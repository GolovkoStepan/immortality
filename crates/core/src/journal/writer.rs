//! Writing and reading journal segments.
//!
//! # The atomicity argument
//!
//! The writer never holds a partially written line in a buffer that is flushed "later": each
//! append is one `write_all` followed by `sync_data`. Two facts make a torn record detectable
//! rather than dangerous:
//!
//! 1. The line's newline is written as part of the same call, so a line without one is the
//!    last line and is incomplete by construction.
//! 2. `seq` and `prev` are inside the line, so a truncated line cannot be mistaken for a valid
//!    record even if it happens to be parseable (it will not be — JSON is prefix-free enough
//!    in practice for a truncated object to fail, and the chain check is the backstop).
//!
//! So a torn tail is repaired by truncating at the last newline, and the number of dropped
//! bytes is recorded as a `journal_truncated_tail` event. That event is written *after* the
//! truncation, into the repaired file, so the repair itself is auditable.
//!
//! # Why the writer owns the chain
//!
//! Callers pass events, not records: `seq` assignment, `prev` chaining and rotation are all
//! decisions that must happen under the same lock and in the same order as the write. If a
//! caller supplied `prev`, a retry after a failed write could reuse a stale digest and fork
//! the chain. [`Journal`] takes `&mut self` for this reason: one writer, no interleaving.

use std::fs::{File, OpenOptions};
use std::io::{BufRead as _, BufReader, Read as _, Seek as _, SeekFrom, Write as _};
use std::path::{Path, PathBuf};

use sha2::{Digest as _, Sha256};

use super::Digest;
use super::record::{JournalEvent, JournalRecord, RecordFields, Scope};
use crate::domain::{TaskId, Timestamp};
use crate::journal::Kind;
use crate::time::{Clock, SystemClock};

/// Maximum size of one segment file before rotation.
///
/// 8 MiB is a deliberate trade: large enough that rotation is rare (a chatty task writes a few
/// hundred bytes per event), small enough that recovery can read a whole segment into memory
/// without thinking about it.
pub const SEGMENT_LIMIT_BYTES: u64 = 8 * 1024 * 1024;

/// The active journal segment.
///
/// One instance per task per process. There is no internal locking: `&mut self` is the lock.
/// The daemon is the only writer (design D-001), so cross-process locking is out of scope;
/// a second daemon is prevented by the daemon lock file, and that is checked before a
/// [`Journal`] is ever opened.
#[derive(Debug)]
pub struct Journal {
    dir: PathBuf,
    file: File,
    path: PathBuf,
    task: TaskId,
    next_seq: u64,
    last_digest: Option<Digest>,
    /// Bytes written to the active segment, tracked in memory rather than `fstat`ed on every
    /// append: the only writer is this struct, so the count cannot drift.
    written: u64,
    /// Where timestamps come from for the writes this struct performs itself.
    ///
    /// Most records arrive with their `at` already chosen by the caller; three do not — the
    /// truncation audit in [`Journal::open`], a rotation's segment name, and every record
    /// [`crate::protocol::ToolCallLog`] writes. Those are exactly the timestamps a golden
    /// journal cannot tolerate being whatever the machine said, so the source is a field rather
    /// than a free function. Production never calls [`Journal::set_clock`]: the default is the
    /// system clock.
    clock: std::sync::Arc<dyn Clock>,
}

impl Journal {
    /// Open (or create) the journal for `task` in `dir`.
    ///
    /// Reads the existing tail to recover `next_seq` and the chain head. A torn tail is
    /// truncated; a chain break is an error, because silently appending after a break would
    /// make the break permanent and invisible.
    pub fn open(dir: impl AsRef<Path>, task: TaskId) -> crate::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| crate::Error::io(&dir, e))?;

        let reader = JournalReader::open(&dir, &task)?;
        let tail = reader.scan()?;
        drop(reader);

        let path = tail.active_path.clone();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)
            .map_err(|e| crate::Error::io(&path, e))?;

        // Truncation has to happen through a writable handle positioned at the tail boundary.
        let mut file = file;
        if tail.valid_bytes < tail.file_len {
            file.set_len(tail.valid_bytes)
                .map_err(|e| crate::Error::io(&path, e))?;
            file.seek(SeekFrom::End(0))
                .map_err(|e| crate::Error::io(&path, e))?;
        }

        let mut journal = Self {
            dir,
            file,
            path,
            task,
            next_seq: tail.next_seq,
            last_digest: tail.last_digest,
            written: tail.valid_bytes,
            clock: std::sync::Arc::new(SystemClock),
        };

        if tail.dropped_bytes > 0 {
            // Audit the repair itself. Written after the truncation, so this record is the
            // first thing a reader of the repaired file sees.
            let dropped = tail.dropped_bytes;
            journal.append(
                RecordFields::new(
                    journal.next_seq,
                    journal.now(),
                    journal.task.clone(),
                    Kind::JournalTruncatedTail,
                ),
                JournalEvent::with_payload(serde_json::json!({ "bytes": dropped })),
            )?;
        }

        Ok(journal)
    }

    /// Install the clock this journal stamps its own records from.
    ///
    /// The runner calls this before it writes anything, so that the runner's `clock` and the
    /// journal's are the same clock rather than two that have to be kept in step by hand. A
    /// journal opened by a one-shot process — `immortality approve` resolving an escalation —
    /// keeps the default.
    pub fn set_clock(&mut self, clock: std::sync::Arc<dyn Clock>) {
        self.clock = clock;
    }

    /// The current instant on this journal's clock.
    ///
    /// Used by the writers that have no caller to hand them a timestamp. A caller that already
    /// knows *when* something happened should pass that `at` explicitly: an event can be
    /// observed before it is written, and `now()` would then say the wrong thing.
    #[must_use]
    pub fn now(&self) -> Timestamp {
        self.clock.now()
    }

    /// Directory holding the segments.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Sequence number the next appended record will get.
    #[must_use]
    pub const fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// Append one event, assigning `seq` and `prev` here. Returns the record as written.
    ///
    /// The returned record is the caller's proof that the effect is durable: it must not be
    /// used to report success to the model before this function returns.
    pub fn append(&mut self, fields: RecordFields, event: JournalEvent) -> crate::Result<JournalRecord> {
        let record = JournalRecord::new(
            RecordFields {
                seq: self.next_seq,
                ..fields
            },
            event,
            self.last_digest,
        );
        let mut line = record.to_line()?;
        line.push('\n');

        if self.written + line.len() as u64 > SEGMENT_LIMIT_BYTES {
            self.rotate()?;
        }

        self.write_line(&line)?;
        // Chain the next record to these exact bytes.
        self.last_digest = Some(hash_line(line.as_bytes()));
        self.next_seq += 1;
        self.written += line.len() as u64;
        Ok(record)
    }

    /// Convenience: append with a bare payload and derive the kind from `fields`.
    pub fn append_event(
        &mut self,
        at: Timestamp,
        kind: Kind,
        payload: serde_json::Value,
    ) -> crate::Result<JournalRecord> {
        let fields = RecordFields::new(self.next_seq, at, self.task.clone(), kind);
        self.append(fields, JournalEvent::with_payload(payload))
    }

    /// Append an event attributed to a subtask and an attempt.
    ///
    /// The scope is not decoration: recovery groups attempts by `(subtask, attempt)` to decide
    /// which ones a crash left `Running`, so a record written without it is invisible to the one
    /// pass that is supposed to close an interrupted attempt. `append_event` stays for records
    /// that genuinely belong to the task rather than to a pass at it.
    pub fn append_scoped(
        &mut self,
        at: Timestamp,
        kind: Kind,
        payload: serde_json::Value,
        scope: &Scope,
    ) -> crate::Result<JournalRecord> {
        let fields = RecordFields {
            seq: self.next_seq,
            at,
            task: self.task.clone(),
            subtask: scope.subtask.clone(),
            attempt: scope.attempt,
            kind,
        };
        self.append(fields, JournalEvent::with_payload(payload))
    }

    /// Append a fully built event with a given kind.
    ///
    /// Unlike [`Self::append_event`], this accepts a [`JournalEvent`] so that an artifact
    /// reference survives. `fields` supplies addressing; `seq` and `prev` are still assigned
    /// here, because those are the writer's business and no caller may set them.
    pub fn append_full(
        &mut self,
        at: Timestamp,
        kind: Kind,
        event: JournalEvent,
    ) -> crate::Result<JournalRecord> {
        let fields = RecordFields::new(self.next_seq, at, self.task.clone(), kind);
        self.append(fields, event)
    }

    /// [`Self::append_full`] with the subtask and attempt the event belongs to.
    pub fn append_full_scoped(
        &mut self,
        at: Timestamp,
        kind: Kind,
        event: JournalEvent,
        scope: &Scope,
    ) -> crate::Result<JournalRecord> {
        let fields = RecordFields {
            seq: self.next_seq,
            at,
            task: self.task.clone(),
            subtask: scope.subtask.clone(),
            attempt: scope.attempt,
            kind,
        };
        self.append(fields, event)
    }

    /// Flush and sync the active segment.
    ///
    /// Appends already sync; this exists for the shutdown path so the last record is
    /// definitely on disk before the process exits and the daemon lock is released.
    pub fn sync(&mut self) -> crate::Result<()> {
        self.file.sync_data().map_err(|e| crate::Error::io(&self.path, e))
    }

    /// Close the active segment by name, leaving it for the next open to pick up.
    ///
    /// Not a "finalize" step: there is no trailer and no marker, because the absence of a
    /// trailer must remain harmless — the whole point is that a killed process leaves a
    /// readable file.
    pub fn close(self) -> crate::Result<()> {
        self.file.sync_all().map_err(|e| crate::Error::io(&self.path, e))
    }

    fn write_line(&mut self, line: &str) -> crate::Result<()> {
        self.file
            .write_all(line.as_bytes())
            .map_err(|e| crate::Error::io(&self.path, e))?;
        // `sync_data`, not `sync_all`: the file's metadata (size) is flushed as part of the
        // data write on every filesystem that matters, and timestamps are not worth a second
        // syscall per event.
        self.file.sync_data().map_err(|e| crate::Error::io(&self.path, e))
    }

    fn rotate(&mut self) -> crate::Result<()> {
        self.file
            .sync_all()
            .map_err(|e| crate::Error::io(&self.path, e))?;
        let stamp = self.now().format("%Y-%m-%dT%H-%M-%S%.3f");
        // Two rotations can land in the same millisecond — a forced rotation in a test, or a
        // single record larger than the segment limit. `create_new` refuses to clobber the
        // previous segment, so step the timestamp forward instead of failing the append.
        let (next, file) = {
            let mut candidate = self.dir.join(format!("JOURNAL.{stamp}.jsonl"));
            let mut attempt = 0u32;
            loop {
                match OpenOptions::new()
                    .create_new(true)
                    .append(true)
                    .read(true)
                    .open(&candidate)
                {
                    Ok(file) => break (candidate, file),
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        attempt += 1;
                        if attempt > 1_000 {
                            return Err(crate::Error::io(&candidate, error));
                        }
                        candidate = self.dir.join(format!("JOURNAL.{stamp}-{attempt:03}.jsonl"));
                    }
                    Err(error) => return Err(crate::Error::io(&candidate, error)),
                }
            }
        };
        // Sync the directory so the new entry survives a power loss. This is the only place
        // that pays for a directory sync, and it happens once per 8 MiB.
        sync_dir(&self.dir)?;
        self.file = file;
        self.path = next;
        self.written = 0;
        Ok(())
    }
}

/// `fsync` a directory so a freshly created entry is durable.
fn sync_dir(dir: &Path) -> crate::Result<()> {
    let handle = File::open(dir).map_err(|e| crate::Error::io(dir, e))?;
    handle.sync_all().map_err(|e| crate::Error::io(dir, e))
}

/// Hash of a journal line, without its newline.
///
/// The newline is excluded so that the digest of a record does not depend on whether the file
/// was written on a platform that would have normalized it; and so that a reader computing the
/// hash from a line has no chance to include the separator by accident.
#[must_use]
pub fn hash_line(line: &[u8]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(line.strip_suffix(b"\n").unwrap_or(line));
    hasher.finalize().into()
}

/// What a scan of the segments found.
#[derive(Debug, Clone)]
pub struct ScanTail {
    /// File new records should go to: the newest segment, or `JOURNAL.jsonl` if none exist.
    pub active_path: PathBuf,
    /// Bytes at the end of the active segment that are a complete record.
    pub valid_bytes: u64,
    /// Current length of the active segment.
    pub file_len: u64,
    /// Bytes of a torn tail that must be dropped. Non-zero means the file was cut short.
    pub dropped_bytes: u64,
    /// Sequence number for the next record.
    pub next_seq: u64,
    /// Digest of the last valid record, for chaining.
    pub last_digest: Option<Digest>,
    /// Number of records seen across all segments.
    pub records: usize,
}

/// A problem found while reading a journal.
///
/// Kept as data rather than an error so that `recover()` can report *all* problems at once:
/// stopping at the first one would hide the rest and make a single recovery pass useless.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadProblem {
    /// The last line of a segment had no terminating newline or did not parse. Recoverable:
    /// nothing after it can exist, and it was never acknowledged to anyone.
    TornTail { segment: PathBuf, bytes: u64 },

    /// A record's `prev` does not match the hash of the record before it.
    ///
    /// Fatal for trust in the file: someone edited, reordered or dropped records. Recovery
    /// must not "fix" this by rewriting the chain.
    ChainBreak { segment: PathBuf, line: usize, seq: u64 },

    /// `seq` did not increase by exactly one.
    ///
    /// Distinct from a chain break because it can also be caused by a bug in the writer, and
    /// the fix differs: a gap usually means records were lost, a duplicate means two writers.
    SequenceGap {
        segment: PathBuf,
        line: usize,
        expected: u64,
        found: u64,
    },
}

impl ReadProblem {
    /// Whether this problem prevents trusting the journal at all.
    ///
    /// A torn tail does not: by construction it is the last thing written and nothing after it
    /// exists. Everything else does.
    #[must_use]
    pub const fn is_fatal(&self) -> bool {
        !matches!(self, Self::TornTail { .. })
    }
}

/// Result of reading every record in a task's journal.
#[derive(Debug, Clone)]
pub struct ReadOutcome {
    pub records: Vec<JournalRecord>,
    pub problems: Vec<ReadProblem>,
    /// True when the last record was unusable and was dropped from `records`.
    pub has_torn_tail: bool,
    /// Hash of the last surviving line, exactly as it appears on disk.
    ///
    /// Carried explicitly rather than recomputed from the parsed record: recomputing means
    /// re-serializing, and a serializer change or a failed serialization would silently
    /// produce a chain head that does not match the file.
    pub last_digest: Option<Digest>,
}

impl ReadOutcome {
    /// Whether any problem makes the journal untrustworthy.
    #[must_use]
    pub fn is_corrupt(&self) -> bool {
        self.problems.iter().any(ReadProblem::is_fatal)
    }
}

/// Reads journal segments without modifying them.
///
/// Separate from [`Journal`] because a reader must work on a directory it cannot write —
/// recovery inspects a crashed task's journal before deciding to repair it, and a read-only
/// inspection should not require write permission.
#[derive(Debug)]
pub struct JournalReader {
    dir: PathBuf,
    task: TaskId,
    segments: Vec<PathBuf>,
}

impl JournalReader {
    /// Find the segments for `task`, oldest first.
    ///
    /// A task's journal is either the single `JOURNAL.jsonl` (created by the first write) or a
    /// set of rotated `JOURNAL.<stamp>.jsonl` files. Both are accepted; the timestamp in the
    /// name sorts correctly because it is zero-padded and date-first.
    pub fn open(dir: impl AsRef<Path>, task: &TaskId) -> crate::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let segments = collect_segments(&dir)?;
        Ok(Self {
            dir,
            task: task.clone(),
            segments,
        })
    }

    /// Read a journal directory without naming a task.
    ///
    /// The `task` field is only used by [`Self::task`]; every record carries its own id. A
    /// caller that has a directory and no id — `grep_journal`, whose argument is the `.agent`
    /// directory — would otherwise have to invent an id to open a file it can already name, and
    /// an invented id in a journal tool is one more thing that can disagree with the records.
    ///
    /// The task stored here is an empty tag, never written and never compared.
    ///
    /// # Errors
    /// The directory could not be listed.
    pub fn open_unnamed(dir: impl AsRef<Path>) -> crate::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let segments = collect_segments(&dir)?;
        Ok(Self {
            dir,
            task: TaskId::from_suffix(""),
            segments,
        })
    }

    /// Read every record and every problem.
    pub fn read(&self) -> crate::Result<ReadOutcome> {
        let tail = self.scan_detailed()?;
        Ok(tail)
    }

    /// Scan just the tail: which file to append to, and where the last valid record ends.
    pub fn scan(&self) -> crate::Result<ScanTail> {
        let outcome = self.scan_detailed()?;
        if outcome.is_corrupt() {
            let first = outcome
                .problems
                .iter()
                .find(|p| p.is_fatal())
                .map(|p| format!("{p:?}"))
                .unwrap_or_default();
            return Err(crate::Error::JournalCorrupt(first));
        }
        // Re-derive the cursor state from the records themselves.
        let last = outcome.records.last();
        let next_seq = last.map_or(0, |r| r.fields.seq + 1);
        let last_digest = outcome.last_digest;
        let active_path = self
            .segments
            .last()
            .cloned()
            .unwrap_or_else(|| self.dir.join("JOURNAL.jsonl"));
        let file_len = std::fs::metadata(&active_path).map_or(0, |m| m.len());
        let valid_bytes = if outcome.has_torn_tail {
            // The torn tail can only be in the last segment, and only at its end: find the
            // offset of the last newline by re-reading the tail of that file.
            last_newline_offset(&active_path)?
        } else {
            file_len
        };
        Ok(ScanTail {
            active_path,
            valid_bytes,
            file_len,
            dropped_bytes: file_len.saturating_sub(valid_bytes),
            next_seq,
            last_digest,
            records: outcome.records.len(),
        })
    }

    fn scan_detailed(&self) -> crate::Result<ReadOutcome> {
        let mut records = Vec::new();
        let mut problems = Vec::new();
        let mut expected_seq: Option<u64> = None;
        let mut previous_digest: Option<Digest> = None;
        let mut has_torn_tail = false;

        for segment in &self.segments {
            let file = match File::open(segment) {
                Ok(file) => file,
                Err(error) => return Err(crate::Error::io(segment, error)),
            };
            let mut reader = BufReader::new(file);
            let mut line = String::new();
            let mut line_number = 0usize;

            loop {
                line.clear();
                let read = reader
                    .read_line(&mut line)
                    .map_err(|e| crate::Error::io(segment, e))?;
                if read == 0 {
                    break;
                }
                line_number += 1;
                let terminated = line.ends_with('\n');
                let line_bytes = line.as_bytes();

                // A torn tail: the final line of the final file that never got its newline.
                // Anything else that fails to parse is corruption, not a torn write.
                if !terminated {
                    if segment == self.segments.last().unwrap_or(segment) {
                        has_torn_tail = true;
                        problems.push(ReadProblem::TornTail {
                            segment: segment.clone(),
                            bytes: read as u64,
                        });
                        break;
                    }
                    problems.push(ReadProblem::ChainBreak {
                        segment: segment.clone(),
                        line: line_number,
                        seq: expected_seq.unwrap_or(0),
                    });
                    break;
                }

                let trimmed = line.trim_end_matches('\n');
                if trimmed.is_empty() {
                    // A blank line inside the file is not a record. It can only get there by
                    // an editor, so treat it as corruption rather than skipping it.
                    problems.push(ReadProblem::ChainBreak {
                        segment: segment.clone(),
                        line: line_number,
                        seq: expected_seq.unwrap_or(0),
                    });
                    continue;
                }

                let record = match JournalRecord::from_line(trimmed) {
                    Ok(record) => record,
                    Err(error) => {
                        return Err(match error {
                            crate::Error::JournalVersion { found, expected } => {
                                crate::Error::JournalVersion { found, expected }
                            }
                            other => crate::Error::JournalCorrupt(format!(
                                "{}:{line_number}: {other}",
                                segment.display()
                            )),
                        });
                    }
                };

                if let Some(expected) = expected_seq
                    && record.fields.seq != expected
                {
                    problems.push(ReadProblem::SequenceGap {
                        segment: segment.clone(),
                        line: line_number,
                        expected,
                        found: record.fields.seq,
                    });
                }
                expected_seq = Some(record.fields.seq + 1);

                match (previous_digest, record.prev) {
                    (None, None) => {}
                    (Some(previous), Some(found)) if previous == found => {}
                    (_, _) => {
                        problems.push(ReadProblem::ChainBreak {
                            segment: segment.clone(),
                            line: line_number,
                            seq: record.fields.seq,
                        });
                    }
                }
                previous_digest = Some(hash_line(line_bytes));
                records.push(record);
            }

            // A torn tail can only be in the last segment; stop at the first one found so that
            // later segments (which should not exist) are not silently accepted.
            if has_torn_tail {
                break;
            }
        }

        Ok(ReadOutcome {
            records,
            problems,
            has_torn_tail,
            last_digest: previous_digest,
        })
    }

    /// Task this reader was opened for.
    #[must_use]
    pub const fn task(&self) -> &TaskId {
        &self.task
    }

    /// Segments found, oldest first.
    #[must_use]
    pub fn segments(&self) -> &[PathBuf] {
        &self.segments
    }
}

/// Every segment in `dir`, oldest first.
///
/// NOT plain lexicographic order. `JOURNAL.jsonl` vs `JOURNAL.2026-...`: at the position after
/// `JOURNAL` one name has `.` (0x2E) and the other has `j` (0x6A), so the un-stamped initial
/// segment would sort *last* — reading the journal backwards. The initial segment is ordered
/// first explicitly, the rest by their parsed stamp.
fn collect_segments(dir: &Path) -> crate::Result<Vec<PathBuf>> {
    let mut segments = Vec::new();
    if dir.is_dir() {
        let entries = std::fs::read_dir(dir).map_err(|e| crate::Error::io(dir, e))?;
        for entry in entries {
            let entry = entry.map_err(|e| crate::Error::io(dir, e))?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if is_segment_name(name) {
                segments.push(entry.path());
            }
        }
    }
    segments.sort_by_key(|path| segment_order(path));
    Ok(segments)
}

/// Whether a file in the journal directory is one of ours.
///
/// Deliberately strict: a stray file that happens to end in `.jsonl` must not be replayed as
/// part of the task's history. Delegates to [`parse_stamp`] so that the reader and the
/// rotation writer cannot disagree about which names exist — they did, once, and the
/// collision-suffixed segments the writer produced were invisible to the reader.
fn is_segment_name(name: &str) -> bool {
    name == "JOURNAL.jsonl" || parse_stamp(name).is_some()
}

/// Sort key for a segment: the initial one first, then chronological by parsed stamp.
fn segment_order(path: &Path) -> (u8, i64, String) {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string();
    if name == "JOURNAL.jsonl" {
        // The initial segment predates every rotation, so it orders before them. The 0 is not
        // a real timestamp; the discriminator field does the work.
        return (0, 0, name);
    }
    match parse_stamp(&name) {
        // Nanoseconds since the epoch: `NaiveDateTime` is comparable, but a plain integer key
        // keeps the tuple cheap and avoids a dependency on the timezone type ordering.
        Some(stamp) => (1, stamp.and_utc().timestamp_millis(), name),
        // An unrecognised name is only reachable through a race with an external writer; put
        // it after the parseable ones so it cannot displace them, and let the chain check
        // report it if it contains foreign records.
        None => (2, 0, name),
    }
}

/// Parse the timestamp out of `JOURNAL.<stamp>[-<n>].jsonl`.
fn parse_stamp(name: &str) -> Option<chrono::NaiveDateTime> {
    let rest = name.strip_prefix("JOURNAL.")?.strip_suffix(".jsonl")?;
    for format in ["%Y-%m-%dT%H-%M-%S%.3f", "%Y-%m-%dT%H-%M-%S"] {
        if let Ok(stamp) = chrono::NaiveDateTime::parse_from_str(rest, format) {
            return Some(stamp);
        }
    }
    // Collision-suffixed form: `...-001`.
    let (stamp, _) = rest.rsplit_once('-')?;
    ["%Y-%m-%dT%H-%M-%S%.3f", "%Y-%m-%dT%H-%M-%S"]
        .iter()
        .find_map(|format| chrono::NaiveDateTime::parse_from_str(stamp, format).ok())
}

/// Offset just past the last newline in a file.
///
/// Used to decide where a torn tail begins. Reading backwards in blocks rather than the whole
/// file matters once a segment is near 8 MiB.
fn last_newline_offset(path: &Path) -> crate::Result<u64> {
    /// Backwards read granularity. One page-ish block bounds the work for an 8 MiB segment to
    /// a single `read` in the common case, where the tail is short.
    const BLOCK: u64 = 64 * 1024;

    let mut file = File::open(path).map_err(|e| crate::Error::io(path, e))?;
    let len = file.metadata().map_err(|e| crate::Error::io(path, e))?.len();
    if len == 0 {
        return Ok(0);
    }
    let mut position = len;
    while position > 0 {
        let take = BLOCK.min(position);
        let start = position - take;
        file.seek(SeekFrom::Start(start))
            .map_err(|e| crate::Error::io(path, e))?;
        let mut buffer = vec![0u8; usize::try_from(take).unwrap_or(0)];
        file.read_exact(&mut buffer)
            .map_err(|e| crate::Error::io(path, e))?;
        if let Some(index) = buffer.iter().rposition(|&b| b == b'\n') {
            return Ok(start + index as u64 + 1);
        }
        position = start;
    }
    // No newline at all: nothing in the file is a complete record.
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::TaskId;
    use chrono::TimeZone as _;

    fn at() -> Timestamp {
        chrono::Utc.with_ymd_and_hms(2026, 11, 12, 10, 14, 3).unwrap()
    }

    fn journal(dir: &Path) -> Journal {
        Journal::open(dir, TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV")).unwrap()
    }

    #[test]
    fn empty_dir_yields_a_journal_at_the_canonical_name() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = journal(dir.path());
        writer
            .append_event(at(), Kind::DaemonStarted, serde_json::json!({}))
            .unwrap();
        let text = std::fs::read_to_string(dir.path().join("JOURNAL.jsonl")).unwrap();
        assert_eq!(text.lines().count(), 1);
        assert!(text.ends_with('\n'));
    }

    #[test]
    fn seq_starts_at_zero_and_increases_by_one() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = journal(dir.path());
        for n in 0..5 {
            let record = writer
                .append_event(at(), Kind::TaskState, serde_json::json!({ "n": n }))
                .unwrap();
            assert_eq!(record.fields.seq, n);
        }
        assert_eq!(writer.next_seq(), 5);
    }

    #[test]
    fn reopening_continues_the_chain() {
        let dir = tempfile::tempdir().unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let first_digest = {
            let mut writer = Journal::open(dir.path(), task.clone()).unwrap();
            writer
                .append_event(at(), Kind::TaskState, serde_json::json!({"n": 0}))
                .unwrap();
            writer
                .append_event(at(), Kind::TaskState, serde_json::json!({"n": 1}))
                .unwrap();
            writer.close().unwrap();
            let reader = JournalReader::open(dir.path(), &task).unwrap();
            let outcome = reader.read().unwrap();
            assert_eq!(outcome.records.len(), 2);
            hash_line(outcome.records.last().unwrap().to_line().unwrap().as_bytes())
        };

        let mut writer = Journal::open(dir.path(), task.clone()).unwrap();
        let record = writer
            .append_event(at(), Kind::TaskState, serde_json::json!({"n": 2}))
            .unwrap();
        assert_eq!(record.fields.seq, 2);
        assert_eq!(record.prev, Some(first_digest));
        writer.close().unwrap();

        let reader = JournalReader::open(dir.path(), &task).unwrap();
        let outcome = reader.read().unwrap();
        assert_eq!(outcome.records.len(), 3);
        assert_eq!(outcome.problems, Vec::new());
    }

    #[test]
    fn a_scoped_record_carries_the_subtask_and_attempt() {
        // Recovery keys attempts by `(subtask, attempt)`. A record written without the scope is
        // invisible to the pass that closes interrupted attempts, so the scope is asserted on
        // both the in-memory record and the bytes on disk.
        let dir = tempfile::tempdir().unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let subtask = crate::domain::SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAA");
        {
            let mut writer = Journal::open(dir.path(), task.clone()).unwrap();
            let record = writer
                .append_scoped(
                    at(),
                    Kind::AttemptStarted,
                    serde_json::json!({ "number": 2 }),
                    &crate::journal::Scope::attempt(subtask.clone(), 2),
                )
                .unwrap();
            assert_eq!(record.fields.subtask.as_ref(), Some(&subtask));
            assert_eq!(record.fields.attempt, Some(2));
            writer.close().unwrap();
        }

        let text = std::fs::read_to_string(dir.path().join("JOURNAL.jsonl")).unwrap();
        let line: serde_json::Value = serde_json::from_str(text.trim_end()).unwrap();
        assert_eq!(line["subtask"], subtask.as_str());
        assert_eq!(line["attempt"], 2);

        let reader = JournalReader::open(dir.path(), &task).unwrap();
        let records = reader.read().unwrap().records;
        assert_eq!(records[0].fields.subtask.as_ref(), Some(&subtask));
        assert_eq!(records[0].fields.attempt, Some(2));
    }

    #[test]
    fn an_unscoped_record_omits_the_scope_fields_entirely() {
        // `skip_serializing_if` is load-bearing: a record about the task must not claim an
        // attempt number it does not have, because `null` and "absent" read differently to
        // every consumer that looks the field up by name.
        let dir = tempfile::tempdir().unwrap();
        let mut writer = journal(dir.path());
        writer
            .append_event(at(), Kind::DaemonStarted, serde_json::json!({}))
            .unwrap();
        let text = std::fs::read_to_string(dir.path().join("JOURNAL.jsonl")).unwrap();
        let line: serde_json::Value = serde_json::from_str(text.trim_end()).unwrap();
        assert!(line.get("subtask").is_none(), "{line}");
        assert!(line.get("attempt").is_none(), "{line}");
    }

    #[test]
    fn a_torn_tail_is_dropped_and_audited() {
        let dir = tempfile::tempdir().unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        {
            let mut writer = Journal::open(dir.path(), task.clone()).unwrap();
            writer
                .append_event(at(), Kind::TaskState, serde_json::json!({"n": 0}))
                .unwrap();
            writer
                .append_event(at(), Kind::TaskState, serde_json::json!({"n": 1}))
                .unwrap();
            writer.close().unwrap();
        }
        let path = dir.path().join("JOURNAL.jsonl");
        let full = std::fs::read_to_string(&path).unwrap();
        let cut = full.len() - 20;
        std::fs::write(&path, &full[..cut]).unwrap();

        let mut writer = Journal::open(dir.path(), task.clone()).unwrap();
        // The repair is itself recorded, so the first new record has an increased seq and
        // there is a truncation event on disk.
        let record = writer
            .append_event(at(), Kind::TaskState, serde_json::json!({"n": 2}))
            .unwrap();
        assert_eq!(record.fields.seq, 2);
        writer.close().unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("journal_truncated_tail"), "{text}");
        let reader = JournalReader::open(dir.path(), &task).unwrap();
        let outcome = reader.read().unwrap();
        assert_eq!(outcome.records.len(), 3, "two original + one audit event");
        assert!(!outcome.is_corrupt());
    }

    #[test]
    fn truncated_at_every_byte_is_always_recoverable() {
        // The exhaustive case the roadmap asks for: cut the file at every possible position.
        // Two things must hold, and nothing stronger is possible: every complete line before
        // the cut survives, and a cut that lands inside a line is reported as a torn tail
        // rather than dropped silently.
        let dir = tempfile::tempdir().unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        {
            let mut writer = Journal::open(dir.path(), task.clone()).unwrap();
            writer
                .append_event(at(), Kind::TaskState, serde_json::json!({"n": 0}))
                .unwrap();
            writer
                .append_event(at(), Kind::TaskState, serde_json::json!({"n": 1}))
                .unwrap();
            writer.close().unwrap();
        }
        let path = dir.path().join("JOURNAL.jsonl");
        let full = std::fs::read(&path).unwrap();

        for cut in 0..=full.len() {
            std::fs::write(&path, &full[..cut]).unwrap();
            let reader = JournalReader::open(dir.path(), &task).unwrap();
            let outcome = reader
                .read()
                .unwrap_or_else(|error| panic!("cut at {cut}: {error}"));
            let complete_lines = full[..cut].iter().filter(|&&b| b == b'\n').count();
            assert_eq!(
                outcome.records.len(),
                complete_lines,
                "cut at {cut} kept the wrong records"
            );
            let mid_line = cut > 0 && full[cut - 1] != b'\n';
            assert_eq!(
                outcome.has_torn_tail, mid_line,
                "cut at {cut} misreported the tail"
            );
            assert!(!outcome.is_corrupt(), "cut at {cut} reported corruption");
        }
    }

    #[test]
    fn a_chain_break_is_fatal_and_not_repaired() {
        let dir = tempfile::tempdir().unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        {
            let mut writer = Journal::open(dir.path(), task.clone()).unwrap();
            for n in 0..3 {
                writer
                    .append_event(at(), Kind::TaskState, serde_json::json!({"n": n}))
                    .unwrap();
            }
            writer.close().unwrap();
        }
        let path = dir.path().join("JOURNAL.jsonl");
        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
        // Rewrite the middle record's payload, leaving its `prev` intact: a plausible silent
        // edit that must be caught.
        lines[1] = lines[1].replace("\"n\":1", "\"n\":99");
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();

        let reader = JournalReader::open(dir.path(), &task).unwrap();
        let outcome = reader.read().unwrap();
        assert!(outcome.is_corrupt(), "{:?}", outcome.problems);
        assert!(
            outcome
                .problems
                .iter()
                .any(|p| matches!(p, ReadProblem::ChainBreak { line: 3, .. })),
            "{:?}",
            outcome.problems
        );

        // `scan` refuses rather than silently starting a new chain.
        let reader = JournalReader::open(dir.path(), &task).unwrap();
        assert!(reader.scan().is_err());
    }

    #[test]
    fn a_deleted_record_is_a_sequence_gap() {
        let dir = tempfile::tempdir().unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        {
            let mut writer = Journal::open(dir.path(), task.clone()).unwrap();
            for n in 0..4 {
                writer
                    .append_event(at(), Kind::TaskState, serde_json::json!({"n": n}))
                    .unwrap();
            }
            writer.close().unwrap();
        }
        let path = dir.path().join("JOURNAL.jsonl");
        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines: Vec<&str> = text.lines().collect();
        lines.remove(1);
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();

        let reader = JournalReader::open(dir.path(), &task).unwrap();
        let outcome = reader.read().unwrap();
        assert!(outcome.is_corrupt());
        assert!(
            outcome.problems.iter().any(|p| matches!(
                p,
                ReadProblem::SequenceGap {
                    expected: 1,
                    found: 2,
                    ..
                }
            )),
            "{:?}",
            outcome.problems
        );
    }

    #[test]
    fn rotation_creates_a_second_segment_that_sorts_after() {
        let dir = tempfile::tempdir().unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let mut writer = Journal::open(dir.path(), task.clone()).unwrap();
        // Force rotation by faking the counter: simpler and faster than writing 8 MiB.
        writer.written = SEGMENT_LIMIT_BYTES;
        writer
            .append_event(at(), Kind::TaskState, serde_json::json!({"after": "rotation"}))
            .unwrap();
        writer.close().unwrap();

        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names.len(), 2, "{names:?}");
        assert!(names.iter().any(|n| n.starts_with("JOURNAL.2")));

        let reader = JournalReader::open(dir.path(), &task).unwrap();
        let outcome = reader.read().unwrap();
        assert_eq!(outcome.records.len(), 1);
        assert_eq!(outcome.problems, Vec::new());
        assert_eq!(outcome.records[0].fields.seq, 0);
    }

    #[test]
    fn segments_are_read_in_chronological_order() {
        let dir = tempfile::tempdir().unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let mut writer = Journal::open(dir.path(), task.clone()).unwrap();
        writer
            .append_event(at(), Kind::TaskState, serde_json::json!({"seg": 0}))
            .unwrap();
        writer.written = SEGMENT_LIMIT_BYTES;
        writer
            .append_event(at(), Kind::TaskState, serde_json::json!({"seg": 1}))
            .unwrap();
        writer.written = SEGMENT_LIMIT_BYTES;
        writer
            .append_event(at(), Kind::TaskState, serde_json::json!({"seg": 2}))
            .unwrap();
        writer.close().unwrap();

        let reader = JournalReader::open(dir.path(), &task).unwrap();
        let outcome = reader.read().unwrap();
        assert_eq!(outcome.records.len(), 3);
        let order: Vec<u64> = outcome.records.iter().map(|r| r.fields.seq).collect();
        assert_eq!(order, [0, 1, 2]);
        assert_eq!(outcome.problems, Vec::new(), "chain must span segments");
    }

    #[test]
    fn segment_names_are_recognised_strictly() {
        assert!(is_segment_name("JOURNAL.jsonl"));
        assert!(is_segment_name("JOURNAL.2026-11-12T10-00-00.jsonl"));
        assert!(is_segment_name("JOURNAL.2026-11-12T10-00-00.123.jsonl"));
        // Collision-suffixed segments are produced by `rotate` when two rotations land in the
        // same millisecond; the reader must see them or the journal silently loses records.
        assert!(is_segment_name("JOURNAL.2026-11-12T10-00-00.123-001.jsonl"));
        assert!(is_segment_name("JOURNAL.2026-11-12T10-00-00-002.jsonl"));
        for bad in [
            "JOURNAL.txt",
            "JOURNAL.2026-11-12.jsonl",
            "journal.jsonl",
            "JOURNAL.2026-13-12T10-00-00.jsonl",
            "JOURNALX.2026-11-12T10-00-00.jsonl",
            "JOURNAL.2026-11-12T10-00-00.jsonl.bak",
            "JOURNAL.2026-11-12T10-00-00.jsonl-001.jsonl",
            "other.jsonl",
        ] {
            assert!(!is_segment_name(bad), "`{bad}` was accepted");
        }
    }

    #[test]
    fn rotation_happens_before_the_limit_is_exceeded() {
        // The segment may not grow past the limit, and one record larger than the limit must
        // still be written: it goes alone into a fresh segment rather than being dropped.
        let dir = tempfile::tempdir().unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let mut writer = Journal::open(dir.path(), task).unwrap();
        let padding = "x".repeat(usize::try_from(SEGMENT_LIMIT_BYTES).unwrap());
        writer
            .append_event(at(), Kind::ModelResponse, serde_json::json!({ "body": padding }))
            .unwrap();
        writer.close().unwrap();

        let mut total = 0u64;
        for entry in std::fs::read_dir(dir.path()).unwrap() {
            total += entry.unwrap().metadata().unwrap().len();
        }
        assert!(total >= SEGMENT_LIMIT_BYTES);
        // Exactly one record survived, so nothing was truncated to fit.
        let reader =
            JournalReader::open(dir.path(), &TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV")).unwrap();
        assert_eq!(reader.read().unwrap().records.len(), 1);
    }

    #[test]
    fn a_foreign_file_in_the_dir_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        std::fs::write(dir.path().join("notes.txt"), b"hello").unwrap();
        let mut writer = Journal::open(dir.path(), task.clone()).unwrap();
        writer
            .append_event(at(), Kind::TaskState, serde_json::json!({}))
            .unwrap();
        writer.close().unwrap();
        let reader = JournalReader::open(dir.path(), &task).unwrap();
        assert_eq!(reader.read().unwrap().records.len(), 1);
    }

    #[test]
    fn reopening_an_empty_journal_starts_at_zero() {
        let dir = tempfile::tempdir().unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let opened = Journal::open(dir.path(), task.clone()).unwrap();
        assert_eq!(opened.next_seq(), 0);
        opened.close().unwrap();

        let reopened = Journal::open(dir.path(), task).unwrap();
        assert_eq!(reopened.next_seq(), 0);
    }

    #[test]
    fn foreign_task_ids_do_not_share_a_journal() {
        let dir = tempfile::tempdir().unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let reader = JournalReader::open(dir.path(), &task).unwrap();
        assert_eq!(reader.read().unwrap().records.len(), 0);
        assert_eq!(reader.segments().len(), 0);
    }

    #[test]
    fn concurrent_appends_from_one_writer_interleave_at_line_boundaries() {
        // The writer is `&mut self`, so the real concurrency question is whether two threads
        // sharing it through a mutex ever produce a partial line. This drives that shape.
        use std::sync::{Arc, Mutex};
        let dir = tempfile::tempdir().unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let journal = Arc::new(Mutex::new(Journal::open(dir.path(), task.clone()).unwrap()));

        let mut handles = Vec::new();
        for thread in 0..4 {
            let journal = Arc::clone(&journal);
            handles.push(std::thread::spawn(move || {
                for n in 0..25 {
                    // The guard is held for the whole append and dropped at the end of the
                    // iteration. Binding it to `mut writer` would drop the guard early — the
                    // lint is right: `MutexGuard`'s `Drop` is significant.
                    let writer = &mut *journal.lock().unwrap();
                    writer
                        .append_event(
                            at(),
                            Kind::TaskState,
                            serde_json::json!({ "thread": thread, "n": n }),
                        )
                        .unwrap();
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        Arc::try_unwrap(journal)
            .ok()
            .unwrap()
            .into_inner()
            .unwrap()
            .close()
            .unwrap();

        let reader = JournalReader::open(dir.path(), &task).unwrap();
        let outcome = reader.read().unwrap();
        assert_eq!(outcome.records.len(), 100);
        assert_eq!(outcome.problems, Vec::new());
        let seqs: Vec<u64> = outcome.records.iter().map(|r| r.fields.seq).collect();
        assert_eq!(seqs, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn the_chain_survives_many_reopenings() {
        // Recovery reopens a journal every time a task resumes. A chain that only holds within
        // one process would pass the single-reopen test and fail in the field.
        let dir = tempfile::tempdir().unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        for n in 0..10 {
            let mut writer = Journal::open(dir.path(), task.clone()).unwrap();
            assert_eq!(writer.next_seq(), n);
            writer
                .append_event(at(), Kind::TaskState, serde_json::json!({ "n": n }))
                .unwrap();
            writer.close().unwrap();
        }
        let reader = JournalReader::open(dir.path(), &task).unwrap();
        let outcome = reader.read().unwrap();
        assert_eq!(outcome.records.len(), 10);
        assert_eq!(outcome.problems, Vec::new());
    }

    #[test]
    fn sync_does_not_disturb_the_chain() {
        let dir = tempfile::tempdir().unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let mut writer = Journal::open(dir.path(), task.clone()).unwrap();
        writer
            .append_event(at(), Kind::TaskState, serde_json::json!({ "n": 0 }))
            .unwrap();
        writer.sync().unwrap();
        writer
            .append_event(at(), Kind::TaskState, serde_json::json!({ "n": 1 }))
            .unwrap();
        writer.close().unwrap();
        let reader = JournalReader::open(dir.path(), &task).unwrap();
        assert_eq!(reader.read().unwrap().problems, Vec::new());
    }

    #[test]
    fn the_active_segment_is_the_last_one() {
        let dir = tempfile::tempdir().unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let mut writer = Journal::open(dir.path(), task.clone()).unwrap();
        writer
            .append_event(at(), Kind::TaskState, serde_json::json!({ "seg": 0 }))
            .unwrap();
        writer.written = SEGMENT_LIMIT_BYTES;
        writer
            .append_event(at(), Kind::TaskState, serde_json::json!({ "seg": 1 }))
            .unwrap();
        let active = writer.path.clone();
        writer.close().unwrap();

        // Reopening must append to the same segment, not create another one.
        let mut reopened = Journal::open(dir.path(), task).unwrap();
        reopened
            .append_event(at(), Kind::TaskState, serde_json::json!({ "seg": 2 }))
            .unwrap();
        assert_eq!(reopened.path, active);
        reopened.close().unwrap();
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
    }
}
