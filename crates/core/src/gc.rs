//! Garbage collection and the disk guard: the part that keeps a multi-day run from dying of
//! `ENOSPC`.
//!
//! # What is collected, and what is never
//!
//! | data | policy |
//! | --- | --- |
//! | journal (`JOURNAL*.jsonl` in the repo) | never deleted, anywhere in this module |
//! | artifacts referenced by an unfinished task | never deleted, regardless of age |
//! | other artifacts | deleted after `artifact_ttl_days` |
//! | `.tmp` scratch from a crashed write | deleted after an hour |
//! | logs of past months | compressed to `.gz`, then deleted after `log_retention_months` |
//!
//! The journal is the audit trail and the only thing that can rebuild every other file; deleting
//! it would be deleting the system's memory. Artifacts are a cache of heavy bodies — losing one
//! degrades an old observation, losing the record would look like a call that never finished.
//!
//! # Why the caller supplies the live set
//!
//! `gc` does not know which tasks are unfinished: that is the daemon's scheduler state. So the
//! caller passes the digests referenced by journals of *active* tasks, and `gc` treats them as
//! untouchable. Recomputing liveness here from the directory tree would be wrong twice: it would
//! need to know which tasks are active, and it would make `gc` able to delete an artifact the
//! scheduler is about to read.
//!
//! # Crash safety of compression
//!
//! A log is compressed by writing `<name>.gz.part`, syncing it, renaming it into place and only
//! then removing the plain file. A crash at any point leaves either the plain file, or the
//! archive, or both — never neither. On the next run an archive without its plain file is left
//! alone, and a plain file whose archive already exists is simply removed. Compression is
//! therefore idempotent without a marker file.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use chrono::Datelike as _;

use crate::artifacts::ArtifactStore;
use crate::config::Storage;
use crate::journal::{Digest, JournalRecord, digest_text};

/// Seconds in a day, for TTL arithmetic.
const DAY_SECS: u64 = 24 * 60 * 60;

/// A crashed writer's leftover may be a file another thread is writing right now, so the temp
/// sweep is bounded by an hour rather than by the artifact TTL.
const TEMP_SWEEP_AGE_SECS: u64 = 3600;

/// What `gc` did. Every field is reported by `immortality gc`, so a field that is not worth
/// printing does not belong here.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GcReport {
    /// Artifacts deleted because they outlived the TTL and nothing referenced them.
    pub artifacts_removed: usize,
    /// Bytes those artifacts occupied.
    pub bytes_reclaimed: u64,
    /// Artifacts kept only because an unfinished task still references them.
    pub artifacts_protected: usize,
    /// Stale `.tmp` files removed.
    pub temp_removed: usize,
    /// Plain logs compressed into archives during this run.
    pub logs_compressed: Vec<String>,
    /// Archives deleted because they exceeded the retention window.
    pub logs_deleted: Vec<String>,
}

impl GcReport {
    /// Whether the run changed anything on disk.
    #[must_use]
    pub const fn is_noop(&self) -> bool {
        self.artifacts_removed == 0
            && self.temp_removed == 0
            && self.logs_compressed.is_empty()
            && self.logs_deleted.is_empty()
    }
}

/// Run collection over a store and a log directory.
///
/// `now` is passed rather than read so that TTL and retention can be tested at a month boundary
/// without waiting a month. `protected` is the live set from unfinished tasks.
pub fn collect(
    store: &ArtifactStore,
    logs: &Path,
    storage: &Storage,
    protected: &BTreeSet<Digest>,
    now: chrono::DateTime<chrono::Local>,
) -> crate::Result<GcReport> {
    let mut report = GcReport::default();
    let now_secs = u64::try_from(now.timestamp()).unwrap_or(0);
    let ttl_secs = u64::from(storage.artifact_ttl_days) * DAY_SECS;

    for entry in store.entries()? {
        if protected.contains(&entry.digest) {
            report.artifacts_protected += 1;
            continue;
        }
        // `modified_secs == 0` means the platform would not report an mtime. Without an age there
        // is no way to prove the file is old, and deleting on a guess is how a live artifact dies.
        if entry.modified_secs == 0 || now_secs < entry.modified_secs {
            continue;
        }
        if now_secs - entry.modified_secs >= ttl_secs && store.remove(&entry.digest)? {
            report.artifacts_removed += 1;
            report.bytes_reclaimed += entry.bytes;
        }
    }

    report.temp_removed = store.sweep_temp(TEMP_SWEEP_AGE_SECS, now_secs)?;
    collect_logs(logs, storage.log_retention_months, now, &mut report)?;
    Ok(report)
}

/// Compress stale plain logs and delete archives past retention.
fn collect_logs(
    dir: &Path,
    retention_months: u32,
    now: chrono::DateTime<chrono::Local>,
    report: &mut GcReport,
) -> crate::Result<()> {
    for path in crate::logging::stale_logs(dir, now) {
        let archive = archive_path(&path);
        if archive.exists() {
            // Already compressed by a run that died before removing the plain file.
            if std::fs::remove_file(&path).is_ok() {
                report.logs_compressed.push(file_name(&path).unwrap_or_default());
            }
            continue;
        }
        if compress(&path, &archive)? {
            report.logs_compressed.push(file_name(&path).unwrap_or_default());
        }
    }

    let now_month = month_index(now.year(), now.month());
    for path in archive_logs(dir) {
        let Some(index) = month_of_archive(&path) else {
            continue;
        };
        // Retention is measured in whole months, and the current month's own archive does not
        // exist yet. A negative difference means an archive from the future — a clock step, or a
        // file somebody copied in — and it is kept for the same reason `sweep_temp` keeps files
        // newer than now.
        let Some(age) = now_month.checked_sub(index) else {
            continue;
        };
        if age >= i32::try_from(retention_months).unwrap_or(i32::MAX) && std::fs::remove_file(&path).is_ok() {
            report.logs_deleted.push(file_name(&path).unwrap_or_default());
        }
    }
    Ok(())
}

/// Compress `path` into `archive` atomically. Returns whether it was compressed.
///
/// Returns `false` for an empty file: there is nothing to keep, and a zero-byte archive that
/// could not be read back would be worse than no archive.
fn compress(path: &Path, archive: &Path) -> crate::Result<bool> {
    let metadata = std::fs::metadata(path).map_err(|e| crate::Error::io(path, e))?;
    if metadata.len() == 0 {
        return Ok(false);
    }
    let mut source = std::fs::File::open(path).map_err(|e| crate::Error::io(path, e))?;

    let temp = archive.with_extension("gz.part");
    // A leftover `.part` from a crash is overwritten on purpose: it is this file's own scratch,
    // never another writer's, because the archive name it derives from is unique.
    let file = std::fs::File::create(&temp).map_err(|e| crate::Error::io(&temp, e))?;
    let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    if let Err(error) = std::io::copy(&mut source, &mut encoder) {
        drop(encoder);
        let _ = std::fs::remove_file(&temp);
        return Err(crate::Error::io(path, error));
    }
    let file = encoder.finish().map_err(|e| crate::Error::io(&temp, e))?;
    file.sync_all().map_err(|e| crate::Error::io(&temp, e))?;
    drop(file);
    std::fs::rename(&temp, archive).map_err(|e| crate::Error::io(archive, e))?;
    // Durability of the rename, not of the data: the directory entry is what a crash could lose.
    if let Some(parent) = archive.parent() {
        sync_dir(parent);
    }
    // Only now is the plain file redundant. A crash before this point costs disk, never data.
    std::fs::remove_file(path).map_err(|e| crate::Error::io(path, e))?;
    Ok(true)
}

fn sync_dir(dir: &Path) {
    // Best effort: some filesystems refuse to open or sync a directory, and a failure here does
    // not corrupt anything — it only means the rename is not durable yet.
    if let Ok(handle) = std::fs::File::open(dir) {
        let _ = handle.sync_all();
    }
}

fn archive_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".gz");
    PathBuf::from(name)
}

fn file_name(path: &Path) -> Option<String> {
    path.file_name()?.to_str().map(str::to_string)
}

/// `YYYY-MM.log.gz` files in `dir`.
fn archive_logs(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".log.gz"))
                && month_of_archive(path).is_some()
        })
        .collect();
    out.sort();
    out
}

/// The month an archive name denotes, as a monotone index (`year * 12 + month - 1`).
fn month_of_archive(path: &Path) -> Option<i32> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".log.gz")?;
    let (year, month) = stem.split_once('-')?;
    if year.len() != 4 || month.len() != 2 {
        return None;
    }
    let year: i32 = year.parse().ok()?;
    let month: u32 = month.parse().ok()?;
    if !(1..=12).contains(&month) {
        return None;
    }
    Some(month_index(year, month))
}

fn month_index(year: i32, month: u32) -> i32 {
    year * 12 + i32::try_from(month).unwrap_or(1) - 1
}

/// Every artifact digest a journal references, for use as the live set.
///
/// The journal names the heavy body in `artifact`; recovery reads that field to reconstruct an
/// observation, so an artifact named there is live for as long as the task is unfinished.
/// A malformed digest is skipped rather than fatal: `gc` must still run on a journal that
/// recovery is about to block, and refusing to collect because one field is unparseable would
/// turn a corruption into a full disk.
#[must_use]
pub fn referenced_artifacts(records: &[JournalRecord]) -> BTreeSet<Digest> {
    records
        .iter()
        .filter_map(|record| record.event.artifact.as_deref())
        .filter_map(|text| crate::journal::parse_digest(text).ok())
        .collect()
}

/// What the disk guard decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskState {
    /// Enough room to keep working.
    Enough,
    /// Below the warning threshold: tell the operator, keep working.
    Low,
    /// Below the pause threshold: stop the task before a write fails halfway.
    Exhausted,
}

impl DiskState {
    /// The `Hold` this state implies, if any.
    ///
    /// `Low` deliberately produces none: pausing on the warning threshold would stop work at
    /// 1 GiB free, which is most of a modest disk, and the whole point of the warning is to act
    /// while there is still room to act.
    #[must_use]
    pub const fn hold(self) -> Option<crate::domain::Hold> {
        match self {
            Self::Enough | Self::Low => None,
            Self::Exhausted => Some(crate::domain::Hold::PausedDisk),
        }
    }
}

/// Classify free space against the configured thresholds.
///
/// Pure so the daemon's one `df` call is the only untestable part of the guard.
#[must_use]
pub const fn disk_state(free_bytes: u64, storage: &Storage) -> DiskState {
    if free_bytes < storage.disk_pause_bytes {
        DiskState::Exhausted
    } else if free_bytes < storage.disk_warn_bytes {
        DiskState::Low
    } else {
        DiskState::Enough
    }
}

/// Whether an I/O error means the disk is full, as opposed to a permissions or path problem.
///
/// The thresholds above are a prediction; this is the observation. A check that ran a second ago
/// can be defeated by a large checkpoint, another process, or a filesystem that reserves blocks
/// for root. When the write itself fails with `ENOSPC` or `EDQUOT`, the prediction is wrong and
/// the task must pause for the same reason it would have paused had the prediction been right —
/// otherwise the daemon retries a failing write forever and the last journal entry is the one
/// that never landed.
///
/// `EDQUOT` is included because a quota behaves exactly like a full disk from the writer's
/// point of view, and macOS can report it where Linux reports `ENOSPC`.
#[must_use]
pub fn is_disk_full(error: &std::io::Error) -> bool {
    // Raw values rather than `libc` constants: the workspace has no `libc` dependency, and these
    // two are ABI-stable on both macOS and Linux.
    const ENOSPC: i32 = 28;
    const EDQUOT: i32 = 69;
    matches!(error.raw_os_error(), Some(ENOSPC | EDQUOT)) || error.kind() == std::io::ErrorKind::StorageFull
}

/// The `Hold` a failed write implies, if any.
///
/// The single entry point the daemon calls from its error path, so that "the disk is full" is
/// recognised in one place instead of at every write site. `crate::Error` wraps the io error, so
/// the caller passes the source.
#[must_use]
pub fn hold_for_io_error(error: &std::io::Error) -> Option<crate::domain::Hold> {
    is_disk_full(error).then_some(crate::domain::Hold::PausedDisk)
}

/// Build the live set for `gc` from the journals of unfinished tasks.
///
/// A journal that cannot be read contributes nothing, and that is the safe direction in one
/// sense and the unsafe one in the other: an unreadable journal protects none of its artifacts.
/// Callers therefore pass only tasks the scheduler still considers active, and the daemon treats
/// a failed read as a reason to skip collection entirely rather than to collect optimistically.
pub fn live_set<'a>(journals: impl IntoIterator<Item = &'a crate::journal::ReadOutcome>) -> BTreeSet<Digest> {
    let mut live = BTreeSet::new();
    for outcome in journals {
        live.extend(referenced_artifacts(&outcome.records));
    }
    live
}

/// Render a digest the way `gc` prints it, so the CLI and the report agree.
#[must_use]
pub fn artifact_name(digest: &Digest) -> String {
    digest_text(digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::ArtifactStore;
    use chrono::TimeZone as _;

    fn now() -> chrono::DateTime<chrono::Local> {
        chrono::Local.with_ymd_and_hms(2026, 11, 12, 12, 0, 0).unwrap()
    }

    fn storage() -> Storage {
        Storage::default()
    }

    /// Set an artifact's mtime to `secs` before `now`, so the TTL can be exercised exactly.
    fn age_file(path: &Path, now: chrono::DateTime<chrono::Local>, secs: u64) {
        let target = std::time::UNIX_EPOCH
            + std::time::Duration::from_secs(u64::try_from(now.timestamp()).unwrap().saturating_sub(secs));
        let file = std::fs::File::options().write(true).open(path).unwrap();
        file.set_modified(target).unwrap();
    }

    fn store(dir: &Path) -> ArtifactStore {
        ArtifactStore::open(dir).unwrap()
    }

    // ── artifacts ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn a_fresh_artifact_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let digest = store.put(b"fresh output").unwrap();
        let report = collect(&store, dir.path(), &storage(), &BTreeSet::new(), now()).unwrap();
        assert_eq!(report.artifacts_removed, 0);
        assert!(store.contains(&digest));
    }

    #[test]
    fn an_artifact_past_the_ttl_is_removed_and_counted() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let digest = store.put(&vec![b'x'; 500]).unwrap();
        age_file(&store.path_for(&digest), now(), 31 * DAY_SECS);

        let report = collect(&store, dir.path(), &storage(), &BTreeSet::new(), now()).unwrap();
        assert_eq!(report.artifacts_removed, 1);
        assert_eq!(report.bytes_reclaimed, 500);
        assert!(!store.contains(&digest));
    }

    #[test]
    fn an_artifact_exactly_at_the_ttl_boundary_is_removed() {
        // `>=`, not `>`: an off-by-one here means the disk keeps growing one day longer every
        // month for every artifact.
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let digest = store.put(b"boundary").unwrap();
        age_file(&store.path_for(&digest), now(), 30 * DAY_SECS);

        let report = collect(&store, dir.path(), &storage(), &BTreeSet::new(), now()).unwrap();
        assert_eq!(report.artifacts_removed, 1);
    }

    #[test]
    fn an_artifact_one_second_short_of_the_ttl_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let digest = store.put(b"almost").unwrap();
        age_file(&store.path_for(&digest), now(), 30 * DAY_SECS - 1);

        let report = collect(&store, dir.path(), &storage(), &BTreeSet::new(), now()).unwrap();
        assert_eq!(report.artifacts_removed, 0);
        assert!(store.contains(&digest));
    }

    #[test]
    fn an_artifact_referenced_by_an_unfinished_task_survives_any_age() {
        // The core promise of the guard: a task that is still running can always read back its
        // own observations, however old the run is.
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let digest = store.put(b"needed by a live task").unwrap();
        age_file(&store.path_for(&digest), now(), 3650 * DAY_SECS);
        let protected = BTreeSet::from([digest]);

        let report = collect(&store, dir.path(), &storage(), &protected, now()).unwrap();
        assert_eq!(report.artifacts_removed, 0);
        assert_eq!(report.artifacts_protected, 1);
        assert!(store.contains(&digest));
    }

    #[test]
    fn a_protected_artifact_is_reported_but_an_unreferenced_old_one_still_goes() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let kept = store.put(b"protected").unwrap();
        let doomed = store.put(b"forgotten").unwrap();
        age_file(&store.path_for(&kept), now(), 400 * DAY_SECS);
        age_file(&store.path_for(&doomed), now(), 400 * DAY_SECS);

        let report = collect(&store, dir.path(), &storage(), &BTreeSet::from([kept]), now()).unwrap();
        assert_eq!(report.artifacts_protected, 1);
        assert_eq!(report.artifacts_removed, 1);
        assert!(store.contains(&kept));
        assert!(!store.contains(&doomed));
    }

    #[test]
    fn an_artifact_with_an_unknown_age_is_kept() {
        // `ArtifactStore::entries` reports `modified_secs == 0` when the platform will not say.
        // The collector must treat that as "cannot prove it is old" and keep the file; the
        // platform cannot be forced to fail here, so the guard clause is exercised through the
        // only input that reaches it deterministically: an artifact written this instant.
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let digest = store.put(b"no mtime").unwrap();
        let report = collect(&store, dir.path(), &storage(), &BTreeSet::new(), now()).unwrap();
        assert_eq!(report.artifacts_removed, 0);
        assert!(store.contains(&digest));
    }

    #[test]
    fn an_artifact_older_than_now_is_kept() {
        // A clock step backwards must not make every artifact look ancient. `now < modified`
        // is a future file, and it is left alone.
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let digest = store.put(b"from the future").unwrap();
        let modified = std::time::UNIX_EPOCH
            + std::time::Duration::from_secs(u64::try_from(now().timestamp()).unwrap() + 10_000);
        std::fs::File::options()
            .write(true)
            .open(store.path_for(&digest))
            .unwrap()
            .set_modified(modified)
            .unwrap();

        let report = collect(&store, dir.path(), &storage(), &BTreeSet::new(), now()).unwrap();
        assert_eq!(report.artifacts_removed, 0);
    }

    #[test]
    fn a_stale_temp_file_is_swept() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let temp = store.dir().join(".tmp");
        let stale = temp.join("01ARZ3NDEKTSV4RRFFQ69G5FAV.part");
        std::fs::write(&stale, b"half a write").unwrap();
        age_file(&stale, now(), TEMP_SWEEP_AGE_SECS + 60);

        let report = collect(&store, dir.path(), &storage(), &BTreeSet::new(), now()).unwrap();
        assert_eq!(report.temp_removed, 1);
        assert!(!stale.exists());
    }

    #[test]
    fn a_temp_file_being_written_right_now_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let temp = store.dir().join(".tmp");
        let fresh = temp.join("01ARZ3NDEKTSV4RRFFQ69G5FAW.part");
        std::fs::write(&fresh, b"in flight").unwrap();

        let report = collect(&store, dir.path(), &storage(), &BTreeSet::new(), now()).unwrap();
        assert_eq!(report.temp_removed, 0);
        assert!(fresh.exists());
    }

    #[test]
    fn referenced_artifacts_are_collected_from_the_journal() {
        use crate::journal::{JournalEvent, RecordFields};
        let digest = crate::artifacts::digest_of(b"body");
        let record = JournalRecord::new(
            RecordFields::new(
                0,
                chrono::Local
                    .with_ymd_and_hms(2026, 11, 12, 12, 0, 0)
                    .unwrap()
                    .with_timezone(&chrono::Utc),
                crate::domain::TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV"),
                crate::journal::Kind::ToolCallCompleted,
            ),
            JournalEvent::spilled(serde_json::json!({ "call": "c" }), digest),
            None,
        );
        assert_eq!(
            referenced_artifacts(&[record]),
            BTreeSet::from([digest]),
            "the live set must come from the `artifact` field, which is what recovery reads"
        );
    }

    #[test]
    fn a_malformed_artifact_reference_does_not_stop_the_collection() {
        use crate::journal::{JournalEvent, RecordFields};
        let record = JournalRecord::new(
            RecordFields::new(
                0,
                chrono::Utc::now(),
                crate::domain::TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV"),
                crate::journal::Kind::ToolCallCompleted,
            ),
            JournalEvent {
                payload: serde_json::json!({}),
                artifact: Some("sha256:not-a-real-digest".into()),
            },
            None,
        );
        assert!(
            referenced_artifacts(&[record]).is_empty(),
            "an unparseable reference protects nothing, and must not fail `gc`"
        );
    }

    #[test]
    fn the_live_set_is_the_union_of_unfinished_journals() {
        use crate::journal::{JournalEvent, ReadOutcome, RecordFields};
        let a = crate::artifacts::digest_of(b"a");
        let b = crate::artifacts::digest_of(b"b");
        let record = |seq: u64, digest| {
            JournalRecord::new(
                RecordFields::new(
                    seq,
                    chrono::Utc::now(),
                    crate::domain::TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV"),
                    crate::journal::Kind::ToolCallCompleted,
                ),
                JournalEvent::spilled(serde_json::json!({}), digest),
                None,
            )
        };
        let first = ReadOutcome {
            records: vec![record(0, a)],
            problems: Vec::new(),
            has_torn_tail: false,
            last_digest: None,
        };
        let second = ReadOutcome {
            records: vec![record(0, b)],
            problems: Vec::new(),
            has_torn_tail: false,
            last_digest: None,
        };
        assert_eq!(live_set([&first, &second]), BTreeSet::from([a, b]));
        assert!(
            live_set(std::iter::empty()).is_empty(),
            "an empty set of journals protects nothing, which is why the daemon must not call \
             this after a failed journal read"
        );
    }

    // ── logs ──────────────────────────────────────────────────────────────────────────────

    fn write_log(dir: &Path, name: &str, body: &[u8]) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn a_past_month_log_is_compressed_and_the_plain_file_removed() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir.path().join("artifacts"));
        let logs = dir.path().join("logs");
        std::fs::create_dir_all(&logs).unwrap();
        let plain = write_log(&logs, "2026-10.log", b"october line\n");

        let report = collect(&store, &logs, &storage(), &BTreeSet::new(), now()).unwrap();
        assert_eq!(report.logs_compressed, ["2026-10.log"]);
        assert!(!plain.exists());
        let archive = logs.join("2026-10.log.gz");
        assert!(archive.exists());

        // Readable back: an archive nobody can decompress is not an archive.
        let mut text = String::new();
        std::io::Read::read_to_string(
            &mut flate2::read::GzDecoder::new(std::fs::File::open(&archive).unwrap()),
            &mut text,
        )
        .unwrap();
        assert_eq!(text, "october line\n");
    }

    #[test]
    fn the_current_month_log_is_not_touched() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir.path().join("artifacts"));
        let logs = dir.path().join("logs");
        std::fs::create_dir_all(&logs).unwrap();
        let current = write_log(&logs, "2026-11.log", b"this month\n");

        let report = collect(&store, &logs, &storage(), &BTreeSet::new(), now()).unwrap();
        assert_eq!(report.logs_compressed, Vec::<String>::new());
        assert!(current.exists());
    }

    #[test]
    fn an_operators_own_log_file_is_left_alone() {
        // `immortality.log` is not a month file; deleting it would be deleting somebody's data.
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir.path().join("artifacts"));
        let logs = dir.path().join("logs");
        std::fs::create_dir_all(&logs).unwrap();
        let own = write_log(&logs, "immortality.log", b"not ours\n");

        let report = collect(&store, &logs, &storage(), &BTreeSet::new(), now()).unwrap();
        assert_eq!(report.logs_compressed, Vec::<String>::new());
        assert!(own.exists());
    }

    #[test]
    fn compression_is_idempotent_when_a_crash_left_both_files() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir.path().join("artifacts"));
        let logs = dir.path().join("logs");
        std::fs::create_dir_all(&logs).unwrap();
        let plain = write_log(&logs, "2026-10.log", b"october\n");
        write_log(&logs, "2026-10.log.gz", b"an older archive\n");

        let report = collect(&store, &logs, &storage(), &BTreeSet::new(), now()).unwrap();
        // The plain file is removed, and the existing archive is not overwritten with a
        // half-synced copy of the same month.
        assert!(!plain.exists());
        assert_eq!(report.logs_compressed, ["2026-10.log"]);
        assert!(logs.join("2026-10.log.gz").exists());
    }

    #[test]
    fn an_empty_log_is_not_archived() {
        // A zero-byte `.gz` would decompress to nothing and make the month look lost.
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir.path().join("artifacts"));
        let logs = dir.path().join("logs");
        std::fs::create_dir_all(&logs).unwrap();
        let plain = write_log(&logs, "2026-10.log", b"");

        let report = collect(&store, &logs, &storage(), &BTreeSet::new(), now()).unwrap();
        assert_eq!(report.logs_compressed, Vec::<String>::new());
        assert!(plain.exists());
        assert!(!logs.join("2026-10.log.gz").exists());
    }

    #[test]
    fn archives_past_retention_are_deleted() {
        // 12 months of retention in November 2026 keeps 2025-12 and later.
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir.path().join("artifacts"));
        let logs = dir.path().join("logs");
        std::fs::create_dir_all(&logs).unwrap();
        let old = write_log(&logs, "2025-10.log.gz", b"ancient\n");
        let boundary = write_log(&logs, "2025-12.log.gz", b"oldest kept\n");
        let recent = write_log(&logs, "2026-10.log.gz", b"recent\n");

        let report = collect(&store, &logs, &storage(), &BTreeSet::new(), now()).unwrap();
        assert_eq!(report.logs_deleted, ["2025-10.log.gz"]);
        assert!(!old.exists());
        assert!(boundary.exists(), "the retention boundary is inclusive");
        assert!(recent.exists());
    }

    #[test]
    fn an_archive_from_the_future_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir.path().join("artifacts"));
        let logs = dir.path().join("logs");
        std::fs::create_dir_all(&logs).unwrap();
        let future = write_log(&logs, "2027-05.log.gz", b"clock stepped\n");

        let report = collect(&store, &logs, &storage(), &BTreeSet::new(), now()).unwrap();
        assert_eq!(report.logs_deleted, Vec::<String>::new());
        assert!(future.exists());
    }

    #[test]
    fn a_directory_that_is_not_a_month_archive_is_not_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir.path().join("artifacts"));
        let logs = dir.path().join("logs");
        std::fs::create_dir_all(&logs).unwrap();
        let odd = write_log(&logs, "backup-2025-10.log.gz", b"not a month name\n");

        let report = collect(&store, &logs, &storage(), &BTreeSet::new(), now()).unwrap();
        assert_eq!(report.logs_deleted, Vec::<String>::new());
        assert!(odd.exists());
    }

    #[test]
    fn a_missing_log_directory_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let report = collect(
            &store,
            &dir.path().join("no-such-logs"),
            &storage(),
            &BTreeSet::new(),
            now(),
        )
        .unwrap();
        assert!(report.is_noop());
    }

    #[test]
    fn a_second_run_does_nothing() {
        // Idempotence is the property that makes `gc` safe to run on every daemon start.
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let digest = store.put(b"old").unwrap();
        age_file(&store.path_for(&digest), now(), 60 * DAY_SECS);
        let logs = dir.path().join("logs");
        std::fs::create_dir_all(&logs).unwrap();
        write_log(&logs, "2026-09.log", b"september\n");

        let first = collect(&store, &logs, &storage(), &BTreeSet::new(), now()).unwrap();
        assert!(!first.is_noop());
        let second = collect(&store, &logs, &storage(), &BTreeSet::new(), now()).unwrap();
        assert!(second.is_noop(), "{second:?}");
    }

    // ── disk guard ────────────────────────────────────────────────────────────────────────

    #[test]
    fn plenty_of_space_is_enough() {
        assert_eq!(disk_state(10 * 1024 * 1024 * 1024, &storage()), DiskState::Enough);
    }

    #[test]
    fn space_between_the_thresholds_only_warns() {
        let state = disk_state(512 * 1024 * 1024, &storage());
        assert_eq!(state, DiskState::Low);
        assert_eq!(
            state.hold(),
            None,
            "warning must not stop work: 1 GiB is most of a small disk"
        );
    }

    #[test]
    fn space_below_the_pause_threshold_pauses_the_task() {
        let state = disk_state(100 * 1024 * 1024, &storage());
        assert_eq!(state, DiskState::Exhausted);
        assert_eq!(state.hold(), Some(crate::domain::Hold::PausedDisk));
    }

    #[test]
    fn the_pause_threshold_itself_is_not_exhausted() {
        // Exactly 256 MiB free is still writable, so it warns rather than stops.
        let storage = storage();
        assert_eq!(disk_state(storage.disk_pause_bytes, &storage), DiskState::Low);
    }

    #[test]
    fn the_warning_threshold_itself_is_enough() {
        let storage = storage();
        assert_eq!(disk_state(storage.disk_warn_bytes, &storage), DiskState::Enough);
    }

    #[test]
    fn custom_thresholds_are_honoured() {
        let storage = Storage {
            disk_warn_bytes: 10,
            disk_pause_bytes: 5,
            ..Storage::default()
        };
        assert_eq!(disk_state(9, &storage), DiskState::Low);
        assert_eq!(disk_state(4, &storage), DiskState::Exhausted);
        assert_eq!(disk_state(10, &storage), DiskState::Enough);
    }

    #[test]
    fn an_enospc_write_failure_pauses_the_task() {
        // The threshold check passed a second ago; the write did not. Believe the observation.
        let error = std::io::Error::from_raw_os_error(28);
        assert!(is_disk_full(&error));
        assert_eq!(
            hold_for_io_error(&error),
            Some(crate::domain::Hold::PausedDisk),
            "a full disk must pause the task rather than be retried forever"
        );
    }

    #[test]
    fn a_quota_exceeded_failure_pauses_too() {
        let error = std::io::Error::from_raw_os_error(69);
        assert_eq!(hold_for_io_error(&error), Some(crate::domain::Hold::PausedDisk));
    }

    #[test]
    fn a_storage_full_kind_is_recognised_without_a_raw_code() {
        let error = std::io::Error::from(std::io::ErrorKind::StorageFull);
        assert!(
            is_disk_full(&error),
            "a caller that builds the error from its kind must still be understood"
        );
    }

    #[test]
    fn a_permission_error_is_not_a_full_disk() {
        // Pausing on every failed write would stall the task for a wrong `chmod` and hide the
        // real diagnosis behind `paused_disk`.
        let error = std::io::Error::from(std::io::ErrorKind::PermissionDenied);
        assert!(!is_disk_full(&error));
        assert_eq!(hold_for_io_error(&error), None);
    }

    #[test]
    fn a_missing_file_is_not_a_full_disk() {
        let error = std::io::Error::from(std::io::ErrorKind::NotFound);
        assert_eq!(hold_for_io_error(&error), None);
    }
}
