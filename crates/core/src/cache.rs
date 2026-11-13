//! The derived-state cache: `attempts.json` and `subtasks.json`.
//!
//! # Why these files exist at all
//!
//! `status` answers in milliseconds because it reads two small JSON files instead of replaying a
//! journal that may hold a hundred thousand records. The trade is explicit: the cache is
//! *derived*, the journal is the truth, and whenever the two disagree the journal wins and the
//! file is rewritten from it (see [`crate::recover`]).
//!
//! # Why writing the cache lives here and not in `recover`
//!
//! Recovery decides *that* a file is stale; this module knows *what the file looks like*. Keeping
//! the schema in one place means the writer, the reader and the divergence check cannot drift —
//! and they would drift first, because all three are written by hand.
//!
//! # Why the file has a version
//!
//! A format change under a `v: 1` reader is how a cache becomes a silent producer of wrong
//! answers. A file with an unknown `v` is reported as a problem and rebuilt, never parsed
//! optimistically.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::atomic::write_atomically;
use crate::domain::SubtaskState;
use crate::ids::SubtaskId;
use crate::journal::JournalRecord;
use crate::recover::{PersistedAttempt, PersistedState, PersistedSubtask};

/// File name for the attempt cache.
pub const ATTEMPTS_FILE: &str = "attempts.json";

/// File name for the subtask-state cache.
pub const SUBTASKS_FILE: &str = "subtasks.json";

const SCHEMA_VERSION: u16 = 1;

/// Where the two cache files for a task live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cache {
    dir: PathBuf,
}

impl Cache {
    /// A cache over `dir`, which is created on the first write.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The directory holding the files.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Path of the attempt cache.
    #[must_use]
    pub fn attempts_path(&self) -> PathBuf {
        self.dir.join(ATTEMPTS_FILE)
    }

    /// Path of the subtask cache.
    #[must_use]
    pub fn subtasks_path(&self) -> PathBuf {
        self.dir.join(SUBTASKS_FILE)
    }

    /// Read both files, treating any of "absent", "unreadable", "wrong version" as empty.
    ///
    /// Empty is the safe answer: a divergence check against an empty cache reports nothing, and
    /// the plan then rewrites the file from the journal only if the journal has something to say.
    /// Refusing to start because a *cache* is corrupt would be the tail wagging the dog.
    #[must_use]
    pub fn read(&self) -> PersistedState {
        PersistedState {
            attempts: self.read_attempts(),
            subtasks: self.read_subtasks(),
        }
    }

    fn read_attempts(&self) -> Vec<PersistedAttempt> {
        let Some(body) = read_json::<AttemptsBody>(&self.attempts_path()) else {
            return Vec::new();
        };
        if body.v != SCHEMA_VERSION {
            return Vec::new();
        }
        body.attempts
            .into_iter()
            .map(|entry| PersistedAttempt {
                subtask: entry.subtask,
                number: entry.number,
                finished: entry.finished,
            })
            .collect()
    }

    fn read_subtasks(&self) -> Vec<PersistedSubtask> {
        let Some(body) = read_json::<SubtasksBody>(&self.subtasks_path()) else {
            return Vec::new();
        };
        if body.v != SCHEMA_VERSION {
            return Vec::new();
        }
        body.subtasks
            .into_iter()
            .map(|entry| PersistedSubtask {
                id: entry.subtask,
                state: entry.state,
            })
            .collect()
    }

    /// Replace both files with `state`, atomically.
    ///
    /// # Errors
    /// The directory cannot be created, or a file cannot be written.
    pub fn write(&self, state: &PersistedState) -> crate::Result<()> {
        let attempts = AttemptsBody {
            v: SCHEMA_VERSION,
            attempts: state
                .attempts
                .iter()
                .map(|attempt| AttemptEntry {
                    subtask: attempt.subtask.clone(),
                    number: attempt.number,
                    finished: attempt.finished,
                })
                .collect(),
        };
        let subtasks = SubtasksBody {
            v: SCHEMA_VERSION,
            subtasks: state
                .subtasks
                .iter()
                .map(|subtask| SubtaskEntry {
                    subtask: subtask.id.clone(),
                    state: subtask.state,
                })
                .collect(),
        };
        write_json(&self.attempts_path(), &attempts)?;
        write_json(&self.subtasks_path(), &subtasks)
    }
}

/// Derive the cache contents from a journal.
///
/// The rules are the journal's own: an attempt is `finished` when the journal closed it (failed,
/// restarted, passed its gates, or the subtask was marked done), and a subtask's state is the last
/// `subtask_state` record written for it.
#[must_use]
pub fn derived(records: &[JournalRecord]) -> PersistedState {
    let mut attempts: Vec<PersistedAttempt> = Vec::new();
    let mut open: std::collections::BTreeSet<(SubtaskId, u32)> = std::collections::BTreeSet::new();
    let mut closed: std::collections::BTreeSet<(SubtaskId, u32)> = std::collections::BTreeSet::new();
    let mut subtasks: Vec<PersistedSubtask> = Vec::new();

    for record in records {
        if let (Some(subtask), Some(number)) = (record.fields.subtask.clone(), record.fields.attempt) {
            let key = (subtask, number);
            match record.fields.kind {
                crate::journal::Kind::AttemptStarted => {
                    open.insert(key);
                }
                crate::journal::Kind::AttemptFailed | crate::journal::Kind::AttemptRestarted => {
                    closed.insert(key);
                }
                crate::journal::Kind::GateResult if verdict_passed(record) => {
                    closed.insert(key);
                }
                crate::journal::Kind::SubtaskState if subtask_state(record) == Some(SubtaskState::Done) => {
                    closed.insert(key);
                }
                _ => {}
            }
        }
        if record.fields.kind == crate::journal::Kind::SubtaskState
            && let (Some(subtask), Some(state)) = (record.fields.subtask.clone(), subtask_state(record))
        {
            match subtasks.iter_mut().find(|entry| entry.id == subtask) {
                Some(entry) => entry.state = state,
                None => subtasks.push(PersistedSubtask { id: subtask, state }),
            }
        }
    }

    for (subtask, number) in open.union(&closed) {
        attempts.push(PersistedAttempt {
            subtask: subtask.clone(),
            number: *number,
            finished: closed.contains(&(subtask.clone(), *number)),
        });
    }
    attempts.sort_by(|left, right| {
        left.subtask
            .cmp(&right.subtask)
            .then_with(|| left.number.cmp(&right.number))
    });
    subtasks.sort_by(|left, right| left.id.cmp(&right.id));

    PersistedState { attempts, subtasks }
}

/// Whether a record is the `gate_result` verdict that says the gates passed.
fn verdict_passed(record: &JournalRecord) -> bool {
    let payload = &record.event.payload;
    payload.get("gate").and_then(|gate| gate.as_str()) == Some("verdict")
        && payload.get("passed").and_then(serde_json::Value::as_bool) == Some(true)
}

fn subtask_state(record: &JournalRecord) -> Option<SubtaskState> {
    record
        .event
        .payload
        .get("state")
        .and_then(|value| serde_json::from_value(value.clone()).ok())
}

#[derive(Debug, Serialize, Deserialize)]
struct AttemptsBody {
    v: u16,
    attempts: Vec<AttemptEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AttemptEntry {
    subtask: SubtaskId,
    number: u32,
    finished: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct SubtasksBody {
    v: u16,
    subtasks: Vec<SubtaskEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SubtaskEntry {
    subtask: SubtaskId,
    state: SubtaskState,
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> crate::Result<()> {
    let mut text = serde_json::to_string_pretty(value)
        .map_err(|error| crate::Error::Protocol(format!("cannot serialize {}: {error}", path.display())))?;
    text.push('\n');
    write_atomically(path, text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::Kind;
    use crate::journal::{JournalEvent, RecordFields, Scope};

    fn task() -> crate::ids::TaskId {
        crate::ids::TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV")
    }

    fn subtask() -> SubtaskId {
        SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV")
    }

    fn record(seq: u64, kind: Kind, scope: Scope, payload: serde_json::Value) -> JournalRecord {
        JournalRecord::new(
            RecordFields {
                seq,
                at: crate::domain::Timestamp::UNIX_EPOCH,
                task: task(),
                subtask: scope.subtask,
                attempt: scope.attempt,
                kind,
            },
            JournalEvent::with_payload(payload),
            None,
        )
    }

    #[test]
    fn a_started_but_unfinished_attempt_is_not_finished() {
        let records = vec![record(
            0,
            Kind::AttemptStarted,
            Scope::attempt(subtask(), 1),
            serde_json::json!({}),
        )];
        let state = derived(&records);
        assert_eq!(state.attempts.len(), 1);
        assert!(!state.attempts[0].finished);
    }

    #[test]
    fn a_failed_attempt_is_finished() {
        let records = vec![
            record(
                0,
                Kind::AttemptStarted,
                Scope::attempt(subtask(), 1),
                serde_json::json!({}),
            ),
            record(
                1,
                Kind::AttemptFailed,
                Scope::attempt(subtask(), 1),
                serde_json::json!({}),
            ),
        ];
        let state = derived(&records);
        assert!(state.attempts[0].finished);
    }

    #[test]
    fn a_green_verdict_closes_the_attempt() {
        // An attempt that passed its gates has no `attempt_failed`, and calling it unfinished
        // inflates the attempt budget for work that succeeded.
        let records = vec![
            record(
                0,
                Kind::AttemptStarted,
                Scope::attempt(subtask(), 1),
                serde_json::json!({}),
            ),
            record(
                1,
                Kind::GateResult,
                Scope::attempt(subtask(), 1),
                serde_json::json!({"gate": "verdict", "passed": true}),
            ),
        ];
        assert!(derived(&records).attempts[0].finished);
    }

    #[test]
    fn a_red_verdict_does_not_close_the_attempt() {
        let records = vec![
            record(
                0,
                Kind::AttemptStarted,
                Scope::attempt(subtask(), 1),
                serde_json::json!({}),
            ),
            record(
                1,
                Kind::GateResult,
                Scope::attempt(subtask(), 1),
                serde_json::json!({"gate": "verdict", "passed": false}),
            ),
        ];
        assert!(!derived(&records).attempts[0].finished);
    }

    #[test]
    fn the_last_subtask_state_wins() {
        let records = vec![
            record(
                0,
                Kind::SubtaskState,
                Scope::subtask(subtask()),
                serde_json::json!({"state": "running"}),
            ),
            record(
                1,
                Kind::SubtaskState,
                Scope::subtask(subtask()),
                serde_json::json!({"state": "done"}),
            ),
        ];
        let state = derived(&records);
        assert_eq!(state.subtasks.len(), 1);
        assert_eq!(state.subtasks[0].state, SubtaskState::Done);
    }

    #[test]
    fn a_round_trip_preserves_both_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = Cache::new(dir.path());
        let state = PersistedState {
            attempts: vec![PersistedAttempt {
                subtask: subtask(),
                number: 2,
                finished: true,
            }],
            subtasks: vec![PersistedSubtask {
                id: subtask(),
                state: SubtaskState::Blocked,
            }],
        };
        cache.write(&state).expect("write");
        assert_eq!(cache.read(), state);
    }

    #[test]
    fn an_absent_cache_reads_as_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(Cache::new(dir.path()).read(), PersistedState::default());
    }

    #[test]
    fn a_corrupt_cache_reads_as_empty_rather_than_refusing_to_start() {
        // A cache is derived; refusing to start because it is unreadable would hand a cache veto
        // power over the journal.
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = Cache::new(dir.path());
        std::fs::write(cache.attempts_path(), "{not json").expect("write");
        std::fs::write(cache.subtasks_path(), r#"{"v":99,"subtasks":[]}"#).expect("write");
        assert_eq!(cache.read(), PersistedState::default());
    }

    #[test]
    fn the_files_are_human_readable() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = Cache::new(dir.path());
        cache.write(&PersistedState::default()).expect("write");
        let text = std::fs::read_to_string(cache.attempts_path()).expect("read");
        assert!(text.contains("\"v\": 1"), "{text}");
        assert!(
            text.ends_with('\n'),
            "a file that ends without a newline reads as torn"
        );
    }
}
