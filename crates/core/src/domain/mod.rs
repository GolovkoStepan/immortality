//! Domain types and their state machines.
//!
//! Three machines, one shape: a closed set of states, a closed set of triggers, and an
//! explicit table of allowed `(state, trigger) → state` edges. Nothing mutates a state
//! field directly; a caller always says *why* it wants a change, and the machine decides
//! whether that is legal.
//!
//! The reason for such strictness is recovery. After a crash the journal is read back and
//! replayed against the state machines; if any record could be applied to any state, a
//! corrupt or hand-edited journal would silently produce an impossible task. Here it
//! produces an error naming the exact edge that does not exist.
//!
//! # Suspension is not a state
//!
//! `waiting_backend` and `paused_disk` look like lifecycle stages in the design document,
//! and the CLI shows them as such. They are modelled separately, as [`Hold`], because
//! they are orthogonal to progress: a task is not "waiting for the backend" *instead of*
//! "verifying", it is verifying *while* the backend is unreachable. Folding them into
//! [`TaskState`] would lose the state to return to, and a lossy `waiting_backend → running`
//! edge would quietly skip verification for every task that outlived a network outage.
//!
//! # What lives here
//!
//! Pure logic only: no filesystem, no clock reads, no journal. Timestamps are passed in,
//! so tests run against a fake clock and the same code path serves the daemon.

pub mod attempt;
pub mod dod;
pub mod subtask;
pub mod task;

pub use attempt::{Attempt, AttemptState, AttemptTrigger, FailureReason};
pub use dod::{Dod, DodCheck, DodInvalid};
pub use subtask::{Subtask, SubtaskReport, SubtaskState, SubtaskTrigger};
pub use task::{Hold, Task, TaskState, TaskTrigger};

pub use crate::ids::{AttemptId, Nonce, SubtaskId, TaskId};

/// An instant in the journal's time base. Always UTC: local time is a display concern.
///
/// Serialized with [`crate::time::millis`] wherever it appears in a file, so that its width
/// does not depend on whether the nanoseconds happened to be zero.
pub type Timestamp = chrono::DateTime<chrono::Utc>;

/// A refused state transition.
///
/// Every variant names the concrete situation so that callers can react instead of matching
/// on strings. The distinction matters: an edge that does not exist is a caller bug, while
/// a suspended task that refuses to advance is an expected condition that the scheduler
/// must act on.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StatusError {
    #[error("task state `{from}` does not accept trigger `{trigger}`")]
    Task { from: TaskState, trigger: TaskTrigger },

    #[error("subtask state `{from}` does not accept trigger `{trigger}`")]
    Subtask {
        from: SubtaskState,
        trigger: SubtaskTrigger,
    },

    #[error("attempt state `{from}` does not accept trigger `{trigger}`")]
    Attempt {
        from: AttemptState,
        trigger: AttemptTrigger,
    },

    #[error("task is already finished as `{state}`")]
    Finished { state: TaskState },

    #[error("task is suspended on `{hold}`; clear the hold before advancing")]
    Suspended { hold: Hold },

    #[error("task is already suspended on `{hold}`")]
    AlreadySuspended { hold: Hold },

    #[error("task is not suspended; there is nothing to clear")]
    NotSuspended,

    #[error("task is suspended on `{hold}`, which `{cleared}` does not clear")]
    WrongHold { hold: Hold, cleared: Hold },
}

/// A persisted value that does not name a known state.
///
/// Separate from [`StatusError`] because it is a schema problem, not a lifecycle problem:
/// the file was written by a newer version, by a hand edit, or by a bug. The caller must
/// stop and escalate rather than guess.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown {kind} state `{value}`")]
pub struct UnknownState {
    pub kind: &'static str,
    pub value: String,
}

impl UnknownState {
    pub(crate) fn new(kind: &'static str, value: &str) -> Self {
        Self {
            kind,
            value: value.to_string(),
        }
    }
}

/// Where an attempt's nonce travels.
///
/// The nonce is the only way to answer "did this `git commit` already happen?" after a
/// crash: it is embedded in the commit message and in the child process environment, and
/// nothing else in the system is unique per attempt.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct NonceTag(String);

impl NonceTag {
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for NonceTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
