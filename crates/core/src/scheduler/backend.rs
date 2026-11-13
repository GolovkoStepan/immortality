//! What to do while the model backend is unreachable.
//!
//! # Why this is a decision and not `if error { sleep }`
//!
//! Three things have to be true at once and each has its own rule:
//!
//! * the task stays **alive** — an outage is not a failure of the work, so it must not spend an
//!   attempt or a restart (D-027);
//! * offline work **continues** — a backend outage is a reason not to think, not a reason to leave
//!   the tree dirty;
//! * after fifteen minutes of *continuous* unavailability a **human** is told, because an outage
//!   that long is news (the design's `unavailable_escalate_after_secs`).
//!
//! The third is the one that goes wrong when it is written inline as a timer: a task that sees the
//! backend at minute fourteen and loses it again at minute fifteen has not been out for fifteen
//! continuous minutes, and a deadline armed at the first failure would escalate about a healthy
//! backend. So the rule is stated over an interval, not over a count of failures.
//!
//! # Why the escalation does not stop the task
//!
//! `backend_unavailable` is not a blocking kind (only `journal_corrupt` is). The task keeps its
//! `waiting_backend` hold, retries, and continues by itself when the backend returns. A human who
//! is told is a human who can fix the backend, not one who must answer a question before work
//! resumes — there is nothing to decide.

use crate::domain::Timestamp;
use crate::escalation::{Answer, Entry, EscalationKind};

/// The deduplication key of the outage escalation.
///
/// Constant, so that a second outage during the same task finds the first question instead of
/// opening a second one. `EscalationStore::ask` is the thing that enforces it; this is the key it
/// is given.
pub const KEY: &str = "backend:unavailable";

/// What the loop does about the backend right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum While {
    /// The backend answered. Nothing to do.
    Available,
    /// It is down, but for less than the threshold: wait, and keep doing offline work.
    Wait,
    /// It has been down continuously for at least the threshold: wait, do offline work, and tell a
    /// human. Emitted on the crossing only — the caller must not open the same question on every
    /// turn, and it does not need to: the key deduplicates, but a journal record per turn is noise.
    Escalate,
}

/// Decide what to do given when the backend was last known good and what time it is.
///
/// `since` is the moment the backend was first found unavailable, `None` while it is healthy. The
/// caller clears it as soon as a request succeeds, which is what makes the interval *continuous*:
/// this function cannot tell a reset from a continuation and does not need to, because the reset is
/// the caller's single line.
#[must_use]
pub fn while_unavailable(since: Option<Timestamp>, now: Timestamp, escalate_after_secs: u64) -> While {
    let Some(since) = since else {
        return While::Available;
    };
    let elapsed = now.signed_duration_since(since).num_seconds();
    if elapsed >= i64::try_from(escalate_after_secs).unwrap_or(i64::MAX) {
        While::Escalate
    } else {
        While::Wait
    }
}

/// Whether `now` is past the threshold, for a caller that has already recorded the escalation.
///
/// Separate from [`while_unavailable`] so that "the human has been told" and "the threshold is
/// reached" stay distinguishable: the first is written to the queue and survives a restart, the
/// second is arithmetic.
#[must_use]
pub fn elapsed_secs(since: Timestamp, now: Timestamp) -> i64 {
    now.signed_duration_since(since).num_seconds()
}

/// The question a human is asked when the backend has been out too long.
///
/// One option, and it is [`Answer::AllowOnce`] — the answer that changes no permission and grants
/// nothing beyond this notification. There is nothing to decide: the task continues on its own when
/// the backend returns. Offering "retry" and "abort" would be a menu whose only honest answer is
/// "wait", and a question that pretends to have options trains a human to click through without
/// reading.
///
/// `AllowOnce` rather than a new variant because [`Answer`] is a closed set of *permission*
/// answers, consumed by `approve`, `Answer::parse` and [`Answer::permits`]. A fourth variant would
/// ripple through all three to express "this one is not a permission question", which is a fact
/// this entry's `kind` already carries. When a human sees `backend_unavailable`, nothing is being
/// asked of them; the journal and the queue are telling them the backend is down.
#[must_use]
pub fn escalation(task: crate::ids::TaskId, since: Timestamp, now: Timestamp, at_secs: i64) -> Entry {
    Entry::new(
        task,
        KEY,
        EscalationKind::BackendUnavailable,
        format!(
            "The model backend has been unreachable since {} ({} minutes). The task is still \
             alive and keeps doing work that does not need the model; it resumes by itself when \
             the backend answers. Nothing needs deciding — this is a notification.",
            crate::time::truncate(since).to_rfc3339(),
            at_secs / 60
        ),
        vec![Answer::AllowOnce],
        now,
    )
}

/// When the backend was first found unavailable, from the journal alone.
///
/// # Why the journal and not a field in memory
///
/// A daemon killed during a fifteen-minute outage must not restart the clock. The journal is the
/// only record that survives `kill -9`, and `backend_unavailable` is already written there; the
/// question is only how to read it back without a clock.
///
/// # Why the *last* record wins
///
/// A `backend_restored` record clears the interval. So the answer is the timestamp of the last
/// `backend_unavailable` that is not followed by a `backend_restored` — reading it as "the first
/// outage ever" would make every later outage look fifteen minutes old the moment it starts.
#[must_use]
pub fn since_from_journal(records: &[crate::journal::JournalRecord]) -> Option<Timestamp> {
    let mut since: Option<Timestamp> = None;
    for record in records {
        match record.fields.kind {
            crate::journal::Kind::BackendUnavailable => since = Some(record.fields.at),
            crate::journal::Kind::BackendRestored => since = None,
            _ => {}
        }
    }
    since
}

/// The payload of the journal record announcing the outage.
///
/// `since` and `at_secs` are both recorded: the first is what the next process needs to keep
/// measuring, the second is what a human reading the journal wants. Deriving either from the other
/// needs a clock, and a journal reader should not need one.
#[must_use]
pub fn unavailable_payload(since: Timestamp, now: Timestamp, at_secs: i64) -> serde_json::Value {
    serde_json::json!({
        "service": KEY,
        "since": crate::time::truncate(since),
        "at": crate::time::truncate(now),
        "unavailable_for_secs": at_secs,
    })
}

#[cfg(test)]
mod tests;
