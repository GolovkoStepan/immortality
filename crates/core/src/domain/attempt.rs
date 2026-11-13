//! Attempts: one bounded pass at a subtask with a fresh context.
//!
//! An attempt is the unit that *costs* something: `max_steps_per_attempt`,
//! `max_tokens_per_attempt` and roughly 45 minutes of wall time. When it runs out, the
//! subtask does not fail — a new attempt starts, optionally after rolling the worktree back
//! to the last green checkpoint and with a different strategy in its brief.
//!
//! # Why the nonce lives on the attempt
//!
//! Recovery has to answer "did this `git commit` already happen?" for a process that died
//! mid-call. The answer is found by searching commit messages and process environments for a
//! token, and the token must be unique per attempt rather than per subtask: the same subtask
//! legitimately tries `cargo fmt` twice, and a per-subtask token would make the second one
//! look like an already-completed effect.
//!
//! # Why strategy is a string
//!
//! L3 escalation restarts an attempt with "a different strategy in the brief", and what
//! counts as different is model-facing prose, not an enum the code branches on. Keeping it a
//! string lets the planner write anything and keeps the type honest about what it knows.

use serde::{Deserialize, Serialize};

use super::{AttemptId, NonceTag, StatusError, SubtaskId, Timestamp};

/// Where an attempt ended up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AttemptState {
    /// Steps are being executed right now.
    Running,
    /// The model claimed done and every gate passed.
    Succeeded,
    /// The attempt ended without passing: budget exhausted, tool error, or a red gate.
    Failed,
    /// The attempt was cut short from outside: a crash, a daemon restart, or an L3 rollback.
    ///
    /// Distinct from [`Failed`](Self::Failed) because it says nothing about the model's
    /// work and must not be counted against the subtask's failure budget: after a machine
    /// reboot, a subtask should not be one attempt closer to `blocked`.
    Abandoned,
}

closed_state!(AttemptState {
    Running => "running",
    Succeeded => "succeeded",
    Failed => "failed",
    Abandoned => "abandoned",
});

/// How an attempt ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AttemptTrigger {
    /// All gates green.
    Succeed,
    /// The attempt itself went wrong: budget, tool, or gate failure attributable to it.
    Fail,
    /// The process or daemon went away; nothing is known about the work.
    Abandon,
}

closed_state!(AttemptTrigger {
    Succeed => "succeed",
    Fail => "fail",
    Abandon => "abandon",
});

impl AttemptState {
    /// Whether anything further can happen to this attempt.
    #[must_use]
    pub const fn is_finished(self) -> bool {
        !matches!(self, Self::Running)
    }

    /// Whether this attempt counts against `max_attempts_per_subtask`.
    ///
    /// `Abandoned` deliberately does not: an attempt killed by a reboot is not evidence that
    /// the model's approach was wrong, and counting it would block a subtask after three
    /// unlucky restarts.
    #[must_use]
    pub const fn counts_as_failure(self) -> bool {
        matches!(self, Self::Failed)
    }

    pub const fn trigger(self, trigger: AttemptTrigger) -> Result<Self, StatusError> {
        let next = match (self, trigger) {
            (Self::Running, AttemptTrigger::Succeed) => Self::Succeeded,
            (Self::Running, AttemptTrigger::Fail) => Self::Failed,
            (Self::Running, AttemptTrigger::Abandon) => Self::Abandoned,
            (state, trigger) => {
                return Err(StatusError::Attempt { from: state, trigger });
            }
        };
        Ok(next)
    }
}

/// Why an attempt stopped before succeeding.
///
/// Recorded on the attempt so that the next brief can name the previous failure and so that
/// the detectors have a metric to act on. Free-form detail is in the journal, not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureReason {
    /// `max_steps_per_attempt` reached.
    StepsExhausted,
    /// `max_tokens_per_attempt` reached.
    TokensExhausted,
    /// Wall-clock budget reached.
    TimeExhausted,
    /// The model looped and the escalation ladder ran out at L5.
    Looped,
    /// The subtask was decomposed by the model into smaller pieces.
    ///
    /// Not a failure of the model's work: the subtask was too large, which is what the split
    /// says. It is its own reason so a metric query can separate "too big" from "too hard".
    Split,
    /// A required command from `ACCEPTANCE.md` failed.
    GateFailed,
    /// The reviewer asked for changes and the changes were not made.
    ReviewRejected,
    /// The backend kept failing beyond the retry policy.
    BackendUnavailable,
    /// The tool layer reported an unrecoverable error.
    ToolError,
    /// A human cancelled the subtask mid-attempt.
    Cancelled,
}

/// One bounded pass at a subtask.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attempt {
    pub id: AttemptId,
    pub subtask: SubtaskId,
    /// 1-based; the first attempt is 1, which is what a human expects to read.
    pub number: u32,
    /// What this pass tries differently, in model-facing prose.
    pub strategy: String,
    /// Embedded in commit messages and child environments for crash-time probes.
    pub nonce: NonceTag,
    pub state: AttemptState,
    /// Checkpoint the worktree was rolled back to before this attempt, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_checkpoint: Option<String>,
    pub started_at: Timestamp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<Timestamp>,
    pub steps_used: u32,
    pub tokens_used: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<FailureReason>,
}

impl Attempt {
    /// Start attempt `number` for a subtask.
    #[must_use]
    pub fn start(subtask: SubtaskId, number: u32, strategy: impl Into<String>, now: Timestamp) -> Self {
        Self {
            id: AttemptId::generate(),
            subtask,
            number,
            strategy: strategy.into(),
            // A fresh nonce per attempt: see the module docs.
            nonce: NonceTag::new(super::super::ids::Nonce::generate().to_string()),
            state: AttemptState::Running,
            base_checkpoint: None,
            started_at: now,
            finished_at: None,
            steps_used: 0,
            tokens_used: 0,
            failure: None,
        }
    }

    /// Record a step's cost. Called once per model call, not per tool call.
    pub const fn charge_step(&mut self, tokens: u64) {
        self.steps_used = self.steps_used.saturating_add(1);
        self.tokens_used = self.tokens_used.saturating_add(tokens);
    }

    /// End the attempt.
    ///
    /// A reason is mandatory for `Fail`: an attempt that failed without a recorded cause is
    /// unusable for the next brief and invisible in `status`, which is how silent loops
    /// start. `Succeed` and `Abandon` reject one.
    pub fn finish(
        &mut self,
        trigger: AttemptTrigger,
        failure: Option<FailureReason>,
        now: Timestamp,
    ) -> Result<AttemptState, super::StatusError> {
        if trigger == AttemptTrigger::Fail && failure.is_none() {
            return Err(super::StatusError::Attempt {
                from: self.state,
                trigger,
            });
        }
        let next = self.state.trigger(trigger)?;
        self.state = next;
        self.failure = failure;
        self.finished_at = Some(now);
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> Timestamp {
        chrono::TimeZone::timestamp_opt(&chrono::Utc, 1_700_000_000, 0)
            .single()
            .unwrap()
    }

    fn attempt() -> Attempt {
        Attempt::start(SubtaskId::generate(), 1, "try the obvious thing", now())
    }

    #[test]
    fn a_fresh_attempt_runs_with_a_nonce() {
        let a = attempt();
        assert_eq!(a.state, AttemptState::Running);
        assert_eq!(a.number, 1);
        assert_ne!(a.nonce.as_str(), "");
        assert!(a.finished_at.is_none());
    }

    #[test]
    fn nonces_differ_between_attempts_of_one_subtask() {
        let subtask = SubtaskId::generate();
        let first = Attempt::start(subtask.clone(), 1, "a", now());
        let second = Attempt::start(subtask, 2, "b", now());
        assert_ne!(first.nonce, second.nonce);
    }

    #[test]
    fn steps_and_tokens_accumulate() {
        let mut a = attempt();
        a.charge_step(1_000);
        a.charge_step(2_500);
        assert_eq!(a.steps_used, 2);
        assert_eq!(a.tokens_used, 3_500);
    }

    #[test]
    fn failing_requires_a_reason() {
        let mut a = attempt();
        assert!(a.finish(AttemptTrigger::Fail, None, now()).is_err());
        assert_eq!(a.state, AttemptState::Running, "state must be untouched");
        a.finish(AttemptTrigger::Fail, Some(FailureReason::StepsExhausted), now())
            .unwrap();
        assert_eq!(a.state, AttemptState::Failed);
        assert_eq!(a.failure, Some(FailureReason::StepsExhausted));
    }

    #[test]
    fn abandoned_does_not_count_as_failure() {
        let mut a = attempt();
        a.finish(AttemptTrigger::Abandon, None, now()).unwrap();
        assert_eq!(a.state, AttemptState::Abandoned);
        assert!(!a.state.counts_as_failure());
    }

    #[test]
    fn a_finished_attempt_cannot_be_finished_again() {
        let mut a = attempt();
        a.finish(AttemptTrigger::Succeed, None, now()).unwrap();
        assert!(
            a.finish(AttemptTrigger::Fail, Some(FailureReason::Looped), now())
                .is_err()
        );
        assert_eq!(a.state, AttemptState::Succeeded);
    }

    #[test]
    fn failure_reasons_serialize_as_snake_case() {
        let json = serde_json::to_string(&FailureReason::GateFailed).unwrap();
        assert_eq!(json, "\"gate_failed\"");
    }

    #[test]
    fn attempt_roundtrips_through_json() {
        let a = attempt();
        let json = serde_json::to_string(&a).unwrap();
        assert_eq!(serde_json::from_str::<Attempt>(&json).unwrap(), a);
    }
}
