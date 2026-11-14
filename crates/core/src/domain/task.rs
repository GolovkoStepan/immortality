//! The task lifecycle: the top-level machine the CLI shows as `status`.
//!
//! ```text
//! queued → planning → awaiting_plan_approval ─┐
//!            │                                │
//!            └────────────── plan approved ───┴→ running → verifying ─→ ready_for_review → merged
//!                                                  │  ↑        │   │
//!                                         block ───┘  │        │   └→ failed_gates ────┐
//!                                                  unblock      │         │             │
//!                                                      └────────┴─────────┘ ← retry ───┘
//! ```
//!
//! `blocked` is reachable from execution *and* from verification: a task whose subtasks are
//! exhausted cannot pass gates either, and making the caller route through `failed_gates`
//! would invent a gate verdict that never happened.
//!
//! The plan may be approved implicitly. Planning by default does not stop for a human, so
//! `plan_approved` is legal straight out of `planning`; `awaiting_plan_approval` exists for
//! the case where a human asked to review before execution.

use serde::{Deserialize, Serialize};

use super::{StatusError, Subtask, TaskId, Timestamp};

/// Where the task is in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TaskState {
    Queued,
    Planning,
    AwaitingPlanApproval,
    Running,
    Verifying,
    ReadyForReview,
    Blocked,
    FailedGates,
    Merged,
    Cancelled,
}

closed_state!(TaskState {
    Queued => "queued",
    Planning => "planning",
    AwaitingPlanApproval => "awaiting_plan_approval",
    Running => "running",
    Verifying => "verifying",
    ReadyForReview => "ready_for_review",
    Blocked => "blocked",
    FailedGates => "failed_gates",
    Merged => "merged",
    Cancelled => "cancelled",
});

/// Why a task should move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TaskTrigger {
    /// A worker picked the queued task up.
    StartPlanning,
    /// `PLAN.md`, `ACCEPTANCE.md` and `TASKS.json` are on disk.
    PlanWritten,
    /// Execution may begin: either no human review was requested, or it happened.
    PlanApproved,
    /// Every subtask is `done` (or `skipped`), nothing left to execute.
    ExecutionFinished,
    /// All `DoD` gates are green.
    GatesPassed,
    /// At least one gate is red.
    GatesFailed,
    /// A human decided the failed gates were addressed and execution should resume.
    Retry,
    /// No independent subtask remains.
    Block,
    /// A human or the scheduler unblocked the task.
    Unblock,
    /// A human merged the task branch.
    Merge,
    /// A human cancelled the task.
    Cancel,
}

closed_state!(TaskTrigger {
    StartPlanning => "start_planning",
    PlanWritten => "plan_written",
    PlanApproved => "plan_approved",
    ExecutionFinished => "execution_finished",
    GatesPassed => "gates_passed",
    GatesFailed => "gates_failed",
    Retry => "retry",
    Block => "block",
    Unblock => "unblock",
    Merge => "merge",
    Cancel => "cancel",
});

impl TaskState {
    /// Sunrise states: nothing is scheduled for them until something changes.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Merged | Self::Cancelled)
    }

    /// Whether the task occupies a worker right now.
    #[must_use]
    pub const fn is_working(self) -> bool {
        matches!(self, Self::Planning | Self::Running | Self::Verifying)
    }

    /// The one place the lifecycle's allowed edges are written down.
    ///
    /// Adding a variant to [`TaskState`] without adding edges here is caught by the
    /// `every_non_terminal_state_has_an_exit` test, which is why this is a `match` and not a
    /// table indexed by discriminant.
    pub const fn trigger(self, trigger: TaskTrigger) -> Result<Self, StatusError> {
        let next = match (self, trigger) {
            (Self::Queued, TaskTrigger::StartPlanning) => Self::Planning,
            (Self::Planning, TaskTrigger::PlanWritten) => Self::AwaitingPlanApproval,
            // Three ways back into `running`: the plan is approved (implicitly or after a
            // review), a failed gate round has been addressed, or a blocked task found an
            // independent subtask. One body, three causes — and the caller can always tell
            // which happened from the trigger it passed.
            (Self::Planning | Self::AwaitingPlanApproval, TaskTrigger::PlanApproved)
            | (Self::FailedGates, TaskTrigger::Retry)
            | (Self::Blocked, TaskTrigger::Unblock) => Self::Running,
            (Self::Running, TaskTrigger::ExecutionFinished) => Self::Verifying,
            (Self::Verifying, TaskTrigger::GatesPassed) => Self::ReadyForReview,
            (Self::Verifying, TaskTrigger::GatesFailed) => Self::FailedGates,
            (Self::Planning | Self::Running | Self::Verifying | Self::FailedGates, TaskTrigger::Block) => {
                Self::Blocked
            }
            (Self::ReadyForReview, TaskTrigger::Merge) => Self::Merged,
            // Cancellation is a human override: it must work from every state that is not
            // already finished, including a blocked task nobody is working on.
            (state, TaskTrigger::Cancel) if !state.is_terminal() => Self::Cancelled,
            (state, trigger) => {
                return Err(StatusError::Task { from: state, trigger });
            }
        };
        Ok(next)
    }
}

/// Why a task is paused without being stopped.
///
/// A hold is orthogonal to the lifecycle: see the module documentation of [`super`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Hold {
    /// The model backend is unreachable. Does not stop offline work.
    WaitingBackend,
    /// Free disk space fell below the hard threshold. Everything stops.
    PausedDisk,
}

closed_state!(Hold {
    WaitingBackend => "waiting_backend",
    PausedDisk => "paused_disk",
});

/// A user goal and its decomposition.
///
/// The subtask vector is flat at depth 1 and each subtask may carry children at depth 2;
/// the config's `max_subtask_depth` is enforced when a split is proposed, not here, because
/// this type must be able to *represent* an invalid tree left behind by a crash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Task {
    pub id: TaskId,
    pub title: String,
    /// The user's words, verbatim. The planner may restate this, but never replaces it.
    pub goal: String,
    pub state: TaskState,
    /// Empty unless the task is paused on something outside its control.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hold: Option<Hold>,
    pub base_branch: String,
    pub branch: String,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    #[serde(default)]
    pub subtasks: Vec<Subtask>,
}

impl Task {
    /// A queued task with no subtasks yet.
    #[must_use]
    pub fn new(
        id: TaskId,
        title: impl Into<String>,
        goal: impl Into<String>,
        base_branch: impl Into<String>,
        now: Timestamp,
    ) -> Self {
        let id_text = id.as_str().to_string();
        Self {
            title: title.into(),
            goal: goal.into(),
            state: TaskState::Queued,
            hold: None,
            // `agent/` is the meaningful branch; the checkpoint branch is separate and is
            // never the task's own branch.
            branch: format!("agent/{id_text}"),
            base_branch: base_branch.into(),
            created_at: now,
            updated_at: now,
            id,
            subtasks: Vec::new(),
        }
    }

    /// Apply a trigger, refusing it if the task is suspended.
    ///
    /// Cancellation is the exception: a human must be able to cancel a task whose backend
    /// died or whose disk filled, without first clearing the condition.
    pub fn transition(&mut self, trigger: TaskTrigger, now: Timestamp) -> Result<TaskState, StatusError> {
        if let Some(hold) = self.hold
            && trigger != TaskTrigger::Cancel
        {
            return Err(StatusError::Suspended { hold });
        }
        let next = self.state.trigger(trigger)?;
        self.state = next;
        self.updated_at = now;
        if next.is_terminal() {
            // A finished task cannot be paused; keeping a stale hold would make `status`
            // claim a merged task is waiting for a backend.
            self.hold = None;
        }
        Ok(next)
    }

    /// Pause the task without changing where it is.
    ///
    /// Returning an error for a second hold is deliberate: two simultaneous causes means the
    /// caller must decide which one to report, and silently replacing `PausedDisk` with
    /// `WaitingBackend` would let a full disk be forgotten.
    pub const fn suspend(&mut self, hold: Hold, now: Timestamp) -> Result<(), StatusError> {
        if self.state.is_terminal() {
            return Err(StatusError::Finished { state: self.state });
        }
        if let Some(existing) = self.hold {
            return Err(StatusError::AlreadySuspended { hold: existing });
        }
        self.hold = Some(hold);
        self.updated_at = now;
        Ok(())
    }

    /// Lift a hold, but only the one the caller believes is in force.
    pub fn clear_hold(&mut self, cleared: Hold, now: Timestamp) -> Result<(), StatusError> {
        let Some(existing) = self.hold else {
            return Err(StatusError::NotSuspended);
        };
        if existing != cleared {
            return Err(StatusError::WrongHold {
                hold: existing,
                cleared,
            });
        }
        self.hold = None;
        self.updated_at = now;
        Ok(())
    }

    /// Look a subtask up by id. Depth 2 is searched, no deeper.
    #[must_use]
    pub fn subtask(&self, id: &super::SubtaskId) -> Option<&Subtask> {
        self.subtasks
            .iter()
            .find(|s| &s.id == id)
            .or_else(|| self.subtasks.iter().find_map(|s| s.child(id)))
    }

    /// Counts for the progress line: `(closed, total)` over the whole tree.
    #[must_use]
    pub fn progress(&self) -> (usize, usize) {
        let closed = self.subtasks.iter().map(Subtask::closed_count).sum();
        let total = self.subtasks.iter().map(Subtask::total_count).sum();
        (closed, total)
    }

    /// Whether execution may be declared finished.
    ///
    /// Kept as a separate predicate rather than folded into the transition so that the runner
    /// can report *which* subtasks still hold the task open. A transition error would say only
    /// "not accepted", which is useless in `status` output.
    ///
    /// A task with no subtasks is *not* ready: an empty decomposition means the planner has
    /// not run, and treating it as finished would let `verifying` pass on nothing at all.
    #[must_use]
    pub fn ready_to_finish_execution(&self) -> bool {
        !self.subtasks.is_empty() && self.subtasks.iter().all(|s| s.state.counts_as_settled())
    }

    /// Subtasks that still hold execution open, deepest first, for reporting.
    #[must_use]
    pub fn open_subtasks(&self) -> Vec<&Subtask> {
        let mut open: Vec<&Subtask> = self
            .subtasks
            .iter()
            .flat_map(|s| std::iter::once(s).chain(s.children.iter()))
            .filter(|s| !s.state.counts_as_settled())
            .collect();
        // Deepest first: the innermost unfinished work is what a human needs to see.
        open.sort_by(|a, b| b.depth.cmp(&a.depth).then_with(|| a.id.cmp(&b.id)));
        open
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn now() -> Timestamp {
        chrono::TimeZone::timestamp_opt(&chrono::Utc, 1_700_000_000, 0)
            .single()
            .unwrap()
    }

    fn task() -> Task {
        Task::new(TaskId::generate(), "t", "g", "main", now())
    }

    #[test]
    fn plan_approval_is_skippable() {
        let mut t = task();
        t.transition(TaskTrigger::StartPlanning, now()).unwrap();
        assert_eq!(
            t.transition(TaskTrigger::PlanApproved, now()).unwrap(),
            TaskState::Running
        );
    }

    #[test]
    fn plan_review_is_an_optional_detour() {
        let mut t = task();
        t.transition(TaskTrigger::StartPlanning, now()).unwrap();
        t.transition(TaskTrigger::PlanWritten, now()).unwrap();
        assert_eq!(t.state, TaskState::AwaitingPlanApproval);
        assert_eq!(
            t.transition(TaskTrigger::PlanApproved, now()).unwrap(),
            TaskState::Running
        );
    }

    #[test]
    fn happy_path_reaches_merged() {
        let mut t = task();
        for trigger in [
            TaskTrigger::StartPlanning,
            TaskTrigger::PlanApproved,
            TaskTrigger::ExecutionFinished,
            TaskTrigger::GatesPassed,
            TaskTrigger::Merge,
        ] {
            t.transition(trigger, now()).unwrap();
        }
        assert_eq!(t.state, TaskState::Merged);
        assert!(t.state.is_terminal());
    }

    #[test]
    fn failed_gates_can_be_retried() {
        let mut t = task();
        for trigger in [
            TaskTrigger::StartPlanning,
            TaskTrigger::PlanApproved,
            TaskTrigger::ExecutionFinished,
            TaskTrigger::GatesFailed,
        ] {
            t.transition(trigger, now()).unwrap();
        }
        assert_eq!(t.state, TaskState::FailedGates);
        t.transition(TaskTrigger::Retry, now()).unwrap();
        assert_eq!(t.state, TaskState::Running);
    }

    #[test]
    fn blocked_task_does_not_skip_verification_on_unblock() {
        // The old lossy model had `waiting_backend -> running`; make sure `blocked` at least
        // keeps the caller honest by requiring a fresh execution round before gates.
        let mut t = task();
        t.transition(TaskTrigger::StartPlanning, now()).unwrap();
        t.transition(TaskTrigger::PlanApproved, now()).unwrap();
        t.transition(TaskTrigger::Block, now()).unwrap();
        assert_eq!(t.state, TaskState::Blocked);
        t.transition(TaskTrigger::Unblock, now()).unwrap();
        assert!(
            t.transition(TaskTrigger::GatesPassed, now()).is_err(),
            "blocked task must not reach ready_for_review without executing"
        );
        assert_eq!(t.state, TaskState::Running);
    }

    #[test]
    fn every_non_terminal_state_can_be_cancelled() {
        for state in TaskState::ALL.iter().copied().filter(|s| !s.is_terminal()) {
            assert_eq!(
                state.trigger(TaskTrigger::Cancel).unwrap(),
                TaskState::Cancelled,
                "{state}"
            );
        }
    }

    #[test]
    fn every_non_terminal_state_has_an_exit() {
        for state in TaskState::ALL.iter().copied().filter(|s| !s.is_terminal()) {
            assert!(
                TaskTrigger::ALL.iter().any(|t| state.trigger(*t).is_ok()),
                "`{state}` has no outgoing edge"
            );
        }
    }

    #[test]
    fn terminal_states_accept_nothing() {
        for state in [TaskState::Merged, TaskState::Cancelled] {
            for trigger in TaskTrigger::ALL {
                assert!(state.trigger(*trigger).is_err(), "`{state}` accepted `{trigger}`");
            }
        }
    }

    #[test]
    fn merged_is_only_reachable_from_ready_for_review() {
        for state in TaskState::ALL {
            let reachable = state.trigger(TaskTrigger::Merge).is_ok();
            assert_eq!(
                reachable,
                *state == TaskState::ReadyForReview,
                "`{state}` → merged: {reachable}"
            );
        }
    }

    #[test]
    fn hold_preserves_the_underlying_state() {
        let mut t = task();
        t.transition(TaskTrigger::StartPlanning, now()).unwrap();
        t.transition(TaskTrigger::PlanApproved, now()).unwrap();
        t.transition(TaskTrigger::ExecutionFinished, now()).unwrap();
        t.suspend(Hold::WaitingBackend, now()).unwrap();

        // Advance is refused wholesale, not silently applied.
        let err = t.transition(TaskTrigger::GatesPassed, now()).unwrap_err();
        assert_eq!(
            err,
            StatusError::Suspended {
                hold: Hold::WaitingBackend
            }
        );
        assert_eq!(t.state, TaskState::Verifying);

        t.clear_hold(Hold::WaitingBackend, now()).unwrap();
        t.transition(TaskTrigger::GatesPassed, now()).unwrap();
        assert_eq!(t.state, TaskState::ReadyForReview);
    }

    #[test]
    fn cancel_works_while_suspended() {
        let mut t = task();
        t.suspend(Hold::PausedDisk, now()).unwrap();
        t.transition(TaskTrigger::Cancel, now()).unwrap();
        assert_eq!(t.state, TaskState::Cancelled);
        assert_eq!(t.hold, None, "a finished task must not keep a hold");
    }

    #[test]
    fn double_suspend_is_refused() {
        let mut t = task();
        t.suspend(Hold::WaitingBackend, now()).unwrap();
        assert_eq!(
            t.suspend(Hold::PausedDisk, now()).unwrap_err(),
            StatusError::AlreadySuspended {
                hold: Hold::WaitingBackend
            }
        );
    }

    #[test]
    fn clearing_the_wrong_hold_is_refused() {
        let mut t = task();
        t.suspend(Hold::PausedDisk, now()).unwrap();
        assert_eq!(
            t.clear_hold(Hold::WaitingBackend, now()).unwrap_err(),
            StatusError::WrongHold {
                hold: Hold::PausedDisk,
                cleared: Hold::WaitingBackend
            }
        );
        assert_eq!(t.hold, Some(Hold::PausedDisk));
    }

    #[test]
    fn clearing_without_a_hold_is_refused() {
        let mut t = task();
        assert_eq!(
            t.clear_hold(Hold::PausedDisk, now()).unwrap_err(),
            StatusError::NotSuspended
        );
    }

    #[test]
    fn suspended_terminal_task_is_impossible() {
        let mut t = task();
        t.transition(TaskTrigger::Cancel, now()).unwrap();
        assert_eq!(
            t.suspend(Hold::PausedDisk, now()).unwrap_err(),
            StatusError::Finished {
                state: TaskState::Cancelled
            }
        );
    }

    #[test]
    fn state_names_roundtrip_through_serde() {
        for state in TaskState::ALL {
            let json = serde_json::to_string(state).unwrap();
            assert_eq!(json, format!("\"{}\"", state.as_str()));
            assert_eq!(&serde_json::from_str::<TaskState>(&json).unwrap(), state);
        }
    }

    #[test]
    fn an_empty_plan_is_not_finished() {
        let t = task();
        assert!(!t.ready_to_finish_execution(), "no subtasks cannot be done");
        assert_eq!(t.progress(), (0, 0));
    }

    #[test]
    fn settled_subtasks_finish_execution_including_blocked() {
        let mut t = task();
        let mut done = Subtask::new(1, "a", "g", crate::domain::Dod::reviewed("w"), now());
        done.state = crate::domain::SubtaskState::Done;
        let mut blocked = Subtask::new(1, "b", "g", crate::domain::Dod::reviewed("w"), now());
        blocked.state = crate::domain::SubtaskState::Blocked;
        t.subtasks = vec![done, blocked];
        assert!(t.ready_to_finish_execution());
        assert_eq!(t.open_subtasks().len(), 0);
    }

    #[test]
    fn open_subtasks_are_reported_deepest_first() {
        let mut t = task();
        let mut busy = Subtask::new(1, "parent", "g", crate::domain::Dod::reviewed("w"), now());
        let child = Subtask::new(2, "child", "g", crate::domain::Dod::reviewed("w"), now());
        busy.children.push(child);
        t.subtasks = vec![busy];
        assert!(!t.ready_to_finish_execution());
        let open = t.open_subtasks();
        assert_eq!(open.len(), 2);
        assert_eq!(open[0].depth, 2, "the deepest open subtask comes first");
    }

    #[test]
    fn schema_field_order_is_stable() {
        // The journal and `TASKS.json` are read by scripts and by humans in diffs. This test
        // does not forbid change; it makes an accidental change loud. Update it together with
        // the schema version when a field is really added or removed.
        // `to_value` goes through a map that sorts keys, so the check has to read the
        // serialized text, not a `Value`.
        let text = serde_json::to_string(&task()).unwrap();
        let expected = [
            "\"id\":",
            "\"title\":",
            "\"goal\":",
            "\"state\":",
            "\"base_branch\":",
            "\"branch\":",
            "\"created_at\":",
            "\"updated_at\":",
            "\"subtasks\":",
        ];
        let mut cursor = 0;
        for key in expected {
            let Some(found) = text[cursor..].find(key) else {
                panic!("`{key}` missing or out of order in {text}");
            };
            cursor += found + key.len();
        }
    }

    #[test]
    fn unknown_state_names_are_rejected() {
        let err = serde_json::from_str::<TaskState>("\"sideways\"").unwrap_err();
        assert!(err.to_string().contains("sideways"), "{err}");
    }

    #[test]
    fn task_branch_is_derived_from_the_id() {
        let t = task();
        assert_eq!(t.branch, format!("agent/{}", t.id));
    }

    proptest! {
        // The lifecycle must stay deadlock-free: from any state a human can always cancel,
        // and a random walk of legal triggers never lands outside the closed set.
        #[test]
        fn random_legal_walk_stays_typed(triggers in proptest::collection::vec(0usize..11, 0..40)) {
            let mut t = task();
            for index in triggers {
                let trigger = TaskTrigger::ALL[index];
                let _ = t.transition(trigger, now());
                prop_assert!(TaskState::ALL.contains(&t.state));
            }
        }

        #[test]
        fn cancel_always_terminates(start in 0usize..10, triggers in proptest::collection::vec(0usize..11, 0..20)) {
            let mut t = task();
            t.state = TaskState::ALL[start];
            for index in triggers {
                let _ = t.transition(TaskTrigger::ALL[index], now());
                if t.state.is_terminal() {
                    break;
                }
            }
            // Either the walk already finished the task (only `merge` reaches `Merged`, and
            // it is legal), or a human can always stop it. What must never happen is a
            // non-terminal state that refuses cancellation.
            if t.state.is_terminal() {
                prop_assert!(matches!(t.state, TaskState::Merged | TaskState::Cancelled));
                prop_assert!(t.transition(TaskTrigger::Cancel, now()).is_err());
            } else {
                t.transition(TaskTrigger::Cancel, now()).unwrap();
                prop_assert_eq!(t.state, TaskState::Cancelled);
            }
        }
    }
}
