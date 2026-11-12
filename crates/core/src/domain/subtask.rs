//! Subtasks: the unit of fresh context, and the only thing that actually executes.
//!
//! A task is a goal; a subtask is a piece of work with its own `DoD`, its own attempts and
//! — crucially — its own context. When a subtask closes, its conversation is discarded and
//! only a structured [`SubtaskReport`] survives. That is what keeps complexity in the task
//! tree instead of in the model's window, and it is the reason a subtask is the granularity
//! everything else is built around.
//!
//! # Depth
//!
//! `depth` is 1 for children of the task and 2 for their children. A subtask may split
//! itself, but the split must not change the parent's `DoD` — that would be a replan and needs
//! a human. The limit is enforced by the runner using `max_subtask_depth`; the type only
//! records the truth.

use serde::{Deserialize, Serialize};

use super::{Dod, StatusError, SubtaskId, Timestamp};

/// Where a subtask stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SubtaskState {
    /// Not started; waiting for the scheduler.
    Pending,
    /// An attempt is executing.
    Running,
    /// `DoD` satisfied and gates green.
    Done,
    /// Retries exhausted. The task moves on to the next independent subtask.
    Blocked,
    /// Deliberately not done: superseded by a split, or irrelevant after a replan.
    Skipped,
}

closed_state!(SubtaskState {
    Pending => "pending",
    Running => "running",
    Done => "done",
    Blocked => "blocked",
    Skipped => "skipped",
});

/// Why a subtask should move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SubtaskTrigger {
    /// The scheduler picked it and an attempt is starting.
    Start,
    /// Every gate passed.
    Complete,
    /// `max_attempts_per_subtask` and the restart budget are exhausted.
    GiveUp,
    /// A human or the planner decided it is no longer needed.
    Skip,
    /// A human unblocked it for another round.
    Reopen,
    /// The subtask was split into children; the parent is replaced by them.
    Split,
}

closed_state!(SubtaskTrigger {
    Start => "start",
    Complete => "complete",
    GiveUp => "give_up",
    Skip => "skip",
    Reopen => "reopen",
    Split => "split",
});

impl SubtaskState {
    /// Whether this subtask no longer needs a worker.
    #[must_use]
    pub const fn is_closed(self) -> bool {
        matches!(self, Self::Done | Self::Blocked | Self::Skipped)
    }

    /// Whether a parent may be considered finished when this is its state.
    ///
    /// `Blocked` counts: the design says a blocked subtask does not stop the task, and the
    /// task's own `blocked` state is entered only when *no* independent subtask remains.
    #[must_use]
    pub const fn counts_as_settled(self) -> bool {
        self.is_closed()
    }

    pub const fn trigger(self, trigger: SubtaskTrigger) -> Result<Self, StatusError> {
        let next = match (self, trigger) {
            // Starting and reopening are the same edge seen from different sides: one begins
            // the first pass, the other begins another. A reviewer may reject a "done"
            // subtask, and that must not require destroying and recreating it.
            (Self::Pending, SubtaskTrigger::Start) | (Self::Blocked | Self::Done, SubtaskTrigger::Reopen) => {
                Self::Running
            }
            (Self::Running, SubtaskTrigger::Complete) => Self::Done,
            (Self::Running, SubtaskTrigger::GiveUp) => Self::Blocked,
            // Skipping and splitting both retire this subtask: a split's parent is replaced by
            // its children, and a split is the standard L4 response to an exhausted subtask.
            // Any state that is not already closed may be split; a pending or running one may
            // be skipped out of the plan entirely.
            (Self::Pending | Self::Running | Self::Blocked, SubtaskTrigger::Skip | SubtaskTrigger::Split) => {
                Self::Skipped
            }
            (state, trigger) => {
                return Err(StatusError::Subtask { from: state, trigger });
            }
        };
        Ok(next)
    }
}

/// The structured result of a subtask, written to state when it closes.
///
/// This is the *only* thing the next subtask inherits. Everything else about the conversation
/// is dropped, so the fields are chosen for what a fresh context actually needs: what was
/// achieved, what is now known, what was tried and rejected, and what changed on disk.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubtaskReport {
    /// What the subtask achieved, in past tense.
    pub outcome: String,
    /// Facts the rest of the task should know, one per line.
    #[serde(default)]
    pub facts: Vec<String>,
    /// Paths changed by this subtask, relative to the repository root.
    #[serde(default)]
    pub files_touched: Vec<std::path::PathBuf>,
    /// Approaches that did not work, so that a later attempt does not repeat them.
    ///
    /// Excluding these from compaction is deliberate and is the single most useful part of
    /// the report: a model that re-proposes a rejected idea is the commonest way a run
    /// burns a day.
    #[serde(default)]
    pub rejected_ideas: Vec<String>,
}

impl SubtaskReport {
    /// A report that says only what happened.
    #[must_use]
    pub fn new(outcome: impl Into<String>) -> Self {
        Self {
            outcome: outcome.into(),
            ..Self::default()
        }
    }
}

/// One unit of execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Subtask {
    pub id: SubtaskId,
    /// 1 for a child of the task, 2 for a child of a child.
    pub depth: u8,
    pub title: String,
    /// What to achieve, written by the planner for a fresh context.
    pub goal: String,
    pub dod: Dod,
    pub state: SubtaskState,
    /// Ids of subtasks that must be settled before this one may start.
    #[serde(default)]
    pub depends_on: Vec<SubtaskId>,
    /// If set, the attempt that last held this subtask.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_attempt: Option<super::AttemptId>,
    /// Settled attempts, oldest first.
    #[serde(default)]
    pub attempts: Vec<super::Attempt>,
    /// Present once the subtask is done or blocked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<SubtaskReport>,
    #[serde(default)]
    pub children: Vec<Self>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

impl Subtask {
    /// A pending subtask at `depth`.
    #[must_use]
    pub fn new(
        depth: u8,
        title: impl Into<String>,
        goal: impl Into<String>,
        dod: Dod,
        now: Timestamp,
    ) -> Self {
        Self {
            id: SubtaskId::generate(),
            depth,
            title: title.into(),
            goal: goal.into(),
            dod,
            state: SubtaskState::Pending,
            depends_on: Vec::new(),
            current_attempt: None,
            attempts: Vec::new(),
            report: None,
            children: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Apply a trigger, keeping timestamps honest.
    pub fn transition(
        &mut self,
        trigger: SubtaskTrigger,
        now: Timestamp,
    ) -> Result<SubtaskState, StatusError> {
        let next = self.state.trigger(trigger)?;
        self.state = next;
        self.updated_at = now;
        if next.is_closed() {
            self.current_attempt = None;
        }
        Ok(next)
    }

    /// Whether every dependency is settled, given a lookup over the sibling list.
    ///
    /// Dependencies that do not resolve are treated as unsatisfied: a dangling id is a
    /// planning bug, and starting the subtask anyway would hide it.
    #[must_use]
    pub fn is_ready(&self, lookup: impl Fn(&SubtaskId) -> Option<SubtaskState>) -> bool {
        self.state == SubtaskState::Pending
            && self
                .depends_on
                .iter()
                .all(|id| lookup(id).is_some_and(SubtaskState::counts_as_settled))
    }

    /// Find a direct child.
    #[must_use]
    pub fn child(&self, id: &SubtaskId) -> Option<&Self> {
        self.children.iter().find(|c| &c.id == id)
    }

    /// Number of subtasks in this subtree that count as settled.
    #[must_use]
    pub fn closed_count(&self) -> usize {
        usize::from(self.state.counts_as_settled())
            + self.children.iter().map(Self::closed_count).sum::<usize>()
    }

    /// Number of subtasks in this subtree.
    #[must_use]
    pub fn total_count(&self) -> usize {
        1 + self.children.iter().map(Self::total_count).sum::<usize>()
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

    fn subtask() -> Subtask {
        Subtask::new(1, "t", "g", Dod::reviewed("works"), now())
    }

    #[test]
    fn happy_path_pending_to_done() {
        let mut s = subtask();
        s.transition(SubtaskTrigger::Start, now()).unwrap();
        s.transition(SubtaskTrigger::Complete, now()).unwrap();
        assert_eq!(s.state, SubtaskState::Done);
    }

    #[test]
    fn running_to_blocked_is_the_give_up_edge() {
        let mut s = subtask();
        s.transition(SubtaskTrigger::Start, now()).unwrap();
        s.transition(SubtaskTrigger::GiveUp, now()).unwrap();
        assert_eq!(s.state, SubtaskState::Blocked);
        assert!(s.state.counts_as_settled());
    }

    #[test]
    fn a_blocked_subtask_may_be_split() {
        // L4 escalation: an exhausted subtask is split rather than failed outright.
        let mut s = subtask();
        s.transition(SubtaskTrigger::Start, now()).unwrap();
        s.transition(SubtaskTrigger::GiveUp, now()).unwrap();
        s.transition(SubtaskTrigger::Split, now()).unwrap();
        assert_eq!(s.state, SubtaskState::Skipped);
    }

    #[test]
    fn a_done_subtask_cannot_be_given_up_on() {
        let mut s = subtask();
        s.transition(SubtaskTrigger::Start, now()).unwrap();
        s.transition(SubtaskTrigger::Complete, now()).unwrap();
        assert!(s.transition(SubtaskTrigger::GiveUp, now()).is_err());
        assert_eq!(s.state, SubtaskState::Done);
    }

    #[test]
    fn a_done_subtask_can_be_reopened_by_a_reviewer() {
        let mut s = subtask();
        s.transition(SubtaskTrigger::Start, now()).unwrap();
        s.transition(SubtaskTrigger::Complete, now()).unwrap();
        s.transition(SubtaskTrigger::Reopen, now()).unwrap();
        assert_eq!(s.state, SubtaskState::Running);
    }

    #[test]
    fn closing_clears_the_current_attempt() {
        let mut s = subtask();
        s.current_attempt = Some(super::super::AttemptId::generate());
        s.transition(SubtaskTrigger::Start, now()).unwrap();
        s.transition(SubtaskTrigger::Complete, now()).unwrap();
        assert_eq!(s.current_attempt, None);
    }

    #[test]
    fn readiness_requires_settled_dependencies() {
        let dependency = SubtaskId::generate();
        let mut c = subtask();
        c.depends_on = vec![dependency.clone()];

        // A dangling id is a planning bug: it must keep the subtask unready, not start it.
        assert!(!c.is_ready(|_| None));

        let lookup_pending = |id: &SubtaskId| (id == &dependency).then_some(SubtaskState::Pending);
        assert!(!c.is_ready(lookup_pending));

        let lookup_done = |id: &SubtaskId| (id == &dependency).then_some(SubtaskState::Done);
        assert!(c.is_ready(lookup_done));
    }

    #[test]
    fn readiness_ignores_non_pending_subtasks() {
        let mut s = subtask();
        s.transition(SubtaskTrigger::Start, now()).unwrap();
        assert!(!s.is_ready(|_| None), "a running subtask is not ready");
    }

    #[test]
    fn counts_include_the_whole_subtree() {
        let mut parent = subtask();
        let mut child = Subtask::new(2, "c", "g", Dod::reviewed("w"), now());
        child.state = SubtaskState::Done;
        parent.children.push(child);
        assert_eq!(parent.total_count(), 2);
        assert_eq!(parent.closed_count(), 1);
        parent.state = SubtaskState::Done;
        assert_eq!(parent.closed_count(), 2);
    }

    #[test]
    fn report_roundtrips_and_defaults_are_empty() {
        let report = SubtaskReport::new("added the endpoint");
        let json = serde_json::to_string(&report).unwrap();
        let back: SubtaskReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report);
        assert_eq!(back.facts, Vec::<String>::new());
    }
}
