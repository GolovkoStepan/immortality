//! Who gets the next turn, and why.
//!
//! # What a "turn" is
//!
//! There are two different resources in this daemon and they are not the same thing:
//!
//! * the **inference slot** — one request at a time by default (D-008), because parallel requests
//!   lower per-request throughput on this backend;
//! * **everything else** — shell, tests, reads, commits, journal writes — which can overlap with
//!   another task's inference.
//!
//! So there are two questions, not one, and collapsing them would be the bug this module exists to
//! prevent: a task waiting for a backend must still be able to commit, and a task that can only
//! commit must not occupy the slot a task ready to think is waiting for.
//!
//! # Why fairness is a rotation and not a priority number
//!
//! A single task can run for hours, and a strict priority order lets it starve every task behind
//! it. Serving the task that was served last at the *back* of the order costs one field and makes
//! the queue behave the way a human expects: two ready tasks alternate turns. Priority still wins
//! over rotation — a task a human just started goes first — but between equals, nobody is passed
//! over twice.
//!
//! # Why `Priority` exists before anything sets it
//!
//! [`Priority::High`] is for a task the human explicitly asked to start. The producer is `start`
//! (M5.5/M5.8), and inventing it here rather than later is deliberate: the ordering rule has to be
//! written and tested against *something*, and adding the field when its first writer appears is
//! how the rule ends up with a branch nobody ever exercised.

use crate::domain::{Hold, TaskId, TaskState, Timestamp};

/// How urgently a task wants the slot.
///
/// Ordered so that comparison is the rule: `High > Normal`, and nothing else has to be said.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Queued work: it will be picked up, but nothing is waiting on it right now.
    Normal,
    /// A task the human asked to run, or one that is already mid-attempt.
    High,
}

impl Priority {
    /// The wire name, for `status` and the journal.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Normal => "normal",
        }
    }
}

/// Why a candidate was not chosen.
///
/// A refusal with a reason rather than a silently shorter list: "nothing is running" is the failure
/// mode this daemon exists to avoid, and a human staring at an idle daemon needs to know which of
/// the reasons applies. Rendered by `status` and `doctor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ineligible {
    /// `merged` or `cancelled`: the work ended.
    Finished,
    /// Nothing can be started: every subtask is settled, or one is already running and the rest
    /// wait on it. The task is waiting for its gates, for a dependant, or for a human.
    NoWork,
    /// A subtask is already `running`: the task is mid-attempt, and a second attempt would fork
    /// its journal. Distinct from [`Ineligible::NoWork`] because the two read as opposite facts
    /// to a person: one is "this is working", the other is "this is stuck".
    InFlight,
    /// `Hold::PausedDisk`: the disk guard fired, and everything about this task stops.
    Disk,
    /// `Hold::WaitingBackend`, in answer to the *inference* question. The task is still eligible
    /// for offline work — that is the whole point of the hold being orthogonal to the lifecycle.
    Backend,
    /// The task's own gates failed: the tree is finished, the verdict is red, and only a human can
    /// move it — by reopening a subtask, changing the plan, or `start --task` to retry.
    ///
    /// Distinct from [`Ineligible::NoWork`] because the two send a person to different places. "No
    /// subtask left to run" over a task that ended `failed_gates` is the report measured on
    /// `black_jack`: five subtasks done, two blocked, the task-level `rspec_pass` timed out, and
    /// `status` said the task had nothing left to do. It had nothing left to *run*; it was waiting
    /// for a human, which is the fact that was missing.
    FailedGates,
    /// A subtask is `blocked` and nothing independent is left: L5's escalation, parked until a
    /// human reopens the subtask or adds one to the plan.
    Blocked,
    /// Every subtask is settled and the gates were green: the task waits for a human to merge.
    ReadyForReview,
}

/// One task, as the queue sees it.
///
/// A plain value rather than a borrow of the plan: the queue must not be able to change the tree,
/// and assembling this costs one plan load per task per turn, which is what `status` does anyway.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub task: TaskId,
    pub priority: Priority,
    pub state: TaskState,
    pub hold: Option<Hold>,
    /// When the task was created, for the FIFO part of the order.
    pub created_at: Timestamp,
    /// Whether some subtask may be started right now.
    ///
    /// This is "ready to think", not "any subtask is open": a `false` here is the difference
    /// between an attempt worth starting and one that would begin with nothing to do, and
    /// treating the second as the first is how a daemon spins.
    pub has_work: bool,
    /// Whether subtasks are open but none may be started.
    ///
    /// Two different situations produce `has_work == false`, and a human reads them differently:
    /// every subtask is settled (waiting for gates or for a person), or one is already `running`
    /// and its dependents are waiting on it. Reporting both as "no subtask left to run" told a
    /// person watching a live run that the task had nothing to do — measured on
    /// `tsk_01M7HJEFXD7YDHGM59Y4X57FTZ`, where `status` said "1 with no subtask left to run"
    /// while the run was in the middle of its first attempt.
    pub in_flight: bool,
}

impl Candidate {
    /// Why this candidate cannot take the inference slot, if it cannot.
    #[must_use]
    pub const fn ineligible_for_inference(&self) -> Option<Ineligible> {
        if self.state.is_terminal() {
            return Some(Ineligible::Finished);
        }
        // The task's own verdict before the system's condition: a `failed_gates` task does not
        // start when the disk frees up or the backend returns, so reporting the outage would name
        // a fix that does not fix it. The system's condition is reported by the other tasks on the
        // line, and it is not this task's answer to "what do I do about it".
        if let Some(parked) = self.parked_reason() {
            return Some(parked);
        }
        match self.hold {
            Some(Hold::PausedDisk) => return Some(Ineligible::Disk),
            Some(Hold::WaitingBackend) => return Some(Ineligible::Backend),
            None => {}
        }
        if !self.has_work {
            // Order matters: "already running" is the stronger, more specific fact, and a person
            // who sees "no subtask left to run" during a live run is looking at a bug.
            return Some(if self.in_flight {
                Ineligible::InFlight
            } else {
                Ineligible::NoWork
            });
        }
        None
    }

    /// Why this candidate cannot do offline work, if it cannot.
    ///
    /// Only the disk guard and the end of the task stop offline work: a backend outage is a reason
    /// not to call the model, not a reason to leave the tree dirty and the journal unflushed.
    #[must_use]
    pub const fn ineligible_for_offline(&self) -> Option<Ineligible> {
        if self.state.is_terminal() {
            return Some(Ineligible::Finished);
        }
        if let Some(parked) = self.parked_reason() {
            return Some(parked);
        }
        if matches!(self.hold, Some(Hold::PausedDisk)) {
            return Some(Ineligible::Disk);
        }
        if !self.has_work {
            // Offline work is not attempted mid-attempt either: a subtask that is `running` has a
            // worker, and a second pass over the tree would be a second writer of one plan.
            return Some(if self.in_flight {
                Ineligible::InFlight
            } else {
                Ineligible::NoWork
            });
        }
        None
    }

    /// The reason a task that has stopped on its own account is not started again.
    ///
    /// Three lifecycle states leave nothing for a worker to do and everything for a human: the
    /// gates are red, a subtask is blocked and nothing independent is left, or the verdict is green
    /// and the branch awaits a merge. All three produced "no subtask left to run" — literally true
    /// and useless, because it names the symptom a person can already see and not the thing to do.
    /// Measured on the `black_jack` task that ended `failed_gates`: `status` said "1 with no
    /// subtask left to run" while the task was waiting for exactly one action, a human reading a
    /// timeout and reopening a subtask.
    ///
    /// `failed_gates` and `ready_for_review` are parked whether or not the tree has work. Both are
    /// verdicts about a run that finished, and a verdict is not something a fresh attempt may
    /// quietly overturn: resuming is `start --task`, which takes the `retry` edge on purpose. A
    /// `blocked` task is different — L5 blocks it only when nothing independent is left, and a
    /// human who adds a subtask makes it runnable again without touching its state, so that one is
    /// parked only when the tree really has nothing to offer.
    const fn parked_reason(&self) -> Option<Ineligible> {
        match self.state {
            TaskState::FailedGates => Some(Ineligible::FailedGates),
            TaskState::ReadyForReview => Some(Ineligible::ReadyForReview),
            TaskState::Blocked if !self.has_work => Some(Ineligible::Blocked),
            _ => None,
        }
    }
}

/// The queue's memory between turns.
///
/// It holds exactly one fact — who was served last — because that is all fairness needs. Anything
/// more would be a second copy of state the journal already has.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Queue {
    last_served: Option<TaskId>,
}

impl Queue {
    /// An empty queue.
    #[must_use]
    pub const fn new() -> Self {
        Self { last_served: None }
    }

    /// Record that `task` just took the inference slot.
    pub fn note_served(&mut self, task: TaskId) {
        self.last_served = Some(task);
    }

    /// Who was served last, for a caller that wants to render it.
    #[must_use]
    pub const fn last_served(&self) -> Option<&TaskId> {
        self.last_served.as_ref()
    }

    /// The task that should get the inference slot next.
    ///
    /// # Why this returns a `Vec` of rejections and not just `None`
    ///
    /// An empty answer is only useful if the caller can explain it. Returning the reasons along
    /// with the choice means `status` can say "three tasks, all waiting for the backend" without
    /// re-deriving the same rules a second time.
    #[must_use]
    pub fn next_for_inference<'a>(
        &self,
        candidates: &'a [Candidate],
    ) -> (Option<&'a Candidate>, Vec<(&'a TaskId, Ineligible)>) {
        self.pick(candidates, Candidate::ineligible_for_inference)
    }

    /// The task that should do offline work next (shell, tests, commits).
    #[must_use]
    pub fn next_for_offline<'a>(
        &self,
        candidates: &'a [Candidate],
    ) -> (Option<&'a Candidate>, Vec<(&'a TaskId, Ineligible)>) {
        self.pick(candidates, Candidate::ineligible_for_offline)
    }

    fn pick<'a>(
        &self,
        candidates: &'a [Candidate],
        ineligible: impl Fn(&Candidate) -> Option<Ineligible>,
    ) -> (Option<&'a Candidate>, Vec<(&'a TaskId, Ineligible)>) {
        let mut eligible: Vec<&Candidate> = Vec::new();
        let mut rejected: Vec<(&TaskId, Ineligible)> = Vec::new();
        for candidate in candidates {
            match ineligible(candidate) {
                Some(reason) => rejected.push((&candidate.task, reason)),
                None => eligible.push(candidate),
            }
        }
        // The order: priority first, then age, then id as a total tiebreak — an untotal order
        // makes the choice depend on the input's order, which is how `status` and the scheduler
        // start disagreeing about what runs next.
        eligible.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.created_at.cmp(&right.created_at))
                .then_with(|| left.task.as_str().cmp(right.task.as_str()))
        });
        // The task served last goes to the back of the group it belongs to, so equal tasks
        // alternate. Confined to the top priority group on purpose: rotating the whole vector
        // would let a `Normal` task displace the `High` one that was just served, which is the
        // one case a human explicitly asked to be first (a test pins this).
        if eligible.len() > 1
            && let Some(last) = &self.last_served
            && eligible[0].task == *last
            && let Some(next) = eligible
                .iter()
                .position(|candidate| candidate.priority == eligible[0].priority && candidate.task != *last)
        {
            eligible.swap(0, next);
        }
        (eligible.first().copied(), rejected)
    }
}

/// Why nothing can run at all, summarised for a human.
///
/// A `Vec<(&TaskId, Ineligible)>` answers "which task" but not "what do I do", and the second
/// question is the one a person with an idle daemon is asking.
#[must_use]
pub fn summarise(rejected: &[(&TaskId, Ineligible)]) -> String {
    if rejected.is_empty() {
        return "no tasks are registered".to_string();
    }
    let mut waiting_backend = 0_usize;
    let mut paused_disk = 0_usize;
    let mut finished = 0_usize;
    let mut waiting_gates = 0_usize;
    let mut in_flight = 0_usize;
    let mut failed_gates = 0_usize;
    let mut blocked = 0_usize;
    let mut ready_for_review = 0_usize;
    for (_, reason) in rejected {
        match reason {
            Ineligible::Backend => waiting_backend += 1,
            Ineligible::Disk => paused_disk += 1,
            Ineligible::Finished => finished += 1,
            Ineligible::NoWork => waiting_gates += 1,
            Ineligible::InFlight => in_flight += 1,
            Ineligible::FailedGates => failed_gates += 1,
            Ineligible::Blocked => blocked += 1,
            Ineligible::ReadyForReview => ready_for_review += 1,
        }
    }
    let mut parts: Vec<String> = Vec::new();
    if waiting_backend > 0 {
        parts.push(format!("{waiting_backend} waiting for the model backend"));
    }
    if paused_disk > 0 {
        parts.push(format!("{paused_disk} paused on free disk space"));
    }
    if in_flight > 0 {
        parts.push(format!("{in_flight} running now"));
    }
    // Parked tasks come before the generic "no subtask" line: a count of them names an action, and
    // a task that ended red is not waiting for a gate to finish. Ordering them first also keeps the
    // sentence a human reads about a finished run at the front of the line.
    if failed_gates > 0 {
        parts.push(format!("{failed_gates} waiting for a human after failed gates"));
    }
    if blocked > 0 {
        parts.push(format!("{blocked} blocked, waiting for a human"));
    }
    if ready_for_review > 0 {
        parts.push(format!("{ready_for_review} ready for review"));
    }
    if waiting_gates > 0 {
        parts.push(format!("{waiting_gates} with no subtask left to run"));
    }
    if finished > 0 {
        parts.push(format!("{finished} finished"));
    }
    parts.join("; ")
}

#[cfg(test)]
mod tests;
