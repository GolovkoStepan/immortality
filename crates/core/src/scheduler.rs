//! Choosing what runs next, and deciding whether a proposed split is legal.
//!
//! # Why this is separate from the runner
//!
//! The runner executes one subtask and knows nothing about its siblings. The scheduler answers
//! the question *before* that: given a tree, which subtask may run, and when a subtask comes
//! back exhausted, is splitting it a legal decomposition or a quiet weakening of the parent's
//! definition of done? Both are pure functions over the tree, which is why they are testable
//! without a model, a journal or a clock.
//!
//! # Why a split is validated rather than trusted
//!
//! A split is the L4 response to a subtask that keeps failing, and it is proposed by a model
//! that has already failed at the subtask three times. The parent is then retired (`Skipped`)
//! and its children carry the guarantee, so the check that matters is: **does the union of the
//! children's gates still imply the parent's?** Two ways to break that, both cheap to write and
//! both silent:
//!
//! - a child drops one of the parent's required checks, so nothing ever runs that command;
//! - the parent needed a reviewer and the children do not, so a diff nobody read passes.
//!
//! Both are refused here, at the moment the split is proposed, when the model can still be told
//! why and the attempt is not yet spent. At run time the symptom would be a green task whose
//! guarantee was never checked.
//!
//! # Why `Blocked` does not stop the task
//!
//! [`SubtaskState::counts_as_settled`] counts `Blocked`, so a subtask waiting on a blocked
//! dependency becomes ready rather than waiting forever. That is the design's rule — one stuck
//! subtask must not idle the task for a day — and it is why [`next`] reports `Stuck` only when
//! something is genuinely unresolved (an unmet dependency that is still pending) rather than
//! whenever any subtask failed.

use crate::domain::{Subtask, SubtaskId, SubtaskState, SubtaskTrigger, Timestamp};
use crate::journal::{Journal, Kind, Scope};
use crate::plan::{Plan, PlanError};

/// One piece of a proposed split, as the model describes it.
///
/// Only the words: the `DoD` is inherited from the parent, because a model that could write its
/// children's gates would be able to weaken the very guarantee the split is supposed to preserve.
/// The parent's `DoD` is the contract; the pieces say where the work goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Piece {
    pub title: String,
    pub goal: String,
}

impl Piece {
    #[must_use]
    pub fn new(title: impl Into<String>, goal: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            goal: goal.into(),
        }
    }
}

/// Why a proposed split cannot be applied.
///
/// Not `Clone`/`PartialEq`: [`PlanError`] wraps I/O and JSON errors, so neither is. Tests match
/// on the variant, which is the part that has meaning.
#[derive(Debug, thiserror::Error)]
pub enum SplitError {
    #[error("subtask `{id}` is `{state}` and can no longer be split")]
    Terminal { id: String, state: SubtaskState },

    #[error("a split must produce at least one child")]
    Empty,

    #[error(
        "a split of `{id}` into a single piece is not a decomposition; it would retire the \
         subtask and run the same work under a new id"
    )]
    NotADecomposition { id: String },

    #[error(
        "splitting `{id}` (depth {depth}) would need depth {needed}, beyond the configured \
         maximum of {max}"
    )]
    TooDeep {
        id: String,
        depth: u8,
        needed: u8,
        max: u8,
    },

    #[error("child `{id}` declares depth {declared} but a child of this parent is at depth {expected}")]
    WrongDepth { id: String, declared: u8, expected: u8 },

    #[error("child `{id}` is `{state}`; a split creates fresh pending work")]
    NotPending { id: String, state: SubtaskState },

    #[error("child `{id}` has an empty title or goal")]
    EmptyText { id: String },

    #[error("child `{id}` has an unusable definition of done: {reason}")]
    BadDod { id: String, reason: String },

    #[error("child `{id}` depends on the parent `{parent}` it is replacing; children run first")]
    ChildDependsOnParent { id: String, parent: String },

    #[error(
        "the split of `{id}` drops the required check `{check}`; no child would ever run it, so \
         the parent's definition of done would no longer be enforced"
    )]
    MissingCheck { id: String, check: String },

    #[error(
        "`{id}` requires a reviewer and child `{child}` does not; an unreviewed diff would carry \
         part of a reviewed guarantee"
    )]
    ReviewDropped { id: String, child: String },

    #[error(transparent)]
    Plan(#[from] PlanError),

    #[error("the split could not be journalled: {0}")]
    Journal(#[from] crate::Error),
}

/// Check that `children` are a legal decomposition of `parent`.
///
/// # Errors
/// Any [`SplitError`]: a terminal parent, no children, a depth beyond `max_depth`, a child at
/// the wrong depth or not pending, a weakened `DoD`, or a child that depends on its parent.
pub fn validate_split(parent: &Subtask, children: &[Subtask], max_depth: u8) -> Result<(), SplitError> {
    // The state machine is the authority on which states may split; asking it here keeps the
    // two from drifting.
    if parent.state.trigger(SubtaskTrigger::Split).is_err() {
        return Err(SplitError::Terminal {
            id: parent.id.to_string(),
            state: parent.state,
        });
    }
    if children.is_empty() {
        return Err(SplitError::Empty);
    }
    let expected = parent.depth.saturating_add(1);
    if expected > max_depth {
        return Err(SplitError::TooDeep {
            id: parent.id.to_string(),
            depth: parent.depth,
            needed: expected,
            max: max_depth,
        });
    }
    if children.len() == 1 {
        // A single child is the parent's work under a new id, and it arrives with a fresh
        // attempts budget — a weak model could dodge `max_attempts_per_subtask` forever by
        // "splitting" into one piece. Decomposition means more than one piece.
        return Err(SplitError::NotADecomposition {
            id: parent.id.to_string(),
        });
    }
    for child in children {
        if child.depth != expected {
            return Err(SplitError::WrongDepth {
                id: child.id.to_string(),
                declared: child.depth,
                expected,
            });
        }
        if child.state != SubtaskState::Pending {
            return Err(SplitError::NotPending {
                id: child.id.to_string(),
                state: child.state,
            });
        }
        if child.title.trim().is_empty() || child.goal.trim().is_empty() {
            return Err(SplitError::EmptyText {
                id: child.id.to_string(),
            });
        }
        child.dod.validate().map_err(|error| SplitError::BadDod {
            id: child.id.to_string(),
            reason: error.to_string(),
        })?;
        if child.depends_on.contains(&parent.id) {
            return Err(SplitError::ChildDependsOnParent {
                id: child.id.to_string(),
                parent: parent.id.to_string(),
            });
        }
    }
    // Every required check of the parent must still run somewhere. Optional checks are
    // informational, so dropping one costs nothing.
    for check in parent.dod.required_checks() {
        let covered = children.iter().any(|child| {
            child
                .dod
                .checks
                .iter()
                .any(|candidate| candidate.id == check.id && candidate.command == check.command)
        });
        if !covered {
            return Err(SplitError::MissingCheck {
                id: parent.id.to_string(),
                check: check.id.clone(),
            });
        }
    }
    // Review is judged per subtask and the parent will not run, so each child that contributes
    // work has to be reviewed for the parent's guarantee to survive.
    if parent.dod.review
        && let Some(child) = children.iter().find(|child| !child.dod.review)
    {
        return Err(SplitError::ReviewDropped {
            id: parent.id.to_string(),
            child: child.id.to_string(),
        });
    }
    Ok(())
}

/// A child that inherits its parent's definition of done unchanged.
///
/// The common case: a subtask is too large, not too weakly specified, so the pieces carry the
/// same gate. Callers that need a narrower child's `DoD` can build it and pass it to
/// [`validate_split`], which will accept anything at least as strong as the parent's.
#[must_use]
pub fn inherit(
    parent: &Subtask,
    title: impl Into<String>,
    goal: impl Into<String>,
    at: Timestamp,
) -> Subtask {
    Subtask::new(
        parent.depth.saturating_add(1),
        title,
        goal,
        parent.dod.clone(),
        at,
    )
}

/// Build the children of a split from the model's pieces.
///
/// The `DoD` is inherited rather than accepted from the model: a subtask proposed after three
/// failed attempts is exactly the one that would like a cheaper gate, and the parent's gate is
/// the thing the split exists to preserve.
#[must_use]
pub fn children_of(parent: &Subtask, pieces: &[Piece], at: Timestamp) -> Vec<Subtask> {
    pieces
        .iter()
        .map(|piece| inherit(parent, piece.title.clone(), piece.goal.clone(), at))
        .collect()
}

/// What to do with a subtask that has exhausted its attempts.
///
/// This is the L4/L5 end of the ladder, and it is a decision rather than a mechanism: the
/// mechanism is [`validate_split`] and [`Plan::split`], the decision is which rung applies.
/// Keeping it separate means the policy is testable against a tree without inventing a model
/// that fails three times.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exhausted {
    /// L4: the subtask is too large, not impossible. Split it and let the pieces run.
    Split,
    /// L5: the subtask cannot be split further. Block it and move to the next independent one.
    Block,
    /// L5: there is nothing else to run. Ask a human rather than idling the task.
    Escalate,
}

/// Decide the rung for an exhausted subtask.
///
/// `has_independent` is whether any other subtask could run if this one were blocked.
///
/// The order is the design's: L4 before L5. Splitting keeps the work in the task and is possible
/// whenever the tree has room, so it is the first answer; blocking is the fallback, and
/// escalation only when blocking would leave the task with nothing to do — the design's
/// `blocked` "с причиной".
///
/// A subtask at `max_depth` is never offered L4: [`validate_split`] would refuse its children,
/// and returning `Split` would hand the caller a decision the mechanism has already vetoed.
///
/// Note what this function does *not* say: how to get the pieces. That needs a model call, so it
/// is the caller's decision whether to spend an attempt asking, and the caller knows the attempt
/// budget this function cannot see.
#[must_use]
pub const fn on_exhausted(subtask: &Subtask, max_depth: u8, has_independent: bool) -> Exhausted {
    if subtask.depth.saturating_add(1) <= max_depth {
        Exhausted::Split
    } else if has_independent {
        Exhausted::Block
    } else {
        Exhausted::Escalate
    }
}

/// Apply a split to the tree and journal it.
///
/// Returns the ids of the children, in the order the model proposed them.
///
/// # Ordering: the tree first, then the journal
///
/// `TASKS.json` is durable state, not a cache — the plan *is* the tree, and the journal is the
/// audit trail of what happened to it. So the write order is chosen by what a crash between the
/// two leaves behind:
///
/// - file first, journal second (this order): a crash leaves a correct tree with the parent
///   retired and its children pending. The work continues; only the audit line is missing, and a
///   reader sees a `Skipped` parent with children, which is exactly what a split looks like.
/// - journal first, file second: a crash leaves a journal that claims a decomposition the tree
///   never received. The next tick would find the parent exhausted again and split it a second
///   time — two different sets of children for one parent.
///
/// The general rule "the journal wins" is about `attempts.json` and `subtasks.json`, which are
/// derived. It does not apply to the plan, which is the thing being recorded.
///
/// # Errors
/// Any [`SplitError`] from [`validate_split`], or the tree cannot be written.
pub fn apply_split(
    plan: &mut Plan,
    journal: &mut Journal,
    parent: &SubtaskId,
    pieces: &[Piece],
    max_depth: u8,
    at: Timestamp,
) -> Result<Vec<SubtaskId>, SplitError> {
    let existing = plan
        .subtask(parent)
        .ok_or_else(|| PlanError::UnknownSubtask {
            id: parent.to_string(),
        })?
        .clone();
    let children = children_of(&existing, pieces, at);
    // Captured before the move: a parent that already had settled children keeps them in the
    // tree, and the journal must name only the pieces this split created.
    let ids: Vec<SubtaskId> = children.iter().map(|child| child.id.clone()).collect();
    plan.split(parent, children, max_depth, at)?;
    let stored = plan.subtask(parent).ok_or_else(|| PlanError::UnknownSubtask {
        id: parent.to_string(),
    })?;
    journal.append_scoped(
        at,
        Kind::SubtaskSplit,
        serde_json::json!({
            "parent": parent.to_string(),
            "state": stored.state.as_str(),
            "depth": stored.depth,
            "children": ids.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "pieces": pieces
                .iter()
                .map(|piece| piece.title.as_str())
                .collect::<Vec<_>>(),
        }),
        &Scope::subtask(parent.clone()),
    )?;
    Ok(ids)
}

/// What the scheduler should do with the tree as it stands.
#[derive(Debug)]
pub enum Next<'a> {
    /// Start an attempt at this subtask.
    Run(&'a Subtask),
    /// Every subtask is settled; the task may move to verification.
    Finished,
    /// Subtasks remain but none may run: an unmet dependency is still pending.
    ///
    /// Not reachable from a plan that passed validation, and that is the point — it is what a
    /// human edit that introduced a cycle would produce, and the message names the subtasks
    /// rather than leaving the daemon to idle.
    Stuck { open: Vec<&'a Subtask> },
}

/// The next thing to do, or why nothing can be done.
///
/// A blocked subtask does not stop anything: it counts as settled, so its dependents and the
/// task itself move on. That is the design's rule and one of the two behaviours this milestone
/// exists to get right.
#[must_use]
pub fn next(plan: &Plan) -> Next<'_> {
    if let Some(subtask) = plan.next_ready() {
        return Next::Run(subtask);
    }
    if plan.all_settled() {
        return Next::Finished;
    }
    Next::Stuck {
        open: plan.open_subtasks(),
    }
}

/// The subtasks that hold the task open, for a human reading `status`.
#[must_use]
pub fn pending_ids(plan: &Plan) -> Vec<SubtaskId> {
    plan.open_subtasks()
        .into_iter()
        .map(|subtask| subtask.id.clone())
        .collect()
}

/// Whether any subtask other than `exhausted` could run once `exhausted` is blocked.
///
/// Asked of the tree rather than answered by scanning for a `Pending` sibling: a sibling whose
/// own dependency has not settled is not work available *now*, and L5 escalates only when
/// blocking would leave the task with nothing to do.
#[must_use]
pub fn has_independent(plan: &Plan, exhausted: &SubtaskId) -> bool {
    plan.next_ready_after_retiring(exhausted).is_some()
}

#[cfg(test)]
mod tests;
