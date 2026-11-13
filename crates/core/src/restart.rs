//! Deciding to restart an attempt, and what the next attempt is told.
//!
//! # What L3 is
//!
//! `docs/design.md` §6 defines the third rung as "roll back the worktree to the last green
//! checkpoint, start with a fresh context, and put a *different strategy* in the brief". Three
//! things have to happen together, because each one alone is useless: without the rollback the
//! next attempt inherits a half-broken tree, without the fresh context it inherits the
//! conversation that was going in circles, and without a new strategy it inherits the same plan.
//!
//! # Why there are two decisions, not one
//!
//! [`Decision::Retry`] keeps the tree and [`Decision::Restart`] rolls it back, and the split is
//! not a refinement — it is what a red gate means. A reviewer read the diff and named what is
//! wrong with it, so the work on disk is the closest thing to a finished subtask that exists;
//! resetting it away and asking the model to reproduce it from a brief is how a nearly-done
//! subtask costs three more attempts. A detector or a budget says something different — *this
//! attempt is not getting there* — and only that is a reason to throw the tree away.
//!
//! # Why the decision is pure
//!
//! Rolling back is git, journalling is fsync, and starting an attempt costs tokens. The decision
//! itself is a function of what ended the attempt and how much budget is left — no I/O, no clock.
//! That split is what makes the two properties testable in microseconds: **a restarted attempt is
//! told why the last one stopped**, and **a subtask that has spent its budget is not restarted
//! again**. The git work is behind [`Checkpoints`], implemented by the daemon.
//!
//! # Why `Abandoned` is not a restart
//!
//! An attempt interrupted from outside — a backend outage, a reboot — says nothing about the
//! model's approach (D-027). It is retried as *the same attempt* when the daemon is ready, with
//! its conversation replayed from the journal, and it never spends a restart. Only a verdict on
//! the work does.

use crate::detect::Detector;
use crate::domain::FailureReason;

/// The most failed approaches a brief repeats.
///
/// A report accumulates rejected ideas for the whole subtask; past a handful the list stops being
/// a warning and starts being a prompt. The newest are the most relevant, which is also why the
/// report itself is bounded when it is written.
pub const MAX_REJECTED_IDEAS: usize = 6;

/// How much of a previous strategy or rejected idea is quoted.
///
/// A strategy is model prose and can be long. Quoting it whole spends the tokens the restart was
/// meant to save, and the first sentence is what names the approach.
pub const MAX_QUOTE_CHARS: usize = 400;

/// A budget that ran out.
///
/// Kept apart from [`crate::context::Verdict`] because this is the *restart* vocabulary: what the
/// next attempt is told about why its predecessor stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Limit {
    /// `max_steps_per_attempt`.
    Steps,
    /// `max_tokens_per_attempt`, or the hard context budget.
    Tokens,
    /// `max_wall_time_per_attempt`.
    Time,
}

impl Limit {
    /// The words used in a brief.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Steps => "steps",
            Self::Tokens => "tokens",
            Self::Time => "wall time",
        }
    }
}

/// Why an attempt ended, as far as the restart decision is concerned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cause {
    /// An L3 detector fired: the attempt was going in circles.
    Detector {
        /// Which signal concluded that another step was worthless.
        detector: Detector,
        /// The measurement that tripped the stop threshold.
        metric: u32,
    },
    /// A budget ran out.
    Limit(Limit),
    /// Nothing progressed for `stall_steps` steps.
    Stalled,
    /// The model called a tool it was never offered.
    UnknownTool { name: String },
    /// The model answered with nothing at all.
    EmptyReply,
    /// The claim was rejected by the gates.
    Gate {
        /// Which gate failed, as the attempt records it.
        reason: FailureReason,
        /// What the gate said, verbatim: this is what the model has to fix.
        why: String,
    },
}

impl Cause {
    /// The wire name, for an `attempt_restarted` record.
    ///
    /// Detector causes use the detector's own name (`d1`…`d5`) so that a metric query over
    /// restarts joins the `detector_fired` records without a translation table.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Detector { detector, .. } => detector.as_str(),
            Self::Limit(limit) => match limit {
                Limit::Steps => "steps",
                Limit::Tokens => "tokens",
                Limit::Time => "time",
            },
            Self::Stalled => "stalled",
            Self::UnknownTool { .. } => "unknown_tool",
            Self::EmptyReply => "empty",
            Self::Gate { .. } => "gate",
        }
    }

    /// What the next attempt is told happened, as one clause.
    #[must_use]
    pub fn observation(&self) -> String {
        match self {
            Self::Detector { detector, metric } => {
                let what = match detector {
                    Detector::ExactRepeat => {
                        format!("it repeated the same tool call {metric} time(s)")
                    }
                    Detector::SemanticRepeat => {
                        format!("different calls kept returning the same result ({metric} time(s))")
                    }
                    Detector::NoProgress => format!(
                        "nothing changed for {metric} steps: the tree, the gates and the facts \
                         were identical"
                    ),
                    Detector::ReasoningLoop => format!(
                        "the message began repeating itself ({metric} occurrence(s)) and the \
                         stream was cut"
                    ),
                    Detector::Budget => "a budget ran out".to_string(),
                };
                format!("{what}; the ladder stopped it (detector {})", detector.as_str())
            }
            Self::Limit(limit) => format!("it ran out of {} (detector d5)", limit.as_str()),
            Self::Stalled => {
                "it produced no tool call for the stall budget, so nothing happened on disk".to_string()
            }
            Self::UnknownTool { name } => {
                format!("it called `{name}`, which is not a tool it was offered")
            }
            Self::EmptyReply => "it answered with nothing".to_string(),
            Self::Gate { why, .. } => format!("its claim was rejected by the gates: {why}"),
        }
    }

    /// The route the next attempt should take instead.
    ///
    /// One or two sentences, and always a *change of method*: "try harder" is what the model was
    /// already doing, and telling it so again is how an attempt becomes three.
    #[must_use]
    pub const fn advice(&self) -> &'static str {
        match self {
            Self::Detector {
                detector: Detector::ExactRepeat,
                ..
            } => {
                "The evidence you need is already in the history you were given. Read the last \
                 observation again, say what it does and does not prove, and take a different \
                 step: another file, a narrower search, or a smaller edit."
            }
            Self::Detector {
                detector: Detector::SemanticRepeat,
                ..
            } => {
                "Two different commands told you the same thing, so a third variant will too. \
                 Change what you are looking at, not how you ask for it."
            }
            Self::Detector {
                detector: Detector::NoProgress,
                ..
            } => {
                "Reading and re-reading is not progress. Make one small edit that lands on disk, \
                 then let a check tell you whether it was right."
            }
            Self::Detector {
                detector: Detector::ReasoningLoop,
                ..
            } => {
                "Write at most a couple of sentences before acting, and do not restate the plan. \
                 One tool call per step."
            }
            Self::Detector {
                detector: Detector::Budget,
                ..
            } => {
                "The budget that ran out was not a step budget, so the plan was too expensive \
                 rather than too long. Do the same work with fewer and cheaper observations, and \
                 finish the subtask as soon as its definition of done holds."
            }
            Self::Limit(Limit::Steps) => {
                "The last plan did not fit in one pass. Work in smaller pieces: one file, one \
                 function, one check per step."
            }
            Self::Limit(Limit::Tokens) => {
                "Keep the prompt small: search and read slices instead of whole files, and do not \
                 repeat long observations back."
            }
            Self::Limit(Limit::Time) => {
                "Something took too long. Prefer fast, bounded commands, and never re-run a \
                 command that already passed."
            }
            Self::Stalled => "Every step must either change a file or run a check. Talking is not a step.",
            Self::UnknownTool { .. } => {
                "Call only the tools listed in this prompt. If what you need is missing, say so \
                 in the summary of `finish_subtask` instead of inventing a tool."
            }
            Self::EmptyReply => {
                "Always answer with a tool call, or with `finish_subtask` once the work is on \
                 disk."
            }
            Self::Gate { .. } => {
                "Fix exactly what the gate named, run the check that failed yourself, and claim \
                 the subtask again only after it passes."
            }
        }
    }
}

/// What ended the attempt, from the caller's point of view.
///
/// A separate type from [`Cause`] because two endings are not failures and must not be turned
/// into one: a verified claim closes the subtask, and a split is a decision about the tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ended {
    /// The claim passed every gate.
    Verified,
    /// The model proposed a decomposition; only the scheduler may apply it.
    Split,
    /// The attempt was interrupted from outside; retry it rather than replacing it.
    Abandoned,
    /// The attempt ended without passing: this is why.
    Failed(Cause),
}

/// How much of the budget is left.
///
/// Passed in rather than read from config so the decision is a function of numbers, not of a
/// config file: a test can ask "with one restart left, is this restarted?" without building one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Allowance {
    /// Attempts that may still start, `max_attempts_per_subtask` minus the ones already used.
    pub attempts_left: u32,
    /// Restarts that may still happen, `max_restarts_per_subtask` minus the ones already used.
    pub restarts_left: u32,
}

/// Whether to restart, and if not, why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Start another attempt with a fresh context and a new strategy, but **leave the tree as it
    /// is**.
    ///
    /// Used when the work on disk is what has to be fixed — a rejected claim. Rolling back would
    /// throw away an almost-passing change and pay for it a second time.
    Retry(Cause),
    /// Start another attempt: roll back to the base checkpoint first, then run with a new
    /// strategy.
    Restart(Cause),
    /// Stop. The scheduler decides what the subtask becomes.
    Stop(Stop),
}

/// Why no further attempt is started.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stop {
    /// The claim was verified; the subtask may close.
    Done,
    /// The model proposed a split; the tree decides whether it is legal.
    Split,
    /// The attempt was interrupted from outside and is retried as the same attempt.
    Abandoned,
    /// The subtask has spent its attempts or its restarts.
    Exhausted(Allowance),
    /// The tree could not be put back into a known state, so continuing would build on unknown.
    ///
    /// Carries the reason so the escalation names the git failure instead of saying "stuck".
    Blocked(String),
}

/// The decision, as a function of what ended and what is left.
///
/// The strategy is **not** built here: it needs the previous strategy and the report's rejected
/// ideas, which are the caller's to keep, and folding them in would make this function depend on
/// prose it does not own. [`strategy`] is the other half.
///
/// # Why a rejected claim keeps the tree
///
/// A red gate is evidence about the work *on disk*, not about the attempt having wandered: the
/// reviewer read a diff and named what is wrong with it. Rolling that diff away and asking the
/// model to reproduce it from a brief is how a nearly-finished subtask costs three more attempts.
/// So the claim path retries in place, and only a detector or a budget — which say "this attempt
/// is not getting there" — roll back.
#[must_use]
pub fn decide(ended: Ended, allowance: Allowance) -> Decision {
    let cause = match ended {
        Ended::Verified => return Decision::Stop(Stop::Done),
        Ended::Split => return Decision::Stop(Stop::Split),
        Ended::Abandoned => return Decision::Stop(Stop::Abandoned),
        Ended::Failed(cause) => cause,
    };
    // `attempts_left == 0` is checked separately from `restarts_left == 0` so the message can say
    // which budget ran out: a subtask blocked after three attempts is a different conversation
    // with a human than one blocked after two restarts.
    if allowance.attempts_left == 0 || allowance.restarts_left == 0 {
        return Decision::Stop(Stop::Exhausted(allowance));
    }
    if matches!(cause, Cause::Gate { .. }) {
        return Decision::Retry(cause);
    }
    Decision::Restart(cause)
}

/// The strategy section of the next attempt's brief.
///
/// `previous` is the strategy the last attempt ran under (empty for an attempt that had none), and
/// `rejected` is the accumulated [`crate::domain::SubtaskReport::rejected_ideas`]. Both are quoted
/// rather than summarized: the model is the one that has to avoid them, and a paraphrase is how a
/// rejected idea comes back under a new name.
#[must_use]
pub fn strategy(cause: &Cause, previous: &str, rejected: &[String]) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    out.push_str("This is a fresh attempt at the same subtask, with no memory of the previous one. ");
    let _ = writeln!(out, "The previous attempt ended because {}.", cause.observation());
    out.push('\n');
    let _ = writeln!(out, "{}", cause.advice());

    if !previous.trim().is_empty() {
        out.push('\n');
        let _ = writeln!(out, "The previous attempt ran under this strategy:");
        let _ = writeln!(out, "> {}", one_line(previous, MAX_QUOTE_CHARS));
        let _ = writeln!(
            out,
            "Do not run it again. Change the method, not only the wording."
        );
    }

    // The newest are the most relevant, and the order inside the block stays chronological: a
    // list read newest-first is a list a model matches against its own last idea.
    let start = rejected.len().saturating_sub(MAX_REJECTED_IDEAS);
    let rejected = &rejected[start..];
    if !rejected.is_empty() {
        out.push('\n');
        let _ = writeln!(out, "These approaches already failed for this subtask:");
        for idea in rejected {
            let _ = writeln!(out, "- {}", one_line(idea, MAX_QUOTE_CHARS));
        }
        out.push('\n');
        let _ = writeln!(out, "Do not propose them again.");
    }
    out.trim_end().to_string()
}

/// One line of text, cut at a character boundary.
///
/// Whitespace is collapsed because the quote goes into a Markdown blockquote: a strategy written
/// over ten lines would otherwise turn the brief's layout into a second strategy.
fn one_line(text: &str, max: usize) -> String {
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max {
        return collapsed;
    }
    let mut cut: String = collapsed.chars().take(max).collect();
    cut.push('…');
    cut
}

/// A snapshot of the working tree, as a restart needs it.
///
/// `sha` is a real commit, so a rollback is `reset --hard` and nothing more; `branch` is the ref
/// that will still point at it if the journal write is lost to a crash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    pub sha: String,
    pub branch: String,
}

/// The tree operations a restart needs, behind a trait.
///
/// Two methods, because those are the two halves of L3's tree work, and their order is the whole
/// crash-safety argument: **snapshot, journal, then move**. The implementation is git in the
/// daemon; a test can fake both and assert the restart's *decisions* without a repository.
pub trait Checkpoints {
    /// Snapshot the tree as it is, without moving it.
    ///
    /// `Ok(None)` means there is nothing to snapshot — a repository with no commits yet — which
    /// is not an error: the attempt simply has no base to roll back to.
    ///
    /// # Errors
    /// Git failed. The caller must not start an attempt it cannot restart.
    fn snapshot(&self, reason: &str) -> Result<Option<Snapshot>, String>;

    /// Move the tree to `sha`.
    ///
    /// The caller takes and journals its own snapshot of the work this destroys, which is why
    /// there is no safety snapshot here: taking one would either duplicate the journal record or
    /// (worse) destroy the tree before anything named the work it threw away.
    ///
    /// # Errors
    /// Git failed, and the tree may be part-way through the reset. The caller stops rather than
    /// continuing on a tree whose state nobody knows.
    fn restore(&self, sha: &str) -> Result<(), String>;
}

#[cfg(test)]
mod tests;
