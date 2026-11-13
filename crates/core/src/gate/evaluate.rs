//! Running the three gates in the right order, and stopping at the first red one.
//!
//! # Why the order lives here
//!
//! Cheap-first is not a preference, it is a token budget. A change that adds a private key is
//! already failed; running a two-minute test suite and paying for a reviewer before saying so
//! wastes the two resources this project exists to conserve. The order is diff sanity, then the
//! commands, then the reviewer, and a red gate short-circuits everything after it.
//!
//! # Why the I/O is behind a trait
//!
//! [`Gates`] is git, the policy gate and the command runner; [`Reviewer`] is a model call. With
//! both faked, the entire decision — which command runs, what a refusal means, what stops the
//! evaluation — is testable in microseconds and without a repository. The daemon supplies the real
//! implementations; the rules stay here.

use super::{CheckOutcome, Diff, Finding, GateReport, ReviewRequest, ReviewVerdict, sanity};
use crate::domain::{Dod, DodCheck};

/// What one command produced, without the check metadata the caller attaches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandRun {
    /// `None` when the process was killed and never reported a code.
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub duration_ms: u64,
    /// The tail of stdout and stderr, already combined and bounded.
    pub output_tail: String,
}

impl CommandRun {
    /// Whether this run counts as passing.
    ///
    /// A timeout is a failure even if the process happened to exit 0: a killed process's status
    /// says nothing about the work, and a check that never finished never proved anything.
    #[must_use]
    pub fn passed(&self) -> bool {
        !self.timed_out && self.exit_code == Some(0)
    }
}

/// Everything the evaluation needs from the world.
///
/// `authorise` is separate from `run` on purpose: the policy decision is a fact about the command
/// and the worktree, not about whether the machine happens to be able to execute it, and keeping
/// them apart is what makes "a refused check is not a passing check" expressible.
pub trait Gates {
    /// The change under review, against `base`.
    ///
    /// `base` is the commit a restarted attempt rolled back to, when the caller knows it. `None`
    /// means "against `HEAD`", which is broader and therefore safe: an over-wide diff costs a
    /// reviewer some reading, an under-wide one approves work nobody saw. The parameter is here
    /// rather than on the implementation because it changes *per attempt*, and a gate world built
    /// once per task would otherwise have to be rebuilt after every restart just to move a SHA.
    fn diff(&self, base: Option<&str>) -> Result<Diff, String>;

    /// Whether the policy allows this command, or why not.
    fn authorise(&self, command: &str) -> Result<(), String>;

    /// Run one command. Called only after `authorise` said yes.
    fn run(&self, command: &str) -> Result<CommandRun, String>;
}

/// The reviewer, which is one model call with no tools.
pub trait Reviewer {
    /// Return the model's raw answer, which [`super::parse_review`] then reads.
    fn review(&self, request: &ReviewRequest) -> Result<String, String>;
}

/// Evaluate every gate the definition of done asks for.
///
/// Stops at the first red required check, because the remaining checks are about a build that
/// already failed and their output would be noise in the journal. Optional checks still run — that
/// is what "informational" means — and a red one changes nothing.
///
/// `reviewer` is `None` only when the daemon has not wired one; the `DoD` decides whether a review
/// is needed, and a `DoD` that asks for one without a reviewer fails closed rather than silently
/// skipping the gate.
///
/// # Errors
/// Only when the diff cannot be produced. Everything after that is a `GateReport`, because "the
/// command failed" and "the reviewer said no" are verdicts, not errors — and a caller that treats
/// them as errors ends up with a subtask that neither closes nor fails.
pub fn evaluate(
    dod: &Dod,
    gates: &dyn Gates,
    reviewer: Option<&dyn Reviewer>,
    base: Option<&str>,
) -> Result<GateReport, String> {
    let diff = gates.diff(base)?;
    let findings = sanity(&diff);
    if !findings.is_empty() {
        // Nothing else can change the verdict, and the cheapest gate is the one that already
        // decided. `checks` stays empty on purpose: "not run" is the truth, and a fabricated
        // passing check here is the bypass the gates exist to close.
        return Ok(GateReport {
            findings,
            checks: Vec::new(),
            review: None,
        });
    }

    let mut report = GateReport {
        findings,
        checks: Vec::new(),
        review: None,
    };
    let mut stopped = false;
    for check in &dod.checks {
        let outcome = run_check(gates, check);
        let failed_required = check.required && !outcome.passed;
        report.checks.push(outcome);
        if failed_required {
            stopped = true;
            break;
        }
    }
    if stopped {
        return Ok(report);
    }

    if dod.review {
        report.review = Some(review(dod, reviewer, &diff));
    }
    Ok(report)
}

/// One check, authorising it first.
fn run_check(gates: &dyn Gates, check: &DodCheck) -> CheckOutcome {
    if let Err(why) = gates.authorise(&check.command) {
        return CheckOutcome::not_run(
            check.id.clone(),
            check.command.clone(),
            check.required,
            format!("the policy refused this command: {why}"),
        );
    }
    match gates.run(&check.command) {
        Ok(run) => CheckOutcome {
            id: check.id.clone(),
            command: check.command.clone(),
            required: check.required,
            exit_code: run.exit_code,
            timed_out: run.timed_out,
            duration_ms: run.duration_ms,
            passed: run.passed(),
            output_tail: run.output_tail,
        },
        Err(why) => CheckOutcome::not_run(
            check.id.clone(),
            check.command.clone(),
            check.required,
            format!("the command could not be started: {why}"),
        ),
    }
}

/// The review, failing closed on every way it can go wrong.
fn review(dod: &Dod, reviewer: Option<&dyn Reviewer>, diff: &Diff) -> ReviewVerdict {
    let Some(reviewer) = reviewer else {
        return ReviewVerdict::RequestChanges {
            issues: vec!["the definition of done requires a review, but no reviewer is wired".to_string()],
        };
    };
    let request = ReviewRequest {
        statement: dod.statement.clone(),
        diff: diff.clone(),
    };
    match reviewer.review(&request) {
        // `parse_review` is closed by default: only an explicit approval approves.
        Ok(answer) => super::parse_review(&answer).0,
        // A backend that did not answer is not a reviewer that said yes.
        Err(why) => ReviewVerdict::RequestChanges {
            issues: vec![format!("the reviewer could not be called: {why}")],
        },
    }
}

/// The findings as one block for a brief or a journal field.
#[must_use]
pub fn findings_text(findings: &[Finding]) -> String {
    findings
        .iter()
        .map(Finding::observation)
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests;
