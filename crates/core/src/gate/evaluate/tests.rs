//! Tests for the order of the gates and for the short-circuit rules.
//!
//! The fake world records every call it receives, so these tests can assert not only the verdict
//! but what was *not* run — which is the half of "cheap first" that a verdict alone cannot show.

use super::*;
use crate::domain::{Dod, DodCheck};
use std::cell::RefCell;
use std::path::PathBuf;

#[derive(Debug, Default)]
struct World {
    diff: RefCell<Diff>,
    runs: RefCell<Vec<String>>,
    refused: RefCell<Vec<String>>,
    /// Per-command result; a missing entry is an exit 0.
    outcomes: RefCell<Vec<(String, CommandRun)>>,
}

impl World {
    fn new(diff: Diff) -> Self {
        Self {
            diff: RefCell::new(diff),
            ..Self::default()
        }
    }

    fn refusing(command: &str) -> Self {
        let world = Self::new(Diff::default());
        world.refused.borrow_mut().push(command.to_string());
        world
    }

    fn answering(command: &str, run: CommandRun) -> Self {
        let world = Self::new(Diff::default());
        world.outcomes.borrow_mut().push((command.to_string(), run));
        world
    }

    fn ran(&self) -> Vec<String> {
        self.runs.borrow().clone()
    }
}

impl Gates for World {
    fn diff(&self) -> Result<Diff, String> {
        Ok(self.diff.borrow().clone())
    }

    fn authorise(&self, command: &str) -> Result<(), String> {
        if self.refused.borrow().iter().any(|refused| refused == command) {
            return Err("needs a human".to_string());
        }
        Ok(())
    }

    fn run(&self, command: &str) -> Result<CommandRun, String> {
        self.runs.borrow_mut().push(command.to_string());
        let outcomes = self.outcomes.borrow();
        if let Some((_, run)) = outcomes.iter().find(|(name, _)| name == command) {
            return Ok(run.clone());
        }
        Ok(ok())
    }
}

#[derive(Debug)]
struct ScriptedReviewer {
    answer: Result<String, String>,
    seen: RefCell<Vec<ReviewRequest>>,
}

impl ScriptedReviewer {
    fn answering(answer: &str) -> Self {
        Self {
            answer: Ok(answer.to_string()),
            seen: RefCell::new(Vec::new()),
        }
    }

    fn failing() -> Self {
        Self {
            answer: Err("backend unreachable".to_string()),
            seen: RefCell::new(Vec::new()),
        }
    }
}

impl Reviewer for ScriptedReviewer {
    fn review(&self, request: &ReviewRequest) -> Result<String, String> {
        self.seen.borrow_mut().push(request.clone());
        self.answer.clone()
    }
}

fn ok() -> CommandRun {
    CommandRun {
        exit_code: Some(0),
        timed_out: false,
        duration_ms: 5,
        output_tail: String::new(),
    }
}

fn failing() -> CommandRun {
    CommandRun {
        exit_code: Some(101),
        timed_out: false,
        duration_ms: 5,
        output_tail: "test failed".to_string(),
    }
}

fn check(id: &str, required: bool) -> DodCheck {
    DodCheck {
        id: id.to_string(),
        command: format!("run-{id}"),
        required,
    }
}

fn dod(checks: Vec<DodCheck>, review: bool) -> Dod {
    Dod {
        statement: "the gate holds".to_string(),
        checks,
        review,
    }
}

fn broken_diff() -> Diff {
    Diff {
        files: vec![crate::gate::ChangedFile {
            path: PathBuf::from("src/x.rs"),
            status: crate::gate::FileStatus::Modified,
            binary: false,
            added_lines: vec!["let api_key = \"sk-live-4f9a8b7c6d5e4f3a2b1c\";".to_string()],
        }],
        text: "+let api_key = \"sk-live-4f9a8b7c6d5e4f3a2b1c\";".to_string(),
        truncated: false,
    }
}

#[test]
fn every_green_gate_passes() {
    let world = World::new(Diff::default());
    let reviewer = ScriptedReviewer::answering(r#"{"verdict":"approve"}"#);
    let report = evaluate(
        &dod(vec![check("build", true), check("test", true)], true),
        &world,
        Some(&reviewer),
    )
    .expect("evaluate");

    assert!(report.is_green());
    assert_eq!(world.ran(), vec!["run-build", "run-test"]);
    assert_eq!(report.review, Some(ReviewVerdict::Approve));
}

#[test]
fn a_broken_diff_stops_before_any_command_or_review() {
    // The whole point of the order: no test suite, no model call, when the change already fails.
    let world = World::new(broken_diff());
    let reviewer = ScriptedReviewer::answering(r#"{"verdict":"approve"}"#);
    let report = evaluate(&dod(vec![check("build", true)], true), &world, Some(&reviewer)).expect("evaluate");

    assert!(!report.is_green());
    assert_eq!(world.ran(), Vec::<String>::new(), "no command should run");
    assert!(reviewer.seen.borrow().is_empty(), "the reviewer must not be paid");
    assert_eq!(report.checks, Vec::<CheckOutcome>::new(), "nothing ran");
    assert!(report.review.is_none());
    assert_ne!(report.findings, Vec::<Finding>::new());
}

#[test]
fn a_red_required_check_stops_the_remaining_commands() {
    let world = World::answering("run-build", failing());
    let reviewer = ScriptedReviewer::answering(r#"{"verdict":"approve"}"#);
    let report = evaluate(
        &dod(vec![check("build", true), check("test", true)], true),
        &world,
        Some(&reviewer),
    )
    .expect("evaluate");

    assert!(!report.is_green());
    assert_eq!(world.ran(), vec!["run-build"], "`test` must not run");
    assert!(reviewer.seen.borrow().is_empty());
    assert!(report.review.is_none());
}

#[test]
fn a_red_optional_check_does_not_stop_anything() {
    let world = World::answering("run-outdated", failing());
    let reviewer = ScriptedReviewer::answering(r#"{"verdict":"approve"}"#);
    let report = evaluate(
        &dod(vec![check("outdated", false), check("test", true)], true),
        &world,
        Some(&reviewer),
    )
    .expect("evaluate");

    assert!(report.is_green());
    assert_eq!(world.ran(), vec!["run-outdated", "run-test"]);
}

#[test]
fn a_refused_command_is_a_red_check_and_nothing_runs() {
    let world = World::refusing("run-build");
    let report = evaluate(&dod(vec![check("build", true)], false), &world, None).expect("evaluate");

    assert!(!report.is_green());
    let outcome = &report.checks[0];
    assert!(!outcome.passed);
    assert!(outcome.output_tail.contains("refused"), "{outcome:?}");
    assert_eq!(world.ran(), Vec::<String>::new());
}

#[test]
fn a_timeout_is_a_failure_even_with_a_zero_exit_code() {
    // A killed process's status is not evidence: the check never finished, so it never proved
    // anything.
    let world = World::answering(
        "run-test",
        CommandRun {
            exit_code: Some(0),
            timed_out: true,
            duration_ms: 600_000,
            output_tail: "still running".to_string(),
        },
    );
    let report = evaluate(&dod(vec![check("test", true)], false), &world, None).expect("evaluate");
    assert!(!report.is_green());
    assert_eq!(report.checks[0].exit_code, Some(0));
    assert!(!report.checks[0].passed);
}

#[test]
fn a_dod_without_a_review_never_calls_the_reviewer() {
    let world = World::new(Diff::default());
    let reviewer = ScriptedReviewer::answering(r#"{"verdict":"request_changes","issues":["no"]}"#);
    let report =
        evaluate(&dod(vec![check("build", true)], false), &world, Some(&reviewer)).expect("evaluate");

    assert!(report.is_green());
    assert!(reviewer.seen.borrow().is_empty());
    assert!(report.review.is_none());
}

#[test]
fn a_required_review_with_no_reviewer_fails_closed() {
    // The unwired path is the one that would otherwise approve everything it saw.
    let world = World::new(Diff::default());
    let report = evaluate(&dod(Vec::new(), true), &world, None).expect("evaluate");

    assert!(!report.is_green());
    assert_eq!(
        report.verdict().failure(),
        Some(crate::domain::FailureReason::ReviewRejected)
    );
    assert!(!report.review.expect("a verdict").approved());
}

#[test]
fn a_reviewer_outage_is_a_rejection_rather_than_an_approval() {
    let world = World::new(Diff::default());
    let reviewer = ScriptedReviewer::failing();
    let report = evaluate(&dod(Vec::new(), true), &world, Some(&reviewer)).expect("evaluate");

    assert!(!report.is_green());
    match report.review.expect("a verdict") {
        ReviewVerdict::RequestChanges { issues } => {
            assert!(issues[0].contains("could not be called"), "{issues:?}");
        }
        ReviewVerdict::Approve => panic!("a backend outage was read as approval"),
    }
}

#[test]
fn the_reviewer_sees_only_the_statement_and_the_diff() {
    // A reviewer that also saw the conversation would be reviewing intentions instead of output,
    // and the request type is what makes that impossible.
    let world = World::new(Diff {
        files: Vec::new(),
        text: "+++ b/src/a.rs\n+fn a() {}".to_string(),
        truncated: false,
    });
    let reviewer = ScriptedReviewer::answering(r#"{"verdict":"approve"}"#);
    evaluate(&dod(Vec::new(), true), &world, Some(&reviewer)).expect("evaluate");

    let seen = reviewer.seen.borrow();
    let request = seen.first().expect("a request");
    assert_eq!(request.statement, "the gate holds");
    assert!(request.diff.text.contains("fn a()"));
}

#[test]
fn a_diff_that_cannot_be_produced_is_the_only_error() {
    #[derive(Debug)]
    struct Broken;
    impl Gates for Broken {
        fn diff(&self) -> Result<Diff, String> {
            Err("git is gone".to_string())
        }
        fn authorise(&self, _command: &str) -> Result<(), String> {
            Ok(())
        }
        fn run(&self, _command: &str) -> Result<CommandRun, String> {
            Ok(ok())
        }
    }
    let error = evaluate(&dod(Vec::new(), false), &Broken, None).expect_err("no diff");
    assert!(error.contains("git is gone"), "{error}");
}

#[test]
fn a_command_that_cannot_start_is_a_red_check() {
    #[derive(Debug)]
    struct Unstartable;
    impl Gates for Unstartable {
        fn diff(&self) -> Result<Diff, String> {
            Ok(Diff::default())
        }
        fn authorise(&self, _command: &str) -> Result<(), String> {
            Ok(())
        }
        fn run(&self, _command: &str) -> Result<CommandRun, String> {
            Err("no such binary".to_string())
        }
    }
    let report = evaluate(&dod(vec![check("test", true)], false), &Unstartable, None).expect("evaluate");
    assert!(!report.is_green());
    assert!(report.checks[0].output_tail.contains("could not be started"));
}

#[test]
fn the_checks_keep_the_order_the_dod_declared() {
    // `build` before `test` costs one compile instead of two, which is why `required_checks`
    // preserves order rather than sorting.
    let world = World::new(Diff::default());
    evaluate(
        &dod(vec![check("build", true), check("test", true)], false),
        &world,
        None,
    )
    .expect("evaluate");
    assert_eq!(world.ran(), vec!["run-build", "run-test"]);
}

#[test]
fn findings_text_is_one_sentence_per_line() {
    let findings = vec![
        Finding::Binary {
            path: PathBuf::from("a.png"),
        },
        Finding::GitMetadata {
            path: PathBuf::from(".git/config"),
        },
    ];
    let text = findings_text(&findings);
    assert_eq!(text.lines().count(), 2, "{text}");
    assert!(text.contains("a.png"));
    assert!(text.contains(".git/config"));
}
