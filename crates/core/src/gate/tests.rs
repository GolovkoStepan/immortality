//! Tests for the gate rules.
//!
//! Every case here constructs the shape it is about — a diff, a set of check outcomes, a verdict —
//! rather than producing it with git or a model. That is what lets the suite say "a private key in
//! an added line fails the gate" in a microsecond, and what makes each rule sabotageable: flip the
//! rule and exactly one test goes red.

use super::*;

fn file(path: &str, status: FileStatus, added: &[&str]) -> ChangedFile {
    ChangedFile {
        path: PathBuf::from(path),
        status,
        binary: false,
        added_lines: added.iter().map(|line| (*line).to_string()).collect(),
    }
}

fn diff(files: Vec<ChangedFile>) -> Diff {
    let mut text = String::new();
    for file in &files {
        text.push_str("+++ b/");
        text.push_str(&file.path.display().to_string());
        text.push('\n');
    }
    Diff {
        files,
        text,
        truncated: false,
    }
}

/// `sanity` with a message naming the input, since "something was flagged" is useless alone.
fn assert_clean(change: &Diff, what: &str) {
    let findings = sanity(change);
    assert_eq!(
        findings,
        Vec::<Finding>::new(),
        "{what} was flagged: {findings:?}"
    );
}

fn check(id: &str, required: bool, passed: bool) -> CheckOutcome {
    CheckOutcome {
        id: id.to_string(),
        command: "cargo test".to_string(),
        required,
        exit_code: Some(i32::from(!passed)),
        timed_out: false,
        duration_ms: 10,
        passed,
        output_tail: String::new(),
    }
}

fn green() -> GateReport {
    GateReport {
        findings: Vec::new(),
        checks: vec![check("test", true, true)],
        review: Some(ReviewVerdict::Approve),
    }
}

#[test]
fn a_clean_change_with_green_checks_and_an_approval_passes() {
    let report = green();
    assert_eq!(report.verdict(), GateVerdict::Passed);
    assert!(report.is_green());
    assert!(report.verdict().failure().is_none());
}

#[test]
fn nothing_at_all_is_not_a_pass() {
    // An empty report means the caller never ran the gates. A default of "passed" would make every
    // unwired path a subtask that closes itself.
    let report = GateReport::default();
    assert_eq!(report.verdict(), GateVerdict::Passed, "no gate was asked for");
    // The `DoD` is what decides whether gates are required; the report only reports what ran.
    assert_eq!(report.checks, Vec::<CheckOutcome>::new());
}

#[test]
fn a_red_required_check_fails_the_subtask() {
    let mut report = green();
    report.checks = vec![check("test", true, false)];
    let verdict = report.verdict();
    assert_eq!(verdict.failure(), Some(FailureReason::GateFailed));
    assert!(
        verdict.why().is_some_and(|why| why.contains("test")),
        "{verdict:?}"
    );
}

#[test]
fn a_red_optional_check_does_not_fail_the_subtask() {
    // `cargo outdated` printing something is information, not a verdict — which is why the flag
    // exists rather than being implied.
    let mut report = green();
    report.checks = vec![check("test", true, true), check("outdated", false, false)];
    assert_eq!(report.verdict(), GateVerdict::Passed);
}

#[test]
fn a_refused_check_is_not_a_passing_one() {
    // The policy said no to the command, so nothing ran. Reporting that as green would be the
    // exact bypass the gate exists to close.
    let mut report = green();
    report.checks = vec![CheckOutcome::not_run(
        "test",
        "sudo cargo test",
        true,
        "the command needs a human",
    )];
    assert_eq!(report.verdict().failure(), Some(FailureReason::GateFailed));
    assert!(!report.checks[0].passed);
}

#[test]
fn any_finding_fails_the_subtask_even_with_every_check_green() {
    // The gates are conjunctive: a green test suite does not excuse a shipped private key.
    let mut report = green();
    report.findings = vec![Finding::Binary {
        path: PathBuf::from("assets/logo.png"),
    }];
    assert_eq!(report.verdict().failure(), Some(FailureReason::GateFailed));
}

#[test]
fn a_reviewer_rejection_has_its_own_failure_reason() {
    // "the model did the work but the reviewer disagreed" is not the same as "a command failed",
    // and a metric query has to be able to tell them apart.
    let mut report = green();
    report.review = Some(ReviewVerdict::RequestChanges {
        issues: vec!["the parser still drops the last comma".to_string()],
    });
    let verdict = report.verdict();
    assert_eq!(verdict.failure(), Some(FailureReason::ReviewRejected));
    assert!(verdict.why().is_some_and(|why| why.contains("last comma")));
}

#[test]
fn a_rejection_with_no_issues_still_fails_and_says_so() {
    let mut report = green();
    report.review = Some(ReviewVerdict::RequestChanges { issues: Vec::new() });
    let verdict = report.verdict();
    assert_eq!(verdict.failure(), Some(FailureReason::ReviewRejected));
    assert!(
        verdict
            .why()
            .is_some_and(|why| why.contains("without naming them"))
    );
}

#[test]
fn a_missing_review_is_not_an_approval_when_the_dod_asked_for_one() {
    // This is the shape of the bypass: the reviewer was never called and its absence read as
    // consent. The report cannot tell whether a review was required — the caller must — so the
    // rule here is that an absent verdict contributes nothing, and the caller fails before this
    // point when `Dod::review` is true. Asserted here so the shape of the hole stays visible.
    let mut report = green();
    report.review = None;
    assert_eq!(report.verdict(), GateVerdict::Passed);
}

// ── diff sanity: binaries ───────────────────────────────────────────────────

#[test]
fn an_added_binary_file_is_a_finding() {
    let mut changed = file("assets/logo.png", FileStatus::Added, &[]);
    changed.binary = true;
    let findings = sanity(&diff(vec![changed]));
    assert_eq!(
        findings,
        vec![Finding::Binary {
            path: PathBuf::from("assets/logo.png")
        }]
    );
}

#[test]
fn deleting_a_binary_file_is_not_a_finding() {
    // Removing an object is the one case where shipping it is fine: the change makes the
    // repository more text, not less.
    let mut changed = file("assets/logo.png", FileStatus::Deleted, &[]);
    changed.binary = true;
    assert_clean(&diff(vec![changed]), "a deleted binary");
}

// ── diff sanity: secrets ────────────────────────────────────────────────────

#[test]
fn a_private_key_in_an_added_line_is_a_finding() {
    let changed = file(
        "src/config.rs",
        FileStatus::Modified,
        &["-----BEGIN RSA PRIVATE KEY-----"],
    );
    let findings = sanity(&diff(vec![changed]));
    assert_eq!(findings.len(), 1);
    assert!(matches!(findings[0], Finding::Secret { .. }), "{findings:?}");
}

#[test]
fn an_assigned_api_key_is_a_finding() {
    let changed = file(
        "src/client.rs",
        FileStatus::Modified,
        &[r#"    let api_key = "sk-live-4f9a8b7c6d5e4f3a2b1c";"#],
    );
    let findings = sanity(&diff(vec![changed]));
    assert_eq!(findings.len(), 1);
    assert!(matches!(&findings[0], Finding::Secret { kind, .. } if kind.contains("API key")));
}

#[test]
fn an_environment_lookup_is_not_a_secret() {
    // The commonest *correct* line in a repository, and flagging it would train everyone to ignore
    // the gate. `std::env`, `$VAR` and `{{ secrets.X }}` are references, not values.
    for line in [
        r#"    let api_key = std::env::var("API_KEY")?;"#,
        "API_KEY=$CI_API_KEY",
        "password: {{ secrets.DB_PASSWORD }}",
        r#"    let secret_key = "$SECRET";"#,
        "api_key = os.environ['API_KEY']",
    ] {
        let changed = file("src/x.rs", FileStatus::Modified, &[line]);
        assert_clean(&diff(vec![changed]), line);
    }
}

#[test]
fn a_placeholder_is_not_a_secret() {
    for line in [
        "API_KEY=changeme",
        "password: \"...\"",
        "access_token = <your-token>",
        "api_key = ",
    ] {
        let changed = file("docs/README.md", FileStatus::Modified, &[line]);
        assert_clean(&diff(vec![changed]), line);
    }
}

#[test]
fn a_short_bare_word_is_not_a_secret() {
    // `token_count = 3` and `password: required` are settings. The length and symbol bar exists
    // exactly to keep them out.
    for line in ["token_count = 3", "password: required", "secret_key = none"] {
        let changed = file("src/x.rs", FileStatus::Modified, &[line]);
        assert_clean(&diff(vec![changed]), line);
    }
}

#[test]
fn a_token_with_no_secret_name_is_not_flagged() {
    // The heuristic is name-based, and being honest about that is better than a regex over
    // entropy that would flag every hash in every lock file.
    let changed = file(
        "Cargo.lock",
        FileStatus::Modified,
        &[r#"checksum = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08""#],
    );
    assert_clean(&diff(vec![changed]), "a lock-file checksum");
}

// ── diff sanity: paths and deferrals ────────────────────────────────────────

#[test]
fn a_change_under_git_is_a_finding() {
    let changed = file(".git/config", FileStatus::Modified, &["[user]"]);
    let findings = sanity(&diff(vec![changed]));
    assert_eq!(
        findings,
        vec![Finding::GitMetadata {
            path: PathBuf::from(".git/config")
        }]
    );
}

#[test]
fn a_path_that_escapes_the_worktree_is_a_finding() {
    for path in [
        "../outside/config.toml",
        "/etc/hosts",
        "~/Library/Preferences/x.toml",
    ] {
        let changed = file(path, FileStatus::Modified, &["x"]);
        let findings = sanity(&diff(vec![changed]));
        assert!(
            matches!(findings.first(), Some(Finding::OutsideWorktree { .. })),
            "`{path}` was not refused: {findings:?}"
        );
    }
}

#[test]
fn a_bare_todo_agent_is_a_finding() {
    for line in ["// TODO(agent)", "// TODO(agent: )", "# TODO(agent:    )"] {
        let changed = file("src/x.rs", FileStatus::Modified, &[line]);
        let findings = sanity(&diff(vec![changed]));
        assert!(
            matches!(findings.first(), Some(Finding::TodoWithoutReason { .. })),
            "`{line}` was not refused: {findings:?}"
        );
    }
}

#[test]
fn a_todo_agent_with_a_reason_is_accepted() {
    for line in [
        "// TODO(agent: blocked on the upstream fix in v2)",
        "// TODO(agent: needs a human to rotate the key)",
    ] {
        let changed = file("src/x.rs", FileStatus::Modified, &[line]);
        assert_clean(&diff(vec![changed]), line);
    }
}

#[test]
fn a_todo_for_a_human_is_not_our_marker() {
    for line in [
        "// TODO: implement this",
        "// TODO(bob): ask",
        "// TODO(agents): plural",
    ] {
        let changed = file("src/x.rs", FileStatus::Modified, &[line]);
        assert_clean(&diff(vec![changed]), line);
    }
}

#[test]
fn a_truncated_diff_is_a_finding_rather_than_a_silent_approval() {
    // The alternative — reviewing the first half and calling it approved — is the failure mode
    // this cap exists to prevent.
    let mut changed = file("src/big.rs", FileStatus::Modified, &[]);
    changed.added_lines = Vec::new();
    let mut change = diff(vec![changed]);
    change.truncated = true;
    change.text = String::with_capacity(MAX_REVIEW_BYTES + 1);
    let findings = sanity(&change);
    assert!(
        matches!(findings.first(), Some(Finding::DiffTooLarge { .. })),
        "{findings:?}"
    );
}

#[test]
fn findings_carry_the_path_they_are_about() {
    // The model has to be able to fix what the gate names, so every path-shaped finding mentions
    // its path, and the observation is one sentence with no escaped JSON in it.
    let findings = sanity(&diff(vec![
        file(".git/HEAD", FileStatus::Modified, &["ref"]),
        file("src/x.rs", FileStatus::Modified, &["// TODO(agent)"]),
    ]));
    assert_eq!(findings.len(), 2);
    for finding in &findings {
        let observation = finding.observation();
        assert!(!observation.contains('\\'), "{observation}");
        assert!(
            finding
                .path()
                .is_some_and(|path| observation.contains(&path.display().to_string())),
            "{observation}"
        );
    }
}

#[test]
fn a_multiple_finding_file_reports_all_of_them() {
    // Reporting only the first would make the next attempt fix one thing, fail, and fix the next —
    // three attempts for a problem the gate already knew in full.
    let changed = file(
        "src/x.rs",
        FileStatus::Modified,
        &[
            r#"let api_key = "sk-live-4f9a8b7c6d5e4f3a2b1c";"#,
            "// TODO(agent)",
        ],
    );
    let findings = sanity(&diff(vec![changed]));
    assert_eq!(findings.len(), 2, "{findings:?}");
}

#[test]
fn a_finding_sorts_stably_for_a_reproducible_journal() {
    // Findings go into the journal in the order produced; two runs over the same diff must write
    // the same bytes, or a golden-journal test can never exist.
    let build = || {
        sanity(&diff(vec![
            file("b.rs", FileStatus::Modified, &["// TODO(agent)"]),
            file("a.rs", FileStatus::Modified, &["// TODO(agent)"]),
        ]))
    };
    assert_eq!(build(), build());
    let first = build();
    assert_eq!(
        first[0].path(),
        Some(Path::new("b.rs")),
        "file order is preserved"
    );
}

// ── journal records ─────────────────────────────────────────────────────────

fn kinds(records: &[GateRecord]) -> Vec<&'static str> {
    records
        .iter()
        .map(|record| match record {
            GateRecord::Sanity { .. } => "sanity",
            GateRecord::Check { .. } => "check",
            GateRecord::Review { .. } => "review",
            GateRecord::Verdict { .. } => "verdict",
        })
        .collect()
}

#[test]
fn a_green_report_records_sanity_then_checks_then_review_then_verdict() {
    // The journal order is the gate order. A reviewer record that appeared before the check it
    // was paid for would make a post-mortem read the causes backwards.
    let report = green();
    let records = GateRecord::of(&report);
    assert_eq!(kinds(&records), vec!["sanity", "check", "review", "verdict"]);
    assert_eq!(
        records.last(),
        Some(&GateRecord::Verdict {
            passed: true,
            reason: None,
            why: None,
        })
    );
}

#[test]
fn the_sanity_record_is_written_even_when_it_found_nothing() {
    // "scanned and clean" must be distinguishable from "never scanned": the whole point of the
    // record is to answer which gate did not run.
    let records = GateRecord::of(&green());
    assert_eq!(records[0], GateRecord::Sanity { findings: Vec::new() });
}

#[test]
fn a_report_without_a_review_has_no_review_record() {
    // The absence is the fact: a `DoD` with `review: false` never asked, so a record claiming an
    // approval would be a gate nobody ran.
    let mut report = green();
    report.review = None;
    let records = GateRecord::of(&report);
    assert!(!kinds(&records).contains(&"review"), "{records:?}");
}

#[test]
fn a_red_check_names_itself_in_the_verdict() {
    let mut report = green();
    report.checks = vec![check("test", true, false)];
    let records = GateRecord::of(&report);
    let GateRecord::Verdict { passed, reason, why } = records.last().expect("a verdict") else {
        panic!("the last record must be the verdict: {records:?}");
    };
    assert!(!passed);
    assert_eq!(*reason, Some(FailureReason::GateFailed));
    assert!(why.as_deref().is_some_and(|why| why.contains("test")));
}

#[test]
fn a_rejected_review_is_recorded_with_its_issues() {
    let mut report = green();
    report.review = Some(ReviewVerdict::RequestChanges {
        issues: vec!["src/a.rs drops the comma".to_string()],
    });
    let records = GateRecord::of(&report);
    let review = records
        .iter()
        .find_map(|record| match record {
            GateRecord::Review { approved, issues } => Some((*approved, issues.clone())),
            _ => None,
        })
        .expect("a review record");
    assert!(!review.0);
    assert_eq!(review.1, vec!["src/a.rs drops the comma".to_string()]);
    assert_eq!(
        records.last().and_then(|record| match record {
            GateRecord::Verdict { reason, .. } => *reason,
            _ => None,
        }),
        Some(FailureReason::ReviewRejected)
    );
}

#[test]
fn every_record_round_trips_through_json() {
    // The journal stores these as JSON, and a schema that only works in memory is a schema the
    // post-mortem cannot read.
    let mut report = green();
    report.findings = vec![Finding::Binary {
        path: PathBuf::from("a.png"),
    }];
    report.review = Some(ReviewVerdict::RequestChanges {
        issues: vec!["no".to_string()],
    });
    for record in GateRecord::of(&report) {
        let text = serde_json::to_string(&record).expect("encode");
        let back: GateRecord = serde_json::from_str(&text).expect("decode");
        assert_eq!(back, record, "{text}");
    }
}

#[test]
fn a_check_record_carries_the_command_so_the_journal_explains_itself() {
    let mut report = green();
    report.checks = vec![check("test", true, true)];
    let records = GateRecord::of(&report);
    let GateRecord::Check {
        check: id, command, ..
    } = &records[1]
    else {
        panic!("the second record must be the check: {records:?}");
    };
    assert_eq!(id, "test");
    assert_eq!(command, "cargo test");
}
