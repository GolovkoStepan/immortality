//! Tests for the restart decision and the strategy text.
//!
//! Every case here is a pair of values in and a value out: no git, no journal, no model. That is
//! what lets the suite state the two rules the milestone exists for — an unsuccessful attempt is
//! told *why* it failed, and a subtask that has spent its budget stops.

use super::*;

fn detector(detector: Detector, metric: u32) -> Cause {
    Cause::Detector { detector, metric }
}

fn room(attempts: u32, restarts: u32) -> Allowance {
    Allowance {
        attempts_left: attempts,
        restarts_left: restarts,
    }
}

#[test]
fn a_verified_claim_closes_rather_than_restarting() {
    // The only happy path: a green verdict is a closure, not a reason to start again. Restarting
    // here would make a green attempt spend a second one on the same work.
    assert_eq!(decide(Ended::Verified, room(3, 2)), Decision::Stop(Stop::Done));
}

#[test]
fn a_split_is_the_trees_decision() {
    // The runner hands the pieces up; whether they are legal is the scheduler's question, so the
    // restart decision must not quietly run another attempt over it.
    assert_eq!(decide(Ended::Split, room(3, 2)), Decision::Stop(Stop::Split));
}

#[test]
fn an_interrupted_attempt_is_not_restarted_from_scratch() {
    // A backend outage is not evidence about the work (D-027), so it spends no restart. That is a
    // different operation from L3, which does spend one — but "no restart" is not "same attempt":
    // the process is gone and the next one starts a fresh conversation.
    assert_eq!(
        decide(Ended::Abandoned, room(3, 2)),
        Decision::Stop(Stop::Abandoned)
    );
}

#[test]
fn a_failed_attempt_with_budget_left_restarts_with_its_cause() {
    let cause = detector(Detector::ExactRepeat, 8);
    assert_eq!(
        decide(Ended::Failed(cause.clone()), room(1, 1)),
        Decision::Restart(cause)
    );
}

#[test]
fn a_spent_attempt_budget_stops_even_with_restarts_left() {
    // The two budgets are independent: `max_attempts_per_subtask` is the outer bound and no
    // number of restarts may go past it. This is the rule that keeps a weak model from splitting
    // or looping its way into infinite tries.
    assert_eq!(
        decide(Ended::Failed(Cause::Stalled), room(0, 5)),
        Decision::Stop(Stop::Exhausted(room(0, 5)))
    );
}

#[test]
fn a_spent_restart_budget_stops_even_with_attempts_left() {
    // The opposite direction, and the one a sabotage of the `||` would miss: an implementation
    // that checked only attempts would restart forever, because attempts and restarts are counted
    // by different callers.
    assert_eq!(
        decide(Ended::Failed(Cause::Stalled), room(5, 0)),
        Decision::Stop(Stop::Exhausted(room(5, 0)))
    );
}

#[test]
fn every_cause_has_a_distinct_wire_name() {
    // `attempt_restarted` records the cause as a string, and two causes sharing a name would make
    // "how often did D1 restart an attempt" unanswerable.
    let causes = [
        detector(Detector::ExactRepeat, 8),
        detector(Detector::SemanticRepeat, 3),
        detector(Detector::NoProgress, 25),
        detector(Detector::ReasoningLoop, 3),
        detector(Detector::Budget, 1),
        detector(Detector::FileChurn, 16),
        Cause::Limit(Limit::Steps),
        Cause::Limit(Limit::Tokens),
        Cause::Limit(Limit::Time),
        Cause::Stalled,
        Cause::UnknownTool {
            name: "no_such_tool".to_string(),
        },
        Cause::EmptyReply,
        Cause::Gate {
            reason: FailureReason::GateFailed,
            why: "the check failed".to_string(),
        },
    ];
    let mut names: Vec<&str> = causes.iter().map(Cause::as_str).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), causes.len(), "{names:?}");
}

#[test]
fn a_gate_failure_quotes_the_gate_verbatim() {
    // The next attempt has to fix what the gate named. A paraphrase here is how the model spends
    // a pass fixing something the gate did not complain about.
    let cause = Cause::Gate {
        reason: FailureReason::ReviewRejected,
        why: "`src/parser.rs` drops the trailing comma case".to_string(),
    };
    let text = strategy(&cause, "", &[]);
    assert!(
        text.contains("`src/parser.rs` drops the trailing comma case"),
        "{text}"
    );
    assert!(text.contains("gate"), "{text}");
}

#[test]
fn the_strategy_names_the_previous_strategy_and_forbids_it() {
    let cause = detector(Detector::NoProgress, 25);
    let text = strategy(&cause, "search the whole repository for the parser", &[]);
    assert!(
        text.contains("search the whole repository for the parser"),
        "{text}"
    );
    assert!(text.contains("Do not run it again"), "{text}");
    // The advice for this detector is about making an edit rather than reading: the restart is
    // useless if the strategy only says "try again".
    assert!(text.contains("edit"), "{text}");
}

#[test]
fn a_first_attempt_has_no_previous_strategy_to_quote() {
    let text = strategy(&Cause::Stalled, "", &[]);
    assert!(!text.contains("previous attempt ran under"), "{text}");
    assert!(text.contains("fresh attempt"), "{text}");
}

#[test]
fn rejected_ideas_are_quoted_and_capped_to_the_newest() {
    let rejected: Vec<String> = (0..MAX_REJECTED_IDEAS + 3)
        .map(|index| format!("idea {index}"))
        .collect();
    let text = strategy(&Cause::Stalled, "", &rejected);
    // The oldest ideas are the ones least likely to be proposed again, and the block has to
    // stay short enough to read.
    assert!(!text.contains("idea 0\n"), "{text}");
    assert!(!text.contains("- idea 0"), "{text}");
    assert!(text.contains(&format!("- idea {}", rejected.len() - 1)), "{text}");
    assert_eq!(text.matches("- idea ").count(), MAX_REJECTED_IDEAS, "{text}");
}

#[test]
fn a_quoted_strategy_is_one_line_and_bounded() {
    let long = "step one\nstep two\n".repeat(200);
    let text = strategy(&Cause::Stalled, &long, &[]);
    let quoted = text
        .lines()
        .find(|line| line.starts_with("> "))
        .expect("the quote");
    let quoted: String = quoted.chars().skip(2).collect();
    assert!(
        quoted.chars().count() <= MAX_QUOTE_CHARS + 1,
        "{}",
        quoted.chars().count()
    );
    assert!(quoted.ends_with('…'), "{quoted}");
    // A multi-line quote would break the brief's headings into a strategy of their own.
    assert_eq!(quoted.matches('\n').count(), 0);
}

#[test]
fn budget_causes_point_at_the_budget_that_ran_out() {
    // Steps, tokens and time ask for different corrections: a restart that told a token-bound
    // attempt to "take smaller steps" would change the wrong thing.
    let steps = strategy(&Cause::Limit(Limit::Steps), "", &[]);
    let tokens = strategy(&Cause::Limit(Limit::Tokens), "", &[]);
    let time = strategy(&Cause::Limit(Limit::Time), "", &[]);
    assert!(steps.contains("steps"), "{steps}");
    assert!(tokens.contains("tokens"), "{tokens}");
    assert!(time.contains("wall time"), "{time}");
    assert_ne!(steps, tokens);
    assert_ne!(tokens, time);
}

#[test]
fn every_cause_produces_advice_that_changes_the_method() {
    // The guard behind the whole rung: a restart whose brief says nothing new is a paid retry of
    // the same attempt. Each cause must produce advice, and no two causes may produce the same
    // advice — a copy-paste in the match is otherwise invisible.
    let causes = [
        detector(Detector::ExactRepeat, 8),
        detector(Detector::SemanticRepeat, 3),
        detector(Detector::NoProgress, 25),
        detector(Detector::ReasoningLoop, 3),
        detector(Detector::Budget, 1),
        Cause::Limit(Limit::Steps),
        Cause::Limit(Limit::Tokens),
        Cause::Limit(Limit::Time),
        Cause::Stalled,
        Cause::UnknownTool {
            name: "x".to_string(),
        },
        Cause::EmptyReply,
        Cause::Gate {
            reason: FailureReason::GateFailed,
            why: "y".to_string(),
        },
    ];
    let mut advice: Vec<&str> = causes.iter().map(Cause::advice).collect();
    for text in &advice {
        assert!(text.len() > 40, "advice is too thin to change anything: {text}");
    }
    advice.sort_unstable();
    advice.dedup();
    assert_eq!(advice.len(), causes.len(), "{advice:?}");
}

#[test]
fn a_stop_names_the_budget_that_ran_out() {
    // The escalation a human reads must say which budget stopped the subtask: "blocked" alone
    // sends them looking through config.
    let Decision::Stop(Stop::Exhausted(allowance)) = decide(Ended::Failed(Cause::Stalled), room(0, 2)) else {
        panic!("expected exhaustion");
    };
    assert_eq!(allowance.attempts_left, 0);
}

#[test]
fn a_rejected_claim_retries_in_place_rather_than_rolling_back() {
    // A red gate read the diff and named what is wrong with it. Rolling that diff away throws
    // away the closest thing to a finished subtask and pays for it twice.
    let cause = Cause::Gate {
        reason: FailureReason::ReviewRejected,
        why: "the error branch is untested".to_string(),
    };
    assert_eq!(
        decide(Ended::Failed(cause.clone()), room(1, 1)),
        Decision::Retry(cause)
    );
}

#[test]
fn a_looping_attempt_rolls_back_instead_of_retrying_in_place() {
    // The tree a detector stopped on is the tree that produced the loop; keeping it is how the
    // next attempt inherits the mess. This is the arm that must not collapse into `Retry`.
    let cause = detector(Detector::NoProgress, 25);
    assert!(matches!(
        decide(Ended::Failed(cause), room(1, 1)),
        Decision::Restart(_)
    ));
}

#[test]
fn a_retry_also_needs_budget_on_both_counts() {
    // `Retry` starts a new attempt, so it spends the same two budgets as `Restart`. A branch that
    // returned `Retry` before checking them would let a gate-failing model run forever.
    for allowance in [room(0, 2), room(2, 0)] {
        assert!(matches!(
            decide(
                Ended::Failed(Cause::Gate {
                    reason: FailureReason::GateFailed,
                    why: "x".to_string(),
                }),
                allowance
            ),
            Decision::Stop(Stop::Exhausted(_))
        ));
    }
}

// ── what the journal says a subtask already spent ────────────────────────────────────────

/// One record with an explicit scope, for the budget reader.
fn scoped(number: u32, kind: Kind, payload: serde_json::Value) -> JournalRecord {
    JournalRecord::new(
        crate::journal::RecordFields {
            seq: 0,
            at: at(),
            task: crate::domain::TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV"),
            subtask: Some(SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAA")),
            attempt: Some(number),
            kind,
        },
        crate::journal::JournalEvent::with_payload(payload),
        None,
    )
}

fn at() -> crate::domain::Timestamp {
    use chrono::TimeZone as _;
    chrono::Utc.with_ymd_and_hms(2026, 11, 12, 10, 14, 3).unwrap()
}

fn here() -> SubtaskId {
    SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAA")
}

#[test]
fn a_subtask_the_journal_never_mentioned_starts_at_one_with_the_whole_budget() {
    assert_eq!(Spent::read(&[], &here()), Spent::EMPTY);
    assert_eq!(Spent::EMPTY.next, 1);
}

#[test]
fn a_failed_attempt_is_charged_and_the_next_number_follows_it() {
    // The budget is about attempts, so a failed one has to be visible; and the number has to move
    // or the next attempt reuses a key the journal already holds.
    let records = vec![scoped(1, Kind::AttemptFailed, serde_json::json!({}))];
    let spent = Spent::read(&records, &here());
    assert_eq!(spent.attempts, 1);
    assert_eq!(spent.restarts, 0);
    assert_eq!(spent.next, 2);
    assert_eq!(
        spent.left(room(3, 2)),
        room(2, 2),
        "a failed attempt spends an attempt, not a restart"
    );
}

#[test]
fn a_ladder_restart_spends_both_budgets() {
    let records = vec![scoped(
        1,
        Kind::AttemptRestarted,
        serde_json::json!({ "cause": "stalled" }),
    )];
    let spent = Spent::read(&records, &here());
    assert_eq!(spent.attempts, 1);
    assert_eq!(spent.restarts, 1);
    assert_eq!(spent.left(room(3, 2)), room(2, 1));
}

#[test]
fn an_abandoned_attempt_spends_nothing_but_still_takes_its_number() {
    // D-027: a reboot is not evidence about the model's approach, so the budget is untouched.
    // The number still has to advance, or the resumed attempt would collide with the one the
    // crash left open.
    let records = vec![
        scoped(
            1,
            Kind::AttemptFailed,
            serde_json::json!({ "state": "abandoned", "ending": "backend" }),
        ),
        scoped(
            2,
            Kind::AttemptRestarted,
            serde_json::json!({ "cause": "recovery" }),
        ),
    ];
    let spent = Spent::read(&records, &here());
    assert_eq!(spent.attempts, 0, "an abandoned attempt is free");
    assert_eq!(spent.restarts, 0, "recovery is not an L3 restart");
    assert_eq!(spent.next, 3, "but the number still moved past both");
    assert_eq!(spent.left(room(3, 2)), room(3, 2));
}

#[test]
fn a_green_verdict_is_charged_but_a_failed_one_is_not() {
    let passed = vec![scoped(
        1,
        Kind::GateResult,
        serde_json::json!({ "gate": "verdict", "passed": true }),
    )];
    assert_eq!(Spent::read(&passed, &here()).attempts, 1);
    // A red verdict does not end an attempt; `attempt_failed` does, and that is charged. Counting
    // the verdict too would spend two attempts for one.
    let failed = vec![scoped(
        1,
        Kind::GateResult,
        serde_json::json!({ "gate": "verdict", "passed": false }),
    )];
    assert_eq!(Spent::read(&failed, &here()).attempts, 0);
}

#[test]
fn another_subtasks_records_do_not_count() {
    // A journal is shared by the whole task; a budget read without the filter would drain one
    // subtask with another's attempts.
    let records = vec![scoped(1, Kind::AttemptFailed, serde_json::json!({}))];
    let elsewhere = SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAB");
    assert_eq!(Spent::read(&records, &elsewhere), Spent::EMPTY);
}

#[test]
fn a_journal_that_recorded_more_than_the_config_allows_stops_the_subtask() {
    // A human can lower `max_attempts_per_subtask` under a running task. Subtracting would wrap
    // to `u32::MAX` and hand the subtask an effectively infinite budget.
    let records: Vec<JournalRecord> = (1..=5)
        .map(|number| scoped(number, Kind::AttemptFailed, serde_json::json!({})))
        .collect();
    let spent = Spent::read(&records, &here());
    assert_eq!(
        spent.left(room(2, 1)),
        room(0, 1),
        "attempts clamp at zero; restarts are untouched, since none of these were L3"
    );
}
