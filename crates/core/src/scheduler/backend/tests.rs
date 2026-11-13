//! What to do while the backend is unreachable.

use super::*;
use crate::journal::{JournalRecord, Kind, RecordFields};
use chrono::TimeZone as _;

fn at(secs: i64) -> Timestamp {
    chrono::Utc
        .timestamp_opt(1_700_000_000 + secs, 0)
        .single()
        .expect("a fixed instant")
}

fn task() -> crate::ids::TaskId {
    crate::ids::TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV")
}

fn record(secs: i64, kind: Kind) -> JournalRecord {
    JournalRecord::new(
        RecordFields::new(1, at(secs), task(), kind),
        crate::journal::JournalEvent::with_payload(serde_json::json!({})),
        None,
    )
}

#[test]
fn a_healthy_backend_needs_no_decision() {
    assert_eq!(while_unavailable(None, at(0), 900), While::Available);
}

#[test]
fn the_threshold_is_continuous_and_not_a_count_of_failures() {
    // Fourteen minutes in, a human is not told. At fifteen, they are. The interesting case is the
    // boundary itself: `>=` and not `>`, because the configuration's number is the promise.
    assert_eq!(while_unavailable(Some(at(0)), at(899), 900), While::Wait);
    assert_eq!(while_unavailable(Some(at(0)), at(899 + 1), 900), While::Escalate);
    assert_eq!(while_unavailable(Some(at(0)), at(3600), 900), While::Escalate);
}

#[test]
fn a_recovery_resets_the_interval() {
    // The rule the caller implements by clearing `since`: a backend that answered at minute 14 was
    // not down for fifteen continuous minutes at minute 15. Stated here as arithmetic so the test
    // fails if the comparison ever becomes cumulative.
    assert_eq!(while_unavailable(Some(at(0)), at(60 * 14), 900), While::Wait);
    let second_outage = at(60 * 14);
    assert_eq!(
        while_unavailable(Some(second_outage), at(60 * 15), 900),
        While::Wait,
        "the interval must be measured from the latest failure, not the first"
    );
}

#[test]
fn the_question_offers_nothing_to_decide() {
    // A menu with "retry"/"abort" would be a question whose only honest answer is "wait", and the
    // one option present changes no permission — acknowledging is not allowing.
    let entry = escalation(task(), at(0), at(1200), 1200);
    assert_eq!(entry.kind, EscalationKind::BackendUnavailable);
    assert_eq!(entry.options, vec![Answer::AllowOnce]);
    assert!(!entry.blocking, "an outage must not stop the task");
    assert!(entry.question.contains("20 minutes"), "{}", entry.question);
    assert_eq!(entry.key, KEY);
}

#[test]
fn the_outage_since_is_read_back_from_the_journal() {
    // After `kill -9` the clock must not restart. The journal is the only record that survives, and
    // the last `backend_unavailable` with no `backend_restored` after it is the answer.
    let records = vec![
        record(0, Kind::AttemptStarted),
        record(100, Kind::BackendUnavailable),
        record(200, Kind::BackendRestored),
        record(300, Kind::BackendUnavailable),
    ];
    assert_eq!(since_from_journal(&records), Some(at(300)));
}

#[test]
fn a_restored_backend_clears_the_interval() {
    let records = vec![
        record(100, Kind::BackendUnavailable),
        record(200, Kind::BackendRestored),
    ];
    assert_eq!(since_from_journal(&records), None);
}

#[test]
fn a_journal_with_no_outage_record_has_no_interval() {
    let records = vec![record(0, Kind::AttemptStarted), record(1, Kind::TaskState)];
    assert_eq!(since_from_journal(&records), None);
    assert_eq!(since_from_journal(&[]), None);
}
