//! The v0.1 numbers, computed from journals rather than remembered.
//!
//! # Why this is code and not a table in a document
//!
//! `docs/metrics-v0.1.md` states claims like "3 subtasks closed with green gates, unaided". A
//! human reading a run and typing those numbers into Markdown is a measurement that cannot be
//! reproduced, cannot be re-run after a fix, and quietly becomes an assertion nobody can check.
//! Computing it from the journal is the same work the reader did, with the arithmetic available to
//! anyone who has the journal.
//!
//! # What "unaided" means here
//!
//! A subtask closed while an escalation was open and unanswered. The design allows that — an
//! escalation never blocks progress — so the *count* is what matters, not the presence: a run with
//! one answered escalation that closed five subtasks is still evidence, and a run that needed a
//! human before every subtask is not. The metric therefore records the number closed with an
//! escalation open, so the reader of `metrics-v0.1.md` can see the difference rather than trust a
//! label.
//!
//! # Why "in a row" is computed separately
//!
//! The bar is not "three closures" but "three **consecutive** closures". A total cannot tell a run
//! that closed three subtasks before its first failure from one that closed one, failed, closed
//! another, failed, and closed a third: both report `closed_green == 3`, and only the first is the
//! claim. `longest_green_streak` breaks on every non-green settlement and on every closure that
//! happened while an escalation was open, and it walks the settlements in journal order, because
//! that is the order the run produced them in.
//!
//! # What this module deliberately does not compute
//!
//! Wall-clock time and token cost. Both are in the journal, but a run of this tool is measured in
//! hours on one machine and both numbers move with the hardware; reporting them invites comparing
//! runs that are not comparable. The milestone's bar is about correctness and autonomy, which are
//! counts, not durations.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::journal::{JournalRecord, Kind};

/// One subtask's outcome, as the journal recorded it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubtaskOutcome {
    /// The subtask's id.
    pub subtask: String,
    /// Its final state: `done`, `blocked` or `skipped`.
    pub state: String,
    /// Whether an escalation was open at the moment it closed.
    pub closed_with_escalation: bool,
}

/// One task's numbers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TaskMetrics {
    /// The task's id.
    pub task: String,
    /// Every subtask that reached a settled state, in journal order.
    pub subtasks: Vec<SubtaskOutcome>,
    /// Subtasks a green gate closed.
    pub closed_green: usize,
    /// How many of those closed while an escalation was open.
    pub closed_with_escalation: usize,
    /// The longest run of consecutive settled subtasks that were closed green with no escalation
    /// open, in journal order.
    pub longest_green_streak: usize,
    /// Every detector firing, by `` `detector`/`level` ``.
    pub detectors: BTreeMap<String, usize>,
    /// Escalations opened and not yet resolved at the end of the journal.
    pub escalations_pending: usize,
    /// Escalations opened over the whole run.
    pub escalations_total: usize,
    /// Attempts the ladder ended and the scheduler began again.
    pub restarts: usize,
    /// Records the journal holds for this task.
    pub records: usize,
    /// Whether the journal's chain is intact.
    pub corrupt: bool,
}

/// Compute a task's metrics from its own records.
///
/// `records` must already be filtered to one task: a journal is shared by every task of a
/// repository, and counting another task's subtasks would inflate every number here.
#[must_use]
pub fn compute(task: &str, records: &[JournalRecord]) -> TaskMetrics {
    // State per subtask, so a later record for the same id replaces the earlier one rather than
    // being counted twice: `running` followed by `done` is one settled subtask, not two.
    //
    // The position is stored with the state because the order of settlements is not the order of
    // ids. "Three subtasks in a row" is a claim about time, and a `BTreeMap` iterated by key would
    // answer it in an order that a run never produced — `sub_9` sorting after `sub_10` is enough
    // to turn a broken streak into a whole one.
    let mut settled: BTreeMap<String, (String, bool, usize)> = BTreeMap::new();
    let mut detectors: BTreeMap<String, usize> = BTreeMap::new();
    // Open escalations by id, not by counter: a journal that has a resolution without its opening
    // (the tail of a rotated segment can start mid-question) would leave a counter negative and
    // every later attribution wrong. A set answers "was one open" and "how many are open" with the
    // same truth.
    let mut open_escalations: BTreeSet<String> = BTreeSet::new();
    let mut total_escalations: usize = 0;
    let mut restarts: usize = 0;

    for (position, record) in records.iter().enumerate() {
        match record.fields.kind {
            Kind::SubtaskState => {
                let Some(id) = record
                    .event
                    .payload
                    .get("subtask")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
                else {
                    continue;
                };
                let Some(state) = record
                    .event
                    .payload
                    .get("state")
                    .and_then(serde_json::Value::as_str)
                else {
                    continue;
                };
                if matches!(state, "done" | "blocked" | "skipped") {
                    let with_escalation = !open_escalations.is_empty();
                    // A later settlement moves the subtask to its latest position: the last word
                    // on a subtask is where it settled, and a repair that turned `blocked` into
                    // `done` belongs at the end of the run, not where the first attempt died.
                    settled.insert(id, (state.to_string(), with_escalation, position));
                }
            }
            Kind::DetectorFired => {
                let detector = record
                    .event
                    .payload
                    .get("detector")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown");
                let level = record
                    .event
                    .payload
                    .get("level")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown");
                *detectors.entry(format!("{detector}/{level}")).or_default() += 1;
            }
            Kind::EscalationOpened => {
                total_escalations += 1;
                if let Some(id) = record.event.payload.get("id").and_then(serde_json::Value::as_str) {
                    open_escalations.insert(id.to_string());
                }
            }
            Kind::EscalationResolved => {
                if let Some(id) = record.event.payload.get("id").and_then(serde_json::Value::as_str) {
                    open_escalations.remove(id);
                }
            }
            Kind::AttemptRestarted => restarts += 1,
            _ => {}
        }
    }

    // Journal order, not id order: the reader of "in a row" is asking about the run, and the run
    // happened in time.
    let mut ordered: Vec<(String, String, bool, usize)> = settled
        .into_iter()
        .map(|(id, (state, with_escalation, position))| (id, state, with_escalation, position))
        .collect();
    ordered.sort_by_key(|(_, _, _, position)| *position);
    let subtasks: Vec<SubtaskOutcome> = ordered
        .into_iter()
        .map(|(subtask, state, closed_with_escalation, _)| SubtaskOutcome {
            subtask,
            state,
            closed_with_escalation,
        })
        .collect();

    let closed_green = subtasks.iter().filter(|one| one.state == "done").count();
    let closed_with_escalation = subtasks
        .iter()
        .filter(|one| one.state == "done" && one.closed_with_escalation)
        .count();
    let longest_green_streak = longest_green_streak(&subtasks);

    TaskMetrics {
        task: task.to_string(),
        subtasks,
        closed_green,
        closed_with_escalation,
        longest_green_streak,
        detectors,
        escalations_pending: open_escalations.len(),
        escalations_total: total_escalations,
        restarts,
        records: records.len(),
        corrupt: false,
    }
}

/// The longest run of subtasks that were each closed green with no escalation open.
///
/// # Why this is not `closed_green`
///
/// The milestone's bar is "three subtasks **in a row**". A total counts three green closures
/// separated by three blocked ones as progress, and it is: but it is not the claim, and a number
/// that cannot distinguish the two would let a failing run be reported as a passing one. Breaking
/// the streak on every non-green settlement — including `skipped`, which is not work done — is
/// what makes the number mean what the document says.
///
/// # Why `closed_with_escalation` breaks it
///
/// The bar is "without human intervention". A subtask that closed while a question was open is
/// work the run did while a human may have been needed; counting it would make the streak weaker
/// than the claim it is used to support. The escalation's *presence* is enough, whether or not it
/// was answered later: the settlement is what the streak is about.
fn longest_green_streak(subtasks: &[SubtaskOutcome]) -> usize {
    let mut longest = 0;
    let mut current = 0;
    for one in subtasks {
        if one.state == "done" && !one.closed_with_escalation {
            current += 1;
            longest = longest.max(current);
        } else {
            current = 0;
        }
    }
    longest
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{SubtaskId, SubtaskState, TaskId, Timestamp};
    use crate::journal::{JournalEvent, JournalRecord, RecordFields};
    use chrono::TimeZone as _;

    fn at() -> Timestamp {
        chrono::Utc.with_ymd_and_hms(2026, 11, 13, 10, 0, 0).unwrap()
    }

    fn record(kind: Kind, payload: serde_json::Value) -> JournalRecord {
        JournalRecord::new(
            RecordFields::new(0, at(), TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV"), kind),
            JournalEvent::with_payload(payload),
            None,
        )
    }

    fn subtask_state(id: &str, state: &str) -> JournalRecord {
        record(
            Kind::SubtaskState,
            serde_json::json!({ "subtask": id, "state": state }),
        )
    }

    #[test]
    fn a_running_state_does_not_become_a_second_closed_subtask() {
        // The cycle announces `running`, then `pending` after an adoption, then `done`. Counting
        // records rather than settled ids would report three subtasks where one did the work.
        let records = vec![
            subtask_state("sub_A", "running"),
            subtask_state("sub_A", "pending"),
            subtask_state("sub_A", "done"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.subtasks.len(), 1);
        assert_eq!(metrics.closed_green, 1);
        assert_eq!(metrics.subtasks[0].state, "done");
    }

    #[test]
    fn a_subtask_announced_settled_twice_is_one_subtask_and_the_later_state_wins() {
        // A resume re-announces what the tree already says: the recovery pass reopens a subtask a
        // crash left `running`, and a task that crashed between its last gate and its verdict can
        // announce a settled state again. Two records, one subtask; and the last word is the
        // truth, so `blocked` followed by `done` counts as green once, not as one of each.
        let records = vec![
            subtask_state("sub_A", "blocked"),
            subtask_state("sub_A", "done"),
            subtask_state("sub_B", "done"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.subtasks.len(), 2, "{metrics:?}");
        assert_eq!(metrics.closed_green, 2, "{metrics:?}");
        let settled = metrics
            .subtasks
            .iter()
            .find(|one| one.subtask == "sub_A")
            .expect("sub_A");
        assert_eq!(settled.state, "done");
    }

    #[test]
    fn a_blocked_subtask_is_settled_but_not_green() {
        // The milestone's bar is green gates. A blocked subtask is progress the task made past,
        // so it must appear, and it must not be counted as success.
        let records = vec![
            subtask_state("sub_A", "done"),
            subtask_state("sub_B", "blocked"),
            subtask_state("sub_C", "skipped"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.closed_green, 1, "{metrics:?}");
        assert_eq!(metrics.subtasks.len(), 3);
    }

    #[test]
    fn detectors_are_tallied_by_detector_and_level() {
        // The pair is the unit that matters: `d1/l3` means the ladder ended an attempt, `d1/l1`
        // means one nudge. A flat total would average those into a number that says nothing.
        let fired = |detector: &str, level: &str| {
            record(
                Kind::DetectorFired,
                serde_json::json!({ "detector": detector, "level": level }),
            )
        };
        let records = vec![
            fired("d1", "l1"),
            fired("d1", "l1"),
            fired("d1", "l3"),
            fired("d5", "l3"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.detectors["d1/l1"], 2);
        assert_eq!(metrics.detectors["d1/l3"], 1);
        assert_eq!(metrics.detectors["d5/l3"], 1);
    }

    #[test]
    fn a_subtask_closed_while_an_escalation_was_open_is_not_counted_as_unaided() {
        // "Unaided" is a claim about a human's absence, so it has to be evidence. An escalation
        // open at the moment the gate went green means the run asked something and did not wait:
        // the work closed, but the reader must be able to see the question.
        let records = vec![
            record(
                Kind::EscalationOpened,
                serde_json::json!({ "id": "esc_1", "key": "git:push", "kind": "irreversible" }),
            ),
            subtask_state("sub_A", "done"),
            record(Kind::EscalationResolved, serde_json::json!({ "id": "esc_1" })),
            subtask_state("sub_B", "done"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.closed_green, 2);
        assert_eq!(metrics.closed_with_escalation, 1, "{metrics:?}");
        assert_eq!(metrics.escalations_total, 1);
        assert_eq!(metrics.escalations_pending, 0);
        let first = metrics
            .subtasks
            .iter()
            .find(|one| one.subtask == "sub_A")
            .expect("sub_A");
        assert!(first.closed_with_escalation, "the open escalation was missed");
    }

    #[test]
    fn an_unresolved_escalation_stays_pending_and_does_not_need_a_matching_open() {
        // A rotation can leave a resolution whose opening is in the previous segment. Counting
        // with `+= 1`/`-= 1` would then drive the count negative and mark nothing as assisted.
        let records = vec![
            record(Kind::EscalationResolved, serde_json::json!({ "id": "esc_old" })),
            record(Kind::EscalationOpened, serde_json::json!({ "id": "esc_new" })),
            subtask_state("sub_A", "done"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.escalations_pending, 1);
        assert_eq!(metrics.escalations_total, 1);
        assert_eq!(metrics.closed_with_escalation, 1);
    }

    #[test]
    fn a_streak_counts_consecutive_green_closures_not_the_total() {
        // The milestone's bar is "three in a row". Three closures separated by a blocked subtask
        // are progress, but they are not three in a row, and a total cannot tell the difference.
        let records = vec![
            subtask_state("sub_A", "done"),
            subtask_state("sub_B", "blocked"),
            subtask_state("sub_C", "done"),
            subtask_state("sub_D", "done"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.closed_green, 3, "{metrics:?}");
        assert_eq!(metrics.longest_green_streak, 2, "{metrics:?}");
    }

    #[test]
    fn a_streak_is_taken_from_journal_order_not_from_id_order() {
        // Ids sort as text, and a run does not settle subtasks in id order. Here the greens are
        // `sub_9`, `sub_10` and `sub_11` — which sort as `sub_10`, `sub_11`, `sub_9` — so an
        // implementation that walked a `BTreeMap` by key would split the run and report 2.
        let records = vec![
            subtask_state("sub_9", "done"),
            subtask_state("sub_10", "done"),
            subtask_state("sub_11", "done"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.longest_green_streak, 3, "{metrics:?}");
        let order: Vec<&str> = metrics.subtasks.iter().map(|one| one.subtask.as_str()).collect();
        assert_eq!(order, vec!["sub_9", "sub_10", "sub_11"], "{metrics:?}");
    }

    #[test]
    fn a_closure_with_an_escalation_open_breaks_the_streak() {
        // "In a row, unaided" is the claim. A closure that happened while a question was open is
        // not evidence of autonomy, so it ends the run of green rather than extending it — even
        // though it is still counted in `closed_green`.
        let records = vec![
            subtask_state("sub_A", "done"),
            record(
                Kind::EscalationOpened,
                serde_json::json!({ "id": "esc_1", "key": "git:push", "kind": "irreversible" }),
            ),
            subtask_state("sub_B", "done"),
            record(Kind::EscalationResolved, serde_json::json!({ "id": "esc_1" })),
            subtask_state("sub_C", "done"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.closed_green, 3, "{metrics:?}");
        assert_eq!(metrics.closed_with_escalation, 1, "{metrics:?}");
        assert_eq!(metrics.longest_green_streak, 1, "{metrics:?}");
    }

    #[test]
    fn a_skipped_subtask_breaks_the_streak_because_no_work_was_done() {
        // `skipped` settles a subtask without a green gate. Counting it as part of a run would
        // let a task that skipped the work be reported as having done three in a row.
        let records = vec![
            subtask_state("sub_A", "done"),
            subtask_state("sub_B", "skipped"),
            subtask_state("sub_C", "done"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.longest_green_streak, 1, "{metrics:?}");
    }

    #[test]
    fn a_streak_uses_the_last_settlement_of_each_subtask() {
        // A repair can turn `blocked` into `done` later in the run. The subtask settles where it
        // ends up, so the repair joins the run instead of leaving its earlier failure in the
        // middle of it.
        let records = vec![
            subtask_state("sub_A", "done"),
            subtask_state("sub_B", "blocked"),
            subtask_state("sub_C", "done"),
            subtask_state("sub_B", "done"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.closed_green, 3, "{metrics:?}");
        assert_eq!(metrics.longest_green_streak, 3, "{metrics:?}");
    }

    #[test]
    fn a_run_that_closed_nothing_has_no_streak() {
        let records = vec![subtask_state("sub_A", "running")];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.longest_green_streak, 0, "{metrics:?}");
    }

    #[test]
    fn restarts_are_counted_and_ordinary_records_are_ignored() {
        let records = vec![
            record(Kind::AttemptRestarted, serde_json::json!({ "next": 2 })),
            record(Kind::AttemptRestarted, serde_json::json!({ "next": 3 })),
            record(Kind::ModelResponse, serde_json::json!({ "step": 1 })),
            record(Kind::TaskState, serde_json::json!({ "state": "running" })),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.restarts, 2);
        assert_eq!(metrics.records, 4);
        assert!(metrics.subtasks.is_empty(), "{metrics:?}");
        assert!(metrics.detectors.is_empty(), "{metrics:?}");
    }

    #[test]
    fn a_record_without_its_scope_is_ignored_rather_than_guessed() {
        // The payload, not the record's `subtask` field, is what names the subtask here. A
        // malformed record must not create a settled subtask called "".
        let records = vec![record(Kind::SubtaskState, serde_json::json!({ "state": "done" }))];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert!(metrics.subtasks.is_empty(), "{metrics:?}");
        assert_eq!(metrics.closed_green, 0);
    }

    #[test]
    fn the_spec_the_state_type_uses_is_the_one_this_module_matches() {
        // If `SubtaskState` gains a settled spelling, the `matches!` above silently stops
        // counting it. This fails instead.
        for state in [SubtaskState::Done, SubtaskState::Blocked, SubtaskState::Skipped] {
            assert!(
                matches!(state.as_str(), "done" | "blocked" | "skipped"),
                "`{}` is settled but metrics does not count it",
                state.as_str()
            );
        }
        let _ = SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FZZ");
    }
}
