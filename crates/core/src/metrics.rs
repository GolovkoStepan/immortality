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

use crate::ids::SubtaskId;
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
    /// Attempts whose gates returned `passed: true`.
    ///
    /// Not the same set as [`Self::closed_green`]: a subtask may take two attempts, and only one
    /// of them ends green; a green attempt may also belong to a subtask that closed `done` on a
    /// later attempt after this one was rolled back. The two numbers differ, and the difference is
    /// where work gets lost.
    pub green_attempts: usize,
    /// Green attempts whose subtask did not end as `done`.
    ///
    /// The `0.2` exit criterion (D-114) is this number being **zero** together with
    /// [`Self::step_exhaustions_without_gates`] being zero and
    /// [`Self::settled_after_step_budget`] being non-zero: no work a gate certified went
    /// uncounted, no steps were spent without asking, and the path the milestone added actually
    /// ran.
    ///
    /// # What a non-zero value does and does not mean
    ///
    /// It means the pair `(green verdict, no closure)` exists in the journal, which is the thing
    /// the milestone set out to eliminate. It does not by itself say the work was thrown away: a
    /// subtask closed `done` and then reopened — by a retry, or by an operator — lands here too.
    /// That is why the same struct carries the per-subtask list: the count says "look", the list
    /// says "here".
    pub green_attempts_not_closed: usize,
    /// Attempts closed because the gates went green when the step budget ran out (D-115).
    ///
    /// Counted from the `settled_by: "gates_after_step_budget"` marker on the attempt record, which
    /// exists precisely so this path is distinguishable from a claim in a journal read months
    /// later. Zero here on a journal that predates D-115 is expected, not a good sign.
    pub settled_after_step_budget: usize,
    /// Attempts the step budget ended where no gate verdict was ever written.
    ///
    /// # Why this number exists
    ///
    /// It is the blind spot that made the first reading of D-114 vacuous. Before D-115 the gates
    /// were consulted only from `settle_claim`, so an attempt that found the fix and then ran out
    /// of steps wrote `attempt_failed { ending: "steps" }` and no `gate_result` at all: the
    /// criterion "no attempt where the gate passed and the subtask did not close" held on the very
    /// run that motivated the milestone.
    ///
    /// A non-zero value marks a journal written before D-115. Its green-attempt counts cannot be
    /// read as evidence about lost work, because the gates simply were not asked on this path —
    /// "the gates said no" and "the gates were never consulted" are the two things this separates.
    pub step_exhaustions_without_gates: usize,
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
    let scan = Scan::of(records);
    let subtasks = scan.settled_subtasks();

    let closed_green = subtasks.iter().filter(|one| one.state == "done").count();
    let closed_with_escalation = subtasks
        .iter()
        .filter(|one| one.state == "done" && one.closed_with_escalation)
        .count();

    TaskMetrics {
        task: task.to_string(),
        green_attempts: scan.green.len(),
        green_attempts_not_closed: scan.lost_green_work(&subtasks),
        step_exhaustions_without_gates: scan.step_exhaustions_without_gates(),
        settled_after_step_budget: scan.settled_after_step_budget,
        closed_with_escalation,
        longest_green_streak: longest_green_streak(&subtasks),
        subtasks,
        closed_green,
        detectors: scan.detectors,
        escalations_pending: scan.open_escalations.len(),
        escalations_total: scan.total_escalations,
        restarts: scan.restarts,
        records: records.len(),
        corrupt: false,
    }
}

/// One pass over a task's journal, kept apart from [`compute`] so each counter has room to say
/// what it is for.
///
/// The fields are private: the numbers a reader sees are computed from them in [`compute`], and a
/// counter that leaks its raw set would invite a caller to re-derive one of the totals differently.
struct Scan {
    /// Every settled subtask state, keyed by id, with the position of the record that said it.
    ///
    /// The position is stored because the order of settlements is not the order of ids. "Three
    /// subtasks in a row" is a claim about time, and a `BTreeMap` iterated by key would answer it in
    /// an order that a run never produced — `sub_9` sorting after `sub_10` is enough to turn a
    /// broken streak into a whole one.
    settled: BTreeMap<String, (String, bool, usize)>,
    detectors: BTreeMap<String, usize>,
    /// Open escalations by id, not by counter: a journal that has a resolution without its opening
    /// (the tail of a rotated segment can start mid-question) would leave a counter negative and
    /// every later attribution wrong. A set answers "was one open" and "how many are open" with the
    /// same truth.
    open_escalations: BTreeSet<String>,
    total_escalations: usize,
    restarts: usize,
    /// Attempts whose `verdict` gate returned `passed: true`, and every attempt with *some*
    /// verdict. Two sets rather than one plus a filter, because the two questions the `0.2`
    /// criterion asks are "did a gate certify this attempt" and "was this attempt ever judged at
    /// all" — and a red verdict must not read as "never consulted".
    green: BTreeSet<(SubtaskId, u32)>,
    judged: BTreeSet<(SubtaskId, u32)>,
    step_exhaustions: Vec<(SubtaskId, u32)>,
    settled_after_step_budget: usize,
}

impl Scan {
    fn of(records: &[JournalRecord]) -> Self {
        let mut scan = Self {
            settled: BTreeMap::new(),
            detectors: BTreeMap::new(),
            open_escalations: BTreeSet::new(),
            total_escalations: 0,
            restarts: 0,
            green: BTreeSet::new(),
            judged: BTreeSet::new(),
            step_exhaustions: Vec::new(),
            settled_after_step_budget: 0,
        };
        for (position, record) in records.iter().enumerate() {
            scan.read(record, position);
        }
        scan
    }

    fn read(&mut self, record: &JournalRecord, position: usize) {
        match record.fields.kind {
            Kind::SubtaskState => self.read_settlement(record, position),
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
                *self.detectors.entry(format!("{detector}/{level}")).or_default() += 1;
            }
            Kind::EscalationOpened => {
                self.total_escalations += 1;
                if let Some(id) = record.event.payload.get("id").and_then(serde_json::Value::as_str) {
                    self.open_escalations.insert(id.to_string());
                }
            }
            Kind::EscalationResolved => {
                if let Some(id) = record.event.payload.get("id").and_then(serde_json::Value::as_str) {
                    self.open_escalations.remove(id);
                }
            }
            Kind::AttemptRestarted => self.restarts += 1,
            Kind::GateResult => self.read_gate(record),
            Kind::AttemptFailed => self.read_attempt_failed(record),
            _ => {}
        }
    }

    fn read_settlement(&mut self, record: &JournalRecord, position: usize) {
        let Some(id) = record
            .event
            .payload
            .get("subtask")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
        else {
            return;
        };
        let Some(state) = record
            .event
            .payload
            .get("state")
            .and_then(serde_json::Value::as_str)
        else {
            return;
        };
        if !matches!(state, "done" | "blocked" | "skipped") {
            return;
        }
        let with_escalation = !self.open_escalations.is_empty();
        // A later settlement moves the subtask to its latest position: the last word on a subtask
        // is where it settled, and a repair that turned `blocked` into `done` belongs at the end of
        // the run, not where the first attempt died.
        self.settled
            .insert(id, (state.to_string(), with_escalation, position));
    }

    fn read_gate(&mut self, record: &JournalRecord) {
        // `passed: true` on the `verdict` gate is the only gate outcome that licenses a closure; a
        // green individual check under a red verdict is still work the system rejected, and reading
        // any green record as certification would report a pass for every run whose first required
        // check happened to succeed.
        let verdict_gate = record
            .event
            .payload
            .get("gate")
            .and_then(serde_json::Value::as_str)
            == Some("verdict");
        if !verdict_gate {
            return;
        }
        // The attempt's identity comes from the record's own scope, which is what every other
        // reader of this journal keys on.
        let (Some(subtask), Some(attempt)) = (record.fields.subtask.clone(), record.fields.attempt) else {
            return;
        };
        self.judged.insert((subtask.clone(), attempt));
        if record
            .event
            .payload
            .get("passed")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        {
            self.green.insert((subtask, attempt));
        }
    }

    fn read_attempt_failed(&mut self, record: &JournalRecord) {
        // The attempt number is in the scope, not in the payload: the payload carries the attempt
        // *id* (`att_…`), which no other record uses for the same purpose.
        let (Some(subtask), Some(attempt)) = (record.fields.subtask.clone(), record.fields.attempt) else {
            return;
        };
        let payload = &record.event.payload;
        if payload.get("settled_by").and_then(serde_json::Value::as_str) == Some("gates_after_step_budget") {
            self.settled_after_step_budget += 1;
        }
        if payload.get("ending").and_then(serde_json::Value::as_str) == Some("steps") {
            self.step_exhaustions.push((subtask, attempt));
        }
    }

    /// Step exhaustions the gates were never asked about.
    ///
    /// Resolved after the whole journal is read rather than inline, because a verdict and the
    /// attempt record it belongs to may arrive in either order in a torn or hand-edited journal,
    /// and asking the finished set is easier to trust than tracking which path wrote what first.
    ///
    /// This is the number that made the first reading of the `0.2` criterion vacuous: before the
    /// gates were consulted on a spent step budget, an attempt that found the fix wrote
    /// `attempt_failed { ending: "steps" }` and no `gate_result` at all.
    fn step_exhaustions_without_gates(&self) -> usize {
        self.step_exhaustions
            .iter()
            .filter(|key| !self.judged.contains(*key))
            .count()
    }

    /// Green attempts whose subtask did not end as `done`.
    ///
    /// Includes the subtask that never settled at all: a gate certified work the run has no
    /// closure for, which is the defect itself rather than a missing record.
    ///
    /// On a journal from a *live* run this can be non-zero for an honest reason — the gates went
    /// green and the process has not yet written the closure — so the criterion is read on a
    /// finished task and the number is not treated as a live health signal.
    fn lost_green_work(&self, subtasks: &[SubtaskOutcome]) -> usize {
        let final_state: BTreeMap<&str, &str> = subtasks
            .iter()
            .map(|one| (one.subtask.as_str(), one.state.as_str()))
            .collect();
        self.green
            .iter()
            .filter(|(subtask, _)| final_state.get(subtask.as_str()).copied() != Some("done"))
            .count()
    }

    /// The settled subtasks in journal order.
    ///
    /// Journal order, not id order: the reader of "in a row" is asking about the run, and the run
    /// happened in time.
    fn settled_subtasks(&self) -> Vec<SubtaskOutcome> {
        let mut ordered: Vec<(String, String, bool, usize)> = self
            .settled
            .iter()
            .map(|(id, (state, with_escalation, position))| {
                (id.clone(), state.clone(), *with_escalation, *position)
            })
            .collect();
        ordered.sort_by_key(|(_, _, _, position)| *position);
        ordered
            .into_iter()
            .map(|(subtask, state, closed_with_escalation, _)| SubtaskOutcome {
                subtask,
                state,
                closed_with_escalation,
            })
            .collect()
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

    /// A record scoped to one attempt, the way the runner writes `gate_result` and
    /// `attempt_failed`. The scope is the only place the attempt number lives for these kinds.
    fn scoped_attempt(kind: Kind, subtask: &str, attempt: u32, payload: serde_json::Value) -> JournalRecord {
        let mut fields = RecordFields::new(0, at(), TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV"), kind);
        fields.subtask = Some(SubtaskId::from_suffix(
            subtask.strip_prefix("sub_").unwrap_or(subtask),
        ));
        fields.attempt = Some(attempt);
        JournalRecord::new(fields, JournalEvent::with_payload(payload), None)
    }

    fn verdict(subtask: &str, attempt: u32, passed: bool) -> JournalRecord {
        scoped_attempt(
            Kind::GateResult,
            subtask,
            attempt,
            serde_json::json!({ "gate": "verdict", "passed": passed, "reason": null, "why": null }),
        )
    }

    fn attempt_failed(subtask: &str, attempt: u32, payload: serde_json::Value) -> JournalRecord {
        scoped_attempt(Kind::AttemptFailed, subtask, attempt, payload)
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
    fn a_green_attempt_whose_subtask_closed_is_not_counted_as_lost() {
        // The ordinary path: claim, gates agree, subtask closes. Every green attempt in the journal
        // belongs to a subtask that ended `done`, so the 0.2 criterion sees nothing to report.
        let records = vec![
            verdict("sub_A", 1, true),
            subtask_state("sub_A", "done"),
            verdict("sub_B", 1, true),
            subtask_state("sub_B", "done"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.green_attempts, 2, "{metrics:?}");
        assert_eq!(metrics.green_attempts_not_closed, 0, "{metrics:?}");
        assert_eq!(metrics.step_exhaustions_without_gates, 0, "{metrics:?}");
    }

    #[test]
    fn a_green_attempt_for_a_subtask_that_never_closed_is_counted_as_lost() {
        // The defect 0.2 exists to eliminate, in its post-D-115 shape: the gates went green and the
        // subtask then closed `blocked` (or not at all). A count of zero would say the run kept
        // what the gate certified.
        let records = vec![verdict("sub_A", 1, true), subtask_state("sub_A", "blocked")];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.green_attempts, 1, "{metrics:?}");
        assert_eq!(metrics.green_attempts_not_closed, 1, "{metrics:?}");
    }

    #[test]
    fn a_green_attempt_with_no_settlement_at_all_is_counted_as_lost() {
        // Not merely "closed badly": the run ended before writing any state. The gate certified
        // work the journal has no closure for, and that is exactly what a reader must see.
        let records = vec![verdict("sub_A", 1, true)];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.green_attempts_not_closed, 1, "{metrics:?}");
    }

    #[test]
    fn a_red_verdict_is_not_a_green_attempt_and_is_not_lost_work() {
        // A rejected claim is the system working. Counting it would make every honest failure look
        // like a defect and drown the number that matters.
        let records = vec![verdict("sub_A", 1, false), subtask_state("sub_A", "done")];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.green_attempts, 0, "{metrics:?}");
        assert_eq!(metrics.green_attempts_not_closed, 0, "{metrics:?}");
    }

    #[test]
    fn only_the_verdict_gate_counts_as_green_not_an_individual_check() {
        // `DoD` writes one record per check plus a verdict. A green check under a red verdict is
        // work the system rejected, and reading any green record as certification would report a
        // pass for every run whose first required check happened to succeed.
        let records = vec![
            scoped_attempt(
                Kind::GateResult,
                "sub_A",
                1,
                serde_json::json!({ "gate": "check", "id": "ok", "passed": true }),
            ),
            verdict("sub_A", 1, false),
            subtask_state("sub_A", "blocked"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.green_attempts, 0, "{metrics:?}");
        assert_eq!(metrics.green_attempts_not_closed, 0, "{metrics:?}");
    }

    #[test]
    fn a_step_exhaustion_with_no_verdict_is_the_pre_d115_blind_spot() {
        // The measurement that made the first reading of D-114 vacuous: the attempt ran out of
        // steps, the gates were never consulted, and the journal therefore contains no green
        // record — so "no green attempt was lost" held on the run that motivated the milestone.
        let records = vec![
            attempt_failed(
                "sub_A",
                1,
                serde_json::json!({ "ending": "steps", "steps": 60, "tokens": 42264 }),
            ),
            subtask_state("sub_A", "blocked"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.green_attempts, 0, "{metrics:?}");
        assert_eq!(metrics.green_attempts_not_closed, 0, "{metrics:?}");
        assert_eq!(
            metrics.step_exhaustions_without_gates, 1,
            "a step exhaustion the gates never saw has to be visible: {metrics:?}"
        );
    }

    #[test]
    fn a_step_exhaustion_whose_gates_spoke_is_not_the_blind_spot() {
        // After D-115 the gates *are* asked when the step budget ends. The count must drop to zero
        // even when the verdict is red: "the gates said no" is not "the gates were never asked".
        let records = vec![
            verdict("sub_A", 1, false),
            attempt_failed("sub_A", 1, serde_json::json!({ "ending": "steps" })),
            subtask_state("sub_A", "blocked"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.step_exhaustions_without_gates, 0, "{metrics:?}");
    }

    #[test]
    fn a_green_settlement_after_the_step_budget_is_counted_as_its_own_path() {
        // `settled_by` exists so a reader months later can tell "the model claimed and the gates
        // agreed" from "the budget ended the loop and the gates happened to agree". Without the
        // count, D-115's whole contribution is invisible in `metrics`.
        let records = vec![
            verdict("sub_A", 1, true),
            attempt_failed(
                "sub_A",
                1,
                serde_json::json!({
                    "ending": "gate",
                    "state": "succeeded",
                    "settled_by": "gates_after_step_budget",
                }),
            ),
            subtask_state("sub_A", "done"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.settled_after_step_budget, 1, "{metrics:?}");
        assert_eq!(metrics.green_attempts_not_closed, 0, "{metrics:?}");
    }

    #[test]
    fn a_subtask_reopened_after_a_green_closure_is_reported_rather_than_hidden() {
        // A green attempt whose subtask later went back to `pending`/`running` is work the run
        // walked away from. It may be justified (a retry, an operator) but the count must show it:
        // this is the one honest reason a finished journal can be non-zero, and a reader needs the
        // number rather than an explanation.
        let records = vec![
            verdict("sub_A", 1, true),
            subtask_state("sub_A", "done"),
            subtask_state("sub_A", "blocked"),
        ];
        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.closed_green, 0, "{metrics:?}");
        assert_eq!(metrics.green_attempts_not_closed, 1, "{metrics:?}");
    }

    #[test]
    fn the_run_that_motivated_the_milestone_cannot_pass_the_criterion() {
        // The third live `black_jack` run, in the shape its journal actually has: five subtasks
        // closed green, one blocked, and two step exhaustions on that one subtask with no verdict
        // anywhere near them. Reading only "green attempts that did not close" says zero — and that
        // reading is what made the first version of the criterion vacuous, because the gates were
        // never asked and so no green record could exist.
        //
        // This test is the criterion's own regression: the run the milestone exists for must fail
        // it, and it must fail for the reason the milestone names, not by accident.
        let mut records = Vec::new();
        for id in ["sub_A", "sub_B", "sub_C", "sub_D", "sub_E"] {
            records.push(verdict(id, 1, true));
            records.push(subtask_state(id, "done"));
        }
        // The sixth subtask is the one the plan cannot satisfy, and its two attempts spent their
        // steps before any gate spoke.
        for attempt in [1, 2] {
            records.push(attempt_failed(
                "sub_F",
                attempt,
                serde_json::json!({ "ending": "steps", "steps": 60, "tokens": 42264 }),
            ));
        }
        records.push(subtask_state("sub_F", "blocked"));

        let metrics = compute("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV", &records);
        assert_eq!(metrics.closed_green, 5, "{metrics:?}");
        assert_eq!(metrics.green_attempts, 5, "{metrics:?}");
        assert_eq!(metrics.green_attempts_not_closed, 0, "{metrics:?}");
        assert_eq!(
            metrics.step_exhaustions_without_gates, 2,
            "the run's own defect has to be visible, or the criterion cannot see it: {metrics:?}"
        );
        assert_eq!(
            metrics.settled_after_step_budget, 0,
            "nothing settled on the code path the milestone added: {metrics:?}"
        );
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
