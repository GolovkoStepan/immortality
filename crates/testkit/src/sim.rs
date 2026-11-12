//! Crash-point simulator for the tool-call protocol.
//!
//! # What this simulates
//!
//! The roadmap's exit criterion for M1 is not "recovery has tests" but "the simulator kills the
//! process at *every* point of the tool-call protocol and `recover` lands in a correct state
//! every time". This module is that simulator: it replays a scripted tool call up to a named
//! point, dies there, then feeds the resulting journal to [`immortality_core::recover::plan`].
//!
//! # Why a real journal and a real crash point
//!
//! A fake `ReadOutcome` built by hand would test the planner against what the author *believes*
//! the writer produces. The interesting failures live in the gap: a field the writer puts in the
//! payload but the planner looks for at the top level, a torn tail the writer produces and the
//! planner mishandles. So every scenario runs through [`Journal`] and
//! [`JournalReader`], and a torn tail is a real half-line appended to a real file.
//!
//! # The invariant that matters
//!
//! A destructive call — one whose effect cannot be known after a crash — must **never** be
//! planned for retry when the probe says `Applied` or `Unknown`. Everything else in this module
//! is bookkeeping; that one rule is the reason the three-record protocol exists.

use std::path::Path;

use immortality_core::domain::TaskId;
use immortality_core::journal::{Journal, JournalReader, Kind, ReadOutcome};
use immortality_core::protocol::{CallId, Idempotency, Intent, Probe, RecoveryAction, ToolCallLog};
use immortality_core::recover::{
    self, LastCheckpoint, PersistedState, Prober, RecoveryReport, RecoveryStep, Verdict, WorkspaceFacts,
};

/// Where in a tool call's life the process died.
///
/// The points are the ones the journal can distinguish. `MidAppend` is not a protocol step but
/// the shape a crash leaves *inside* a step: a line without its newline, which is the only kind
/// of torn record the writer can produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CrashPoint {
    /// Before `intent`: the call does not exist and nothing can have happened.
    BeforeIntent,
    /// `intent` is durable; the tool has not been launched by this process.
    AfterIntent,
    /// Between `intent` and `started`: the tool may already be running.
    BeforeStarted,
    /// `started` is durable; the tool is running or died mid-run.
    AfterStarted,
    /// Between `started` and `completed`: the classic "did the `rm` happen?" window.
    BeforeCompleted,
    /// `completed` is durable but the model never saw the result.
    AfterCompleted,
    /// A line was half-written and the process died mid-`write_all`.
    MidAppend,
}

impl CrashPoint {
    /// Every point, in protocol order.
    pub const ALL: [Self; 7] = [
        Self::BeforeIntent,
        Self::AfterIntent,
        Self::BeforeStarted,
        Self::AfterStarted,
        Self::BeforeCompleted,
        Self::AfterCompleted,
        Self::MidAppend,
    ];

    /// Whether any record of the call is durable at this point.
    #[must_use]
    pub const fn call_was_recorded(self) -> bool {
        !matches!(self, Self::BeforeIntent)
    }

    /// Whether the call is closed in the journal at this point.
    #[must_use]
    pub const fn call_was_closed(self) -> bool {
        matches!(self, Self::AfterCompleted)
    }
}

impl std::fmt::Display for CrashPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::BeforeIntent => "before_intent",
            Self::AfterIntent => "after_intent",
            Self::BeforeStarted => "before_started",
            Self::AfterStarted => "after_started",
            Self::BeforeCompleted => "before_completed",
            Self::AfterCompleted => "after_completed",
            Self::MidAppend => "mid_append",
        };
        f.write_str(name)
    }
}

/// A journal left behind by a crash, with the facts recovery would gather.
#[derive(Debug)]
pub struct Scenario {
    pub point: CrashPoint,
    pub idempotency: Idempotency,
    pub task: TaskId,
    pub call: CallId,
    pub outcome: ReadOutcome,
}

impl Scenario {
    /// Plan recovery for this scenario, with `probe` as the tool's answer.
    ///
    /// `checkpoint` is the SHA of the last green checkpoint, or `None` when the task has none —
    /// which is a separate branch in the planner, not a detail.
    #[must_use]
    pub fn report(&self, probe: Probe, checkpoint: Option<&str>) -> RecoveryReport {
        let input = recover::RecoveryInput {
            journal: &self.outcome,
            workspace: WorkspaceFacts::healthy("agent/tsk_01", "abc"),
            persisted: PersistedState::default(),
            last_checkpoint: checkpoint.map(|sha| LastCheckpoint {
                sha: sha.to_string(),
                kind: immortality_core::checkpoint::CheckpointKind::Hidden,
            }),
            expected_branch: "agent/tsk_01",
        };
        recover::plan(&input, &Fixed(probe))
    }

    /// The steps that re-run a tool, in plan order.
    #[must_use]
    pub fn retried_calls(&self, report: &RecoveryReport) -> Vec<String> {
        report
            .steps
            .iter()
            .filter_map(|step| match step {
                RecoveryStep::RetryCall { call, .. } => Some(call.as_str().to_string()),
                _ => None,
            })
            .collect()
    }
}

/// A prober that answers the same thing every time, so the test controls the branch.
#[derive(Debug, Clone, Copy)]
pub struct Fixed(pub Probe);

impl Prober for Fixed {
    fn probe(&self, _intent: &Intent) -> Probe {
        self.0
    }
}

/// Replay a tool call up to `point` and die there, leaving the journal on disk.
///
/// The call is `rm -rf build` — anonymous but destructive — or a `read_file` when `idempotency`
/// makes it safe to repeat. The arguments differ so that a duplicate-detection bug cannot hide
/// behind two calls that happen to hash the same.
///
/// The directory is the caller's: tests own their temp dirs (D-024), and a helper that created
/// one would make `Scenario` responsible for cleanup.
pub fn crash_at(
    dir: &Path,
    point: CrashPoint,
    idempotency: Idempotency,
) -> immortality_core::Result<Scenario> {
    let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
    let call = CallId::from_model("call_sim");
    let tool = if idempotency.is_safe_to_retry() {
        "read_file"
    } else {
        "shell"
    };
    let intent = Intent::new(
        call.clone(),
        tool,
        &serde_json::json!({ "command": "rm -rf build" }),
        idempotency,
        "remove the build directory",
    );

    if point != CrashPoint::BeforeIntent {
        let mut journal = Journal::open(dir, task.clone())?;
        {
            let mut log = ToolCallLog::new(&mut journal);
            log.intent(
                &intent,
                serde_json::json!({
                    "tool": tool,
                    "idempotency": idempotency,
                    "expected_effect": intent.expected_effect,
                    "args_digest": intent.args_digest,
                }),
            )?;
            if matches!(
                point,
                CrashPoint::AfterStarted | CrashPoint::BeforeCompleted | CrashPoint::AfterCompleted
            ) {
                log.started(&call, serde_json::json!({ "pid": 4242 }))?;
            }
            if point == CrashPoint::AfterCompleted {
                log.completed(&call, serde_json::json!({ "exit_code": 0 }), None)?;
            }
        }
        journal.close()?;
    }

    if point == CrashPoint::MidAppend {
        // A half-written line: the writer writes the newline in the same `write_all`, so a line
        // without one is provably the tail. Appending it by hand is the only way to reproduce
        // the state without a power cut.
        let path = dir.join("JOURNAL.jsonl");
        let mut bytes = std::fs::read(&path).map_err(|e| immortality_core::Error::io(&path, e))?;
        bytes.extend_from_slice(
            br#"{"v":1,"seq":99,"ts":"2026-11-12T10:14:03.221Z","kind":"tool_call_completed""#,
        );
        std::fs::write(&path, &bytes).map_err(|e| immortality_core::Error::io(&path, e))?;
    }

    let outcome = JournalReader::open(dir, &task)?.read()?;
    Ok(Scenario {
        point,
        idempotency,
        task,
        call,
        outcome,
    })
}

/// Assert every recovery invariant for one crash point and one probe answer.
///
/// Written as assertions rather than as a returned list because the failure message has to name
/// the point: "the destructive call was retried" is useless without knowing where the process
/// died.
pub fn verify(dir: &Path, point: CrashPoint, idempotency: Idempotency, probe: Probe) {
    let scenario = crash_at(dir, point, idempotency)
        .unwrap_or_else(|e| panic!("{point}: the simulator could not build a journal: {e}"));
    let report = scenario.report(probe, Some("deadbeef"));

    // 1. Every unfinished call has a disposition, and no disposition is invented.
    assert!(
        recover::all_calls_addressed(&scenario.outcome.records, &report),
        "{point}: the plan does not answer every open call: {report:?}"
    );

    // 2. A torn tail is repaired, not blocked.
    if point == CrashPoint::MidAppend {
        assert!(
            matches!(report.steps.first(), Some(RecoveryStep::DropTornTail { .. })),
            "{point}: a torn tail must be dropped first: {:?}",
            report.steps
        );
        assert_ne!(
            report.verdict,
            Verdict::Blocked,
            "{point}: a torn tail is recoverable and must not block the task"
        );
    }

    // 3. A closed or never-recorded call leaves nothing to do.
    if point.call_was_closed() || !point.call_was_recorded() {
        assert_eq!(report.verdict, Verdict::Healthy, "{point}: {report:?}");
        assert!(report.calls.is_empty(), "{point}: {report:?}");
        assert!(report.steps.is_empty(), "{point}: {report:?}");
        check_journal_matches_verdict(&scenario);
        return;
    }

    // 4. The call is open and has exactly one disposition.
    assert_eq!(
        report.calls.len(),
        1,
        "{point}: one open call, one disposition: {report:?}"
    );
    assert_eq!(report.calls[0].call, scenario.call, "{point}");
    check_disposition(&scenario, &report, probe);

    // 5. A repair is recorded, and the record is the last step.
    assert_eq!(report.verdict, Verdict::Repaired, "{point}: {report:?}");
    assert!(
        matches!(report.steps.last(), Some(RecoveryStep::RecordRecovery { .. })),
        "{point}: the audit trail must be written after the repair it describes: {:?}",
        report.steps
    );

    // 6. The journal really says what the verdict claims.
    check_journal_matches_verdict(&scenario);
}

/// Assert the planner did the right thing about an open call, given the probe's answer.
fn check_disposition(scenario: &Scenario, report: &RecoveryReport, probe: Probe) {
    let point = scenario.point;
    let recovered = &report.calls[0];
    let retried = scenario.retried_calls(report);

    // An `Idempotent` call is never probed: repeating it is safe by definition, so the probe
    // answer is not consulted and must not change the plan. Asserting this catches a planner
    // that starts wasting a probe on `read_file`.
    if scenario.idempotency == Idempotency::Idempotent {
        assert_eq!(
            retried,
            [scenario.call.as_str()],
            "{point}: an idempotent call is retried without a probe: {report:?}"
        );
        assert_eq!(recovered.action, RecoveryAction::Retry, "{point}");
        assert!(recovered.effect.is_none(), "{point}: no probe was needed");
        return;
    }

    match probe {
        // A known-absent effect: the call did not happen, so re-running it is the repair.
        Probe::NotApplied => {
            assert_eq!(
                retried,
                [scenario.call.as_str()],
                "{point}: an absent effect must be retried exactly once: {report:?}"
            );
            // `action` names what recovery had to *ask* (`VerifyContent` for `ByContent`,
            // `Probe` for `Unknown`), not what it does with the answer.
            assert_ne!(recovered.action, RecoveryAction::Abandon, "{point}");
        }
        // An applied effect: the observation is synthesized, the tool never runs again.
        Probe::Applied => {
            assert!(
                retried.is_empty(),
                "{point}: a call whose effect is present must not run again: {report:?}"
            );
            assert!(
                report
                    .steps
                    .iter()
                    .any(|step| matches!(step, RecoveryStep::SynthesizeApplied { .. })),
                "{point}: the model still needs an observation: {:?}",
                report.steps
            );
        }
        // Unknown: roll back, never retry. A destructive call must not run twice, which is the
        // whole reason the probe exists.
        Probe::Unknown => {
            assert!(
                retried.is_empty(),
                "{point}: a call of unknown effect must never be blindly retried: {report:?}"
            );
            assert!(
                report
                    .steps
                    .iter()
                    .any(|step| matches!(step, RecoveryStep::RollbackToCheckpoint { .. })),
                "{point}: with a checkpoint the plan must roll back: {:?}",
                report.steps
            );
        }
    }
}

/// Assert the simulated journal contains the records the crash point claims it does.
fn check_journal_matches_verdict(scenario: &Scenario) {
    let closed = scenario
        .outcome
        .records
        .iter()
        .any(|record| record.fields.kind == Kind::ToolCallCompleted);
    assert_eq!(
        closed,
        scenario.point.call_was_closed(),
        "{}: the scenario did not write the records it claims",
        scenario.point
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch() -> tempfile::TempDir {
        tempfile::tempdir().expect("a temp dir")
    }

    #[test]
    fn every_crash_point_produces_a_correct_plan_for_every_idempotency_class() {
        // The roadmap's M1 exit criterion, in one test: kill the process at each protocol point
        // and prove recovery is correct rather than plausible.
        for point in CrashPoint::ALL {
            for idempotency in [
                Idempotency::Idempotent,
                Idempotency::ByContent,
                Idempotency::Unknown,
            ] {
                for probe in [Probe::Applied, Probe::NotApplied, Probe::Unknown] {
                    let dir = scratch();
                    verify(dir.path(), point, idempotency, probe);
                }
            }
        }
    }

    #[test]
    fn a_destructive_call_is_never_retried_when_the_effect_is_uncertain() {
        // The single invariant that justifies the three-record protocol, asserted on its own so
        // that a refactor breaking it fails with a readable name rather than inside a loop.
        for point in [
            CrashPoint::AfterIntent,
            CrashPoint::AfterStarted,
            CrashPoint::BeforeCompleted,
        ] {
            let dir = scratch();
            let scenario = crash_at(dir.path(), point, Idempotency::Unknown).unwrap();
            for probe in [Probe::Applied, Probe::Unknown] {
                let report = scenario.report(probe, Some("deadbeef"));
                assert!(
                    scenario.retried_calls(&report).is_empty(),
                    "{point} with {probe:?}: a destructive call ran or may have run and was \
                     planned for retry: {report:?}"
                );
            }
        }
    }

    #[test]
    fn without_a_checkpoint_an_undecidable_effect_is_abandoned_not_retried() {
        let dir = scratch();
        let scenario = crash_at(dir.path(), CrashPoint::AfterStarted, Idempotency::Unknown).unwrap();
        let report = scenario.report(Probe::Unknown, None);
        assert!(
            report
                .steps
                .iter()
                .any(|step| matches!(step, RecoveryStep::AbandonCall { .. })),
            "no checkpoint means no safe rollback, and no retry: {:?}",
            report.steps
        );
        assert!(scenario.retried_calls(&report).is_empty(), "{report:?}");
    }

    #[test]
    fn an_idempotent_call_is_retried_whether_or_not_it_started() {
        // The mirror of the rule above: `read_file` may safely run twice, so the cheap path is
        // taken and no probe is consulted.
        for point in [CrashPoint::AfterIntent, CrashPoint::AfterStarted] {
            let dir = scratch();
            let scenario = crash_at(dir.path(), point, Idempotency::Idempotent).unwrap();
            let report = scenario.report(Probe::Unknown, Some("deadbeef"));
            assert_eq!(
                scenario.retried_calls(&report),
                [scenario.call.as_str()],
                "{point}: {report:?}"
            );
        }
    }

    #[test]
    fn a_torn_tail_is_dropped_and_counted_before_anything_else() {
        let dir = scratch();
        let scenario = crash_at(dir.path(), CrashPoint::MidAppend, Idempotency::Unknown).unwrap();
        assert!(scenario.outcome.has_torn_tail);
        let report = scenario.report(Probe::NotApplied, Some("deadbeef"));
        assert!(report.dropped_tail_bytes > 0, "{report:?}");
        assert!(matches!(
            report.steps.first(),
            Some(RecoveryStep::DropTornTail { .. })
        ));
    }

    #[test]
    fn the_simulated_journal_has_the_records_the_point_claims() {
        // Guards the simulator itself: if `crash_at` stopped writing `started`, every invariant
        // above would still pass while testing nothing.
        let kinds = |point| {
            let dir = scratch();
            let scenario = crash_at(dir.path(), point, Idempotency::Unknown).unwrap();
            scenario
                .outcome
                .records
                .iter()
                .map(|record| record.fields.kind)
                .collect::<Vec<Kind>>()
        };
        assert_eq!(kinds(CrashPoint::BeforeIntent), Vec::<Kind>::new());
        assert_eq!(kinds(CrashPoint::AfterIntent), [Kind::ToolCallIntent]);
        assert_eq!(
            kinds(CrashPoint::AfterStarted),
            [Kind::ToolCallIntent, Kind::ToolCallStarted]
        );
        assert_eq!(
            kinds(CrashPoint::AfterCompleted),
            [
                Kind::ToolCallIntent,
                Kind::ToolCallStarted,
                Kind::ToolCallCompleted
            ]
        );
        // `MidAppend` is the same as `AfterIntent` plus a torn line, which the reader drops.
        assert_eq!(kinds(CrashPoint::MidAppend), [Kind::ToolCallIntent]);
    }
}
