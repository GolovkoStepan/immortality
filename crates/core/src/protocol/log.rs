//! Driving the three-record protocol: ordering, completion, and open-call detection.
//!
//! [`ToolCallLog`] wraps a [`Journal`] and is the only sanctioned way to record a tool call.
//! It exists so that the ordering rules are enforced in one place instead of being remembered
//! at every call site:
//!
//! - `intent` before `started`, always;
//! - `started` only for a call that has an `intent`;
//! - `completed`/`failed` exactly once, and only after `started` or explicitly as a
//!   never-started failure;
//! - a duplicate `call_id` is a bug, not a retry.
//!
//! Recovering open calls is a *read* operation over a finished journal, so it lives in
//! [`open_calls`] rather than on the writer: a recovering daemon has no writer to hold yet.
//!
//! # The one ordering that looks wrong
//!
//! `completed` is written *before* the model sees the result — that is the documented rule and
//! it is what makes the journal authoritative. The consequence is that a crash immediately
//! after `completed` leaves a result the model never saw. That is the safe direction: on
//! recovery the observation is replayed into the conversation from the journal, so the model
//! sees exactly what was durable, and no tool runs twice.

use std::collections::BTreeMap;

use crate::Result;
use crate::journal::{Journal, JournalEvent, Kind};
use crate::protocol::{CallId, Idempotency, Intent, OpenCall, Probe};

/// A journal-backed log of tool calls for one attempt.
#[derive(Debug)]
pub struct ToolCallLog<'journal> {
    journal: &'journal mut Journal,
    /// Calls whose `intent` has been written but which are not finished.
    ///
    /// Kept in memory for ordering checks within one process run. It is not the source of
    /// truth — the journal is — and it deliberately does not survive a restart: after a restart
    /// the in-flight state is exactly what `recover` has to reconstruct.
    open: BTreeMap<CallId, Intent>,
}

impl<'journal> ToolCallLog<'journal> {
    /// Start logging into `journal`.
    #[must_use]
    pub const fn new(journal: &'journal mut Journal) -> Self {
        Self {
            journal,
            open: BTreeMap::new(),
        }
    }

    /// Record the decision to run a tool call. Must be the first thing written.
    pub fn intent(&mut self, intent: &Intent, payload: serde_json::Value) -> Result<()> {
        if self.open.contains_key(&intent.call) {
            return Err(crate::Error::Protocol(format!(
                "call `{}` already has an intent",
                intent.call
            )));
        }
        let record = self.journal.append_event(
            crate::time::now(),
            Kind::ToolCallIntent,
            with_call(&intent.call, payload),
        )?;
        debug_assert_eq!(record.fields.kind, Kind::ToolCallIntent);
        self.open.insert(intent.call.clone(), intent.clone());
        Ok(())
    }

    /// Record that the process actually started.
    ///
    /// `handle` is the pid and start instant for a child process, or `null` for a tool that runs
    /// in-process. It is recorded because "is this pid still alive?" is the only evidence
    /// available for a `start_process` probe after a crash.
    pub fn started(&mut self, call: &CallId, handle: serde_json::Value) -> Result<()> {
        self.require_open(call)?;
        self.journal
            .append_event(crate::time::now(), Kind::ToolCallStarted, with_call(call, handle))?;
        Ok(())
    }

    /// Record a successful outcome. **Call this before handing the output to the model.**
    ///
    /// `output` is the full body; when it is large the caller should have spilled it to the
    /// artifact store and pass a summary in `observation` instead. This method does not decide:
    /// the tool layer knows its own output size, and a policy buried here would be invisible at
    /// the call site where it matters.
    pub fn completed(
        &mut self,
        call: &CallId,
        observation: serde_json::Value,
        artifact: Option<crate::journal::Digest>,
    ) -> Result<()> {
        self.require_open(call)?;
        self.open.remove(call);
        // The call id goes into the payload like every other protocol record: recovery groups
        // by id, and a terminal record without one cannot close anything.
        let observation = with_call(call, observation);
        let event = match artifact {
            Some(digest) => JournalEvent::spilled(observation, digest),
            None => JournalEvent::with_payload(observation),
        };
        self.journal
            .append_full(crate::time::now(), Kind::ToolCallCompleted, event)
            .map(|_| ())
    }

    /// Record a failure. The call is finished; the model must be told.
    pub fn failed(&mut self, call: &CallId, error: serde_json::Value) -> Result<()> {
        self.require_open(call)?;
        self.open.remove(call);
        self.journal
            .append_event(crate::time::now(), Kind::ToolCallFailed, with_call(call, error))?;
        Ok(())
    }

    /// Record that the effect is unknown: the process died or the call could not be completed.
    ///
    /// Distinct from `failed`: a failure says the tool ran and returned an error, unknown says
    /// nobody knows. Only the latter triggers a probe.
    pub fn unknown(&mut self, call: &CallId, reason: serde_json::Value) -> Result<()> {
        self.require_open(call)?;
        self.open.remove(call);
        self.journal
            .append_event(crate::time::now(), Kind::ToolCallUnknown, with_call(call, reason))?;
        Ok(())
    }

    /// Record the verdict of a probe, as required by recovery.
    pub fn effect(&mut self, call: &CallId, effect: Probe, detail: serde_json::Value) -> Result<()> {
        let kind = match effect {
            Probe::Applied => Kind::EffectVerified,
            Probe::NotApplied => Kind::EffectAbsent,
            // A probe that could not decide is not a verdict; it must not be recorded as one.
            Probe::Unknown => {
                return Err(crate::Error::Protocol(format!(
                    "call `{call}`: `Unknown` is not a probe verdict; leave the call open"
                )));
            }
        };
        self.journal
            .append_event(crate::time::now(), kind, with_call(call, detail))?;
        Ok(())
    }

    /// Calls still open in this process. A non-empty value at shutdown is a bug.
    #[must_use]
    pub fn open_calls(&self) -> Vec<&CallId> {
        self.open.keys().collect()
    }

    /// Whether the log believes `call` is finished.
    #[must_use]
    pub fn is_open(&self, call: &CallId) -> bool {
        self.open.contains_key(call)
    }

    fn require_open(&self, call: &CallId) -> Result<()> {
        if !self.open.contains_key(call) {
            return Err(crate::Error::Protocol(format!(
                "call `{call}` has no open intent; a record out of order is a bug in the caller"
            )));
        }
        Ok(())
    }
}

/// A `call` field merged into every protocol payload.
///
/// The call id is duplicated in the payload *and* implied by the event kind, because recovery
/// groups records by id and the kind alone would force a per-kind parser.
fn with_call(call: &CallId, payload: serde_json::Value) -> serde_json::Value {
    match payload {
        serde_json::Value::Object(mut map) => {
            map.insert("call".into(), serde_json::Value::String(call.as_str().into()));
            serde_json::Value::Object(map)
        }
        other => serde_json::json!({ "call": call.as_str(), "detail": other }),
    }
}

/// An unfinished call found by scanning a journal, with its position for reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundOpenCall {
    pub open: OpenCall,
    /// `seq` of the `intent` record, so a human can jump to it in the file.
    pub intent_seq: u64,
    /// `seq` of the `started` record, if any.
    pub started_seq: Option<u64>,
}

/// Find calls that have an `intent` but no terminal record.
///
/// Scans in `seq` order, which is the order the records were written. A terminal record without
/// an intent is ignored here rather than reported: it is a schema violation caught by the
/// journal's own chain check, and inventing a call for it would fabricate history.
#[must_use]
pub fn open_calls(records: &[crate::journal::JournalRecord]) -> Vec<FoundOpenCall> {
    let mut pending: BTreeMap<String, FoundOpenCall> = BTreeMap::new();
    // Preserve discovery order for output stability: a human reading `recovery` output expects
    // the same order as the journal, not alphabetical order of ids.
    let mut order: Vec<String> = Vec::new();

    for record in records {
        let Some(call) = record.fields.kind.is_tool_call().then(|| {
            record
                .event
                .payload
                .get("call")
                .and_then(|v| v.as_str())
                .map(str::to_string)
        }) else {
            continue;
        };
        let Some(call) = call else { continue };

        match record.fields.kind {
            Kind::ToolCallIntent => {
                // A second intent for the same call means the id was reused; the *first* one
                // governs, because it is the one whose effect could have happened.
                if pending.contains_key(&call) {
                    continue;
                }
                let intent = intent_from_payload(&call, &record.event.payload);
                order.push(call.clone());
                pending.insert(
                    call,
                    FoundOpenCall {
                        open: OpenCall {
                            intent,
                            started: false,
                            marked_unknown: false,
                        },
                        intent_seq: record.fields.seq,
                        started_seq: None,
                    },
                );
            }
            Kind::ToolCallStarted => {
                if let Some(found) = pending.get_mut(&call) {
                    found.open.started = true;
                    found.started_seq = Some(record.fields.seq);
                }
            }
            Kind::ToolCallUnknown => {
                if let Some(found) = pending.get_mut(&call) {
                    found.open.marked_unknown = true;
                }
            }
            Kind::ToolCallCompleted | Kind::ToolCallFailed => {
                pending.remove(&call);
            }
            _ => {}
        }
    }

    order
        .into_iter()
        .filter_map(|call| pending.remove(&call))
        .collect()
}

/// Rebuild an [`Intent`] from a journal payload.
///
/// Tolerant on purpose: a payload written by an older build may miss `expected_effect`, and
/// losing the whole recovery pass over a cosmetic field would be worse than showing an empty
/// string. The fields that decide behaviour — `idempotency`, `call` — are required.
fn intent_from_payload(call: &str, payload: &serde_json::Value) -> Intent {
    let idempotency = payload
        .get("idempotency")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        // An unclassifiable call is treated as the most dangerous case. Guessing "idempotent"
        // here would turn a corrupt journal into a duplicated destructive command.
        .unwrap_or(Idempotency::Unknown);
    Intent {
        call: CallId::from_model(call),
        tool: payload
            .get("tool")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string(),
        args_digest: payload
            .get("args_digest")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        idempotency,
        expected_effect: payload
            .get("expected_effect")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::TaskId;
    use crate::journal::JournalReader;
    use chrono::TimeZone as _;

    fn task() -> TaskId {
        TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV")
    }

    fn intent(name: &str, idempotency: Idempotency) -> Intent {
        Intent::new(
            CallId::from_model(name),
            "shell",
            &serde_json::json!({"command": "ls"}),
            idempotency,
            "list",
        )
    }

    fn open(dir: &std::path::Path) -> Journal {
        Journal::open(dir, task()).unwrap()
    }

    #[test]
    fn the_happy_path_writes_three_records_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        let call = intent("call_1", Idempotency::Idempotent);
        {
            let mut log = ToolCallLog::new(&mut journal);
            log.intent(&call, serde_json::json!({ "tool": "shell" })).unwrap();
            log.started(&call.call, serde_json::json!({ "pid": 42 })).unwrap();
            log.completed(&call.call, serde_json::json!({ "exit_code": 0 }), None)
                .unwrap();
            assert_eq!(log.open_calls().len(), 0);
        }
        journal.close().unwrap();

        let reader = JournalReader::open(dir.path(), &task()).unwrap();
        let outcome = reader.read().unwrap();
        let kinds: Vec<Kind> = outcome.records.iter().map(|r| r.fields.kind).collect();
        assert_eq!(
            kinds,
            [
                Kind::ToolCallIntent,
                Kind::ToolCallStarted,
                Kind::ToolCallCompleted
            ]
        );
        let seqs: Vec<u64> = outcome.records.iter().map(|r| r.fields.seq).collect();
        assert_eq!(seqs, [0, 1, 2]);
        assert_eq!(open_calls(&outcome.records).len(), 0);
    }

    #[test]
    fn the_call_id_is_present_in_every_payload() {
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        let call = intent("call_7", Idempotency::Idempotent);
        {
            let mut log = ToolCallLog::new(&mut journal);
            log.intent(&call, serde_json::json!({ "tool": "shell" })).unwrap();
            log.started(&call.call, serde_json::Value::Null).unwrap();
            log.completed(&call.call, serde_json::json!({}), None).unwrap();
        }
        journal.close().unwrap();
        let reader = JournalReader::open(dir.path(), &task()).unwrap();
        for record in reader.read().unwrap().records {
            assert_eq!(record.event.payload["call"], "call_7", "{record:?}");
        }
    }

    #[test]
    fn a_record_without_an_intent_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        let mut log = ToolCallLog::new(&mut journal);
        let call = CallId::from_model("call_missing");
        let error = log
            .started(&call, serde_json::json!({}))
            .expect_err("started without intent must fail");
        assert!(matches!(error, crate::Error::Protocol(_)), "{error}");
        assert!(log.completed(&call, serde_json::json!({}), None).is_err());
        assert!(log.failed(&call, serde_json::json!({})).is_err());
    }

    #[test]
    fn a_duplicate_intent_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        let mut log = ToolCallLog::new(&mut journal);
        let call = intent("call_dup", Idempotency::Idempotent);
        log.intent(&call, serde_json::json!({})).unwrap();
        assert!(log.intent(&call, serde_json::json!({})).is_err());
    }

    #[test]
    fn completing_twice_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        let mut log = ToolCallLog::new(&mut journal);
        let call = intent("call_twice", Idempotency::Idempotent);
        log.intent(&call, serde_json::json!({})).unwrap();
        log.started(&call.call, serde_json::json!({})).unwrap();
        log.completed(&call.call, serde_json::json!({}), None).unwrap();
        assert!(log.completed(&call.call, serde_json::json!({}), None).is_err());
    }

    #[test]
    fn a_never_started_call_can_be_marked_unknown_only_after_intent() {
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        let mut log = ToolCallLog::new(&mut journal);
        let call = intent("call_unknown", Idempotency::Unknown);
        log.intent(&call, serde_json::json!({})).unwrap();
        log.unknown(&call.call, serde_json::json!({"reason": "timeout"}))
            .unwrap();
        assert_eq!(log.open_calls().len(), 0);
    }

    #[test]
    fn a_probe_verdict_cannot_be_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        let mut log = ToolCallLog::new(&mut journal);
        let call = CallId::from_model("call_probe");
        assert!(
            log.effect(&call, Probe::Unknown, serde_json::json!({})).is_err(),
            "unknown is not a verdict"
        );
        log.effect(&call, Probe::Applied, serde_json::json!({"commit": "abc"}))
            .unwrap();
        log.effect(&call, Probe::NotApplied, serde_json::json!({}))
            .unwrap();
    }

    #[test]
    fn open_calls_finds_an_intent_without_started() {
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        {
            let mut log = ToolCallLog::new(&mut journal);
            log.intent(
                &intent("call_a", Idempotency::Idempotent),
                serde_json::json!({"tool": "shell", "idempotency": "idempotent"}),
            )
            .unwrap();
        }
        journal.close().unwrap();

        let reader = JournalReader::open(dir.path(), &task()).unwrap();
        let found = open_calls(&reader.read().unwrap().records);
        assert_eq!(found.len(), 1);
        assert!(!found[0].open.started);
        assert_eq!(
            found[0].open.required_action(),
            crate::protocol::RecoveryAction::Retry
        );
        assert_eq!(found[0].intent_seq, 0);
        assert_eq!(found[0].started_seq, None);
    }

    #[test]
    fn open_calls_finds_a_started_call_without_a_terminal_record() {
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        {
            let mut log = ToolCallLog::new(&mut journal);
            let call = intent("call_b", Idempotency::Unknown);
            log.intent(
                &call,
                serde_json::json!({"tool": "shell", "idempotency": "unknown"}),
            )
            .unwrap();
            log.started(&call.call, serde_json::json!({"pid": 1})).unwrap();
        }
        journal.close().unwrap();

        let reader = JournalReader::open(dir.path(), &task()).unwrap();
        let found = open_calls(&reader.read().unwrap().records);
        assert_eq!(found.len(), 1);
        assert!(found[0].open.started);
        assert_eq!(
            found[0].open.required_action(),
            crate::protocol::RecoveryAction::Probe
        );
        assert_eq!(found[0].started_seq, Some(1));
    }

    #[test]
    fn a_failed_call_is_not_open() {
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        {
            let mut log = ToolCallLog::new(&mut journal);
            let call = intent("call_f", Idempotency::Unknown);
            log.intent(
                &call,
                serde_json::json!({"tool": "shell", "idempotency": "unknown"}),
            )
            .unwrap();
            log.started(&call.call, serde_json::json!({})).unwrap();
            log.failed(&call.call, serde_json::json!({"exit_code": 2}))
                .unwrap();
        }
        journal.close().unwrap();
        let reader = JournalReader::open(dir.path(), &task()).unwrap();
        assert_eq!(open_calls(&reader.read().unwrap().records).len(), 0);
    }

    #[test]
    fn a_call_recorded_unknown_stays_open_for_recovery() {
        // `unknown` is not terminal: the whole point is that the next boot must probe it.
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        {
            let mut log = ToolCallLog::new(&mut journal);
            let call = intent("call_u", Idempotency::Unknown);
            log.intent(
                &call,
                serde_json::json!({"tool": "shell", "idempotency": "unknown"}),
            )
            .unwrap();
            log.unknown(&call.call, serde_json::json!({"reason": "daemon died"}))
                .unwrap();
        }
        journal.close().unwrap();
        let reader = JournalReader::open(dir.path(), &task()).unwrap();
        let found = open_calls(&reader.read().unwrap().records);
        assert_eq!(found.len(), 1);
        assert!(found[0].open.marked_unknown);
        assert_eq!(
            found[0].open.required_action(),
            crate::protocol::RecoveryAction::Probe
        );
    }

    #[test]
    fn a_terminal_record_without_an_intent_does_not_invent_a_call() {
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        journal
            .append_event(
                crate::time::now(),
                Kind::ToolCallCompleted,
                serde_json::json!({"call": "ghost"}),
            )
            .unwrap();
        journal.close().unwrap();
        let reader = JournalReader::open(dir.path(), &task()).unwrap();
        assert_eq!(open_calls(&reader.read().unwrap().records).len(), 0);
    }

    #[test]
    fn calls_are_reported_in_journal_order() {
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        {
            let mut log = ToolCallLog::new(&mut journal);
            for name in ["call_z", "call_a", "call_m"] {
                log.intent(
                    &intent(name, Idempotency::Idempotent),
                    serde_json::json!({"idempotency": "idempotent"}),
                )
                .unwrap();
            }
        }
        journal.close().unwrap();
        let reader = JournalReader::open(dir.path(), &task()).unwrap();
        let found = open_calls(&reader.read().unwrap().records);
        let names: Vec<&str> = found.iter().map(|f| f.open.intent.call.as_str()).collect();
        assert_eq!(names, ["call_z", "call_a", "call_m"]);
    }

    #[test]
    fn a_completed_call_with_a_spilled_body_keeps_its_summary() {
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        let digest = crate::artifacts::digest_of(b"huge output");
        {
            let mut log = ToolCallLog::new(&mut journal);
            let call = intent("call_spill", Idempotency::Idempotent);
            log.intent(&call, serde_json::json!({"idempotency": "idempotent"}))
                .unwrap();
            log.started(&call.call, serde_json::json!({})).unwrap();
            log.completed(&call.call, serde_json::json!({"bytes": 11}), Some(digest))
                .unwrap();
        }
        journal.close().unwrap();
        let reader = JournalReader::open(dir.path(), &task()).unwrap();
        let records = reader.read().unwrap().records;
        let completed = records
            .iter()
            .find(|r| r.fields.kind == Kind::ToolCallCompleted)
            .unwrap();
        assert_eq!(completed.event.payload["bytes"], 11);
        assert_eq!(
            completed.event.artifact.as_deref(),
            Some(crate::journal::digest_text(&digest).as_str())
        );
        assert_eq!(open_calls(&records).len(), 0);
    }

    #[test]
    fn an_unclassifiable_intent_is_treated_as_unknown() {
        // A corrupted or older payload must not be optimistically retried.
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        journal
            .append_event(
                crate::time::now(),
                Kind::ToolCallIntent,
                serde_json::json!({"call": "call_legacy", "tool": "shell"}),
            )
            .unwrap();
        journal.close().unwrap();
        let reader = JournalReader::open(dir.path(), &task()).unwrap();
        let found = open_calls(&reader.read().unwrap().records);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].open.intent.idempotency, Idempotency::Unknown);
        assert_eq!(found[0].open.intent.tool, "shell");
        assert_eq!(found[0].open.intent.expected_effect, "");
    }

    #[test]
    fn a_crash_between_intent_and_started_is_probed_end_to_end() {
        // The exact scenario the roadmap requires: kill the process at every point of the
        // protocol and prove recovery lands somewhere correct.
        for crash_point in ["after_intent", "after_started"] {
            let dir = tempfile::tempdir().unwrap();
            let mut journal = open(dir.path());
            let call = intent("call_crash", Idempotency::Unknown);
            {
                let mut log = ToolCallLog::new(&mut journal);
                log.intent(
                    &call,
                    serde_json::json!({
                        "tool": "shell",
                        "idempotency": "unknown",
                        "expected_effect": "remove the build directory",
                    }),
                )
                .unwrap();
                if crash_point == "after_started" {
                    log.started(&call.call, serde_json::json!({"pid": 4242})).unwrap();
                }
                // The process dies here. Nothing else is written.
            }
            journal.close().unwrap();

            // Recovery: read, find the open call, probe, record the verdict, retry.
            let reader = JournalReader::open(dir.path(), &task()).unwrap();
            let found = open_calls(&reader.read().unwrap().records);
            assert_eq!(found.len(), 1, "{crash_point}");
            assert_eq!(
                found[0].open.required_action(),
                crate::protocol::RecoveryAction::Probe,
                "{crash_point}: an unknown call must never be blindly retried"
            );
            assert_eq!(found[0].open.started, crash_point == "after_started");

            let mut journal = Journal::open(dir.path(), task()).unwrap();
            {
                let mut log = ToolCallLog::new(&mut journal);
                log.effect(
                    &call.call,
                    crate::protocol::Probe::NotApplied,
                    serde_json::json!({"checked": "build/ exists"}),
                )
                .unwrap();
                log.intent(
                    &call,
                    serde_json::json!({"tool": "shell", "idempotency": "unknown"}),
                )
                .unwrap();
                log.started(&call.call, serde_json::json!({"pid": 4243})).unwrap();
                log.completed(&call.call, serde_json::json!({"exit_code": 0}), None)
                    .unwrap();
            }
            journal.close().unwrap();

            let reader = JournalReader::open(dir.path(), &task()).unwrap();
            let records = reader.read().unwrap().records;
            assert_eq!(open_calls(&records).len(), 0, "{crash_point}");
            let kinds: Vec<Kind> = records.iter().map(|r| r.fields.kind).collect();
            assert!(kinds.contains(&Kind::EffectAbsent), "{crash_point}: {kinds:?}");
            assert_eq!(
                records
                    .iter()
                    .filter(|r| r.fields.kind == Kind::ToolCallCompleted)
                    .count(),
                1,
                "{crash_point}: the retry must finish exactly once"
            );
        }
    }

    #[test]
    fn a_crash_between_started_and_completed_keeps_the_call_open() {
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        let call = intent("call_mid", Idempotency::ByContent);
        {
            let mut log = ToolCallLog::new(&mut journal);
            log.intent(
                &call,
                serde_json::json!({"tool": "write_file", "idempotency": "by_content"}),
            )
            .unwrap();
            log.started(&call.call, serde_json::Value::Null).unwrap();
        }
        journal.close().unwrap();

        let reader = JournalReader::open(dir.path(), &task()).unwrap();
        let found = open_calls(&reader.read().unwrap().records);
        assert_eq!(found.len(), 1);
        assert_eq!(
            found[0].open.required_action(),
            crate::protocol::RecoveryAction::VerifyContent
        );
    }

    #[test]
    fn distinct_calls_do_not_close_each_other() {
        // A copy-paste bug that drops the call id would make the second completion close the
        // first call. Catch it with two overlapping calls.
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        {
            let mut log = ToolCallLog::new(&mut journal);
            let a = intent("call_a", Idempotency::Idempotent);
            let b = intent("call_b", Idempotency::Idempotent);
            log.intent(&a, serde_json::json!({"idempotency": "idempotent"}))
                .unwrap();
            log.intent(&b, serde_json::json!({"idempotency": "idempotent"}))
                .unwrap();
            log.started(&a.call, serde_json::json!({})).unwrap();
            log.started(&b.call, serde_json::json!({})).unwrap();
            log.completed(&a.call, serde_json::json!({}), None).unwrap();
        }
        journal.close().unwrap();

        let reader = JournalReader::open(dir.path(), &task()).unwrap();
        let found = open_calls(&reader.read().unwrap().records);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].open.intent.call.as_str(), "call_b");
    }

    #[test]
    fn timestamp_comes_from_the_clock_not_the_caller() {
        // Guards against a caller passing a stale timestamp and making the journal look like it
        // was written earlier than it was.
        let dir = tempfile::tempdir().unwrap();
        let mut journal = open(dir.path());
        let before = chrono::Utc::now();
        {
            let mut log = ToolCallLog::new(&mut journal);
            log.intent(
                &intent("call_now", Idempotency::Idempotent),
                serde_json::json!({"idempotency": "idempotent"}),
            )
            .unwrap();
        }
        journal.close().unwrap();
        let reader = JournalReader::open(dir.path(), &task()).unwrap();
        let records = reader.read().unwrap().records;
        let after = chrono::Utc::now();
        let stamp = records[0].fields.at;
        assert!(stamp >= before - chrono::Duration::milliseconds(1));
        assert!(stamp <= after);
        let _ = chrono::Utc.timestamp_opt(0, 0);
    }
}
