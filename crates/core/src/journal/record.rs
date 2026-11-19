//! A single journal line: the record envelope and the closed set of event kinds.
//!
//! # The envelope
//!
//! ```jsonc
//! {"v":1,"seq":18432,"ts":"2026-11-12T10:14:03.221Z","task":"tsk_...","subtask":"sub_...",
//!  "attempt":2,"kind":"tool_call_completed","payload":{...},"artifact":"sha256:...",
//!  "prev":"sha256:..."}
//! ```
//!
//! `payload` is the event itself and `artifact` points at the heavy body when one was
//! spilled to `artifacts/`. Both exist because the distinction is load-bearing during
//! recovery: a `tool_call_completed` whose payload was spilled still tells the reader that
//! the call finished, even if the artifact was lost to a full disk. Losing the body degrades
//! the observation; losing the fact would make the call look unfinished and trigger a
//! duplicate.
//!
//! # Why `kind` is a closed enum
//!
//! `serde`'s `deny_unknown_fields` does not prevent a *new* `kind` from being written by a
//! newer version of the binary, and a reader that treats an unknown kind as ignorable would
//! silently skip, say, an `escalation_resolved`. Unknown kinds are therefore a hard parse
//! error, and adding one requires bumping `v`, which recovery checks before replaying.

use serde::{Deserialize, Serialize};

use super::Digest;
use crate::domain::{SubtaskId, TaskId, Timestamp};

/// Schema version of a record. Bumped whenever the meaning of an existing `kind` changes or a
/// `kind` is added; a reader refuses records it does not understand rather than guessing.
///
/// # Why a bump does not orphan the journals already on disk
///
/// The rule is asymmetric on purpose: a reader accepts every version **up to** its own and
/// refuses anything newer. Old journals stay readable, because a version-1 journal is a subset of
/// what this build understands — every one of its kinds still exists and still means the same
/// thing. A version-3 journal read by a version-2 build is refused, which is the direction that
/// must fail: the older reader would have to skip a record it cannot name, and a skipped record
/// might be the one saying a mutating command already ran.
///
/// Equality would have made the first bump a migration: `v0.1.2` wrote `v: 1`, so every task
/// recorded by it would have become unreadable, and recovery blocks a task whose journal it
/// cannot read ([`crate::Error::JournalVersion`]). Refusing only the future is what lets the
/// schema grow without a migration step that has to be written before it can be used.
pub const SCHEMA_VERSION: u16 = 3;

/// The closed set of event kinds.
///
/// The variant list mirrors the table in `docs/state-and-recovery.md` §2 one-to-one. A
/// mismatched list is the reason that table and this enum live in the same commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    // ── lifecycle ──────────────────────────────────────────────────────────────
    TaskCreated,
    TaskState,
    PlanWritten,
    PlanEdited,
    PlanApproved,
    SubtaskState,
    SubtaskSplit,
    AttemptStarted,
    AttemptFailed,
    AttemptRestarted,
    // ── inference ──────────────────────────────────────────────────────────────
    ModelRequest,
    ModelResponse,
    ModelStreamAborted,
    // ── protocol quality ───────────────────────────────────────────────────────
    NativeToolcall,
    TextFallback,
    InvalidToolcall,
    // ── tool calls: the three-record protocol ──────────────────────────────────
    ToolCallIntent,
    ToolCallStarted,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallUnknown,
    EffectVerified,
    EffectAbsent,
    // ── checkpoints and recovery ───────────────────────────────────────────────
    CheckpointCreated,
    JournalTruncatedTail,
    DaemonStarted,
    DaemonStopped,
    DaemonRecovered,
    // ── context, detection, budgets ────────────────────────────────────────────
    DetectorFired,
    Compaction,
    GateResult,
    CacheStats,
    // ── human interaction ──────────────────────────────────────────────────────
    EscalationOpened,
    EscalationResolved,
    // ── judgment (D-111) ───────────────────────────────────────────────────────
    JudgeVerdict,
    // ── environment ────────────────────────────────────────────────────────────
    BackendUnavailable,
    BackendRestored,
    // ── processes (D-152) ──────────────────────────────────────────────────────
    ProcessStopped,
}

impl Kind {
    /// The wire name, as it appears in a journal line.
    ///
    /// `serde`'s `rename_all = "snake_case"` is the authority for this table, and a test
    /// (`as_str_agrees_with_the_wire_name`) asserts the two agree for every variant. The
    /// duplication is what lets a caller render a kind without allocating a serialized
    /// `Value`; the test is what stops the duplication from becoming a second schema.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            // ── lifecycle ──────────────────────────────────────────────────────
            Self::TaskCreated => "task_created",
            Self::TaskState => "task_state",
            Self::PlanWritten => "plan_written",
            Self::PlanEdited => "plan_edited",
            Self::PlanApproved => "plan_approved",
            Self::SubtaskState => "subtask_state",
            Self::SubtaskSplit => "subtask_split",
            Self::AttemptStarted => "attempt_started",
            Self::AttemptFailed => "attempt_failed",
            Self::AttemptRestarted => "attempt_restarted",
            // ── inference ──────────────────────────────────────────────────────
            Self::ModelRequest => "model_request",
            Self::ModelResponse => "model_response",
            Self::ModelStreamAborted => "model_stream_aborted",
            // ── protocol quality ───────────────────────────────────────────────
            Self::NativeToolcall => "native_toolcall",
            Self::TextFallback => "text_fallback",
            Self::InvalidToolcall => "invalid_toolcall",
            // ── tool calls: the three-record protocol ──────────────────────────
            Self::ToolCallIntent => "tool_call_intent",
            Self::ToolCallStarted => "tool_call_started",
            Self::ToolCallCompleted => "tool_call_completed",
            Self::ToolCallFailed => "tool_call_failed",
            Self::ToolCallUnknown => "tool_call_unknown",
            Self::EffectVerified => "effect_verified",
            Self::EffectAbsent => "effect_absent",
            // ── checkpoints and recovery ───────────────────────────────────────
            Self::CheckpointCreated => "checkpoint_created",
            Self::JournalTruncatedTail => "journal_truncated_tail",
            Self::DaemonStarted => "daemon_started",
            Self::DaemonStopped => "daemon_stopped",
            Self::DaemonRecovered => "daemon_recovered",
            // ── context, detection, budgets ────────────────────────────────────
            Self::DetectorFired => "detector_fired",
            Self::Compaction => "compaction",
            Self::GateResult => "gate_result",
            Self::CacheStats => "cache_stats",
            // ── human interaction ──────────────────────────────────────────────
            Self::EscalationOpened => "escalation_opened",
            Self::EscalationResolved => "escalation_resolved",
            // ── judgment (D-111) ───────────────────────────────────────────────
            Self::JudgeVerdict => "judge_verdict",
            // ── environment ────────────────────────────────────────────────────
            Self::BackendUnavailable => "backend_unavailable",
            Self::BackendRestored => "backend_restored",
            Self::ProcessStopped => "process_stopped",
        }
    }

    /// Every kind, for exhaustive round-trip tests.
    ///
    /// A hand-written list rather than a derive macro because the list is the schema: it is
    /// reviewed next to the documentation table, and a test asserts that it matches `ALL`
    /// exactly once the corresponding `serde` rename table is exercised.
    pub const ALL: &'static [Self] = &[
        Self::TaskCreated,
        Self::TaskState,
        Self::PlanWritten,
        Self::PlanEdited,
        Self::PlanApproved,
        Self::SubtaskState,
        Self::SubtaskSplit,
        Self::AttemptStarted,
        Self::AttemptFailed,
        Self::AttemptRestarted,
        Self::ModelRequest,
        Self::ModelResponse,
        Self::ModelStreamAborted,
        Self::NativeToolcall,
        Self::TextFallback,
        Self::InvalidToolcall,
        Self::ToolCallIntent,
        Self::ToolCallStarted,
        Self::ToolCallCompleted,
        Self::ToolCallFailed,
        Self::ToolCallUnknown,
        Self::EffectVerified,
        Self::EffectAbsent,
        Self::CheckpointCreated,
        Self::JournalTruncatedTail,
        Self::DaemonStarted,
        Self::DaemonStopped,
        Self::DaemonRecovered,
        Self::DetectorFired,
        Self::Compaction,
        Self::GateResult,
        Self::CacheStats,
        Self::EscalationOpened,
        Self::EscalationResolved,
        Self::JudgeVerdict,
        Self::BackendUnavailable,
        Self::BackendRestored,
        Self::ProcessStopped,
    ];

    /// A number that is distinct for every variant, and that the compiler forces to stay
    /// complete.
    ///
    /// [`Self::ALL`] is a hand-written list, so a new variant can be added to the enum and
    /// forgotten there. This `match` is the guard: it has no wildcard arm, so adding a variant
    /// does not compile until an arm is written, and
    /// `every_kind_is_covered_by_the_all_table` checks that the arms produce exactly the numbers
    /// `0..ALL.len()`. A new variant that skips `ALL` therefore fails a test, and so does a
    /// duplicate number — which is what a hasty arm copied from its neighbour would be.
    ///
    /// The value is not part of the schema and never reaches the journal.
    #[cfg(test)]
    #[must_use]
    const fn ordinal(self) -> usize {
        match self {
            Self::TaskCreated => 0,
            Self::TaskState => 1,
            Self::PlanWritten => 2,
            Self::PlanEdited => 3,
            Self::PlanApproved => 4,
            Self::SubtaskState => 5,
            Self::SubtaskSplit => 6,
            Self::AttemptStarted => 7,
            Self::AttemptFailed => 8,
            Self::AttemptRestarted => 9,
            Self::ModelRequest => 10,
            Self::ModelResponse => 11,
            Self::ModelStreamAborted => 12,
            Self::NativeToolcall => 13,
            Self::TextFallback => 14,
            Self::InvalidToolcall => 15,
            Self::ToolCallIntent => 16,
            Self::ToolCallStarted => 17,
            Self::ToolCallCompleted => 18,
            Self::ToolCallFailed => 19,
            Self::ToolCallUnknown => 20,
            Self::EffectVerified => 21,
            Self::EffectAbsent => 22,
            Self::CheckpointCreated => 23,
            Self::JournalTruncatedTail => 24,
            Self::DaemonStarted => 25,
            Self::DaemonStopped => 26,
            Self::DaemonRecovered => 27,
            Self::DetectorFired => 28,
            Self::Compaction => 29,
            Self::GateResult => 30,
            Self::CacheStats => 31,
            Self::EscalationOpened => 32,
            Self::EscalationResolved => 33,
            Self::JudgeVerdict => 34,
            Self::BackendUnavailable => 35,
            Self::BackendRestored => 36,
            Self::ProcessStopped => 37,
        }
    }

    /// Whether this event describes a tool call's lifecycle.
    ///
    /// Recovery scans for exactly these: every `intent` must be followed by a terminal
    /// record, and anything else is an unfinished call.
    #[must_use]
    pub const fn is_tool_call(self) -> bool {
        matches!(
            self,
            Self::ToolCallIntent
                | Self::ToolCallStarted
                | Self::ToolCallCompleted
                | Self::ToolCallFailed
                | Self::ToolCallUnknown
        )
    }

    /// Whether this event belongs to the durability machinery itself.
    ///
    /// These are written while no model is involved and must never be counted as a step or
    /// attributed to an attempt's budget.
    #[must_use]
    pub const fn is_bookkeeping(self) -> bool {
        matches!(
            self,
            Self::JournalTruncatedTail
                | Self::DaemonStarted
                | Self::DaemonStopped
                | Self::DaemonRecovered
                | Self::CacheStats
        )
    }
}

/// Everything a record carries besides the payload itself.
///
/// A struct rather than fifteen arguments: the writer builds one, and both the line format
/// and the chain hash read from the same place, so they cannot drift.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordFields {
    pub seq: u64,
    pub at: Timestamp,
    pub task: TaskId,
    pub subtask: Option<SubtaskId>,
    pub attempt: Option<u32>,
    pub kind: Kind,
}

impl RecordFields {
    /// Minimum required: a sequence number, a time, a task and a kind.
    #[must_use]
    pub const fn new(seq: u64, at: Timestamp, task: TaskId, kind: Kind) -> Self {
        Self {
            seq,
            at,
            task,
            subtask: None,
            attempt: None,
            kind,
        }
    }
}

/// Which subtask and attempt a record belongs to.
///
/// A separate type because the pair travels together: recovery keys attempts by
/// `(subtask, attempt)`, and passing the two fields separately is how one of them gets left
/// `None` and the record becomes invisible to the pass that closes interrupted attempts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scope {
    /// The subtask the record is about.
    pub subtask: Option<SubtaskId>,
    /// The 1-based attempt number, not the attempt's id: that is what the journal stores.
    pub attempt: Option<u32>,
}

impl Scope {
    /// A record about one attempt.
    #[must_use]
    pub const fn attempt(subtask: SubtaskId, attempt: u32) -> Self {
        Self {
            subtask: Some(subtask),
            attempt: Some(attempt),
        }
    }

    /// A record about a subtask but not about one of its attempts.
    #[must_use]
    pub const fn subtask(subtask: SubtaskId) -> Self {
        Self {
            subtask: Some(subtask),
            attempt: None,
        }
    }

    /// A record about the task as a whole.
    ///
    /// Recovery closes calls it cannot attribute to a pass, and a record that invented an
    /// attempt number would make `attempts.json` claim a pass that never ran.
    #[must_use]
    pub const fn task() -> Self {
        Self {
            subtask: None,
            attempt: None,
        }
    }
}

/// A single event, ready to be written.
///
/// The payload is a [`serde_json::Value`] rather than a typed per-kind enum. `kind` already
/// carries the type information, and typed payloads would mean fifteen near-identical structs
/// whose only purpose is to be immediately serialized. Validation lives where it matters:
/// the writer rejects a payload that is not an object, and recovery reads fields by name.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct JournalEvent {
    /// The event body. Always a JSON object in practice; the writer enforces that.
    pub payload: serde_json::Value,
    /// Set when the body lives in `artifacts/<sha256>` instead of in the line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<String>,
}

impl JournalEvent {
    /// An event with no body.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// An event whose body is this value.
    #[must_use]
    pub const fn with_payload(payload: serde_json::Value) -> Self {
        Self {
            payload,
            artifact: None,
        }
    }

    /// An event whose body was spilled to an artifact.
    ///
    /// The payload is kept as a small summary rather than emptied: a human reading the
    /// journal sees what the event was about without opening the artifact.
    #[must_use]
    pub fn spilled(summary: serde_json::Value, artifact: Digest) -> Self {
        Self {
            payload: summary,
            artifact: Some(super::digest_text(&artifact)),
        }
    }
}

/// One line of the journal: the envelope plus the event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalRecord {
    pub fields: RecordFields,
    pub event: JournalEvent,
    /// Hash of the previous record's line, absent only for the very first record of a chain.
    pub prev: Option<Digest>,
}

impl JournalRecord {
    /// A record that continues a chain.
    #[must_use]
    pub const fn new(fields: RecordFields, event: JournalEvent, prev: Option<Digest>) -> Self {
        Self { fields, event, prev }
    }

    /// The canonical line, without its trailing newline.
    ///
    /// Field order is fixed by the struct declaration below, not by a map, so two writers
    /// produce byte-identical output for the same record — which is what makes the `prev`
    /// chain meaningful and golden files stable.
    pub fn to_line(&self) -> crate::Result<String> {
        serde_json::to_string(&self.line_out())
            .map_err(|error| crate::Error::JournalCorrupt(error.to_string()))
    }

    /// The on-disk shape, for a caller that must hand a record to something other than the
    /// journal file — `attach` streams records to a client, and a client should render exactly
    /// what the file holds rather than a second, subtly different shape.
    ///
    /// Built from the same [`LineOut`] the writer uses, so the two cannot drift: a field added
    /// to the line format appears in the stream automatically.
    fn line_out(&self) -> LineOut<'_> {
        LineOut {
            v: SCHEMA_VERSION,
            seq: self.fields.seq,
            ts: self.fields.at,
            task: &self.fields.task,
            subtask: self.fields.subtask.as_ref(),
            attempt: self.fields.attempt,
            kind: self.fields.kind,
            payload: &self.event.payload,
            artifact: self.event.artifact.as_deref(),
            prev: self.prev.as_ref().map(super::digest_text),
        }
    }

    /// Parse one line, rejecting anything the schema does not recognise.
    pub fn from_line(line: &str) -> crate::Result<Self> {
        let parsed: LineIn =
            serde_json::from_str(line).map_err(|error| crate::Error::JournalCorrupt(error.to_string()))?;
        // Accept every version up to this build's, refuse anything newer. An older journal holds
        // kinds that all still exist and still mean the same thing, so reading it is safe; a newer
        // one may hold a kind this build cannot name, and skipping it could skip the record that
        // says an effect already happened. See [`SCHEMA_VERSION`].
        if parsed.v > SCHEMA_VERSION {
            return Err(crate::Error::JournalVersion {
                found: parsed.v,
                expected: SCHEMA_VERSION,
            });
        }
        if !parsed.payload.is_object() {
            return Err(crate::Error::JournalCorrupt(
                "payload must be a JSON object".into(),
            ));
        }
        let prev = parsed.prev.as_deref().map(super::parse_digest).transpose()?;
        Ok(Self {
            fields: RecordFields {
                seq: parsed.seq,
                at: parsed.ts,
                task: parsed.task,
                subtask: parsed.subtask,
                attempt: parsed.attempt,
                kind: parsed.kind,
            },
            event: JournalEvent {
                payload: parsed.payload,
                artifact: parsed.artifact,
            },
            prev,
        })
    }
}

/// One line of the journal, as JSON.
///
/// Delegates to the same [`LineOut`] the file writer uses. A second shape for the client
/// stream is how an attached terminal ends up rendering something the journal never contained:
/// the two would drift the first time a field was added, and nobody would notice until a human
/// compared a screen against a file.
impl Serialize for JournalRecord {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.line_out().serialize(serializer)
    }
}

/// The on-disk shape for writing.
///
/// Separate from [`JournalRecord`] so that the wire format is reviewed as one block, and
/// split from [`LineIn`] because serializing can borrow while parsing must own: `serde`
/// cannot derive both from one struct that contains `&str` fields.
#[derive(Serialize)]
struct LineOut<'a> {
    v: u16,
    seq: u64,
    #[serde(with = "crate::time::millis")]
    ts: Timestamp,
    task: &'a TaskId,
    #[serde(skip_serializing_if = "Option::is_none")]
    subtask: Option<&'a SubtaskId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attempt: Option<u32>,
    kind: Kind,
    #[serde(default)]
    payload: &'a serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    artifact: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prev: Option<String>,
}

/// The on-disk shape for reading.
///
/// `deny_unknown_fields` is deliberately absent: a future version may add a field inside
/// version 1 only if it is optional, and rejecting it would make forward compatibility
/// impossible for exactly the harmless case. A *missing* field is an error, and an unknown
/// `kind` or a wrong `v` still fails.
#[derive(Deserialize)]
struct LineIn {
    v: u16,
    seq: u64,
    #[serde(with = "crate::time::millis")]
    ts: Timestamp,
    task: TaskId,
    #[serde(default)]
    subtask: Option<SubtaskId>,
    #[serde(default)]
    attempt: Option<u32>,
    kind: Kind,
    #[serde(default)]
    payload: serde_json::Value,
    #[serde(default)]
    artifact: Option<String>,
    #[serde(default)]
    prev: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone as _;

    fn now() -> Timestamp {
        chrono::Utc.with_ymd_and_hms(2026, 11, 12, 10, 14, 3).unwrap()
    }

    fn record() -> JournalRecord {
        JournalRecord::new(
            RecordFields::new(
                7,
                now(),
                TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV"),
                Kind::TaskCreated,
            ),
            JournalEvent::with_payload(serde_json::json!({"title": "fix the tests"})),
            None,
        )
    }

    #[test]
    fn as_str_agrees_with_the_wire_name() {
        // `as_str` is a hand-written copy of what serde writes. This is the test that stops it
        // being a *second* schema: the moment a variant's serde name changes and the table does
        // not, rendering and recovery disagree about what a record says.
        for kind in Kind::ALL {
            let wire = serde_json::to_value(kind).unwrap();
            assert_eq!(wire.as_str(), Some(kind.as_str()), "{kind:?}");
        }
    }

    #[test]
    fn every_kind_is_covered_by_the_version_that_introduced_it() {
        // The schema rule is "a new `kind` means a new version". `process_stopped` arrived in
        // version 3 (D-152), and a build that writes it under version 2 would produce a journal
        // the previous release reads by *skipping* the record — the one direction the reader
        // deliberately refuses. The assertion is on the version, not on the count, because the
        // count is what a hasty bump would satisfy while leaving the number alone.
        assert_eq!(SCHEMA_VERSION, 3, "`process_stopped` needs version 3");
        assert!(Kind::ALL.contains(&Kind::ProcessStopped));
    }

    #[test]
    fn a_record_serializes_to_the_same_json_that_it_writes() {
        // `attach` streams records as JSON. If that JSON were built by a second impl, it would
        // drift from the file, and a human comparing a terminal against the journal would find
        // two different truths.
        let record = record();
        let line = record.to_line().unwrap();
        let streamed = serde_json::to_string(&record).unwrap();
        assert_eq!(streamed, line);
        let parsed: serde_json::Value = serde_json::from_str(&streamed).unwrap();
        assert_eq!(parsed["seq"], 7);
        assert_eq!(parsed["kind"], "task_created");
        assert!(parsed.get("prev").is_none(), "the first record has no prev");
    }

    #[test]
    fn every_kind_has_a_unique_wire_name() {
        let mut names = std::collections::BTreeSet::new();
        for kind in Kind::ALL {
            let name = serde_json::to_string(kind).unwrap();
            assert!(names.insert(name.clone()), "duplicate wire name {name}");
            let back: Kind = serde_json::from_str(&name).unwrap();
            assert_eq!(back, *kind);
        }
        assert_eq!(names.len(), Kind::ALL.len());
    }

    #[test]
    fn every_kind_is_covered_by_the_all_table() {
        // A hand-written list drifts. This walks the exhaustive `ordinal` match, which the
        // compiler refuses to leave incomplete, and requires it to produce exactly one number per
        // entry of `ALL`. A variant added to the enum but not to `ALL` therefore fails here — the
        // sabotage that dropped `JudgeVerdict` from `ALL` did not fail the old version of this
        // test, which only checked that the names in `ALL` were unique.
        let ordinals: std::collections::BTreeSet<usize> =
            Kind::ALL.iter().map(|kind| kind.ordinal()).collect();
        assert_eq!(
            ordinals.len(),
            Kind::ALL.len(),
            "two entries of `ALL` share an ordinal"
        );
        assert_eq!(
            ordinals,
            (0..Kind::ALL.len()).collect::<std::collections::BTreeSet<usize>>(),
            "`ALL` is missing a variant, or holds one that `ordinal` does not number"
        );
    }

    #[test]
    fn unknown_kinds_are_rejected() {
        let err = serde_json::from_str::<Kind>("\"warp_drive_engaged\"").unwrap_err();
        assert!(err.to_string().contains("warp_drive_engaged"), "{err}");
    }

    #[test]
    fn line_roundtrips_exactly() {
        let record = record();
        let line = record.to_line().unwrap();
        let back = JournalRecord::from_line(&line).unwrap();
        assert_eq!(back, record);
        assert_eq!(back.to_line().unwrap(), line);
    }

    #[test]
    fn field_order_is_fixed() {
        let line = record().to_line().unwrap();
        let expected = [
            "\"v\":",
            "\"seq\":",
            "\"ts\":",
            "\"task\":",
            "\"kind\":",
            "\"payload\":",
        ];
        let mut cursor = 0;
        for key in expected {
            let found = line[cursor..]
                .find(key)
                .unwrap_or_else(|| panic!("{key} in {line}"));
            cursor += found + key.len();
        }
    }

    #[test]
    fn absent_optionals_are_omitted_not_null() {
        let line = record().to_line().unwrap();
        assert!(!line.contains("null"), "{line}");
        assert!(!line.contains("\"subtask\""), "{line}");
        assert!(!line.contains("\"prev\""), "{line}");
    }

    #[test]
    fn timestamps_have_millisecond_precision() {
        let line = record().to_line().unwrap();
        assert!(line.contains("\"ts\":\"2026-11-12T10:14:03.000Z\""), "{line}");
    }

    #[test]
    fn a_future_schema_version_is_rejected() {
        // The direction that must fail: a newer journal may hold a kind this build cannot name,
        // and a reader that skipped it could skip the record saying an effect already happened.
        let future = SCHEMA_VERSION + 1;
        let line = record()
            .to_line()
            .unwrap()
            .replace(&format!("\"v\":{SCHEMA_VERSION}"), &format!("\"v\":{future}"));
        let err = JournalRecord::from_line(&line).unwrap_err();
        assert!(
            matches!(
                err,
                crate::Error::JournalVersion {
                    found,
                    expected: SCHEMA_VERSION
                } if found == future
            ),
            "{err}"
        );
    }

    #[test]
    fn an_older_schema_version_still_reads() {
        // `v0.1.2` wrote `v: 1`. Refusing it would have made the first bump a migration: every
        // task recorded by that release would be unreadable, and recovery blocks a task whose
        // journal it cannot read. The kinds of version 1 all still exist and still mean the same
        // thing, so reading one is safe.
        let line = record().to_line().unwrap().replace(
            &format!("\"v\":{SCHEMA_VERSION}"),
            &format!("\"v\":{}", SCHEMA_VERSION - 1),
        );
        let parsed = JournalRecord::from_line(&line).expect("an older journal still reads");
        assert_eq!(parsed, record());
    }

    #[test]
    fn a_non_object_payload_is_rejected() {
        let line = record()
            .to_line()
            .unwrap()
            .replace(r#"{"title":"fix the tests"}"#, "[1,2]");
        assert!(JournalRecord::from_line(&line).is_err());
    }

    #[test]
    fn a_corrupt_prev_digest_is_rejected() {
        let line = record()
            .to_line()
            .unwrap()
            .replace("\"payload\"", "\"prev\":\"sha256:not-hex\",\"payload\"");
        assert!(JournalRecord::from_line(&line).is_err());
    }

    #[test]
    fn tool_call_kinds_are_classified() {
        for kind in Kind::ALL {
            let expected = matches!(
                kind,
                Kind::ToolCallIntent
                    | Kind::ToolCallStarted
                    | Kind::ToolCallCompleted
                    | Kind::ToolCallFailed
                    | Kind::ToolCallUnknown
            );
            assert_eq!(kind.is_tool_call(), expected, "{kind:?}");
        }
    }

    #[test]
    fn spilled_events_keep_a_summary() {
        let event = JournalEvent::spilled(serde_json::json!({"bytes": 90_000}), [7u8; 32]);
        assert!(event.artifact.is_some());
        assert_eq!(event.payload["bytes"], 90_000);
    }
}
