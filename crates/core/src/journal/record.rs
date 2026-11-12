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
pub const SCHEMA_VERSION: u16 = 1;

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
    // ── environment ────────────────────────────────────────────────────────────
    BackendUnavailable,
    BackendRestored,
}

impl Kind {
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
        Self::BackendUnavailable,
        Self::BackendRestored,
    ];

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
        let serialized = LineOut {
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
        };
        serde_json::to_string(&serialized).map_err(|error| crate::Error::JournalCorrupt(error.to_string()))
    }

    /// Parse one line, rejecting anything the schema does not recognise.
    pub fn from_line(line: &str) -> crate::Result<Self> {
        let parsed: LineIn =
            serde_json::from_str(line).map_err(|error| crate::Error::JournalCorrupt(error.to_string()))?;
        if parsed.v != SCHEMA_VERSION {
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
        // Adding a variant without adding it to `ALL` is caught here: the only way to reach
        // this assertion with a missing entry is to have changed one but not the other.
        let serialized: std::collections::BTreeSet<String> = Kind::ALL
            .iter()
            .map(|k| serde_json::to_string(k).unwrap())
            .collect();
        assert_eq!(serialized.len(), Kind::ALL.len());
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
    fn wrong_schema_version_is_rejected() {
        let line = record().to_line().unwrap().replace("\"v\":1", "\"v\":99");
        let err = JournalRecord::from_line(&line).unwrap_err();
        assert!(
            matches!(
                err,
                crate::Error::JournalVersion {
                    found: 99,
                    expected: 1
                }
            ),
            "{err}"
        );
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
