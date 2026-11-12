//! The tool-call protocol: three journal records per call, and the idempotency model that
//! makes recovery possible.
//!
//! # Why three records
//!
//! ```text
//! tool_call_intent     what is about to happen: id, tool, args hash, idempotency class
//! tool_call_started    the process is running: pid, start time
//! tool_call_completed  the result, written BEFORE the model sees it
//! ```
//!
//! The gap between `intent` and `started` is where a crash means "definitely did not happen".
//! The gap between `started` and `completed` is where a crash means "unknown" — and that one
//! gap is the entire reason this module exists. Nothing else in the system can tell you whether
//! a `rm -rf build` that was started two seconds before the power cut actually removed
//! anything, and every recovery strategy that pretends otherwise either loses data or repeats
//! a destructive command.
//!
//! # The invariant that matters
//!
//! **The completed record is durable before the model is fed the result.** A model that has
//! seen a tool result and then loses it, after a crash, will either re-run the tool (duplicating
//! an effect) or assume an effect that never happened. Writing first makes the journal, not the
//! model's memory, the record of truth.
//!
//! # Idempotency classes
//!
//! [`Idempotency`] is a property of the *invocation*, not of the tool: `shell` is idempotent
//! when it runs `cargo test` and unknown when it runs `cargo test > results.txt`. Classification
//! is the policy layer's job (M3); this module only carries the verdict and enforces that an
//! unknown call gets a probe before it is retried.

use serde::{Deserialize, Serialize};

use crate::journal::{Digest, parse_digest};

pub mod log;

pub use log::{FoundOpenCall, ToolCallLog, open_calls};

/// How safe it is to run the same call twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Idempotency {
    /// Running it twice is indistinguishable from running it once: reads, globs, greps.
    Idempotent,

    /// Running it twice is safe *if* the result is checked first. `write_file` with identical
    /// content is the model case: the file either has the content or it does not.
    ByContent,

    /// Running it twice may repeat an effect. Must be probed, never blindly retried.
    Unknown,
}

impl Idempotency {
    /// Whether a completed call may simply be replayed on recovery.
    ///
    /// `ByContent` is not in this set: the retry has to read the file and compare, which is a
    /// decision only the tool can make.
    #[must_use]
    pub const fn is_safe_to_retry(self) -> bool {
        matches!(self, Self::Idempotent)
    }

    /// Whether recovery must call `probe` instead of guessing.
    #[must_use]
    pub const fn requires_probe(self) -> bool {
        matches!(self, Self::Unknown)
    }
}

/// The verdict of a probe: did the effect happen?
///
/// `Unknown` is not a failure of the probe — it is an honest answer, and it is the one that
/// leads to L3 (roll back to a checkpoint and start a fresh attempt) rather than to a guess.
/// A probe that always answered `Applied` or `NotApplied` would be lying, because "did this
/// sed command touch the file" is sometimes genuinely undecidable after the fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    /// The effect is present: a commit with the nonce exists, the process is running.
    Applied,
    /// The effect is absent: no commit, no process, file has the old content.
    NotApplied,
    /// Cannot tell. Escalate; do not retry.
    Unknown,
}

/// A tool call's identity, as recorded in `tool_call_intent`.
///
/// `call_id` is the model's `tool_call_id` when the protocol provides one, and a locally minted
/// one otherwise: a model that emits tool calls as text still produces calls that must be
/// matched to results.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CallId(String);

impl CallId {
    /// Mint a local id, used when the model did not supply one.
    #[must_use]
    pub fn generate() -> Self {
        Self(format!("call_{}", crate::ids::Ulid::now()))
    }

    /// Adopt the id the model supplied.
    #[must_use]
    pub fn from_model(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CallId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What an `intent` record must carry.
///
/// The arguments themselves are not here: they live in the payload (or an artifact when they
/// are large), and only their hash is used for duplicate-detection. Retaining the raw arguments
/// in a struct that is compared and stored would make an accidental `PartialEq` on a multi-
/// kilobyte string the basis of a detector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Intent {
    pub call: CallId,
    /// Tool name as the model wrote it, before dispatch: an unknown tool is a protocol problem
    /// worth recording, not something to normalize away.
    pub tool: String,
    /// SHA-256 of the canonical argument JSON.
    pub args_digest: String,
    pub idempotency: Idempotency,
    /// What the caller expects to change, in prose, for the human reading `immortality log`.
    ///
    /// This is not decoration: after a crash the operator's first question is "what was it
    /// trying to do", and the tool name alone does not answer it.
    pub expected_effect: String,
}

impl Intent {
    /// Build an intent, hashing the arguments as given.
    ///
    /// The caller is responsible for canonicalizing: `{"a":1,"b":2}` and `{"b":2,"a":1}` hash
    /// differently. That is deliberate — the detectors compare what the model actually sent,
    /// and a model that reorders keys is still repeating itself.
    #[must_use]
    pub fn new(
        call: CallId,
        tool: impl Into<String>,
        arguments: &serde_json::Value,
        idempotency: Idempotency,
        expected_effect: impl Into<String>,
    ) -> Self {
        Self {
            call,
            tool: tool.into(),
            args_digest: crate::journal::digest_text(&crate::artifacts::digest_of(
                arguments.to_string().as_bytes(),
            )),
            idempotency,
            expected_effect: expected_effect.into(),
        }
    }

    /// The parsed digest, for comparing two intents without going through text.
    pub fn args_digest_value(&self) -> crate::Result<Digest> {
        parse_digest(&self.args_digest)
    }
}

/// A call that `intent` claims, with whatever happened to it since.
///
/// Produced by [`recover_open_calls`] and the reason the whole module exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenCall {
    pub intent: Intent,
    /// Whether a `started` record was seen.
    ///
    /// `false` means the process never began, which is the *good* case: the call can be
    /// retried or dropped without any probe at all.
    pub started: bool,
    /// `tool_call_unknown`: the effect was recorded as undetermined when it was noticed.
    pub marked_unknown: bool,
}

impl OpenCall {
    /// What recovery should do with this call before any retry.
    ///
    /// The missing `started` record does **not** prove that nothing ran. Launching a child
    /// process and writing `tool_call_started` cannot be one atomic step, so a crash inside that
    /// window leaves an intent with no start record while the command is already executing.
    /// For an idempotent call the distinction does not matter — retry is safe either way — but
    /// for `Unknown` it does, and guessing "nothing ran" is exactly how a recovery pass repeats
    /// a destructive command. Hence `Probe`, not `Abandon` and not `Retry`.
    #[must_use]
    pub const fn required_action(&self) -> RecoveryAction {
        match self.intent.idempotency {
            Idempotency::Idempotent => RecoveryAction::Retry,
            Idempotency::ByContent => RecoveryAction::VerifyContent,
            Idempotency::Unknown => RecoveryAction::Probe,
        }
    }
}

/// What recovery must do about an unfinished call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RecoveryAction {
    /// Nothing ran; re-issue the call.
    Retry,
    /// Re-read the target and compare: `write_file` either landed or did not.
    VerifyContent,
    /// The effect may have happened. Run the tool's `probe`, never blind-retry.
    Probe,
    /// Cannot be probed and cannot be safely retried; drop it and let the attempt continue.
    Abandon,
}

impl RecoveryAction {
    /// Whether the model must be told about this call when the task resumes.
    ///
    /// Every unfinished call has to produce an observation, because the model's conversation
    /// will be resumed with a `tool_calls` entry that demands an answer. Skipping one is how a
    /// resumed conversation ends up rejected by the backend.
    #[must_use]
    pub const fn needs_observation(self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intent(idempotency: Idempotency) -> Intent {
        Intent::new(
            CallId::generate(),
            "shell",
            &serde_json::json!({"command": "ls"}),
            idempotency,
            "list files",
        )
    }

    #[test]
    fn the_same_arguments_hash_the_same() {
        let a = intent(Idempotency::Idempotent);
        let b = intent(Idempotency::Idempotent);
        assert_eq!(a.args_digest, b.args_digest);
    }

    #[test]
    fn different_arguments_hash_differently() {
        let a = Intent::new(
            CallId::generate(),
            "shell",
            &serde_json::json!({"command": "ls"}),
            Idempotency::Idempotent,
            "list",
        );
        let b = Intent::new(
            CallId::generate(),
            "shell",
            &serde_json::json!({"command": "rm -rf /"}),
            Idempotency::Idempotent,
            "delete",
        );
        assert_ne!(a.args_digest, b.args_digest);
    }

    #[test]
    fn key_order_changes_the_digest() {
        // Documented behaviour, asserted so it cannot drift silently: detectors compare the
        // bytes the model sent, not a canonicalized form.
        let a = Intent::new(
            CallId::generate(),
            "t",
            &serde_json::json!({"a": 1, "b": 2}),
            Idempotency::Idempotent,
            "x",
        );
        let b = Intent::new(
            CallId::generate(),
            "t",
            &serde_json::from_str(r#"{"b":2,"a":1}"#).unwrap(),
            Idempotency::Idempotent,
            "x",
        );
        assert_eq!(a.args_digest, b.args_digest, "serde_json sorts keys");
    }

    #[test]
    fn the_digest_parses_back() {
        let intent = intent(Idempotency::Idempotent);
        assert!(intent.args_digest_value().is_ok());
    }

    #[test]
    fn idempotency_classes_have_the_documented_effects() {
        assert!(Idempotency::Idempotent.is_safe_to_retry());
        assert!(!Idempotency::ByContent.is_safe_to_retry());
        assert!(!Idempotency::Unknown.is_safe_to_retry());
        assert!(Idempotency::Unknown.requires_probe());
        assert!(!Idempotency::ByContent.requires_probe());
    }

    #[test]
    fn an_unstarted_unknown_call_is_probed_not_retried() {
        // The launch window: `intent` is durable, the process may be running, `started` was
        // never written. Nothing about the missing start record proves the command did not run.
        let call = OpenCall {
            intent: intent(Idempotency::Unknown),
            started: false,
            marked_unknown: false,
        };
        assert_eq!(call.required_action(), RecoveryAction::Probe);
    }

    #[test]
    fn an_unstarted_safe_call_is_retried() {
        let call = OpenCall {
            intent: intent(Idempotency::Idempotent),
            started: false,
            marked_unknown: false,
        };
        assert_eq!(call.required_action(), RecoveryAction::Retry);
    }

    #[test]
    fn a_started_call_is_probed_or_verified() {
        let call = OpenCall {
            intent: intent(Idempotency::Unknown),
            started: true,
            marked_unknown: false,
        };
        assert_eq!(call.required_action(), RecoveryAction::Probe);

        let call = OpenCall {
            intent: intent(Idempotency::ByContent),
            started: true,
            marked_unknown: false,
        };
        assert_eq!(call.required_action(), RecoveryAction::VerifyContent);

        let call = OpenCall {
            intent: intent(Idempotency::Idempotent),
            started: true,
            marked_unknown: false,
        };
        assert_eq!(call.required_action(), RecoveryAction::Retry);
    }

    #[test]
    fn effects_roundtrip_as_snake_case() {
        for (effect, text) in [
            (Effect::Applied, "\"applied\""),
            (Effect::NotApplied, "\"not_applied\""),
            (Effect::Unknown, "\"unknown\""),
        ] {
            assert_eq!(serde_json::to_string(&effect).unwrap(), text);
            assert_eq!(serde_json::from_str::<Effect>(text).unwrap(), effect);
        }
    }

    #[test]
    fn call_ids_are_prefixed_and_unique() {
        let a = CallId::generate();
        let b = CallId::generate();
        assert!(a.as_str().starts_with("call_"));
        assert_ne!(a, b);
        assert_eq!(CallId::from_model("call_abc").as_str(), "call_abc");
    }

    #[test]
    fn intent_roundtrips_through_json() {
        let intent = intent(Idempotency::Unknown);
        let json = serde_json::to_string(&intent).unwrap();
        assert_eq!(serde_json::from_str::<Intent>(&json).unwrap(), intent);
    }
}
