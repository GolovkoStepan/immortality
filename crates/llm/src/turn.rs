//! The assistant's answer as the engine sees it.
//!
//! # Why this is separate from the wire types
//!
//! [`crate::wire`] mirrors the `OpenAI` JSON, which means it carries fields the engine never
//! looks at (`object`, `system_fingerprint`, `logprobs`) and lacks fields it needs (a defaulted
//! `finish_reason`, a `Usage` that folds Strata's `timings.cache_n` in). This module is the
//! normalised form: one turn, whether it arrived as one JSON body or as four hundred SSE chunks.
//!
//! # Reasoning is not content
//!
//! The live backend emits `reasoning_content` as a separate field, and it is kept separate
//! here. Concatenating it into `content` would poison the history sent back: the model would
//! see its own thinking as if it had said it, and the next turn would treat the reasoning as a
//! committed statement. It is also how the thinking-truncation failure is detected, which needs
//! the two fields to be distinguishable.

use serde::{Deserialize, Serialize};

/// A finished assistant turn.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantTurn {
    /// What the model said. May be empty when the turn is tool calls only.
    pub content: String,
    /// `reasoning_content`, kept out of [`Self::content`] deliberately.
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    /// Normalised to `"stop"` when the backend omitted it.
    pub finish_reason: String,
    pub usage: Option<Usage>,
}

impl AssistantTurn {
    /// Whether the turn asks for tools.
    #[must_use]
    pub const fn is_tool_call(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// The failure mode measured on the live backend: the model spent the whole budget
    /// thinking and never produced an answer. Must not be treated as "done".
    ///
    /// The three conditions are all load-bearing. `finish_reason == "length"` alone is a normal
    /// truncation of a long answer; empty content alone is normal for a tool-call turn. It is
    /// the combination with nothing usable produced that wastes the step.
    #[must_use]
    pub fn is_thinking_truncated(&self) -> bool {
        self.finish_reason == FINISH_LENGTH && self.content.trim().is_empty() && self.tool_calls.is_empty()
    }

    /// Whether the backend cut the answer off at the token limit.
    #[must_use]
    pub fn is_truncated(&self) -> bool {
        self.finish_reason == FINISH_LENGTH
    }

    /// Whether the turn carries anything the engine can act on.
    ///
    /// An empty turn with `finish_reason: stop` is a model failure, not a finished task: there
    /// is nothing to show a human and nothing to execute, so accepting it would end the attempt
    /// successfully with no work done.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.content.trim().is_empty() && self.tool_calls.is_empty()
    }
}

/// `finish_reason` value for a length cut-off. Kept as a constant because the string is
/// compared in three places and a typo would silently disable the truncation checks.
pub const FINISH_LENGTH: &str = "length";

/// One requested tool call, exactly as the model produced it.
///
/// Created with a valid id in every path that produces it (a synthesized `call_<index>` for a
/// model that omitted one), because the id is the key the journal uses to match intent with
/// completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    /// Always `"function"`; modelled because the wire format requires the field.
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: CalledFunction,
}

fn function_type() -> String {
    "function".to_string()
}

impl ToolCall {
    /// A function call, with the wire's `type: "function"` filled in.
    #[must_use]
    pub fn function(id: impl Into<String>, name: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind: function_type(),
            function: CalledFunction {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalledFunction {
    pub name: String,
    /// Raw JSON string, exactly as the model produced it.
    ///
    /// Not parsed here on purpose: validation against the tool's schema is the tool layer's
    /// job, and it must reject malformed JSON with a structured error rather than substitute
    /// `{}` and execute a call with no arguments.
    pub arguments: String,
}

impl CalledFunction {
    /// Parse the arguments, reporting the tool name in the error.
    pub fn parse(&self) -> serde_json::Result<serde_json::Value> {
        serde_json::from_str(&self.arguments)
    }
}

/// Token accounting for one turn.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
    /// Tokens served from the prefix cache. Strata reports this under `timings.cache_n`;
    /// a miss on a grown prefix is the difference between 1 s and 90 s per step.
    #[serde(default)]
    pub cache_n: u32,
}

impl Usage {
    /// The prompt share that hit the cache, or `None` when nothing was cached and the ratio
    /// would be misleading.
    ///
    /// This is the number `doctor` prints and the one worth logging every step: prefix
    /// discipline is invisible without it, and a regression costs 400× per step.
    #[must_use]
    pub fn cache_hit_ratio(&self) -> Option<f64> {
        if self.prompt_tokens == 0 || self.cache_n == 0 {
            return None;
        }
        Some(f64::from(self.cache_n) / f64::from(self.prompt_tokens))
    }

    /// Whether the backend reported nothing at all.
    ///
    /// `cache_n` counts, deliberately: a usage block that carries only a cache count comes from
    /// a streamed `timings`-only server, and dropping it would make every report claim a 0 %
    /// hit rate.
    #[must_use]
    pub const fn is_absent(&self) -> bool {
        self.prompt_tokens == 0 && self.completion_tokens == 0 && self.total_tokens == 0 && self.cache_n == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(content: &str, finish: &str) -> AssistantTurn {
        AssistantTurn {
            content: content.to_string(),
            finish_reason: finish.to_string(),
            ..AssistantTurn::default()
        }
    }

    #[test]
    fn thinking_truncation_needs_all_three_conditions() {
        // The case measured on the live backend: budget spent on reasoning, nothing usable.
        let mut truncated = turn("", FINISH_LENGTH);
        truncated.reasoning = "hmm".into();
        assert!(truncated.is_thinking_truncated());

        // Length cut-off with content is an ordinary truncation, not a model failure.
        assert!(!turn("partial answer", FINISH_LENGTH).is_thinking_truncated());
        // Empty content with a normal stop is still a failure, but a different one: retrying
        // with a bigger budget would not help, so it must not be classified this way.
        assert!(!turn("", "stop").is_thinking_truncated());
        // Tool calls are content for this purpose.
        let mut with_call = turn("", FINISH_LENGTH);
        with_call.tool_calls.push(ToolCall::function("c", "f", "{}"));
        assert!(!with_call.is_thinking_truncated());
    }

    #[test]
    fn an_empty_turn_is_not_a_finished_task() {
        assert!(turn("  \n", "stop").is_empty());
        assert!(!turn("done", "stop").is_empty());
        let mut with_call = turn("", "tool_calls");
        with_call.tool_calls.push(ToolCall::function("c", "f", "{}"));
        assert!(!with_call.is_empty());
    }

    #[test]
    fn tool_calls_serialise_with_the_type_field() {
        let call = ToolCall::function("call_1", "read_file", r#"{"path":"a"}"#);
        let text = serde_json::to_string(&call).unwrap();
        assert!(text.contains(r#""type":"function""#), "{text}");
    }

    #[test]
    fn a_call_without_an_explicit_type_deserialises_as_a_function() {
        // The live backend always sends `type`, but a proxy that drops it must not make the
        // whole response unreadable.
        let call: ToolCall =
            serde_json::from_str(r#"{"id":"c","function":{"name":"f","arguments":"{}"}}"#).unwrap();
        assert_eq!(call.kind, "function");
    }

    #[test]
    fn cache_ratio_is_absent_when_nothing_was_cached() {
        let usage = Usage {
            prompt_tokens: 1000,
            cache_n: 0,
            ..Usage::default()
        };
        assert_eq!(usage.cache_hit_ratio(), None);
        let usage = Usage {
            prompt_tokens: 1000,
            cache_n: 950,
            ..Usage::default()
        };
        let ratio = usage.cache_hit_ratio().unwrap();
        assert!((ratio - 0.95).abs() < 1e-9);
    }

    #[test]
    fn cache_ratio_survives_a_zero_prompt_count() {
        // A backend that reports `cache_n` but omits `prompt_tokens` would otherwise divide by
        // zero. Returning `None` is the honest answer: the ratio is unknown, not infinite.
        let usage = Usage {
            prompt_tokens: 0,
            cache_n: 30,
            ..Usage::default()
        };
        assert_eq!(usage.cache_hit_ratio(), None);
    }
}
