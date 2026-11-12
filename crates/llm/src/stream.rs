//! Streamed turn assembly.
//!
//! The live backend emits tool-call arguments **character by character**, and the `OpenAI`
//! format allows several calls in one response distinguished only by `index`. Both facts are
//! load-bearing: a naive parser that assumes one JSON blob per delta loses calls, and one
//! that concatenates reasoning into content poisons the history sent back to the model.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A finished assistant turn.
#[derive(Debug, Clone, Default)]
pub struct AssistantTurn {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: String,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
    /// Tokens served from the prefix cache. Strata reports this under `timings.cache_n`;
    /// a miss on a grown prefix is the difference between 1 s and 90 s per step.
    #[serde(default)]
    pub cache_n: u32,
}

impl AssistantTurn {
    #[must_use]
    pub const fn is_tool_call(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// The failure mode we measured on the live backend: the model spent the whole budget
    /// thinking and never produced an answer. Must not be treated as "done".
    #[must_use]
    pub fn is_thinking_truncated(&self) -> bool {
        self.finish_reason == "length" && self.content.trim().is_empty() && self.tool_calls.is_empty()
    }
}

/// Accumulates streamed deltas into a turn. Tolerates interleaved calls, gaps in `index`,
/// and arguments split mid-token.
#[derive(Debug, Default)]
pub struct TurnAccumulator {
    content: String,
    reasoning: String,
    calls: Vec<PartialCall>,
    finish_reason: String,
    usage: Option<Usage>,
}

#[derive(Debug, Default)]
struct PartialCall {
    index: usize,
    id: String,
    name: String,
    arguments: String,
}

impl TurnAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one decoded SSE `data:` payload.
    ///
    /// Every field is optional: the backend sends empty deltas, keep-alive-adjacent chunks
    /// and usage-only chunks, and none of them are errors.
    pub fn push_json(&mut self, chunk: &str) -> Result<()> {
        let value: Value = serde_json::from_str(chunk).context("malformed SSE data payload")?;

        if let Some(usage) = value.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(serde_json::from_value(usage.clone()).unwrap_or_default());
        }
        // Strata reports cache hits outside `usage`, under `timings`.
        if let Some(cache_n) = value
            .get("timings")
            .and_then(|t| t.get("cache_n"))
            .and_then(serde_json::Value::as_u64)
        {
            let usage = self.usage.get_or_insert_with(Usage::default);
            // Saturating: a cache cannot serve more tokens than the model accepts, and a
            // truncated counter is a better outcome than a panic on absurd input.
            usage.cache_n = u32::try_from(cache_n).unwrap_or(u32::MAX);
        }
        if let Some(err) = value.get("error") {
            bail!("backend error in stream: {err}");
        }

        let Some(choice) = value.get("choices").and_then(|c| c.get(0)) else {
            return Ok(());
        };

        if let Some(reason) = choice.get("finish_reason").and_then(serde_json::Value::as_str)
            && !reason.is_empty()
        {
            self.finish_reason = reason.to_string();
        }

        let Some(delta) = choice.get("delta") else {
            return Ok(());
        };

        if let Some(text) = delta.get("content").and_then(serde_json::Value::as_str) {
            self.content.push_str(text);
        }
        // Reasoning is kept separate on purpose: concatenating it into `content` poisons the
        // history we send back.
        if let Some(text) = delta.get("reasoning_content").and_then(serde_json::Value::as_str) {
            self.reasoning.push_str(text);
        }

        if let Some(calls) = delta.get("tool_calls").and_then(serde_json::Value::as_array) {
            for call in calls {
                let index =
                    usize::try_from(call.get("index").and_then(serde_json::Value::as_u64).unwrap_or(0))
                        .unwrap_or(0);
                let slot = self.slot(index);
                if let Some(id) = call.get("id").and_then(serde_json::Value::as_str)
                    && !id.is_empty()
                {
                    slot.id = id.to_string();
                }
                if let Some(function) = call.get("function") {
                    if let Some(name) = function.get("name").and_then(serde_json::Value::as_str)
                        && !name.is_empty()
                    {
                        slot.name = name.to_string();
                    }
                    if let Some(args) = function.get("arguments").and_then(serde_json::Value::as_str) {
                        slot.arguments.push_str(args);
                    }
                }
            }
        }
        Ok(())
    }

    /// Find or create the accumulator slot for a tool-call `index`.
    ///
    /// Indices from the live backend are interleaved and not guaranteed contiguous, so they
    /// cannot be used as a vector position.
    fn slot(&mut self, index: usize) -> &mut PartialCall {
        if let Some(position) = self.calls.iter().position(|c| c.index == index) {
            return &mut self.calls[position];
        }
        self.calls.push(PartialCall {
            index,
            ..Default::default()
        });
        // `last_mut` cannot fail: a push just succeeded.
        self.calls
            .last_mut()
            .unwrap_or_else(|| unreachable!("just pushed"))
    }

    /// Consume the accumulator and produce the turn to hand to the model.
    ///
    /// Arguments are returned raw; validating them against the tool schema is the tool
    /// layer's job, and it must reject malformed JSON rather than guess at `{}`.
    pub fn finish(self) -> AssistantTurn {
        let mut calls: Vec<PartialCall> = self.calls;
        calls.sort_by_key(|c| c.index);
        AssistantTurn {
            content: self.content,
            reasoning: self.reasoning,
            tool_calls: calls
                .into_iter()
                .map(|c| ToolCall {
                    id: if c.id.is_empty() {
                        format!("call_{}", c.index)
                    } else {
                        c.id
                    },
                    kind: "function".into(),
                    function: CalledFunction {
                        name: c.name,
                        arguments: c.arguments,
                    },
                })
                .collect(),
            finish_reason: if self.finish_reason.is_empty() {
                "stop".into()
            } else {
                self.finish_reason
            },
            usage: self.usage,
        }
    }
}

/// One requested tool call, exactly as the model produced it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default)]
    pub kind: String,
    pub function: CalledFunction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalledFunction {
    pub name: String,
    /// Raw JSON string. Validation against the tool schema happens in the tool layer.
    pub arguments: String,
}

impl CalledFunction {
    pub fn parse(&self) -> Result<Value> {
        serde_json::from_str(&self.arguments)
            .with_context(|| format!("tool `{}` returned invalid JSON arguments", self.name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembles_character_by_character_arguments() {
        let chunks = [
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_5388","type":"function","function":{"name":"read_file","arguments":""}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"path\":"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"src"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"/main.rs\""}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":300,"completion_tokens":40,"total_tokens":340},"timings":{"cache_n":280}}"#,
        ];
        let mut acc = TurnAccumulator::new();
        for chunk in chunks {
            acc.push_json(chunk).unwrap();
        }
        let turn = acc.finish();
        assert_eq!(turn.finish_reason, "tool_calls");
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].function.name, "read_file");
        assert_eq!(
            turn.tool_calls[0].function.parse().unwrap()["path"],
            "src/main.rs"
        );
        let usage = turn.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 300);
        assert_eq!(usage.cache_n, 280);
    }

    #[test]
    fn keeps_interleaved_calls_apart() {
        let chunks = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"read_file","arguments":"{\"pa"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","function":{"name":"write_file","arguments":"{\"pa"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"x\"}"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"th\":\"y\",\"content\":\"z\"}"}}]}}]}"#,
        ];
        let mut acc = TurnAccumulator::new();
        for chunk in chunks {
            acc.push_json(chunk).unwrap();
        }
        let turn = acc.finish();
        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(turn.tool_calls[0].function.parse().unwrap()["path"], "x");
        assert_eq!(turn.tool_calls[1].function.parse().unwrap()["content"], "z");
    }

    #[test]
    fn reasoning_is_not_merged_into_content() {
        let mut acc = TurnAccumulator::new();
        acc.push_json(r#"{"choices":[{"delta":{"reasoning_content":"thinking..."}}]}"#)
            .unwrap();
        acc.push_json(r#"{"choices":[{"delta":{"content":"answer"}}]}"#)
            .unwrap();
        let turn = acc.finish();
        assert_eq!(turn.content, "answer");
        assert_eq!(turn.reasoning, "thinking...");
    }

    #[test]
    fn detects_thinking_truncation() {
        let mut acc = TurnAccumulator::new();
        acc.push_json(r#"{"choices":[{"delta":{"reasoning_content":"hmm"},"finish_reason":null}]}"#)
            .unwrap();
        acc.push_json(r#"{"choices":[{"delta":{},"finish_reason":"length"}]}"#)
            .unwrap();
        assert!(acc.finish().is_thinking_truncated());
    }

    #[test]
    fn thinking_truncated_is_false_when_content_present() {
        let mut acc = TurnAccumulator::new();
        acc.push_json(r#"{"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}"#)
            .unwrap();
        acc.push_json(r#"{"choices":[{"delta":{},"finish_reason":"length"}]}"#)
            .unwrap();
        assert!(!acc.finish().is_thinking_truncated());
    }

    #[test]
    fn missing_index_defaults_to_zero() {
        let mut acc = TurnAccumulator::new();
        acc.push_json(
            r#"{"choices":[{"delta":{"tool_calls":[{"id":"c","function":{"name":"f","arguments":"{}"}}]}}]}"#,
        )
        .unwrap();
        assert_eq!(acc.finish().tool_calls.len(), 1);
    }

    #[test]
    fn empty_delta_and_missing_choices_are_ignored() {
        let mut acc = TurnAccumulator::new();
        acc.push_json(r#"{"choices":[]}"#).unwrap();
        acc.push_json(r#"{"choices":[{"delta":{}}]}"#).unwrap();
        acc.push_json(r#"{"id":"x","object":"chat.completion.chunk"}"#)
            .unwrap();
        assert_eq!(acc.finish().tool_calls, Vec::new());
    }

    #[test]
    fn backend_error_inside_stream_is_an_error() {
        let mut acc = TurnAccumulator::new();
        assert!(acc.push_json(r#"{"error":{"message":"boom"}}"#).is_err());
    }

    #[test]
    fn malformed_payload_is_an_error() {
        let mut acc = TurnAccumulator::new();
        assert!(acc.push_json("not json").is_err());
    }

    #[test]
    fn finish_reason_defaults_to_stop() {
        let mut acc = TurnAccumulator::new();
        acc.push_json(r#"{"choices":[{"delta":{"content":"hi"}}]}"#)
            .unwrap();
        assert_eq!(acc.finish().finish_reason, "stop");
    }

    #[test]
    fn generated_ids_when_model_omits_them() {
        let mut acc = TurnAccumulator::new();
        acc.push_json(r#"{"choices":[{"delta":{"tool_calls":[{"index":2,"function":{"name":"f","arguments":"{}"}}]}}]}"#).unwrap();
        assert_eq!(acc.finish().tool_calls[0].id, "call_2");
    }
}
