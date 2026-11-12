//! Minimal OpenAI-compatible chat client plus the streamed tool-call accumulator.
//!
//! Both pieces are intended to move into `immortality-llm`. The accumulator exists because
//! the live backend (Strata / `swift-iq2_xs`) emits tool-call arguments **character by
//! character**, and the OpenAI spec allows interleaved calls distinguished only by `index`.

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::sse::SseParser;

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Strata / Qwen3 template switch. Probed: disabling removes `reasoning_content`
    /// entirely and cuts latency ~2.8x without breaking tool-calling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<Value>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: Vec::new(),
            tools: None,
            tool_choice: None,
            temperature: Some(0.0),
            max_tokens: Some(1024),
            stream: Some(true),
            chat_template_kwargs: Some(serde_json::json!({"enable_thinking": false})),
        }
    }

    pub fn thinking(mut self, enabled: bool) -> Self {
        self.chat_template_kwargs = Some(serde_json::json!({ "enable_thinking": enabled }));
        self
    }

    pub fn system(mut self, text: impl Into<String>) -> Self {
        self.messages.push(Message::system(text));
        self
    }

    pub fn user(mut self, text: impl Into<String>) -> Self {
        self.messages.push(Message::user(text));
        self
    }

    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Self { role: "system".into(), content: Some(text.into()), tool_calls: None, tool_call_id: None, name: None }
    }
    pub fn user(text: impl Into<String>) -> Self {
        Self { role: "user".into(), content: Some(text.into()), tool_calls: None, tool_call_id: None, name: None }
    }
    /// Echo the assistant turn that requested tools.
    pub fn assistant_tool_calls(calls: Vec<ToolCall>) -> Self {
        Self { role: "assistant".into(), content: None, tool_calls: Some(calls), tool_call_id: None, name: None }
    }
    pub fn tool_result(call_id: impl Into<String>, name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(call_id.into()),
            name: Some(name.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDef,
}

impl Tool {
    pub fn function(name: &str, description: &str, parameters: Value) -> Self {
        Self {
            kind: "function".into(),
            function: FunctionDef { name: name.into(), description: description.into(), parameters },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default)]
    pub kind: String,
    pub function: CalledFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalledFunction {
    pub name: String,
    /// Raw JSON string, exactly as the model produced it. Validation happens later.
    pub arguments: String,
}

impl CalledFunction {
    pub fn parse(&self) -> Result<Value> {
        serde_json::from_str(&self.arguments)
            .with_context(|| format!("tool `{}` returned invalid JSON arguments", self.name))
    }
}

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
    pub fn is_tool_call(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// The failure mode we measured on the live backend: the model spent the whole budget
    /// thinking and never produced an answer. Must not be treated as "done".
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

    pub fn push_json(&mut self, chunk: &str) -> Result<()> {
        let value: Value = serde_json::from_str(chunk).context("malformed SSE data payload")?;

        if let Some(usage) = value.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(serde_json::from_value(usage.clone()).unwrap_or_default());
        }
        // Strata reports cache hits outside `usage`, under `timings`.
        if let Some(cache_n) = value
            .get("timings")
            .and_then(|t| t.get("cache_n"))
            .and_then(|n| n.as_u64())
        {
            let usage = self.usage.get_or_insert_with(Usage::default);
            usage.cache_n = cache_n as u32;
        }
        if let Some(err) = value.get("error") {
            bail!("backend error in stream: {err}");
        }

        let Some(choice) = value.get("choices").and_then(|c| c.get(0)) else {
            return Ok(());
        };

        if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
            if !reason.is_empty() {
                self.finish_reason = reason.to_string();
            }
        }

        let Some(delta) = choice.get("delta") else {
            return Ok(());
        };

        if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
            self.content.push_str(text);
        }
        // Reasoning is kept separate on purpose: concatenating it into `content` poisons the
        // history we send back.
        if let Some(text) = delta.get("reasoning_content").and_then(|c| c.as_str()) {
            self.reasoning.push_str(text);
        }

        if let Some(calls) = delta.get("tool_calls").and_then(|c| c.as_array()) {
            for call in calls {
                let index = call.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                let slot = self.slot(index);
                if let Some(id) = call.get("id").and_then(|i| i.as_str()) {
                    if !id.is_empty() {
                        slot.id = id.to_string();
                    }
                }
                if let Some(function) = call.get("function") {
                    if let Some(name) = function.get("name").and_then(|n| n.as_str()) {
                        if !name.is_empty() {
                            slot.name = name.to_string();
                        }
                    }
                    if let Some(args) = function.get("arguments").and_then(|a| a.as_str()) {
                        slot.arguments.push_str(args);
                    }
                }
            }
        }
        Ok(())
    }

    fn slot(&mut self, index: usize) -> &mut PartialCall {
        if let Some(pos) = self.calls.iter().position(|c| c.index == index) {
            return &mut self.calls[pos];
        }
        self.calls.push(PartialCall { index, ..Default::default() });
        let pos = self.calls.len() - 1;
        &mut self.calls[pos]
    }

    pub fn finish(self) -> AssistantTurn {
        let mut calls: Vec<PartialCall> = self.calls;
        calls.sort_by_key(|c| c.index);
        AssistantTurn {
            content: self.content,
            reasoning: self.reasoning,
            tool_calls: calls
                .into_iter()
                .map(|c| ToolCall {
                    id: if c.id.is_empty() { format!("call_{}", c.index) } else { c.id },
                    kind: "function".into(),
                    function: CalledFunction { name: c.name, arguments: c.arguments },
                })
                .collect(),
            finish_reason: if self.finish_reason.is_empty() { "stop".into() } else { self.finish_reason },
            usage: self.usage,
        }
    }
}

pub struct Client {
    base_url: String,
    http: reqwest::Client,
}

impl Client {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            http: reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("http client"),
        }
    }

    pub async fn models(&self) -> Result<Vec<String>> {
        #[derive(Deserialize)]
        struct List {
            data: Vec<Item>,
        }
        #[derive(Deserialize)]
        struct Item {
            id: String,
        }
        let list: List = self.http.get(format!("{}/models", self.base_url)).send().await?.json().await?;
        Ok(list.data.into_iter().map(|i| i.id).collect())
    }

    /// One streaming turn. Prints what it sees when `verbose`, returns the assembled turn.
    pub async fn stream_turn(&self, request: &ChatRequest, verbose: bool) -> Result<AssistantTurn> {
        let response = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .json(request)
            .send()
            .await
            .context("request failed")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("HTTP {status}: {body}");
        }

        let mut stream = response.bytes_stream();
        let mut parser = SseParser::new();
        let mut acc = TurnAccumulator::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("stream broke")?;
            for event in parser.push(&chunk) {
                if event.data.trim() == "[DONE]" {
                    if verbose {
                        eprintln!("  [stream] done");
                    }
                    return Ok(acc.finish());
                }
                if event.data.trim().is_empty() {
                    continue;
                }
                if verbose {
                    // Show only deltas that carry signal: content and tool-call fragments.
                    if let Ok(parsed) = serde_json::from_str::<Value>(&event.data) {
                        if let Some(delta) = parsed
                            .get("choices")
                            .and_then(|c| c.get(0))
                            .and_then(|c| c.get("delta"))
                        {
                            if let Some(reason) = parsed
                                .get("choices")
                                .and_then(|c| c.get(0))
                                .and_then(|c| c.get("finish_reason"))
                                .and_then(|r| r.as_str())
                            {
                                eprintln!("  [stream] finish={reason}");
                            }
                            let _ = delta;
                        }
                    }
                }
                acc.push_json(&event.data)?;
            }
        }
        if let Some(event) = parser.finish() {
            if event.data.trim() != "[DONE]" && !event.data.trim().is_empty() {
                acc.push_json(&event.data)?;
            }
        }
        Ok(acc.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembles_character_by_character_arguments() {
        // Captured shape from the live backend.
        let chunks = [
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_5388","type":"function","function":{"name":"read_file","arguments":""}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"path\":"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"src"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"/main.rs\""}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":300,"completion_tokens":40,"total_tokens":340}}"#,
        ];
        let mut acc = TurnAccumulator::new();
        for chunk in chunks {
            acc.push_json(chunk).unwrap();
        }
        let turn = acc.finish();
        assert_eq!(turn.finish_reason, "tool_calls");
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].function.name, "read_file");
        assert_eq!(turn.tool_calls[0].function.parse().unwrap()["path"], "src/main.rs");
        assert_eq!(turn.usage.unwrap().prompt_tokens, 300);
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
        assert_eq!(turn.tool_calls[0].function.name, "read_file");
        assert_eq!(turn.tool_calls[0].function.parse().unwrap()["path"], "x");
        assert_eq!(turn.tool_calls[1].function.name, "write_file");
        assert_eq!(turn.tool_calls[1].function.parse().unwrap()["content"], "z");
    }

    #[test]
    fn reasoning_is_not_merged_into_content() {
        let mut acc = TurnAccumulator::new();
        acc.push_json(r#"{"choices":[{"delta":{"reasoning_content":"thinking..."}}]}"#).unwrap();
        acc.push_json(r#"{"choices":[{"delta":{"content":"answer"}}]}"#).unwrap();
        let turn = acc.finish();
        assert_eq!(turn.content, "answer");
        assert_eq!(turn.reasoning, "thinking...");
    }

    #[test]
    fn detects_thinking_truncation() {
        let mut acc = TurnAccumulator::new();
        acc.push_json(r#"{"choices":[{"delta":{"reasoning_content":"hmm"},"finish_reason":null}]}"#).unwrap();
        acc.push_json(r#"{"choices":[{"delta":{},"finish_reason":"length"}]}"#).unwrap();
        assert!(acc.finish().is_thinking_truncated());
    }

    #[test]
    fn missing_index_defaults_to_zero() {
        let mut acc = TurnAccumulator::new();
        acc.push_json(r#"{"choices":[{"delta":{"tool_calls":[{"id":"c","function":{"name":"f","arguments":"{}"}}]}}]}"#).unwrap();
        let turn = acc.finish();
        assert_eq!(turn.tool_calls.len(), 1);
    }

    #[test]
    fn empty_delta_and_missing_choices_are_ignored() {
        let mut acc = TurnAccumulator::new();
        acc.push_json(r#"{"choices":[]}"#).unwrap();
        acc.push_json(r#"{"choices":[{"delta":{}}]}"#).unwrap();
        acc.push_json(r#"{"id":"x","object":"chat.completion.chunk"}"#).unwrap();
        assert!(acc.finish().tool_calls.is_empty());
    }

    #[test]
    fn backend_error_inside_stream_is_an_error() {
        let mut acc = TurnAccumulator::new();
        assert!(acc.push_json(r#"{"error":{"message":"boom"}}"#).is_err());
    }
}
