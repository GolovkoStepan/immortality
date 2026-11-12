//! The JSON that goes over the wire, and nothing else.
//!
//! # Why a separate module from [`crate::turn`]
//!
//! These types are a faithful description of the `OpenAI` protocol, warts included: fields that
//! are `null` rather than absent, a `finish_reason` that is nullable, `tool_calls` nested two
//! levels deep, usage reported under two different keys depending on the server. Keeping them
//! apart means a serde annotation cannot accidentally change what the engine sees, and the
//! conversion is one place to test.
//!
//! # The two backend quirks that shape this file
//!
//! - **`chat_template_kwargs`.** `swift-iq2_xs` reasons by default and spends `max_tokens` on
//!   it. Sending `{"enable_thinking": false}` removes reasoning entirely and cuts latency ~2.8×.
//!   It is not part of the `OpenAI` schema, so it is a field of its own and omitted when unused.
//! - **`timings.cache_n`.** Strata reports prefix-cache hits outside `usage`, as a sibling of
//!   `choices`. A client that only reads `usage` reports 0 % cache hits on every request and
//!   nobody notices the prompt discipline regressing.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::turn::{AssistantTurn, FINISH_LENGTH, ToolCall, Usage};

/// A chat completion request.
///
/// Serialised with `skip_serializing_if` throughout so that an absent option is *absent* rather
/// than `null`. Some `OpenAI`-compatible servers treat an explicit `null` as a type error, and
/// the default for an omitted field is what the server chooses, which is what we want.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Strata / Qwen3 template switch. See the module docs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<Value>,
    /// Turns a streamed request into one that also returns token counts.
    ///
    /// Without it the `OpenAI` protocol sends `usage: null` on every streamed chunk, and the
    /// only way to count tokens is to estimate — which is exactly what the budget must not do.
    /// Servers that do not recognise it ignore it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

impl ChatRequest {
    /// A request with the project's defaults: temperature 0, streaming, thinking off.
    ///
    /// `temperature: 0` is not a stylistic choice. The agent's decisions are recorded in a
    /// journal and replayed on recovery, so two runs of the same journal should produce the same
    /// plan; sampling makes a recovery differ from the run it is recovering.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: Vec::new(),
            tools: None,
            tool_choice: None,
            temperature: Some(0.0),
            max_tokens: Some(DEFAULT_MAX_TOKENS),
            stream: Some(true),
            chat_template_kwargs: Some(json!({ "enable_thinking": false })),
            stream_options: Some(StreamOptions { include_usage: true }),
        }
    }

    /// Append a message, returning the request for chaining.
    #[must_use]
    pub fn with(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    #[must_use]
    pub fn with_system(self, text: impl Into<String>) -> Self {
        self.with(Message::system(text))
    }

    #[must_use]
    pub fn with_user(self, text: impl Into<String>) -> Self {
        self.with(Message::user(text))
    }

    #[must_use]
    pub fn with_tools(mut self, tools: Vec<ToolDef>) -> Self {
        self.tools = Some(tools);
        self
    }

    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Turn reasoning on or off, leaving the rest of the template kwargs untouched.
    ///
    /// Written as a merge rather than a replacement because the field is the backend's general
    /// escape hatch: a future profile may set something else there, and clobbering it would
    /// silently drop that setting.
    #[must_use]
    pub fn with_thinking(mut self, enabled: bool) -> Self {
        let mut kwargs = match self.chat_template_kwargs.take() {
            Some(Value::Object(map)) => map,
            _ => serde_json::Map::new(),
        };
        kwargs.insert("enable_thinking".into(), Value::Bool(enabled));
        self.chat_template_kwargs = Some(Value::Object(kwargs));
        self
    }
}

/// Default completion budget.
///
/// 1024 was enough for every S1 scenario, including a turn with five parallel `write_file`
/// calls. Larger values invite the model to ramble; smaller ones truncate tool arguments,
/// which is the one truncation that produces a *wrong* call rather than a missing one.
pub const DEFAULT_MAX_TOKENS: u32 = 1024;

/// One message in the conversation.
///
/// One struct rather than an enum because that is what the protocol is: a `role` and optional
/// fields, where which fields matter depends on the role. An enum would need a custom
/// `Serialize` to reproduce the wire format, and a mismatched custom serializer is a class of
/// bug that costs an afternoon to find.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    #[must_use]
    pub fn system(text: impl Into<String>) -> Self {
        Self::plain("system", text)
    }

    #[must_use]
    pub fn user(text: impl Into<String>) -> Self {
        Self::plain("user", text)
    }

    #[must_use]
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::plain("assistant", text)
    }

    fn plain(role: &str, text: impl Into<String>) -> Self {
        Self {
            role: role.to_string(),
            content: Some(text.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// The assistant turn that requested tools, echoed back so the server can match the answers.
    #[must_use]
    pub fn assistant_tool_calls(calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(calls),
            tool_call_id: None,
            name: None,
        }
    }

    /// A tool's result.
    ///
    /// `tool_call_id` is mandatory in the protocol; the name is sent too because several
    /// `OpenAI`-compatible servers (including Strata's template) use it to render the result,
    /// and one that expects it and gets `null` renders an empty message.
    #[must_use]
    pub fn tool_result(
        call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(call_id.into()),
            name: Some(name.into()),
        }
    }

    /// An assistant turn replayed from the journal: content plus its tool calls, if any.
    ///
    /// The journal stores the turn with the id the model produced, and replay must reproduce it
    /// byte for byte or the backend rejects the following `tool` message as answering an
    /// unknown call.
    #[must_use]
    pub fn from_turn(turn: &AssistantTurn) -> Self {
        let content = if turn.content.is_empty() {
            None
        } else {
            Some(turn.content.clone())
        };
        let tool_calls = if turn.tool_calls.is_empty() {
            None
        } else {
            Some(turn.tool_calls.clone())
        };
        Self {
            role: "assistant".to_string(),
            content,
            tool_calls,
            tool_call_id: None,
            name: None,
        }
    }
}

/// A tool offered to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDef {
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDef,
}

impl ToolDef {
    #[must_use]
    pub fn function(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            kind: "function".to_string(),
            function: FunctionDef {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: String,
    /// JSON Schema for the arguments.
    pub parameters: Value,
}

/// A non-streamed chat completion response.
///
/// Used by the health probe and by any caller that wants one answer rather than a stream. The
/// agent loop always streams, because a 60-second silent `POST` cannot be distinguished from a
/// hung server, while a stream shows progress and can be cut off mid-generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatResponse {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
    /// Strata's cache counters, reported outside `usage`.
    #[serde(default)]
    pub timings: Option<Timings>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Choice {
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub message: Option<ResponseMessage>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseMessage {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Backend timings. Only `cache_n` is load-bearing; the rest is kept for `doctor`'s report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Timings {
    #[serde(default)]
    pub cache_n: u32,
    #[serde(default)]
    pub prompt_n: u32,
    #[serde(default)]
    pub predicted_n: u32,
}

impl ChatResponse {
    /// The first choice as a normalised turn.
    ///
    /// A response with no choices is a model failure, not an empty success: returning
    /// `AssistantTurn::default()` would look like a task that finished with nothing to say.
    #[must_use]
    pub fn into_turn(self) -> Option<AssistantTurn> {
        let choice = self.choices.into_iter().next()?;
        let message = choice.message.unwrap_or_default();
        let mut usage = self.usage.unwrap_or_default();
        if let Some(timings) = self.timings {
            usage.cache_n = timings.cache_n;
            if usage.prompt_tokens == 0 {
                usage.prompt_tokens = timings.prompt_n;
            }
            if usage.completion_tokens == 0 {
                usage.completion_tokens = timings.predicted_n;
            }
        }
        Some(AssistantTurn {
            content: message.content.unwrap_or_default(),
            reasoning: message.reasoning_content.unwrap_or_default(),
            tool_calls: message.tool_calls.unwrap_or_default(),
            finish_reason: choice
                .finish_reason
                .filter(|reason| !reason.is_empty())
                .unwrap_or_else(|| "stop".to_string()),
            usage: (!usage.is_absent()).then_some(usage),
        })
    }
}

impl Default for ResponseMessage {
    fn default() -> Self {
        Self {
            role: "assistant".to_string(),
            content: None,
            reasoning_content: None,
            tool_calls: None,
        }
    }
}

/// Which body field the server used to report the cache hit count, for `doctor`.
///
/// Kept as a value rather than a comment because the answer changed between servers and a
/// future one may put it somewhere else: making the location explicit means a new location is a
/// visible change rather than a silent 0 % in every report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheReport {
    /// Reported under `timings.cache_n`.
    Timings,
    /// Reported under `usage.cache_n` (some forks of the `llama.cpp` server).
    Usage,
    /// Not reported at all.
    Absent,
}

impl ChatResponse {
    /// Where the cache counter came from.
    #[must_use]
    pub fn cache_report(&self) -> CacheReport {
        if self.timings.is_some_and(|timings| timings.cache_n > 0) {
            return CacheReport::Timings;
        }
        if self.usage.is_some_and(|usage| usage.cache_n > 0) {
            return CacheReport::Usage;
        }
        CacheReport::Absent
    }

    /// Whether the answer was cut off before it produced anything usable.
    #[must_use]
    pub fn is_thinking_truncated(&self) -> bool {
        self.choices.first().is_some_and(|choice| {
            let finish_matches = choice
                .finish_reason
                .as_deref()
                .is_some_and(|reason| reason == FINISH_LENGTH);
            let message = choice.message.as_ref();
            let no_content = message
                .and_then(|message| message.content.as_deref())
                .is_none_or(|content| content.trim().is_empty());
            let no_calls = message
                .and_then(|message| message.tool_calls.as_ref())
                .is_none_or(Vec::is_empty);
            finish_matches && no_content && no_calls
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_omits_unset_options_instead_of_sending_null() {
        let request = ChatRequest::new("m");
        let text = serde_json::to_string(&request).unwrap();
        assert!(!text.contains("null"), "{text}");
        assert!(text.contains(r#""enable_thinking":false"#), "{text}");
        assert!(text.contains(r#""include_usage":true"#), "{text}");
    }

    #[test]
    fn turning_thinking_on_keeps_other_template_kwargs() {
        // A profile may set other template options; flipping reasoning must not drop them.
        let request = ChatRequest::new("m").with_thinking(true).with_thinking(false);
        let kwargs = request.chat_template_kwargs.unwrap();
        assert_eq!(kwargs["enable_thinking"], false);
    }

    #[test]
    fn extra_template_kwargs_survive_a_thinking_switch() {
        let mut request = ChatRequest::new("m");
        request.chat_template_kwargs = Some(json!({ "enable_thinking": true, "other": 7 }));
        let request = request.with_thinking(false);
        let kwargs = request.chat_template_kwargs.unwrap();
        assert_eq!(kwargs["enable_thinking"], false);
        assert_eq!(kwargs["other"], 7);
    }

    #[test]
    fn a_tool_result_carries_the_call_id() {
        // Without the id the backend rejects the whole conversation: it has an answer that
        // matches no question.
        let message = Message::tool_result("call_1", "read_file", "ok");
        let json = serde_json::to_value(&message).unwrap();
        assert_eq!(json["role"], "tool");
        assert_eq!(json["tool_call_id"], "call_1");
        assert_eq!(json["name"], "read_file");
    }

    #[test]
    fn replaying_a_turn_round_trips_its_tool_calls() {
        let turn = AssistantTurn {
            content: String::new(),
            reasoning: "secret".into(),
            tool_calls: vec![ToolCall::function("call_1", "read_file", r#"{"path":"a"}"#)],
            finish_reason: "tool_calls".into(),
            usage: None,
        };
        let message = Message::from_turn(&turn);
        let json = serde_json::to_value(&message).unwrap();
        assert!(json.get("content").is_none(), "empty content is omitted: {json}");
        assert_eq!(json["tool_calls"][0]["id"], "call_1");
        // Reasoning must never be replayed: the model would read its own thinking as speech.
        assert!(!json.to_string().contains("secret"));
    }

    #[test]
    fn a_stop_reason_of_null_becomes_stop() {
        let response: ChatResponse = serde_json::from_str(
            r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":null}]}"#,
        )
        .unwrap();
        assert_eq!(response.into_turn().unwrap().finish_reason, "stop");
    }

    #[test]
    fn cache_n_from_timings_reaches_usage() {
        let response: ChatResponse = serde_json::from_str(
            r#"{"choices":[{"message":{"content":"x"}}],
                "usage":{"prompt_tokens":260,"completion_tokens":2,"total_tokens":262},
                "timings":{"cache_n":250}}"#,
        )
        .unwrap();
        assert_eq!(response.cache_report(), CacheReport::Timings);
        let usage = response.into_turn().unwrap().usage.unwrap();
        assert_eq!(usage.cache_n, 250);
        assert_eq!(usage.prompt_tokens, 260);
    }

    #[test]
    fn timings_fill_in_a_missing_usage_block() {
        // Some servers stream only `timings`. Treating that as "no usage" would force the
        // budget to fall back to an estimate on every step.
        let response: ChatResponse = serde_json::from_str(
            r#"{"choices":[{"message":{"content":"x"}}],"timings":{"prompt_n":100,"predicted_n":5,"cache_n":90}}"#,
        )
        .unwrap();
        let usage = response.into_turn().unwrap().usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.cache_n, 90);
    }

    #[test]
    fn no_choices_is_not_an_empty_success() {
        let response: ChatResponse = serde_json::from_str(r#"{"choices":[]}"#).unwrap();
        assert!(response.into_turn().is_none());
    }

    #[test]
    fn a_response_with_no_usage_reports_none_rather_than_zeros() {
        // `Some(Usage::default())` would look like a measured zero and poison the calibration.
        let response: ChatResponse =
            serde_json::from_str(r#"{"choices":[{"message":{"content":"x"}}]}"#).unwrap();
        assert!(response.into_turn().unwrap().usage.is_none());
    }

    #[test]
    fn thinking_truncation_is_detected_on_the_wire_type_too() {
        // `reqwest` gives the status and body before deserialisation, and a truncated body is
        // classified before it ever becomes a turn.
        let response: ChatResponse = serde_json::from_str(
            r#"{"choices":[{"message":{"role":"assistant","content":null,"reasoning_content":"hm"},"finish_reason":"length"}]}"#,
        )
        .unwrap();
        assert!(response.is_thinking_truncated());
    }

    #[test]
    fn a_turn_with_only_content_is_not_a_thinking_truncation() {
        let response: ChatResponse = serde_json::from_str(
            r#"{"choices":[{"message":{"content":"partial"},"finish_reason":"length"}]}"#,
        )
        .unwrap();
        assert!(!response.is_thinking_truncated());
    }

    #[test]
    fn the_request_round_trips_through_serde() {
        let request = ChatRequest::new("swift-iq2_xs")
            .with_system("sys")
            .with_user("hi")
            .with_tools(vec![ToolDef::function(
                "read_file",
                "read a file",
                json!({"type": "object"}),
            )]);
        let text = serde_json::to_string(&request).unwrap();
        let back: ChatRequest = serde_json::from_str(&text).unwrap();
        assert_eq!(back, request);
        assert_eq!(back.messages.len(), 2);
        assert_eq!(back.tools.unwrap()[0].function.name, "read_file");
    }

    #[test]
    fn a_function_definition_serialises_as_the_wire_expects() {
        let tool = ToolDef::function("t", "d", json!({"type": "object", "properties": {}}));
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "t");
    }

    #[test]
    fn a_full_request_matches_the_reference_json() {
        // The reference body is what was verified against the live backend during S1. Compared
        // as a whole because the interesting failures are omissions: a dropped
        // `stream_options` silences usage reporting, a dropped `chat_template_kwargs` costs
        // 2.8x latency, and neither shows up in a field-by-field assertion.
        //
        // Key order is not asserted — JSON objects are unordered — but the exact set of keys,
        // and the shapes of their values, are.
        let request = ChatRequest::new("swift-iq2_xs")
            .with_system("sys")
            .with_user("hi")
            .with_max_tokens(512)
            .with_tools(vec![ToolDef::function(
                "read_file",
                "read a file",
                json!({
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }),
            )]);

        let actual = serde_json::to_value(&request).unwrap();
        let expected = json!({
            "model": "swift-iq2_xs",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }
            }],
            "temperature": 0.0,
            "max_tokens": 512,
            "stream": true,
            "chat_template_kwargs": {"enable_thinking": false},
            "stream_options": {"include_usage": true}
        });
        assert_eq!(actual, expected);
    }

    #[test]
    fn a_request_without_tools_omits_the_field_entirely() {
        // An empty `tools: []` makes some servers reject the request and others switch to a
        // code path that ignores tool calls in the conversation history.
        let text = serde_json::to_string(&ChatRequest::new("m")).unwrap();
        assert!(!text.contains("tools"), "{text}");
        assert!(!text.contains("tool_choice"), "{text}");
    }
}
