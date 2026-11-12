//! Streamed turn assembly.
//!
//! The live backend emits tool-call arguments **character by character**, and the `OpenAI`
//! format allows several calls in one response distinguished only by `index`. Both facts are
//! load-bearing: a naive parser that assumes one JSON blob per delta loses calls, and one
//! that concatenates reasoning into content poisons the history sent back to the model.
//!
//! # What this module believes about the stream
//!
//! - `index` is the identity of a call and is **not** contiguous: the backend sends `0`, then
//!   `1`, then more of `0`. It is therefore a key, never a vector position.
//! - `id` and `function.name` arrive once, in the first fragment for that index. Later
//!   fragments carry only `arguments`, and a fragment that repeats the id is still the same
//!   call — overwriting is correct, appending is not.
//! - Empty deltas, `choices: []` and usage-only chunks are normal, not errors.
//! - `timings.cache_n` arrives as a sibling of `choices`, not inside `usage`.
//!
//! # Truncation
//!
//! A stream that ends without `[DONE]` is not an error by itself: a server may close
//! cleanly after the `finish_reason` chunk. [`TurnAccumulator::finish`] therefore reports what
//! it has, and [`StreamEnd`] tells the caller whether the stream was properly terminated — a
//! decision for the caller, because a short answer and a cut-off one look identical otherwise.

use serde_json::Value;

use crate::sse::SseParser;
use crate::turn::{AssistantTurn, ToolCall, Usage};

/// How a stream ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamEnd {
    /// The server sent `data: [DONE]`.
    Done,
    /// The byte stream ended without `[DONE]`, but the accumulated turn is usable.
    ///
    /// This is the shape of a clean close by a server that does not send `[DONE]`, and also of
    /// a connection that dropped after the last chunk. The two are indistinguishable here; the
    /// difference is whether `finish_reason` is set, which the caller can check.
    Closed,
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
    /// Whether the backend ever sent a non-empty `finish_reason`.
    ///
    /// Separate from `finish_reason`, which is normalised to `"stop"` for the engine's benefit.
    /// The normalisation is right for a completed turn and *wrong* for diagnosing one: without
    /// this flag a stream that died mid-answer looks exactly like an answer that ended, and the
    /// client cannot tell a retryable transport failure from a finished turn.
    saw_finish_reason: bool,
    /// Deltas seen, for the "the model said nothing at all" diagnostic.
    deltas: u64,
}

#[derive(Debug, Default)]
struct PartialCall {
    index: usize,
    id: String,
    name: String,
    arguments: String,
}

impl TurnAccumulator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one decoded SSE `data:` payload.
    ///
    /// Errors only on malformed JSON or an in-band `error` object. Everything else is optional
    /// by design: a missing field means the backend had nothing to say about it in this chunk,
    /// not that the stream is broken.
    pub fn push_json(&mut self, chunk: &str) -> Result<(), StreamError> {
        let value: Value = serde_json::from_str(chunk).map_err(|error| StreamError::Malformed {
            detail: error.to_string(),
        })?;
        self.deltas += 1;

        if let Some(usage) = value.get("usage").filter(|usage| !usage.is_null()) {
            // A usage block that does not fit the schema is not worth failing the turn over:
            // the counters are diagnostics, and the answer is still valid without them.
            self.usage = serde_json::from_value(usage.clone()).ok();
        }
        // Strata reports cache hits outside `usage`, under `timings`.
        if let Some(cache_n) = value
            .get("timings")
            .and_then(|timings| timings.get("cache_n"))
            .and_then(Value::as_u64)
        {
            let usage = self.usage.get_or_insert_with(Usage::default);
            // Saturating: a cache cannot serve more tokens than the model accepts, and a
            // truncated counter is a better outcome than a panic on absurd input.
            usage.cache_n = u32::try_from(cache_n).unwrap_or(u32::MAX);
        }
        if let Some(error) = value.get("error").filter(|error| !error.is_null()) {
            return Err(StreamError::Backend {
                detail: error.to_string(),
            });
        }

        let Some(choice) = value.get("choices").and_then(|choices| choices.get(0)) else {
            return Ok(());
        };

        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str)
            && !reason.is_empty()
        {
            self.finish_reason = reason.to_string();
            self.saw_finish_reason = true;
        }

        let Some(delta) = choice.get("delta") else {
            return Ok(());
        };

        if let Some(text) = delta.get("content").and_then(Value::as_str) {
            self.content.push_str(text);
        }
        // Reasoning is kept separate on purpose: concatenating it into `content` poisons the
        // history we send back.
        if let Some(text) = delta.get("reasoning_content").and_then(Value::as_str) {
            self.reasoning.push_str(text);
        }

        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                let index =
                    usize::try_from(call.get("index").and_then(Value::as_u64).unwrap_or(0)).unwrap_or(0);
                let slot = self.slot(index);
                if let Some(id) = call.get("id").and_then(Value::as_str)
                    && !id.is_empty()
                {
                    slot.id = id.to_string();
                }
                if let Some(function) = call.get("function") {
                    if let Some(name) = function.get("name").and_then(Value::as_str)
                        && !name.is_empty()
                    {
                        slot.name = name.to_string();
                    }
                    if let Some(args) = function.get("arguments").and_then(Value::as_str) {
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
        if let Some(position) = self.calls.iter().position(|call| call.index == index) {
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

    /// Whether the backend reported a `finish_reason`.
    ///
    /// A stream that closes without one was cut off: the protocol says a completed turn always
    /// carries it, and the connection closing is not evidence that the model finished.
    #[must_use]
    pub const fn saw_finish_reason(&self) -> bool {
        self.saw_finish_reason
    }

    /// How many `data:` payloads were consumed. A turn assembled from zero deltas is a
    /// different failure from one assembled from a hundred that said nothing.
    #[must_use]
    pub const fn deltas(&self) -> u64 {
        self.deltas
    }

    /// Consume the accumulator and produce the turn to hand to the model.
    ///
    /// Arguments are returned raw; validating them against the tool schema is the tool
    /// layer's job, and it must reject malformed JSON rather than guess at `{}`.
    #[must_use]
    pub fn finish(self) -> AssistantTurn {
        let mut calls: Vec<PartialCall> = self.calls;
        calls.sort_by_key(|call| call.index);
        AssistantTurn {
            content: self.content,
            reasoning: self.reasoning,
            tool_calls: calls.into_iter().map(PartialCall::into_tool_call).collect(),
            finish_reason: if self.finish_reason.is_empty() {
                "stop".to_string()
            } else {
                self.finish_reason
            },
            usage: self.usage.filter(|usage| !usage.is_absent()),
        }
    }
}

impl PartialCall {
    fn into_tool_call(self) -> ToolCall {
        ToolCall::function(
            if self.id.is_empty() {
                // A synthesized id keeps the journal's promise that every call has one. The
                // index is used rather than a counter so that the id is stable across a replay
                // of the same stream.
                format!("call_{}", self.index)
            } else {
                self.id
            },
            self.name,
            self.arguments,
        )
    }
}

/// Why a stream could not be assembled.
///
/// Deliberately not `anyhow::Error`: the caller classifies these into retryable and fatal, and
/// that classification is part of the contract (M2.3). A stringly error would force the caller
/// to match on text.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StreamError {
    /// A `data:` payload was not JSON. Retrying will not fix it; the backend is broken or the
    /// proxy mangled the body.
    #[error("malformed SSE payload: {detail}")]
    Malformed { detail: String },
    /// The backend reported an error inside the stream, after a 200 status.
    ///
    /// This is the failure that looks like success to an HTTP client: the status was fine, the
    /// body is an SSE error event. Treating it as a completed turn with no content would lose
    /// the reason.
    #[error("backend error in stream: {detail}")]
    Backend { detail: String },
}

/// Drives an SSE byte stream into a [`TurnAccumulator`].
///
/// Separate from the accumulator so that the accumulator stays testable by string, and so that
/// the parser's byte-level state lives next to the only code that feeds it.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    parser: SseParser,
    accumulator: TurnAccumulator,
    done: bool,
}

impl StreamDecoder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed raw bytes, returning nothing: the result is read once at the end.
    ///
    /// Returning events as they arrive would let a caller act on half a tool call, which is
    /// exactly the bug the three-record protocol exists to prevent.
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), StreamError> {
        self.push_with(bytes, |_| true).map(|_| ())
    }

    /// Feed raw bytes, calling `on_text` with the content each payload appends.
    ///
    /// Per **payload**, not per transport chunk, and that distinction is the whole point:
    /// `reqwest` is free to coalesce several SSE events into one `Bytes`, so a callback driven
    /// by chunk boundaries would hand a loop detector a hundred characters at once and could
    /// only stop *after* the whole run had arrived. Driven by payloads, it sees generation
    /// almost token by token, which is what "cut the stream off mid-sentence" requires.
    ///
    /// Returns `Ok(false)` when `on_text` asked to stop. Everything accumulated so far stays
    /// readable; the caller decides what a partial stream means (for reasoning-loop detection
    /// it means [`crate::Failure::Cancelled`], never a usable turn).
    ///
    /// # Errors
    /// See [`Self::push`].
    pub fn push_with(
        &mut self,
        bytes: &[u8],
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<bool, StreamError> {
        for event in self.parser.push(bytes) {
            let data = event.data.trim();
            if data == "[DONE]" {
                self.done = true;
                continue;
            }
            if data.is_empty() {
                continue;
            }
            // Deltas after `[DONE]` are ignored rather than an error: a keep-alive or a stray
            // chunk after the terminator is harmless, and failing the whole turn over it would
            // discard a complete answer.
            if self.done {
                continue;
            }
            let before = self.accumulator.content.len();
            self.accumulator.push_json(data)?;
            let after = self.accumulator.content.len();
            if after > before && !on_text(&self.accumulator.content[before..]) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Finish the stream, flushing any unterminated final event.
    ///
    /// An unterminated final `data:` line is a torn stream. Its content is *not* used: a
    /// half-written JSON object cannot be parsed, and using a half-parsed turn would mean
    /// acting on arguments the model never finished producing.
    pub fn finish(mut self) -> (AssistantTurn, StreamEnd) {
        // A trailing event that was not terminated by a blank line is dropped: it is the torn
        // tail, and the same rule that governs the journal governs the stream. The one thing
        // worth reading out of it is a `[DONE]` terminator, because a server that forgot the
        // final blank line still ended cleanly.
        if let Some(event) = self.parser.finish()
            && event.data.trim() == "[DONE]"
        {
            self.done = true;
        }
        let ended = if self.done {
            StreamEnd::Done
        } else {
            StreamEnd::Closed
        };
        (self.accumulator.finish(), ended)
    }

    /// Whether `[DONE]` has been seen.
    #[must_use]
    pub const fn is_done(&self) -> bool {
        self.done
    }

    /// The content accumulated so far.
    ///
    /// Exposed so a detector can inspect generation *while* it happens — reasoning-loop
    /// detection has to cut a stream off mid-sentence, and it cannot wait for `finish`.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.accumulator.content
    }

    /// Byte length of the content so far, for a caller that wants only the new part.
    #[must_use]
    pub const fn content_len(&self) -> usize {
        self.accumulator.content.len()
    }

    /// The content appended after `offset`, which must be a previous [`Self::content_len`].
    ///
    /// Returning only the new part keeps a per-delta detector linear. Handing it the whole
    /// accumulated text instead makes loop detection quadratic in the answer length, on the
    /// code path that is supposed to *stop* runaway generation.
    #[must_use]
    pub fn content_since(&self, offset: usize) -> &str {
        self.accumulator.content.get(offset..).unwrap_or_default()
    }

    /// The reasoning accumulated so far.
    #[must_use]
    pub fn reasoning(&self) -> &str {
        &self.accumulator.reasoning
    }

    /// Whether the backend reported a `finish_reason` before the stream ended.
    #[must_use]
    pub const fn saw_finish_reason(&self) -> bool {
        self.accumulator.saw_finish_reason()
    }

    /// Deltas consumed so far.
    #[must_use]
    pub const fn deltas(&self) -> u64 {
        self.accumulator.deltas()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn feed(decoder: &mut StreamDecoder, payloads: &[Value]) {
        for payload in payloads {
            let line = format!("data: {payload}\n\n");
            decoder.push(line.as_bytes()).expect("valid payload");
        }
    }

    #[test]
    fn assembles_character_by_character_arguments() {
        // The exact shape captured from the live backend.
        let chunks = [
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_5388","type":"function","function":{"name":"read_file","arguments":""}}]},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{"}}]},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"path\":"}}]},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"src"}}]},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"/main.rs\""}}]},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":300,"completion_tokens":40,"total_tokens":340},"timings":{"cache_n":280}}),
        ];
        let mut decoder = StreamDecoder::new();
        feed(&mut decoder, &chunks);
        let (turn, _) = decoder.finish();

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
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"read_file","arguments":"{\"pa"}}]}}]}),
            json!({"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","function":{"name":"write_file","arguments":"{\"pa"}}]}}]}),
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"x\"}"}}]}}]}),
            json!({"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"th\":\"y\",\"content\":\"z\"}"}}]}}]}),
        ];
        let mut decoder = StreamDecoder::new();
        feed(&mut decoder, &chunks);
        let (turn, _) = decoder.finish();

        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(turn.tool_calls[0].function.name, "read_file");
        assert_eq!(turn.tool_calls[0].function.parse().unwrap()["path"], "x");
        assert_eq!(turn.tool_calls[1].function.name, "write_file");
        assert_eq!(turn.tool_calls[1].function.parse().unwrap()["content"], "z");
    }

    #[test]
    fn three_calls_with_gaps_in_index_stay_in_order() {
        // Not contiguous and not starting at zero: `index` is a key, so a vector position
        // derived from it would panic or truncate.
        let mut accumulator = TurnAccumulator::new();
        for (index, name) in [(7, "a"), (2, "b"), (5, "c")] {
            accumulator
                .push_json(
                    &json!({"choices":[{"delta":{"tool_calls":[{"index":index,"id":format!("id_{name}"),"function":{"name":name,"arguments":"{}"}}]}}]})
                    .to_string(),
                )
                .unwrap();
        }
        let turn = accumulator.finish();
        let names: Vec<&str> = turn
            .tool_calls
            .iter()
            .map(|call| call.function.name.as_str())
            .collect();
        assert_eq!(names, ["b", "c", "a"], "sorted by index, not arrival");
    }

    #[test]
    fn reasoning_is_not_merged_into_content() {
        let mut decoder = StreamDecoder::new();
        feed(
            &mut decoder,
            &[
                json!({"choices":[{"delta":{"reasoning_content":"thinking..."}}]}),
                json!({"choices":[{"delta":{"content":"answer"}}]}),
            ],
        );
        let (turn, _) = decoder.finish();
        assert_eq!(turn.content, "answer");
        assert_eq!(turn.reasoning, "thinking...");
    }

    #[test]
    fn detects_thinking_truncation() {
        let mut decoder = StreamDecoder::new();
        feed(
            &mut decoder,
            &[
                json!({"choices":[{"delta":{"reasoning_content":"hmm"},"finish_reason":null}]}),
                json!({"choices":[{"delta":{},"finish_reason":"length"}]}),
            ],
        );
        assert!(decoder.finish().0.is_thinking_truncated());
    }

    #[test]
    fn thinking_truncated_is_false_when_content_present() {
        let mut decoder = StreamDecoder::new();
        feed(
            &mut decoder,
            &[
                json!({"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}),
                json!({"choices":[{"delta":{},"finish_reason":"length"}]}),
            ],
        );
        assert!(!decoder.finish().0.is_thinking_truncated());
    }

    #[test]
    fn missing_index_defaults_to_zero() {
        let mut decoder = StreamDecoder::new();
        feed(
            &mut decoder,
            &[
                json!({"choices":[{"delta":{"tool_calls":[{"id":"c","function":{"name":"f","arguments":"{}"}}]}}]}),
            ],
        );
        let (turn, _) = decoder.finish();
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].id, "c");
    }

    #[test]
    fn empty_delta_and_missing_choices_are_ignored() {
        let mut decoder = StreamDecoder::new();
        feed(
            &mut decoder,
            &[
                json!({"choices":[]}),
                json!({"choices":[{"delta":{}}]}),
                json!({"id":"x","object":"chat.completion.chunk"}),
            ],
        );
        let (turn, end) = decoder.finish();
        assert_eq!(turn.tool_calls, Vec::new());
        assert_eq!(end, StreamEnd::Closed);
    }

    #[test]
    fn backend_error_inside_stream_is_an_error() {
        let mut accumulator = TurnAccumulator::new();
        let error = accumulator
            .push_json(r#"{"error":{"message":"boom"}}"#)
            .unwrap_err();
        assert!(matches!(error, StreamError::Backend { .. }), "{error}");
        assert!(error.to_string().contains("boom"));
    }

    #[test]
    fn malformed_payload_is_an_error() {
        let mut accumulator = TurnAccumulator::new();
        assert!(matches!(
            accumulator.push_json("not json"),
            Err(StreamError::Malformed { .. })
        ));
    }

    #[test]
    fn a_null_error_field_is_not_an_error() {
        // The backend sends `"error": null` on healthy chunks, and treating that as an error
        // would fail every request.
        let mut accumulator = TurnAccumulator::new();
        assert!(
            accumulator
                .push_json(r#"{"error":null,"choices":[{"delta":{"content":"ok"}}]}"#)
                .is_ok()
        );
        assert_eq!(accumulator.finish().content, "ok");
    }

    #[test]
    fn finish_reason_defaults_to_stop() {
        let mut decoder = StreamDecoder::new();
        feed(&mut decoder, &[json!({"choices":[{"delta":{"content":"hi"}}]})]);
        assert_eq!(decoder.finish().0.finish_reason, "stop");
    }

    #[test]
    fn generated_ids_when_model_omits_them() {
        let mut decoder = StreamDecoder::new();
        feed(
            &mut decoder,
            &[
                json!({"choices":[{"delta":{"tool_calls":[{"index":2,"function":{"name":"f","arguments":"{}"}}]}}]}),
            ],
        );
        assert_eq!(decoder.finish().0.tool_calls[0].id, "call_2");
    }

    #[test]
    fn done_is_recognised_and_ends_the_stream() {
        let mut decoder = StreamDecoder::new();
        decoder
            .push(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\ndata: [DONE]\n\n")
            .unwrap();
        assert!(decoder.is_done());
        let (turn, end) = decoder.finish();
        assert_eq!(turn.content, "x");
        assert_eq!(end, StreamEnd::Done);
    }

    #[test]
    fn a_closed_stream_without_done_is_reported_as_closed() {
        // The distinction matters: `Closed` with a set `finish_reason` is a clean close, while
        // `Closed` without one is a connection that dropped mid-answer.
        let mut decoder = StreamDecoder::new();
        decoder
            .push(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
            .unwrap();
        let (_, end) = decoder.finish();
        assert_eq!(end, StreamEnd::Closed);
    }

    #[test]
    fn a_torn_final_payload_is_dropped_not_half_applied() {
        // A stream that dies mid-JSON must not execute the tool call it was describing.
        let mut decoder = StreamDecoder::new();
        decoder
            .push(b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"tool")
            .unwrap();
        let (turn, end) = decoder.finish();
        assert_eq!(turn.content, "ok");
        assert!(turn.tool_calls.is_empty(), "a half-written call is not a call");
        assert_eq!(end, StreamEnd::Closed);
    }

    #[test]
    fn deltas_after_done_are_ignored() {
        let mut decoder = StreamDecoder::new();
        decoder
            .push(b"data: [DONE]\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}\n\n")
            .unwrap();
        assert_eq!(decoder.finish().0.content, "");
    }

    #[test]
    fn timings_only_usage_is_not_mistaken_for_an_empty_turn() {
        let mut decoder = StreamDecoder::new();
        feed(
            &mut decoder,
            &[
                json!({"choices":[{"delta":{"content":"hi"}}]}),
                json!({"choices":[],"timings":{"cache_n":10,"prompt_n":20,"predicted_n":1}}),
            ],
        );
        // `cache_n` alone does not make a full usage block: prompt and completion are the
        // fields the budget reads, and a `Some` with them both zero would calibrate against zeros.
        let usage = decoder.finish().0.usage.unwrap();
        assert_eq!(usage.cache_n, 10);
        assert_eq!(usage.prompt_tokens, 0);
    }

    #[test]
    fn one_byte_at_a_time_survives_the_whole_pipeline() {
        // The parser has this test alone; here it proves the decoder above it does not
        // reassemble payloads in a way that assumes chunk boundaries.
        let source = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c\",",
            "\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":1}\"}}]}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let mut decoder = StreamDecoder::new();
        for byte in source.as_bytes() {
            decoder.push(std::slice::from_ref(byte)).unwrap();
        }
        let (turn, end) = decoder.finish();
        assert_eq!(end, StreamEnd::Done);
        assert_eq!(turn.tool_calls[0].function.parse().unwrap()["a"], 1);
    }

    #[test]
    fn a_keep_alive_comment_does_not_end_the_turn() {
        let mut decoder = StreamDecoder::new();
        decoder
            .push(b": keep-alive\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n: keep-alive\n\ndata: [DONE]\n\n")
            .unwrap();
        let (turn, end) = decoder.finish();
        assert_eq!(turn.content, "x");
        assert_eq!(end, StreamEnd::Done);
    }
}
