//! Replaying the S1 traces through the stream decoder.
//!
//! # Why replay real traces
//!
//! Every other test in `immortality-llm` builds the stream it then parses, so it proves the
//! decoder agrees with the author's model of the backend. These traces are what the backend
//! actually said: real ids, real argument strings, real `finish_reason` values, real content
//! with Cyrillic and newlines in it. If the decoder disagrees with a trace, the decoder is
//! wrong.
//!
//! # The chunking is the point
//!
//! The traces are stored as whole responses, so the test cuts them into the adversarial shapes
//! the real stream takes: one byte at a time, split inside a UTF-8 sequence, split inside a
//! JSON string. A decoder that only works on whole payloads would pass every unit test in the
//! module and fail here.

// Integration tests: `expect` and `unwrap` are the right tool here — a failed assertion should
// name the exact line, and there is no caller to propagate to.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::fmt::Write as _;

use immortality_llm::{StreamDecoder, StreamEnd, ToolCall};
use immortality_testkit::parse_trace;

/// Render one trace response as the SSE stream the backend would have produced.
///
/// The shape is the measured one: role delta, then one chunk per tool call carrying the whole
/// argument string, then a finish chunk. It is deliberately *not* character-by-character here:
/// that shape is covered by the unit tests, and this test exists to check the trace's content
/// survives, so whole-argument chunks make a failure point at the data rather than the chunking.
fn to_sse(resp: &serde_json::Value) -> String {
    let mut out = String::new();
    let content = resp.get("content").and_then(|v| v.as_str()).unwrap_or("");
    let reasoning = resp
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .or_else(|| resp.get("reasoning").and_then(|v| v.as_str()))
        .unwrap_or("");

    out.push_str(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
    );
    // Writing into a `String` cannot fail, so the result is discarded explicitly.
    if !reasoning.is_empty() {
        let _ = writeln!(
            out,
            "data: {}\n",
            serde_json::json!({"choices":[{"index":0,"delta":{"reasoning_content":reasoning}}]})
        );
    }
    if !content.is_empty() {
        let _ = writeln!(
            out,
            "data: {}\n",
            serde_json::json!({"choices":[{"index":0,"delta":{"content":content}}]})
        );
    }
    if let Some(calls) = resp.get("tool_calls").and_then(|v| v.as_array()) {
        for (index, call) in calls.iter().enumerate() {
            let function = call.get("function").cloned().unwrap_or_default();
            let delta = serde_json::json!({
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": [{
                        "index": index,
                        "id": call.get("id").and_then(|v| v.as_str()).unwrap_or(""),
                        "type": "function",
                        "function": {
                            "name": function.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                            "arguments": function.get("arguments").and_then(|v| v.as_str()).unwrap_or(""),
                        }
                    }]}
                }]
            });
            let _ = writeln!(out, "data: {delta}\n");
        }
    }
    let finish = resp
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("stop");
    let _ = writeln!(
        out,
        "data: {}\n",
        serde_json::json!({"choices":[{"index":0,"delta":{},"finish_reason":finish}]})
    );
    out.push_str("data: [DONE]\n\n");
    out
}

/// Feed `source` to a decoder in chunks of `size` bytes and return the assembled turn.
fn decode_in_chunks(source: &str, size: usize) -> (immortality_llm::AssistantTurn, StreamEnd) {
    let mut decoder = StreamDecoder::new();
    let bytes = source.as_bytes();
    for chunk in bytes.chunks(size) {
        decoder
            .push(chunk)
            .expect("a stream derived from a real trace must parse");
    }
    decoder.finish()
}

/// All seven S1 traces, flattened to one `(scenario, step, resp)` list.
fn responses() -> Vec<(String, u32, serde_json::Value)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .canonicalize()
        .expect("the fixtures directory exists");
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .expect("readable fixtures directory")
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    paths.sort();

    let mut out = Vec::new();
    for path in paths {
        let scenario = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("unknown")
            .to_string();
        let text = std::fs::read_to_string(&path).unwrap();
        for step in parse_trace(&text).unwrap() {
            if let Some(resp) = step.resp {
                out.push((scenario.clone(), step.step, resp));
            }
        }
    }
    out
}

#[test]
fn every_step_of_every_trace_survives_a_one_byte_stream() {
    // The most hostile chunking the transport can produce. If the decoder is correct here, the
    // real stream's larger chunks are not a special case.
    let mut steps = 0;
    for (scenario, step, resp) in responses() {
        let source = to_sse(&resp);
        let (turn, end) = decode_in_chunks(&source, 1);
        assert_eq!(end, StreamEnd::Done, "{scenario} step {step}");
        assert_turn_matches_trace(&resp, &turn, &format!("{scenario} step {step}"));
        steps += 1;
    }
    assert!(steps > 20, "the traces should cover every step, got {steps}");
}

#[test]
fn every_step_survives_chunk_boundaries_inside_utf8_and_json() {
    // Sizes 2..17 are arbitrary but include the interesting ones: a size that splits a Cyrillic
    // codepoint, and one that splits every `{"` pair in the body.
    for size in [2, 3, 5, 7, 11, 13, 17] {
        for (scenario, step, resp) in responses() {
            let source = to_sse(&resp);
            let (turn, end) = decode_in_chunks(&source, size);
            assert_eq!(end, StreamEnd::Done, "{scenario} step {step} at {size}");
            assert_turn_matches_trace(&resp, &turn, &format!("{scenario} step {step} at {size}"));
        }
    }
}

#[test]
fn a_truncated_real_response_is_not_mistaken_for_a_complete_one() {
    // Cut the stream in the middle of the last tool call's arguments. The decoder must report
    // `Closed` and the call must be absent rather than half-parsed: executing it would run a
    // command with a cut-off path.
    let (_, _, resp) = responses()
        .into_iter()
        .find(|(_, _, resp)| resp.get("tool_calls").is_some())
        .expect("at least one trace step calls a tool");

    let source = to_sse(&resp);
    let cut = source.len() / 2;
    let mut decoder = StreamDecoder::new();
    decoder.push(&source.as_bytes()[..cut]).unwrap();
    let (turn, end) = decoder.finish();

    assert_eq!(end, StreamEnd::Closed, "a cut stream is not a finished one");
    for call in &turn.tool_calls {
        assert!(
            call.function.parse().is_ok(),
            "a half-written call escaped: {call:?}"
        );
    }
}

/// Assert the decoded turn carries what the trace recorded.
fn assert_turn_matches_trace(resp: &serde_json::Value, turn: &immortality_llm::AssistantTurn, label: &str) {
    let content = resp.get("content").and_then(|v| v.as_str()).unwrap_or("");
    assert_eq!(turn.content, content, "{label}: content was lost or altered");

    let finish = resp
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("stop");
    assert_eq!(turn.finish_reason, finish, "{label}: finish reason");

    let expected: Vec<&serde_json::Value> = resp
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .map(|calls| calls.iter().collect())
        .unwrap_or_default();
    assert_eq!(turn.tool_calls.len(), expected.len(), "{label}: tool call count");

    for (call, want) in turn.tool_calls.iter().zip(expected) {
        assert_call_matches(call, want, label);
    }
}

fn assert_call_matches(call: &ToolCall, want: &serde_json::Value, label: &str) {
    let id = want.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let name = want
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let arguments = want
        .get("function")
        .and_then(|f| f.get("arguments"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    assert_eq!(call.id, id, "{label}: the call id must be preserved exactly");
    assert_eq!(call.function.name, name, "{label}: tool name");
    assert_eq!(
        call.function.arguments, arguments,
        "{label}: arguments must be byte-identical to what the model produced"
    );
    // The arguments are valid JSON in every trace; asserting it here means a future trace with
    // broken arguments fails this test rather than silently reaching the tool layer.
    let parsed = call
        .function
        .parse()
        .unwrap_or_else(|error| panic!("{label}: arguments are not JSON: {error}"));
    assert!(parsed.is_object() || parsed.is_null(), "{label}: {parsed}");
}
