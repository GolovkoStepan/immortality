//! The client against a scripted backend: streaming, classification, cancellation, health.
//!
//! # Why these are integration tests
//!
//! Every property that matters here lives in the composition of `reqwest`, the SSE decoder and
//! the classifier. A unit test of `classify_status` proves the mapping; it cannot prove that a
//! `429` with `Retry-After` is what the client actually sees, or that a stream cut off mid-call
//! ends as `Cancelled` rather than as a half-parsed turn.
//!
//! # The mock is the test's statement about the backend
//!
//! See `immortality_testkit::mock`: it answers a script, so a failure here points at the client
//! rather than at the mock's idea of a model.

// Integration tests: `expect` and `unwrap` are the right tool here — a failed assertion should
// name the exact line, and there is no caller to propagate to.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::time::Duration;

use immortality_llm::{
    Backend, ChatRequest, Failure, ModelFault, RetryKind, backoff, classify_turn, should_retry,
};
use immortality_testkit::{MockServer, Reply};

fn backend(server: &MockServer) -> Backend {
    Backend::new(server.base_url(), Duration::from_secs(2), Duration::from_secs(10)).expect("a valid request")
}

fn request() -> ChatRequest {
    ChatRequest::new("swift-iq2_xs")
        .with_system("you are a coder")
        .with_user("hi")
}

#[tokio::test]
async fn a_streamed_text_answer_is_assembled_character_by_character() {
    let server = MockServer::single(Reply::streamed_text("Done.")).await;
    let turn = backend(&server).turn(&request()).await.expect("a valid turn");

    assert_eq!(turn.content, "Done.");
    assert_eq!(turn.finish_reason, "stop");
    let usage = turn.usage.expect("the mock reports usage");
    assert_eq!(usage.cache_n, 32, "timings.cache_n must reach usage");
    assert_eq!(server.request_count(), 1);
}

#[tokio::test]
async fn a_streamed_tool_call_reassembles_its_arguments() {
    let server = MockServer::single(Reply::streamed_tool_call(
        "call_1",
        "read_file",
        r#"{"path":"src/main.rs"}"#,
    ))
    .await;
    let turn = backend(&server).turn(&request()).await.expect("a valid turn");

    assert_eq!(turn.finish_reason, "tool_calls");
    assert_eq!(turn.tool_calls.len(), 1);
    assert_eq!(turn.tool_calls[0].id, "call_1");
    assert_eq!(
        turn.tool_calls[0].function.parse().unwrap()["path"],
        "src/main.rs"
    );
}

#[tokio::test]
async fn what_was_sent_is_what_the_contract_requires() {
    // The three fields whose absence is invisible and expensive: thinking off, usage on, and a
    // streaming flag. A mock that did not read the body could not catch a dropped one.
    let server = MockServer::single(Reply::streamed_text("ok")).await;
    backend(&server).turn(&request()).await.expect("a valid turn");

    let body = server.last_body().expect("the server saw a JSON body");
    assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert_eq!(body["stream"], true);
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["model"], "swift-iq2_xs");
    assert_eq!(body["messages"][0]["role"], "system");
}

#[tokio::test]
async fn a_500_is_retryable_and_a_400_is_fatal() {
    let server = MockServer::single(Reply::Status {
        status: 500,
        body: "boom".into(),
    })
    .await;
    let failure = backend(&server).turn(&request()).await.unwrap_err();
    assert!(failure.is_retryable(), "{failure}");
    assert!(
        matches!(
            failure,
            Failure::Retryable {
                kind: RetryKind::ServerError,
                ..
            }
        ),
        "{failure}"
    );

    let server = MockServer::single(Reply::Status {
        status: 400,
        body: "bad schema".into(),
    })
    .await;
    let failure = backend(&server).turn(&request()).await.unwrap_err();
    assert!(failure.is_fatal(), "{failure}");
    assert!(failure.to_string().contains("bad schema"), "{failure}");
}

#[tokio::test]
async fn a_429_carries_its_retry_after_hint() {
    let server = MockServer::single(Reply::rate_limited(7)).await;
    let failure = backend(&server).turn(&request()).await.unwrap_err();
    assert!(
        matches!(
            failure,
            Failure::Retryable {
                kind: RetryKind::RateLimited,
                ..
            }
        ),
        "{failure}"
    );
    assert!(failure.to_string().contains("Retry-After 7"), "{failure}");
}

#[tokio::test]
async fn an_error_event_inside_a_200_stream_is_a_retryable_failure() {
    // The failure an HTTP client cannot see: status 200, body says "no".
    let server = MockServer::single(Reply::in_band_error("model crashed")).await;
    let failure = backend(&server).turn(&request()).await.unwrap_err();
    assert!(
        matches!(
            failure,
            Failure::Retryable {
                kind: RetryKind::BackendErrorEvent,
                ..
            }
        ),
        "{failure}"
    );
    assert!(failure.to_string().contains("model crashed"), "{failure}");
}

#[tokio::test]
async fn a_stream_cut_off_mid_answer_is_broken_not_partial() {
    // The decoder drops the torn tail, so the turn is short — but the client must not report it
    // as a completed answer, because the attempt would be recorded as done with half the work.
    let server = MockServer::single(Reply::SseCut {
        payloads: vec![
            serde_json::json!({"choices":[{"index":0,"delta":{"content":"half an ans"}}]}).to_string(),
        ],
    })
    .await;
    let failure = backend(&server).turn(&request()).await.unwrap_err();
    assert!(failure.is_retryable(), "{failure}");
    assert!(
        matches!(
            failure,
            Failure::Retryable {
                kind: RetryKind::BrokenStream,
                ..
            }
        ),
        "{failure}"
    );
}

#[tokio::test]
async fn a_torn_tool_call_is_never_returned_as_a_call() {
    // The cut lands inside the arguments. Executing the truncated command is the disaster this
    // test exists to prevent.
    let full = r#"{"command":"rm -rf build"}"#;
    let half = &full[..10];
    let server = MockServer::single(Reply::SseCut {
        payloads: vec![
            serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{
                "index":0,"id":"c","type":"function",
                "function":{"name":"shell","arguments":half}
            }]}}]})
            .to_string(),
        ],
    })
    .await;

    match backend(&server).turn(&request()).await {
        Err(failure) => assert!(failure.is_retryable(), "{failure}"),
        Ok(turn) => panic!("a torn call escaped: {turn:?}"),
    }
}

#[tokio::test]
async fn an_empty_stream_is_a_model_failure_not_a_finished_turn() {
    let server = MockServer::single(Reply::SseEmpty).await;
    let failure = backend(&server).turn(&request()).await.unwrap_err();
    assert!(failure.is_model_failure(), "{failure}");
}

#[tokio::test]
async fn a_thinking_truncation_is_reported_as_a_model_failure() {
    // The measured live failure: the budget went on reasoning and the answer is empty.
    let server = MockServer::single(Reply::Sse {
        payloads: vec![
            serde_json::json!({"choices":[{"index":0,"delta":{"reasoning_content":"hmm"}}]}).to_string(),
        ],
        finish_reason: "length".into(),
    })
    .await;

    let failure = backend(&server).turn(&request()).await.unwrap_err();
    assert!(failure.is_model_failure(), "{failure}");
    assert!(failure.to_string().contains("ThinkingTruncated"), "{failure}");
}

#[tokio::test]
async fn a_closed_connection_is_a_connect_failure() {
    // Port 1 on loopback: reserved, nothing listens.
    let backend = Backend::new(
        "http://127.0.0.1:1/v1",
        Duration::from_millis(500),
        Duration::from_secs(2),
    )
    .unwrap();
    let failure = backend.turn(&request()).await.unwrap_err();
    assert!(failure.is_retryable(), "{failure}");
    assert!(
        matches!(
            failure,
            Failure::Retryable {
                kind: RetryKind::Connect,
                ..
            }
        ),
        "{failure}"
    );
}

#[tokio::test]
async fn a_silent_backend_times_out_rather_than_hanging() {
    // A backend that accepts and then says nothing is the shape of a wedged model. The timeout
    // must fire, and must be classified as retryable: the request was fine, the server was not.
    let server = MockServer::single(Reply::HangUp).await;
    let backend = Backend::new(
        server.base_url(),
        Duration::from_millis(500),
        Duration::from_millis(300),
    )
    .unwrap();
    let failure = backend.turn(&request()).await.unwrap_err();
    assert!(failure.is_retryable(), "{failure}");
}

#[tokio::test]
async fn health_succeeds_on_a_working_backend_and_fails_on_a_broken_one() {
    let server = MockServer::single(Reply::json_turn("OK")).await;
    backend(&server).health("swift-iq2_xs").await.expect("healthy");

    let server = MockServer::single(Reply::Status {
        status: 503,
        body: "down".into(),
    })
    .await;
    let failure = backend(&server).health("swift-iq2_xs").await.unwrap_err();
    assert!(failure.is_retryable(), "{failure}");
}

#[tokio::test]
async fn models_lists_ids_and_ignores_the_rest() {
    let server = MockServer::single(Reply::Json {
        status: 200,
        body: r#"{"object":"list","data":[{"id":"swift-iq2_xs","object":"model","owned_by":"local"}]}"#
            .into(),
    })
    .await;
    let ids = backend(&server).models().await.expect("a model list");
    assert_eq!(ids, ["swift-iq2_xs"]);
}

#[tokio::test]
async fn models_reports_a_non_json_body_as_fatal() {
    // A proxy's HTML error page. Retrying cannot help: the URL is wrong or the proxy is
    // misconfigured.
    let server = MockServer::single(Reply::Json {
        status: 200,
        body: "<html>not found</html>".into(),
    })
    .await;
    let failure = backend(&server).models().await.unwrap_err();
    assert!(failure.is_fatal(), "{failure}");
}

#[tokio::test]
async fn a_streamed_turn_can_be_aborted_mid_generation() {
    // Reasoning-loop detection (D4) must cut the stream off, not wait for the model to finish
    // five thousand tokens of the same paragraph.
    let server = MockServer::single(Reply::streamed_text("abcdefghij")).await;
    let mut seen = String::new();
    let failure = backend(&server)
        .turn_streaming(&request(), |delta| {
            seen.push_str(delta);
            // Stop after five characters.
            seen.chars().count() < 5
        })
        .await
        .unwrap_err();

    assert_eq!(failure, Failure::Cancelled);
    assert!(
        seen.chars().count() < 10,
        "generation was not cut short: {seen:?}"
    );
}

#[tokio::test]
async fn a_streamed_turn_that_is_not_aborted_returns_the_whole_answer() {
    let server = MockServer::single(Reply::streamed_text("abcdefghij")).await;
    let turn = backend(&server)
        .turn_streaming(&request(), |_| true)
        .await
        .expect("a valid turn");
    assert_eq!(turn.content, "abcdefghij");
}

#[tokio::test]
async fn the_retry_schedule_converges_instead_of_hammering() {
    // A slow test on purpose: it asserts the shape the daemon's loop will follow, and a
    // regression to a fixed 100 ms delay would be invisible in a unit test of `backoff`.
    let failure = Failure::retryable(RetryKind::ServerError, "500");
    let mut attempts = 0;
    let mut wait = Duration::ZERO;
    for attempt in 1.. {
        attempts += 1;
        if !should_retry(&failure, attempt) {
            break;
        }
        wait += backoff(attempt);
    }
    assert_eq!(attempts, immortality_llm::MAX_ATTEMPTS);
    // 1+2+4+8+16 = 31 s over five retries: patient, and nothing like a tight loop.
    assert!(
        wait >= Duration::from_secs(31) && wait <= Duration::from_secs(120),
        "the whole retry budget is {wait:?}"
    );
}

#[tokio::test]
async fn a_truncated_turn_with_content_is_not_silently_accepted() {
    // `classify_turn` is the last gate before the turn reaches the loop; a truncated answer
    // ends mid-edit and its tool arguments are often invalid JSON.
    let server = MockServer::single(Reply::Sse {
        payloads: vec![
            serde_json::json!({"choices":[{"index":0,"delta":{"content":"I will edit"}}]}).to_string(),
        ],
        finish_reason: "length".into(),
    })
    .await;
    let failure = backend(&server).turn(&request()).await.unwrap_err();
    assert!(failure.is_model_failure(), "{failure}");
    assert!(failure.to_string().contains("Truncated"), "{failure}");
}

#[tokio::test]
async fn a_complete_short_answer_is_accepted_despite_the_token_budget() {
    // The guard is not "any `length` is fatal": a stream that reaches `[DONE]` with content and a
    // `stop` reason is a normal answer, however short.
    let server = MockServer::single(Reply::streamed_text("ok")).await;
    let turn = backend(&server).turn(&request()).await.unwrap();
    assert_eq!(classify_turn(&turn), None);
    assert!(!turn.is_truncated());
}

#[tokio::test]
async fn a_502_after_a_long_silence_is_retried() {
    // The measured shape: the backend computes for a while and then gives up. Retrying is the
    // right answer; escalating would abandon every large task on its first attempt.
    let server = MockServer::single(Reply::slow(
        Duration::from_millis(100),
        Reply::Status {
            status: 502,
            body: "upstream".into(),
        },
    ))
    .await;
    let failure = backend(&server).turn(&request()).await.unwrap_err();
    assert_eq!(
        failure,
        Failure::retryable(RetryKind::ServerError, "HTTP 502: upstream")
    );
}

#[tokio::test]
async fn a_non_streamed_turn_reads_usage_and_timings() {
    let server = MockServer::single(Reply::json_turn("answer")).await;
    let turn = backend(&server).turn_blocking(&request()).await.unwrap();
    assert_eq!(turn.content, "answer");
    let usage = turn.usage.expect("usage is reported");
    assert_eq!(usage.cache_n, 20);
    assert_eq!(usage.prompt_tokens, 24);

    // The request must not carry streaming options once it is forced to be blocking.
    let body = server.last_body().unwrap();
    assert_eq!(body["stream"], false);
    assert!(body.get("stream_options").is_none(), "{body}");
}

#[tokio::test]
async fn a_model_fault_is_surfaced_with_its_kind() {
    // Distinguishing the three faults is what lets the loop react differently: a truncated
    // answer gets a bigger budget, an empty one gets a different prompt or profile.
    let server = MockServer::single(Reply::Sse {
        payloads: Vec::new(),
        finish_reason: "stop".into(),
    })
    .await;
    let failure = backend(&server).turn(&request()).await.unwrap_err();
    match failure {
        Failure::ModelFailure {
            fault: ModelFault::EmptyTurn,
            ..
        } => {}
        other => panic!("expected an empty-turn fault, got {other}"),
    }
}
