//! The mock backend itself: determinism, script consumption, and that it stops.
//!
//! # Why the mock is tested
//!
//! Every client test is only as good as the backend it runs against. A mock that silently reorders
//! its script, repeats when it should not, or keeps a task alive after the test finishes produces
//! failures that look like client bugs and passes that mean nothing. So the harness gets its own
//! tests: the script is consumed exactly as written, the last reply repeats, the recorded requests
//! are the ones actually sent, and dropping the server leaves nothing running.

// Integration tests: `expect` and `unwrap` are the right tool here — a failed assertion should
// name the exact line, and there is no caller to propagate to.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::time::Duration;

use immortality_llm::{Backend, ChatRequest};
use immortality_testkit::mock::script_from_trace;
use immortality_testkit::{MockServer, Reply};

fn request() -> ChatRequest {
    ChatRequest::new("swift-iq2_xs").with_user("hi")
}

fn backend(server: &MockServer) -> Backend {
    Backend::new(server.base_url(), Duration::from_secs(2), Duration::from_secs(10)).expect("a valid client")
}

#[tokio::test]
async fn a_script_is_consumed_in_order_and_the_last_reply_repeats() {
    // Three different answers, then the third forever. The order is the whole contract: a client
    // test that expects "429, then 200" is meaningless if the mock may serve them the other way.
    let server = MockServer::start(vec![
        Reply::Status {
            status: 500,
            body: "first".into(),
        },
        Reply::Status {
            status: 502,
            body: "second".into(),
        },
        Reply::streamed_text("third"),
    ])
    .await;
    let backend = backend(&server);

    let first = backend.turn(&request()).await.unwrap_err();
    assert!(first.to_string().contains("first"), "{first}");
    let second = backend.turn(&request()).await.unwrap_err();
    assert!(second.to_string().contains("second"), "{second}");

    for _ in 0..3 {
        let turn = backend.turn(&request()).await.expect("the last reply repeats");
        assert_eq!(turn.content, "third");
    }
    assert_eq!(server.request_count(), 5);
}

#[tokio::test]
async fn the_recorded_requests_are_the_ones_that_were_sent() {
    let server = MockServer::single(Reply::streamed_text("ok")).await;
    backend(&server)
        .turn(&ChatRequest::new("m1").with_user("first"))
        .await
        .expect("a turn");
    backend(&server)
        .turn(&ChatRequest::new("m2").with_user("second"))
        .await
        .expect("a turn");

    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].body.as_ref().unwrap()["model"], "m1");
    assert_eq!(requests[1].body.as_ref().unwrap()["model"], "m2");
    assert_eq!(requests[0].path, "/v1/chat/completions");
}

#[tokio::test]
async fn a_looping_reply_never_gets_bored() {
    // D1 fires on the third identical call. A backend that ran out of script would make a loop
    // detector look like it worked when what actually happened is that the script ended.
    let server = MockServer::single(Reply::looping_tool_call("read_file", r#"{"path":"a"}"#)).await;
    let backend = backend(&server);
    for _ in 0..5 {
        let turn = backend.turn(&request()).await.expect("a turn");
        assert_eq!(turn.tool_calls[0].function.name, "read_file");
        assert_eq!(turn.tool_calls[0].id, "call_loop");
    }
    assert_eq!(server.request_count(), 5);
}

#[tokio::test]
async fn a_looping_text_reply_can_be_cut_off_by_the_callback() {
    // The property D4 relies on: the backend keeps producing, and the client's callback is what
    // stops it. If the reply were finite, this test would pass without the callback working.
    let server = MockServer::single(Reply::looping_text("the same sentence ", 200)).await;
    let mut seen = String::new();
    let failure = backend(&server)
        .turn_streaming(&request(), |delta| {
            seen.push_str(delta);
            seen.chars().count() < 40
        })
        .await
        .unwrap_err();
    assert_eq!(failure, immortality_llm::Failure::Cancelled);
    assert!(
        seen.chars().count() < 100,
        "the callback did not stop generation: {} chars",
        seen.chars().count()
    );
}

#[tokio::test]
async fn a_server_that_never_answers_is_bounded_by_the_timeout() {
    // `HangUp` covers a backend that closes immediately; this covers one that accepts and then
    // says nothing, which is the shape of a wedged model. Without the timeout this test would
    // hang forever rather than fail, so it also proves the timeout is wired at all.
    let server = MockServer::single(Reply::slow(Duration::from_secs(30), Reply::streamed_text("late"))).await;
    let backend = Backend::new(
        server.base_url(),
        Duration::from_millis(500),
        Duration::from_millis(250),
    )
    .expect("a valid client");

    let started = std::time::Instant::now();
    let failure = backend.turn(&request()).await.unwrap_err();
    assert!(failure.is_retryable(), "{failure}");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the timeout did not fire: {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn dropping_the_server_stops_the_accept_loop() {
    // The port is released when the task is aborted, so a second server can bind. A leaked task
    // would keep the port and make the next test flaky in a way that looks like a socket bug.
    let first = MockServer::single(Reply::streamed_text("a")).await;
    let url_one = first.base_url();
    drop(first);

    let second = MockServer::single(Reply::streamed_text("b")).await;
    assert_ne!(url_one, second.base_url());
    let turn = backend(&second).turn(&request()).await.expect("a turn");
    assert_eq!(turn.content, "b");
}

#[tokio::test]
async fn a_trace_becomes_a_script_that_replays_its_tool_calls() {
    // The S1 fixtures are what the backend really said. Replaying one through the client is the
    // closest a test gets to the live system, and it cannot drift from the protocol because it
    // *is* the protocol as captured.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let text = std::fs::read_to_string(dir.join("refactor.jsonl")).expect("the fixture exists");
    let script = script_from_trace(&text).expect("a valid trace");

    let first = script.first().expect("a non-empty script").clone();
    let server = MockServer::start(script).await;
    let turn = backend(&server).turn(&request()).await.expect("a turn");

    match first {
        Reply::Sse {
            ref finish_reason, ..
        } => assert_eq!(&turn.finish_reason, finish_reason),
        other => panic!("a trace should produce a streamed reply, got {other:?}"),
    }
    assert_eq!(turn.tool_calls.len(), 1);
    assert!(
        turn.tool_calls[0].id.starts_with("call_"),
        "{:?}",
        turn.tool_calls[0].id
    );
}

#[tokio::test]
async fn a_trace_step_marked_fatal_becomes_an_in_band_error() {
    // The spike recorded a failing request as text, not as a status code. Inventing a `500` would
    // be a claim the fixture does not support.
    let script = script_from_trace(
        r#"{"step":1,"fatal":"the model produced no output"}
{"step":2,"resp":{"content":"recovered","finish_reason":"stop"}}"#,
    )
    .expect("a valid trace");

    let server = MockServer::start(script).await;
    let backend = backend(&server);
    let failure = backend.turn(&request()).await.unwrap_err();
    assert!(
        failure.to_string().contains("no output"),
        "the fault text must survive: {failure}"
    );
    let turn = backend.turn(&request()).await.expect("the second step");
    assert_eq!(turn.content, "recovered");
}

#[tokio::test]
async fn a_trace_reply_round_trips_through_the_script_format() {
    // The scenario format is plain JSON, so a test can keep a script in a file and a reviewer can
    // read what the backend is pretending to do.
    let reply = Reply::streamed_text("hello");
    let text = serde_json::to_string(&reply).expect("serialisable");
    let back: Reply = serde_json::from_str(&text).expect("deserialisable");
    assert_eq!(back, reply);

    let delayed = Reply::slow(Duration::from_millis(15), Reply::in_band_error("boom"));
    let text = serde_json::to_string(&delayed).expect("serialisable");
    let back: Reply = serde_json::from_str(&text).expect("deserialisable");
    assert_eq!(back, delayed);
}
