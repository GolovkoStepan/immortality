//! Why a turn failed, and what the loop should do about it.
//!
//! # Why this is a type and not an `anyhow::Error`
//!
//! Every caller of the client has to answer exactly one question: **may this be retried?**
//! A string cannot answer it without the caller matching on text, and matching on text is how a
//! `400` gets retried six times and a `502` gets escalated to a human.
//!
//! The three classes are not a ranking, they are different responses:
//!
//! - [`Failure::Retryable`] — the same request may succeed later. Backoff and send it again.
//! - [`Failure::ModelFailure`] — the backend worked and produced nothing usable. Retrying the
//!   identical request usually reproduces it, so the retry must change something (a larger
//!   `max_tokens`, a shorter prompt) or escalate.
//! - [`Failure::Fatal`] — the request itself is wrong. Repeating it is guaranteed to fail the
//!   same way, so the loop must stop and say so.
//!
//! Silence on the network is in the first class, and that is the case worth spelling out: a
//! 600-second timeout after a 130K-token prefill was measured on this backend, so a timeout is
//! evidence about the prompt's size, not about the request being invalid.

use std::time::Duration;

use crate::turn::AssistantTurn;

/// Retry budget for one request.
///
/// Six attempts is what the design settled on. The first backoff is 1 s so a transient daemon
/// hiccup costs nothing, and the last is 60 s so a backend restarting its model gets a full
/// minute of quiet before it is asked again.
pub const MAX_ATTEMPTS: u32 = 6;

/// Longest single backoff. Never exceeded, however many attempts are allowed.
pub const MAX_BACKOFF: Duration = Duration::from_secs(60);

/// Why a retry might help.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryKind {
    /// `429`: the backend is busy. The `Retry-After` header, when present, overrides backoff.
    RateLimited,
    /// `5xx`: the backend failed. This includes the measured `502` that arrives *after* a long
    /// prefill — it is a failure of the attempt, not of the request.
    ServerError,
    /// The connection was refused or the DNS lookup failed.
    Connect,
    /// The request or the response timed out. On this backend a timeout during prefill says the
    /// prompt is large, not that the request is malformed.
    Timeout,
    /// The stream ended before `[DONE]` and before a `finish_reason` was seen.
    BrokenStream,
    /// A payload inside a 200 response was not parseable.
    MalformedStream,
    /// The backend reported an error event mid-stream.
    BackendErrorEvent,
}

/// What was wrong with a turn the backend considered successful.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFault {
    /// `finish_reason: length`, no content, no tool calls: the budget went on reasoning. The
    /// measured failure mode of `swift-iq2_xs` with thinking enabled.
    ThinkingTruncated,
    /// A complete turn with nothing in it.
    EmptyTurn,
    /// The answer was cut off at the token limit, whether or not it produced content. Content
    /// from a truncated turn may end mid-sentence and tool arguments may be invalid JSON.
    Truncated,
}

/// How a request failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Failure {
    /// The same request may succeed after a pause.
    #[error("retryable backend failure ({kind:?}): {detail}")]
    Retryable { kind: RetryKind, detail: String },

    /// The backend answered, but the answer is unusable.
    #[error("model failure ({fault:?}): {detail}")]
    ModelFailure { fault: ModelFault, detail: String },

    /// The request is wrong and repeating it cannot help.
    #[error("fatal backend failure: {detail}")]
    Fatal { detail: String },

    /// The caller withdrew the request. Not a failure of the backend and never retried.
    #[error("request cancelled")]
    Cancelled,
}

impl Failure {
    #[must_use]
    pub fn retryable(kind: RetryKind, detail: impl Into<String>) -> Self {
        Self::Retryable {
            kind,
            detail: detail.into(),
        }
    }

    #[must_use]
    pub fn fatal(detail: impl Into<String>) -> Self {
        Self::Fatal {
            detail: detail.into(),
        }
    }

    #[must_use]
    pub fn model(fault: ModelFault, detail: impl Into<String>) -> Self {
        Self::ModelFailure {
            fault,
            detail: detail.into(),
        }
    }

    /// Whether sending the identical request again could produce a different outcome.
    ///
    /// `ModelFailure` returns `false` on purpose. The design's retry for a truncated turn
    /// doubles `max_tokens`, which is a *different* request; a caller that retries verbatim
    /// would reproduce the truncation and burn the budget. Making the caller construct a new
    /// request is the cheap way to force that decision to be visible.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Self::Retryable { .. })
    }

    /// Whether the loop should stop and report a problem rather than try again.
    #[must_use]
    pub const fn is_fatal(&self) -> bool {
        matches!(self, Self::Fatal { .. })
    }

    /// Whether the failure says the model produced nothing usable.
    #[must_use]
    pub const fn is_model_failure(&self) -> bool {
        matches!(self, Self::ModelFailure { .. })
    }
}

/// Classify an HTTP status.
///
/// The boundaries are the design's, and two of them are surprising enough to be worth stating:
///
/// - `408 Request Timeout` is retryable, not fatal: a slow prefill is the normal cause here.
/// - `502` is retryable rather than fatal. It was measured arriving after 300 s of prefill on a
///   130K prompt, which means the backend gave up mid-computation, not that the request was
///   invalid.
///
/// Anything unlisted in 4xx is fatal (the request is wrong) and anything else is fatal (an
/// unexpected status is not something to hammer).
#[must_use]
pub fn classify_status(status: u16, body: &str) -> Failure {
    let detail = format!("HTTP {status}: {}", truncate(body, 400));
    match status {
        408 | 429 => Failure::retryable(RetryKind::RateLimited, detail),
        500..=599 => Failure::retryable(RetryKind::ServerError, detail),
        _ => Failure::fatal(detail),
    }
}

/// Bound a body included in an error message.
///
/// An error capping a 200 KB HTML error page is unreadable in a terminal and expensive in a
/// journal. The head is enough to identify the cause; the full body is in the log if needed.
fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let head: String = text.chars().take(limit).collect();
    format!("{head}… ({} chars total)", text.chars().count())
}

/// Classify a transport error from `reqwest`.
///
/// Split from [`classify_status`] because the interesting cases are not statuses: a connect
/// timeout and a read timeout look the same to a caller that only sees "timed out", but the
/// first says the host is down and the second says the prompt is large.
#[must_use]
pub fn classify_transport(error: &reqwest::Error) -> Failure {
    let detail = transport_detail(error);
    if error.is_timeout() {
        return Failure::retryable(RetryKind::Timeout, detail);
    }
    if error.is_connect() {
        return Failure::retryable(RetryKind::Connect, detail);
    }
    if error.is_body() || error.is_decode() {
        // A body that died mid-stream: the connection dropped, so the same request may work.
        return Failure::retryable(RetryKind::BrokenStream, detail);
    }
    Failure::retryable(RetryKind::BrokenStream, detail)
}

/// What `reqwest` says, followed by what caused it.
///
/// `reqwest::Error`'s own `Display` is `error sending request for url (<url>)` and nothing else —
/// it names the endpoint and stops there. The reason (`connection refused`, `connection reset by
/// peer`, a DNS failure, a body that died mid-stream) is in its `source()` chain, which
/// `to_string()` does not walk.
///
/// That matters because `detail` is the field that answers "why did this attempt end" (D-125):
/// it was added so that "the backend died" and "the model was truncated" would stop reading the
/// same in the journal. Measured on `qna_rails`, 2026-11-16: two abandoned attempts both reached
/// the journal as `error sending request for url (…)` with no cause at all, so a human reading
/// the journal could not tell a flaky network from a dropped connection from a bug in the client.
fn transport_detail(error: &reqwest::Error) -> String {
    use std::error::Error as _;

    let mut detail = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        let text = cause.to_string();
        // A repeated reason adds a way to read the same fact twice — and a chain long enough to
        // loop would otherwise never end.
        if !detail.contains(&text) {
            detail.push_str(": ");
            detail.push_str(&text);
        }
        source = cause.source();
    }
    detail
}

/// Classify a completed turn.
///
/// A turn is only accepted when it carries something the engine can act on. The alternative —
/// accepting an empty turn — ends the attempt as successful with no work done, which is the
/// worst possible outcome: the journal says the subtask finished.
#[must_use]
pub fn classify_turn(turn: &AssistantTurn) -> Option<Failure> {
    if turn.is_thinking_truncated() {
        return Some(Failure::model(
            ModelFault::ThinkingTruncated,
            format!(
                "finish_reason=length, {} chars of reasoning, empty content and no tool calls",
                turn.reasoning.chars().count()
            ),
        ));
    }
    if turn.is_empty() {
        return Some(Failure::model(
            ModelFault::EmptyTurn,
            format!("finish_reason={}", turn.finish_reason),
        ));
    }
    if turn.is_truncated() {
        return Some(Failure::model(
            ModelFault::Truncated,
            format!(
                "finish_reason=length after {} chars of content",
                turn.content.chars().count()
            ),
        ));
    }
    None
}

/// The pause before attempt `attempt`, counting from 1.
///
/// Doubling from one second with a 60-second cap: 1, 2, 4, 8, 16, 32, 60, 60…
///
/// No jitter, deliberately. Jitter exists to spread a thundering herd of independent clients;
/// this daemon is the only client and it issues one request at a time. In
/// exchange, the schedule is reproducible, so a test can assert it and an operator reading a log
/// can tell whether a backend was down for two minutes or twenty.
#[must_use]
pub fn backoff(attempt: u32) -> Duration {
    /// The first pause. One second: a backend hiccup should cost a step almost nothing.
    const BASE_SECS: u64 = 1;
    let exponent = attempt.saturating_sub(1).min(6);
    let secs = BASE_SECS.saturating_mul(1u64 << exponent);
    Duration::from_secs(secs).min(MAX_BACKOFF)
}

/// Whether a failed attempt should be followed by another.
///
/// `attempt` is 1-based. The rule is a plain attempt count rather than a time budget: a 60-second
/// backoff at attempt six is already ten minutes of patience, and a time budget would make the
/// last attempt arrive at a different moment on every run.
#[must_use]
pub const fn should_retry(failure: &Failure, attempt: u32) -> bool {
    failure.is_retryable() && attempt < MAX_ATTEMPTS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_statuses_are_retryable_and_client_errors_are_fatal() {
        for status in [408, 429, 500, 502, 503, 504] {
            let failure = classify_status(status, "");
            assert!(failure.is_retryable(), "{status} must be retryable: {failure}");
        }
        for status in [400, 401, 403, 404, 422] {
            let failure = classify_status(status, "");
            assert!(failure.is_fatal(), "{status} must be fatal: {failure}");
        }
    }

    #[test]
    fn a_502_after_prefill_is_retried() {
        // Measured: 502 arrives after 300 s of prefill on a 130K prompt. Treating it as fatal
        // would abandon every large task on the first attempt.
        let failure = classify_status(502, "upstream error");
        assert_eq!(
            failure,
            Failure::retryable(RetryKind::ServerError, "HTTP 502: upstream error")
        );
    }

    #[test]
    fn a_redirect_is_fatal_rather_than_followed_blindly() {
        // `reqwest` follows redirects; one reaching here means the redirect looped or the policy
        // refused it. Retrying the same request cannot help.
        assert!(classify_status(301, "").is_fatal());
    }

    #[test]
    fn backoff_doubles_and_then_stops_growing() {
        let schedule: Vec<u64> = (1..=8).map(|attempt| backoff(attempt).as_secs()).collect();
        assert_eq!(schedule, [1, 2, 4, 8, 16, 32, 60, 60]);
    }

    #[test]
    fn backoff_is_zero_safe() {
        // Attempt 0 does not exist, but a caller that passes one must not underflow.
        assert_eq!(backoff(0), Duration::from_secs(1));
    }

    #[tokio::test]
    async fn a_timeout_stays_a_timeout_and_still_names_its_cause() {
        // The classification is the load-bearing half: a timeout says the prompt was large and a
        // connect failure says the host is down, and `should_retry` treats the two the same only
        // because both are worth retrying. A `detail` that grew a cause must not have changed
        // which arm was taken, or an operator would start reading `Timeout` where the truth is
        // `Connect`.
        //
        // Silence is the honest way to produce a timeout: the request is sent and the client's
        // own deadline expires before anything answers.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(120))
            .build()
            .expect("client");
        let error = client
            .get(format!("http://{addr}/"))
            .send()
            .await
            .expect_err("a silent listener must time out");
        let failure = classify_transport(&error);
        let Failure::Retryable { kind, detail } = &failure else {
            panic!("a transport error is always retryable: {failure}");
        };
        assert_eq!(
            *kind,
            RetryKind::Timeout,
            "a timeout must stay a timeout: {failure}"
        );
        // The cause is what the fix adds; the url alone is what it replaced.
        assert!(
            detail.contains("127.0.0.1"),
            "the endpoint should be named: {detail}"
        );
        assert!(
            detail.matches(':').count() > 2,
            "the client's own sentence is the whole detail again: {detail}"
        );
    }

    #[tokio::test]
    async fn a_transport_error_keeps_the_reason_it_was_caused_by() {
        // Live `qna_rails`, 2026-11-16 16:40 and 16:48. Two attempts were abandoned and both
        // reached the journal as the same sentence:
        //
        //   retryable backend failure (BrokenStream): error sending request for url
        //   (http://llm-server:47832/v1/chat/completions)
        //
        // That is `reqwest::Error`'s own `Display`, and it names the url and nothing else. The
        // reason — connection reset, connection refused, DNS, a body that died mid-stream — lives
        // in the error's `source()` chain, which `to_string()` does not walk.
        //
        // `detail` exists for exactly this question (D-125): telling "the backend died" from "the
        // model was truncated" is what the field was added for, and an outage whose cause is
        // invisible cannot be told from any other outage. Two different causes read identically
        // in the journal, so a human reading it cannot tell a flaky network from a bug.
        //
        // A closed port is a real transport failure produced by `reqwest` itself, not a hand-made
        // struct: the test has to exercise the same `Display`/`source` split the live run hit.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        drop(listener); // nothing is listening now, so a connect is refused
        let error = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/chat/completions"))
            .send()
            .await
            .expect_err("a closed port must fail");
        let failure = classify_transport(&error);
        let text = failure.to_string();
        assert!(
            text.contains("refused") || text.contains("connect"),
            "the transport error lost its cause: {text}"
        );
        // The url is not the reason; keeping it is fine, keeping only it is not.
        assert!(
            text.contains(&port.to_string()),
            "the failing endpoint should still be named: {text}"
        );
    }

    #[test]
    fn the_last_permitted_attempt_is_not_retried() {
        let failure = Failure::retryable(RetryKind::ServerError, "x");
        assert!(should_retry(&failure, 1));
        assert!(should_retry(&failure, MAX_ATTEMPTS - 1));
        assert!(!should_retry(&failure, MAX_ATTEMPTS));
    }

    #[test]
    fn a_model_failure_is_not_retried_verbatim() {
        // The retry for a truncated turn changes `max_tokens`; retrying the identical request
        // would reproduce the same truncation. The caller must decide, so this is not retryable.
        let failure = Failure::model(ModelFault::Truncated, "cut off");
        assert!(!should_retry(&failure, 1));
        assert!(!failure.is_retryable());
        assert!(
            !failure.is_fatal(),
            "it is not fatal either — it is a different decision"
        );
    }

    #[test]
    fn a_cancelled_request_is_never_retried() {
        assert!(!should_retry(&Failure::Cancelled, 1));
    }

    #[test]
    fn a_thinking_truncated_turn_is_a_model_failure() {
        let turn = AssistantTurn {
            content: String::new(),
            reasoning: "lots of thinking".into(),
            tool_calls: Vec::new(),
            finish_reason: "length".into(),
            usage: None,
        };
        let failure = classify_turn(&turn).expect("must be a failure");
        assert_eq!(
            failure,
            Failure::model(
                ModelFault::ThinkingTruncated,
                "finish_reason=length, 16 chars of reasoning, empty content and no tool calls"
            )
        );
    }

    #[test]
    fn an_empty_turn_is_a_model_failure_not_a_finished_subtask() {
        let turn = AssistantTurn {
            finish_reason: "stop".into(),
            ..AssistantTurn::default()
        };
        assert_eq!(
            classify_turn(&turn),
            Some(Failure::model(ModelFault::EmptyTurn, "finish_reason=stop"))
        );
    }

    #[test]
    fn a_turn_with_content_and_calls_is_accepted() {
        let turn = AssistantTurn {
            content: "done".into(),
            finish_reason: "stop".into(),
            ..AssistantTurn::default()
        };
        assert_eq!(classify_turn(&turn), None);
    }

    #[test]
    fn a_truncated_turn_with_content_is_still_reported() {
        // Content from a truncated turn ends mid-sentence, and tool arguments from one are
        // often invalid JSON. Accepting it would execute a half-written command.
        let turn = AssistantTurn {
            content: "I will now edit".into(),
            finish_reason: "length".into(),
            ..AssistantTurn::default()
        };
        assert_eq!(
            classify_turn(&turn),
            Some(Failure::model(
                ModelFault::Truncated,
                "finish_reason=length after 15 chars of content"
            ))
        );
    }

    #[test]
    fn a_tool_call_turn_is_accepted_even_with_no_content() {
        let turn = AssistantTurn {
            content: String::new(),
            reasoning: String::new(),
            tool_calls: vec![crate::ToolCall::function("c", "f", "{}")],
            finish_reason: "tool_calls".into(),
            usage: None,
        };
        assert_eq!(classify_turn(&turn), None);
    }

    #[test]
    fn a_long_error_body_is_truncated_in_the_message() {
        let body = "x".repeat(10_000);
        let failure = classify_status(400, &body);
        let text = failure.to_string();
        assert!(
            text.len() < 1000,
            "the message must stay readable: {} bytes",
            text.len()
        );
        assert!(text.contains("10000 chars total"), "{text}");
    }

    #[test]
    fn a_short_error_body_is_kept_whole() {
        let failure = classify_status(400, "bad request");
        assert!(failure.to_string().contains("bad request"));
        assert!(!failure.to_string().contains("chars total"));
    }
}
