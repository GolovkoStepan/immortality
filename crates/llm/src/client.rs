//! The HTTP client: one request, one turn, cancellation, and health.
//!
//! # What it deliberately does not do
//!
//! **It does not retry.** [`Backend::turn`] performs exactly one attempt and returns
//! [`Failure`]. The retry loop lives above it because retrying is a *policy* decision that
//! involves the journal: an attempt that failed mid-stream may have already written a
//! `tool_call_started` record, and the next attempt is then a different protocol step, not a
//! repeat. Putting the loop inside the client would hide that.
//!
//! **It does not hold a semaphore.** Serialising turns is the caller's business: the daemon runs
//! one step at a time, and a client that blocked inside `turn` would hide the wait.
//!
//! # Cancellation
//!
//! [`Backend::turn_streaming`] takes a callback that can end generation mid-answer, and
//! returns [`Failure::Cancelled`] rather than a partial turn. A half-parsed tool call must never
//! reach the executor, for exactly the reason the journal drops a torn tail.

use std::time::Duration;

use futures_util::StreamExt as _;
use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderValue, RETRY_AFTER};

use crate::failure::{Failure, ModelFault, RetryKind, classify_status, classify_transport, classify_turn};
use crate::stream::{StreamDecoder, StreamEnd, StreamError};
use crate::turn::AssistantTurn;
use crate::wire::{ChatRequest, ChatResponse, Message};

/// The `reqwest` client, with the pool lifetime this backend requires.
///
/// Split out so a test can build the same client with a different idle window: the setting is
/// what the regression is about, and a test that could only use the production value would have
/// to wait three seconds to observe the difference. `pool_idle_timeout` is the one knob here that
/// is not merely a timeout — see [`DEFAULT_POOL_IDLE_TIMEOUT`].
fn build_http(
    connect_timeout: Duration,
    request_timeout: Duration,
    pool_idle_timeout: Duration,
) -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .connect_timeout(connect_timeout)
        .timeout(request_timeout)
        .pool_idle_timeout(pool_idle_timeout)
        .build()
}

/// An `OpenAI`-compatible backend.
///
/// Cloned freely: `reqwest::Client` is an `Arc` inside, so a clone shares the connection pool.
#[derive(Debug, Clone)]
pub struct Backend {
    base_url: String,
    http: reqwest::Client,
    /// Bound on the whole request, not just the first byte. See [`DEFAULT_REQUEST_TIMEOUT`].
    request_timeout: Duration,
}

/// Request timeout for the agent loop.
///
/// 600 s, measured: a successful 200K-token prefill took 254 s and a `502` arrived after 300 s.
/// Anything below 600 s turns a large but healthy prefill into a timeout, and a timeout is
/// retryable, so the symptom would be "every large task retries forever" rather than an error.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

/// Connect timeout. Ten seconds is generous for a LAN host and short enough that a dead backend
/// is reported before the first step's patience runs out.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// How long an idle socket may sit in the pool before the client stops trusting it.
///
/// `reqwest`'s default is 90 seconds, and that default assumes the server keeps a silent
/// connection open at least that long. This backend does not: it answers every response with
/// `Keep-Alive: timeout=5`, so it closes an idle socket after five seconds while the pool goes on
/// handing it out for another eighty-five.
///
/// The consequence is not a slow request, it is a failed one. A pooled socket the server has
/// already closed is only discovered when the next request is written into it, and the client
/// reports `connection closed before message completed` from `.send()` — before a single byte of
/// the prompt is sent. Retrying is the daemon's job (D-046), so each of those ends an attempt.
///
/// Measured against the live backend on 2026-11-16 with a six-second gap between requests: 9 of
/// 25 requests failed with exactly that message under the default pool, and 0 of 25 once this
/// setting was below the server's five-second window. On `qna_rails` the same failure abandoned
/// three attempts at 16:40, 16:48 and 17:05, each 6.2-6.5 s after the previous response.
///
/// Three seconds, not five: the client and the server would otherwise be counting down the same
/// five seconds from different starting points, and a socket the server closes a moment before
/// the pool discards it is exactly the race this constant exists to remove.
pub const DEFAULT_POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(3);

impl Backend {
    /// Build a client from configuration.
    ///
    /// # Errors
    /// Returns the underlying `reqwest` error when the TLS backend or the timeouts are invalid,
    /// which is a build-time problem rather than a runtime one.
    pub fn new(
        base_url: impl Into<String>,
        connect_timeout: Duration,
        request_timeout: Duration,
    ) -> Result<Self, reqwest::Error> {
        let http = build_http(connect_timeout, request_timeout, DEFAULT_POOL_IDLE_TIMEOUT)?;
        Ok(Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            http,
            request_timeout,
        })
    }

    /// A client with the project's defaults.
    ///
    /// # Errors
    /// See [`Backend::new`].
    pub fn with_defaults(base_url: impl Into<String>) -> Result<Self, reqwest::Error> {
        Self::new(base_url, DEFAULT_CONNECT_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
    }

    /// The configured base URL, without a trailing slash.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The configured request timeout.
    #[must_use]
    pub const fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    /// List the model ids the backend serves.
    ///
    /// Used by `doctor` and by the daemon at startup: a model name that is not in this list
    /// fails on the first step with whatever the server decides to do, which on Strata was a
    /// `400` that reads like a schema error.
    ///
    /// # Errors
    /// A [`Failure`], classified like any other request.
    pub async fn models(&self) -> Result<Vec<String>, Failure> {
        let url = format!("{}/models", self.base_url);
        let response = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|error| classify_transport(&error))?;

        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(|error| classify_transport(&error))?;
        if !(200..300).contains(&status) {
            return Err(classify_status(status, &body));
        }

        let list: ModelList = serde_json::from_str(&body)
            .map_err(|error| Failure::fatal(format!("`/models` is not the expected JSON: {error}")))?;
        Ok(list
            .data
            .into_iter()
            .map(|item| item.id)
            .filter(|id| !id.is_empty())
            .collect())
    }

    /// One streamed turn, no retries.
    ///
    /// The body is streamed because a 60-second silent `POST` is indistinguishable from a hung
    /// server, while a stream shows progress and can be cut off mid-generation — which is what
    /// reasoning-loop detection needs to do.
    ///
    /// # Errors
    /// A [`Failure`] classified for the caller's retry policy.
    pub async fn turn(&self, request: &ChatRequest) -> Result<AssistantTurn, Failure> {
        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .http
            .post(&url)
            .headers(stream_headers())
            .json(request)
            .send()
            .await
            .map_err(|error| classify_transport(&error))?;

        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let retry_after = retry_after(&response);
            let body = response.text().await.unwrap_or_default();
            let failure = classify_status(status, &body);
            // A `Retry-After` on a 429 is the server telling us how long to wait; ignoring it
            // means either retrying too early (another 429) or sleeping far too long.
            if let (Failure::Retryable { kind, .. }, Some(hint)) = (&failure, retry_after) {
                return Err(Failure::Retryable {
                    kind: *kind,
                    detail: format!("HTTP {status}, Retry-After {hint}s"),
                });
            }
            return Err(failure);
        }

        let mut decoder = StreamDecoder::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| classify_transport(&error))?;
            decoder.push(&chunk).map_err(classify_stream_error)?;
        }

        let saw_finish = decoder.saw_finish_reason();
        let (turn, end) = decoder.finish();
        // A stream that closed without `[DONE]` *and* without a `finish_reason` was cut off. The
        // two conditions are separate on purpose: a server that omits `[DONE]` but sets
        // `finish_reason` closes cleanly, and rejecting that would fail every step.
        if end == StreamEnd::Closed && !saw_finish {
            return Err(Failure::retryable(
                RetryKind::BrokenStream,
                "the stream closed before [DONE] and without a finish_reason",
            ));
        }
        if let Some(failure) = classify_turn(&turn) {
            return Err(failure);
        }
        Ok(turn)
    }

    /// One non-streamed turn.
    ///
    /// For the health probe and for callers that need one complete answer. The agent loop uses
    /// [`Backend::turn`] instead; see the module docs for why.
    ///
    /// # Errors
    /// A [`Failure`] classified for the caller's retry policy.
    pub async fn turn_blocking(&self, request: &ChatRequest) -> Result<AssistantTurn, Failure> {
        let mut request = request.clone();
        request.stream = Some(false);
        request.stream_options = None;

        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .http
            .post(&url)
            .header(CONTENT_TYPE, "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|error| classify_transport(&error))?;

        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(|error| classify_transport(&error))?;
        if !(200..300).contains(&status) {
            return Err(classify_status(status, &body));
        }

        let parsed: ChatResponse = serde_json::from_str(&body).map_err(|error| {
            Failure::retryable(
                RetryKind::MalformedStream,
                format!("a 200 response was not the expected JSON: {error}"),
            )
        })?;
        let turn = parsed
            .into_turn()
            .ok_or_else(|| Failure::model(ModelFault::EmptyTurn, "a 200 response contained no choices"))?;
        if let Some(failure) = classify_turn(&turn) {
            return Err(failure);
        }
        Ok(turn)
    }

    /// A cheap liveness probe: a one-token completion.
    ///
    /// Not `/models`: listing models does not touch the model at all, and the interesting
    /// failure — prefill is alive but generation is wedged — is invisible to it. One token is
    /// enough to prove the whole pipeline runs.
    ///
    /// # Errors
    /// A [`Failure`] if the backend cannot complete a single token.
    pub async fn health(&self, model: &str) -> Result<(), Failure> {
        let mut request = ChatRequest::new(model)
            .with_user("Say OK")
            .with_max_tokens(4)
            .with_thinking(false);
        request.stream = Some(false);
        request.stream_options = None;

        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .http
            .post(&url)
            .header(CONTENT_TYPE, "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|error| classify_transport(&error))?;

        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(|error| classify_transport(&error))?;
        if !(200..300).contains(&status) {
            return Err(classify_status(status, &body));
        }
        // Deliberately not `classify_turn`: a four-token budget legitimately truncates, and the
        // probe asks only whether the backend can produce anything at all.
        let parsed: ChatResponse = serde_json::from_str(&body).map_err(|error| {
            Failure::retryable(
                RetryKind::MalformedStream,
                format!("health probe returned non-JSON: {error}"),
            )
        })?;
        if parsed.choices.is_empty() {
            return Err(Failure::model(
                ModelFault::EmptyTurn,
                "health probe returned no choices",
            ));
        }
        Ok(())
    }

    /// A streamed turn whose callback sees each new piece of content as it arrives.
    ///
    /// This is what reasoning-loop detection (D4) needs: it must be able to abort generation
    /// mid-sentence, not after the model has written five thousand tokens of the same paragraph.
    /// The callback returns `false` to abort, and an aborted stream returns
    /// [`Failure::Cancelled`] with no partial turn.
    ///
    /// The callback runs per SSE payload, not per transport chunk: `reqwest` may coalesce
    /// several events into one `Bytes`, and a detector driven by chunk boundaries would see a
    /// hundred characters at once and could only stop after all of them had arrived.
    ///
    /// # Errors
    /// A [`Failure`] classified for the caller's retry policy.
    pub async fn turn_streaming<F>(
        &self,
        request: &ChatRequest,
        mut on_text: F,
    ) -> Result<AssistantTurn, Failure>
    where
        F: FnMut(&str) -> bool,
    {
        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .http
            .post(&url)
            .headers(stream_headers())
            .json(request)
            .send()
            .await
            .map_err(|error| classify_transport(&error))?;

        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let body = response.text().await.unwrap_or_default();
            return Err(classify_status(status, &body));
        }

        let mut decoder = StreamDecoder::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| classify_transport(&error))?;
            let keep_going = decoder
                .push_with(&chunk, &mut on_text)
                .map_err(classify_stream_error)?;
            if !keep_going {
                return Err(Failure::Cancelled);
            }
        }

        let saw_finish = decoder.saw_finish_reason();
        let (turn, end) = decoder.finish();
        if end == StreamEnd::Closed && !saw_finish {
            return Err(Failure::retryable(
                RetryKind::BrokenStream,
                "the stream closed before [DONE] and without a finish_reason",
            ));
        }
        if let Some(failure) = classify_turn(&turn) {
            return Err(failure);
        }
        Ok(turn)
    }

    /// A conversational round-trip used by tests and probes: send messages, get a turn back.
    ///
    /// # Errors
    /// See [`Backend::turn_blocking`].
    pub async fn chat(&self, model: &str, messages: Vec<Message>) -> Result<AssistantTurn, Failure> {
        let mut request = ChatRequest::new(model).with_thinking(false);
        request.messages = messages;
        self.turn_blocking(&request).await
    }
}

/// Convert a decoder error into a classified failure.
///
/// A malformed payload and an in-band backend error are both retryable, for different reasons:
/// the first is usually a proxy mangling a chunk, the second is the backend giving up mid-turn.
/// Neither is fatal, because the request itself was well formed.
fn classify_stream_error(error: StreamError) -> Failure {
    match error {
        StreamError::Malformed { detail } => Failure::retryable(RetryKind::MalformedStream, detail),
        StreamError::Backend { detail } => Failure::retryable(RetryKind::BackendErrorEvent, detail),
    }
}

/// Headers for a streamed request.
///
/// `Accept: text/event-stream` is what makes some servers actually stream instead of buffering
/// the whole answer and sending it as one chunk — which looks like working streaming in a test
/// and like a hang in production.
fn stream_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers
}

/// The `/models` response, as far as it matters here.
///
/// `object: "list"` and every field except `id` are ignored: this exists to answer "does the
/// configured model exist", not to describe the server.
#[derive(Debug, serde::Deserialize)]
struct ModelList {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

#[derive(Debug, serde::Deserialize)]
struct ModelEntry {
    #[serde(default)]
    id: String,
}

/// Read `Retry-After` as whole seconds.
///
/// The HTTP-date form is legal and is ignored here: this backend never sends it, and parsing a
/// date to decide a one-second sleep is not worth the failure mode of parsing it wrong.
fn retry_after(response: &reqwest::Response) -> Option<u64> {
    response
        .headers()
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_base_url_loses_its_trailing_slash() {
        // Otherwise every URL becomes `//chat/completions`, which a proxy may 404 while the
        // server it fronts would have accepted it — a bug that only appears behind a proxy.
        let backend = Backend::with_defaults("http://host:1/v1/").unwrap();
        assert_eq!(backend.base_url(), "http://host:1/v1");
    }

    #[test]
    fn a_pooled_socket_is_never_trusted_past_the_servers_keep_alive() {
        // Live `qna_rails`, 2026-11-16. Three attempts were abandoned at 16:40, 16:48 and 17:05,
        // each 6.2-6.5 s after the previous response, all with the same sentence:
        //
        //   retryable backend failure (BrokenStream): error sending request for url (…):
        //   client error (SendRequest): connection closed before message completed
        //
        // The backend answers every response with `Keep-Alive: timeout=5`, so it drops an idle
        // socket after five seconds while `reqwest` keeps it for ninety. `.send()` then finds the
        // connection dead before one byte of the prompt is written — which is what made this a
        // failed attempt rather than a slow one, since retrying belongs to the daemon (D-046).
        //
        // Measured against the live backend, six seconds apart: 9 of 25 requests failed with that
        // message under the default pool, 0 of 25 with the pool below the server's window.
        //
        // # Why this is a constant and not a socket test
        //
        // A local fake cannot reproduce it. On loopback the server's FIN reaches the client at
        // once, and `hyper` marks the connection closed before it returns to the pool — the
        // failing case never occurs, so a test built on a fake server passes with the fix
        // removed. The live failure needs the FIN to be late or absent, which is what a real
        // network path does and a loopback does not. The honest test is therefore the rule
        // itself: the pool must let go strictly before the server says it will.
        assert!(
            DEFAULT_POOL_IDLE_TIMEOUT < Duration::from_secs(5),
            "the pool must let go before the server's five-second keep-alive, or it hands out \
             sockets the server has already closed: {DEFAULT_POOL_IDLE_TIMEOUT:?}"
        );
    }

    #[test]
    fn the_default_timeout_is_the_measured_one() {
        // A regression to 30 s or 120 s would look like a performance problem, not a bug.
        assert_eq!(DEFAULT_REQUEST_TIMEOUT, Duration::from_secs(600));
    }
}
