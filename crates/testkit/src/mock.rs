//! A scripted `OpenAI`-compatible server for tests.
//!
//! # Why this exists
//!
//! The retry policy, the stream decoder and the error classifier are only correct together, and
//! the cases that matter cannot be produced by a happy-path mock: a stream that dies halfway, a
//! `200` whose first payload is an error event, a `429` with `Retry-After`, a `502` after a long
//! silence. This server answers a *script*, so a test says what the backend does step by step
//! and then asserts what the client did about it.
//!
//! # Deliberately hand-rolled
//!
//! A web framework would test the framework's HTTP stack and bring a dependency that exists only
//! in tests. The surface needed here is one endpoint, one method, one header set — thirty lines
//! of `AsyncRead`/`AsyncWrite`. What it does implement is the awkward part: it reads the request
//! *body* so a test can assert what was actually sent, including `chat_template_kwargs` and
//! `stream_options`.
//!
//! # Not a fake model
//!
//! It never generates anything. A test that needs "the model loops" writes a script whose
//! chunks repeat; a test that needs "the model thinks too long" writes `finish_reason: length`
//! with reasoning. The script is the test's statement about the backend, not the mock's
//! cleverness — which is what makes a failure point at the client.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::{TcpListener, TcpStream};

/// One scripted answer.
///
/// The variants are the shapes that break clients, not the shapes a model produces: everything
/// here was either observed on the live backend or is the failure mode the retry policy exists
/// for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Reply {
    /// A complete SSE stream: role delta, then `chunks` as `data:` payloads, then a finish
    /// chunk and `[DONE]`.
    ///
    /// `chunks` are raw JSON payload strings, so a test can produce interleaved tool calls,
    /// character-by-character arguments, an in-band error event — anything.
    Sse {
        /// Payloads sent as `data: <payload>\n\n` before the terminator.
        payloads: Vec<String>,
        /// Sent in the final chunk as `finish_reason`, before `[DONE]`.
        finish_reason: String,
    },
    /// An SSE stream cut off after `payloads`, with no finish chunk and no `[DONE]`.
    SseCut { payloads: Vec<String> },
    /// A complete stream that sends `[DONE]` immediately, with no content at all.
    SseEmpty,
    /// A non-streamed JSON body with the given status.
    Json { status: u16, body: String },
    /// An error status with a plain-text body.
    Status { status: u16, body: String },
    /// An error status with a `Retry-After` header.
    RateLimited { retry_after: u64 },
    /// Accept the connection and close it without a byte of response.
    HangUp,
    /// Wait `delay` and then answer with `then`.
    Slow { delay: Duration, then: Box<Self> },
}

/// Turn a recorded trace into a script.
///
/// # Why the mock can be driven by a fixture
///
/// The seven S1 traces are what the backend really said. Feeding them back through the client is
/// the closest a test can get to the live system without one — and unlike a hand-written reply it
/// cannot drift away from the protocol, because it *is* the protocol as captured.
///
/// A step marked `fatal` becomes an in-band error: that is how the spike recorded a failing
/// request, and it is deliberately not a status code — the traces do not record one, and inventing
/// a `500` would be a claim the fixture does not support.
///
/// # Errors
/// A malformed trace line, or a step with neither a response nor a fault and therefore nothing to
/// serve. The second is an error rather than a skip because a script that silently drops a step
/// makes the following request answer with the wrong reply, which is a confusing way to fail.
pub fn script_from_trace(text: &str) -> Result<Vec<Reply>, serde_json::Error> {
    let steps: Vec<crate::TraceStep> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;

    Ok(steps
        .into_iter()
        .map(|step| match (step.fatal, step.resp) {
            (Some(fault), _) => Reply::in_band_error(&fault),
            (None, Some(resp)) => Reply::from_trace_response(&resp),
            (None, None) => Reply::HangUp,
        })
        .collect())
}

impl Reply {
    /// A streamed answer whose content is exactly `content`, split into one chunk per character.
    ///
    /// One character per chunk is the live backend's actual behaviour and the cruellest
    /// chunking a decoder has to survive.
    #[must_use]
    pub fn streamed_text(content: &str) -> Self {
        Self::Sse {
            payloads: content
                .chars()
                .map(|ch| {
                    serde_json::json!({"choices":[{"index":0,"delta":{"content":ch.to_string()}}]})
                        .to_string()
                })
                .collect(),
            finish_reason: "stop".to_string(),
        }
    }

    /// A streamed answer containing one tool call whose arguments arrive one character at a time.
    #[must_use]
    pub fn streamed_tool_call(id: &str, name: &str, arguments: &str) -> Self {
        let mut payloads = vec![
            serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{
                "index":0,"id":id,"type":"function",
                "function":{"name":name,"arguments":""}
            }]}}]})
            .to_string(),
        ];
        for ch in arguments.chars() {
            payloads.push(
                serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{
                    "index":0,"function":{"arguments":ch.to_string()}
                }]}}]})
                .to_string(),
            );
        }
        Self::Sse {
            payloads,
            finish_reason: "tool_calls".to_string(),
        }
    }

    /// A complete non-streamed answer, as [`crate::llm_reply`] helpers produce it.
    #[must_use]
    pub fn json_turn(content: &str) -> Self {
        Self::Json {
            status: 200,
            body: serde_json::json!({
                "id": "chatcmpl-mock",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 24, "completion_tokens": 3, "total_tokens": 27},
                "timings": {"cache_n": 20}
            })
            .to_string(),
        }
    }

    /// A reply that arrives only after `delay`.
    #[must_use]
    pub fn slow(delay: Duration, then: Self) -> Self {
        Self::Slow {
            delay,
            then: Box::new(then),
        }
    }

    /// A `429` that asks the client to wait.
    #[must_use]
    pub const fn rate_limited(retry_after: u64) -> Self {
        Self::RateLimited { retry_after }
    }

    /// A `200` whose stream reports an error in-band, after the status was already sent.
    ///
    /// The failure an HTTP client cannot see: the status is fine and the body says "no".
    #[must_use]
    pub fn in_band_error(message: &str) -> Self {
        Self::Sse {
            payloads: vec![
                serde_json::json!({"error":{"message":message,"type":"server_error"}}).to_string(),
            ],
            finish_reason: "stop".to_string(),
        }
    }

    /// Render one recorded `resp` object as the stream the backend would have sent.
    ///
    /// The shape is the measured one, widened to whole-payload chunks: the per-character chunking
    /// the live backend really does is covered by the unit tests and by `Reply::streamed_text`,
    /// while a replayed trace wants to fail on its *data*, not on its chunking.
    #[must_use]
    pub fn from_trace_response(resp: &serde_json::Value) -> Self {
        let mut payloads = Vec::new();

        if let Some(reasoning) = resp
            .get("reasoning_content")
            .or_else(|| resp.get("reasoning"))
            .and_then(serde_json::Value::as_str)
            && !reasoning.is_empty()
        {
            payloads.push(
                serde_json::json!({"choices":[{"index":0,"delta":{"reasoning_content":reasoning}}]})
                    .to_string(),
            );
        }
        if let Some(content) = resp.get("content").and_then(serde_json::Value::as_str)
            && !content.is_empty()
        {
            payloads
                .push(serde_json::json!({"choices":[{"index":0,"delta":{"content":content}}]}).to_string());
        }
        for (index, call) in resp
            .get("tool_calls")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            payloads.push(
                serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{
                    "index": index,
                    "id": call.get("id").and_then(serde_json::Value::as_str).unwrap_or(""),
                    "type": "function",
                    "function": {
                        "name": call.get("function").and_then(|f| f.get("name"))
                            .and_then(serde_json::Value::as_str).unwrap_or(""),
                        "arguments": call.get("function").and_then(|f| f.get("arguments"))
                            .and_then(serde_json::Value::as_str).unwrap_or(""),
                    }
                }]}}]})
                .to_string(),
            );
        }

        let finish_reason = resp
            .get("finish_reason")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("stop")
            .to_string();
        Self::Sse {
            payloads,
            finish_reason,
        }
    }

    /// The same tool call, forever.
    ///
    /// Detector D1 fires on the third identical `(name, args)`; this reply exists so that test has
    /// a backend which never gets bored. It is the one scripted shape that must *not* be finite:
    /// a loop detector that is fed a script which runs out would pass by the script ending rather
    /// than by the detector working.
    #[must_use]
    pub fn looping_tool_call(name: &str, arguments: &str) -> Self {
        Self::streamed_tool_call("call_loop", name, arguments)
    }

    /// The same sentence, repeated until the caller stops reading.
    ///
    /// Reasoning-loop detection (D4) has the same requirement as D1: the backend must keep
    /// producing, or a detector that never fires looks identical to one that fired.
    #[must_use]
    pub fn looping_text(sentence: &str, repeats: usize) -> Self {
        Self::streamed_text(&sentence.repeat(repeats))
    }
}

/// A request the server received.
#[derive(Debug, Clone)]
pub struct ReceivedRequest {
    /// The request path, e.g. `/v1/chat/completions`.
    pub path: String,
    /// The body, parsed as JSON when it was JSON.
    pub body: Option<serde_json::Value>,
    /// The body exactly as sent, for asserting byte-level details.
    pub raw_body: String,
}

/// A running mock server.
///
/// Dropping it aborts the accept loop; the port is released by the runtime, so tests do not have
/// to join anything.
#[derive(Debug)]
pub struct MockServer {
    addr: std::net::SocketAddr,
    /// Every request, in arrival order.
    received: Arc<Mutex<Vec<ReceivedRequest>>>,
    task: tokio::task::JoinHandle<()>,
}

impl MockServer {
    /// Start with a script. The last reply repeats for every further request, so a test about
    /// retries does not have to enumerate the retry count.
    ///
    /// Binds port 0 so tests run in parallel, and only on `127.0.0.1`.
    pub async fn start(script: Vec<Reply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a loopback port");
        let addr = listener.local_addr().expect("a bound socket has an address");
        let received: Arc<Mutex<Vec<ReceivedRequest>>> = Arc::new(Mutex::new(Vec::new()));
        let queue = Arc::new(Mutex::new(VecDeque::from(script)));

        let recorder = Arc::clone(&received);
        let task = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let reply = {
                    let mut queue = queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                    // The last reply repeats: a script that ends in "500" describes a backend
                    // that stays broken, not one that recovers on the second request.
                    if queue.len() > 1 {
                        queue.pop_front()
                    } else {
                        queue.front().cloned()
                    }
                };
                let Some(reply) = reply else { continue };

                let request = read_request(&mut stream).await;
                if let Some(request) = request {
                    recorder
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(request);
                }
                serve(&mut stream, reply).await;
                let _ = stream.shutdown().await;
            }
        });

        Self { addr, received, task }
    }

    /// A server that always answers with one reply.
    pub async fn single(reply: Reply) -> Self {
        Self::start(vec![reply]).await
    }

    /// The base URL, including the `/v1` suffix the client expects.
    #[must_use]
    pub fn base_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    /// The bare authority, for a caller that builds its own URL.
    #[must_use]
    pub const fn addr(&self) -> std::net::SocketAddr {
        self.addr
    }

    /// Every request received so far, in order.
    #[must_use]
    pub fn requests(&self) -> Vec<ReceivedRequest> {
        self.received
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// How many requests arrived.
    #[must_use]
    pub fn request_count(&self) -> usize {
        self.received
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// The most recent request body, or `None` if nothing arrived.
    #[must_use]
    pub fn last_body(&self) -> Option<serde_json::Value> {
        self.requests().last().and_then(|request| request.body.clone())
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        // The accept loop is the only thing holding the port; aborting it is the whole cleanup.
        self.task.abort();
    }
}

/// Read one HTTP/1.1 request, including its body.
async fn read_request(stream: &mut TcpStream) -> Option<ReceivedRequest> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];

    // Read headers.
    let header_end = loop {
        let read = stream.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(position) = find_double_crlf(&buffer) {
            break position;
        }
    };

    let head = String::from_utf8_lossy(&buffer[..header_end]).to_string();
    let path = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("")
        .to_string();

    let content_length = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())?
        })
        .unwrap_or(0);

    let body_start = header_end + 4;
    while buffer.len() < body_start + content_length {
        let read = stream.read(&mut chunk).await.ok()?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    let raw_body = String::from_utf8_lossy(
        buffer
            .get(body_start..body_start + content_length)
            .unwrap_or_default(),
    )
    .to_string();

    Some(ReceivedRequest {
        path,
        body: serde_json::from_str(&raw_body).ok(),
        raw_body,
    })
}

/// The byte offset of the `\r\n\r\n` that ends the header block.
fn find_double_crlf(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

/// Write one scripted reply.
async fn serve(stream: &mut TcpStream, reply: Reply) {
    match reply {
        Reply::HangUp => {}
        Reply::Status { status, body } => {
            write_plain(stream, status, "text/plain", &body).await;
        }
        Reply::Json { status, body } => {
            write_plain(stream, status, "application/json", &body).await;
        }
        Reply::RateLimited { retry_after } => {
            let response = format!(
                "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\n\
                 Retry-After: {retry_after}\r\nContent-Length: 22\r\nConnection: close\r\n\r\n\
                 {{\"error\":\"rate limit\"}}"
            );
            let _ = stream.write_all(response.as_bytes()).await;
        }
        Reply::Sse {
            payloads,
            finish_reason,
        } => {
            write_sse_head(stream).await;
            for payload in payloads {
                if write_sse_chunk(stream, &payload).await.is_err() {
                    return;
                }
            }
            let finish = serde_json::json!({
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 7, "total_tokens": 47},
                "timings": {"cache_n": 32}
            });
            if write_sse_chunk(stream, &finish.to_string()).await.is_err() {
                return;
            }
            let _ = stream.write_all(b"data: [DONE]\n\n").await;
            let _ = stream.flush().await;
        }
        Reply::SseCut { payloads } => {
            write_sse_head(stream).await;
            for payload in payloads {
                if write_sse_chunk(stream, &payload).await.is_err() {
                    return;
                }
            }
            // No finish, no `[DONE]`: the connection ends mid-answer.
            let _ = stream.flush().await;
        }
        Reply::SseEmpty => {
            write_sse_head(stream).await;
            let _ = stream.write_all(b"data: [DONE]\n\n").await;
            let _ = stream.flush().await;
        }
        Reply::Slow { delay, then } => {
            tokio::time::sleep(delay).await;
            Box::pin(serve(stream, *then)).await;
        }
    }
}

async fn write_plain(stream: &mut TcpStream, status: u16, content_type: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        reason(status),
        body.len()
    );
    let _ = stream.write_all(response.as_bytes()).await;
    let _ = stream.flush().await;
}

async fn write_sse_head(stream: &mut TcpStream) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
    let _ = stream.write_all(head.as_bytes()).await;
}

async fn write_sse_chunk(stream: &mut TcpStream, payload: &str) -> std::io::Result<()> {
    stream
        .write_all(format!("data: {payload}\n\n").as_bytes())
        .await?;
    stream.flush().await
}

/// The reason phrase for the statuses this server produces.
///
/// A real server always sends one; a client that parses the status line strictly would reject a
/// bare `HTTP/1.1 500\r\n`, and the mock should not be the reason a test fails.
const fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        408 => "Request Timeout",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Unknown",
    }
}
