//! The client half of the unix-socket protocol.
//!
//! # Why the client performs exactly one attempt (D-046)
//!
//! The retry loop belongs to the daemon, which owns the attempt budget and the journal. A
//! client that retried on its own would turn one flaky connection into N attempts that the
//! daemon never counted, and — worse — a client retrying a `start` could ask twice whether it
//! had already succeeded. So this module connects, says `hello`, sends the request, reads the
//! answer, and returns. Whether to try again is the caller's decision, and the only caller is
//! a human at a terminal who can see what happened.
//!
//! # Why the token is read rather than passed in
//!
//! It lives in `daemon.token` next to the socket, mode `0600`, and is the daemon's own secret. A
//! CLI flag for it would put a secret in the shell history and in `ps` output for no benefit: the
//! client is on the same machine by construction, because that is what a unix socket is.
//!
//! # Why a missing socket is a specific error
//!
//! "No daemon running" is the single most common failure of every command here, and the
//! difference between `ENOENT` from `connect` and `ConnectionRefused` is the difference between
//! "start the daemon" and "the socket is a corpse". Collapsing both into "connection failed"
//! sends the human to the wrong fix.

// `pub`, not `pub(crate)`: this is a binary crate, so the module is only reachable from the
// crate root and `pub(crate)` on its items is what `clippy::redundant_pub_crate` objects to.
// `unreachable_pub` then objects to `pub`. The pair is unsatisfiable for a private module in a
// binary, so the reachability lint is allowed here and only here.
#![allow(unreachable_pub)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use immortality_daemon::rpc::{self, ErrorCode, Request, Response};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader};
use tokio::net::UnixStream;

/// How long a command waits for the daemon before giving up.
///
/// A `status` answers in microseconds, so a long wait means the daemon is wedged. Hanging the
/// terminal forever is the failure mode this project exists to avoid.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a request that asks the daemon to **think** may take.
///
/// `plan --goal` and `start "<goal>"` call the model inside the request: the planner writes up to
/// `MAX_PLAN_TOKENS` of subtasks, and the backend's own timeout is 600 s. Measured on 2026-11-16
/// against the live backend, the same one-line goal took 22.9 s once and 29.6 s another time, and
/// then three times in a row the client gave up at 30.00 s — while the daemon finished the plan
/// seconds later, so the human saw "the daemon did not answer within 30 seconds" over a plan that
/// had in fact been written. A timeout that fires just before success is worse than no timeout:
/// it invites a retry that pays for a second plan.
///
/// Bounded rather than absent. `attach` waits forever because silence there is the model thinking
/// and the stream is the product; a request needs an answer, and a daemon wedged in the planner
/// must still be reported.
const THINKING_TIMEOUT: Duration = Duration::from_mins(15);

/// How long a request may wait, given what it asks the daemon to do.
///
/// Only two shapes make the daemon call the model before it answers, and both are the same
/// question — "build a plan": `plan` carries a `goal`, and `start` carries a `goal` rather than a
/// `task`. Everything else answers from disk. The distinction is by *shape* rather than by name
/// because both methods have a second shape that answers at once (`plan` without a goal reports
/// the tree on disk; `start --task` resumes), and a name-only rule would make the fast shape wait
/// a quarter of an hour for a socket that has already closed.
///
/// Measured on 2026-11-16 against the live backend: the planning shape took 22.9 s once and
/// 29.6 s another time, then three runs in a row hit the 30-second deadline while the daemon
/// finished and wrote the plan seconds later. A deadline that fires just before success invites a
/// retry that pays for a second plan.
fn deadline_for(method: &str, params: &Value, answer: Duration) -> Duration {
    let plans = match method {
        "plan" | "start" => params.get("goal").is_some(),
        _ => false,
    };
    if plans { THINKING_TIMEOUT } else { answer }
}

/// How many reconnects that deliver nothing in a row are tolerated before giving up.
///
/// A daemon that closes every connection immediately is broken. One reconnect is the normal
/// lagged-subscriber case; more than a few with no record between them is a loop, and a silent
/// loop is worse than an error message.
const MAX_FRUITLESS_RECONNECTS: u32 = 3;

/// Something the client has to say that is not a journal record.
///
/// Only reconnection so far. It exists because silence on screen means two very different things
/// — "the model is thinking" and "the stream died" — and a run lasts hours, so the human will
/// eventually be looking at both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Notice {
    /// The stream ended and a reconnect is being attempted from `from`.
    Reconnecting {
        /// The first sequence number the client has not seen.
        from: u64,
        /// Which consecutive attempt this is, counting from 1.
        attempt: u32,
    },
}

/// A connection to a running daemon.
pub struct Client {
    write: tokio::net::unix::OwnedWriteHalf,
    reader: BufReader<tokio::net::unix::OwnedReadHalf>,
    next_id: u64,
    token: String,
    /// Remembered so a stream that the daemon closes can be re-opened. The file descriptor
    /// cannot be read back into a path, so the path is kept rather than derived.
    socket: PathBuf,
    /// Deadline for a *request* answer.
    ///
    /// A field rather than the constant, so a test can shorten it. It must never be applied to
    /// the event stream: an attached run is idle for as long as the model thinks, and a cold
    /// prefill here was measured at 166 s.
    answer_timeout: Duration,
}

impl Client {
    /// Connect to `socket` and prove the token before returning.
    ///
    /// # Errors
    /// The daemon is not running, the socket exists but cannot be reached, the token file is
    /// missing, or the token is refused. Each has its own message because each has its own fix.
    pub async fn connect(socket: &Path, token_path: &Path) -> Result<Self> {
        // The socket is checked first, and this order is load-bearing. A machine that never ran
        // a daemon has neither file, and reporting the missing token would send the human to
        // fix a secret rather than to start the daemon — the wrong problem, and one they cannot
        // solve by reading the message.
        if !socket.exists() {
            bail!(
                "no daemon is running on {}.\n\
                 Start one with `immortality serve`; nothing was changed.",
                socket.display()
            );
        }
        let token = read_token(token_path)?;
        let mut client = Self::open(socket, token).await?;
        // `hello` is the handshake, the token check *and* the version negotiation: a wrong token
        // or a different protocol is refused here, before any command runs, rather than after it
        // looked like it was going to work. The version is sent, not merely compared afterwards:
        // the daemon is the side that can say which of the two problems this is.
        let response = client
            .request("hello", json!({ "version": rpc::VERSION }))
            .await?;
        if let Some(error) = response.error {
            // The distinction matters to a human: a version complaint means both sides are whole
            // and disagreeing, so restoring the token would change nothing.
            if error.code == rpc::ErrorCode::VersionMismatch {
                bail!(
                    "{}\n\
                     Restart the daemon with this binary; nothing was changed.",
                    error.message
                );
            }
            bail!(
                "the daemon refused this client: {}\n\
                 The token in {} does not match the running daemon. Restart the daemon, or\n\
                 restore the token file it was started with.",
                error.message,
                token_path.display()
            );
        }
        let version = response
            .result
            .as_ref()
            .and_then(|result| result.get("version"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        // A daemon too old to negotiate answered without a version or with an older one. Kept as
        // a second check rather than replaced by the request field: an older daemon silently
        // ignores `params`, and silence is not agreement.
        if version != u64::from(rpc::VERSION) {
            bail!(
                "the daemon speaks protocol {version} and this client speaks {}.\n\
                 A client from a newer build cannot talk to an older daemon: restart the daemon\n\
                 with the same build as this binary.",
                rpc::VERSION
            );
        }
        Ok(client)
    }

    /// The socket connection itself, without the handshake.
    async fn open(socket: &Path, token: String) -> Result<Self> {
        let stream = UnixStream::connect(socket)
            .await
            .map_err(|error| match error.kind() {
                std::io::ErrorKind::NotFound => anyhow::anyhow!(
                    "no daemon is running on {}.\n\
                 Start one with `immortality serve`; nothing was changed.",
                    socket.display()
                ),
                std::io::ErrorKind::PermissionDenied => anyhow::anyhow!(
                    "{} exists but is not reachable by this user.\n\
                 The socket is mode 0600 by design: run the client as the user who runs the\n\
                 daemon.",
                    socket.display()
                ),
                std::io::ErrorKind::ConnectionRefused => anyhow::anyhow!(
                    "{} exists but nothing is accepting connections on it.\n\
                 A previous daemon probably died without cleaning up; start a fresh one.",
                    socket.display()
                ),
                _ => anyhow::anyhow!("cannot connect to {}: {error}", socket.display()),
            })?;
        let (read, write) = stream.into_split();
        Ok(Self {
            write,
            reader: BufReader::new(read),
            next_id: 0,
            token,
            socket: socket.to_path_buf(),
            answer_timeout: ANSWER_TIMEOUT,
        })
    }

    /// Send one request and read exactly one response.
    ///
    /// The deadline depends on what the request asks the daemon to do; see [`deadline_for`]. It is
    /// decided here rather than at each call site so a call site cannot forget it, and so the two
    /// are testable together through this one function.
    ///
    /// # Errors
    /// The socket died mid-request, the daemon sent a message that is not a response, or the
    /// daemon did not answer within the deadline its method allows.
    pub async fn request(&mut self, method: &str, params: Value) -> Result<Response> {
        let deadline = deadline_for(method, &params, self.answer_timeout);
        self.request_within(method, params, deadline).await
    }

    async fn request_within(&mut self, method: &str, params: Value, deadline: Duration) -> Result<Response> {
        self.next_id += 1;
        let request = Request {
            id: self.next_id,
            method: method.to_string(),
            params,
            token: Some(self.token.clone()),
        };
        let line = rpc::encode(&request).map_err(|error| anyhow::anyhow!("{error}"))?;
        self.write
            .write_all(format!("{line}\n").as_bytes())
            .await
            .context("the daemon closed the connection while sending the request")?;
        self.write
            .flush()
            .await
            .context("the daemon closed the connection while sending the request")?;
        let line = self.read_line_within(Some(deadline)).await?;
        rpc::decode(line.trim_end()).map_err(|error| anyhow::anyhow!("{error}"))
    }

    /// Attach and hand every journal record to `on_event` until the terminal is interrupted.
    ///
    /// Reconnects on its own when the daemon closes the stream because this client fell behind:
    /// the daemon's contract is that `since_seq` makes a reconnect lossless, so a human watching
    /// a run should not have to notice and restart the command. The number of reconnects that
    /// deliver no record at all is bounded, because a daemon that closes every connection
    /// immediately is broken and a silent retry loop would hide it forever.
    ///
    /// # Errors
    /// The daemon refuses the attach, the stream dies for a reason a reconnect cannot fix, or
    /// the daemon cannot be reached again.
    pub async fn attach(
        &mut self,
        task: &str,
        since_seq: u64,
        mut on_event: impl FnMut(&Value),
        mut on_notice: impl FnMut(Notice),
    ) -> Result<()> {
        let mut from = since_seq;
        let mut fruitless_reconnects = 0u32;
        loop {
            let response = self
                .request("attach", json!({"task": task, "since_seq": from}))
                .await?;
            let result = match (response.result, response.error) {
                (Some(result), _) => result,
                (None, Some(error)) => bail!("{}", explain(&error)),
                (None, None) => {
                    bail!("the daemon answered attach with neither a result nor an error")
                }
            };
            for record in result
                .get("backlog")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or_default()
            {
                on_event(record);
            }
            from = result.get("next_seq").and_then(Value::as_u64).unwrap_or(from);

            // The acknowledgement is done; the connection is the event stream now.
            let before = from;
            self.stream(&mut from, &mut on_event).await?;
            if from == before {
                fruitless_reconnects += 1;
                if fruitless_reconnects > MAX_FRUITLESS_RECONNECTS {
                    bail!(
                        "the daemon closed the stream {fruitless_reconnects} times in a row without\n\
                         sending a single event. It is running but not streaming; restart it."
                    );
                }
            } else {
                fruitless_reconnects = 0;
            }
            tracing::info!("the daemon closed the stream; re-attaching from {from}");
            // The screen would otherwise simply stop changing, and a stopped screen is
            // indistinguishable from a model that is thinking. Saying it out loud is the
            // difference between "wait" and "the daemon died".
            on_notice(Notice::Reconnecting {
                from,
                // The count was just incremented, so this is the attempt about to be made:
                // the first reconnection announces itself as 1, not as 2.
                attempt: fruitless_reconnects,
            });
            *self = Self::reopen(&self.socket, &self.token).await?;
        }
    }

    /// Read events until the daemon closes the stream.
    ///
    /// Every close is the same to the client — `read_line` returns EOF either way — so this
    /// reports nothing and the caller decides. What distinguishes "fell behind" from "daemon
    /// gone" is whether the reconnect succeeds, which is the only reliable test and is why the
    /// reconnect is attempted rather than guessed at.
    async fn stream(&mut self, from: &mut u64, on_event: &mut impl FnMut(&Value)) -> Result<()> {
        loop {
            // No deadline: this is the event stream, not a request. An attached run is legitimately
            // silent for as long as the model thinks, and a cold prefill on this backend measured
            // 166 s. A 30-second answer deadline here would disconnect a healthy run.
            let line = match self.read_line_within(None).await {
                Ok(line) => line,
                Err(error) if error.to_string().contains("closed the connection") => return Ok(()),
                Err(error) => return Err(error),
            };
            let event: rpc::Event =
                rpc::decode(line.trim_end()).map_err(|error| anyhow::anyhow!("{error}"))?;
            if let Some(seq) = event.record.get("seq").and_then(Value::as_u64) {
                *from = seq + 1;
            }
            on_event(&event.record);
        }
    }

    /// Re-open a connection and repeat the handshake.
    async fn reopen(socket: &Path, token: &str) -> Result<Self> {
        let mut client = Self::open(socket, token.to_string()).await.with_context(|| {
            format!(
                "the daemon closed the stream and {} is no longer reachable; is it still \
                     running?",
                socket.display()
            )
        })?;
        let response = client
            .request("hello", json!({ "version": rpc::VERSION }))
            .await?;
        if let Some(error) = response.error {
            bail!("the daemon refused the reconnect: {}", error.message);
        }
        Ok(client)
    }

    /// Read one line, waiting at most `deadline`.
    ///
    /// `None` means "wait as long as the daemon needs", and the stream passes `None`: a human
    /// watching a run expects silence while the model works, not a client that gives up on it.
    async fn read_line_within(&mut self, deadline: Option<Duration>) -> Result<String> {
        let mut line = String::new();
        let read = match deadline {
            Some(deadline) => tokio::time::timeout(deadline, self.reader.read_line(&mut line))
                .await
                .map_err(|_| {
                    anyhow::anyhow!("the daemon did not answer within {} seconds", deadline.as_secs())
                })??,
            None => self.reader.read_line(&mut line).await?,
        };
        if read == 0 {
            bail!("the daemon closed the connection");
        }
        Ok(line)
    }
}

/// Read the token file.
///
/// # Errors
/// The file is missing or empty. A missing token is almost always a state root that was never
/// initialised, so the message says that rather than "permission denied" one layer later.
fn read_token(path: &Path) -> Result<String> {
    let text = std::fs::read_to_string(path).with_context(|| {
        format!(
            "cannot read {}.\n\
             This file holds the daemon's token; if it is missing, no daemon was ever started\n\
             for this state directory.",
            path.display()
        )
    })?;
    let token = text.trim();
    if token.is_empty() {
        bail!(
            "{} is empty, so no client can prove it belongs here",
            path.display()
        );
    }
    Ok(token.to_string())
}

/// Render an RPC error for a human.
///
/// The `NotFound` and `Unauthorized` cases get a next step because they are the two a human can
/// fix without reading source: the task id was wrong, or the daemon was started by someone else.
#[must_use]
pub fn explain(error: &rpc::RpcError) -> String {
    match error.code {
        ErrorCode::NotFound => format!(
            "{}\n\
             Nothing was changed. `immortality status` lists what this daemon knows about.",
            error.message
        ),
        ErrorCode::Unauthorized => format!(
            "{}\n\
             The socket is mode 0600 and the token lives in daemon.token: run the client as the\n\
             user who owns the daemon.",
            error.message
        ),
        ErrorCode::VersionMismatch | ErrorCode::UnknownMethod => format!(
            "{}\n\
             The daemon and this client are different builds; restart the daemon with this binary.",
            error.message
        ),
        ErrorCode::BadParams | ErrorCode::Internal => error.message.clone(),
    }
}

/// The default socket and token paths, resolved through `StatePaths`.
#[must_use]
pub fn default_endpoints() -> (PathBuf, PathBuf) {
    let state = immortality_core::paths::StatePaths::discover();
    (state.daemon_socket(), state.daemon_token())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_token_is_read_without_its_trailing_newline() {
        // The daemon writes the token with a newline, so a client that kept it would never
        // match and every command would report a wrong token.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("daemon.token");
        std::fs::write(&path, "secret\n").unwrap();
        assert_eq!(read_token(&path).unwrap(), "secret");
    }

    #[test]
    fn a_token_is_trimmed_of_surrounding_whitespace() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("daemon.token");
        std::fs::write(&path, "  secret  \r\n").unwrap();
        assert_eq!(read_token(&path).unwrap(), "secret");
    }

    #[test]
    fn a_missing_token_names_the_file_and_says_no_daemon_was_started() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("daemon.token");
        let error = read_token(&path).unwrap_err().to_string();
        assert!(error.contains("daemon.token"), "{error}");
        assert!(error.contains("no daemon was ever started"), "{error}");
    }

    #[test]
    fn an_empty_token_is_refused_because_it_would_authorise_everything() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("daemon.token");
        std::fs::write(&path, "  \n").unwrap();
        let error = read_token(&path).unwrap_err().to_string();
        assert!(error.contains("empty"), "{error}");
    }

    #[test]
    fn not_found_and_unauthorised_errors_carry_a_next_step() {
        // These are the two a human can fix without reading source, so the message has to say
        // what to do rather than just what went wrong.
        let not_found = explain(&rpc::RpcError::new(ErrorCode::NotFound, "no task tsk_01"));
        assert!(not_found.contains("no task tsk_01"), "{not_found}");
        assert!(not_found.contains("immortality status"), "{not_found}");
        assert!(not_found.contains("Nothing was changed"), "{not_found}");

        let unauthorised = explain(&rpc::RpcError::new(
            ErrorCode::Unauthorized,
            "the token does not match",
        ));
        assert!(
            unauthorised.contains("the token does not match"),
            "{unauthorised}"
        );
        assert!(unauthorised.contains("0600"), "{unauthorised}");
    }

    #[test]
    fn a_version_or_method_error_points_at_the_builds() {
        for code in [ErrorCode::VersionMismatch, ErrorCode::UnknownMethod] {
            let message = explain(&rpc::RpcError::new(code, "nope"));
            assert!(message.contains("different builds"), "{code:?}: {message}");
            assert!(message.contains("restart the daemon"), "{code:?}: {message}");
        }
    }

    #[test]
    fn bad_params_and_internal_errors_are_passed_through_unchanged() {
        // Nothing actionable to add: the daemon's message already names the field or the bug.
        for code in [ErrorCode::BadParams, ErrorCode::Internal] {
            let message = explain(&rpc::RpcError::new(code, "field `task` is required"));
            assert_eq!(message, "field `task` is required");
        }
    }

    #[test]
    fn the_default_endpoints_are_beside_each_other_under_one_state_root() {
        // The client finds the token by the socket's side, not by a second rule that could
        // drift; the daemon's startup check depends on the same layout.
        let (socket, token) = default_endpoints();
        assert_eq!(socket.parent(), token.parent());
        assert_eq!(socket.file_name().unwrap(), "daemon.sock");
        assert_eq!(token.file_name().unwrap(), "daemon.token");
    }

    /// A fake daemon that answers `hello`, then serves one scripted event line.
    ///
    /// Hand-written rather than the real server: the point is to control *when* bytes arrive, and
    /// the real server cannot be told to stay silent for two seconds.
    fn fake_daemon(dir: &Path, event_delay: Duration) -> std::path::PathBuf {
        let socket = dir.join("daemon.sock");
        let path = socket.clone();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            runtime.block_on(async move {
                let listener = tokio::net::UnixListener::bind(&path).expect("bind");
                let (stream, _) = listener.accept().await.expect("accept");
                let (read, mut write) = stream.into_split();
                let mut lines = tokio::io::BufReader::new(read).lines();
                // `hello`
                if lines.next_line().await.ok().flatten().is_none() {
                    return;
                }
                write
                    .write_all(b"{\"id\":1,\"result\":{\"version\":1}}\n")
                    .await
                    .ok();
                // `attach`
                if lines.next_line().await.ok().flatten().is_none() {
                    return;
                }
                write
                    .write_all(b"{\"id\":2,\"result\":{\"backlog\":[],\"next_seq\":0}}\n")
                    .await
                    .ok();
                // Silence for longer than any request deadline, then one event.
                tokio::time::sleep(event_delay).await;
                write
                    .write_all(b"{\"record\":{\"seq\":0,\"kind\":\"task_state\",\"task\":\"t\"}}\n")
                    .await
                    .ok();
                tokio::time::sleep(Duration::from_secs(5)).await;
            });
        });
        // The listener binds inside the thread; wait until the path exists.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !socket.exists() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        socket
    }

    /// A fake daemon that answers `hello` and `attach`, then closes the stream immediately.
    ///
    /// This is the lagged-subscriber case from the server's side: the client must reconnect and
    /// say so, rather than either hanging silently or giving up.
    fn closing_daemon(dir: &Path) -> std::path::PathBuf {
        let socket = dir.join("daemon.sock");
        let path = socket.clone();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            runtime.block_on(async move {
                let listener = tokio::net::UnixListener::bind(&path).expect("bind");
                loop {
                    let Ok((stream, _)) = listener.accept().await else {
                        return;
                    };
                    let (read, mut write) = stream.into_split();
                    let mut lines = tokio::io::BufReader::new(read).lines();
                    if lines.next_line().await.ok().flatten().is_none() {
                        continue;
                    }
                    let _ = write.write_all(b"{\"id\":1,\"result\":{\"version\":1}}\n").await;
                    if lines.next_line().await.ok().flatten().is_none() {
                        continue;
                    }
                    let _ = write
                        .write_all(b"{\"id\":2,\"result\":{\"backlog\":[],\"next_seq\":0}}\n")
                        .await;
                    // Close at once: every connection is fruitless.
                }
            });
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !socket.exists() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        socket
    }

    #[tokio::test]
    async fn a_daemon_closing_every_stream_is_reported_and_bounded() {
        // Two failures in one: the human must be told the stream closed (a frozen screen looks
        // like a thinking model), and the loop must stop instead of reconnecting forever.
        let dir = tempfile::tempdir().expect("tempdir");
        let socket = closing_daemon(dir.path());
        let token_path = dir.path().join("daemon.token");
        std::fs::write(&token_path, "t\n").expect("token");

        let mut client = Client::connect(&socket, &token_path).await.expect("connect");
        let mut notices = Vec::new();
        let error = client
            .attach(
                "t",
                0,
                |_| {},
                |notice| {
                    notices.push(notice);
                },
            )
            .await
            .expect_err("a daemon that never streams must not be retried forever");

        assert!(
            error.to_string().contains("without"),
            "the message must say what happened: {error}"
        );
        // Exactly the tolerated reconnections, numbered from one. An off-by-one here would
        // tell the human "attempt 2" on the first retry, which reads as a lost attempt.
        let attempts: Vec<u32> = notices
            .iter()
            .map(|notice| match notice {
                Notice::Reconnecting { attempt, .. } => *attempt,
            })
            .collect();
        assert_eq!(attempts, vec![1, 2, 3], "{notices:?}");
        assert_eq!(
            attempts.len(),
            usize::try_from(MAX_FRUITLESS_RECONNECTS).unwrap(),
            "the tolerated count is the announced count"
        );
    }

    #[tokio::test]
    async fn a_silent_stream_is_not_a_timed_out_request() {
        // A real bug: the stream read through `read_line`, which applies the 30-second answer
        // deadline. A run where the model thought for longer than that was disconnected as if
        // the daemon had hung, and a cold prefill here is 166 s — so it happened every time.
        let dir = tempfile::tempdir().expect("tempdir");
        let socket = fake_daemon(dir.path(), Duration::from_millis(300));
        let token_path = dir.path().join("daemon.token");
        std::fs::write(&token_path, "t\n").expect("token");

        let mut client = Client::connect(&socket, &token_path).await.expect("connect");
        // Shorter than the silence above: the request path would fail here, the stream must not.
        client.answer_timeout = Duration::from_millis(50);

        let mut seen = Vec::new();
        let outcome = tokio::time::timeout(
            Duration::from_secs(3),
            client.attach(
                "t",
                0,
                |record| {
                    seen.push(record.get("seq").and_then(Value::as_u64));
                },
                |_| {},
            ),
        )
        .await;
        // `attach` loops forever by design; the event arriving before the timeout is the point.
        assert!(outcome.is_err(), "attach should still be streaming, not returned");
        assert_eq!(seen, vec![Some(0)], "the silent stream must still deliver");
    }

    /// A daemon that answers `hello`, then answers the next request after `delay`.
    ///
    /// The listener is bound **before** the thread starts. Binding inside it raced the client:
    /// under a full parallel test run `Client::connect` reached a socket that did not exist yet and
    /// failed with "the daemon closed the connection" — seen once in three whole-binary runs and
    /// never in isolation, which is the shape of a race and not of a broken test.
    fn slow_request_daemon(dir: &Path, delay: Duration) -> std::path::PathBuf {
        let socket = dir.join("daemon.sock");
        let listener = std::os::unix::net::UnixListener::bind(&socket).expect("bind");
        listener
            .set_nonblocking(true)
            .expect("the listener is handed to tokio");
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            runtime.block_on(async move {
                let listener = tokio::net::UnixListener::from_std(listener).expect("a non-blocking listener");
                let (stream, _) = listener.accept().await.expect("accept");
                let (read, mut write) = stream.into_split();
                let mut lines = tokio::io::BufReader::new(read).lines();
                // `hello`
                if lines.next_line().await.ok().flatten().is_none() {
                    return;
                }
                write
                    .write_all(b"{\"id\":1,\"result\":{\"version\":1}}\n")
                    .await
                    .ok();
                // The one request, answered only after the planner-shaped wait.
                if lines.next_line().await.ok().flatten().is_none() {
                    return;
                }
                tokio::time::sleep(delay).await;
                write
                    .write_all(b"{\"id\":2,\"result\":{\"created\":true}}\n")
                    .await
                    .ok();
            });
        });
        socket
    }

    #[tokio::test]
    async fn a_planning_request_outlives_the_status_deadline() {
        // The bug, measured three times in a row on 2026-11-16: `plan --goal` and `start "<goal>"`
        // call the backend inside the request, and the planner's answer took 22.9 s once and
        // 29.6 s another time. The 30-second deadline sized for `status` therefore fired just
        // before success, and the plan was written to disk anyway — the human saw "the daemon did
        // not answer within 30 seconds" over a plan that existed.
        let dir = tempfile::tempdir().expect("tempdir");
        let socket = slow_request_daemon(dir.path(), Duration::from_millis(400));
        let token_path = dir.path().join("daemon.token");
        std::fs::write(&token_path, "t\n").expect("token");

        let mut client = Client::connect(&socket, &token_path).await.expect("connect");
        // Shorter than the daemon's wait: only the planning rule can survive it.
        client.answer_timeout = Duration::from_millis(100);
        let response = client
            .request("plan", json!({"goal": "x"}))
            .await
            .expect("a planning request waits for the planner");
        assert_eq!(
            response.result.expect("result")["created"],
            json!(true),
            "the answer the daemon owed still arrives"
        );
    }

    #[test]
    fn the_deadline_follows_the_shape_of_the_request() {
        // The rule `a_planning_request_outlives_the_status_deadline` depends on, stated directly:
        // the two shape-sensitive methods get the long deadline only when they carry a goal, and
        // everything else gets the short one. Without this, a call site could keep the 30-second
        // wait in place and the socket test above would still pass on a fast backend.
        let short = Duration::from_secs(30);
        for (method, params, long) in [
            ("plan", json!({"goal": "x"}), true),
            ("start", json!({"goal": "x"}), true),
            // The fast shapes of the same methods: a plan that already exists, a task being resumed.
            ("plan", json!({"project": "/tmp/p"}), false),
            ("start", json!({"task": "tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV"}), false),
            // Everything else answers from disk.
            ("status", json!({}), false),
            ("log", json!({"task": "t"}), false),
            ("metrics", json!({}), false),
            ("gc", json!({"dry_run": true}), false),
            ("attach", json!({"task": "t"}), false),
        ] {
            let got = deadline_for(method, &params, short);
            assert_eq!(
                got,
                if long { THINKING_TIMEOUT } else { short },
                "`{method}` with {params} got {got:?}"
            );
        }
    }
}
