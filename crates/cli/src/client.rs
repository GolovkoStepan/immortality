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

/// How many reconnects that deliver nothing in a row are tolerated before giving up.
///
/// A daemon that closes every connection immediately is broken. One reconnect is the normal
/// lagged-subscriber case; more than a few with no record between them is a loop, and a silent
/// loop is worse than an error message.
const MAX_FRUITLESS_RECONNECTS: u32 = 3;

/// A connection to a running daemon.
pub struct Client {
    write: tokio::net::unix::OwnedWriteHalf,
    reader: BufReader<tokio::net::unix::OwnedReadHalf>,
    next_id: u64,
    token: String,
    /// Remembered so a stream that the daemon closes can be re-opened. The file descriptor
    /// cannot be read back into a path, so the path is kept rather than derived.
    socket: PathBuf,
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
        // `hello` is the handshake *and* the token check: a wrong token is refused here, before
        // any command runs, rather than after it looked like it was going to work.
        let response = client.request("hello", json!({})).await?;
        if let Some(error) = response.error {
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
        })
    }

    /// Send one request and read exactly one response.
    ///
    /// # Errors
    /// The socket died mid-request, or the daemon sent a message that is not a response.
    pub async fn request(&mut self, method: &str, params: Value) -> Result<Response> {
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
        let line = self.read_line().await?;
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
            let line = match self.read_line().await {
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
        let response = client.request("hello", json!({})).await?;
        if let Some(error) = response.error {
            bail!("the daemon refused the reconnect: {}", error.message);
        }
        Ok(client)
    }

    /// Read one line, with a deadline.
    async fn read_line(&mut self) -> Result<String> {
        let mut line = String::new();
        let read = tokio::time::timeout(ANSWER_TIMEOUT, self.reader.read_line(&mut line))
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "the daemon did not answer within {} seconds",
                    ANSWER_TIMEOUT.as_secs()
                )
            })??;
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
}
