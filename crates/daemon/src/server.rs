//! The unix-socket server: token check, method dispatch, and the event stream.
//!
//! # Why the loop is per connection and not a select over all clients
//!
//! A `status` request is answered in microseconds and an `attach` connection then goes quiet
//! until the next journal record. Blocking a thread per connection would mean a thread per
//! attached terminal, held for hours — the usual shape of a daemon that grows a thread leak
//! nobody notices. Each connection is a small task instead, and the shared state behind it is
//! `Arc<Mutex<…>>` only where a request mutates something.
//!
//! # Why attach replays from `since_seq`
//!
//! A client killed and restarted — the roadmap's own test — must not lose events, and must not
//! re-render ones it already saw. The journal is append-only and ordered per task, so the
//! client's last seen `seq` is all the state needed: `since_seq` is an inclusive lower bound and
//! the daemon starts there. A client that was away for ten thousand records catches up in one
//! read rather than by polling.
//!
//! # Why the fan-out is a broadcast and not a queue per client
//!
//! A slow or dead client must not slow the daemon, and there is no reason for one: the
//! authoritative record is the journal, and a client that misses broadcast messages can ask for
//! `since_seq` again. A bounded channel that drops for the laggard and a replay on reconnect is
//! the whole design; an unbounded queue per client is how a daemon dies of a terminal left open
//! on a laptop that went to sleep.

use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use immortality_core::delegation::DelegationStore;
use immortality_core::escalation::{Answer, EscalationStore};
use immortality_core::ids::{EscalationId, TaskId};
use immortality_core::journal::{JournalReader, JournalRecord};
use immortality_core::registry::Registry;
use serde_json::{Value, json};
use tokio::io::{AsyncWriteExt as _, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::broadcast;

use crate::rpc::{self, ErrorCode, Event, Request, Response, RpcError};

/// How many journal records a laggard client may fall behind before it is told to re-attach.
const EVENT_BUFFER: usize = 1024;

/// Everything the handlers need, in one place so a handler cannot reach for a global.
#[derive(Debug, Clone)]
pub struct ServerState {
    /// State root: the registry, task directories and `daemon.token`.
    pub state_root: PathBuf,
    /// The token a client must present.
    pub token: String,
    /// How deep a subtask tree may grow, from `LoopLimits::max_subtask_depth`.
    ///
    /// Carried here because `status` has to *read* plans — `Plan::load` validates depth, so a
    /// reader without the limit cannot load a tree at all. Defaulting it per call would make the
    /// limit a property of the reader rather than of the daemon, and a plan that one call accepts
    /// and another refuses is the kind of disagreement that shows up as "it ran yesterday".
    pub max_subtask_depth: u8,
}

impl ServerState {
    /// State reading its token from `state_root`.
    ///
    /// # Errors
    /// `daemon.token` is missing or unreadable. A daemon without its token would have to accept
    /// every client, and "no token configured" is not a way to run an unauthenticated daemon by
    /// accident.
    pub fn discover(state_root: impl Into<PathBuf>) -> immortality_core::Result<Self> {
        let state_root = state_root.into();
        let token_path = state_root.join("daemon.token");
        let token = std::fs::read_to_string(&token_path)
            .map_err(|error| immortality_core::Error::io(&token_path, error))?;
        let token = token.trim().to_string();
        if token.is_empty() {
            return Err(immortality_core::Error::Protocol(format!(
                "{} is empty; a daemon with no token accepts every client",
                token_path.display()
            )));
        }
        Ok(Self {
            state_root,
            token,
            max_subtask_depth: immortality_core::config::Config::default()
                .loop_
                .max_subtask_depth,
        })
    }

    /// The same state with `config`'s plan limits.
    ///
    /// The token and state root are the daemon's; only the limits come from configuration. A caller
    /// that has just loaded the config must use this, or `status` will read plans against built-in
    /// limits and refuse a tree the scheduler would happily run.
    #[must_use]
    pub const fn with_config(mut self, config: &immortality_core::config::Config) -> Self {
        self.max_subtask_depth = config.loop_.max_subtask_depth;
        self
    }

    /// The registry file.
    #[must_use]
    pub fn registry_path(&self) -> PathBuf {
        self.state_root.join(immortality_core::registry::FILE_NAME)
    }

    /// A task's state directory, found through its project.
    ///
    /// The directory is *named* here, not required to exist: scheduling a task is what creates
    /// it, and a `status` or `escalations` call made before that must still find the task.
    ///
    /// # Errors
    /// No project claims the task. Guessing the project from the current directory would make
    /// the answer depend on where the human happened to run the CLI.
    pub fn task_dir(&self, task: &TaskId) -> immortality_core::Result<PathBuf> {
        let registry = Registry::open(self.registry_path())?;
        registry.owner_of(&self.state_root, task)?.map_or_else(
            || {
                Err(immortality_core::Error::Protocol(format!(
                    "no project in {} owns task {task}; run `immortality status` to see what exists",
                    self.registry_path().display()
                )))
            },
            |project| Ok(project.state_dir(&self.state_root).join(task.as_str())),
        )
    }

    /// A task's journal directory: `.agent/` in its project.
    ///
    /// # Errors
    /// No project claims the task.
    pub fn journal_dir(&self, task: &TaskId) -> immortality_core::Result<PathBuf> {
        let registry = Registry::open(self.registry_path())?;
        registry.owner_of(&self.state_root, task)?.map_or_else(
            || {
                Err(immortality_core::Error::Protocol(format!(
                    "no project in {} owns task {task}",
                    self.registry_path().display()
                )))
            },
            |project| Ok(project.agent_dir()),
        )
    }
}

/// A running server, with the handle the daemon keeps to broadcast events.
#[derive(Debug, Clone)]
pub struct Events {
    sender: broadcast::Sender<Arc<JournalRecord>>,
}

impl Events {
    /// A fresh broadcast channel.
    #[must_use]
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(EVENT_BUFFER);
        Self { sender }
    }

    /// Publish a journal record to every attached client.
    ///
    /// A send with no subscribers is not an error: the daemon runs whether or not a terminal is
    /// attached, which is the entire point of it being a daemon.
    pub fn publish(&self, record: Arc<JournalRecord>) {
        // Ignoring the error is deliberate: `Err` here means nobody is listening.
        let _ = self.sender.send(record);
    }

    /// A subscriber for one connection.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<JournalRecord>> {
        self.sender.subscribe()
    }
}

impl Default for Events {
    fn default() -> Self {
        Self::new()
    }
}

/// Bind the socket, having removed a stale one.
///
/// # Errors
/// The parent directory cannot be created, an existing socket cannot be removed, or the bind
/// fails. The permissions are set to `0600` immediately after binding: between `bind` and
/// `chmod` the socket has the umask's permissions, and a world-writable socket for a few
/// microseconds is still a window worth closing before anything can connect.
pub fn bind(path: &Path) -> immortality_core::Result<UnixListener> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| immortality_core::Error::io(parent, error))?;
    }
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(immortality_core::Error::io(path, error)),
    }
    let listener = UnixListener::bind(path).map_err(|error| immortality_core::Error::io(path, error))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .map_err(|error| immortality_core::Error::io(path, error))?;
    Ok(listener)
}

/// Serve until the listener errors or the future is dropped.
///
/// # Errors
/// `listener.accept` failed for a reason other than a per-connection error, which ends the
/// server.
pub async fn serve(listener: UnixListener, state: ServerState, events: Events) -> std::io::Result<()> {
    let state = Arc::new(state);
    let events = Arc::new(events);
    loop {
        let (stream, _) = listener.accept().await?;
        let state = Arc::clone(&state);
        let events = Arc::clone(&events);
        // A connection is its own task: one attached terminal stuck on a slow read must not
        // stop another from being answered.
        tokio::spawn(async move {
            if let Err(error) = connection(stream, &state, &events).await {
                tracing::debug!("connection ended: {error}");
            }
        });
    }
}

/// One connection: read lines, answer requests, stream events.
async fn connection(stream: UnixStream, state: &ServerState, events: &Events) -> std::io::Result<()> {
    let (read, mut write) = stream.into_split();
    let mut reader = BufReader::new(read);
    let mut line = String::new();
    // The token the client proved, once it has. `None` until `hello` succeeds.
    let mut authorised = false;

    loop {
        line.clear();
        // Bounded while reading, not only after: `read_line` would grow the buffer to whatever
        // the client sends, so a limit checked afterwards is not a limit at all.
        let read = match read_bounded_line(&mut reader, &mut line, rpc::MAX_LINE_BYTES).await? {
            ReadLine::Eof => return Ok(()),
            ReadLine::TooLong => {
                let response = Response::failed(
                    0,
                    RpcError::new(ErrorCode::BadParams, "the request line is too long"),
                );
                write_line(&mut write, &response).await?;
                return Ok(());
            }
            ReadLine::Line(count) => count,
        };
        if read == 0 {
            return Ok(());
        }
        let request: Request = match rpc::decode(line.trim_end()) {
            Ok(request) => request,
            Err(error) => {
                // No id to answer with: the request did not parse. Say so and stay connected,
                // because a client that sent one bad line may still send a good one.
                let response = Response::failed(0, RpcError::new(ErrorCode::BadParams, error.to_string()));
                write_line(&mut write, &response).await?;
                continue;
            }
        };

        // `hello` is the one method answerable before the token is proven; it *is* the proof.
        let Some(token) = request.token.as_deref() else {
            let response = Response::failed(
                request.id,
                RpcError::new(ErrorCode::Unauthorized, "no token; send `hello` first"),
            );
            write_line(&mut write, &response).await?;
            continue;
        };
        if token != state.token {
            // Closed, not just refused: a client with the wrong token is one that should not
            // keep asking, and leaving it connected only gives it more chances to guess.
            let response = Response::failed(
                request.id,
                RpcError::new(ErrorCode::Unauthorized, "the token does not match daemon.token"),
            );
            write_line(&mut write, &response).await?;
            return Ok(());
        }

        if request.method == "hello" {
            let response = hello(request.id, &request.params);
            // Only a successful handshake authorises the rest of the connection. A version
            // refusal leaves it unauthorised: the two builds have just agreed they disagree, and
            // marking the client authorised would let the next request be interpreted by exactly
            // the daemon it was told not to talk to.
            authorised = response.error.is_none();
            write_line(&mut write, &response).await?;
            continue;
        }
        if !authorised {
            let response = Response::failed(
                request.id,
                RpcError::new(ErrorCode::Unauthorized, "send `hello` before any other method"),
            );
            write_line(&mut write, &response).await?;
            continue;
        }

        // `attach` is the one method that subscribes, and it must subscribe *before* reading
        // the backlog: a record written between the two would otherwise be in neither the
        // backlog nor the stream, which is the one way an attaching client can miss a step.
        // Subscribing first risks a duplicate instead, and a duplicate is removed by the
        // `seq` watermark below — a lost record cannot be removed by anything.
        let subscriber = if request.method == "attach" {
            Some(events.subscribe())
        } else {
            None
        };

        let response = match dispatch(&request, state) {
            Ok(result) => Response::ok(request.id, result),
            Err(error) => Response::failed(request.id, error),
        };
        write_line(&mut write, &response).await?;

        if let Some(mut subscriber) = subscriber {
            if response.error.is_some() {
                continue;
            }
            // Everything up to `next_seq` is already in the acknowledgement, so the stream
            // starts after it. Without this watermark the client renders the backlog twice.
            let after = response
                .result
                .as_ref()
                .and_then(|result| result.get("next_seq"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let task = request
                .params
                .get("task")
                .and_then(Value::as_str)
                .and_then(|task| TaskId::parse(task).ok());
            // From here the connection is a stream until the client leaves or falls behind.
            return stream_events(&mut write, &mut subscriber, task.as_ref(), after).await;
        }
    }
}

/// What a bounded read produced.
enum ReadLine {
    /// The peer closed the connection.
    Eof,
    /// A complete line, `count` bytes including the newline.
    Line(usize),
    /// The line exceeded the limit; the rest of it is discarded.
    TooLong,
}

/// Read one newline-terminated line, refusing to buffer more than `limit` bytes.
///
/// `AsyncBufReadExt::read_line` has no limit and grows its `String` to whatever arrives, so a
/// single client can make the daemon allocate until it dies. This reads through the buffer and
/// stops at the limit, dropping the rest of the oversized line so the next read starts at a
/// message boundary rather than in the middle of the one that was refused.
async fn read_bounded_line<R: tokio::io::AsyncBufRead + Unpin>(
    reader: &mut R,
    out: &mut String,
    limit: usize,
) -> std::io::Result<ReadLine> {
    use tokio::io::AsyncBufReadExt as _;

    let mut total = 0usize;
    let mut too_long = false;
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Ok(if total == 0 {
                ReadLine::Eof
            } else if too_long {
                ReadLine::TooLong
            } else {
                ReadLine::Line(total)
            });
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let done = newline.is_some();
        let slice = newline.map_or(available, |index| &available[..=index]);
        // Copy only what still fits. The count keeps advancing so `TooLong` is decided by the
        // real line length rather than by how much happened to be kept.
        if !too_long {
            if total + slice.len() > limit {
                too_long = true;
            } else {
                // The buffer holds bytes and the slice is a valid prefix of a UTF-8 stream that
                // came from a socket; `from_utf8` is checked below rather than assumed.
                match std::str::from_utf8(slice) {
                    Ok(text) => out.push_str(text),
                    Err(_) => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "the request line is not valid UTF-8",
                        ));
                    }
                }
            }
        }
        let consumed = slice.len();
        total += consumed;
        reader.consume(consumed);
        if done {
            return Ok(if too_long {
                ReadLine::TooLong
            } else {
                ReadLine::Line(total)
            });
        }
    }
}

/// Push events until the client goes away.
///
/// `after` is the first sequence number the client has not seen, and `task` filters the shared
/// journal down to the one task the client asked about. Both are needed even though the backlog
/// was already filtered: the broadcast carries every task of every project on this daemon, so a
/// stream without these two guards would hand a client another task's work.
async fn stream_events(
    write: &mut tokio::net::unix::OwnedWriteHalf,
    subscriber: &mut broadcast::Receiver<Arc<JournalRecord>>,
    task: Option<&TaskId>,
    after: u64,
) -> std::io::Result<()> {
    loop {
        match subscriber.recv().await {
            Ok(record) => {
                if record.fields.seq < after {
                    continue;
                }
                if task.is_some_and(|task| &record.fields.task != task) {
                    continue;
                }
                let event = Event {
                    record: serde_json::to_value(&*record).map_err(|error| encode_error(&error))?,
                };
                write_line(write, &event).await?;
            }
            // The client fell behind and the buffer dropped records. The journal is the source
            // of truth, so the honest answer is to stop and let it re-attach with `since_seq`
            // rather than to serve it a stream with a hole in it.
            Err(broadcast::error::RecvError::Lagged(skipped)) => {
                tracing::debug!("attached client lagged by {skipped} records; closing");
                return Ok(());
            }
            Err(broadcast::error::RecvError::Closed) => return Ok(()),
        }
    }
}

/// A serialization failure, as an I/O error the connection loop already handles.
fn encode_error(error: &serde_json::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
}

/// Answer one method.
///
/// Every handler at this milestone answers from local files, so this is synchronous. M4's
/// scheduler handlers will need a runtime, and this becomes `async` again then; declaring it
/// `async` now would only imply a suspension point that does not exist.
/// `hello`: prove the token was accepted, and negotiate the protocol version.
///
/// Version negotiation lives here rather than in the client because a client comparing versions
/// *after* a successful handshake has already been handed code that assumed the answer. The
/// daemon can tell the difference between "your token is wrong" and "your build is different",
/// and a client that only checked on its own could not.
///
/// A mismatched version is a refusal, not a disconnect: the human running an old terminal against
/// a new daemon needs the message, and a client that omitted `version` is still answerable.
fn hello(id: u64, params: &Value) -> Response {
    // A malformed `params` is treated as "no version claimed": the token was already proven, and
    // an unparseable extra field is not a reason to refuse a client that speaks this build.
    let theirs = serde_json::from_value::<rpc::HelloParams>(params.clone())
        .ok()
        .and_then(|claimed| claimed.version);
    if let Some(theirs) = theirs
        && theirs != rpc::VERSION
    {
        return Response::failed(
            id,
            RpcError::new(
                ErrorCode::VersionMismatch,
                format!(
                    "this daemon speaks protocol {} and this client speaks {theirs}",
                    rpc::VERSION
                ),
            ),
        );
    }
    Response::ok(
        id,
        json!({
            "version": rpc::VERSION,
            "protocol": "immortality",
        }),
    )
}

fn dispatch(request: &Request, state: &ServerState) -> Result<Value, RpcError> {
    match request.method.as_str() {
        "status" => status(state, &request.params),
        "escalations" => escalations(state, &request.params),
        "approve" => approve(state, &request.params),
        "attach" => attach(state, &request.params),
        "hello" => Ok(json!({"version": rpc::VERSION})),
        other => Err(RpcError::new(
            ErrorCode::UnknownMethod,
            format!(
                "`{other}` is not a method in protocol version {}; known methods: {}",
                rpc::VERSION,
                rpc::METHODS.join(", ")
            ),
        )),
    }
}

/// `status`: every task the daemon knows about, or one.
fn status(state: &ServerState, params: &Value) -> Result<Value, RpcError> {
    let registry = Registry::open(state.registry_path()).map_err(|error| internal(&error))?;
    let wanted = params
        .get("task")
        .and_then(Value::as_str)
        .map(TaskId::parse)
        .transpose()
        .map_err(|error| RpcError::new(ErrorCode::BadParams, error.to_string()))?;

    let mut projects = Vec::new();
    for project in registry.projects() {
        projects.push(json!({
            "id": project.id,
            "path": project.path,
            "remote": project.remote,
        }));
    }

    let mut tasks = Vec::new();
    let known = registry
        .tasks(&state.state_root)
        .map_err(|error| internal(&error))?;
    for (project, task, dir) in known {
        if wanted.as_ref().is_some_and(|wanted| wanted != &task) {
            continue;
        }
        let escalations =
            EscalationStore::open(&dir, task.clone()).map_or(0, |store| store.open_entries().count());
        let delegations = DelegationStore::new(dir.join("delegations"))
            .list()
            .map_or(0, |reports| reports.len());
        // One `JOURNAL.jsonl` per repository holds every task of that repository, and `seq`
        // counts lines in the file rather than records of one task. A handler that forgot to
        // filter would report another task's last record as this task's, and attach would
        // stream another task's work into the terminal.
        tasks.push(json!({
            "task": task,
            "project": project.id,
            "state_dir": dir,
            "open_escalations": escalations,
            "delegations": delegations,
            "journal": journal_summary(&project, &task),
        }));
    }

    if wanted.is_some() && tasks.is_empty() {
        let task = wanted.map(|task| task.to_string()).unwrap_or_default();
        return Err(RpcError::new(
            ErrorCode::NotFound,
            format!("no task `{task}` in this daemon's registry"),
        ));
    }

    // Why nothing is running, when nothing is. An idle daemon with an empty table is the failure
    // this project exists to avoid, so the answer carries the reason and `status` renders it: the
    // four reasons have four different fixes, and "no tasks" is not one of them.
    let scheduling = scheduling_reason(state);

    Ok(json!({
        "projects": projects,
        "tasks": tasks,
        "scheduling": scheduling,
        "version": rpc::VERSION,
    }))
}

/// What the scheduler would do right now, and why it would do nothing.
///
/// Read through the same [`crate::scheduling::candidates`] the scheduler uses, so `status` cannot
/// report a task as runnable when the queue has already rejected it for a reason it does not know.
/// A read failure is reported as text rather than omitted: a human whose daemon is idle needs to
/// know that the answer could not be computed, which is different from "nothing to run".
fn scheduling_reason(state: &ServerState) -> Value {
    let mut config = immortality_core::config::Config::default();
    config.loop_.max_subtask_depth = state.max_subtask_depth;
    let registry = match Registry::open(state.registry_path()) {
        Ok(registry) => registry,
        Err(error) => return json!({"error": error.to_string()}),
    };
    let candidates = match crate::scheduling::candidates(&registry, &state.state_root, &config) {
        Ok(candidates) => candidates,
        Err(error) => return json!({"error": error.to_string()}),
    };
    let queue = immortality_core::scheduler::queue::Queue::new();
    let all: Vec<immortality_core::scheduler::queue::Candidate> = candidates
        .into_iter()
        .map(|scheduled| scheduled.candidate)
        .collect();
    let (chosen, rejected) = queue.next_for_inference(&all);
    json!({
        "next": chosen.map(|candidate| candidate.task.clone()),
        "reason": if chosen.is_some() {
            String::new()
        } else {
            immortality_core::scheduler::queue::summarise(&rejected)
        },
    })
}

/// A one-line summary of one task's share of its repository's journal.
///
/// One `JOURNAL.jsonl` per repository holds every task of that repository, and `seq` counts
/// lines in the file rather than records of one task. A report that forgot to filter would
/// show another task's last record as this task's, so the filter lives here and every caller
/// gets it.
///
/// A journal that cannot be read is reported as `{"error": ...}` rather than as zero records:
/// "unreadable" and "empty" lead a human to different actions, and silently reporting empty
/// would hide a corrupt journal behind a plausible-looking zero.
fn journal_summary(project: &immortality_core::registry::Project, task: &TaskId) -> Value {
    let outcome = match JournalReader::open(project.agent_dir(), task).and_then(|reader| reader.read()) {
        Ok(outcome) => outcome,
        Err(error) => return json!({"error": error.to_string()}),
    };
    let owned: Vec<&JournalRecord> = outcome
        .records
        .iter()
        .filter(|record| &record.fields.task == task)
        .collect();
    let last = owned.last().map(|record| {
        json!({
            "seq": record.fields.seq,
            "kind": record.fields.kind.as_str(),
            "at": record.fields.at,
        })
    });
    json!({
        "records": owned.len(),
        "next_seq": owned.last().map_or(0, |record| record.fields.seq + 1),
        "last": last,
        "corrupt": outcome.is_corrupt(),
    })
}

/// `escalations`: the open questions, optionally for one task.
fn escalations(state: &ServerState, params: &Value) -> Result<Value, RpcError> {
    let registry = Registry::open(state.registry_path()).map_err(|error| internal(&error))?;
    let wanted = params
        .get("task")
        .and_then(Value::as_str)
        .map(TaskId::parse)
        .transpose()
        .map_err(|error| RpcError::new(ErrorCode::BadParams, error.to_string()))?;

    let mut entries = Vec::new();
    let known = registry
        .tasks(&state.state_root)
        .map_err(|error| internal(&error))?;
    for (_, task, dir) in known {
        if wanted.as_ref().is_some_and(|wanted| wanted != &task) {
            continue;
        }
        let store = match EscalationStore::open(&dir, task.clone()) {
            Ok(store) => store,
            Err(error) => return Err(internal(&error)),
        };
        for entry in store.open_entries() {
            entries.push(json!({
                "id": entry.id,
                "task": entry.task,
                "key": entry.key,
                "kind": entry.kind,
                "question": entry.question,
                "options": entry.options,
                "blocking": entry.blocking,
                "created_at": entry.created_at,
            }));
        }
    }
    Ok(json!({"escalations": entries}))
}

/// `approve`: answer one escalation.
fn approve(state: &ServerState, params: &Value) -> Result<Value, RpcError> {
    let id = params
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::new(ErrorCode::BadParams, "`id` is required"))?;
    let id =
        EscalationId::parse(id).map_err(|error| RpcError::new(ErrorCode::BadParams, error.to_string()))?;
    let answer_name = params
        .get("answer")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::new(ErrorCode::BadParams, "`answer` is required"))?;
    let answer =
        Answer::parse(answer_name).map_err(|error| RpcError::new(ErrorCode::BadParams, error.to_string()))?;

    let registry = Registry::open(state.registry_path()).map_err(|error| internal(&error))?;
    let known = registry
        .tasks(&state.state_root)
        .map_err(|error| internal(&error))?;
    for (_, task, dir) in known {
        let Ok(mut store) = EscalationStore::open(&dir, task.clone()) else {
            continue;
        };
        if store.get(&id).is_none() {
            continue;
        }
        let entry = store
            .answer(&id, answer, immortality_core::time::now())
            .map_err(|error| internal(&error))?;
        return Ok(json!({
            "id": entry.id,
            "task": entry.task,
            "answer": entry.answer,
            "resolved_at": entry.resolved_at,
        }));
    }
    Err(RpcError::new(
        ErrorCode::NotFound,
        format!("no escalation `{id}` in any task of this daemon"),
    ))
}

/// `attach`: acknowledge, then stream from `since_seq`.
///
/// Synchronous on purpose: it reads the journal and returns the backlog; the streaming happens
/// in `stream_events` on the connection. An `async` here would suggest an `await` that is not
/// there.
fn attach(state: &ServerState, params: &Value) -> Result<Value, RpcError> {
    let task = params
        .get("task")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::new(ErrorCode::BadParams, "`task` is required"))?;
    let task = TaskId::parse(task).map_err(|error| RpcError::new(ErrorCode::BadParams, error.to_string()))?;
    let since = params.get("since_seq").and_then(Value::as_u64).unwrap_or(0);

    let dir = state
        .journal_dir(&task)
        .map_err(|error| RpcError::new(ErrorCode::NotFound, error.to_string()))?;
    let outcome = JournalReader::open(&dir, &task)
        .and_then(|reader| reader.read())
        .map_err(|error| internal(&error))?;

    // The backlog goes in the acknowledgement so a client joining mid-run renders the history
    // without a second round trip. Filtered by task for the same reason as `status`: the file
    // is shared with every other task in the repository. `since_seq` is an inclusive lower
    // bound and `next_seq` is the first record the client has not seen.
    let owned: Vec<&JournalRecord> = outcome
        .records
        .iter()
        .filter(|record| record.fields.task == task)
        .collect();
    let backlog: Vec<Value> = owned
        .iter()
        .filter(|record| record.fields.seq >= since)
        .filter_map(|record| serde_json::to_value(record).ok())
        .collect();
    let next = owned
        .iter()
        .map(|record| record.fields.seq)
        .filter(|seq| *seq >= since)
        .max()
        .map_or(since, |last| last + 1);

    Ok(json!({
        "task": task,
        "since_seq": since,
        "next_seq": next,
        "backlog": backlog,
        "corrupt": outcome.is_corrupt(),
    }))
}

/// Write one message as a line.
async fn write_line<T: serde::Serialize + Sync>(
    write: &mut tokio::net::unix::OwnedWriteHalf,
    message: &T,
) -> std::io::Result<()> {
    let text = rpc::encode(message)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string()))?;
    write.write_all(text.as_bytes()).await?;
    write.write_all(b"\n").await?;
    write.flush().await
}

/// A core error as an internal RPC error.
fn internal(error: &immortality_core::Error) -> RpcError {
    RpcError::new(ErrorCode::Internal, error.to_string())
}
