//! The socket server, exercised over a real unix socket.
//!
//! # Why a real socket and not the handler functions
//!
//! Every bug this file is looking for lives in the part that a direct call to `dispatch` skips:
//! the token check, the ordering of `hello` before other methods, the `0600` permission, the
//! backlog/stream seam, and what happens to a client that dies mid-stream. Calling the handler
//! would test the JSON shapes and none of the protocol.
//!
//! # Why the tests drive the client by hand
//!
//! There is no client yet at this milestone, and writing one first would let a shared bug pass
//! both sides. Each test speaks the wire format literally — `Request`s in, lines out — so the
//! assertions are about bytes a client would actually receive.

// Integration tests: a failed assertion should name the exact line, and there is no caller to
// propagate to.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::os::unix::fs::PermissionsExt as _;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use immortality_core::escalation::{Answer, Entry, EscalationKind, EscalationStore};
use immortality_core::ids::TaskId;
use immortality_core::journal::{Journal, JournalReader, Kind};
use immortality_core::registry::Registry;
use immortality_daemon::rpc::{self, ErrorCode, Request, Response};
use immortality_daemon::server::{self, Events, ServerState};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader};
use tokio::net::UnixStream;

/// A task id used by every test, so a failure names one thing.
fn task() -> TaskId {
    TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV")
}

/// A second task in the *same* repository, for the filtering tests.
fn other_task() -> TaskId {
    TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAW")
}

/// A daemon's state root, its socket, and the project it knows about.
struct Fixture {
    /// Kept alive for the whole test: dropping it deletes the socket and the repository.
    _dir: tempfile::TempDir,
    /// The `~/Library/Application Support/immortality` equivalent.
    state_root: std::path::PathBuf,
    /// The repository root; `.agent/` lives here.
    repo: std::path::PathBuf,
    socket: std::path::PathBuf,
    token: String,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let state_root = dir.path().join("state-root");
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::create_dir_all(&state_root).unwrap();
        let token = "test-token-0123456789".to_string();
        std::fs::write(state_root.join("daemon.token"), &token).unwrap();

        // A registry entry is what links a task id to its repository, and the socket path is
        // kept short because `sun_path` is 104 bytes on macOS and a tempdir is already long.
        let mut registry = Registry::open(state_root.join(immortality_core::registry::FILE_NAME)).unwrap();
        registry.register(&repo, None).unwrap();

        Self {
            socket: dir.path().join("s.sock"),
            _dir: dir,
            state_root,
            repo,
            token,
        }
    }

    fn server_state(&self) -> ServerState {
        ServerState::discover(&self.state_root).unwrap()
    }

    /// The project whose path is the only one registered, for a test that needs to name a task's
    /// state directory before the task exists.
    fn sole_project(&self) -> immortality_core::registry::Project {
        self.server_state_registry().projects()[0].clone()
    }

    fn server_state_registry(&self) -> Registry {
        Registry::open(self.state_root.join(immortality_core::registry::FILE_NAME)).unwrap()
    }

    /// A task's state directory, as `status` and `escalations` compute it.
    fn task_dir(&self, task: &TaskId) -> std::path::PathBuf {
        self.sole_project()
            .state_dir(&self.state_root)
            .join(task.as_str())
    }

    /// The repository's `.agent/` directory.
    fn agent_dir(&self) -> std::path::PathBuf {
        self.repo.join(immortality_core::paths::AGENT_DIR)
    }

    /// Write `count` records for `task` into the shared journal, returning their sequence
    /// numbers. The journal is per repository and holds every task of that repository, which
    /// is exactly the fact the filtering tests exercise.
    fn journal_records(&self, task: &TaskId, count: usize) -> Vec<u64> {
        let mut journal = Journal::open(self.agent_dir(), task.clone()).unwrap();
        (0..count)
            .map(|index| {
                journal
                    .append_event(
                        immortality_core::time::now(),
                        Kind::TaskState,
                        json!({"n": index}),
                    )
                    .unwrap()
                    .fields
                    .seq
            })
            .collect()
    }

    /// Ask the fixture's escalation store a question, so a daemon has something to answer.
    fn ask(&self, task: &TaskId, key: &str) -> Entry {
        let mut store = EscalationStore::open(&self.task_dir(task), task.clone()).unwrap();
        let entry = Entry::new(
            task.clone(),
            key,
            EscalationKind::ExternalAction,
            "may I push?",
            Vec::new(),
            immortality_core::time::now(),
        );
        match store.ask(entry).unwrap() {
            immortality_core::escalation::Opened::New(entry) => entry,
            other => panic!("expected a new entry, got {other:?}"),
        }
    }

    /// Run the server on its own task, returning the join handle. The socket appears before the
    /// handle is returned, because every test then connects rather than sleeping.
    fn serve(&self) -> tokio::task::JoinHandle<()> {
        let listener = server::bind(&self.socket).unwrap();
        let state = self.server_state();
        let events = Events::new();
        // Bind happened synchronously, so a client may connect now; `serve` only accepts.
        tokio::spawn(async move {
            let _ = server::serve(listener, state, events).await;
        })
    }

    /// A raw connection, with `hello` already accepted.
    async fn client(&self) -> Client {
        let mut client = Client::connect(&self.socket, &self.token).await;
        let response = client.request("hello", json!({})).await;
        assert!(response.error.is_none(), "hello failed: {response:?}");
        client
    }
}

/// One hand-driven connection.
struct Client {
    write: tokio::net::unix::OwnedWriteHalf,
    reader: BufReader<tokio::net::unix::OwnedReadHalf>,
    next_id: u64,
    /// The token sent with every request, as a real client does.
    token: Option<String>,
}

impl Client {
    /// Connect without saying `hello`, so a test can observe the refusal.
    async fn connect(socket: &Path, token: &str) -> Self {
        let stream = UnixStream::connect(socket).await.unwrap();
        let (read, write) = stream.into_split();
        Self {
            write,
            reader: BufReader::new(read),
            next_id: 0,
            token: Some(token.to_string()),
        }
    }

    /// Send one request and read exactly one response.
    async fn request(&mut self, method: &str, params: Value) -> Response {
        self.next_id += 1;
        let request = Request {
            id: self.next_id,
            method: method.to_string(),
            params,
            token: self.token.clone(),
        };
        let line = rpc::encode(&request).unwrap();
        self.write.write_all(line.as_bytes()).await.unwrap();
        self.write.write_all(b"\n").await.unwrap();
        self.write.flush().await.unwrap();
        self.read_response().await
    }

    /// Send a raw line, for the tests that are about malformed input.
    async fn send_raw(&mut self, line: &str) {
        self.write.write_all(line.as_bytes()).await.unwrap();
        self.write.write_all(b"\n").await.unwrap();
        self.write.flush().await.unwrap();
    }

    async fn read_line(&mut self) -> String {
        let mut line = String::new();
        let read = tokio::time::timeout(Duration::from_secs(5), self.reader.read_line(&mut line))
            .await
            .expect("the daemon did not answer within five seconds")
            .unwrap();
        assert!(read > 0, "the daemon closed the connection");
        line
    }

    async fn read_response(&mut self) -> Response {
        let line = self.read_line().await;
        rpc::decode(line.trim_end()).unwrap()
    }

    /// Read one event line.
    async fn read_event(&mut self) -> Value {
        let line = self.read_line().await;
        let event: rpc::Event = rpc::decode(line.trim_end()).unwrap();
        event.record
    }

    /// Whether the peer closed the connection, within the test's patience.
    async fn is_closed(&mut self) -> bool {
        let mut line = String::new();
        tokio::time::timeout(Duration::from_secs(5), self.reader.read_line(&mut line))
            .await
            .expect("the daemon neither answered nor closed within five seconds")
            .unwrap_or(0)
            == 0
    }
}

// ── transport ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn the_socket_is_created_with_mode_0600() {
    // The umask gives 0755; a socket that any local user can connect to is a daemon that any
    // local user can drive, and the token is the second lock, not the first.
    let fixture = Fixture::new();
    let listener = server::bind(&fixture.socket).unwrap();
    let mode = std::fs::metadata(&fixture.socket).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600, "socket mode was {mode:o}");
    drop(listener);
}

#[tokio::test]
async fn binding_twice_replaces_the_stale_socket() {
    // A daemon that was `kill -9`'d leaves its socket file behind, and the next start must not
    // fail with "address already in use" — which is what happens without the removal.
    let fixture = Fixture::new();
    let first = server::bind(&fixture.socket).unwrap();
    drop(first);
    assert!(fixture.socket.exists(), "the file survives a dropped listener");
    let _second = server::bind(&fixture.socket).unwrap();
}

#[tokio::test]
async fn a_wrong_token_is_refused_and_the_connection_is_closed() {
    let fixture = Fixture::new();
    let _server = fixture.serve();

    let mut client = Client::connect(&fixture.socket, "not-the-token").await;
    let response = client.request("hello", json!({})).await;
    let error = response.error.expect("a wrong token must be refused");
    assert_eq!(error.code, ErrorCode::Unauthorized);
    // Closed, not merely refused: a client that may not talk should not get more guesses.
    assert!(client.is_closed().await);
}

#[tokio::test]
async fn a_request_without_a_token_is_refused_but_the_connection_stays() {
    let fixture = Fixture::new();
    let _server = fixture.serve();

    let mut client = Client::connect(&fixture.socket, "").await;
    client.token = None;
    let response = client.request("status", json!({})).await;
    let error = response.error.expect("a token is required");
    assert_eq!(error.code, ErrorCode::Unauthorized);
    assert!(error.message.contains("no token"), "{error:?}");

    // A client that forgot the token on one request can still say `hello` and continue: this
    // is a mistake to fix, not an attack to close the door on.
    client.token = Some(fixture.token.clone());
    let response = client.request("hello", json!({})).await;
    assert!(response.error.is_none(), "{response:?}");
}

#[tokio::test]
async fn the_token_alone_is_not_enough_hello_must_come_first() {
    let fixture = Fixture::new();
    let _server = fixture.serve();

    // A correct token, but no `hello`: the socket accepted the connection and the token is
    // valid, so the only thing left is the ordering rule.
    let mut client = Client::connect(&fixture.socket, &fixture.token).await;
    let response = client.request("status", json!({})).await;
    let error = response.error.expect("status before hello must be refused");
    assert_eq!(error.code, ErrorCode::Unauthorized);
    assert!(error.message.contains("hello"), "{error:?}");
}

#[tokio::test]
async fn hello_refuses_a_client_that_speaks_another_protocol() {
    // The negotiation the roadmap asks for: a newer client must be told its *build* is the
    // problem, not be fed answers meant for a different schema. The daemon is the side that can
    // tell that apart from a bad token, which is why the version is sent with `hello`.
    let fixture = Fixture::new();
    let _server = fixture.serve();

    let mut client = Client::connect(&fixture.socket, &fixture.token).await;
    let response = client
        .request("hello", json!({ "version": rpc::VERSION + 1 }))
        .await;
    let error = response.error.expect("a foreign protocol must be refused");
    assert_eq!(error.code, ErrorCode::VersionMismatch);
    assert!(error.message.contains(&rpc::VERSION.to_string()), "{error:?}");

    // The refusal does not authorise the connection: the next request is still refused, and a
    // daemon has not silently started serving a build it just said it cannot talk to.
    let status = client.request("status", json!({})).await;
    assert_eq!(
        status.error.expect("still unauthenticated").code,
        ErrorCode::Unauthorized
    );
}

#[tokio::test]
async fn hello_without_a_version_is_accepted() {
    // A human debugging with `printf` must still be able to hand-shake: the field is optional,
    // and its absence means "I am not claiming a version" rather than "I claim zero".
    let fixture = Fixture::new();
    let _server = fixture.serve();

    let mut client = Client::connect(&fixture.socket, &fixture.token).await;
    let response = client.request("hello", json!({})).await;
    assert!(response.error.is_none(), "{response:?}");
    assert_eq!(response.result.unwrap()["version"], rpc::VERSION);
}

#[tokio::test]
async fn a_version_mismatch_is_not_reported_as_a_bad_token() {
    // The human-facing consequence of doing the negotiation on the daemon: a stale token and a
    // stale binary produce different messages, because the fix is different.
    let fixture = Fixture::new();
    let _server = fixture.serve();

    let mut stale = Client::connect(&fixture.socket, "wrong-token").await;
    let token_error = stale.request("hello", json!({ "version": rpc::VERSION })).await;
    assert_eq!(token_error.error.expect("refused").code, ErrorCode::Unauthorized);

    let mut wrong_version = Client::connect(&fixture.socket, &fixture.token).await;
    let version_error = wrong_version
        .request("hello", json!({ "version": rpc::VERSION + 7 }))
        .await;
    assert_eq!(
        version_error.error.expect("refused").code,
        ErrorCode::VersionMismatch
    );
}

#[tokio::test]
async fn hello_reports_the_protocol_version() {
    let fixture = Fixture::new();
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    client.next_id += 1;
    let request = Request {
        id: client.next_id,
        method: "hello".to_string(),
        params: Value::Null,
        token: Some(fixture.token.clone()),
    };
    client
        .write
        .write_all(format!("{}\n", rpc::encode(&request).unwrap()).as_bytes())
        .await
        .unwrap();
    let response = client.read_response().await;
    let result = response.result.unwrap();
    assert_eq!(result["version"], rpc::VERSION);
    assert_eq!(result["protocol"], "immortality");
}

#[tokio::test]
async fn a_malformed_line_is_answered_without_killing_the_connection() {
    let fixture = Fixture::new();
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    client.send_raw("this is not JSON").await;
    let response = client.read_response().await;
    let error = response.error.expect("garbage must be refused");
    assert_eq!(error.code, ErrorCode::BadParams);

    // The next request on the same connection still works: one bad line is a client bug, not a
    // reason to make the human reconnect.
    let response = client.request("status", json!({})).await;
    assert!(response.error.is_none(), "{response:?}");
}

#[tokio::test]
async fn an_oversized_line_is_refused_and_closes_the_connection() {
    // Not because the daemon is strict, but because the alternative is that a client can make
    // the daemon allocate without bound. The close is needed too: the rest of the line is still
    // in flight and there is no way to resynchronise in the middle of a message.
    let fixture = Fixture::new();
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    let huge = "x".repeat(rpc::MAX_LINE_BYTES + 1024);
    client.send_raw(&huge).await;
    let response = client.read_response().await;
    let error = response.error.expect("an oversized line must be refused");
    assert_eq!(error.code, ErrorCode::BadParams);
    assert!(error.message.contains("too long"), "{error:?}");
    assert!(client.is_closed().await);
}

#[tokio::test]
async fn an_unknown_method_names_the_known_ones() {
    let fixture = Fixture::new();
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    let response = client.request("push", json!({})).await;
    let error = response.error.expect("an unknown method must be refused");
    assert_eq!(error.code, ErrorCode::UnknownMethod);
    assert!(error.message.contains("status"), "{error:?}");
    // The list is the build's own `METHODS`, so it cannot claim a method that will be refused or
    // omit one that works: both would make the refusal send a human to the wrong place. Read from
    // the constant rather than written out here — a literal is a second copy that goes stale, as
    // the assertion this replaced did when `gc` was implemented.
    for method in immortality_daemon::rpc::METHODS {
        assert!(error.message.contains(method), "`{method}` missing: {error:?}");
    }
    // `push` appears in the sentence ("`push` is not a method"), so the check is against the list
    // after the colon, which is the part a human reads as "what can I call instead".
    let known = error.message.split("known methods: ").nth(1).expect("a list");
    assert!(!known.contains("push"), "{error:?}");
}

// ── status ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn status_lists_the_projects_and_tasks_the_daemon_knows() {
    let fixture = Fixture::new();
    fixture.journal_records(&task(), 3);
    std::fs::create_dir_all(fixture.task_dir(&task())).unwrap();
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    let result = client.request("status", json!({})).await.result.unwrap();
    assert_eq!(result["version"], rpc::VERSION);
    let projects = result["projects"].as_array().unwrap();
    assert_eq!(projects.len(), 1);
    assert!(projects[0]["path"].as_str().unwrap().ends_with("repo"));

    let tasks = result["tasks"].as_array().unwrap();
    assert_eq!(tasks.len(), 1);
    let listed = &tasks[0];
    assert_eq!(listed["task"], task().as_str());
    assert_eq!(listed["journal"]["records"], 3);
    // `next_seq` is the first unseen sequence number. Sequences start at 0, so three records
    // are 0, 1, 2 and the next is 3.
    assert_eq!(listed["journal"]["next_seq"], 3);
    assert_eq!(listed["journal"]["corrupt"], false);
}

#[tokio::test]
async fn status_reports_only_this_tasks_records_from_a_shared_journal() {
    // One `JOURNAL.jsonl` per repository holds every task of that repository. A handler that
    // forgot to filter would report the other task's records as this task's, and a human would
    // read a state that belongs to different work.
    let fixture = Fixture::new();
    fixture.journal_records(&task(), 2);
    fixture.journal_records(&other_task(), 5);
    std::fs::create_dir_all(fixture.task_dir(&task())).unwrap();
    std::fs::create_dir_all(fixture.task_dir(&other_task())).unwrap();
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    let result = client
        .request("status", json!({"task": task().as_str()}))
        .await
        .result
        .unwrap();
    let tasks = result["tasks"].as_array().unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0]["task"], task().as_str());
    assert_eq!(tasks[0]["journal"]["records"], 2);
    assert_eq!(tasks[0]["journal"]["next_seq"], 2);
}

#[tokio::test]
async fn status_for_a_known_task_that_does_not_exist_is_not_found() {
    let fixture = Fixture::new();
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    let unknown = TaskId::generate();
    let response = client.request("status", json!({"task": unknown.as_str()})).await;
    let error = response.error.expect("an unknown task must be refused");
    assert_eq!(error.code, ErrorCode::NotFound);
    assert!(error.message.contains(unknown.as_str()), "{error:?}");
}

#[tokio::test]
async fn status_with_a_malformed_task_id_is_bad_params() {
    let fixture = Fixture::new();
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    let response = client.request("status", json!({"task": "not-a-task"})).await;
    let error = response.error.expect("a bad id must be refused");
    assert_eq!(error.code, ErrorCode::BadParams);
}

#[tokio::test]
async fn status_counts_the_open_escalations() {
    let fixture = Fixture::new();
    std::fs::create_dir_all(fixture.task_dir(&task())).unwrap();
    fixture.ask(&task(), "external:git:push");
    fixture.ask(&task(), "external:net:example.com");
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    let result = client.request("status", json!({})).await.result.unwrap();
    assert_eq!(result["tasks"][0]["open_escalations"], 2);
}

// ── escalations and approve ───────────────────────────────────────────────────

#[tokio::test]
async fn escalations_lists_open_questions_and_approve_answers_one() {
    let fixture = Fixture::new();
    std::fs::create_dir_all(fixture.task_dir(&task())).unwrap();
    let entry = fixture.ask(&task(), "external:git:push");
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    let listed = client.request("escalations", json!({})).await.result.unwrap();
    let entries = listed["escalations"].as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["id"], entry.id.as_str());
    assert_eq!(entries[0]["key"], "external:git:push");
    assert_eq!(entries[0]["kind"], "external_action");
    assert_eq!(entries[0]["question"], "may I push?");
    assert_eq!(entries[0]["blocking"], false);
    // The three answers a human may give, in the CLI's order.
    assert_eq!(
        entries[0]["options"],
        json!(["allow_once", "allow_for_task", "deny"])
    );

    let answered = client
        .request(
            "approve",
            json!({"id": entry.id.as_str(), "answer": "allow_for_task"}),
        )
        .await
        .result
        .unwrap();
    assert_eq!(answered["id"], entry.id.as_str());
    assert_eq!(answered["answer"], "allow_for_task");

    // Answered questions leave the open list, which is what makes the listing usable.
    let listed = client.request("escalations", json!({})).await.result.unwrap();
    assert_eq!(listed["escalations"], json!([]));

    // And the answer is durable, not just returned.
    let store = EscalationStore::open(&fixture.task_dir(&task()), task()).unwrap();
    assert!(store.allowance("external:git:push"));
}

#[tokio::test]
async fn approving_an_unknown_escalation_is_not_found() {
    let fixture = Fixture::new();
    std::fs::create_dir_all(fixture.task_dir(&task())).unwrap();
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    let unknown = immortality_core::ids::EscalationId::generate();
    let response = client
        .request("approve", json!({"id": unknown.as_str(), "answer": "deny"}))
        .await;
    let error = response.error.expect("an unknown escalation must be refused");
    assert_eq!(error.code, ErrorCode::NotFound);
    assert!(error.message.contains(unknown.as_str()), "{error:?}");
}

#[tokio::test]
async fn approve_requires_a_valid_id_and_answer() {
    let fixture = Fixture::new();
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    for params in [
        json!({"answer": "deny"}),
        json!({"id": "esc_nonsense", "answer": "deny"}),
        json!({"id": immortality_core::ids::EscalationId::generate().as_str()}),
        json!({"id": immortality_core::ids::EscalationId::generate().as_str(), "answer": "sure"}),
    ] {
        let response = client.request("approve", params.clone()).await;
        let error = response.error.expect("malformed params must be refused");
        assert_eq!(error.code, ErrorCode::BadParams, "{params}: {error:?}");
    }
}

#[tokio::test]
async fn answering_the_same_question_twice_with_the_same_answer_is_not_an_error() {
    // The CLI retries after a lost socket far more often than a human double-clicks.
    let fixture = Fixture::new();
    std::fs::create_dir_all(fixture.task_dir(&task())).unwrap();
    let entry = fixture.ask(&task(), "external:git:push");
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    let params = json!({"id": entry.id.as_str(), "answer": "deny"});
    let first = client.request("approve", params.clone()).await;
    assert!(first.error.is_none(), "{first:?}");
    let second = client.request("approve", params).await;
    assert!(second.error.is_none(), "{second:?}");
    assert_eq!(second.result.unwrap()["answer"], "deny");
}

#[tokio::test]
async fn a_conflicting_second_answer_is_refused_rather_than_silently_replaced() {
    // Changing a decision after the agent has already acted on it is a yes-then-no that the
    // agent cannot honour; the store refuses, and the RPC must surface that as an error.
    let fixture = Fixture::new();
    std::fs::create_dir_all(fixture.task_dir(&task())).unwrap();
    let entry = fixture.ask(&task(), "external:git:push");
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    let first = client
        .request(
            "approve",
            json!({"id": entry.id.as_str(), "answer": "allow_once"}),
        )
        .await;
    assert!(first.error.is_none(), "{first:?}");
    let second = client
        .request("approve", json!({"id": entry.id.as_str(), "answer": "deny"}))
        .await;
    assert!(second.error.is_some(), "a conflicting answer must be refused");
}

#[tokio::test]
async fn escalations_can_be_filtered_to_one_task() {
    let fixture = Fixture::new();
    std::fs::create_dir_all(fixture.task_dir(&task())).unwrap();
    std::fs::create_dir_all(fixture.task_dir(&other_task())).unwrap();
    fixture.ask(&task(), "external:git:push");
    fixture.ask(&other_task(), "external:net:example.com");
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    let listed = client
        .request("escalations", json!({"task": other_task().as_str()}))
        .await
        .result
        .unwrap();
    let entries = listed["escalations"].as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["task"], other_task().as_str());
}

// ── attach ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn attach_replays_the_backlog_from_since_seq_inclusively() {
    // `since_seq` is an inclusive lower bound (D-066), so a client that last saw 2 asks for 2
    // and gets 2 again. Exclusive would silently skip the record after a re-attach.
    let fixture = Fixture::new();
    fixture.journal_records(&task(), 4);
    std::fs::create_dir_all(fixture.task_dir(&task())).unwrap();
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    // Sequences run 0, 1, 2, 3. Asking from 2 (inclusive) gives the last two.
    let result = client
        .request("attach", json!({"task": task().as_str(), "since_seq": 2}))
        .await
        .result
        .unwrap();
    let backlog = result["backlog"].as_array().unwrap();
    let seqs: Vec<u64> = backlog
        .iter()
        .map(|record| record["seq"].as_u64().unwrap())
        .collect();
    assert_eq!(seqs, vec![2, 3]);
    assert_eq!(result["since_seq"], 2);
    // The first record the client has not seen is 4, and that is where the stream starts.
    assert_eq!(result["next_seq"], 4);

    // The streamed records are the journal's own JSON, not a second shape.
    let written = JournalReader::open(fixture.agent_dir(), &task())
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(backlog[0]["kind"], Kind::TaskState.as_str());
    assert_eq!(backlog[0]["payload"], written.records[2].event.payload);
    assert_eq!(backlog[0]["task"], task().as_str());
}

#[tokio::test]
async fn attach_with_no_backlog_starts_at_the_first_record() {
    let fixture = Fixture::new();
    fixture.journal_records(&task(), 2);
    std::fs::create_dir_all(fixture.task_dir(&task())).unwrap();
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    let result = client
        .request("attach", json!({"task": task().as_str()}))
        .await
        .result
        .unwrap();
    assert_eq!(result["since_seq"], 0);
    assert_eq!(result["backlog"].as_array().unwrap().len(), 2);
    assert_eq!(result["next_seq"], 2);
}

#[tokio::test]
async fn attach_to_an_unknown_task_is_not_found() {
    let fixture = Fixture::new();
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    let unknown = TaskId::generate();
    let response = client.request("attach", json!({"task": unknown.as_str()})).await;
    let error = response.error.expect("an unknown task must be refused");
    assert_eq!(error.code, ErrorCode::NotFound);
}

#[tokio::test]
async fn attach_without_a_task_is_bad_params() {
    let fixture = Fixture::new();
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    let response = client.request("attach", json!({})).await;
    let error = response.error.expect("`task` is required");
    assert_eq!(error.code, ErrorCode::BadParams);
}

#[tokio::test]
async fn attach_with_no_daemon_running_is_a_connect_error_not_a_hang() {
    // The roadmap's "clear error when the daemon is absent". A connect to a socket that does
    // not exist fails immediately; this asserts the *client* side of that contract with the
    // exact errno a CLI prints.
    let fixture = Fixture::new();
    let result = UnixStream::connect(&fixture.socket).await;
    let error = result.expect_err("no daemon is listening");
    assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
}

// ── the stream ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn two_clients_attached_at_once_both_see_the_same_events() {
    // The roadmap's "two clients simultaneously" test, and the reason the daemon is the owner
    // of state: a second terminal is a subscriber, not a second engine.
    let fixture = Fixture::new();
    std::fs::create_dir_all(fixture.task_dir(&task())).unwrap();
    let listener = server::bind(&fixture.socket).unwrap();
    let state = fixture.server_state();
    let events = Events::new();
    let server_events = events.clone();
    tokio::spawn(async move {
        let _ = server::serve(listener, state, events).await;
    });

    let mut first = fixture.client().await;
    let mut second = fixture.client().await;
    for client in [&mut first, &mut second] {
        let result = client
            .request("attach", json!({"task": task().as_str()}))
            .await
            .result
            .unwrap();
        assert_eq!(result["next_seq"], 0);
        assert_eq!(result["backlog"], json!([]));
    }

    // Publish through the daemon's own handle, as the runner will after a tool call.
    let mut journal = Journal::open(fixture.agent_dir(), task()).unwrap();
    let record = journal
        .append_event(
            immortality_core::time::now(),
            Kind::ToolCallCompleted,
            json!({"call": "call_1"}),
        )
        .unwrap();
    server_events.publish(Arc::new(record.clone()));

    for client in [&mut first, &mut second] {
        let event = client.read_event().await;
        assert_eq!(event["seq"], 0);
        assert_eq!(event["kind"], "tool_call_completed");
        assert_eq!(event["payload"]["call"], "call_1");
    }
}

#[tokio::test]
async fn a_client_that_reconnects_with_since_seq_sees_no_duplicate_and_no_gap() {
    // The roadmap's "client killed and restarted" test. The dangerous bug is the seam: a record
    // written between the backlog read and the subscription is in neither the backlog nor the
    // stream, and the client never learns it happened.
    let fixture = Fixture::new();
    std::fs::create_dir_all(fixture.task_dir(&task())).unwrap();
    let first_two = fixture.journal_records(&task(), 2);
    let listener = server::bind(&fixture.socket).unwrap();
    let state = fixture.server_state();
    let events = Events::new();
    let server_events = events.clone();
    tokio::spawn(async move {
        let _ = server::serve(listener, state, events).await;
    });

    let mut first = fixture.client().await;
    let result = first
        .request("attach", json!({"task": task().as_str()}))
        .await
        .result
        .unwrap();
    assert_eq!(result["backlog"].as_array().unwrap().len(), 2);
    let next = result["next_seq"].as_u64().unwrap();
    assert_eq!(next, first_two.last().unwrap() + 1);

    // A record arrives while the client is attached, then the client dies. Publishing before
    // the kill is what leaves the record in the journal but possibly unread.
    let mut journal = Journal::open(fixture.agent_dir(), task()).unwrap();
    let third = journal
        .append_event(immortality_core::time::now(), Kind::TaskState, json!({"n": 2}))
        .unwrap();
    server_events.publish(Arc::new(third.clone()));
    let event = first.read_event().await;
    assert_eq!(event["seq"], third.fields.seq);
    drop(first);

    // Re-attach from the last seen sequence: exactly the third record comes back, once.
    let mut second = fixture.client().await;
    let result = second
        .request(
            "attach",
            json!({"task": task().as_str(), "since_seq": third.fields.seq}),
        )
        .await
        .result
        .unwrap();
    let backlog = result["backlog"].as_array().unwrap();
    assert_eq!(backlog.len(), 1, "the replay must be exactly the unseen record");
    assert_eq!(backlog[0]["seq"], third.fields.seq);
    assert_eq!(result["next_seq"], third.fields.seq + 1);
}

#[tokio::test]
async fn the_stream_drops_a_record_the_backlog_already_contained() {
    // Subscribing before reading the backlog closes the lost-record window, but it means the
    // broadcast can carry a record the acknowledgement already contained. Without the `next_seq`
    // watermark the client renders it twice, which looks like a duplicate tool call — and a
    // duplicate `tool_call_completed` is exactly what recovery is built to distrust.
    let fixture = Fixture::new();
    fixture.journal_records(&task(), 3);
    std::fs::create_dir_all(fixture.task_dir(&task())).unwrap();
    let listener = server::bind(&fixture.socket).unwrap();
    let state = fixture.server_state();
    let events = Events::new();
    let server_events = events.clone();
    tokio::spawn(async move {
        let _ = server::serve(listener, state, events).await;
    });

    let mut client = fixture.client().await;
    let result = client
        .request("attach", json!({"task": task().as_str()}))
        .await
        .result
        .unwrap();
    assert_eq!(result["backlog"].as_array().unwrap().len(), 3);
    assert_eq!(result["next_seq"], 3);

    // Exactly the race: a record already in the backlog arrives on the broadcast after the
    // acknowledgement. The harness publishes record 0 again, which is what the daemon would do
    // if its write and its publish straddled the client's attach.
    let written = JournalReader::open(fixture.agent_dir(), &task())
        .unwrap()
        .read()
        .unwrap();
    server_events.publish(Arc::new(written.records[0].clone()));

    // Then a genuinely new record. The client must see *it* next, proving the older one was
    // dropped rather than merely not yet delivered.
    let mut journal = Journal::open(fixture.agent_dir(), task()).unwrap();
    let fresh = journal
        .append_event(
            immortality_core::time::now(),
            Kind::ToolCallCompleted,
            json!({"call": "call_fresh"}),
        )
        .unwrap();
    assert_eq!(fresh.fields.seq, 3);
    server_events.publish(Arc::new(fresh.clone()));

    let event = client.read_event().await;
    assert_eq!(event["seq"], 3, "the replayed record must not be streamed again");
    assert_eq!(event["payload"]["call"], "call_fresh");
}

#[tokio::test]
async fn attach_replays_only_the_named_tasks_backlog() {
    // The backlog is read from the repository-wide journal, so it needs the same filter the
    // stream has. Without it, attaching to one task renders another task's history first.
    let fixture = Fixture::new();
    fixture.journal_records(&other_task(), 3);
    fixture.journal_records(&task(), 2);
    std::fs::create_dir_all(fixture.task_dir(&task())).unwrap();
    std::fs::create_dir_all(fixture.task_dir(&other_task())).unwrap();
    let _server = fixture.serve();

    let mut client = fixture.client().await;
    let result = client
        .request("attach", json!({"task": task().as_str()}))
        .await
        .result
        .unwrap();
    let backlog = result["backlog"].as_array().unwrap();
    assert_eq!(backlog.len(), 2);
    for record in backlog {
        assert_eq!(record["task"], task().as_str());
    }
}

#[tokio::test]
async fn the_stream_carries_only_the_attached_tasks_records() {
    // The broadcast is daemon-wide: every task, every project. A filter after the fact is the
    // only thing stopping another task's work from appearing in this terminal.
    let fixture = Fixture::new();
    std::fs::create_dir_all(fixture.task_dir(&task())).unwrap();
    std::fs::create_dir_all(fixture.task_dir(&other_task())).unwrap();
    let listener = server::bind(&fixture.socket).unwrap();
    let state = fixture.server_state();
    let events = Events::new();
    let server_events = events.clone();
    tokio::spawn(async move {
        let _ = server::serve(listener, state, events).await;
    });

    let mut client = fixture.client().await;
    client
        .request("attach", json!({"task": task().as_str()}))
        .await
        .result
        .unwrap();

    // The other task's record arrives first, then this task's. The client must see only the
    // second, and must see it rather than time out waiting for the first to be filtered.
    let mut journal = Journal::open(fixture.agent_dir(), other_task()).unwrap();
    let other = journal
        .append_event(
            immortality_core::time::now(),
            Kind::TaskState,
            json!({"who": "other"}),
        )
        .unwrap();
    server_events.publish(Arc::new(other));
    let mut journal = Journal::open(fixture.agent_dir(), task()).unwrap();
    let mine = journal
        .append_event(
            immortality_core::time::now(),
            Kind::TaskState,
            json!({"who": "mine"}),
        )
        .unwrap();
    server_events.publish(Arc::new(mine.clone()));

    let event = client.read_event().await;
    assert_eq!(event["task"], task().as_str());
    assert_eq!(event["payload"]["who"], "mine");
    assert_eq!(event["seq"], mine.fields.seq);
}

#[tokio::test]
async fn publishing_with_nobody_attached_is_not_an_error() {
    // The daemon runs whether or not a terminal is attached; that is the point of a daemon.
    let events = Events::new();
    let mut journal = Journal::open(tempfile::tempdir().unwrap().path(), task()).unwrap();
    let record = journal
        .append_event(immortality_core::time::now(), Kind::DaemonStarted, json!({}))
        .unwrap();
    events.publish(Arc::new(record));
}

// ── state discovery ───────────────────────────────────────────────────────────

#[test]
fn a_missing_token_is_a_startup_error_not_an_open_daemon() {
    // A daemon that could not find its token must refuse to start, not accept everyone.
    let dir = tempfile::tempdir().unwrap();
    let error = ServerState::discover(dir.path()).unwrap_err();
    assert!(error.to_string().contains("daemon.token"), "{error}");
}

#[test]
fn an_empty_token_is_refused_because_it_would_authorise_everything() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("daemon.token"), "   \n").unwrap();
    let error = ServerState::discover(dir.path()).unwrap_err();
    assert!(error.to_string().contains("empty"), "{error}");
}

#[test]
fn a_token_is_read_without_its_trailing_newline() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("daemon.token"), "secret\n").unwrap();
    let state = ServerState::discover(dir.path()).unwrap();
    assert_eq!(state.token, "secret");
}

#[test]
fn a_task_no_project_claims_has_no_state_directory() {
    // Guessing the project from the process's current directory would make the answer depend
    // on where the human ran the CLI from.
    let fixture = Fixture::new();
    let state = fixture.server_state();
    let error = state.task_dir(&TaskId::generate()).unwrap_err();
    assert!(error.to_string().contains("no project"), "{error}");
}

/// The escalation options field is `Vec<Answer>`; this asserts the wire names stay stable.
#[test]
fn escalation_answers_keep_their_wire_names() {
    assert_eq!(Answer::ALL.len(), 3);
    assert_eq!(Answer::AllowOnce.as_str(), "allow_once");
}
