//! The CLI against a real daemon on a real socket.
//!
//! # Why this exists on top of the server's own tests
//!
//! Those drive the wire by hand. This runs the actual binary as a child process and the actual
//! server in a background task, so the things a unit test cannot see are covered: the exit
//! code, stdout versus stderr, that `--json` is parseable, and — most of all — that the two
//! ends agree on the wire format. A field renamed on one side passes every test on the other.
//!
//! # Why the assertions are about text, not structure
//!
//! A human reads this output. Asserting that the id appears is what verifies the whole chain
//! from journal to terminal, including the rendering that the server's tests never touch.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use assert_cmd::prelude::*;
use immortality_core::escalation::{Entry, EscalationKind, EscalationStore};
use immortality_core::ids::TaskId;
use immortality_core::journal::{Journal, Kind};
use immortality_core::registry::Registry;
use immortality_daemon::server::{self, Events, ServerState};
use predicates::prelude::*;

/// A task id used by every test here.
fn task() -> TaskId {
    TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV")
}

/// A daemon in a tempdir, with a registered project and one task.
struct Daemon {
    dir: tempfile::TempDir,
    state_root: PathBuf,
    repo: PathBuf,
    socket: PathBuf,
    token_path: PathBuf,
    global_config: PathBuf,
    /// What the planner should answer, when a test sets one.
    model: Option<std::sync::Arc<Answer>>,
}

/// A model that answers with a fixed string.
///
/// Lives in this file rather than in production because it has no reason to exist there: the
/// seam it satisfies is [`immortality_daemon::runner::Model`], and a fake belongs on the test
/// side of it.
#[derive(Debug)]
struct Answer(String);

impl immortality_daemon::runner::Model for Answer {
    fn complete(
        &self,
        _request: &immortality_daemon::runner::ModelRequest,
    ) -> Result<immortality_daemon::runner::Reply, String> {
        Ok(immortality_daemon::runner::Reply::text(self.0.clone()))
    }
}

impl Daemon {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let state_root = dir.path().join("state");
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::create_dir_all(&state_root).unwrap();

        let token_path = state_root.join("daemon.token");
        std::fs::write(&token_path, "test-token\n").unwrap();

        // Where the client computes it: `daemon_socket()` is `<state root>/daemon.sock`. The
        // fixture used to invent a shorter path, which made every test connect to nothing and
        // report "no daemon is running" — a green-looking suite that tested the error message.
        let socket = state_root.join("daemon.sock");
        let global_config = dir.path().join("global.toml");
        std::fs::write(
            &global_config,
            format!(
                "[backend]\nbase_url = \"http://127.0.0.1:1/v1\"\nmodel = \"test-model\"\nconnect_timeout_secs = 1\n\n[logging]\ndir = \"{}\"\n",
                dir.path().join("logs").display()
            ),
        )
        .unwrap();

        let mut registry = Registry::open(state_root.join(immortality_core::registry::FILE_NAME)).unwrap();
        registry.register(&repo, None).unwrap();

        Self {
            dir,
            state_root,
            repo,
            socket,
            token_path,
            global_config,
            model: None,
        }
    }

    /// Answer the next planner call with `answer`.
    fn answering(mut self, answer: &str) -> Self {
        self.model = Some(std::sync::Arc::new(Answer(answer.to_string())));
        self
    }

    /// Where a task's machine-local state lives.
    fn task_dir(&self) -> PathBuf {
        let registry = Registry::open(self.state_root.join(immortality_core::registry::FILE_NAME)).unwrap();
        registry.projects()[0]
            .state_dir(&self.state_root)
            .join(task().as_str())
    }

    /// Write `count` journal records for the task.
    fn journal(&self, count: usize) {
        let agent = self.repo.join(immortality_core::paths::AGENT_DIR);
        let mut journal = Journal::open(&agent, task()).unwrap();
        for index in 0..count {
            journal
                .append_event(
                    immortality_core::time::now(),
                    Kind::TaskState,
                    serde_json::json!({"n": index}),
                )
                .unwrap();
        }
    }

    /// Ask a question of the task's escalation queue.
    fn ask(&self) -> Entry {
        std::fs::create_dir_all(self.task_dir()).unwrap();
        let mut store = EscalationStore::open(&self.task_dir(), task()).unwrap();
        let entry = Entry::new(
            task(),
            "external:git:push",
            EscalationKind::ExternalAction,
            "may I push to origin?",
            Vec::new(),
            immortality_core::time::now(),
        );
        match store.ask(entry).unwrap() {
            immortality_core::escalation::Opened::New(entry) => entry,
            other => panic!("expected a new escalation, got {other:?}"),
        }
    }

    /// Start the server on its own OS thread and wait until the socket is accepting.
    ///
    /// Its own thread with its own runtime, not `tokio::spawn`: these tests call
    /// `Command::output()`, which blocks the calling thread. On a single-threaded test runtime
    /// that blocking call *is* the runtime, so a spawned server is never polled and every
    /// command times out after 30 seconds. `bind` is inside the runtime as well, because
    /// `tokio::net::UnixListener::bind` registers with the reactor and panics without one. A
    /// separate runtime is also closer to the real deployment, where the daemon is a different
    /// process.
    fn serve(&self) {
        let socket = self.socket.clone();
        let state_root = self.state_root.clone();
        let model = self.model.clone();
        let (ready, wait) = std::sync::mpsc::channel::<Result<(), String>>();
        std::thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            // Everything runs inside `block_on`, `bind` included: building a runtime does not
            // enter it, and `tokio::net::UnixListener::bind` registers with the reactor. A
            // `bind` outside the entered runtime panics with "there is no reactor running".
            runtime.block_on(async move {
                let listener = match server::bind(&socket) {
                    Ok(listener) => listener,
                    Err(error) => {
                        let _ = ready.send(Err(error.to_string()));
                        return;
                    }
                };
                let state = match ServerState::discover(&state_root) {
                    Ok(state) => state,
                    Err(error) => {
                        let _ = ready.send(Err(error.to_string()));
                        return;
                    }
                };
                let state = match model {
                    Some(model) => state.with_model(model),
                    None => state,
                };
                // Sent after `bind`: the listening socket exists, so the test's connect is
                // queued by the kernel even before `accept` runs.
                let _ = ready.send(Ok(()));
                if let Err(error) = server::serve(listener, state, Events::new()).await {
                    tracing::debug!("the test daemon stopped: {error}");
                }
            });
        });
        match wait.recv() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => panic!("the daemon could not start: {error}"),
            Err(error) => panic!("the daemon thread died before starting: {error}"),
        }
    }

    /// The client binary, pointed at this daemon.
    fn cli(&self) -> Command {
        let mut command = Command::cargo_bin("immortality").expect("binary built");
        command
            .env("IMMORTALITY_STATE_DIR", &self.state_root)
            .arg("--global-config")
            .arg(&self.global_config);
        command
    }

    fn token_path(&self) -> &Path {
        &self.token_path
    }

    /// The fixture's tempdir, for a test that needs a sibling directory next to the state root.
    fn root(&self) -> &Path {
        self.dir.path()
    }
}

#[test]
fn status_prints_a_task_that_exists_only_in_the_journal() {
    // A task is real as soon as it journals: its machine-local directory is created lazily.
    // Reporting "no tasks" for a run that has visibly started is the failure this covers.
    let daemon = Daemon::new();
    daemon.journal(3);
    daemon.serve();

    daemon
        .cli()
        .arg("status")
        .assert()
        .success()
        .stdout(predicate::str::contains(task().as_str()))
        .stdout(predicate::str::contains("3 record"))
        .stdout(predicate::str::contains("task_state"));
}

#[test]
fn human_output_to_a_pipe_carries_no_escape_sequences() {
    // Assert-cmd captures stdout through a pipe, which is exactly the logged-or-redirected case.
    // This is the end-to-end proof of the `should_colour` rule: the unit tests pin the decision,
    // and this pins that every command actually consults it.
    let daemon = Daemon::new();
    daemon.journal(1);
    daemon.ask();
    daemon.serve();

    for arguments in [vec!["status"], vec!["escalations"]] {
        let output = daemon.cli().args(&arguments).output().unwrap();
        assert!(output.status.success(), "{arguments:?}");
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(
            !text.contains('\u{1b}'),
            "{arguments:?} wrote escape sequences to a pipe: {text:?}"
        );
        assert!(!text.is_empty(), "{arguments:?} printed nothing");
    }
}

#[test]
fn status_json_is_machine_readable() {
    let daemon = Daemon::new();
    daemon.journal(2);
    daemon.serve();

    let output = daemon.cli().arg("status").arg("--json").output().unwrap();
    assert!(output.status.success());
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["tasks"][0]["task"], task().as_str());
    assert_eq!(value["tasks"][0]["journal"]["records"], 2);
    assert_eq!(
        value["projects"][0]["path"],
        serde_json::json!(daemon.repo.canonicalize().unwrap())
    );
}

#[test]
fn status_for_an_unknown_task_exits_nonzero_with_the_reason() {
    let daemon = Daemon::new();
    daemon.journal(1);
    daemon.serve();

    let unknown = TaskId::generate();
    daemon
        .cli()
        .arg("status")
        .arg(unknown.as_str())
        .assert()
        .failure()
        .stderr(predicate::str::contains("no task"))
        .stderr(predicate::str::contains("immortality status"));
}

#[test]
fn escalations_lists_a_question_and_approve_answers_it() {
    let daemon = Daemon::new();
    daemon.journal(1);
    let entry = daemon.ask();
    daemon.serve();

    daemon
        .cli()
        .arg("escalations")
        .assert()
        .success()
        .stdout(predicate::str::contains(entry.id.as_str()))
        .stdout(predicate::str::contains("may I push to origin?"))
        .stdout(predicate::str::contains("allow_once"));

    daemon
        .cli()
        .arg("approve")
        .arg(entry.id.as_str())
        .arg("allow_for_task")
        .assert()
        .success()
        .stdout(predicate::str::contains("allow_for_task"));

    // The answer is durable: the queue agrees, and the listing is empty afterwards.
    let store = EscalationStore::open(&daemon.task_dir(), task()).unwrap();
    assert!(store.allowance("external:git:push"));
    daemon
        .cli()
        .arg("escalations")
        .assert()
        .success()
        .stdout(predicate::str::contains("nothing is waiting"));
}

#[test]
fn approve_with_an_unknown_id_exits_nonzero() {
    let daemon = Daemon::new();
    daemon.journal(1);
    daemon.serve();

    let unknown = immortality_core::ids::EscalationId::generate();
    daemon
        .cli()
        .arg("approve")
        .arg(unknown.as_str())
        .arg("deny")
        .assert()
        .failure()
        .stderr(predicate::str::contains("no escalation"));
}

#[test]
fn attach_replays_the_backlog_through_the_renderer() {
    // The whole chain: journal file, server filter, wire, client decode, one-line render.
    let daemon = Daemon::new();
    daemon.journal(2);
    daemon.serve();

    // `attach` streams forever, so the test kills it after the backlog arrives. `timeout` is
    // not available cross-platform, so the assertion is on the bytes written before the kill.
    let mut child = daemon
        .cli()
        .arg("attach")
        .arg(task().as_str())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    let stdout = child.stdout.take().unwrap();
    let mut reader = std::io::BufReader::new(stdout);
    let mut line = String::new();
    std::io::BufRead::read_line(&mut reader, &mut line).unwrap();
    // The rendered line is `<seq right-aligned>  <kind>  <detail>`, so the fields are split
    // rather than searched: `contains('0')` would also be satisfied by a seq of 10 or a digit
    // in the payload.
    let mut fields = line.split_whitespace();
    assert_eq!(fields.next(), Some("0"), "first line was {line:?}");
    assert_eq!(fields.next(), Some("task_state"), "first line was {line:?}");
    // Styling is decided by `should_colour`, and stdout here is a pipe. A redirected run must be
    // a clean file with no escape sequences at all.
    assert!(!line.contains('\u{1b}'), "a pipe must get plain text: {line:?}");

    let mut second = String::new();
    std::io::BufRead::read_line(&mut reader, &mut second).unwrap();
    let mut fields = second.split_whitespace();
    assert_eq!(fields.next(), Some("1"), "second line was {second:?}");
    assert_eq!(fields.next(), Some("task_state"), "second line was {second:?}");

    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn attach_without_a_task_when_several_exist_refuses_to_guess() {
    let daemon = Daemon::new();
    daemon.journal(1);
    // A second task, so the choice is ambiguous.
    let agent = daemon.repo.join(immortality_core::paths::AGENT_DIR);
    let mut other = Journal::open(&agent, TaskId::generate()).unwrap();
    other
        .append_event(
            immortality_core::time::now(),
            Kind::TaskState,
            serde_json::json!({}),
        )
        .unwrap();
    daemon.serve();

    daemon
        .cli()
        .arg("attach")
        .assert()
        .failure()
        .stderr(predicate::str::contains("name the one to watch"));
}

#[test]
fn attach_with_no_tasks_says_to_start_one() {
    let daemon = Daemon::new();
    daemon.serve();

    daemon
        .cli()
        .arg("attach")
        .assert()
        .failure()
        .stderr(predicate::str::contains("immortality start"));
}

#[test]
fn attach_with_json_prints_one_record_per_line() {
    let daemon = Daemon::new();
    daemon.journal(1);
    daemon.serve();

    let mut child = daemon
        .cli()
        .arg("attach")
        .arg(task().as_str())
        .arg("--json")
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut reader = std::io::BufReader::new(stdout);
    let mut line = String::new();
    std::io::BufRead::read_line(&mut reader, &mut line).unwrap();
    let value: serde_json::Value = serde_json::from_str(line.trim_end()).unwrap();
    assert_eq!(value["task"], task().as_str());
    assert_eq!(value["kind"], "task_state");
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn a_wrong_token_is_reported_as_a_token_problem() {
    // The socket is reachable and the token is wrong. The message must not send the human to
    // "start the daemon", which is already running.
    let daemon = Daemon::new();
    daemon.journal(1);
    daemon.serve();
    std::fs::write(daemon.token_path(), "a-different-token\n").unwrap();

    daemon
        .cli()
        .arg("status")
        .assert()
        .failure()
        .stderr(predicate::str::contains("refused this client"));
}

#[test]
fn serve_brings_up_a_daemon_that_the_other_commands_can_reach() {
    // The end-to-end proof that the CLI's own daemon works: a real child process, a real socket,
    // a real token minted on first start. Before this, `serve` was a "not implemented until M5"
    // stub and every socket command could only ever be tested against an in-process server.
    let daemon = Daemon::new();
    daemon.journal(2);
    // The fixture writes a token to keep the in-process server honest; remove it so the minting
    // path is what runs here.
    std::fs::remove_file(daemon.token_path()).unwrap();
    assert!(!daemon.socket.exists(), "the fixture starts with no socket");

    let mut child = daemon
        .cli()
        .arg("serve")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    // Wait for the socket rather than sleeping a fixed time: `bind` is the last step before the
    // server accepts, so the file existing is the signal that a connect will be queued.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !daemon.socket.exists() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(daemon.socket.exists(), "the daemon never bound its socket");

    // The token was minted, is non-empty, and is mode 0600.
    let token = std::fs::read_to_string(daemon.token_path()).expect("minted token");
    assert_ne!(token.trim(), "", "an empty token would authorise everyone");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = std::fs::metadata(daemon.token_path())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "token mode {mode:o}");
    }

    // An existing token is never rewritten: a restart must not lock out a terminal attached to
    // the previous daemon, and a token that changes silently is one a human cannot trust.
    let minted = token;
    daemon.cli().arg("status").assert().success();
    assert_eq!(
        std::fs::read_to_string(daemon.token_path()).expect("reread"),
        minted,
        "a second start replaced the token"
    );

    // `status` reaches it and sees the journal.
    daemon
        .cli()
        .arg("status")
        .assert()
        .success()
        .stdout(predicate::str::contains(task().as_str()))
        .stdout(predicate::str::contains("2 record"));

    // A second `serve` on the same state root refuses instead of racing on the journal.
    daemon
        .cli()
        .arg("serve")
        .assert()
        .failure()
        .stderr(predicate::str::contains("already running"));

    let _ = child.kill();
    let _ = child.wait();

    // `kill -9` cannot run any cleanup, so the socket file usually survives. The next start must
    // still come up: `server::bind` removes a stale file before binding. This is the case that
    // matters after a crash, and it is the reason a leftover socket is not an error.
    assert!(daemon.socket.exists(), "SIGKILL leaves the socket behind");

    let mut child = daemon
        .cli()
        .arg("serve")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut reachable = false;
    while std::time::Instant::now() < deadline {
        let probe = daemon.cli().arg("status").output().unwrap();
        if probe.status.success() && String::from_utf8_lossy(&probe.stdout).contains("2 record") {
            reachable = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let _ = child.kill();
    let _ = child.wait();
    assert!(reachable, "a restart over a killed daemon never became reachable");
}

#[test]
fn sigterm_removes_the_socket_and_the_lock() {
    // The cleanup path a human actually triggers: Ctrl-C and `kill` are both signals, and the
    // help text promises the socket goes away. Before this was handled the process died inside
    // `accept`, no destructor ran, and the socket stayed behind — a documented promise that was
    // simply false. `SIGKILL` cannot be caught; that case is covered by the restart test.
    let daemon = Daemon::new();
    daemon.journal(1);

    let mut child = daemon
        .cli()
        .arg("serve")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !daemon.socket.exists() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(daemon.socket.exists(), "the daemon never bound its socket");

    let killed = std::process::Command::new("kill")
        .arg("-TERM")
        .arg(child.id().to_string())
        .status()
        .expect("send SIGTERM");
    assert!(killed.success());

    // Waiting on the child reaps it, which is the synchronization: by the time `wait` returns,
    // the process has run its shutdown path or died, so the filesystem is settled.
    let status = child.wait().unwrap();
    assert!(
        status.success(),
        "a signalled daemon must exit by its own cleanup path, got {status:?}"
    );

    assert!(!daemon.socket.exists(), "the socket must be gone after SIGTERM");
    assert!(
        !daemon.state_root.join("daemon.lock").exists(),
        "the lock file must be gone after SIGTERM"
    );
    // The token is not cleanup: removing it would force every client to re-read a different
    // secret, and the next start would mint a new one anyway.
    assert!(
        daemon.token_path().exists(),
        "the token survives a clean shutdown"
    );
}

#[test]
fn init_registers_the_repository_so_the_daemon_can_find_it_later() {
    let daemon = Daemon::new();
    let project = daemon.root().join("fresh");
    std::fs::create_dir_all(&project).unwrap();

    let mut command = Command::cargo_bin("immortality").expect("binary built");
    command
        .env("IMMORTALITY_STATE_DIR", &daemon.state_root)
        .arg("--global-config")
        .arg(&daemon.global_config)
        .current_dir(&project)
        .arg("init")
        .assert()
        .success()
        .stdout(predicate::str::contains("registered project"));

    let registry = Registry::open(daemon.state_root.join(immortality_core::registry::FILE_NAME)).unwrap();
    let found = registry
        .by_path(&project)
        .expect("the repository must be registered after init");
    assert!(found.path.ends_with("fresh"));
    assert!(!daemon.token_path().as_os_str().is_empty());
}

// ── plan --goal, end to end ──────────────────────────────────────────────────

/// A planner answer the compiler accepts.
fn plan_answer() -> String {
    serde_json::json!({
        "plan": "# Plan\n\nAdd the parser.",
        "acceptance": "`cargo test` passes and `parser::parse` exists.",
        "subtasks": [{
            "name": "parse",
            "title": "write the parser",
            "goal": "add `parser::parse`",
            "dod": {
                "statement": "the parser handles the cases",
                "checks": [{ "id": "test", "command": "cargo test" }],
                "review": false,
            },
        }],
    })
    .to_string()
}

#[test]
fn plan_goal_writes_the_three_files_and_prints_them() {
    // The whole chain over a real socket: the CLI sends its cwd, the daemon resolves the project,
    // asks the model, validates against its own depth limit, and writes. Every step here has a
    // unit test; this one proves the two ends agree about the request and the answer.
    let daemon = Daemon::new().answering(&plan_answer());
    daemon.serve();

    daemon
        .cli()
        .current_dir(&daemon.repo)
        .args(["plan", "--goal", "add a parser"])
        .assert()
        .success()
        .stdout(predicate::str::contains("add a parser"))
        .stdout(predicate::str::contains("TASKS.json"))
        .stdout(predicate::str::contains("write the parser"));

    for name in ["PLAN.md", "ACCEPTANCE.md", "TASKS.json"] {
        assert!(
            daemon.repo.join(".agent").join(name).exists(),
            "{name} was not written"
        );
    }
}

#[test]
fn plan_goal_json_is_machine_readable() {
    let daemon = Daemon::new().answering(&plan_answer());
    daemon.serve();

    let output = daemon
        .cli()
        .current_dir(&daemon.repo)
        .args(["plan", "--goal", "add a parser", "--json"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["count"], 1);
    assert_eq!(value["created"], true);
    assert_eq!(value["subtasks"][0]["state"], "pending");
    assert_eq!(value["max_subtask_depth"], 2);
}

#[test]
fn a_second_plan_goal_refuses_to_overwrite_without_force() {
    // The roadmap's contract, proven over the wire: a human may rewrite these files before the
    // run, and a second `plan` must not quietly discard that.
    let daemon = Daemon::new().answering(&plan_answer());
    daemon.serve();
    daemon
        .cli()
        .current_dir(&daemon.repo)
        .args(["plan", "--goal", "add a parser"])
        .assert()
        .success();

    let edited = "# my own plan\n";
    std::fs::write(daemon.repo.join(".agent").join("PLAN.md"), edited).unwrap();

    daemon
        .cli()
        .current_dir(&daemon.repo)
        .args(["plan", "--goal", "add a parser"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--force"))
        .stderr(predicate::str::contains("PLAN.md"));

    assert_eq!(
        std::fs::read_to_string(daemon.repo.join(".agent").join("PLAN.md")).unwrap(),
        edited,
        "the human's edit must survive a refused plan"
    );

    daemon
        .cli()
        .current_dir(&daemon.repo)
        .args(["plan", "--goal", "add a parser", "--force"])
        .assert()
        .success();
    let replaced = std::fs::read_to_string(daemon.repo.join(".agent").join("PLAN.md")).unwrap();
    assert!(!replaced.contains("my own plan"), "{replaced}");
}
