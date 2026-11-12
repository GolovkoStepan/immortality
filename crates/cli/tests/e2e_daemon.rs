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
        }
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
    // The rendered line is `<seq right-aligned>  <kind>`, so the fields are split rather than
    // searched: `contains('0')` would also be satisfied by a seq of 10 or a payload digit.
    let mut fields = line.split_whitespace();
    assert_eq!(fields.next(), Some("0"), "first line was {line:?}");
    assert_eq!(fields.next(), Some("task_state"), "first line was {line:?}");

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
