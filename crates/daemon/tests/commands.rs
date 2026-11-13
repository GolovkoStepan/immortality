//! `plan`, `log`, `gc` and `merge`, driven against a real repository and journal.
//!
//! # Why these do not go through the socket
//!
//! `server_socket.rs` already proves that a request reaches a handler over a real connection.
//! What these commands can get wrong is *what they read and what they touch* — the journal's
//! filter, the plan's depth limit, the live set `gc` must protect, the git guards `merge` must
//! refuse on. All of that is decided before a byte goes on the wire, so calling the handlers
//! directly keeps the failure message about the actual rule.
//!
//! # Why `merge` runs real git
//!
//! The guards it needs to get right (a dirty tree, a missing branch, a real conflict) are git's
//! answers, not ours. A fake `GitRepo` would prove that the handler calls something.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use immortality_core::domain::{Dod, Subtask, SubtaskState, TaskId, TaskState};
use immortality_core::journal::{Journal, Kind};
use immortality_core::registry::Registry;
use immortality_daemon::commands;
use immortality_daemon::rpc::ErrorCode;
use immortality_daemon::server::ServerState;

fn task() -> TaskId {
    TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV")
}

/// A git repository with one commit, registered as a project.
struct Fixture {
    dir: tempfile::TempDir,
    repo: PathBuf,
    state_root: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let repo = dir.path().join("repo");
        let state_root = dir.path().join("state-root");
        std::fs::create_dir_all(repo.join(".agent")).expect("agent dir");
        std::fs::create_dir_all(&state_root).expect("state root");
        std::fs::write(repo.join(".agent").join("ACCEPTANCE.md"), "# acceptance\n").expect("acceptance");
        // `ServerState::discover` refuses to start without a token, on purpose; a fixture that
        // skipped it would be testing a daemon that accepts every client.
        std::fs::write(state_root.join("daemon.token"), "fixture-token").expect("token");
        std::fs::write(repo.join(".agent").join("TASKS.json"), plan_json()).expect("tasks");
        git(&repo, &["init", "-q", "-b", "main"]);
        git(&repo, &["config", "user.email", "t@example.com"]);
        git(&repo, &["config", "user.name", "t"]);
        std::fs::write(repo.join(".gitignore"), ".agent/\n").expect("gitignore");
        std::fs::write(repo.join("README.md"), "# fixture\n").expect("readme");
        git(&repo, &["add", "-A"]);
        git(&repo, &["commit", "-q", "-m", "initial"]);

        let mut registry =
            Registry::open(state_root.join(immortality_core::registry::FILE_NAME)).expect("registry");
        registry.register(&repo, None).expect("register");
        let project = registry.projects()[0].clone();
        std::fs::create_dir_all(project.state_dir(&state_root).join(task().as_str())).expect("task dir");
        Self {
            dir,
            repo,
            state_root,
        }
    }

    /// The one project this fixture registered.
    fn sole_project(&self) -> immortality_core::registry::Project {
        Registry::open(self.state_root.join(immortality_core::registry::FILE_NAME))
            .expect("registry")
            .projects()[0]
            .clone()
    }

    fn state(&self) -> ServerState {
        ServerState::discover(&self.state_root).expect("state")
    }

    fn journal(&self) -> Journal {
        Journal::open(self.repo.join(".agent"), task()).expect("journal")
    }

    fn record_merged_ready(&self) {
        let mut journal = self.journal();
        journal
            .append_event(
                immortality_core::time::now(),
                Kind::TaskState,
                serde_json::json!({"state": TaskState::ReadyForReview.as_str()}),
            )
            .expect("ready");
        journal.close().expect("close");
    }

    /// Commit a file on the task branch, as the agent would.
    fn agent_commit(&self, name: &str) {
        git(
            &self.repo,
            &["checkout", "-q", "-b", &format!("agent/{}", task().as_str())],
        );
        std::fs::write(self.repo.join(name), "work\n").expect("write");
        git(&self.repo, &["add", "-A"]);
        git(&self.repo, &["commit", "-q", "-m", "the task's work"]);
        git(&self.repo, &["checkout", "-q", "main"]);
    }

    fn temp(&self) -> &Path {
        self.dir.path()
    }
}

fn git(repo: &Path, args: &[&str]) {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("LC_ALL", "C")
        .output()
        .expect("run git");
    assert!(
        output.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn plan_json() -> String {
    let subtask = Subtask::new(
        1,
        "write the parser",
        "parse the input",
        Dod::reviewed("the parser works"),
        immortality_core::time::now(),
    );
    serde_json::to_string_pretty(&serde_json::json!({
        "v": 1,
        "subtasks": [subtask],
    }))
    .expect("json")
}

const fn params(value: serde_json::Value) -> serde_json::Value {
    value
}

#[test]
fn plan_reports_the_tree_with_the_daemons_depth_limit() {
    let fixture = Fixture::new();
    let state = fixture.state();
    let result = commands::plan(&state, &params(serde_json::json!({"project": fixture.repo}))).expect("plan");
    assert!(result.get("load_error").is_none(), "{result}");
    assert_eq!(result["count"], 1);
    assert_eq!(result["subtasks"][0]["title"], "write the parser");
    // The limit is carried, not defaulted: `Plan::load` validates against it, so a report that
    // omitted it could not explain a refusal.
    assert_eq!(
        result["max_subtask_depth"].as_u64(),
        Some(u64::from(state.config.loop_.max_subtask_depth))
    );
}

#[test]
fn a_plan_that_cannot_be_read_is_a_reported_answer_not_a_transport_failure() {
    // The distinction the CLI depends on: a human ran `plan` to see what is wrong with the file.
    // Reporting it as `BadParams` would turn "your TASKS.json is invalid" into "you typed the
    // command wrong".
    let fixture = Fixture::new();
    std::fs::write(fixture.repo.join(".agent").join("TASKS.json"), "{ not json }").expect("write");
    let result = commands::plan(
        &fixture.state(),
        &params(serde_json::json!({"project": fixture.repo})),
    )
    .expect("the request succeeds");
    let error = result["load_error"].as_str().expect("a load error");
    assert!(error.contains("TASKS.json"), "{error}");
}

#[test]
fn a_request_with_no_project_is_refused_rather_than_read_from_the_daemons_cwd() {
    // The daemon's working directory is `/` under launchd and arbitrary under a hand-started
    // `serve`. Reading it would answer about a repository the human never named — and the failure
    // would look like "your project is not registered" while they are standing in it.
    let fixture = Fixture::new();
    let error = commands::plan(&fixture.state(), &params(serde_json::json!({})))
        .expect_err("must refuse rather than guess");
    assert_eq!(error.code, ErrorCode::BadParams);
    assert!(error.message.contains("working directory"), "{}", error.message);
}

#[test]
fn plan_for_an_unknown_directory_names_it() {
    let fixture = Fixture::new();
    let error = commands::plan(
        &fixture.state(),
        &params(serde_json::json!({"project": fixture.temp().join("elsewhere")})),
    )
    .expect_err("must refuse");
    assert_eq!(error.code, ErrorCode::NotFound);
    assert!(error.message.contains("elsewhere"), "{}", error.message);
}

#[test]
fn log_shows_only_this_tasks_records() {
    // One journal per repository holds every task of that repository. An unfiltered transcript
    // would show another task's work as if it were this one's.
    let fixture = Fixture::new();
    {
        let mut mine = fixture.journal();
        mine.append_event(
            immortality_core::time::now(),
            Kind::TaskState,
            serde_json::json!({"state": "running"}),
        )
        .expect("mine");
        mine.close().expect("close");
    }
    {
        let other = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FZZ");
        let mut theirs = Journal::open(fixture.repo.join(".agent"), other).expect("journal");
        theirs
            .append_event(
                immortality_core::time::now(),
                Kind::TaskState,
                serde_json::json!({"state": "merged"}),
            )
            .expect("theirs");
        theirs.close().expect("close");
    }

    let result = commands::log(
        &fixture.state(),
        &params(serde_json::json!({"task": task().as_str()})),
    )
    .expect("log");
    let records = result["records"].as_array().expect("records");
    assert_eq!(records.len(), 1, "{result}");
    assert_eq!(records[0]["task"], task().as_str());
    assert_eq!(records[0]["payload"]["state"], "running");
}

#[test]
fn log_for_an_unregistered_task_is_a_refusal() {
    let fixture = Fixture::new();
    let error = commands::log(
        &fixture.state(),
        &params(serde_json::json!({"task": "tsk_01ARZ3NDEKTSV4RRFFQ69G5FZZ"})),
    )
    .expect_err("must refuse");
    assert_eq!(error.code, ErrorCode::NotFound);
}

#[test]
fn gc_protects_the_artifacts_of_an_unfinished_task() {
    // The one fact `gc` cannot get from the filesystem. Deleting the body of a tool result the
    // scheduler is about to replay turns a resumable task into a corrupt one.
    let fixture = Fixture::new();
    let store =
        immortality_core::artifacts::ArtifactStore::open(fixture.state().artifacts_dir()).expect("store");
    let digest = store.put(b"the body of a large tool result").expect("put");
    let name = immortality_core::gc::artifact_name(&digest);

    {
        use immortality_core::journal::{JournalEvent, RecordFields};

        let mut journal = fixture.journal();
        let fields = RecordFields::new(0, immortality_core::time::now(), task(), Kind::ToolCallCompleted);
        journal
            .append(
                fields,
                JournalEvent::spilled(serde_json::json!({"call": "c"}), digest),
            )
            .expect("record");
        journal.close().expect("close");
    }
    // `name` is what `gc` prints; the live set itself is keyed by the digest.
    assert_eq!(name.len(), "sha256:".len() + 64, "a digest is `sha256:<hex>`");
    assert!(name.starts_with("sha256:"), "{name}");

    let result = commands::collect_garbage(&fixture.state(), &params(serde_json::json!({"dry_run": true})))
        .expect("gc");
    assert_eq!(result["protected"], 1, "{result}");
    assert_eq!(result["unfinished"][0], task().as_str());
}

#[test]
fn gc_refuses_to_run_when_a_journal_is_unreadable() {
    // "Unreadable" is not "empty": treating it as empty would put that task's artifacts in the
    // deletable set, which destroys the record of the one task a human is debugging.
    let fixture = Fixture::new();
    std::fs::write(
        fixture.repo.join(".agent").join("JOURNAL.jsonl"),
        "not a journal\n",
    )
    .expect("write");
    let error =
        commands::collect_garbage(&fixture.state(), &params(serde_json::json!({}))).expect_err("must refuse");
    assert_eq!(error.code, ErrorCode::Internal);
    assert!(
        error.message.contains("nothing may be collected"),
        "{}",
        error.message
    );
}

#[test]
fn a_task_known_only_from_its_journal_is_still_owned_by_its_project() {
    // `owner_of` has two traces to go on: the task's state directory and the project's journal.
    // A task that has a journal but no state directory (the directory is machine-local and created
    // lazily) is owned all the same, and every command must agree on that. When they disagreed,
    // `merge` answered "no project owns task" for a task `status` was listing.
    let fixture = Fixture::new();
    fixture.record_merged_ready();
    std::fs::remove_dir_all(
        fixture
            .sole_project()
            .state_dir(&fixture.state_root)
            .join(task().as_str()),
    )
    .expect("remove the state directory");

    // `status`'s own trace: the task is listed.
    let registry =
        Registry::open(fixture.state_root.join(immortality_core::registry::FILE_NAME)).expect("registry");
    let listed = registry.tasks(&fixture.state_root).expect("tasks");
    assert!(
        listed.iter().any(|(_, known, _)| known == &task()),
        "the journal alone must make the task visible"
    );

    // And `log`, which resolves the owner before reading, finds it.
    let result = commands::log(
        &fixture.state(),
        &params(serde_json::json!({ "task": task().as_str() })),
    )
    .expect("log must find the owner through the journal");
    assert_eq!(result["count"], 1, "{result}");
}

#[test]
fn merge_refuses_a_task_that_is_not_ready_for_review() {
    // D-011 gives the merge to a human because the gates are not the whole judgement. Merging a
    // running task would publish work the gates have never seen.
    let fixture = Fixture::new();
    fixture.agent_commit("work.txt");
    let error = commands::merge(
        &fixture.state(),
        &params(serde_json::json!({"task": task().as_str()})),
    )
    .expect_err("must refuse");
    assert_eq!(error.code, ErrorCode::BadParams);
    assert!(error.message.contains("ready_for_review"), "{}", error.message);
}

#[test]
fn merge_refuses_a_dirty_main_checkout() {
    // Merging into a dirty tree mixes the agent's work with whatever a human was editing, and the
    // resulting commit belongs to neither of them.
    let fixture = Fixture::new();
    fixture.record_merged_ready();
    fixture.agent_commit("work.txt");
    std::fs::write(fixture.repo.join("uncommitted.txt"), "human work\n").expect("write");
    let error = commands::merge(
        &fixture.state(),
        &params(serde_json::json!({"task": task().as_str()})),
    )
    .expect_err("must refuse");
    assert_eq!(error.code, ErrorCode::BadParams);
    assert!(error.message.contains("uncommitted"), "{}", error.message);
}

#[test]
fn merge_refuses_a_branch_that_does_not_exist() {
    let fixture = Fixture::new();
    fixture.record_merged_ready();
    let error = commands::merge(
        &fixture.state(),
        &params(serde_json::json!({"task": task().as_str()})),
    )
    .expect_err("must refuse");
    assert_eq!(error.code, ErrorCode::NotFound);
    assert!(
        error.message.contains(&format!("agent/{}", task().as_str())),
        "{}",
        error.message
    );
}

#[test]
fn merge_lands_the_branch_and_journals_merged() {
    let fixture = Fixture::new();
    fixture.record_merged_ready();
    fixture.agent_commit("work.txt");

    let result = commands::merge(
        &fixture.state(),
        &params(serde_json::json!({"task": task().as_str()})),
    )
    .expect("merge");
    assert_eq!(result["state"], "merged");
    // The file is on main, and a merge commit exists: `--no-ff` keeps the task's history visible
    // instead of fast-forwarding it away.
    assert!(
        fixture.repo.join("work.txt").exists(),
        "the branch must be merged into main"
    );
    let log = Command::new("git")
        .arg("-C")
        .arg(&fixture.repo)
        .args(["log", "--oneline"])
        .output()
        .expect("git log");
    let text = String::from_utf8_lossy(&log.stdout);
    assert!(text.contains(&format!("merge {}", task().as_str())), "{text}");
}

#[test]
fn a_conflicted_merge_is_aborted_and_says_so() {
    // The state nobody asked for is a repository with a half-merged index: every later command
    // then lies about what is on disk. Aborting restores the pre-merge tree.
    let fixture = Fixture::new();
    fixture.record_merged_ready();
    fixture.agent_commit("shared.txt");
    std::fs::write(fixture.repo.join("shared.txt"), "a different line\n").expect("write");
    git(&fixture.repo, &["add", "-A"]);
    git(&fixture.repo, &["commit", "-q", "-m", "main's own change"]);

    let error = commands::merge(
        &fixture.state(),
        &params(serde_json::json!({"task": task().as_str()})),
    )
    .expect_err("must refuse");
    assert_eq!(error.code, ErrorCode::Internal);
    assert!(error.message.contains("aborted"), "{}", error.message);
    assert!(
        error.message.contains("git merge"),
        "the message must name the manual fix: {}",
        error.message
    );

    // Aborted, not left in progress: no merge state, and the tree is clean.
    let status = Command::new("git")
        .arg("-C")
        .arg(&fixture.repo)
        .args(["status", "--porcelain"])
        .output()
        .expect("git status");
    assert!(
        String::from_utf8_lossy(&status.stdout).trim().is_empty(),
        "the index must be clean after an aborted merge"
    );
    assert!(!fixture.repo.join(".git").join("MERGE_HEAD").exists());

    // And the task is still `ready_for_review`: the journal is written only after git succeeds.
    // Recording `merged` first would leave a task that looks merged and is not, which no later
    // command could detect.
    let log = commands::log(
        &fixture.state(),
        &params(serde_json::json!({"task": task().as_str()})),
    )
    .expect("log");
    let merged = log["records"]
        .as_array()
        .expect("records")
        .iter()
        .any(|record| record["payload"]["state"] == "merged");
    assert!(!merged, "a failed merge must not be journalled as merged: {log}");
}

#[test]
fn merged_is_the_last_state_the_journal_records() {
    // `merge` writes the journal *after* git succeeds: a record written first would announce work
    // that is not on disk, and a crash in between would leave a task that looks merged and is not.
    let fixture = Fixture::new();
    fixture.record_merged_ready();
    fixture.agent_commit("work.txt");
    commands::merge(
        &fixture.state(),
        &params(serde_json::json!({"task": task().as_str()})),
    )
    .expect("merge");

    let result = commands::log(
        &fixture.state(),
        &params(serde_json::json!({"task": task().as_str()})),
    )
    .expect("log");
    let records = result["records"].as_array().expect("records");
    let last = records.last().expect("a last record");
    assert_eq!(last["kind"], "task_state");
    assert_eq!(last["payload"]["state"], "merged");
}

/// The state a subtask is in must not be confused with the task's, and `plan` reports the tree.
#[test]
fn plan_shows_the_state_each_subtask_is_in() {
    let fixture = Fixture::new();
    let mut body: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture.repo.join(".agent").join("TASKS.json")).unwrap(),
    )
    .expect("plan");
    body["subtasks"][0]["state"] = serde_json::json!(SubtaskState::Done.as_str());
    std::fs::write(
        fixture.repo.join(".agent").join("TASKS.json"),
        serde_json::to_string(&body).expect("json"),
    )
    .expect("write");

    let result = commands::plan(
        &fixture.state(),
        &params(serde_json::json!({"project": fixture.repo})),
    )
    .expect("plan");
    assert_eq!(result["subtasks"][0]["state"], "done");
}

// ── metrics ────────────────────────────────────────────────────────────────────────────

/// Append to this fixture's journal the records a run would leave.
fn record_run(fixture: &Fixture, records: &[(Kind, serde_json::Value)]) {
    let mut journal = fixture.journal();
    for (kind, payload) in records {
        journal
            .append_event(immortality_core::time::now(), *kind, payload.clone())
            .expect("append");
    }
    journal.close().expect("close");
}

#[test]
fn metrics_count_what_the_journal_recorded_for_the_task() {
    // The document claims "three subtasks closed with green gates". This is the assertion that
    // the number the daemon prints is the number in the file — two closed, one blocked.
    let fixture = Fixture::new();
    record_run(
        &fixture,
        &[
            (
                Kind::SubtaskState,
                serde_json::json!({"subtask": "sub_A", "state": "running"}),
            ),
            (
                Kind::SubtaskState,
                serde_json::json!({"subtask": "sub_A", "state": "done"}),
            ),
            (
                Kind::DetectorFired,
                serde_json::json!({"detector": "d1", "level": "l1", "metric": 1}),
            ),
            (
                Kind::SubtaskState,
                serde_json::json!({"subtask": "sub_B", "state": "blocked"}),
            ),
            (Kind::AttemptRestarted, serde_json::json!({"next": 2})),
        ],
    );

    let result = commands::metrics(
        &fixture.state(),
        &params(serde_json::json!({"task": task().as_str()})),
    )
    .expect("metrics");
    assert_eq!(result["count"], 1, "{result}");
    let report = &result["tasks"][0];
    assert_eq!(report["task"], task().as_str());
    assert_eq!(report["closed_green"], 1);
    assert_eq!(report["closed_with_escalation"], 0);
    assert_eq!(report["restarts"], 1);
    assert_eq!(report["detectors"]["d1/l1"], 1);
    assert_eq!(report["corrupt"], false);
    assert_eq!(report["subtasks"].as_array().expect("subtasks").len(), 2);
}

#[test]
fn metrics_for_an_unregistered_task_names_it_instead_of_returning_nothing() {
    // An empty list of numbers is indistinguishable from "this task did fine", so an unknown id
    // has to be a refusal.
    let fixture = Fixture::new();
    let error = commands::metrics(
        &fixture.state(),
        &params(serde_json::json!({"task": "tsk_01ARZ3NDEKTSV4RRFFQ69G5FZZ"})),
    )
    .expect_err("must refuse");
    assert_eq!(error.code, ErrorCode::NotFound);
    assert!(
        error.message.contains("01ARZ3NDEKTSV4RRFFQ69G5FZZ"),
        "{}",
        error.message
    );
}

#[test]
fn metrics_without_a_task_report_every_task_the_daemon_knows() {
    // The multi-task shape is what `docs/metrics-v0.1.md` reads when it says "the three dogfood
    // runs". One task per repository, so the fixture needs a second journal in the same repo.
    let fixture = Fixture::new();
    record_run(
        &fixture,
        &[(
            Kind::SubtaskState,
            serde_json::json!({"subtask": "sub_A", "state": "done"}),
        )],
    );
    let other = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FZZ");
    {
        let mut theirs = Journal::open(fixture.repo.join(".agent"), other).expect("journal");
        theirs
            .append_event(
                immortality_core::time::now(),
                Kind::SubtaskState,
                serde_json::json!({"subtask": "sub_Z", "state": "blocked"}),
            )
            .expect("append");
        theirs.close().expect("close");
    }

    let result = commands::metrics(&fixture.state(), &params(serde_json::json!({}))).expect("metrics");
    assert_eq!(result["count"], 2, "{result}");
}
