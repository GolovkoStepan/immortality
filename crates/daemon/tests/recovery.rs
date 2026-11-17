//! Recovery execution against a real git repository, a real journal and a real tool registry.
//!
//! # Why integration and not unit tests
//!
//! The two claims M5.2 has to make are "the probe sees what the tool wrote" and "the executor
//! changes the world the plan named". Both are claims about the filesystem: a `write_file` probe
//! is only right if it reads the *same path* the tool's gate resolved, and a rollback is only
//! right if git really moved. Mocking either would test the mock.
//!
//! # What is arranged by hand
//!
//! The journal is written directly rather than by a crashed runner. A crash point is an *absence*
//! of records, and producing one by actually killing a process would make each test a timing
//! experiment. Writing the prefix a crash leaves is exact: `intent` without `started`, `started`
//! without `completed`, `rollback` without `attempt_restarted`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use immortality_core::artifacts::ArtifactStore;
use immortality_core::cache::Cache;
use immortality_core::checkpoint::GitRepo;
use immortality_core::config::Policy;
use immortality_core::domain::TaskId;
use immortality_core::escalation::{Answer, EscalationStore};
use immortality_core::ids::SubtaskId;
use immortality_core::journal::{Journal, JournalReader, Kind, ReadOutcome, Scope};
use immortality_core::protocol::{CallId, Idempotency, Intent, Probe};
use immortality_core::recover::{self, Prober, RecoveryInput, Verdict, WorkspaceFacts};
use immortality_core::time::{Clock as _, FixedClock};
use immortality_daemon::recovery::{self as exec, Executed, RecoveryDeps, ToolProbe};
use immortality_tools::{PolicyGate, Registry, WriteFile};
use serde_json::json;

const TASK: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAV";
const SUBTASK: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAA";

/// A repository, a state directory and a journal, laid out the way the daemon lays them out.
struct Fixture {
    /// Kept alive so the repository and state directories outlive every test body.
    dir: tempfile::TempDir,
    /// The task worktree: where tools write, and what a rollback resets.
    worktree: PathBuf,
    /// The project's `.agent/`, where the journal lives.
    ///
    /// Deliberately *outside* the worktree, exactly as in production: a rollback runs `reset
    /// --hard` in the worktree, and a journal stored there would be deleted by the very recovery
    /// that needs to read it. Keeping the fixture honest about that layout is what caught this.
    journal_root: PathBuf,
    /// Machine-local state for this task.
    task_dir: PathBuf,
    task: TaskId,
    clock: Arc<FixedClock>,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let worktree = dir.path().join("worktree");
        let journal_root = dir.path().join("project").join(".agent");
        let state_root = dir.path().join("state");
        std::fs::create_dir_all(&worktree).expect("worktree");
        std::fs::create_dir_all(&journal_root).expect("journal root");
        std::fs::create_dir_all(&state_root).expect("state root");
        for args in [
            vec!["init", "-q", "-b", "main"],
            vec!["config", "user.email", "test@example.com"],
            vec!["config", "user.name", "test"],
        ] {
            let status = Command::new("git")
                .arg("-C")
                .arg(&worktree)
                .args(&args)
                .status()
                .expect("git");
            assert!(status.success(), "git {args:?} failed");
        }
        std::fs::write(worktree.join("README.md"), "start\n").expect("readme");
        let status = Command::new("git")
            .arg("-C")
            .arg(&worktree)
            .args(["add", "--all"])
            .status()
            .expect("git add");
        assert!(status.success());
        let status = Command::new("git")
            .arg("-C")
            .arg(&worktree)
            .args(["commit", "-q", "-m", "initial"])
            .status()
            .expect("git commit");
        assert!(status.success());

        let task = TaskId::from_suffix(TASK);
        let task_dir = state_root.join("state").join("project").join(task.as_str());
        std::fs::create_dir_all(&task_dir).expect("task dir");
        // The task branch the worktree is supposed to be on.
        let branch = exec::expected_branch(&task);
        let status = Command::new("git")
            .arg("-C")
            .arg(&worktree)
            .args(["checkout", "-q", "-b", &branch])
            .status()
            .expect("checkout");
        assert!(status.success());

        Self {
            dir,
            worktree,
            journal_root,
            task_dir,
            task,
            clock: Arc::new(FixedClock::fixture()),
        }
    }

    fn journal_dir(&self) -> PathBuf {
        self.journal_root.clone()
    }

    fn journal(&self) -> Journal {
        let mut journal = Journal::open(self.journal_dir(), self.task.clone()).expect("journal");
        journal.set_clock(Arc::clone(&self.clock) as Arc<dyn immortality_core::time::Clock>);
        journal
    }

    fn records(&self) -> ReadOutcome {
        JournalReader::open(self.journal_dir(), &self.task)
            .expect("reader")
            .read()
            .expect("read")
    }

    fn gate(&self) -> Arc<PolicyGate> {
        Arc::new(
            PolicyGate::with_defaults(
                &self.worktree,
                Policy {
                    workspace_root: self.dir.path().to_path_buf(),
                    ..Policy::default()
                },
            )
            .expect("gate"),
        )
    }

    fn deps(&self) -> RecoveryDeps {
        RecoveryDeps {
            task: self.task.clone(),
            task_dir: self.task_dir.clone(),
            // The fixture's only git repository is the worktree itself.
            repository: self.worktree.clone(),
            worktree: self.worktree.clone(),
            branch: exec::expected_branch(&self.task),
            gate: self.gate(),
            registry: Arc::new(Registry::new(vec![Arc::new(WriteFile)]).expect("registry")),
            artifacts: ArtifactStore::open(self.dir.path().join("artifacts")).expect("artifacts"),
            clock: Arc::clone(&self.clock) as Arc<dyn immortality_core::time::Clock>,
        }
    }

    fn probe(&self) -> ToolProbe<'_> {
        // Leaked on purpose: the probe borrows a gate and a repo that must outlive the call, and
        // a test that threaded `Arc`s through every assertion would obscure what is being tested.
        let gate: &'static PolicyGate = Box::leak(Box::new(
            PolicyGate::with_defaults(
                &self.worktree,
                Policy {
                    workspace_root: self.dir.path().to_path_buf(),
                    ..Policy::default()
                },
            )
            .expect("gate"),
        ));
        let repo: &'static GitRepo = Box::leak(Box::new(GitRepo::new(&self.worktree)));
        ToolProbe {
            gate,
            repo,
            // `DelegationStore::for_task` appends the task id, so the probe takes the directory
            // *containing* the task's state directory.
            delegation_root: self.task_dir.parent().expect("the task dir has a parent"),
            task: &self.task,
            task_dir: &self.task_dir,
        }
    }

    fn input<'a>(&'a self, records: &'a ReadOutcome, workspace: WorkspaceFacts) -> RecoveryInput<'a> {
        RecoveryInput {
            journal: records,
            workspace,
            // Read the cache the way the daemon will: a divergence is only visible if the plan
            // sees what is on disk.
            persisted: Cache::new(&self.task_dir).read(),
            last_checkpoint: exec::last_checkpoint(&records.records),
            // `agent/` + the task id, which already carries its `tsk_` prefix.
            expected_branch: "agent/tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV",
        }
    }

    /// Journal an intent for a call, and optionally its `started` record.
    fn open_call(&self, call: &str, tool: &str, arguments: &serde_json::Value, started: bool) {
        let mut journal = self.journal();
        let intent = Intent::new(
            CallId::from_model(call),
            tool,
            arguments,
            // The class comes from the tool that will execute the retry, not from a copy of its
            // table. The copy had already drifted: `escalate` and `kill_process` declare
            // `ByContent` and a hand-written match put them in its `Idempotent` default, so tests
            // exercised an arm production never reaches.
            self.class_of(tool, arguments),
            format!("call {call}"),
        );
        let scope = Scope::attempt(SubtaskId::from_suffix(SUBTASK), 1);
        {
            // One log for both records: the ordering guard lives in the log's in-memory map, and a
            // second log would refuse a `started` whose `intent` it never saw.
            let mut log = immortality_core::protocol::ToolCallLog::scoped(&mut journal, scope);
            log.intent(&intent, arguments).expect("intent");
            if started {
                log.started(&intent.call, serde_json::Value::Null)
                    .expect("started");
            }
        }
        journal.close().expect("close");
    }

    /// The idempotency class the production registry declares for `tool`.
    ///
    /// Taken from the tool, not from a copy of its table. The copy this replaces had already
    /// drifted from the code — `escalate` and `kill_process` were in its `Idempotent` default arm
    /// while both declare `ByContent` — so every test built on it exercised a path production
    /// never takes, which is exactly what its own comment warned about.
    fn class_of(&self, tool: &str, arguments: &serde_json::Value) -> Idempotency {
        let deps = immortality_daemon::toolset::ToolsetDeps::for_task(
            &immortality_core::config::Config::default(),
            self.dir.path(),
            &self.task_dir,
            &self.worktree,
            self.dir.path(),
            self.task.clone(),
        )
        .expect("toolset deps");
        let registry = immortality_daemon::toolset::build(&deps).expect("registry");
        let found = registry
            .get(tool)
            .unwrap_or_else(|| panic!("`{tool}` is not a production tool, so a test cannot name its class"));
        found.idempotency(&immortality_tools::Args::new(arguments.clone()))
    }

    fn run_plan(&self) -> (ReadOutcome, recover::RecoveryReport, Executed) {
        // Read first, open second. `Journal::open` truncates a torn tail and journals the repair,
        // so a plan built after that open cannot see the tail and would report a clean start for a
        // crash recovery. This is also the order the daemon must use.
        let records = self.records();
        let workspace = clean(workspace_facts(&self.worktree));
        let report = recover::plan(&self.input(&records, workspace), &self.probe());
        let mut journal = self.journal();
        assert!(
            recover::all_calls_addressed(&records.records, &report),
            "the plan must address every open call"
        );
        let deps = self.deps();
        let mut executor = exec::Executor::new(&deps, Some(&mut journal), records.clone(), &report);
        let executed = executor.run(&report.steps).expect("execution");
        journal.sync().expect("sync");
        (records, report, executed)
    }
}

/// Commit everything in `worktree`, so a test that wants a clean tree gets one.
fn commit_all(worktree: &Path) {
    for args in [
        vec!["add", "--all"],
        vec![
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=test",
            "commit",
            "-q",
            "-m",
            "work",
        ],
    ] {
        let status = Command::new("git")
            .arg("-C")
            .arg(worktree)
            .args(&args)
            .status()
            .expect("git");
        assert!(status.success(), "git {args:?} failed");
    }
}

/// `WorkspaceFacts` for a worktree, in the shape recovery expects.
fn workspace_facts(worktree: &Path) -> WorkspaceFacts {
    let repo = GitRepo::new(worktree);
    let branch = Command::new("git")
        .arg("-C")
        .arg(worktree)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|text| text != "HEAD");
    WorkspaceFacts {
        exists: true,
        is_repository: true,
        branch,
        head: repo.head().ok().flatten(),
        dirty: repo.is_dirty().unwrap_or(false),
    }
}

/// Force `dirty` false where a test wants a clean start.
///
/// The plan genuinely behaves differently for a dirty tree, and a test that meant to exercise
/// one behaviour must not silently exercise the other because the fixture touched a file.
const fn clean(mut facts: WorkspaceFacts) -> WorkspaceFacts {
    facts.dirty = false;
    facts
}

// ── the probe ──────────────────────────────────────────────────────────────────────────────

#[test]
fn a_write_that_landed_is_reported_applied_and_is_not_rerun() {
    // The exact bug the D-081 arguments seam was added for: without the content, this probe can
    // only answer `Unknown`, and `Unknown` rolls the tree back over a file that is complete.
    let fixture = Fixture::new();
    std::fs::write(fixture.worktree.join("src.rs"), "pub fn x() {}\n").expect("write");
    let call = "call_write";
    let arguments = json!({"path": "src.rs", "content": "pub fn x() {}\n"});
    fixture.open_call(call, "write_file", &arguments, true);

    let (_records, report, executed) = fixture.run_plan();
    assert_eq!(report.calls[0].effect, Some(Probe::Applied));
    assert!(
        !executed.steps.iter().any(|step| step.contains("re-ran")),
        "an applied write must not be re-run: {executed:?}"
    );

    let records = fixture.records();
    assert!(
        records
            .records
            .iter()
            .any(|record| record.fields.kind == Kind::EffectVerified),
        "the probe verdict must be journalled"
    );
    assert!(
        records
            .records
            .iter()
            .any(|record| record.fields.kind == Kind::ToolCallCompleted),
        "the call must be closed"
    );
}

#[test]
fn a_write_that_did_not_land_is_rerun() {
    let fixture = Fixture::new();
    let call = "call_write";
    let arguments = json!({"path": "src.rs", "content": "pub fn x() {}\n"});
    fixture.open_call(call, "write_file", &arguments, true);

    let (_records, report, executed) = fixture.run_plan();
    assert_eq!(report.calls[0].effect, Some(Probe::NotApplied));
    assert!(
        executed
            .steps
            .iter()
            .any(|step| step.contains("re-ran write_file")),
        "a write that never landed must be re-issued: {executed:?}"
    );
    assert_eq!(
        std::fs::read_to_string(fixture.worktree.join("src.rs")).expect("the retry wrote it"),
        "pub fn x() {}\n"
    );
}

#[test]
fn an_edit_that_landed_is_applied() {
    let fixture = Fixture::new();
    std::fs::write(fixture.worktree.join("src.rs"), "let x = 1;\n").expect("write");
    fixture.open_call(
        "call_edit",
        "edit_file",
        &json!({"path": "src.rs", "old": "let x = 0;", "new": "let x = 1;"}),
        true,
    );

    let (_records, report, _executed) = fixture.run_plan();
    assert_eq!(report.calls[0].effect, Some(Probe::Applied));
}

#[test]
fn an_edit_whose_text_is_nowhere_is_unknown_rather_than_a_guess() {
    // A half-applied edit is indistinguishable from an unapplied one when neither the old nor the
    // new text is present. Guessing `NotApplied` would run the edit again; guessing `Applied`
    // would leave a broken file.
    let fixture = Fixture::new();
    std::fs::write(fixture.worktree.join("src.rs"), "something else entirely\n").expect("write");
    fixture.open_call(
        "call_edit",
        "edit_file",
        &json!({"path": "src.rs", "old": "let x = 0;", "new": "let x = 1;"}),
        true,
    );

    let (_records, report, _executed) = fixture.run_plan();
    assert_eq!(report.calls[0].effect, Some(Probe::Unknown));
}

#[test]
fn a_git_commit_is_found_by_its_trailer() {
    let fixture = Fixture::new();
    let call = "call_commit";
    // The commit the tool would have made: the trailer names the call id.
    std::fs::write(fixture.worktree.join("src.rs"), "x\n").expect("write");
    let status = Command::new("git")
        .arg("-C")
        .arg(&fixture.worktree)
        .args([
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=test",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            &format!("feat: thing\n\nImmortality-Call: {call}"),
        ])
        .status()
        .expect("commit");
    assert!(status.success());
    fixture.open_call(
        call,
        "git",
        &json!({"subcommand": "commit", "message": "feat: thing"}),
        true,
    );

    let (_records, report, executed) = fixture.run_plan();
    assert_eq!(report.calls[0].effect, Some(Probe::Applied));
    assert!(!executed.steps.iter().any(|step| step.contains("re-ran")));
}

#[test]
fn a_commit_that_never_happened_is_not_applied() {
    let fixture = Fixture::new();
    fixture.open_call(
        "call_commit",
        "git",
        &json!({"subcommand": "commit", "message": "feat: thing"}),
        true,
    );
    let (_records, report, _executed) = fixture.run_plan();
    assert_eq!(report.calls[0].effect, Some(Probe::NotApplied));
}

#[test]
fn a_shell_command_is_unknown_and_never_retried() {
    // The class this mechanism exists for. Even a command that clearly did nothing is `Unknown`
    // because nothing about the tree says whether it ran.
    let fixture = Fixture::new();
    fixture.open_call("call_shell", "shell", &json!({"command": "rm -rf build"}), true);

    let (_records, report, executed) = fixture.run_plan();
    assert_eq!(report.calls[0].effect, Some(Probe::Unknown));
    assert!(
        !executed.steps.iter().any(|step| step.contains("re-ran shell")),
        "a destructive shell call must never be retried: {executed:?}"
    );
}

#[test]
fn a_call_without_stored_arguments_is_closed_not_guessed() {
    // An `intent` written by an older build has no `arguments`. Retrying with invented ones is
    // the duplicate this project must not have; the model gets an observation instead.
    let fixture = Fixture::new();
    let mut journal = fixture.journal();
    journal
        .append_scoped(
            fixture.clock.now(),
            Kind::ToolCallIntent,
            json!({
                "call": "call_old",
                "tool": "write_file",
                "idempotency": "idempotent",
                "expected_effect": "write src.rs",
            }),
            &Scope::attempt(SubtaskId::from_suffix(SUBTASK), 1),
        )
        .expect("intent");
    journal.close().expect("close");

    let (_records, report, executed) = fixture.run_plan();
    assert_eq!(
        report.calls[0].action,
        immortality_core::protocol::RecoveryAction::Retry
    );
    assert!(
        executed.steps.iter().any(|step| step.contains("abandoned")),
        "a call with no arguments must be abandoned, not invented: {executed:?}"
    );
    assert!(!fixture.worktree.join("src.rs").exists());
}

// ── the executor ──────────────────────────────────────────────────────────────────────────

#[test]
fn a_recovery_writes_its_own_contract_records() {
    let fixture = Fixture::new();
    fixture.open_call(
        "call_write",
        "write_file",
        &json!({"path": "src.rs", "content": "x\n"}),
        true,
    );

    let (_records, report, _executed) = fixture.run_plan();
    assert_eq!(report.verdict, Verdict::Repaired);

    let records = fixture.records();
    let recovered = records
        .records
        .iter()
        .find(|record| record.fields.kind == Kind::DaemonRecovered)
        .expect("daemon_recovered");
    assert!(
        recovered.event.payload["summary"]
            .as_str()
            .unwrap_or_default()
            .contains("unfinished tool calls"),
        "{:?}",
        recovered.event.payload
    );
    // The last record of the pass must be the summary: the audit describes the repair, so it is
    // written after it.
    assert_eq!(
        records.records.last().map(|record| record.fields.kind),
        Some(Kind::DaemonRecovered)
    );
}

#[test]
fn a_torn_tail_is_reported_and_the_plan_still_runs() {
    let fixture = Fixture::new();
    // A real half-line at the end of a real journal.
    fixture.journal().close().expect("create the journal");
    let path = fixture.journal_dir().join("JOURNAL.jsonl");
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .expect("open");
    file.write_all(b"{\"v\":1,\"seq\":999").expect("torn");
    file.sync_all().expect("sync");
    drop(file);

    // The plan reads the journal *before* anything opens it for writing: `Journal::open` is what
    // truncates the tail, so a plan built after that has no tail to report.
    let records = fixture.records();
    assert!(
        records
            .problems
            .iter()
            .any(|problem| matches!(problem, immortality_core::journal::ReadProblem::TornTail { .. })),
        "{:?}",
        records.problems
    );
    let (_records, report, executed) = fixture.run_plan();
    assert_eq!(report.verdict, Verdict::Repaired);
    assert!(
        executed.steps.iter().any(|step| step.contains("torn tail")),
        "{executed:?}"
    );
}

#[test]
fn a_divergent_cache_is_rewritten_from_the_journal() {
    let fixture = Fixture::new();
    let mut journal = fixture.journal();
    journal
        .append_scoped(
            fixture.clock.now(),
            Kind::SubtaskState,
            json!({ "state": "running" }),
            &Scope::subtask(SubtaskId::from_suffix(SUBTASK)),
        )
        .expect("state");
    journal.close().expect("close");

    // The cache claims `done`; the journal says `running`. The journal wins.
    let cache = Cache::new(&fixture.task_dir);
    cache
        .write(&immortality_core::recover::PersistedState {
            attempts: Vec::new(),
            subtasks: vec![immortality_core::recover::PersistedSubtask {
                id: SubtaskId::from_suffix(SUBTASK),
                state: immortality_core::domain::SubtaskState::Done,
            }],
        })
        .expect("cache");

    let (_records, report, _executed) = fixture.run_plan();
    assert_eq!(report.divergences.len(), 1, "{:?}", report.divergences);
    let rewritten = Cache::new(&fixture.task_dir).read();
    assert_eq!(
        rewritten.subtasks[0].state,
        immortality_core::domain::SubtaskState::Running,
        "the journal's state must land on disk"
    );
}

#[test]
fn a_dirty_worktree_is_snapshotted_with_a_journalled_sha() {
    let fixture = Fixture::new();
    std::fs::write(fixture.worktree.join("wip.rs"), "unsaved work\n").expect("write");

    let mut journal = fixture.journal();
    let records = fixture.records();
    let workspace = workspace_facts(&fixture.worktree);
    assert!(workspace.dirty, "the fixture must actually be dirty");
    let report = recover::plan(&fixture.input(&records, workspace), &fixture.probe());
    let deps = fixture.deps();
    let mut executor = exec::Executor::new(&deps, Some(&mut journal), records, &report);
    executor.run(&report.steps).expect("execution");
    journal.sync().expect("sync");

    let after = fixture.records();
    let preserved = after
        .records
        .iter()
        .find(|record| {
            record.fields.kind == Kind::CheckpointCreated && record.event.payload["role"] == "preserved"
        })
        .expect("a preserved checkpoint");
    let sha = preserved.event.payload["sha"].as_str().expect("sha");
    let repo = GitRepo::new(&fixture.worktree);
    assert!(repo.has_commit(sha), "the journalled snapshot must exist");
}

#[test]
fn a_lost_worktree_is_recreated_from_the_branch() {
    let fixture = Fixture::new();
    // The branch exists (the fixture made it); the worktree directory is gone.
    std::fs::remove_dir_all(&fixture.worktree).expect("remove worktree");
    // Register the worktree removal so git does not consider it locked.
    let status = Command::new("git")
        .arg("-C")
        .arg(fixture.dir.path().join("worktree"))
        .args(["worktree", "prune"])
        .status();
    let _ = status;

    let report = {
        let records = fixture.records();
        recover::plan(
            &fixture.input(&records, WorkspaceFacts::absent()),
            &fixture.probe(),
        )
    };
    assert!(
        report
            .steps
            .iter()
            .any(|step| matches!(step, recover::RecoveryStep::RecreateWorktree { .. })),
        "{:?}",
        report.steps
    );
}

#[test]
fn an_open_attempt_is_closed_when_recovery_interrupts_it() {
    // Every open attempt is closed, not only one a rollback supersedes: the process that owned it
    // is gone and the next one starts a fresh conversation. This case also plans a rollback — the
    // open `shell` call is `Unknown` with a checkpoint recorded — but the closure no longer
    // depends on it, and the record's `cause` says recovery rather than a failure.
    let fixture = Fixture::new();
    {
        let mut journal = fixture.journal();
        let scope = Scope::attempt(SubtaskId::from_suffix(SUBTASK), 1);
        journal
            .append_scoped(fixture.clock.now(), Kind::AttemptStarted, json!({}), &scope)
            .expect("attempt_started");
        let head = GitRepo::new(&fixture.worktree)
            .head()
            .expect("head")
            .expect("a commit");
        journal
            .append_scoped(
                fixture.clock.now(),
                Kind::CheckpointCreated,
                json!({ "sha": head, "branch": exec::expected_branch(&fixture.task), "role": "base" }),
                &Scope::task(),
            )
            .expect("checkpoint");
        journal.sync().expect("sync");
    }
    fixture.open_call("call_shell", "shell", &json!({"command": "rm -rf build"}), true);

    let records = fixture.records();
    let report = recover::plan(
        &fixture.input(&records, workspace_facts(&fixture.worktree)),
        &fixture.probe(),
    );
    let deps = fixture.deps();
    let mut journal = fixture.journal();
    let mut executor = exec::Executor::new(&deps, Some(&mut journal), records, &report);
    executor.run(&report.steps).expect("execution");
    journal.sync().expect("sync");

    let after = fixture.records();
    let restarted = after
        .records
        .iter()
        .find(|record| record.fields.kind == Kind::AttemptRestarted)
        .expect("the attempt must be closed");
    assert_eq!(restarted.event.payload["cause"], json!("recovery"));
    assert_eq!(restarted.fields.attempt, Some(1));
}

#[test]
fn a_rolled_back_shell_call_is_closed_on_disk_so_the_repair_does_not_repeat() {
    // The live failure this pins: a `kill -9` inside `cargo test` left an unknown `shell` call.
    // The first restart rolled the tree back and left the call open; the second restart found the
    // same call, rolled the tree back *again* — this time over the commit the following attempt
    // had already made — and the subtask stayed `done` with a red gate.
    //
    // The claim here is about the file, not the plan: after the executor runs, the journal must
    // hold a terminal record for the call, so `open_calls` is empty and a second start plans no
    // rollback at all.
    let fixture = Fixture::new();
    {
        let mut journal = fixture.journal();
        let head = GitRepo::new(&fixture.worktree)
            .head()
            .expect("head")
            .expect("a commit");
        journal
            .append_scoped(
                fixture.clock.now(),
                Kind::CheckpointCreated,
                json!({ "sha": head, "branch": exec::expected_branch(&fixture.task), "role": "base" }),
                &Scope::task(),
            )
            .expect("checkpoint");
        journal.sync().expect("sync");
    }
    fixture.open_call("call_shell", "shell", &json!({"command": "rm -rf build"}), true);

    let (_records, report, _executed) = fixture.run_plan();
    assert!(
        report
            .steps
            .iter()
            .any(|step| matches!(step, recover::RecoveryStep::RollbackToCheckpoint { .. })),
        "the fixture must plan the L3 rollback: {:?}",
        report.steps
    );

    let after = fixture.records();
    assert!(
        immortality_core::protocol::open_calls(&after.records).is_empty(),
        "the rollback answered the call, so the call must be closed: {:?}",
        immortality_core::protocol::open_calls(&after.records)
    );

    // The second start is the one that loses work if this is wrong.
    let second = recover::plan(
        &fixture.input(&after, workspace_facts(&fixture.worktree)),
        &fixture.probe(),
    );
    assert!(
        !second
            .steps
            .iter()
            .any(|step| matches!(step, recover::RecoveryStep::RollbackToCheckpoint { .. })),
        "a closed call must not plan a second rollback: {:?}",
        second.steps
    );
}

#[test]
fn a_corrupt_journal_blocks_the_task_and_asks_a_human() {
    let fixture = Fixture::new();
    // A chain break: a record whose `prev` does not match.
    let path = fixture.journal_dir().join("JOURNAL.jsonl");
    std::fs::create_dir_all(path.parent().unwrap()).expect("dir");
    std::fs::write(
        &path,
        concat!(
            r#"{"v":1,"seq":0,"ts":"2026-11-12T10:00:00.000Z","task":"tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV","kind":"task_created","prev":null,"payload":{}}"#,
            "\n",
            r#"{"v":1,"seq":1,"ts":"2026-11-12T10:00:01.000Z","task":"tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV","kind":"daemon_started","prev":"sha256:0000000000000000000000000000000000000000000000000000000000000000","payload":{}}"#,
            "\n",
        ),
    )
    .expect("write");

    let records = fixture.records();
    assert!(records.is_corrupt(), "the fixture must break the chain");
    let report = recover::plan(
        &fixture.input(&records, workspace_facts(&fixture.worktree)),
        &fixture.probe(),
    );
    assert_eq!(report.verdict, Verdict::Blocked);

    // A broken journal cannot be opened for writing, which is exactly why `BlockTask` records the
    // stop outside it. The executor is given an empty journal writer over a *separate* file, so any
    // attempt to write the block into the journal would show up here as a bad record rather than
    // as a panic inside `open`.
    let scratch = fixture.dir.path().join("scratch-journal");
    let mut journal = Journal::open(&scratch, fixture.task.clone()).expect("scratch journal");
    journal.set_clock(Arc::clone(&fixture.clock) as Arc<dyn immortality_core::time::Clock>);
    let deps = fixture.deps();
    let mut executor = exec::Executor::new(&deps, Some(&mut journal), records, &report);
    let executed = executor.run(&report.steps).expect("execution");
    journal.sync().expect("sync");
    assert!(
        executed
            .steps
            .iter()
            .any(|step| step.contains("blocked the task")),
        "{executed:?}"
    );

    // The marker is outside the journal, because the journal is the thing that cannot be trusted.
    let marker: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture.task_dir.join("panic.json")).expect("panic.json"),
    )
    .expect("json");
    assert_eq!(marker["state"], json!("blocked"));
    assert_eq!(marker["cause"], json!("journal_corrupt"));
    assert!(
        marker["reason"]
            .as_str()
            .unwrap_or_default()
            .contains("chain broken"),
        "{marker}"
    );
    // And no `daemon_recovered` was written into the scratch journal either.
    let store = EscalationStore::open(&fixture.task_dir, fixture.task.clone()).expect("store");
    let entry = store.by_key("recovery:journal_corrupt").expect("the question");
    assert!(entry.is_open(), "a human must still have to answer");
    assert_eq!(entry.options, Answer::ALL.to_vec());
}

#[test]
fn recovery_is_idempotent_on_a_second_pass() {
    // The daemon may crash during its own recovery. The second pass must find nothing to do
    // rather than re-run the retry or roll back twice.
    //
    // The retried write leaves an untracked file, and a dirty tree genuinely earns a snapshot on
    // the next start — so the file is committed before the second pass. Without that, this test
    // would fail on `PreserveDirtyWork` and say nothing about idempotency.
    let fixture = Fixture::new();
    fixture.open_call(
        "call_write",
        "write_file",
        &json!({"path": "src.rs", "content": "x\n"}),
        true,
    );
    fixture.run_plan();
    commit_all(&fixture.worktree);

    let records = fixture.records();
    let report = recover::plan(
        &fixture.input(&records, clean(workspace_facts(&fixture.worktree))),
        &fixture.probe(),
    );
    assert!(
        report.calls.is_empty(),
        "the answered call must not be answered again: {:?}",
        report.calls
    );
    assert_eq!(
        report.verdict,
        Verdict::Healthy,
        "the second pass must be clean: {:?}",
        report.steps
    );
    // Exactly one terminal record for the call, however many passes ran.
    let terminals = records
        .records
        .iter()
        .filter(|record| {
            record.fields.kind == Kind::ToolCallCompleted
                && record.event.payload["call"] == json!("call_write")
        })
        .count();
    assert_eq!(terminals, 1, "the call was answered more than once");
}

#[test]
fn an_unfinished_task_is_recovered_and_a_merged_one_is_not() {
    let fixture = Fixture::new();
    let records = fixture.records();
    assert!(
        exec::is_unfinished(&records.records),
        "a journal with no state record still has work ahead"
    );

    let mut journal = fixture.journal();
    journal
        .append_scoped(
            fixture.clock.now(),
            Kind::TaskState,
            json!({ "state": "merged" }),
            &Scope::task(),
        )
        .expect("state");
    journal.close().expect("close");

    let records = fixture.records();
    assert!(!exec::is_unfinished(&records.records));
}

#[test]
fn daemon_started_and_stopped_bracket_a_session() {
    let fixture = Fixture::new();
    let journal = exec::daemon_started(
        &fixture.journal_dir(),
        fixture.task.clone(),
        fixture.clock.as_ref(),
    )
    .expect("started");
    journal.close().expect("close");
    exec::daemon_stopped(&fixture.journal_dir(), fixture.task.clone(), fixture.clock.now()).expect("stopped");

    let records = fixture.records();
    let kinds: Vec<Kind> = records.records.iter().map(|record| record.fields.kind).collect();
    assert_eq!(kinds, vec![Kind::DaemonStarted, Kind::DaemonStopped]);
}

#[test]
fn the_state_cache_is_left_consistent_after_a_repair() {
    // The last step rewrites the cache even when no divergence was found: the pass may have closed
    // an attempt, and a cache that still called it open would make the next `status` lie.
    let fixture = Fixture::new();
    {
        let mut journal = fixture.journal();
        let scope = Scope::attempt(SubtaskId::from_suffix(SUBTASK), 1);
        journal
            .append_scoped(fixture.clock.now(), Kind::AttemptStarted, json!({}), &scope)
            .expect("attempt_started");
        journal
            .append_scoped(
                fixture.clock.now(),
                Kind::AttemptFailed,
                json!({ "reason": "the model stalled" }),
                &scope,
            )
            .expect("attempt_failed");
        journal
            .append_scoped(
                fixture.clock.now(),
                Kind::SubtaskState,
                json!({ "state": "blocked" }),
                &Scope::subtask(SubtaskId::from_suffix(SUBTASK)),
            )
            .expect("subtask_state");
        journal.sync().expect("sync");
    }
    fixture.open_call(
        "call_write",
        "write_file",
        &json!({"path": "src.rs", "content": "x\n"}),
        true,
    );
    fixture.run_plan();

    // The claim is equality with the journal, not a hand-picked field: the cache is a derived
    // view, and any field it gets wrong is a `status` that lies.
    let records = fixture.records();
    let expected = immortality_core::cache::derived(&records.records);
    let cache = Cache::new(&fixture.task_dir).read();
    assert_eq!(cache, expected);
    assert!(
        cache.attempts.iter().all(|attempt| attempt.finished),
        "the cache must reflect the failed attempt: {cache:?}"
    );
    assert_eq!(
        cache.subtasks[0].state,
        immortality_core::domain::SubtaskState::Blocked
    );
}

#[test]
fn the_probe_never_answers_applied_for_an_unknown_tool() {
    let fixture = Fixture::new();
    let probe = fixture.probe();
    let intent = Intent::new(
        CallId::from_model("call_x"),
        "a_tool_that_does_not_exist",
        &json!({}),
        immortality_core::protocol::Idempotency::Unknown,
        "do something",
    );
    assert_eq!(
        probe.probe(&intent, Some(&json!({"anything": true}))),
        Probe::Unknown
    );
}

#[test]
fn the_prober_reads_through_the_same_gate_the_tool_used() {
    // A path outside the worktree is refused by the gate's `resolve`; a probe that joined the raw
    // argument onto the worktree itself would read a different file than the tool wrote.
    let fixture = Fixture::new();
    let probe = fixture.probe();
    let intent = Intent::new(
        CallId::from_model("call_outside"),
        "write_file",
        &json!({}),
        immortality_core::protocol::Idempotency::ByContent,
        "write outside",
    );
    let arguments = json!({"path": "../../../etc/passwd", "content": "x"});
    assert_eq!(probe.probe(&intent, Some(&arguments)), Probe::Unknown);
}

#[test]
fn every_writing_tool_that_cannot_be_replayed_is_not_safe_to_retry() {
    // The v0.1 criterion "0 double-applied non-idempotent actions" is only as good as the classes
    // the tools declare, and the class decides what recovery does with an interrupted call: a tool
    // that says `Idempotent` is re-run blindly, one that says `ByContent` or `Unknown` is not.
    //
    // This test exists because the previous version of this file asked a hand-written table
    // instead of the tool. That table had already drifted — `escalate` and `kill_process` declare
    // `ByContent` and the table put them in its `Idempotent` default — so the tests were asserting
    // about an arm no tool takes. Reading the registry is what makes the claim about the code that
    // ships.
    let fixture = Fixture::new();
    let deps = immortality_daemon::toolset::ToolsetDeps::for_task(
        &immortality_core::config::Config::default(),
        fixture.dir.path(),
        &fixture.task_dir,
        &fixture.worktree,
        fixture.dir.path(),
        fixture.task.clone(),
    )
    .expect("toolset deps");
    let registry = immortality_daemon::toolset::build(&deps).expect("registry");

    // Every tool whose replay repeats an effect, with arguments that name the dangerous thing it
    // does. `shell` and `start_process` are where a replay runs a second process; `git commit` is
    // where it makes a second commit; `escalate` and `delegate` write queue records; the two state
    // and file writers rewrite content.
    //
    // `kill_process` is deliberately absent: killing an already-dead process does nothing, so
    // `Idempotent` is the honest class there and the second kill finds nothing to repeat.
    let writing: [(&str, serde_json::Value); 7] = [
        ("shell", json!({"command": "rm -rf build"})),
        ("start_process", json!({"command": "sleep 60"})),
        ("git", json!({"subcommand": "commit", "message": "x"})),
        ("escalate", json!({"question": "may I?"})),
        ("delegate", json!({"brief": "x", "scope": "y"})),
        ("write_file", json!({"path": "a.txt", "content": "x"})),
        ("update_state", json!({"file": "facts", "content": "x"})),
    ];
    for (name, arguments) in writing {
        let tool = registry
            .get(name)
            .unwrap_or_else(|| panic!("`{name}` is missing from every_tool_set"));
        let class = tool.idempotency(&immortality_tools::Args::new(arguments));
        assert!(
            !class.is_safe_to_retry(),
            "`{name}` declares `{class:?}`, so recovery would replay it blindly; a doubled \
             non-idempotent action is the one v0.1 criterion this test exists to keep honest"
        );
    }
}
