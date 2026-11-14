//! The startup pass, end to end: a real repository, a real journal, real tools.
//!
//! # Why this cannot be a unit test
//!
//! Three of the four claims M5.2 wiring makes are about the *filesystem as the daemon finds it*:
//! the journal is in the project's `.agent/` and not in the worktree, a torn tail is still on disk
//! when the plan is built, and a blocked task's journal cannot be opened at all. Each of those is a
//! layout fact, and a fixture that gets the layout wrong tests the fixture.
//!
//! # What is simulated
//!
//! Nothing in the crash. The journal is written as a crashed process would leave it, and the
//! daemon's own startup path — [`startup::recover_task`] — is called exactly as `serve` calls it.

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
use immortality_core::config::{Config, Policy};
use immortality_core::domain::TaskId;
use immortality_core::ids::SubtaskId;
use immortality_core::journal::{Journal, JournalReader, Kind, Scope};
use immortality_core::protocol::ToolCallLog;
use immortality_core::recover::Verdict;
use immortality_core::time::{Clock as _, FixedClock};
use immortality_core::worktree::WorktreeManager;
use immortality_daemon::startup::{self, RecoveryRequest};
use immortality_daemon::toolset::{self, ToolsetDeps};
use immortality_tools::PolicyGate;
use serde_json::json;

const TASK: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAV";
const SUBTASK: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAA";

/// A project, its worktree and a state root, laid out the way the daemon lays them out.
struct Fixture {
    dir: tempfile::TempDir,
    /// The project's main checkout. The journal lives in `<project>/.agent/`.
    project: PathBuf,
    /// `<parent>/<name>-agent/<task>`, created by `git worktree add`.
    worktree: PathBuf,
    state_root: PathBuf,
    task: TaskId,
    clock: Arc<FixedClock>,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        // The worktree base dir must be a sibling of the project, and `WorktreeManager` derives
        // that from the repository's own name, so the project has to have one.
        let project = dir.path().join("project");
        std::fs::create_dir_all(&project).expect("project");
        for args in [
            vec!["init", "-q", "-b", "main"],
            vec!["config", "user.email", "test@example.com"],
            vec!["config", "user.name", "test"],
        ] {
            let status = Command::new("git")
                .arg("-C")
                .arg(&project)
                .args(&args)
                .status()
                .expect("git");
            assert!(status.success(), "git {args:?} failed");
        }
        std::fs::write(project.join("README.md"), "start\n").expect("readme");
        std::fs::create_dir_all(project.join(".agent")).expect("agent dir");
        let status = Command::new("git")
            .arg("-C")
            .arg(&project)
            .args(["add", "--all"])
            .status()
            .expect("git add");
        assert!(status.success());
        let status = Command::new("git")
            .arg("-C")
            .arg(&project)
            .args(["commit", "-q", "-m", "initial"])
            .status()
            .expect("git commit");
        assert!(status.success());

        let task = TaskId::from_suffix(TASK);
        let manager = WorktreeManager::new(&project);
        let worktree = manager.path_for(task.as_str());
        manager
            .ensure(task.as_str(), &format!("agent/{task}"), Some("main"))
            .expect("worktree");

        let state_root = dir.path().join("state");
        let task_dir = state_root.join(task.as_str());
        std::fs::create_dir_all(&task_dir).expect("task dir");

        Self {
            dir,
            project,
            worktree,
            state_root,
            task,
            clock: Arc::new(FixedClock::fixture()),
        }
    }

    fn journal_dir(&self) -> PathBuf {
        toolset::journal_dir(&self.project)
    }

    fn journal(&self) -> Journal {
        let mut journal = Journal::open(self.journal_dir(), self.task.clone()).expect("journal");
        journal.set_clock(Arc::clone(&self.clock) as Arc<dyn immortality_core::time::Clock>);
        journal
    }

    fn records(&self) -> immortality_core::journal::ReadOutcome {
        JournalReader::open(self.journal_dir(), &self.task)
            .expect("reader")
            .read()
            .expect("read")
    }

    fn task_dir(&self) -> PathBuf {
        self.state_root.join(self.task.as_str())
    }

    /// Write an interrupted call: an intent, and a started record when the process got that far.
    fn open_call(&self, call: &str, tool: &str, arguments: &serde_json::Value, started: bool) {
        let mut journal = self.journal();
        let scope = Scope::attempt(SubtaskId::from_suffix(SUBTASK), 1);
        let request = self.request();
        let class = idempotency_for(&request.registry, tool, arguments);
        {
            // One log for both records: the ordering guard lives in its in-memory map, and a
            // second log would refuse a `started` whose `intent` it never saw.
            let mut log = ToolCallLog::scoped(&mut journal, scope);
            let intent = immortality_core::protocol::Intent::new(
                immortality_core::protocol::CallId::from_model(call),
                tool,
                arguments,
                class,
                format!("call {call}"),
            );
            log.intent(&intent, arguments).expect("intent");
            if started {
                log.started(&intent.call, serde_json::Value::Null)
                    .expect("started");
            }
        }
        journal.close().expect("close");
    }

    /// Journal the first record inside an attempt, which is what makes a task "begun".
    ///
    /// Recovery deliberately ignores a journal that carries only task-wide bookkeeping, so every
    /// fixture that expects a repair has to say that a process reached an attempt first.
    fn begin_attempt(&self) {
        let mut journal = self.journal();
        journal
            .append_scoped(
                self.clock.now(),
                Kind::AttemptStarted,
                json!({}),
                &Scope::attempt(SubtaskId::from_suffix(SUBTASK), 1),
            )
            .expect("attempt_started");
        journal.close().expect("close");
    }

    /// The request `serve` builds, with a registry of the real tools.
    fn request(&self) -> RecoveryRequest {
        let deps = ToolsetDeps::for_task(
            &Config::default(),
            &self.state_root,
            self.task_dir(),
            &self.worktree,
            &self.project,
            self.task.clone(),
        )
        .expect("toolset deps");
        RecoveryRequest {
            task: self.task.clone(),
            repository: self.project.clone(),
            journal_dir: self.journal_dir(),
            task_dir: self.task_dir(),
            registry: Arc::new(toolset::build(&deps).expect("registry")),
            gate: Arc::clone(&deps.gate),
            artifacts: ArtifactStore::open(self.dir.path().join("artifacts")).expect("artifacts"),
            clock: Arc::clone(&self.clock) as Arc<dyn immortality_core::time::Clock>,
        }
    }

    fn recover(&self) -> startup::Recovered {
        startup::recover_task(&self.request()).expect("recovery")
    }
}

/// The idempotency class each tool declares, taken from the tools themselves.
///
/// # Why this is not a match on the tool's name
///
/// It used to be one, and it had already drifted from the code: `escalate` and `kill_process`
/// both declare `ByContent`, and the hand-written table put them in its `_ => Idempotent` arm.
/// Every test built on that table was therefore exercising an arm the tool never reaches — the
/// exact failure its own comment warned about, made invisible by the copy.
///
/// Taking the class from the registry `request()` builds is the version that cannot drift: a tool
/// that changes its class changes what these tests exercise, in the same commit.
fn idempotency_for(
    registry: &immortality_tools::Registry,
    tool: &str,
    arguments: &serde_json::Value,
) -> immortality_core::protocol::Idempotency {
    let found = registry
        .get(tool)
        .unwrap_or_else(|| panic!("`{tool}` is not a production tool, so its class is unknown"));
    found.idempotency(&immortality_tools::Args::new(arguments.clone()))
}

/// A gate over the fixture's worktree, for the cases that assert on policy refusals.
fn gate(fixture: &Fixture) -> Arc<PolicyGate> {
    Arc::new(
        PolicyGate::with_defaults(
            &fixture.worktree,
            Policy {
                workspace_root: fixture.dir.path().to_path_buf(),
                ..Policy::default()
            },
        )
        .expect("gate"),
    )
}

fn write_file(worktree: &Path, name: &str, contents: &str) {
    std::fs::write(worktree.join(name), contents).expect("write");
}

#[test]
fn a_write_that_landed_is_closed_without_running_again() {
    // The bug the D-081 arguments seam exists for: without the stored content the probe can only
    // answer `Unknown`, and `Unknown` rolls the tree back over a complete file.
    let fixture = Fixture::new();
    write_file(&fixture.worktree, "src.rs", "pub fn x() {}\n");
    fixture.open_call(
        "call_write",
        "write_file",
        &json!({"path": "src.rs", "content": "pub fn x() {}\n"}),
        true,
    );

    let recovered = fixture.recover();
    assert_eq!(recovered.verdict, Verdict::Repaired);
    assert!(
        recovered
            .executed
            .steps
            .iter()
            .any(|step| step.contains("already applied")),
        "{:?}",
        recovered.executed
    );
    // And the file is still the same file: a second write would have rewritten it, which is
    // invisible for identical content — so the record is what pins it.
    let records = fixture.records();
    let terminals = records
        .records
        .iter()
        .filter(|record| record.fields.kind == Kind::ToolCallCompleted)
        .count();
    assert_eq!(terminals, 1);
    assert_eq!(
        std::fs::read_to_string(fixture.worktree.join("src.rs")).expect("read"),
        "pub fn x() {}\n"
    );
}

#[test]
fn a_write_that_did_not_land_is_re_run() {
    let fixture = Fixture::new();
    fixture.open_call(
        "call_write",
        "write_file",
        &json!({"path": "src.rs", "content": "pub fn y() {}\n"}),
        true,
    );

    let recovered = fixture.recover();
    assert_eq!(recovered.verdict, Verdict::Repaired);
    assert_eq!(
        std::fs::read_to_string(fixture.worktree.join("src.rs")).expect("read"),
        "pub fn y() {}\n",
        "the interrupted write must actually happen"
    );
}

#[test]
fn the_plan_is_built_before_the_journal_is_opened_for_writing() {
    // `Journal::open` truncates the tail and journals the repair. A plan built after the open
    // reports a clean start for a crash — the one thing recovery must never do.
    let fixture = Fixture::new();
    fixture.begin_attempt();
    let path = fixture.journal_dir().join("JOURNAL.jsonl");
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .expect("open");
    file.write_all(b"{\"v\":1,\"seq\":999").expect("torn");
    file.sync_all().expect("sync");
    drop(file);

    let recovered = fixture.recover();
    assert_eq!(recovered.verdict, Verdict::Repaired);
    assert!(
        recovered
            .executed
            .steps
            .iter()
            .any(|step| step.contains("torn tail")),
        "{:?}",
        recovered.executed
    );
    // The repair is journalled after the truncation, into the repaired file.
    let records = fixture.records();
    assert!(
        records
            .records
            .iter()
            .any(|record| record.fields.kind == Kind::JournalTruncatedTail),
        "{:?}",
        records.records.iter().map(|r| r.fields.kind).collect::<Vec<_>>()
    );
    assert!(recovered.summary.contains("torn"), "{}", recovered.summary);
}

#[test]
fn a_corrupt_journal_blocks_the_task_without_writing_to_it() {
    let fixture = Fixture::new();
    fixture.open_call(
        "call_write",
        "write_file",
        &json!({"path": "a.rs", "content": "x"}),
        true,
    );
    // Break the chain by rewriting a middle line's `prev`.
    let path = fixture.journal_dir().join("JOURNAL.jsonl");
    let text = std::fs::read_to_string(&path).expect("read");
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    assert!(lines.len() >= 2, "the fixture must produce a chain");
    lines[1] = lines[1].replace("\"prev\":\"sha256:", "\"prev\":\"sha256:beef");
    std::fs::write(&path, lines.join("\n") + "\n").expect("break");
    let before = std::fs::read_to_string(&path).expect("read");

    let recovered = fixture.recover();
    assert_eq!(recovered.verdict, Verdict::Blocked, "{:?}", recovered.executed);

    // The journal is byte-identical: repairing a broken chain silently would erase the evidence.
    assert_eq!(
        std::fs::read_to_string(&path).expect("read"),
        before,
        "recovery wrote to a journal it declared unreadable"
    );

    // The block is durable outside the journal, where a human can find it.
    let marker: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture.task_dir().join("panic.json")).expect("panic.json"),
    )
    .expect("json");
    assert_eq!(marker["state"], json!("blocked"));
    assert_eq!(marker["cause"], json!("journal_corrupt"));

    // `fixture.task` is moved: this is the test's last use of it, and the clone clippy flags is
    // the kind of needless copy that hides ownership.
    let task = fixture.task.clone();
    let store =
        immortality_core::escalation::EscalationStore::open(&fixture.task_dir(), task).expect("store");
    let entry = store.by_key("recovery:journal_corrupt").expect("question");
    assert!(entry.is_open());
    assert!(entry.blocking, "a corrupt journal stops everything");
}

#[test]
fn a_dirty_worktree_is_snapshotted_before_anything_moves() {
    let fixture = Fixture::new();
    fixture.begin_attempt();
    write_file(&fixture.worktree, "wip.rs", "unsaved work\n");

    let recovered = fixture.recover();
    assert_eq!(recovered.verdict, Verdict::Repaired);
    let records = fixture.records();
    let preserved = records
        .records
        .iter()
        .find(|record| {
            record.fields.kind == Kind::CheckpointCreated
                && record.event.payload["role"] == json!("preserved")
        })
        .expect("a preserved checkpoint");
    let sha = preserved.event.payload["sha"].as_str().expect("sha");
    assert!(
        GitRepo::new(&fixture.worktree).has_commit(sha),
        "the journalled snapshot must exist"
    );
}

#[test]
fn an_unknown_effect_rolls_back_and_closes_the_attempt() {
    // A mutating shell command is the class the mechanism exists for: no cheap evidence, so the
    // honest answer is `Unknown`, and `Unknown` means roll back rather than re-run.
    let fixture = Fixture::new();
    let head = GitRepo::new(&fixture.worktree)
        .head()
        .expect("head")
        .expect("a commit");
    {
        let mut journal = fixture.journal();
        let scope = Scope::attempt(SubtaskId::from_suffix(SUBTASK), 1);
        journal
            .append_scoped(fixture.clock.now(), Kind::AttemptStarted, json!({}), &scope)
            .expect("attempt_started");
        journal
            .append_scoped(
                fixture.clock.now(),
                Kind::CheckpointCreated,
                json!({ "sha": head, "branch": format!("agent/{}", fixture.task), "role": "base" }),
                &Scope::task(),
            )
            .expect("checkpoint");
        journal.close().expect("close");
    }
    fixture.open_call("call_shell", "shell", &json!({"command": "rm -rf build"}), true);

    let recovered = fixture.recover();
    assert_eq!(recovered.verdict, Verdict::Repaired);
    assert!(
        recovered
            .executed
            .steps
            .iter()
            .any(|step| step.contains("rolled the tree back")),
        "{:?}",
        recovered.executed
    );

    let records = fixture.records();
    // The attempt is closed: a plain restart resumes it, a rollback ends it, and leaving it
    // `Running` would make the subtask wait forever for a process that is gone.
    assert!(
        records
            .records
            .iter()
            .any(|record| record.fields.kind == Kind::AttemptRestarted),
        "{:?}",
        records.records.iter().map(|r| r.fields.kind).collect::<Vec<_>>()
    );
    // The journal survived the rollback: it lives outside the worktree.
    assert!(!records.records.is_empty(), "`reset --hard` deleted the journal");
}

#[test]
fn a_finished_task_is_not_touched() {
    let fixture = Fixture::new();
    {
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
    }
    write_file(&fixture.worktree, "wip.rs", "should stay untracked\n");
    let before = fixture.records().records.len();

    let recovered = fixture.recover();
    assert_eq!(recovered.verdict, Verdict::Healthy);
    assert!(!recovered.repaired());
    assert_eq!(recovered.summary, "finished; not recovered");
    assert_eq!(
        fixture.records().records.len(),
        before,
        "a merged task's history must not grow during recovery"
    );
}

#[test]
fn only_a_pass_that_opened_the_journal_reports_a_session() {
    // `serve` writes `daemon_stopped` for exactly the tasks whose `session_open` is true. A
    // `daemon_stopped` without its `daemon_started` is a session that never began, and the pairing
    // is what makes the journal readable as a sequence of daemon lifetimes.
    let fresh = Fixture::new();
    let untouched = fresh.recover();
    assert!(
        !untouched.session_open,
        "a task that never began opened a session"
    );

    let begun = Fixture::new();
    begun.begin_attempt();
    let opened = begun.recover();
    assert!(
        opened.session_open,
        "a pass that opened the journal must bracket its session"
    );
}

#[test]
fn a_blocked_pass_reports_no_session() {
    // A blocked task's journal cannot be opened at all, so claiming a session would make `serve`
    // try to write `daemon_stopped` into the very file it just declared unreadable.
    let fixture = Fixture::new();
    // Two records: a chain needs a second line to break.
    fixture.open_call(
        "call_write",
        "write_file",
        &json!({ "path": "a.rs", "content": "x" }),
        true,
    );
    let path = fixture.journal_dir().join("JOURNAL.jsonl");
    let text = std::fs::read_to_string(&path).expect("read");
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    assert!(lines.len() >= 2, "the fixture must produce a chain");
    lines[1] = lines[1].replace("\"prev\":\"sha256:", "\"prev\":\"sha256:beef");
    std::fs::write(&path, lines.join("\n") + "\n").expect("break");

    let recovered = fixture.recover();
    assert_eq!(recovered.verdict, Verdict::Blocked);
    assert!(!recovered.session_open);
}

#[test]
fn a_lost_worktree_is_recreated_from_the_repository() {
    // The bug that made `RecoveryDeps.repository` a separate field: `WorktreeManager` was built
    // from the *worktree*, so a deleted worktree made it look for `<deleted>/../<deleted>-agent`
    // and create nothing. Recovery would report success over a tree that was still missing.
    let fixture = Fixture::new();
    fixture.begin_attempt();
    std::fs::remove_dir_all(&fixture.worktree).expect("remove worktree");
    assert!(!fixture.worktree.exists());

    let recovered = fixture.recover();
    assert_eq!(recovered.verdict, Verdict::Repaired, "{:?}", recovered.executed);
    assert!(
        fixture.worktree.is_dir(),
        "the worktree was not recreated; steps: {:?}",
        recovered.executed
    );
    let facts = WorktreeManager::new(&fixture.project)
        .find(fixture.task.as_str())
        .expect("find")
        .expect("a worktree");
    assert_eq!(
        facts.branch.as_deref(),
        Some(format!("agent/{}", fixture.task).as_str())
    );
}

#[test]
fn a_delegated_report_written_by_the_tool_is_found_by_the_probe() {
    // The defect this pins: `ToolProbe` was given the machine state root while the tool writes
    // under `<root>/state/<project>/<task>/delegations`, so a whole delegated run was invisible
    // and would have been repeated — spending a sub-agent's budget twice. The report here is
    // written by `DelegationStore` exactly as the tool writes it, not by a fixture that guesses.
    use immortality_core::delegation::{DelegationId, DelegationReport, DelegationStore, Finish};
    use immortality_core::domain::SubtaskReport;

    let fixture = Fixture::new();
    fixture.begin_attempt();

    let call = "call_01ARZ3NDEKTSV4RRFFQ69G5FAV";
    // The tool resolves its store as `state_root/<task>/delegations` where `state_root` is the
    // directory holding the task's state directory.
    let tool_root = fixture.task_dir().parent().expect("parent").to_path_buf();
    let store = DelegationStore::for_task(&tool_root, &fixture.task);
    let report = DelegationReport::new(
        DelegationId::for_call(call),
        "brief",
        Vec::new(),
        5,
        fixture.clock.now(),
    )
    .with_outcome(1, Finish::Done, SubtaskReport::new("done"), None);
    store.write(&report).expect("write");

    fixture.open_call(
        call,
        "delegate",
        &json!({ "brief": "brief", "max_steps": 5 }),
        true,
    );

    let recovered = fixture.recover();
    assert!(
        recovered
            .executed
            .steps
            .iter()
            .any(|step| step.contains("already applied")),
        "the probe did not find the report; steps: {:?}",
        recovered.executed
    );
}

#[test]
fn a_task_that_never_began_is_not_recovered() {
    // The end-to-end case that caught this: the CLI's own fixture journals two task-wide records
    // for a registered project, and recovery used to add a `daemon_started`, ask git to recreate a
    // branch nobody created, and grow the history on every daemon start. A task with no record
    // inside an attempt has nothing a crash could have interrupted.
    let fixture = Fixture::new();
    {
        let mut journal = fixture.journal();
        journal
            .append_event(
                fixture.clock.now(),
                Kind::TaskState,
                json!({ "state": "running" }),
            )
            .expect("state");
        journal.close().expect("close");
    }
    let before = fixture.records().records.len();

    let recovered = fixture.recover();
    assert_eq!(recovered.verdict, Verdict::Healthy);
    assert!(!recovered.repaired());
    assert_eq!(recovered.summary, "not started; nothing to recover");
    assert_eq!(
        fixture.records().records.len(),
        before,
        "recovery must not add a record to a task that has not started"
    );
    assert!(
        !fixture.worktree.join("panic.json").exists(),
        "a task with no work is not blocked"
    );
}

#[test]
fn the_state_cache_ends_up_equal_to_the_journal() {
    let fixture = Fixture::new();
    // A stale cache that claims the opposite of the journal.
    let stale = immortality_core::recover::PersistedState {
        attempts: Vec::new(),
        subtasks: Vec::new(),
    };
    Cache::new(fixture.task_dir()).write(&stale).expect("write");
    fixture.open_call(
        "call_write",
        "write_file",
        &json!({"path": "src.rs", "content": "x\n"}),
        true,
    );

    fixture.recover();
    let expected = immortality_core::cache::derived(&fixture.records().records);
    assert_eq!(Cache::new(fixture.task_dir()).read(), expected);
}

#[test]
fn a_path_outside_the_worktree_is_never_written() {
    // The prober must read through the gate, and so must the re-issued call: a path outside the
    // worktree is denied, so no probe can claim it landed and no retry can write it.
    let fixture = Fixture::new();
    fixture.open_call(
        "call_write",
        "write_file",
        &json!({"path": "../../etc/immortality-test", "content": "nope\n"}),
        false,
    );

    let recovered = fixture.recover();
    assert!(recovered.repaired());
    assert!(
        !fixture.dir.path().join("etc").exists(),
        "recovery escaped the worktree"
    );
    // The denial is the gate's, not this test's: the same call the tool would have made is
    // refused for the same reason.
    match gate(&fixture).check(
        immortality_tools::Effect::Mutate,
        &[immortality_tools::PathUse::write("../../etc/immortality-test")],
    ) {
        immortality_tools::Verdict::Denied(denial) => {
            assert!(denial.reason.contains("outside the task worktree"), "{denial:?}");
        }
        other => panic!("a path outside the worktree must be denied, got {other:?}"),
    }
}

#[test]
fn recovery_is_idempotent_across_two_passes() {
    let fixture = Fixture::new();
    fixture.open_call(
        "call_write",
        "write_file",
        &json!({"path": "src.rs", "content": "x\n"}),
        true,
    );
    fixture.recover();
    // Commit the retried write so the second pass sees a clean tree; otherwise it legitimately
    // earns a snapshot and this test would say nothing about idempotency.
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
            .arg(&fixture.worktree)
            .args(&args)
            .status()
            .expect("git");
        assert!(status.success());
    }

    let second = fixture.recover();
    assert_eq!(second.verdict, Verdict::Healthy, "{:?}", second.executed);
    let terminals = fixture
        .records()
        .records
        .iter()
        .filter(|record| {
            record.fields.kind == Kind::ToolCallCompleted
                && record.event.payload["call"] == json!("call_write")
        })
        .count();
    assert_eq!(terminals, 1, "the call was answered twice");
}
