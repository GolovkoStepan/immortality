//! `start`: the daemon's own run cycle, end to end.
//!
//! # What this file proves that `simulator.rs` cannot
//!
//! The simulator proves the cycle against a fake world. This proves the *wiring*: that the RPC
//! handler mints a task, plans when there is no plan, builds a real worktree, installs the journal
//! sink, spawns the run on its own thread and lets the cycle reach a verdict — all through the
//! function `serve` calls. The fake part is only the model, which is the seam
//! [`immortality_daemon::runner::Model`] exists for; everything else is a real repository, a real
//! journal and real git.
//!
//! # Why the plan is sometimes written by the planner and sometimes by hand
//!
//! Both are normal first invocations: a goal with nothing on disk is planned, and a goal with an
//! edited `TASKS.json` is respected. A fixture that always used one of them would leave the other
//! path untested, and the difference is where "it silently replaced my edits" lives.
//!
//! # Why the model is scripted by role
//!
//! One `Model` answers all three callers — planner, executor, reviewer — and they are told apart by
//! their system prompt. A fake that answered by call order would pass even if the handler asked the
//! wrong role, which is the bug that would make a run hang on a reviewer that was never called.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use immortality_core::config::{Config, Policy};
use immortality_core::domain::{Dod, DodCheck, Subtask, SubtaskState, TaskState, Timestamp};
use immortality_core::journal::{Journal, JournalReader, JournalRecord, Kind};
use immortality_core::plan::SCHEMA_VERSION;
use immortality_core::registry::Registry;
use immortality_core::time::{Clock as _, SystemClock};
use immortality_daemon::runner::{FINISH_SUBTASK, Model, ModelRequest, Reply};
use immortality_daemon::server::{Events, ServerState};
use immortality_daemon::{start, startup};
use serde_json::json;

/// How long a run may take before the test calls it hung.
///
/// Generous on purpose: the run does real `git worktree add`, a real gate command and an `fsync`
/// per journal record, and a loaded CI machine is slower than this developer's laptop. A hung run
/// is the failure this project exists to prevent, so the test waits long enough to be about
/// correctness and fails loudly if the cycle never ends.
const RUN_BUDGET: Duration = Duration::from_secs(120);

/// The goal every test uses. Quoted by the strategy, so a resumed run can be checked against it.
const GOAL: &str = "add a notes file";

// ── the visible world ───────────────────────────────────────────────────────────────────

/// A registered git repository, a state root with a token, and the config the daemon runs with.
struct Fixture {
    dir: tempfile::TempDir,
    repo: PathBuf,
    state_root: PathBuf,
    events: Events,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let repo = dir.path().join("repo");
        let state_root = dir.path().join("state");
        std::fs::create_dir_all(&repo).expect("repo");
        std::fs::create_dir_all(&state_root).expect("state root");
        std::fs::write(state_root.join("daemon.token"), "fixture-token").expect("token");
        std::fs::write(repo.join("README.md"), "# demo\n").expect("readme");
        // `ServerState::discover` refuses to run without a token, and `Registry::register`
        // canonicalises its argument, so a fixture that skipped either would test a daemon that
        // cannot exist.
        git(&repo, &["init", "-q", "-b", "main"]);
        git(&repo, &["config", "user.email", "test@example.com"]);
        git(&repo, &["config", "user.name", "test"]);
        git(&repo, &["add", "--all"]);
        git(&repo, &["commit", "-q", "-m", "initial"]);

        let mut registry =
            Registry::open(state_root.join(immortality_core::registry::FILE_NAME)).expect("registry");
        registry.register(&repo, None).expect("register");

        Self {
            dir,
            repo,
            state_root,
            events: Events::new(),
        }
    }

    /// The configuration `serve` would load, with the workspace root moved over the fixture.
    ///
    /// The policy gate confines reads to `workspace_root`, and the default is `~/rust_projects`;
    /// leaving it there would turn every tool call in these tests into a policy denial, which is
    /// a different test than the one this file is about.
    fn config(&self) -> Config {
        Config {
            policy: Policy {
                workspace_root: self.dir.path().to_path_buf(),
                ..Policy::default()
            },
            ..Config::default()
        }
    }

    /// The state the handler sees, with `model` as its model seam.
    fn state(&self, model: Arc<dyn Model>) -> ServerState {
        ServerState::discover(&self.state_root)
            .expect("state")
            .with_config(&self.config())
            .with_model(model)
            .with_events(self.events.clone())
    }

    fn agent(&self) -> PathBuf {
        self.repo.join(immortality_core::paths::AGENT_DIR)
    }

    fn worktree(&self, task: &str) -> PathBuf {
        immortality_core::worktree::WorktreeManager::new(&self.repo).path_for(task)
    }

    /// The records of one task, oldest first.
    fn records(&self, task: &str) -> Vec<JournalRecord> {
        let task = immortality_core::ids::TaskId::parse(task)
            .unwrap_or_else(|error| panic!("bad task id in the fixture: {error}"));
        JournalReader::open(self.agent(), &task)
            .expect("reader")
            .read()
            .expect("records")
            .records
            .into_iter()
            .filter(|record| record.fields.task == task)
            .collect()
    }

    /// The task states the journal announced, in order.
    fn states(&self, task: &str) -> Vec<String> {
        self.records(task)
            .iter()
            .filter(|record| record.fields.kind == Kind::TaskState)
            .filter_map(|record| {
                record
                    .event
                    .payload
                    .get("state")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            })
            .collect()
    }

    /// Wait until the journal says `wanted`, or fail with what it did say.
    fn wait_for(&self, task: &str, wanted: TaskState) -> TaskState {
        let deadline = Instant::now() + RUN_BUDGET;
        loop {
            let states = self.states(task);
            if let Some(last) = states.last()
                && last == wanted.as_str()
            {
                return wanted;
            }
            assert!(
                Instant::now() <= deadline,
                "the run of {task} never reached `{wanted}`; it announced {states:?}"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Wait until the run thread has released its slot, so a follow-up start is not refused.
    fn wait_until_idle(state: &ServerState) {
        let deadline = Instant::now() + RUN_BUDGET;
        loop {
            let idle = state.running.lock().expect("run set").is_empty();
            if idle {
                return;
            }
            assert!(Instant::now() <= deadline, "the run never released its slot");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Write `TASKS.json` with one subtask in `state`, and the acceptance file it needs.
    fn write_plan(&self, state: SubtaskState) -> Subtask {
        let mut subtask = notes_subtask(SystemClock.now());
        subtask.state = state;
        std::fs::create_dir_all(self.agent()).expect("agent dir");
        std::fs::write(
            self.agent().join("ACCEPTANCE.md"),
            "# Acceptance\n\n`true` exits zero.\n",
        )
        .expect("acceptance");
        let body = json!({ "v": SCHEMA_VERSION, "subtasks": [subtask] });
        std::fs::write(
            self.agent().join("TASKS.json"),
            serde_json::to_string_pretty(&body).expect("json"),
        )
        .expect("tasks");
        subtask
    }
}

/// A subtask whose gate is a command that always succeeds.
///
/// `true` is a classified `Read` program, so the policy gate allows it inside the worktree with
/// no escalation — which is what keeps this file about the cycle and not about approvals.
fn notes_subtask(at: Timestamp) -> Subtask {
    Subtask::new(
        1,
        "prove the cycle runs",
        "run the check and claim the subtask",
        Dod {
            statement: "the check exits zero".to_string(),
            checks: vec![DodCheck {
                id: "ok".to_string(),
                command: "true".to_string(),
                required: true,
            }],
            review: false,
        },
        at,
    )
}

fn git(repo: &Path, args: &[&str]) {
    let output = std::process::Command::new("git")
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

// ── the model ───────────────────────────────────────────────────────────────────────────

/// What the executor should do on its turn.
#[derive(Clone)]
enum Work {
    /// Claim the subtask immediately.
    Claim,
    /// Write a file on the first turn, then claim.
    WriteThenClaim,
    /// Write a file, then fail as an unreachable backend would.
    WriteThenFail,
    /// Wait until the test releases it, then claim.
    Wait(Arc<Gate>),
}

/// A latch the test opens when it wants a blocked run to continue.
#[derive(Debug, Default)]
struct Gate {
    open: Mutex<bool>,
    changed: Condvar,
}

impl Gate {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn block(&self) {
        let mut open = self.open.lock().expect("gate");
        while !*open {
            open = self.changed.wait(open).expect("gate wait");
        }
        drop(open);
    }

    fn release(&self) {
        *self.open.lock().expect("gate") = true;
        self.changed.notify_all();
    }
}

/// One model for all three roles, scripted by the system prompt it is given.
struct Scripted {
    work: Work,
    planner_calls: AtomicUsize,
    reviewer_calls: AtomicUsize,
    executor_calls: AtomicUsize,
}

impl Scripted {
    fn new(work: Work) -> Arc<Self> {
        Arc::new(Self {
            work,
            planner_calls: AtomicUsize::new(0),
            reviewer_calls: AtomicUsize::new(0),
            executor_calls: AtomicUsize::new(0),
        })
    }

    fn planner_calls(&self) -> usize {
        self.planner_calls.load(Ordering::SeqCst)
    }

    fn reviewer_calls(&self) -> usize {
        self.reviewer_calls.load(Ordering::SeqCst)
    }

    fn executor_calls(&self) -> usize {
        self.executor_calls.load(Ordering::SeqCst)
    }
}

impl std::fmt::Debug for Scripted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scripted").finish_non_exhaustive()
    }
}

impl Model for Scripted {
    fn complete(&self, request: &ModelRequest) -> Result<Reply, String> {
        // The first message is the system prompt by construction: the runner and the planner both
        // build their request with `with_system` before any user turn.
        let system = request
            .messages
            .first()
            .and_then(|message| message.content.clone())
            .unwrap_or_default();

        if system == immortality_core::planner::SYSTEM {
            self.planner_calls.fetch_add(1, Ordering::SeqCst);
            return Ok(Reply::text(PLAN_ANSWER));
        }
        if system == immortality_core::gate::REVIEW_SYSTEM {
            self.reviewer_calls.fetch_add(1, Ordering::SeqCst);
            return Ok(Reply::text(r#"{"verdict":"approve"}"#));
        }

        self.executor_calls.fetch_add(1, Ordering::SeqCst);
        match &self.work {
            Work::Claim => Ok(claim()),
            Work::WriteThenClaim => {
                if self.executor_calls() % 2 == 1 {
                    Ok(write_call("call_write"))
                } else {
                    Ok(claim())
                }
            }
            Work::Wait(gate) => {
                gate.block();
                Ok(claim())
            }
            Work::WriteThenFail => {
                // The first executor turn writes; the next one reports the backend gone. The
                // attempt is then abandoned, which is the state `start --task` has to resume.
                if self.executor_calls() == 1 {
                    Ok(write_call("call_write"))
                } else {
                    Err("connection refused: the backend is not listening".to_string())
                }
            }
        }
    }
}

fn claim() -> Reply {
    Reply::call("call_finish", FINISH_SUBTASK, r#"{"summary":"the check passes"}"#)
}

fn write_call(id: &str) -> Reply {
    Reply::call(
        id,
        "write_file",
        r#"{"path":"src/notes.txt","content":"written by the run\n"}"#,
    )
}

/// A planner answer the compiler accepts: one subtask, one real check, no ids.
const PLAN_ANSWER: &str = r##"{
  "plan": "# Plan\n\nAdd the notes file.",
  "acceptance": "`true` exits zero.",
  "subtasks": [{
    "name": "notes",
    "title": "prove the cycle runs",
    "goal": "run the check and claim the subtask",
    "dod": {
      "statement": "the check exits zero",
      "checks": [{ "id": "ok", "command": "true" }],
      "review": false
    }
  }]
}"##;

/// Call the handler the way `serve` does.
fn start_now(state: &ServerState, params: &serde_json::Value) -> Result<serde_json::Value, String> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(start::start(state, params))
        .map_err(|error| error.message)
}

// ── the happy path ──────────────────────────────────────────────────────────────────────

#[test]
fn a_goal_becomes_a_task_that_reaches_ready_for_review() {
    // The milestone's exit criterion, on the production path: one RPC call, a real worktree, the
    // cycle's own verdict. Nothing here fakes the journal, git or the gates.
    let fixture = Fixture::new();
    let model = Scripted::new(Work::WriteThenClaim);
    let state = fixture.state(model.clone());

    let answer = start_now(&state, &json!({ "project": fixture.repo, "goal": GOAL }))
        .expect("the task must be admitted");
    let task = answer["task"].as_str().expect("a task id").to_string();
    assert_eq!(answer["admitted"], true);
    assert_eq!(answer["planned"], true, "there was no plan, so one was written");
    assert_eq!(answer["state"], "queued");

    // The state directory exists before the answer, so every other command can act on the id.
    assert!(
        fixture
            .state_root
            .join("state")
            .join(answer["project"].as_str().unwrap())
            .join(&task)
            .is_dir()
    );

    fixture.wait_for(&task, TaskState::ReadyForReview);
    Fixture::wait_until_idle(&state);

    // The lifecycle, in order: the plan was loaded and approved before any work, and the verdict
    // came after the tree settled. A cycle that skipped `verifying` would have accepted the work
    // without running the task-level gates.
    let states = fixture.states(&task);
    assert_eq!(
        states,
        vec![
            "queued",
            "planning",
            "awaiting_plan_approval",
            "running",
            "verifying",
            "ready_for_review",
        ],
        "the run took an unexpected path: {states:?}"
    );

    // Both roles were actually asked. A run that reached the verdict without a reviewer means the
    // acceptance gate was skipped; one that never asked the planner means a plan came from nowhere.
    assert_eq!(model.planner_calls(), 1);
    assert_eq!(model.reviewer_calls(), 1);

    // The worktree is real and on the task's branch, which is what a human merges from.
    let worktree = fixture.worktree(&task);
    assert!(worktree.is_dir(), "the run must have a worktree");
    assert!(worktree.join("src").join("notes.txt").is_file());
    let branch = std::process::Command::new("git")
        .arg("-C")
        .arg(&worktree)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .expect("git");
    assert_eq!(
        String::from_utf8_lossy(&branch.stdout).trim(),
        format!("agent/{task}")
    );
}

#[test]
fn an_existing_plan_is_run_and_not_replaced_without_force() {
    // The roadmap's contract: a human may rewrite `TASKS.json` before the run. Regenerating it
    // because a goal was also given would silently discard exactly that edit.
    let fixture = Fixture::new();
    let subtask = fixture.write_plan(SubtaskState::Pending);
    let model = Scripted::new(Work::Claim);
    let state = fixture.state(model.clone());

    let answer = start_now(&state, &json!({ "project": fixture.repo, "goal": GOAL }))
        .expect("the plugin must be admitted");
    let task = answer["task"].as_str().expect("a task id").to_string();
    assert_eq!(answer["planned"], false, "a plan on disk must not be re-planned");
    assert_eq!(
        model.planner_calls(),
        0,
        "a model call was spent on a plan that already existed"
    );

    fixture.wait_for(&task, TaskState::ReadyForReview);
    Fixture::wait_until_idle(&state);
    // The tree the run executed is the one that was on disk, not a regenerated one.
    let text = std::fs::read_to_string(fixture.agent().join("TASKS.json")).expect("tasks");
    assert!(text.contains(&subtask.id.to_string()), "{text}");
}

#[test]
fn force_re_plans_and_spends_exactly_one_model_call() {
    let fixture = Fixture::new();
    fixture.write_plan(SubtaskState::Pending);
    let model = Scripted::new(Work::Claim);
    let state = fixture.state(model.clone());

    let answer = start_now(
        &state,
        &json!({ "project": fixture.repo, "goal": GOAL, "force": true }),
    )
    .expect("the task must be admitted");
    assert_eq!(answer["planned"], true);
    assert_eq!(model.planner_calls(), 1);
    fixture.wait_for(answer["task"].as_str().expect("id"), TaskState::ReadyForReview);
}

#[test]
fn force_without_a_goal_is_refused_before_anything_is_written() {
    // `--force` means "replace the tree", and replacing needs the new tree's source. Refusing is
    // the only honest answer; silently re-running the old plan would make the flag a no-op.
    let fixture = Fixture::new();
    fixture.write_plan(SubtaskState::Pending);
    let model = Scripted::new(Work::Claim);
    let state = fixture.state(model.clone());

    let error = start_now(&state, &json!({ "project": fixture.repo, "force": true }))
        .expect_err("force without a goal must be refused");
    assert!(error.contains("--force"), "{error}");
    assert!(error.contains("goal"), "{error}");
    assert_eq!(model.planner_calls(), 0);
    // Nothing ran: the run slot is empty and no task was minted.
    assert!(state.running.lock().expect("run set").is_empty());
}

#[test]
fn a_goal_with_no_plan_and_no_goal_is_refused_with_the_missing_piece() {
    let fixture = Fixture::new();
    let model = Scripted::new(Work::Claim);
    let state = fixture.state(model);

    let error =
        start_now(&state, &json!({ "project": fixture.repo })).expect_err("nothing to run must be refused");
    assert!(error.contains("no plan"), "{error}");
    assert!(error.contains("goal"), "{error}");
}

// ── the run slot ────────────────────────────────────────────────────────────────────────

#[test]
fn a_second_start_of_a_running_task_is_refused_and_names_the_first() {
    // The journal is single-writer, so a second run of one task would fork its chain — the one
    // unrecoverable state this design has. The refusal is what makes that unrepresentable.
    let fixture = Fixture::new();
    let gate = Gate::new();
    let model = Scripted::new(Work::Wait(Arc::clone(&gate)));
    let state = fixture.state(model);

    let answer = start_now(&state, &json!({ "project": fixture.repo, "goal": GOAL }))
        .expect("the first admission must succeed");
    let task = answer["task"].as_str().expect("a task id").to_string();
    // The run is blocked inside the model, so the slot is certainly held here.
    let error =
        start_now(&state, &json!({ "task": task })).expect_err("a second run of one task must be refused");
    assert!(error.contains("already running"), "{error}");
    assert!(error.contains(&task), "the refusal must name the task: {error}");
    assert!(
        error.contains("attach"),
        "the refusal must say where to watch: {error}"
    );

    gate.release();
    fixture.wait_for(&task, TaskState::ReadyForReview);
    Fixture::wait_until_idle(&state);
}

#[test]
fn a_finished_task_cannot_be_resumed() {
    // `ready_for_review` is a verdict, not a pause. Resuming it would be a second claim about
    // work already judged, and the lifecycle has no edge for that.
    let fixture = Fixture::new();
    fixture.write_plan(SubtaskState::Done);
    let task = write_journal(&fixture, &[("ready_for_review", TaskState::ReadyForReview)]);
    let model = Scripted::new(Work::Claim);
    let state = fixture.state(model);

    let error = start_now(&state, &json!({ "task": task })).expect_err("a finished task must not resume");
    assert!(error.contains("ready_for_review"), "{error}");
    assert!(error.contains("unfinished"), "{error}");
}

// ── resume, and the crash it exists for ────────────────────────────────────────────────

#[test]
fn a_verifying_task_owes_only_the_verdict() {
    // Every subtask is settled and the task crashed before its gates ran. The resume must *not*
    // re-drive the tree — that would run the work twice — and must reach the verdict.
    let fixture = Fixture::new();
    let subtask = fixture.write_plan(SubtaskState::Done);
    let task = write_journal(&fixture, &[("verifying", TaskState::Verifying)]);
    let model = Scripted::new(Work::Claim);
    let state = fixture.state(model.clone());

    let answer = start_now(&state, &json!({ "task": task })).expect("the resume must be admitted");
    assert_eq!(answer["resumed"], true);
    assert_eq!(answer["state"], "verifying");

    fixture.wait_for(&task, TaskState::ReadyForReview);
    Fixture::wait_until_idle(&state);
    // The settled subtask was not run again: no executor turn was spent, and the state sequence
    // goes straight from `verifying` to the verdict.
    assert_eq!(model.executor_calls(), 0);
    let states = fixture.states(&task);
    assert_eq!(
        states,
        vec!["verifying", "ready_for_review"],
        "a verifying resume must not take execution edges: {states:?}"
    );
    // The subtask is still `done` — a resumed verdict changes nothing about the work.
    let text = std::fs::read_to_string(fixture.agent().join("TASKS.json")).expect("tasks");
    assert!(text.contains(&subtask.id.to_string()), "{text}");
}

#[test]
fn an_outage_leaves_the_task_running_and_a_later_start_finishes_it() {
    // A backend that went away is a condition, not a verdict. The task must stay resumable with
    // its worktree and plan intact, and the next `start --task` must finish the job.
    let fixture = Fixture::new();
    let first = Scripted::new(Work::WriteThenFail);
    let state = fixture.state(first);

    let answer = start_now(&state, &json!({ "project": fixture.repo, "goal": GOAL }))
        .expect("the first admission must succeed");
    let task = answer["task"].as_str().expect("a task id").to_string();
    Fixture::wait_until_idle(&state);

    // The last state is `running`, and the outage is recorded once. `blocked` here would lose the
    // plan and the checkpoint over a reboot.
    let states = fixture.states(&task);
    assert_eq!(states.last().map(String::as_str), Some("running"), "{states:?}");
    let records = fixture.records(&task);
    let outages = records
        .iter()
        .filter(|record| record.fields.kind == Kind::BackendUnavailable)
        .count();
    assert_eq!(outages, 1, "the outage must be journaled exactly once");
    // The file the attempt wrote is still on disk: an outage must not roll work back.
    assert!(fixture.worktree(&task).join("src").join("notes.txt").is_file());

    // A new daemon process considers the task idle, and the resume is the same code path a
    // restart takes.
    let second_model = Scripted::new(Work::Claim);
    let second = fixture.state(second_model.clone());
    let resumed = start_now(&second, &json!({ "task": task })).expect("the resume must be admitted");
    assert_eq!(resumed["resumed"], true);
    fixture.wait_for(&task, TaskState::ReadyForReview);
    Fixture::wait_until_idle(&second);

    // The goal survived the process boundary: the resumed attempt's strategy is the task's goal.
    // Without it the fresh context would know what each subtask does and not why.
    let strategies: Vec<String> = fixture
        .records(&task)
        .iter()
        .filter(|record| record.fields.kind == Kind::AttemptStarted)
        .filter_map(|record| {
            record
                .event
                .payload
                .get("strategy")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
        .collect();
    assert!(
        strategies.iter().any(|strategy| strategy == GOAL),
        "the resumed attempt lost the goal: {strategies:?}"
    );
    assert_eq!(second_model.executor_calls(), 1);
}

#[test]
fn a_subtask_a_kill_left_running_is_reopened_instead_of_deadlocking_the_resume() {
    // The gap between the tree and the process: `kill -9` lands between the `running` write and
    // anything else, so the plan says a worker owns the subtask and no worker exists. The cycle
    // must reopen it — `next_ready` offers only `pending`, and a resume that trusted the tree
    // would report `Stuck` and idle forever, which is the failure the crash tests exist for.
    let fixture = Fixture::new();
    fixture.write_plan(SubtaskState::Running);
    let task = write_journal(&fixture, &[("running", TaskState::Running)]);
    let model = Scripted::new(Work::Claim);
    let state = fixture.state(model.clone());

    let answer = start_now(&state, &json!({ "task": task })).expect("the resume must be admitted");
    assert_eq!(answer["resumed"], true);

    fixture.wait_for(&task, TaskState::ReadyForReview);
    Fixture::wait_until_idle(&state);
    // The reopened subtask was actually executed, and the tree records the honest path: the
    // adoption announced `pending` from the fixture's `running` tree, then the pass closed it.
    assert_eq!(model.executor_calls(), 1);
    let subtask_states: Vec<String> = fixture
        .records(&task)
        .iter()
        .filter(|record| record.fields.kind == Kind::SubtaskState)
        .filter_map(|record| {
            record
                .event
                .payload
                .get("state")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
        .collect();
    assert_eq!(
        subtask_states,
        vec!["pending".to_string(), "done".to_string()],
        "the leftover `running` was not reopened: {subtask_states:?}"
    );
    let text = std::fs::read_to_string(fixture.agent().join("TASKS.json")).expect("tasks");
    // Nothing in the tree may still claim a worker.
    assert!(!text.contains("\"running\""), "{text}");
}

#[test]
fn a_torn_tail_from_a_kill_is_repaired_and_the_run_continues() {
    // The roadmap's crash test: `kill -9` mid-subtask leaves a torn last line, and the run has to
    // continue after the startup pass repairs it. The torn tail is produced the way the kernel
    // would: by truncating the file inside the last line.
    let fixture = Fixture::new();
    let first = Scripted::new(Work::WriteThenFail);
    let state = fixture.state(first);
    let answer = start_now(&state, &json!({ "project": fixture.repo, "goal": GOAL }))
        .expect("the first admission must succeed");
    let task = answer["task"].as_str().expect("a task id").to_string();
    Fixture::wait_until_idle(&state);

    tear_tail(&fixture.agent());

    // Exactly what `serve` runs before binding the socket.
    let recovered = recover(&fixture, &task);
    assert!(
        recovered.repaired(),
        "a torn tail must be repaired at startup: {recovered:?}"
    );

    let second = Scripted::new(Work::Claim);
    let resumed = fixture.state(second);
    start_now(&resumed, &json!({ "task": task })).expect("the run must resume after the repair");
    fixture.wait_for(&task, TaskState::ReadyForReview);
    Fixture::wait_until_idle(&resumed);
    // The repair did not damage the chain: the journal still reads cleanly to the new verdict.
    let outcome = JournalReader::open(
        fixture.agent(),
        &immortality_core::ids::TaskId::parse(&task).expect("id"),
    )
    .expect("reader")
    .read()
    .expect("records");
    assert!(!outcome.is_corrupt(), "{:?}", outcome.problems);
}

// ── attach ──────────────────────────────────────────────────────────────────────────────

#[test]
fn a_live_run_publishes_its_records_through_the_daemons_events() {
    // `attach` is the only window into a run that takes hours. Records must reach the broadcast as
    // they are written — not when the run ends — so the test subscribes first and then checks what
    // arrived, rather than sleeping and hoping.
    let fixture = Fixture::new();
    let model = Scripted::new(Work::WriteThenClaim);
    let state = fixture.state(model);
    let mut events = fixture.events.subscribe();

    let answer = start_now(&state, &json!({ "project": fixture.repo, "goal": GOAL }))
        .expect("the task must be admitted");
    let task = answer["task"].as_str().expect("a task id").to_string();
    fixture.wait_for(&task, TaskState::ReadyForReview);
    Fixture::wait_until_idle(&state);

    let mut kinds = Vec::new();
    let mut states = Vec::new();
    let mut saw_our_task = false;
    loop {
        match events.try_recv() {
            Ok(record) => {
                if record.fields.task.as_str() != task {
                    continue;
                }
                saw_our_task = true;
                kinds.push(record.fields.kind.as_str().to_string());
                if record.fields.kind == Kind::TaskState
                    && let Some(state) = record
                        .event
                        .payload
                        .get("state")
                        .and_then(serde_json::Value::as_str)
                {
                    states.push(state.to_string());
                }
            }
            // `Lagged` cannot happen: the receiver exists from before the first record and the
            // buffer is larger than any run's journal. Reporting it as a failure is right,
            // because a test that could not observe the stream proves nothing about it.
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(skipped)) => {
                panic!("the test fell {skipped} records behind the daemon's buffer");
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => break,
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => {
                panic!("the daemon's broadcast closed before the run finished");
            }
        }
    }

    assert!(
        saw_our_task,
        "no record of the run reached the daemon's broadcast: {kinds:?}"
    );
    // The stream covers the work, not just the admission: the run's own lifecycle and at least one
    // tool call arrived live.
    assert!(kinds.iter().any(|kind| kind == "attempt_started"), "{kinds:?}");
    assert!(
        kinds.iter().any(|kind| kind == "tool_call_completed"),
        "{kinds:?}"
    );
    assert_eq!(
        states.last().map(String::as_str),
        Some("ready_for_review"),
        "{states:?}"
    );
}

// ── helpers that write the world a crash leaves ────────────────────────────────────────

/// Append a `running`/`verifying` history to a task's journal and return the id.
///
/// The task id is minted here because a resumed task has to exist before the daemon is asked about
/// it: `owner_of` finds it from the journal records, and `start --task` refuses an id nobody owns.
fn write_journal(fixture: &Fixture, states: &[(&str, TaskState)]) -> String {
    let task = immortality_core::ids::TaskId::generate();
    // The state directory is one of the two traces a task has; the journal record is the other.
    let task_dir = fixture.state_root.join("state").join("repo").join(task.as_str());
    std::fs::create_dir_all(&task_dir).expect("task dir");
    let mut journal = Journal::open(fixture.agent(), task.clone()).expect("journal");
    journal
        .append_event(
            SystemClock.now(),
            Kind::TaskCreated,
            json!({ "title": GOAL, "goal": GOAL }),
        )
        .expect("task_created");
    for (name, state) in states {
        let stored = *state;
        assert_eq!(stored.as_str(), *name, "the fixture's name and state must agree");
        journal
            .append_event(SystemClock.now(), Kind::TaskState, json!({ "state": name }))
            .expect("task_state");
    }
    journal.close().expect("close");
    task.to_string()
}

/// Truncate the journal inside its last line, as a `kill -9` during a write would.
fn tear_tail(agent: &Path) {
    let path = agent.join("JOURNAL.jsonl");
    let text = std::fs::read_to_string(&path).expect("journal");
    let last_newline = text.trim_end_matches('\n').rfind('\n').expect("two lines");
    let keep = last_newline + 1;
    // Half of the last line's bytes, so what is left cannot parse and is not itself a record.
    let tail = text.len() - keep;
    let cut = keep + tail / 2;
    std::fs::write(&path, &text.as_bytes()[..cut]).expect("tear");
}

/// The startup pass `serve` runs before binding the socket.
fn recover(fixture: &Fixture, task: &str) -> startup::Recovered {
    let task = immortality_core::ids::TaskId::parse(task).expect("task id");
    let project = Registry::open(fixture.state_root.join(immortality_core::registry::FILE_NAME))
        .expect("registry")
        .projects()
        .first()
        .expect("a registered project")
        .clone();
    let task_dir = project.state_dir(&fixture.state_root).join(task.as_str());
    let worktree = fixture.worktree(task.as_str());
    let config = fixture.config();
    let deps = immortality_daemon::toolset::ToolsetDeps::for_task(
        &config,
        &fixture.state_root,
        &task_dir,
        &worktree,
        task.clone(),
    )
    .expect("toolset deps");
    let registry = Arc::new(immortality_daemon::toolset::build(&deps).expect("tool registry"));
    let request = startup::RecoveryRequest {
        task,
        repository: fixture.repo.clone(),
        journal_dir: fixture.agent(),
        task_dir,
        registry,
        gate: Arc::clone(&deps.gate),
        artifacts: immortality_core::artifacts::ArtifactStore::open(fixture.state_root.join("artifacts"))
            .expect("artifacts"),
        clock: Arc::new(SystemClock),
    };
    startup::recover_task(&request).expect("recovery")
}
