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
use immortality_core::journal::{Journal, JournalReader, JournalRecord, Kind, Scope};
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
    /// Run a program no table knows, then claim. The judge decides whether it may run.
    UnknownProgramThenClaim,
    /// Delegate a brief, then claim. The sub-agent tries to delegate onward first, then works.
    DelegateThenClaim,
    /// Delegate with a one-step budget to a sub-agent that never claims, then claim the parent.
    DelegateToABriefThatRunsOut,
    /// Ask the human a question, then claim. The question must reach the journal.
    EscalateThenClaim,
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
    /// Refuse an empty diff the way the real gate does.
    ///
    /// Off by default: most fixtures in this file never produce work at all (`Work::Claim` closes
    /// a subtask without touching a file), so a reviewer that refused every empty diff would fail
    /// them for a reason they are not about. The production test that *does* produce work turns
    /// this on, and that is the case where "approved an empty diff" is the bug.
    refuse_empty_diff: bool,
    planner_calls: AtomicUsize,
    reviewer_calls: AtomicUsize,
    executor_calls: AtomicUsize,
    judge_calls: AtomicUsize,
    /// Turns taken by a delegated brief, counted separately so one `Work` can script both the
    /// parent and the sub-agent from the same `Model`.
    brief_calls: AtomicUsize,
}

impl Scripted {
    fn new(work: Work) -> Arc<Self> {
        Arc::new(Self {
            work,
            refuse_empty_diff: false,
            planner_calls: AtomicUsize::new(0),
            reviewer_calls: AtomicUsize::new(0),
            executor_calls: AtomicUsize::new(0),
            judge_calls: AtomicUsize::new(0),
            brief_calls: AtomicUsize::new(0),
        })
    }

    /// The same script, refusing an empty diff.
    fn strict(work: Work) -> Arc<Self> {
        Arc::new(Self {
            work,
            refuse_empty_diff: true,
            planner_calls: AtomicUsize::new(0),
            reviewer_calls: AtomicUsize::new(0),
            executor_calls: AtomicUsize::new(0),
            judge_calls: AtomicUsize::new(0),
            brief_calls: AtomicUsize::new(0),
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

    fn judge_calls(&self) -> usize {
        self.judge_calls.load(Ordering::SeqCst)
    }

    fn brief_calls(&self) -> usize {
        self.brief_calls.load(Ordering::SeqCst)
    }
}

impl std::fmt::Debug for Scripted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scripted").finish_non_exhaustive()
    }
}

/// The reviewer's answer, optionally refusing an empty diff the way the real gate does.
///
/// A reviewer that approves anything passes even when there is nothing to review, and "nothing to
/// review" is exactly what a missing commit looks like: the work sits in `HEAD`, the diff against
/// it is empty, and the task closes having merged nothing. `strict` makes the production test fail
/// for the same reason a live run did.
fn review(request: &ModelRequest, refuse_empty_diff: bool) -> Reply {
    if refuse_empty_diff {
        let prompt = request
            .messages
            .iter()
            .filter_map(|message| message.content.as_deref())
            .collect::<Vec<_>>()
            .join("\n");
        if prompt.contains("the diff is empty") {
            return Reply::text(r#"{"verdict":"request_changes","issues":["the diff is empty"]}"#);
        }
    }
    Reply::text(r#"{"verdict":"approve"}"#)
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
            return Ok(review(request, self.refuse_empty_diff));
        }
        if system == immortality_tools::judge::SYSTEM {
            self.judge_calls.fetch_add(1, Ordering::SeqCst);
            return Ok(Reply::text(r#"{"verdict":"allow"}"#));
        }

        self.executor_calls.fetch_add(1, Ordering::SeqCst);
        // The sub-agent is told apart by its brief, not by call order: `brief_for` writes
        // `# Subtask: delegated brief`, which appears in no parent prompt. Order-based scripting
        // would pass even if the tool called the wrong agent.
        let is_brief = system.contains(BRIEF_MARKER);
        if is_brief {
            self.brief_calls.fetch_add(1, Ordering::SeqCst);
        }
        match &self.work {
            Work::WriteThenClaim => {
                if self.executor_calls() % 2 == 1 {
                    Ok(write_call("call_write"))
                } else {
                    Ok(claim())
                }
            }
            Work::UnknownProgramThenClaim => {
                if self.executor_calls() == 1 {
                    Ok(Reply::call(
                        "call_unknown",
                        "shell",
                        r#"{"command":"frobnicate --all"}"#,
                    ))
                } else {
                    Ok(claim())
                }
            }
            Work::DelegateThenClaim if is_brief => {
                // Three turns, in this order: try to delegate onward (which must be refused),
                // write a file, claim. The nested attempt is first so the refusal is the first
                // thing the sub-agent's journal records.
                match self.brief_calls() {
                    1 => Ok(Reply::call(
                        "call_nested",
                        "delegate",
                        r#"{"brief":"write the notes file","scope":["src"]}"#,
                    )),
                    2 => Ok(write_call("call_brief_write")),
                    _ => Ok(Reply::call(
                        "call_brief_finish",
                        FINISH_SUBTASK,
                        r#"{"summary":"the brief wrote src/notes.txt"}"#,
                    )),
                }
            }
            Work::DelegateToABriefThatRunsOut if is_brief => {
                // Never claims, so the brief's only way out is its step budget.
                Ok(write_call("call_brief_write"))
            }
            Work::DelegateToABriefThatRunsOut => {
                if self.executor_calls() == 1 {
                    Ok(Reply::call(
                        "call_delegate",
                        "delegate",
                        r#"{"brief":"write src/notes.txt","scope":["src"],"max_steps":1}"#,
                    ))
                } else {
                    Ok(claim())
                }
            }
            Work::DelegateThenClaim => {
                if self.executor_calls() == 1 {
                    Ok(Reply::call(
                        "call_delegate",
                        "delegate",
                        r#"{"brief":"write src/notes.txt with one line","scope":["src"]}"#,
                    ))
                } else {
                    Ok(claim())
                }
            }
            Work::Wait(gate) => {
                gate.block();
                Ok(claim())
            }
            Work::EscalateThenClaim if self.executor_calls() == 1 => Ok(ask()),
            // After the question the model finishes, which is what a run that is not blocked
            // does: the escalation is a side record, not a stop.
            Work::Claim | Work::EscalateThenClaim => Ok(claim()),
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

/// The heading `brief_for` writes and no parent prompt contains.
const BRIEF_MARKER: &str = "# Subtask: delegated brief";

fn claim() -> Reply {
    Reply::call("call_finish", FINISH_SUBTASK, r#"{"summary":"the check passes"}"#)
}

/// The one `escalate` call in this file: a question that neither the plan nor the model answers.
fn ask() -> Reply {
    Reply::call(
        "call_escalate",
        "escalate",
        r#"{"key":"network:example.com","question":"May I fetch the docs?","kind":"external_action"}"#,
    )
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
    let model = Scripted::strict(Work::WriteThenClaim);
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

    // And the work is *on* that branch, not merely in the worktree. This is the half that was
    // missing: the cycle closed subtasks without committing them, so `agent/<task>` stayed on its
    // base commit. Every run measured — `word-count`, `word-count-fresh`, `kill-test` — had the
    // work present only as uncommitted changes and in `agent-ckpt/<task>`, which made
    // `immortality merge` report success having merged nothing.
    let ahead = std::process::Command::new("git")
        .arg("-C")
        .arg(&fixture.repo)
        .args(["rev-list", "--count", &format!("HEAD..agent/{task}")])
        .output()
        .expect("git");
    let ahead = String::from_utf8_lossy(&ahead.stdout).trim().to_string();
    assert!(
        ahead != "0",
        "the closed subtask's work must be committed to `agent/{task}`; without it `merge` has \
         nothing to land and says so wrongly"
    );
}

#[test]
fn an_unknown_program_is_decided_by_the_judge_on_the_production_path() {
    // D-111's whole point, through `start::start`: a program in no table used to stop the attempt
    // and wait for a human. Here the model is asked, answers `allow`, and the command runs — and
    // the verdict it cost is in the journal, so a restart does not pay again.
    let fixture = Fixture::new();
    let model = Scripted::new(Work::UnknownProgramThenClaim);
    let state = fixture.state(model.clone());
    fixture.write_plan(SubtaskState::Pending);

    let answer = start_now(&state, &json!({ "project": fixture.repo, "goal": GOAL }))
        .expect("the task must be admitted");
    let task = answer["task"].as_str().expect("a task id").to_string();
    fixture.wait_for(&task, TaskState::ReadyForReview);
    Fixture::wait_until_idle(&state);

    assert_eq!(
        model.judge_calls(),
        1,
        "the judge was not wired into the production gate, so the unknown program escalated"
    );
    let record = fixture
        .records(&task)
        .into_iter()
        .find(|record| record.fields.kind == Kind::JudgeVerdict)
        .expect("the verdict must be in the journal, or every restart pays again");
    assert_eq!(
        record.event.payload["command"],
        json!("frobnicate --all"),
        "the record has to name the command a human can check"
    );
    assert_eq!(record.event.payload["verdict"], json!("allow"));
}

#[test]
fn a_delegated_brief_gets_its_own_journal_and_cannot_delegate_onward() {
    // The production path for `delegate`, through `start::start`. Three things are asserted that
    // nothing cheaper can prove:
    //
    // 1. the slot the tool was built over is filled — `toolset::build` runs before the `Runner`
    //    exists, so a build that forgot to install one would leave `NoSubAgent` and the model
    //    would get "no sub-agent wired in" instead of a report;
    // 2. the sub-agent writes its own journal under `delegations/<del_id>/`, not into the task's
    //    journal, which the parent holds open for the whole call;
    // 3. a sub-agent that calls `delegate` is refused, rather than recurring until budgets run out.
    let fixture = Fixture::new();
    fixture.write_plan(SubtaskState::Pending);
    let model = Scripted::new(Work::DelegateThenClaim);
    let state = fixture.state(model.clone());

    let answer = start_now(&state, &json!({ "project": fixture.repo, "goal": GOAL }))
        .expect("the task must be admitted");
    let task = answer["task"].as_str().expect("a task id").to_string();
    let project = answer["project"].as_str().expect("a project id").to_string();
    fixture.wait_for(&task, TaskState::ReadyForReview);
    Fixture::wait_until_idle(&state);

    // Exactly three: the refused nested `delegate`, the write, the claim. Zero means the slot was
    // never filled; four or more means the nested call was allowed to start a second sub-agent.
    assert_eq!(
        model.brief_calls(),
        3,
        "the sub-agent's turns are not the three the script expects"
    );

    let task_id = immortality_core::ids::TaskId::parse(&task).expect("task id");
    let store = immortality_core::delegation::DelegationStore::for_task(
        &fixture.state_root.join("state").join(&project),
        &task_id,
    );
    let reports = store.list().expect("the report directory must exist");
    assert_eq!(reports.len(), 1, "one delegation was asked for: {reports:?}");
    let report = &reports[0];
    assert_eq!(report.finish, immortality_core::delegation::Finish::Done);
    assert_eq!(report.steps, 3);
    assert!(
        report.report.outcome.contains("src/notes.txt"),
        "the parent must inherit what the brief achieved: {:?}",
        report.report
    );
    assert_eq!(
        report.report.files_touched,
        vec![std::path::PathBuf::from("src/notes.txt")],
        "the files the brief wrote are the report's most useful field"
    );

    // Its own journal, beside the report and under the same id.
    let brief_dir = store.dir().join(report.id.as_str());
    let brief_journal = brief_dir.join("JOURNAL.jsonl");
    assert!(
        brief_journal.is_file(),
        "the sub-agent wrote no journal of its own: {} is missing",
        brief_journal.display()
    );
    let brief_records = JournalReader::open(&brief_dir, &task_id)
        .expect("brief journal")
        .read()
        .expect("brief records")
        .records;
    // The nested `delegate` was refused, and the attempt went on to do the work.
    let refusal = brief_records
        .iter()
        .find(|record| record.fields.kind == Kind::ToolCallFailed)
        .expect("the nested delegate must be recorded as a failed call");
    assert!(
        refusal.event.payload.to_string().contains("nested_delegation"),
        "the refusal has to name the reason: {}",
        refusal.event.payload
    );
    assert!(
        brief_records
            .iter()
            .any(|record| record.fields.kind == Kind::ToolCallCompleted),
        "the brief did the work after the refusal"
    );

    // The task's own journal must not contain the sub-agent's steps: one writer per segment, and
    // the parent's recovery sees a delegation as one call.
    let parent = fixture.records(&task);
    assert!(
        parent
            .iter()
            .any(|record| record.fields.kind == Kind::ToolCallIntent
                && record.event.payload["tool"] == json!("delegate")),
        "the parent's journal must record the delegation it asked for"
    );
    assert!(
        !parent.iter().any(|record| {
            record.fields.kind == Kind::ToolCallCompleted
                && record.event.payload["call"] == json!("call_brief_write")
        }),
        "the sub-agent's calls leaked into the task journal"
    );
}

#[test]
fn a_question_asked_by_the_model_reaches_the_journal_it_is_read_from() {
    // The 0.3 criterion is read out of the journal as `escalations_total`, and before this the
    // production toolset wrote the question to the queue file and told the journal *nothing*: the
    // number was zero not because nobody asked, but because the record was never written.
    //
    // Three things have to hold at once for the question to be real, and the test asserts each:
    //
    // 1. the record is in the task's journal, so `immortality metrics` can count it;
    // 2. it is there *once*, and after `tool_call_completed` rather than before the call that
    //    asked — a record written from the tool would be a second writer on the segment the
    //    runner holds open, which duplicates `seq` and breaks the chain (D-117);
    // 3. the question is durably in the queue, so a human can actually answer it.
    let fixture = Fixture::new();
    fixture.write_plan(SubtaskState::Pending);
    let model = Scripted::new(Work::EscalateThenClaim);
    let state = fixture.state(model);

    let answer = start_now(&state, &json!({ "project": fixture.repo, "goal": GOAL }))
        .expect("the task must be admitted");
    let task = answer["task"].as_str().expect("a task id").to_string();
    let project = answer["project"].as_str().expect("a project id").to_string();
    fixture.wait_for(&task, TaskState::ReadyForReview);
    Fixture::wait_until_idle(&state);

    let records = fixture.records(&task);
    // The journal is not corrupt: a second writer would have made the reader say so.
    let opened: Vec<&JournalRecord> = records
        .iter()
        .filter(|record| record.fields.kind == Kind::EscalationOpened)
        .collect();
    assert_eq!(
        opened.len(),
        1,
        "the question must be in the journal exactly once; kinds: {:?}",
        records
            .iter()
            .map(|record| record.fields.kind)
            .collect::<Vec<_>>()
    );
    let opened_at = records
        .iter()
        .position(|record| record.fields.kind == Kind::EscalationOpened)
        .expect("the opened record");
    let completed_at = records
        .iter()
        .position(|record| {
            record.fields.kind == Kind::ToolCallCompleted
                && record.event.payload["call"] == json!("call_escalate")
        })
        .expect("the escalate call completed");
    assert!(
        opened_at > completed_at,
        "the record is written by the journal's owner after the call, not by the tool: \
         opened at {opened_at}, completed at {completed_at}"
    );
    assert_eq!(opened[0].event.payload["key"], json!("network:example.com"));

    // The metrics a reader of the criterion sees.
    let report = immortality_core::metrics::compute(&task, &records);
    assert_eq!(
        report.escalations_total, 1,
        "`escalations_total` is the number the milestone is read from: {report:?}"
    );

    // And the human can answer it, because the queue file is the working set.
    let task_id = immortality_core::ids::TaskId::parse(&task).expect("task id");
    let store = immortality_core::escalation::EscalationStore::open(
        &fixture
            .state_root
            .join("state")
            .join(&project)
            .join(task_id.as_str()),
        task_id,
    )
    .expect("the queue directory");
    assert_eq!(
        store.open_entries().count(),
        1,
        "the question is not in the queue, so it cannot be answered"
    );
}

#[test]
fn a_brief_that_spends_its_steps_is_reported_as_exhausted_not_done() {
    // The one number the sub-agent owns is its step budget, and the parent reads the outcome to
    // decide whether to retry with a bigger brief or do the work itself. Reporting `done` because
    // the loop stopped rather than because the model claimed would make that decision blind — and
    // the parent's own gates would still be green, since the work the brief did was real.
    //
    // Driven through the production path, because the mapping lives in the daemon's `SubAgent`
    // implementation and nowhere else: a unit test of the mapping would not prove that the tool
    // receives what the daemon produced.
    let fixture = Fixture::new();
    fixture.write_plan(SubtaskState::Pending);
    let model = Scripted::new(Work::DelegateToABriefThatRunsOut);
    let state = fixture.state(model);

    let answer = start_now(&state, &json!({ "project": fixture.repo, "goal": GOAL }))
        .expect("the task must be admitted");
    let task = answer["task"].as_str().expect("a task id").to_string();
    let project = answer["project"].as_str().expect("a project id").to_string();
    fixture.wait_for(&task, TaskState::ReadyForReview);
    Fixture::wait_until_idle(&state);

    let task_id = immortality_core::ids::TaskId::parse(&task).expect("task id");
    let store = immortality_core::delegation::DelegationStore::for_task(
        &fixture.state_root.join("state").join(&project),
        &task_id,
    );
    let reports = store.list().expect("reports");
    assert_eq!(reports.len(), 1, "{reports:?}");
    let report = &reports[0];
    assert_eq!(
        report.finish,
        immortality_core::delegation::Finish::Exhausted,
        "a brief that never claimed was reported as finished: {report:?}"
    );
    assert_eq!(
        report.max_steps, 1,
        "the call's budget is what the report must record"
    );
    assert_eq!(report.steps, 1, "one step was all it was allowed");
    assert!(
        report.error.as_deref().is_some_and(|why| why.contains("step")),
        "the parent needs to be told the brief ran out of steps: {:?}",
        report.error
    );

    // And the brief ran no gates of its own. A brief works in the parent's worktree, so evaluating
    // a `DoD` here would spend the parent's acceptance commands on a half-finished tree — and the
    // verdict would be about the whole worktree, including work the parent did before delegating.
    // `gate_result` in the brief's journal is that mistake's fingerprint.
    let brief_records = JournalReader::open(store.dir().join(report.id.as_str()), &task_id)
        .expect("brief journal")
        .read()
        .expect("brief records")
        .records;
    assert!(
        !brief_records
            .iter()
            .any(|record| record.fields.kind == Kind::GateResult),
        "the brief evaluated gates: {:?}",
        brief_records
            .iter()
            .filter(|record| record.fields.kind == Kind::GateResult)
            .map(|record| record.event.payload.clone())
            .collect::<Vec<_>>()
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
fn a_resumed_subtask_neither_reuses_an_attempt_number_nor_restarts_its_budget() {
    // The defect a live `kill -9` exposed. `TASKS.json` carries an `attempts` field that
    // production never appended to — `cache::derived` fills `attempts.json`, a different file — so
    // the cycle began at attempt 1 with the whole budget on every process start. A journal from
    // that run holds two `attempt_started` records under one key and one of them never closed.
    //
    // The fixture journals two spent attempts and leaves the second open, which is the shape a
    // crash mid-attempt leaves behind. After the resume the third attempt must be numbered 3, not
    // 1, and the budget must arrive with two of three attempts already gone.
    let fixture = Fixture::new();
    // `false` is a classified `Read` program, so a red gate costs no escalation — the subtask
    // simply never passes and every attempt spends the budget. That is what makes the second half
    // of this test observable: a claim that failed is a charged attempt.
    let mut subtask = fixture.write_plan(SubtaskState::Running);
    subtask.dod.checks[0].command = "false".to_string();
    std::fs::write(
        fixture.agent().join("TASKS.json"),
        serde_json::to_string_pretty(&json!({ "v": SCHEMA_VERSION, "subtasks": [subtask] })).expect("json"),
    )
    .expect("tasks");
    let task = write_journal(&fixture, &[("running", TaskState::Running)]);
    {
        let mut journal = Journal::open(
            fixture.agent(),
            immortality_core::ids::TaskId::parse(&task).expect("task id"),
        )
        .expect("journal");
        let mut spend = |number: u32, kind: Kind| {
            journal
                .append_scoped(
                    SystemClock.now(),
                    kind,
                    json!({ "attempt": format!("att_{number}"), "number": number }),
                    &Scope::attempt(subtask.id.clone(), number),
                )
                .expect("attempt record");
        };
        spend(1, Kind::AttemptStarted);
        spend(1, Kind::AttemptFailed);
        spend(2, Kind::AttemptStarted);
        journal.close().expect("close");
    }
    let model = Scripted::new(Work::Claim);
    let state = fixture.state(model.clone());

    start_now(&state, &json!({ "task": task })).expect("the resume must be admitted");
    Fixture::wait_until_idle(&state);

    let records = fixture.records(&task);
    let numbers: Vec<u64> = records
        .iter()
        .filter(|record| record.fields.kind == Kind::AttemptStarted)
        .filter_map(|record| record.fields.attempt.map(u64::from))
        .collect();
    // Attempt 1 failed and attempt 2 was left open by the crash, so the resume starts at 3 and
    // gets one more. A process that began again at 1 would produce a repeated key — the live
    // journal held `[1, 2, 1]`, and the next attempt would collide with the open one.
    assert_eq!(
        numbers,
        vec![1, 2, 3, 4],
        "a resumed subtask must continue counting attempts: {numbers:?}"
    );
    // One of the three attempts was charged before the crash, and the open one is free (D-027:
    // recovery's closure says `cause: recovery`). Two were left, and exactly two ran. A restarted
    // budget would have run three, and the red gate would have driven the numbers past 4.
    assert_eq!(
        model.executor_calls(),
        2,
        "the budget must survive the process boundary"
    );
    let states: Vec<String> = records
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
        states.last().map(String::as_str),
        Some("blocked"),
        "the last attempt was spent, so the subtask is exhausted: {states:?}"
    );
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
        &fixture.repo,
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
