//! A whole task, from `TASKS.json` to `ready_for_review`, driven in microseconds.
//!
//! # What this file is for
//!
//! The M4 exit criterion is not "the runner has tests" but "the simulator runs a full cycle —
//! plan, subtasks, gates, `ready_for_review` — and every detector fires on its own scenario".
//! Every piece of that cycle already has its own tests; what none of them can catch is the seam
//! between the pieces: a restart that leaves an attempt open, a gate verdict written before the
//! attempt closes, a `subtask_state` record that names a state the tree never received.
//!
//! So this is the integration test of the loop as a loop. It drives the real
//! [`AttemptSession`], the real `core::gate::evaluate`, the real `core::plan` writes and the real
//! `core::scheduler` decisions, and it fakes only what a test must fake: the model, the clock,
//! git, and the two gate seams (`Gates` and `Reviewer`) that would otherwise run a compiler and a
//! second model call.
//!
//! # Why the clock is fake and the wall clock is measured
//!
//! Attempt budgets are 45 minutes and a scenario has to finish in under 30 seconds. The
//! [`FixedClock`] makes the first possible without the second; the assertion at the end of each
//! test makes the second a fact rather than a hope, because a simulator that got slower is how a
//! per-iteration feedback loop turns into a suite nobody runs.
//!
//! # Why the journal is compared with a golden
//!
//! The journal *is* the contract: recovery reads it after a crash, and a record whose shape
//! drifted is a record recovery misreads. A golden makes the whole sequence — order, scope,
//! payload, relative time — reviewable in a diff instead of implied by fifteen assertions. The
//! one thing it cannot contain is the attempt id, which is a ULID by design; [`normalise`]
//! replaces those with their order of appearance and says so.
//!
//! # Why the crash scenarios cut lines out of a real journal
//!
//! A crash leaves the file with a prefix of its records, so that is exactly what the tests
//! simulate: run the scenario, stop the process at a record boundary, hand the surviving journal
//! to `core::recover`. Building a `ReadOutcome` by hand would test the planner against what this
//! file believes the writer produces, which is the bug the planner exists to catch.

// Integration tests: a failed assertion should name the exact line, and there is no caller to
// propagate to.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use immortality_core::artifacts::ArtifactStore;
use immortality_core::config::{Detectors, Policy};
use immortality_core::context::{Budget, CharsCounter};
use immortality_core::domain::{Dod, DodCheck, Subtask, SubtaskState, TaskState, Timestamp};
use immortality_core::gate::{ChangedFile, Diff, FileStatus, Gates, ReviewRequest, Reviewer};
use immortality_core::ids::{SubtaskId, TaskId};
use immortality_core::journal::{Journal, JournalReader, JournalRecord, Kind, ReadOutcome};
use immortality_core::plan::{Plan, SCHEMA_VERSION};
use immortality_core::protocol::Probe as ProbeAnswer;
use immortality_core::recover::{
    self, LastCheckpoint, PersistedState, Prober, RecoveryInput, RecoveryStep, Verdict, WorkspaceFacts,
};
use immortality_core::restart::{Checkpoints, Snapshot, Stop};
use immortality_core::time::{Clock as _, FixedClock};
use immortality_daemon::cycle;
use immortality_daemon::runner::{AttemptLimits, FINISH_SUBTASK, Model, Reply, Runner};
use immortality_llm::wire::ChatRequest;
use immortality_tools::{PolicyGate, ReadFile, Registry, WriteFile};

/// The task every scenario uses. Fixed so that the golden is stable.
const TASK: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAV";
/// The subtask every scenario starts from.
const SUBTASK: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAA";
/// How much wall time one model call is pretended to take.
///
/// Ninety seconds is a plausible prefill and completion for a 64K prompt, and it makes the
/// golden's `dt_ms` column show which records belong to which step. It also keeps a whole
/// scenario well inside the 45-minute attempt budget, so no scenario ends on time by accident.
const STEP_WALL_TIME: Duration = Duration::from_secs(90);
/// A ceiling on how long one scenario may take in real time.
const SCENARIO_BUDGET: Duration = Duration::from_secs(30);

// ── the world ────────────────────────────────────────────────────────────────────────────

/// A task directory with a plan in it, and the clock everything is measured on.
struct Sim {
    /// Kept for its `Drop`: the temp directory lives as long as the scenario.
    _dir: tempfile::TempDir,
    root: PathBuf,
    /// The directory the policy gate confines reads to: the repository, its worktrees, and
    /// whatever else the human keeps beside them.
    workspace: PathBuf,
    worktree: PathBuf,
    task: TaskId,
    clock: Arc<FixedClock>,
}

impl Sim {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("repo");
        // The worktree is a *sibling* of the repository, and both live under the workspace root,
        // because that is how `WorktreeManager` lays them out (`<parent>/<name>-agent/<task>`) and
        // how the policy gate is built in production. Putting it outside `workspace_root` would
        // make every `read_file` a policy denial, and the scenario would be testing the denial
        // path instead of the loop.
        let workspace = dir.path().join("workspace");
        let worktree = workspace.join("repo-agent").join(TASK);
        std::fs::create_dir_all(root.join(".agent")).expect("agent dir");
        std::fs::create_dir_all(&worktree).expect("worktree");
        std::fs::write(
            root.join(".agent").join("ACCEPTANCE.md"),
            "# acceptance\ncargo test\n",
        )
        .expect("acceptance");
        let clock = Arc::new(FixedClock::fixture());
        let sim = Self {
            _dir: dir,
            root,
            workspace,
            worktree,
            task: TaskId::from_suffix(TASK),
            clock,
        };
        sim.write_plan(&[parser_subtask()]);
        sim
    }

    /// Write `TASKS.json`, the file `Plan::load` reads.
    fn write_plan(&self, subtasks: &[Subtask]) {
        let body = serde_json::json!({ "v": SCHEMA_VERSION, "subtasks": subtasks });
        std::fs::write(
            self.root.join(".agent").join("TASKS.json"),
            serde_json::to_string_pretty(&body).expect("json"),
        )
        .expect("write plan");
    }

    fn plan(&self) -> Plan {
        Plan::load(&self.root, 2).expect("the scenario's plan must be loadable")
    }

    fn journal_dir(&self) -> PathBuf {
        self.root.join(".agent")
    }

    fn journal(&self) -> Journal {
        let mut journal = Journal::open(self.journal_dir(), self.task.clone()).expect("journal");
        // Set before anything is written, so the first record is already on the scenario's
        // clock: the runner re-installs it on every attempt, but the lifecycle records the
        // driver writes happen outside an attempt.
        journal.set_clock(Arc::clone(&self.clock) as Arc<dyn immortality_core::time::Clock>);
        journal
    }

    /// The runner the daemon would build, with the seams a test can control.
    ///
    /// The gate world is a parameter rather than a field on [`Sim`]: most scenarios are about the
    /// loop, and one that cares which commands ran must be able to say so without every other
    /// scenario carrying the recorder.
    fn runner(&self, model: Arc<dyn Model>, gates: Arc<dyn Gates + Send + Sync>) -> Runner {
        let gate = PolicyGate::with_defaults(
            &self.worktree,
            Policy {
                workspace_root: self.workspace.clone(),
                ..Policy::default()
            },
        )
        .expect("policy gate");
        Runner {
            model,
            model_name: "sim-model".to_string(),
            registry: Arc::new(
                Registry::new(vec![Arc::new(WriteFile), Arc::new(ReadFile)]).expect("registry"),
            ),
            artifacts: ArtifactStore::open(self.root.join("artifacts")).expect("artifacts"),
            gate: Arc::new(gate),
            system_prompt: "You are an agent.".to_string(),
            max_tool_output_bytes: 8192,
            max_completion_tokens: 8192,
            enable_thinking: false,
            budget: Budget {
                soft: 32_000,
                hard: 64_000,
            },
            limits: AttemptLimits {
                max_steps: 12,
                max_tokens: 100_000,
                stall_steps: 4,
                max_wall_time: Duration::from_mins(45),
            },
            counter: Arc::new(CharsCounter::default()),
            task: self.task.clone(),
            // Small thresholds: the scenario is about the *sequence* a detector produces, not
            // about the default eight repeats, and a golden with eight identical tool calls in
            // it buries the interesting record.
            detectors: Detectors {
                exact_repeat_warn: 1,
                exact_repeat_strategy: 2,
                exact_repeat_stop: 3,
                ..Detectors::default()
            },
            max_subtask_depth: 2,
            delegation_depth: 0,
            gates: Some(gates),
            reviewer: Some(Arc::new(FakeReviewer)),
            checkpoints: Some(Arc::new(FakeCheckpoints::default())),
            clock: Arc::clone(&self.clock) as Arc<dyn immortality_core::time::Clock>,
            // The simulator's registry has no `escalate`, so there is nothing to collect and
            // nothing to drain.
            escalations: Vec::new(),
        }
    }
}

/// The one subtask the plan starts with.
fn parser_subtask() -> Subtask {
    let at = FixedClock::fixture().now();
    let mut subtask = Subtask::new(
        1,
        "parse trailing commas",
        "accept a trailing comma in argument lists",
        Dod {
            statement: "trailing commas parse".to_string(),
            checks: vec![
                DodCheck {
                    id: "build".to_string(),
                    command: "cargo build".to_string(),
                    required: true,
                    timeout_secs: None,
                },
                DodCheck {
                    id: "test".to_string(),
                    command: "cargo test".to_string(),
                    required: true,
                    timeout_secs: None,
                },
            ],
            review: true,
        },
        at,
    );
    subtask.id = SubtaskId::from_suffix(SUBTASK);
    subtask
}

// ── the fakes ────────────────────────────────────────────────────────────────────────────

/// A model that answers from a script and spends [`STEP_WALL_TIME`] per call.
///
/// The clock advance is not decoration: it is what makes the journal's `dt_ms` column meaningful
/// and what would end an attempt on its 45-minute budget if a scenario scripted enough steps.
#[derive(Debug)]
struct Script {
    replies: Mutex<VecDeque<Reply>>,
    clock: Arc<FixedClock>,
    calls: Mutex<u32>,
}

impl Script {
    fn new(replies: Vec<Reply>, clock: Arc<FixedClock>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies.into()),
            clock,
            calls: Mutex::new(0),
        })
    }

    fn calls(&self) -> u32 {
        self.calls.lock().map(|count| *count).unwrap_or_default()
    }
}

impl Model for Script {
    fn complete(&self, _request: &ChatRequest) -> Result<Reply, String> {
        self.clock
            .advance(chrono::Duration::from_std(STEP_WALL_TIME).expect("a short step"));
        if let Ok(mut calls) = self.calls.lock() {
            *calls += 1;
        }
        self.replies
            .lock()
            .ok()
            .and_then(|mut queue| queue.pop_front())
            .ok_or_else(|| "the scenario scripted no reply for this step".to_string())
    }
}

/// The gate world: a fixed diff, every check green.
#[derive(Debug, Default)]
struct FakeGates;

impl FakeGates {
    const fn green() -> Self {
        Self
    }
}

impl Gates for FakeGates {
    fn diff(&self, _base: Option<&str>) -> std::result::Result<Diff, String> {
        Ok(Diff {
            files: vec![ChangedFile {
                path: PathBuf::from("src/parser.rs"),
                status: FileStatus::Added,
                binary: false,
                added_lines: vec!["pub fn parse() {}".to_string()],
            }],
            text: "--- /dev/null\n+++ b/src/parser.rs\n+pub fn parse() {}\n".to_string(),
            truncated: false,
        })
    }

    fn authorise(&self, _command: &str) -> std::result::Result<(), String> {
        Ok(())
    }

    fn run(
        &self,
        _command: &str,
        _timeout: Option<std::time::Duration>,
    ) -> std::result::Result<immortality_core::gate::CommandRun, String> {
        Ok(immortality_core::gate::CommandRun {
            exit_code: Some(0),
            timed_out: false,
            duration_ms: 1_500,
            output_tail: "ok".to_string(),
        })
    }
}

/// A reviewer that approves. The rejection path has its own tests; here the point is that the
/// cycle reaches `ready_for_review`.
#[derive(Debug)]
struct FakeReviewer;

impl Reviewer for FakeReviewer {
    fn review(&self, _request: &ReviewRequest) -> std::result::Result<String, String> {
        Ok(r#"{"verdict":"approve"}"#.to_string())
    }
}

/// Git, without a repository.
///
/// Shas are generated in sequence so that a golden shows *which* checkpoint a record names
/// without depending on what a real repository would hash.
#[derive(Debug, Default)]
struct FakeCheckpoints {
    next: Mutex<u32>,
    restored: Mutex<Vec<String>>,
}

impl Checkpoints for FakeCheckpoints {
    fn snapshot(&self, _reason: &str) -> std::result::Result<Option<Snapshot>, String> {
        // Scoped, so the guard is released before the result is built: an early drop of a
        // `MutexGuard` would not fail here, but the next `snapshot` would deadlock.
        let number = {
            let mut next = self.next.lock().map_err(|error| error.to_string())?;
            *next += 1;
            *next
        };
        Ok(Some(Snapshot {
            sha: format!("{number:040x}"),
            branch: format!("agent/tsk_{TASK}"),
        }))
    }

    fn restore(&self, sha: &str) -> std::result::Result<(), String> {
        self.restored
            .lock()
            .map_err(|error| error.to_string())?
            .push(sha.to_string());
        Ok(())
    }

    /// A commit to the task branch, with a SHA generated the same way a snapshot's is.
    ///
    /// The simulator is about the cycle's decisions, so the message is all a golden can assert;
    /// the SHA exists because a caller may journal it.
    fn commit(&self, message: &str) -> std::result::Result<Option<String>, String> {
        let number = {
            let mut next = self.next.lock().map_err(|error| error.to_string())?;
            *next += 1;
            *next
        };
        let _ = message;
        Ok(Some(format!("{number:040x}")))
    }
}

// ── the cycle ────────────────────────────────────────────────────────────────────────────

/// What a full cycle ended with.
#[derive(Debug)]
struct CycleEnd {
    state: TaskState,
    stop: Stop,
    /// How many times the model was asked. A cycle that reached `ready_for_review` without
    /// spending a step would be a cycle that did not run the loop.
    model_calls: u32,
}

/// Run the daemon's scheduler loop over `plan`, until the task reaches a verdict.
///
/// This is the piece M5 will own. It lives here so that M4's exit criterion has a test now, and
/// it deliberately contains no policy of its own: every decision is delegated to
/// `scheduler::next`, `AttemptSession::drive`, `Plan::set_state` and `core::gate::evaluate`, and
/// the only thing this function adds is the *order* the design fixes — tree first, journal second.
fn run_cycle(sim: &Sim, script: Vec<Reply>) -> CycleEnd {
    run_cycle_with(
        sim,
        script,
        Arc::new(FakeGates::green()),
        cycle::Limits {
            attempts: 3,
            restarts: 2,
            max_depth: 2,
        },
    )
}

/// The same driver, with the gate world and the budgets the caller chooses.
fn run_cycle_with(
    sim: &Sim,
    script: Vec<Reply>,
    gates: Arc<dyn Gates + Send + Sync>,
    limits: cycle::Limits,
) -> CycleEnd {
    let model = Script::new(script, Arc::clone(&sim.clock));
    let runner = sim.runner(Arc::clone(&model) as Arc<dyn Model>, gates);
    let mut journal = sim.journal();
    let mut plan = sim.plan();

    // The loop itself is `immortality_daemon::cycle::run` — the same function the daemon calls
    // for `immortality start`. That is the point of this file after M5.8: a scenario that passes
    // here is a statement about the shipped cycle, not about a copy of it kept beside the tests.
    // What stays local is the fake world (model, clock, git, gates), which is what makes a run
    // take microseconds.
    let outcome = cycle::run(
        &runner,
        &mut plan,
        &mut journal,
        &cycle::Start::resuming(TaskState::Queued, "extend the existing parser", None),
        limits,
    )
    .expect("the cycle writes its own records");
    journal.close().expect("close journal");

    CycleEnd {
        state: outcome.state,
        stop: outcome.stop,
        model_calls: model.calls(),
    }
}

// ── the scenario ────────────────────────────────────────────────────────────────────────

/// The replies a scenario runs, in order, grouped by the attempt that consumes them.
///
/// Attempt 1 writes the file once, then reads it three times with identical arguments: D1's
/// exact-repeat ladder reaches its stop threshold at the third repeat, so the attempt ends
/// `Looped` and the session restarts it — snapshot, journal, rollback. Attempt 2 does the work
/// and claims, and the gates close it.
fn scenario_script() -> Vec<Reply> {
    let write = Reply::call(
        "call_write",
        "write_file",
        r#"{"path":"src/parser.rs","content":"pub fn parse() {}\n"}"#,
    );
    let read = Reply::call("call_read", "read_file", r#"{"path":"src/parser.rs"}"#);
    vec![
        // ── attempt 1 ──
        write.clone(),
        read.clone(),
        read.clone(),
        read,
        // ── attempt 2, after the rollback ──
        write,
        Reply::call(
            "call_finish",
            FINISH_SUBTASK,
            r#"{"summary":"trailing commas parse"}"#,
        ),
    ]
}

fn records(dir: &Path, task: &TaskId) -> ReadOutcome {
    JournalReader::open(dir, task)
        .expect("reader")
        .read()
        .expect("records")
}

// ── goldens ─────────────────────────────────────────────────────────────────────────────

/// The journal as a stable, reviewable text: offsets instead of dates, ordered ids instead of
/// ULIDs.
///
/// Two things are replaced and nothing else. The `ts` field becomes `dt_ms`, milliseconds since
/// the scenario started, because the absolute date is a fixture nobody should have to re-read in
/// a diff and the *distribution* of records over time is a fact worth seeing. An attempt id
/// becomes `att_1`, `att_2`, … in order of appearance, because [`Attempt::start`] mints a ULID
/// from the system clock and the OS entropy pool — the only non-deterministic thing in the whole
/// run, and a value no correct behaviour depends on.
fn normalise(outcome: &ReadOutcome, origin: Timestamp) -> String {
    let mut ids: Vec<String> = Vec::new();
    let mut rows = Vec::new();
    for record in &outcome.records {
        let mut row = serde_json::Map::new();
        row.insert("seq".into(), serde_json::json!(record.fields.seq));
        row.insert(
            "dt_ms".into(),
            serde_json::json!(record.fields.at.signed_duration_since(origin).num_milliseconds()),
        );
        row.insert("kind".into(), serde_json::json!(record.fields.kind.as_str()));
        if let Some(subtask) = &record.fields.subtask {
            row.insert("subtask".into(), serde_json::json!(subtask.to_string()));
        }
        if let Some(attempt) = record.fields.attempt {
            row.insert("attempt".into(), serde_json::json!(attempt));
        }
        row.insert(
            "payload".into(),
            order_attempt_ids(&record.event.payload, &mut ids),
        );
        rows.push(serde_json::Value::Object(row));
    }
    immortality_testkit::golden::canonical_json(&serde_json::Value::Array(rows))
}

/// Replace every attempt ULID in `value` with its index in `seen`.
fn order_attempt_ids(value: &serde_json::Value, seen: &mut Vec<String>) -> serde_json::Value {
    match value {
        serde_json::Value::String(text) if text.starts_with("att_") => {
            let found = seen.iter().position(|known| known == text);
            let index = found.unwrap_or_else(|| {
                seen.push(text.clone());
                seen.len() - 1
            });
            serde_json::json!(format!("att_{}", index + 1))
        }
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), order_attempt_ids(value, seen)))
                .collect(),
        ),
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(|item| order_attempt_ids(item, seen)).collect())
        }
        other => other.clone(),
    }
}

// ── crash injection ────────────────────────────────────────────────────────────────────

/// Cut the journal so that it holds only the records up to and including the `occurrence`-th
/// record of `kind`.
///
/// Returns the number of records that survived. This is what a crash at a record boundary
/// leaves: a chain that is intact, a sequence that starts at zero, and a tail that stops
/// mid-run.
fn die_after(dir: &Path, kind: Kind, occurrence: usize) -> usize {
    let path = dir.join("JOURNAL.jsonl");
    let text = std::fs::read_to_string(&path).expect("journal");
    let lines: Vec<&str> = text.lines().collect();
    let mut seen = 0;
    let mut keep = lines.len();
    for (index, line) in lines.iter().enumerate() {
        let parsed: serde_json::Value = serde_json::from_str(line).expect("a written record is JSON");
        if parsed.get("kind").and_then(serde_json::Value::as_str) == Some(kind.as_str()) {
            seen += 1;
            if seen == occurrence {
                keep = index + 1;
                break;
            }
        }
    }
    assert!(
        seen >= occurrence,
        "the scenario wrote no {occurrence}-th `{}`; it wrote {seen}",
        kind.as_str()
    );
    let mut out = lines[..keep].join("\n");
    out.push('\n');
    std::fs::write(&path, out).expect("rewrite journal");
    keep
}

/// Append a half-written line, the only torn tail the writer can produce.
fn tear_tail(dir: &Path) {
    use std::io::Write as _;
    let path = dir.join("JOURNAL.jsonl");
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .expect("journal");
    file.write_all(br#"{"v":1,"seq":99,"ts":"2026-11-"#)
        .expect("tear");
}

/// A prober with a fixed answer, so the test controls the branch recovery takes.
#[derive(Debug)]
struct Fixed(ProbeAnswer);

impl Prober for Fixed {
    fn probe(
        &self,
        _intent: &immortality_core::protocol::Intent,
        _arguments: Option<&serde_json::Value>,
    ) -> ProbeAnswer {
        self.0
    }
}

/// Plan recovery for the journal as it now stands on disk.
fn recover_now(dir: &Path, task: &TaskId, probe: ProbeAnswer) -> recover::RecoveryReport {
    let outcome = records(dir, task);
    let input = RecoveryInput {
        journal: &outcome,
        workspace: WorkspaceFacts::healthy(format!("agent/tsk_{TASK}"), "abc123"),
        persisted: PersistedState::default(),
        last_checkpoint: Some(LastCheckpoint {
            sha: "f".repeat(40),
            kind: immortality_core::checkpoint::CheckpointKind::Hidden,
        }),
        expected_branch: &format!("agent/tsk_{TASK}"),
    };
    recover::plan(&input, &Fixed(probe))
}

// ── tests ──────────────────────────────────────────────────────────────────────────────

/// A gate world that records every command it was asked to run and answers as told.
#[derive(Debug)]
struct Recording {
    answer: std::result::Result<i32, String>,
    ran: Mutex<Vec<(String, Option<std::time::Duration>)>>,
}

impl Recording {
    fn refuse(command_marker: &str) -> Arc<Self> {
        Arc::new(Self {
            answer: Err(format!("refused: {command_marker}")),
            ran: Mutex::new(Vec::new()),
        })
    }

    fn ran(&self) -> Vec<String> {
        self.ran
            .lock()
            .map(|ran| ran.iter().map(|(command, _)| command.clone()).collect())
            .unwrap_or_default()
    }
}

impl Gates for Recording {
    fn diff(&self, _base: Option<&str>) -> std::result::Result<Diff, String> {
        FakeGates.diff(None)
    }

    fn authorise(&self, command: &str) -> std::result::Result<(), String> {
        if self.answer.is_err() && command.contains("cannot-be-run") {
            return Err("the policy refused this command".to_string());
        }
        Ok(())
    }

    fn run(
        &self,
        command: &str,
        timeout: Option<std::time::Duration>,
    ) -> std::result::Result<immortality_core::gate::CommandRun, String> {
        if let Ok(mut ran) = self.ran.lock() {
            ran.push((command.to_string(), timeout));
        }
        self.answer.as_ref().map_or_else(
            |why| Err(why.clone()),
            |code| {
                Ok(immortality_core::gate::CommandRun {
                    exit_code: Some(*code),
                    timed_out: false,
                    duration_ms: 1,
                    output_tail: "recorded".to_string(),
                })
            },
        )
    }
}

/// A plan with one achieved subtask and one that runs out of attempts.
///
/// `cargo test` is declared by the achieved subtask and `bundle exec rspec` by the one that
/// blocks, so "the blocked subtask's command never ran" is a single inequality on a recorded list
/// rather than a claim about which command was first.
fn blocked_second_subtask() -> [Subtask; 2] {
    let at = FixedClock::fixture().now();
    let mut achieved = Subtask::new(
        1,
        "parse trailing commas",
        "accept a trailing comma",
        Dod {
            statement: "the unit tests pass".to_string(),
            checks: vec![DodCheck {
                id: "test".to_string(),
                command: "cargo test".to_string(),
                required: true,
                timeout_secs: None,
            }],
            review: false,
        },
        at,
    );
    achieved.id = SubtaskId::from_suffix(SUBTASK);
    achieved.state = SubtaskState::Pending;

    let mut failing = Subtask::new(
        1,
        "verify the whole suite",
        "the suite is green",
        Dod {
            statement: "rspec passes".to_string(),
            checks: vec![DodCheck {
                id: "rspec".to_string(),
                command: "bundle exec rspec".to_string(),
                required: true,
                timeout_secs: None,
            }],
            review: false,
        },
        at,
    );
    failing.id = SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAB");
    failing.state = SubtaskState::Pending;
    [achieved, failing]
}

#[test]
fn a_blocked_subtask_fails_the_task_by_name_and_its_command_never_runs() {
    // The third `black_jack` run. Two subtasks were `blocked`, their work had been discarded, and
    // the task-level bar still ran the first `done` subtask's `rspec_pass` — so the verdict a
    // human read was `rspec_pass timed out after 120070 ms`, a deadline for a suite that had
    // nothing to do with what was missing. After D-150 the dead subtask's work is not even in the
    // tree, which made the command a judgement of somebody else's code under its id.
    //
    // Two properties, and both are the point: the task does not reach `ready_for_review`, and the
    // record that says why names the subtask instead of a command.
    let sim = Sim::new();
    let plan = blocked_second_subtask();
    sim.write_plan(&plan);
    let gates = Recording::refuse("the suite cannot pass");

    // One attempt writes and claims; the second subtask is refused by the gate every time, so it
    // spends its budget and blocks.
    let script = vec![
        Reply::call(
            "call_write",
            "write_file",
            r#"{"path":"src/parser.rs","content":"pub fn parse() {}\n"}"#,
        ),
        Reply::call("call_finish", FINISH_SUBTASK, r#"{"summary":"done"}"#),
        Reply::call("call_finish_2", FINISH_SUBTASK, r#"{"summary":"done"}"#),
    ];
    let end = run_cycle_with(
        &sim,
        script,
        gates.clone(),
        cycle::Limits {
            attempts: 1,
            restarts: 0,
            max_depth: 2,
        },
    );

    assert_eq!(end.state, TaskState::FailedGates, "stop was {:?}", end.stop);
    // The blocked subtask's check legitimately ran during its own attempt — that is how it was
    // found red and blocked. What must not happen is a second run at task level, against a tree
    // its work was discarded from: exactly one occurrence, not two.
    let ran = gates.ran();
    let rspec = ran.iter().filter(|command| command.contains("rspec")).count();
    assert_eq!(
        rspec, 1,
        "the blocked subtask's check must not run again at task level: {ran:?}"
    );
    // The whole task-level bar was refused before it ran anything, not only the blocked subtask's
    // line: a bar that judged the achieved subtask's command while the plan was incomplete would
    // report a green on the part that was done, and `evaluate` stopping at the first red check is
    // exactly why the *reason* was a deadline instead of the missing subtask.
    assert_eq!(
        ran.len(),
        2,
        "one attempt-level run per subtask and no task-level bar at all: {ran:?}"
    );
    let records = records(&sim.journal_dir(), &sim.task);
    let verdict = records
        .records
        .iter()
        .rev()
        .find(|record| {
            record.fields.kind == Kind::GateResult
                && record.event.payload["gate"] == serde_json::json!("verdict")
                && record.fields.subtask.is_none()
        })
        .expect("a task-level verdict");
    let why = verdict.event.payload["why"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    assert!(why.contains("verify the whole suite"), "{why}");
    assert!(why.contains("blocked"), "{why}");
}

#[test]
fn a_whole_task_reaches_ready_for_review_and_the_journal_is_the_golden_one() {
    let started = Instant::now();
    let sim = Sim::new();
    let end = run_cycle(&sim, scenario_script());

    assert_eq!(end.state, TaskState::ReadyForReview, "stop was {:?}", end.stop);
    assert_eq!(end.stop, Stop::Done);
    // Four steps in the looping attempt, two in the one that closed it. Not a magic number: it
    // is the script's length, and a cycle that used fewer would not have run every reply.
    assert_eq!(end.model_calls, 6);

    let outcome = records(&sim.journal_dir(), &sim.task);
    assert!(outcome.problems.is_empty(), "{:?}", outcome.problems);
    assert!(!outcome.has_torn_tail);

    // The lifecycle facts the golden is not allowed to be the only evidence for.
    let kinds: Vec<Kind> = outcome.records.iter().map(|r| r.fields.kind).collect();
    assert!(kinds.contains(&Kind::TaskState));
    assert!(kinds.contains(&Kind::SubtaskState));
    assert_eq!(
        kinds.iter().filter(|kind| **kind == Kind::AttemptStarted).count(),
        2,
        "one attempt looped, one finished"
    );
    assert_eq!(
        kinds
            .iter()
            .filter(|kind| **kind == Kind::AttemptRestarted)
            .count(),
        1
    );
    let verdicts: Vec<&JournalRecord> = outcome
        .records
        .iter()
        .filter(|record| {
            record.fields.kind == Kind::GateResult
                && record.event.payload.get("gate").and_then(|gate| gate.as_str()) == Some("verdict")
        })
        .collect();
    assert!(
        verdicts.len() >= 2,
        "the attempt verdict and the task verdict both exist: {}",
        verdicts.len()
    );
    assert!(
        verdicts
            .iter()
            .all(|record| record.event.payload["passed"] == serde_json::json!(true))
    );

    // Every attempt-scoped record names a subtask and an attempt: a record that borrowed the
    // task's scope would make recovery treat it as belonging to no pass.
    for record in &outcome.records {
        if record.fields.kind.is_tool_call() {
            assert!(record.fields.subtask.is_some(), "{:?}", record.fields.kind);
            assert!(record.fields.attempt.is_some(), "{:?}", record.fields.kind);
        }
    }

    let golden = normalise(&outcome, FixedClock::fixture().now());
    // `check_in`, not `check`: `check` resolves the golden directory with *testkit's*
    // `CARGO_MANIFEST_DIR`, which would put this crate's golden in another crate's tree. The
    // golden belongs beside the test that produces it, where a reviewer reads the diff.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("golden");
    immortality_testkit::golden::check_in(&dir, "full-cycle.journal", &golden).assert_match();

    assert!(
        started.elapsed() < SCENARIO_BUDGET,
        "a scenario must not cost more than {SCENARIO_BUDGET:?}; this took {:?}",
        started.elapsed()
    );
}

#[test]
fn the_same_scenario_twice_produces_the_same_journal() {
    // The guard the golden cannot be: a golden blessed once and never re-derived drifts with the
    // code. Here two independent runs must agree record for record.
    //
    // Compared through `normalise` rather than as raw bytes, and that is not a weaker test. The
    // raw file contains the attempt ULID and, in `prev`, the SHA-256 of every preceding line — so
    // the one non-deterministic value in the system is hashed into every record after it. Byte
    // equality could only hold if the id were mocked, which would stop the test from covering the
    // real `Attempt::start`. Substituting ids in the *parsed* journal and then comparing is the
    // comparison that is both deterministic and about the thing that matters: the sequence, the
    // scopes, the payloads and the relative times.
    let started = Instant::now();
    let first = Sim::new();
    run_cycle(&first, scenario_script());
    let second = Sim::new();
    run_cycle(&second, scenario_script());

    let origin = FixedClock::fixture().now();
    let first_outcome = records(&first.journal_dir(), &first.task);
    let second_outcome = records(&second.journal_dir(), &second.task);
    assert_eq!(
        first_outcome.records.len(),
        second_outcome.records.len(),
        "both runs must write the same number of records"
    );
    let one = normalise(&first_outcome, origin);
    let two = normalise(&second_outcome, origin);
    assert_eq!(one, two, "two runs of one scenario must not differ");
    assert_ne!(
        one,
        serde_json::json!([]),
        "a scenario that wrote nothing proves nothing"
    );

    assert!(
        started.elapsed() < SCENARIO_BUDGET,
        "two scenarios must not cost more than {SCENARIO_BUDGET:?}; they took {:?}",
        started.elapsed()
    );
}

#[test]
fn the_journal_stamps_relative_time_so_the_golden_has_no_wall_clock_in_it() {
    // `dt_ms` is the column the golden leans on: it must be the injected clock's, and it must be
    // monotone. A run whose timestamps came from `Utc::now()` would be a golden that changes
    // every time anybody runs the tests.
    let sim = Sim::new();
    run_cycle(&sim, scenario_script());
    let outcome = records(&sim.journal_dir(), &sim.task);
    let origin = FixedClock::fixture().now();
    let mut previous = 0;
    for record in &outcome.records {
        let offset = record.fields.at.signed_duration_since(origin).num_milliseconds();
        assert!(offset >= previous, "{:?} went backwards", record.fields.kind);
        assert_eq!(
            offset % i64::try_from(STEP_WALL_TIME.as_millis()).expect("fits"),
            0,
            "a timestamp off the step grid is not from the injected clock"
        );
        previous = offset;
    }
    assert!(previous > 0, "the scenario must be more than instant");
}

#[test]
fn a_crash_after_a_write_intent_is_recovered_from_the_tools_own_answer() {
    // The classic window: the model asked for a write, the journal recorded the intent, and the
    // process died. `write_file` is `ByContent`, so the answer is not "retry" but "look at the
    // file" — and the two answers lead to different plans.
    let started = Instant::now();
    let sim = Sim::new();
    run_cycle(&sim, scenario_script());
    let kept = die_after(&sim.journal_dir(), Kind::ToolCallIntent, 1);
    assert!(kept > 0);

    let missing = recover_now(&sim.journal_dir(), &sim.task, ProbeAnswer::NotApplied);
    assert_eq!(missing.verdict, Verdict::Repaired);
    assert_eq!(missing.calls.len(), 1);
    assert!(
        missing
            .steps
            .iter()
            .any(|step| matches!(step, RecoveryStep::RetryCall { tool, .. } if tool == "write_file")),
        "{:?}",
        missing.steps
    );

    let present = recover_now(&sim.journal_dir(), &sim.task, ProbeAnswer::Applied);
    assert!(
        present
            .steps
            .iter()
            .any(|step| matches!(step, RecoveryStep::SynthesizeApplied { .. })),
        "{:?}",
        present.steps
    );
    assert!(
        !present
            .steps
            .iter()
            .any(|step| matches!(step, RecoveryStep::RetryCall { .. })),
        "an applied effect must never be repeated: {:?}",
        present.steps
    );

    assert!(recover::all_calls_addressed(
        &records(&sim.journal_dir(), &sim.task).records,
        &present
    ));

    assert!(
        started.elapsed() < SCENARIO_BUDGET,
        "a crash scenario must not cost more than {SCENARIO_BUDGET:?}; this took {:?}",
        started.elapsed()
    );
}

#[test]
fn a_crash_between_the_rollback_marker_and_its_confirmation_completes_the_rollback() {
    // The other ordering the design protects: the marker is written *before* git moves the tree,
    // and `attempt_restarted` after. A crash in that window leaves a tree nobody knows and one
    // record naming the base — so recovery finishes the job and closes the attempt, because L3
    // is defined as a fresh context, not a resumed one.
    let sim = Sim::new();
    run_cycle(&sim, scenario_script());
    die_after(&sim.journal_dir(), Kind::CheckpointCreated, 3);

    let heat = records(&sim.journal_dir(), &sim.task);
    assert!(
        !heat
            .records
            .iter()
            .any(|record| record.fields.kind == Kind::AttemptRestarted),
        "the cut must land before the confirmation"
    );

    let report = recover_now(&sim.journal_dir(), &sim.task, ProbeAnswer::Applied);
    assert_eq!(report.verdict, Verdict::Repaired);
    let sha = report
        .steps
        .iter()
        .find_map(|step| match step {
            RecoveryStep::RollbackToCheckpoint { sha, .. } => Some(sha.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no rollback planned: {:?}", report.steps));
    // The *base*: the marker says where the tree is about to move to, not what it is about to
    // destroy. Checkpoint 1 is the base, checkpoint 2 is the discarded work.
    assert_eq!(
        sha,
        format!("{:040x}", 1),
        "the marker names the base, not the discarded work"
    );

    // No `AbandonAttempt`, and that is the correct answer rather than a gap: `attempt_failed` is
    // written when the detector ends the attempt, *before* the restart is decided. So the attempt
    // is already closed by the time the marker exists, and closing it again would be recovery
    // inventing an event the journal does not support. `AbandonAttempt` is for an attempt a crash
    // left `Running`, which is a different window.
    assert!(
        !report
            .steps
            .iter()
            .any(|step| matches!(step, RecoveryStep::AbandonAttempt { .. })),
        "the attempt was closed by its own `attempt_failed`: {:?}",
        report.steps
    );
    let failures = records(&sim.journal_dir(), &sim.task)
        .records
        .iter()
        .filter(|record| record.fields.kind == Kind::AttemptFailed)
        .count();
    assert_eq!(failures, 1, "the failed record is what closes the attempt");
}

#[test]
fn a_crash_before_any_effect_invents_no_work_but_closes_the_attempt() {
    // Two separate claims, and the test is only useful if it holds both.
    //
    // Recovery inventing work is a bug in its own right: an attempt that died between its start
    // record and its first tool call has done nothing, so no call may be probed, retried,
    // synthesized or abandoned, and nothing may be invented about the state files.
    //
    // The attempt itself, though, is over — the process that owned it is gone, and the shipped
    // runner starts a fresh conversation for whatever comes next (`StepLoop` builds its context
    // from the stable layers; `model_response` keeps the reply's length, not its text). Leaving it
    // open is what a live `kill -9` left behind: two `attempt_started` records under one key.
    let sim = Sim::new();
    run_cycle(&sim, scenario_script());
    die_after(&sim.journal_dir(), Kind::AttemptStarted, 1);

    let report = recover_now(&sim.journal_dir(), &sim.task, ProbeAnswer::Unknown);
    assert_eq!(report.calls.len(), 0, "{:?}", report.calls);
    assert_eq!(report.divergences.len(), 0, "{:?}", report.divergences);
    // Exactly one step that touches the work, and it is the closure: anything else would be
    // recovery inventing work the crashed attempt never did. `RecordRecovery` is the audit line
    // every non-clean pass writes, not an effect on the tree or the calls.
    let effects: Vec<&RecoveryStep> = report
        .steps
        .iter()
        .filter(|step| !matches!(step, RecoveryStep::RecordRecovery { .. }))
        .collect();
    assert_eq!(effects.len(), 1, "{:?}", report.steps);
    assert!(
        matches!(
            effects.first(),
            Some(RecoveryStep::AbandonAttempt { number: 1, .. })
        ),
        "{:?}",
        report.steps
    );
    // Closing an attempt writes a record, so the pass is a repair rather than `clean`; reporting
    // `Healthy` while the executor appends `attempt_restarted` would make the audit trail describe
    // a pass that did not happen.
    assert_eq!(report.verdict, Verdict::Repaired, "{:?}", report.steps);
}

#[test]
fn a_torn_tail_is_dropped_and_counted() {
    // The one damage the writer can do to itself: a line without its newline. It must be
    // truncated, audited, and must not take the surviving records with it.
    let sim = Sim::new();
    run_cycle(&sim, scenario_script());
    let before = records(&sim.journal_dir(), &sim.task).records.len();
    tear_tail(&sim.journal_dir());

    let outcome = records(&sim.journal_dir(), &sim.task);
    assert!(outcome.has_torn_tail);
    assert_eq!(outcome.records.len(), before, "a torn tail is not a lost record");

    let report = recover_now(&sim.journal_dir(), &sim.task, ProbeAnswer::Applied);
    assert_eq!(report.verdict, Verdict::Repaired);
    assert!(
        matches!(report.steps.first(), Some(RecoveryStep::DropTornTail { .. })),
        "{:?}",
        report.steps
    );
    assert!(report.dropped_tail_bytes > 0);
}
