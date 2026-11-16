//! `start`: begin a task and run it to a verdict, in the daemon.
//!
//! # Why the run is a background task
//!
//! A cycle is hours. The RPC answer is "the task started", not "the task finished": a client that
//! blocked for the whole run would be a client that times out, and the human would have no way to
//! know whether it was still going. The answer carries the task id, the state it was accepted in,
//! and where to watch it — `immortality attach <task>`.
//!
//! # Why `start` may plan first
//!
//! The roadmap's exit criterion for this milestone is that `immortality start "<goal>"` runs the
//! full cycle *in the daemon*. A goal with no plan on disk is the normal first invocation, and
//! making the human run `plan --goal` first would mean `start` could only ever resume. So a goal
//! with no plan is handed to the planner (M5.7) first, on the same inference slot, and the run
//! begins against the tree that was just written. A plan that is already there is respected, not
//! regenerated: the contract is that a human may rewrite those files before the run.
//!
//! # Why the task is created here and not by the client
//!
//! `TaskId` is a ULID and the registry is machine-local (D-011's counterpart for state): two
//! clients minting ids for the same intent would be two tasks. The daemon mints, journals
//! `task_created` and `queued`, and only then returns, so the id in the answer is already real to
//! every other command.
//!
//! # Why the model is built inside the run thread
//!
//! [`crate::runner::LlmModel`] owns a current-thread runtime, and a runtime cannot be dropped from
//! an async context (see [`crate::planner`]). The whole cycle — synchronous by design, because
//! steps are sequential and one inference runs at a time — therefore runs on its own thread, and
//! every runtime it creates and destroys lives there.
//!
//! # Why a second `start` for the same task is refused
//!
//! [`immortality_core::journal::Journal`] is single-writer by construction: two writers on one
//! file collide on `seq` and fork the chain, which is the one unrecoverable state the design has.
//! The daemon therefore keeps the set of tasks it is currently running and refuses a second
//! admission with a message naming the first. A daemon restart clears the set, which is correct:
//! the process that held the journal is gone, and its recovery pass has already repaired the tail.
//!
//! # Why a backend outage leaves the task `running`
//!
//! A model failure ends the attempt `Abandoned`, journals `backend_unavailable`, and leaves the
//! task where it was. Blocking or failing it would turn a temporary condition into a verdict and
//! lose the plan, the worktree and the checkpoint with it. `start --task` (or a later admission)
//! picks it up when the backend returns.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use immortality_core::checkpoint::GitRepo;
use immortality_core::config::Config;
use immortality_core::context::{Budget, CharsCounter, TokenCounter};
use immortality_core::domain::{TaskId, TaskState};
use immortality_core::journal::{Journal, JournalReader, Kind};
use immortality_core::plan::Plan;
use immortality_core::registry::{Project, Registry};
use immortality_core::scheduler::backend;
use immortality_core::worktree::WorktreeManager;
use immortality_llm::Backend;
use serde_json::Value;

use crate::cycle::{self, Limits, Start};
use crate::gates::{CheckRunner, ModelReviewer, RealGates, task_base};
use crate::rpc::{ErrorCode, RpcError};
use crate::runner::{AttemptLimits, LlmModel, Model, Runner};
use crate::server::{Events, ServerState};
use crate::toolset::{self, ToolsetDeps};

fn bad_params(message: impl Into<String>) -> RpcError {
    RpcError::new(ErrorCode::BadParams, message)
}

fn internal(message: impl Into<String>) -> RpcError {
    RpcError::new(ErrorCode::Internal, message)
}

/// The state a fresh task is admitted in.
///
/// `queued` is the only entry point the lifecycle has for a new task, and it is what the cycle's
/// own planning edges expect to start from.
const ADMITTED: TaskState = TaskState::Queued;

/// `start`: admit a task and run the cycle in the background.
///
/// The request names either an existing `task` (resume) or a `goal` (new task). A resume is the
/// same code path further down: the cycle reads the plan from disk and continues from wherever the
/// tree is, which is exactly what a restarted daemon needs.
///
/// # Errors
/// An empty goal with no plan to run, an unregistered project, a plan that cannot be read or
/// written, a task already running in this process, or a thread that cannot be started.
pub async fn start(state: &ServerState, params: &Value) -> Result<Value, RpcError> {
    if let Some(task) = params.get("task").and_then(Value::as_str) {
        return resume(state, task);
    }

    let project = crate::commands::project_for(state, params)?;
    let goal = params
        .get("goal")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|goal| !goal.is_empty())
        .map(str::to_string);
    let depth = state.config.loop_.max_subtask_depth;
    let force = params.get("force").and_then(Value::as_bool).unwrap_or(false);

    // The plan is the tree the run executes, so it has to exist before a task does. A plan on
    // disk is *respected* — a human may have edited it — unless `--force` asks for a new one; a
    // plan that is absent is planned from the goal. Missing both is a usage error that names
    // which of the two the human left out, not a silent re-plan.
    let on_disk = immortality_core::planner::existing(&project.path);
    let (plan, planned) = if on_disk.any() && !force {
        let plan = Plan::load(&project.path, depth).map_err(|error| {
            bad_params(format!(
                "the plan in {} cannot be read ({error}); fix it, or pass `--force` with a goal \
                 to replace it",
                project.path.display()
            ))
        })?;
        (plan, false)
    } else {
        let goal = goal.clone().ok_or_else(|| {
            if force {
                bad_params(
                    "`--force` re-plans an existing tree, and re-planning needs a goal: \
                     `immortality start \"what to do\" --force`",
                )
            } else {
                bad_params(format!(
                    "there is no plan to run in {}; give a goal — \
                     `immortality start \"what to do\"` — or write one with \
                     `immortality plan --goal`",
                    project.path.display()
                ))
            }
        })?;
        (plan_now(state, &project, &goal, force).await?, true)
    };
    if plan.subtasks().is_empty() {
        return Err(bad_params(
            "the plan has no subtasks; nothing would run".to_string(),
        ));
    }

    let task = TaskId::generate();
    let at = immortality_core::time::now();
    let task_dir = project.state_dir(&state.state_root).join(task.as_str());
    std::fs::create_dir_all(&task_dir)
        .map_err(|error| internal(format!("cannot create {}: {error}", task_dir.display())))?;

    // The journal is the task's existence: `Registry::owner_of` finds a task by its records as
    // well as by its state directory, and `status` lists it from the same trace. Writing
    // `task_created` before the answer is what makes the returned id usable immediately.
    let mut journal = Journal::open(project.agent_dir(), task.clone())
        .map_err(|error| internal(format!("cannot open the journal: {error}")))?;
    let title = goal.clone().unwrap_or_else(|| "a planned task".to_string());
    journal
        .append_event(
            at,
            Kind::TaskCreated,
            serde_json::json!({ "title": title, "goal": goal }),
        )
        .and_then(|_| {
            journal.append_event(
                at,
                Kind::TaskState,
                serde_json::json!({ "state": ADMITTED.as_str() }),
            )
        })
        .and_then(|_| journal.close())
        .map_err(|error| internal(format!("cannot journal the start: {error}")))?;

    let run = Run {
        config: task_config(state, &project)?,
        state_root: state.state_root.clone(),
        project: project.clone(),
        task: task.clone(),
        task_dir,
        start: Start::resuming(
            ADMITTED,
            goal.clone().unwrap_or_default(),
            // A fresh admission has no outage behind it: the journal was just created.
            None,
        ),
        goal: goal.clone().unwrap_or_default(),
        model: state.model.clone(),
        events: state.events.clone(),
    };
    spawn(state, run)?;

    Ok(serde_json::json!({
        "task": task,
        "project": project.id,
        "path": project.path,
        "state": ADMITTED.as_str(),
        "goal": goal,
        "subtasks": plan.subtasks().len(),
        "admitted": true,
        "resumed": false,
        "planned": planned,
    }))
}

/// Ask the planner for a tree and load what it wrote.
async fn plan_now(state: &ServerState, project: &Project, goal: &str, force: bool) -> Result<Plan, RpcError> {
    let params = serde_json::json!({
        "project": project.path,
        "goal": goal,
        "force": force,
    });
    crate::planner::write_plan(state, &params).await?;
    Plan::load(&project.path, state.config.loop_.max_subtask_depth)
        .map_err(|error| internal(format!("the plan that was just written cannot be read: {error}")))
}

/// Resume a task that already exists.
///
/// The task keeps its id, its journal and its tree; the cycle continues from whatever the journal
/// says the last state was. This is the code path both `kill -9` recovery and a human's
/// `start --task` take, which is why the state is read from the journal rather than passed in.
fn resume(state: &ServerState, task: &str) -> Result<Value, RpcError> {
    let task = TaskId::parse(task).map_err(|error| bad_params(error.to_string()))?;
    let registry = Registry::open(state.registry_path())
        .map_err(|error| internal(format!("cannot read the registry: {error}")))?;
    let project = registry
        .owner_of(&state.state_root, &task)
        .map_err(|error| internal(format!("cannot find the task's project: {error}")))?
        .cloned()
        .ok_or_else(|| RpcError::new(ErrorCode::NotFound, format!("no project owns task {task}")))?;

    let depth = state.config.loop_.max_subtask_depth;
    let plan = Plan::load(&project.path, depth)
        .map_err(|error| bad_params(format!("cannot resume {task}: {error}")))?;
    let (current, goal, outage) = history(&project, &task)?;
    // A finished task has nothing to resume, and the lifecycle is the authority on "finished":
    // a list of terminal states here would be the copy that goes stale. `verifying` is resumable
    // — the subtasks are settled and only the verdict is owed — which is why it is not "terminal"
    // but also not a valid `enter_running` edge; the cycle handles it as its own resume.
    if !resumable(current) {
        return Err(bad_params(format!(
            "task {task} is `{current}`; only an unfinished task can be resumed"
        )));
    }

    let task_dir = project.state_dir(&state.state_root).join(task.as_str());
    std::fs::create_dir_all(&task_dir)
        .map_err(|error| internal(format!("cannot create {}: {error}", task_dir.display())))?;

    let run = Run {
        config: task_config(state, &project)?,
        state_root: state.state_root.clone(),
        project: project.clone(),
        task: task.clone(),
        task_dir,
        start: Start::resuming(current, goal.clone(), outage),
        goal,
        model: state.model.clone(),
        events: state.events.clone(),
    };
    spawn(state, run)?;

    Ok(serde_json::json!({
        "task": task,
        "project": project.id,
        "path": project.path,
        "state": current.as_str(),
        "subtasks": plan.subtasks().len(),
        "admitted": false,
        "resumed": true,
        "planned": false,
    }))
}

/// Whether the cycle can pick `state` up.
///
/// The same set [`crate::cycle::enter_running`] accepts, plus `verifying`: there the work is done
/// and only the task-level verdict is owed, which the cycle runs without re-driving the tree.
const fn resumable(state: TaskState) -> bool {
    matches!(
        state,
        TaskState::Queued
            | TaskState::Planning
            | TaskState::AwaitingPlanApproval
            | TaskState::Running
            | TaskState::Verifying
            | TaskState::Blocked
            | TaskState::FailedGates
    )
}

/// What the journal says about a task: where it stopped, what it was for, and since when the
/// backend has been missing.
fn history(
    project: &Project,
    task: &TaskId,
) -> Result<(TaskState, String, Option<immortality_core::domain::Timestamp>), RpcError> {
    let outcome = JournalReader::open(project.agent_dir(), task)
        .and_then(|reader| reader.read())
        .map_err(|error| internal(format!("cannot read the journal of {task}: {error}")))?;
    let records: Vec<immortality_core::journal::JournalRecord> = outcome
        .records
        .iter()
        .filter(|record| &record.fields.task == task)
        .cloned()
        .collect();
    let state = crate::scheduling::last_task_state(&records)
        // A task with journal records and no state record was interrupted between its creation
        // and its first transition; `queued` is where it still is.
        .unwrap_or(ADMITTED);
    // The goal is quoted by every restart's strategy, and a fresh process has nothing else: the
    // plan says what each subtask does, but only this says what the task is *for*.
    let goal = records
        .iter()
        .find(|record| record.fields.kind == Kind::TaskCreated)
        .and_then(|record| record.event.payload.get("goal"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    // Read, not re-detected: a daemon killed during a fifteen-minute outage must not restart the
    // clock, or the escalation re-arms on every process start.
    let outage = backend::since_from_journal(&records);
    Ok((state, goal, outage))
}

/// The daemon's config with this repository's project file laid over it.
///
/// Applied here, at the only place a project is in hand, because `serve` has no project when it
/// loads: `config::load` finds the project layer by walking up from the process's working
/// directory, and under launchd that directory is `/`. The result is one config per run, built
/// from the same global file for every task, which is what makes the documented three-layer order
/// (global → project → flags) true for the daemon rather than only for a client run inside the
/// repository.
fn task_config(state: &ServerState, project: &Project) -> Result<Config, RpcError> {
    immortality_core::config::with_project_file(&state.config, &project.path).map_err(|error| {
        internal(format!(
            "the project config in {} cannot be read: {error}",
            project
                .path
                .join(immortality_core::paths::AGENT_DIR)
                .join("config.toml")
                .display()
        ))
    })
}

/// Everything a run needs, owned so it can move to the thread that runs it.
struct Run {
    config: Config,
    state_root: PathBuf,
    project: Project,
    task: TaskId,
    task_dir: PathBuf,
    start: Start,
    /// The task's goal, from `task_created` or the admission request.
    ///
    /// Separate from `start.strategy`, which is what makes *this pass* different: the judge is
    /// asked whether a command fits the task, and the pass's strategy says nothing about that.
    goal: String,
    /// The model a test injected, or `None` to build the real client on the thread.
    model: Option<Arc<dyn Model>>,
    /// Where the run announces its records, so `attach` sees a progressing task rather than only
    /// its finished journal.
    events: Option<Events>,
}

/// Claim the task's single run slot and start the cycle on its own thread.
///
/// The thread is not joined: a cycle is hours and the RPC answer is not its result. What it does
/// report is that the thread was spawned — a daemon that cannot spawn should say so now, not leave
/// a task looking started and doing nothing.
fn spawn(state: &ServerState, run: Run) -> Result<(), RpcError> {
    let task = run.task.clone();
    let slot = RunSlot::claim(state, task.clone())?;
    let named = task.clone();
    std::thread::Builder::new()
        .name(format!("cycle-{task}"))
        .spawn(move || {
            // The slot is dropped at the end of this closure whatever the outcome, so a run that
            // panics or errors does not leave the task permanently unstartable.
            let _slot = slot;
            if let Err(error) = execute(run) {
                tracing::warn!("the cycle for {task} ended: {error}");
            }
        })
        .map_err(|error| internal(format!("cannot start the run of {named}: {error}")))?;
    Ok(())
}

/// One admission's claim on a task, released when the run ends.
///
/// A struct with a `Drop` rather than paired `begin`/`end` calls: the release has to happen on
/// every path out of the thread, and a path that forgot it would make the task look permanently
/// running.
#[derive(Debug)]
struct RunSlot {
    running: Arc<std::sync::Mutex<std::collections::BTreeSet<TaskId>>>,
    task: TaskId,
}

impl RunSlot {
    fn claim(state: &ServerState, task: TaskId) -> Result<Self, RpcError> {
        // The guard is dropped before the `Self` is built: a struct holding the mutex it just
        // released would be a deadlock on the next claim, and clippy's early-drop warning is
        // pointing at exactly that shape.
        let claimed = {
            let mut running = state
                .running
                .lock()
                .map_err(|_| internal("the run bookkeeping is poisoned; restart the daemon".to_string()))?;
            running.insert(task.clone())
        };
        if !claimed {
            return Err(RpcError::new(
                ErrorCode::BadParams,
                format!(
                    "task {task} is already running in this daemon; \
                     watch it with `immortality attach {task}`"
                ),
            ));
        }
        Ok(Self {
            running: Arc::clone(&state.running),
            task,
        })
    }
}

impl Drop for RunSlot {
    fn drop(&mut self) {
        if let Ok(mut running) = self.running.lock() {
            running.remove(&self.task);
        }
    }
}

/// The fork point of `task`'s branch, as the task-level gate's diff base.
///
/// Separated from [`execute`] only to keep that function inside the line budget; the reasoning is
/// in [`crate::gates::task_base`].
fn gate_base(worktree: &Path, repository: &Path, task: &TaskId) -> Option<String> {
    task_base(&GitRepo::new(worktree), repository, &format!("agent/{task}"))
}

/// Everything the cycle needs, built on the run's own thread.
///
/// Every error here is the daemon's own wiring — a missing worktree, an unreadable plan — not the
/// model's; a model failure ends an attempt, which is not an error.
fn execute(run: Run) -> Result<TaskState, String> {
    let Run {
        config,
        state_root,
        project,
        task,
        task_dir,
        start,
        goal,
        model,
        events,
    } = run;
    let branch = format!("agent/{task}");
    let worktree = WorktreeManager::new(&project.path)
        .ensure(task.as_str(), &branch, None)
        .map_err(|error| format!("cannot create the worktree: {error}"))?;
    let deps = ToolsetDeps::for_task(
        &config,
        state_root,
        task_dir,
        worktree.path.clone(),
        project.path.clone(),
        task.clone(),
    )
    .map_err(|error| format!("cannot build the tool layer: {error}"))?
    // The daemon's one notification channel, chosen once. Without this the toolset's default is
    // `Silent`, and an escalation would be recorded on disk while nobody was told — the exact
    // hole the 0.3 milestone is about.
    .notifying(immortality_tools::system());

    let model: Arc<dyn Model> = match model {
        Some(model) => model,
        None => Arc::new(
            LlmModel::new(backend(&config)?)
                .map_err(|error| format!("cannot start the model client: {error}"))?,
        ),
    };
    let judge = judged(&config, &project, &task, Arc::clone(&model))?;
    let deps = deps.with_judge(judge, &goal);
    let registry = toolset::build(&deps).map_err(|error| format!("cannot build the tools: {error}"))?;

    let gate = Arc::clone(&deps.gate);
    let runner = Runner {
        model: Arc::clone(&model),
        model_name: config.models.executor.clone(),
        registry: Arc::new(registry),
        artifacts: deps.artifacts,
        gate: Arc::clone(&gate),
        system_prompt: system_prompt(&start.strategy),
        max_tool_output_bytes: config.context.max_tool_output_bytes,
        budget: Budget {
            soft: config.context.soft_budget_tokens,
            hard: config.context.hard_budget_tokens,
        },
        limits: AttemptLimits {
            max_steps: config.loop_.max_steps_per_attempt,
            max_tokens: config.loop_.max_tokens_per_attempt,
            stall_steps: config.loop_.stall_steps,
            // `AttemptLimits` has no wall-clock field in the config; the design's ~45 minutes is
            // sixty per step at the default sixty steps, and deriving it keeps the two honest
            // instead of introducing a second number nobody tunes.
            max_wall_time: Duration::from_secs(u64::from(config.loop_.max_steps_per_attempt) * 60),
        },
        counter: Arc::new(CharsCounter::default()) as Arc<dyn TokenCounter + Send + Sync>,
        task: task.clone(),
        detectors: config.detectors.clone(),
        gates: Some(Arc::new(RealGates::new(
            worktree.path.clone(),
            Arc::clone(&gate),
            CheckRunner::new(worktree.path.clone()),
            gate_base(&worktree.path, &project.path, &task),
        ))),
        reviewer: Some(Arc::new(ModelReviewer::new(
            Arc::clone(&model),
            config.models.utility.clone(),
        ))),
        max_subtask_depth: config.loop_.max_subtask_depth,
        // The top-level agent. Only a delegated brief is deeper, and it is the only runner
        // that sets this to anything else.
        delegation_depth: 0,
        checkpoints: Some(Arc::new(
            crate::checkpoints::GitCheckpoints::new(worktree.path, branch.clone())
                // A model that left the worktree detached or on another branch would otherwise
                // have its rollback land on the wrong ref.
                .restoring(branch),
        )),
        clock: Arc::new(immortality_core::time::SystemClock),
        // The same collector the registry's `escalate` holds: the tool could not write the
        // journal, and this runner is the only thing holding it (D-117).
        escalations: vec![Arc::clone(&deps.escalations)],
    };

    wire_delegation(&runner, &deps.sub_agent);

    let mut plan = Plan::load(&project.path, config.loop_.max_subtask_depth)
        .map_err(|error| format!("cannot read the plan: {error}"))?;
    let mut journal = Journal::open(project.agent_dir(), task)
        .map_err(|error| format!("cannot open the journal: {error}"))?;
    if let Some(events) = events {
        // A clone of the daemon's broadcast: the run's records reach `attach` as they happen.
        journal.set_sink(Arc::new(events));
    }
    let outcome = cycle::run(
        &runner,
        &mut plan,
        &mut journal,
        &start,
        Limits {
            attempts: config.loop_.max_attempts_per_subtask,
            restarts: config.loop_.max_restarts_per_subtask,
            max_depth: config.loop_.max_subtask_depth,
        },
    )
    .map_err(|error| error.to_string())?;
    journal
        .close()
        .map_err(|error| format!("cannot close the journal: {error}"))?;
    Ok(outcome.state)
}

/// Make `delegate` real, once the registry and the runner both exist.
///
/// # Why this is a separate call and not part of building the runner
///
/// The order is load-bearing and a compiler cannot check it. `toolset::build` creates the registry,
/// and with it the `delegate` tool over `slot`; the runner needs that registry, so the tool exists
/// before the runner does. [`crate::sub_agent::install`] then fills the slot. Do it the other way
/// round and the slot would be replaced with a fresh empty one after the tool had already captured
/// it — both sides are behind an `Arc`, so nothing would complain and every delegated call would
/// answer "no sub-agent wired in".
///
/// One install per run: every attempt, restart and subtask of this task shares the runner
/// installed here.
fn wire_delegation(runner: &Runner, slot: &immortality_tools::SubAgentSlot) {
    let _brief = crate::sub_agent::install(runner, slot);
    // A `debug` line rather than `info`: "the daemon wired itself correctly" is not news. What
    // matters is the test that a delegated call now writes a report.
    tracing::debug!(
        wired = slot.is_wired(),
        "the sub-agent runner is installed behind `delegate`"
    );
}

/// The judge for one run: the model, the cache, and what the journal already remembers.
///
/// Wrapped in [`immortality_tools::judge::Recorded`] here rather than in the tool layer for the
/// reason D-111 gives: the policy crate must not know what a model is, and the cache is a decision
/// about paying twice, not about policy.
fn judged(
    config: &Config,
    project: &Project,
    task: &TaskId,
    model: Arc<dyn Model>,
) -> Result<Arc<dyn immortality_tools::judge::Judge>, String> {
    Ok(Arc::new(immortality_tools::judge::Recorded::seeded(
        Arc::new(crate::judge::ModelJudge::new(
            model,
            config.models.utility.clone(),
        )),
        immortality_tools::judge::policy_fingerprint(&config.policy),
        recorded_verdicts(project, task)?,
    )))
}

/// The judge verdicts a previous run of this task already paid for.
///
/// Read from the journal before it is opened for writing, so a restart does not ask the judge
/// about every unknown program again. A record whose payload cannot be read back is skipped one
/// at a time: the cost is a repeated judgement, and refusing the task over one damaged line would
/// turn a model call into a dead end.
fn recorded_verdicts(
    project: &Project,
    task: &TaskId,
) -> Result<Vec<immortality_tools::judge::Verdict>, String> {
    let outcome = JournalReader::open(project.agent_dir(), task)
        .and_then(|reader| reader.read())
        .map_err(|error| format!("cannot read the journal for the judge's cache: {error}"))?;
    Ok(outcome
        .records
        .iter()
        .filter(|record| record.fields.kind == Kind::JudgeVerdict)
        .filter_map(|record| immortality_tools::judge::Verdict::from_payload(&record.event.payload))
        .collect())
}

/// The backend client described by the configuration.
fn backend(config: &Config) -> Result<Backend, String> {
    Backend::new(
        config.backend.base_url.clone(),
        Duration::from_secs(config.backend.connect_timeout_secs),
        Duration::from_secs(config.backend.request_timeout_secs),
    )
    .map_err(|error| format!("cannot build the HTTP client: {error}"))
}

/// What the executor is told before its history starts.
///
/// The goal is quoted because a resumed task's fresh context has nothing else: the plan's subtask
/// goals carry the work, but the *why* is only in the task's goal.
fn system_prompt(goal: &str) -> String {
    if goal.trim().is_empty() {
        return "You are an autonomous coding agent working one subtask at a time.".to_string();
    }
    format!(
        "You are an autonomous coding agent. The task's overall goal, for context:\n\n{goal}\n\n\
         Work the subtask you are given, using the tools available. Claim it finished with \
         `finish_subtask` only when the work is actually done; the gates decide whether it is."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use immortality_core::domain::Timestamp;
    use immortality_core::journal::Journal;
    use std::path::Path;

    fn project(dir: &Path) -> Project {
        Project {
            id: "prj_test".to_string(),
            path: dir.to_path_buf(),
            remote: None,
        }
    }

    fn at(secs: i64) -> Timestamp {
        chrono::DateTime::from_timestamp(secs, 0).expect("timestamp")
    }

    #[test]
    fn recorded_verdicts_returns_only_what_the_judge_wrote() {
        // The restart path: every `judge_verdict` record is a judgement already paid for, so it
        // must come back into the cache. The filter is the thing that can silently break — a
        // missing kind match reads as an empty cache, and an empty cache is not an error, it is
        // just the model being asked again.
        let dir = tempfile::tempdir().expect("tempdir");
        let task = TaskId::generate();
        let allowed = immortality_tools::judge::Verdict {
            signature: "frobnicate --all".to_string(),
            policy: "p1".to_string(),
            command: "frobnicate --all".to_string(),
            judgement: immortality_tools::judge::Judgement::Allow,
        };
        let agent = dir.path().join(immortality_core::paths::AGENT_DIR);
        let mut journal = Journal::open(&agent, task.clone()).expect("journal");
        journal
            .append_event(
                at(1),
                Kind::TaskCreated,
                serde_json::json!({"title": "t", "goal": "g"}),
            )
            .expect("created");
        journal
            .append_event(at(2), Kind::JudgeVerdict, allowed.to_payload())
            .expect("verdict");
        // A record of another kind with a payload that happens to parse: it must not become a
        // verdict, or the cache would fill with judgements nobody made.
        journal
            .append_event(at(3), Kind::GateResult, allowed.to_payload())
            .expect("gate");
        journal.close().expect("close");

        let verdicts = recorded_verdicts(&project(dir.path()), &task).expect("read");
        assert_eq!(verdicts, vec![allowed]);
    }

    #[test]
    fn a_journal_with_no_verdicts_yields_an_empty_cache() {
        // A first run, and the common case. Distinct from a read failure, which is an error and
        // has to reach the caller: an empty cache would silently re-pay, a failure has to say why.
        let dir = tempfile::tempdir().expect("tempdir");
        let task = TaskId::generate();
        let agent = dir.path().join(immortality_core::paths::AGENT_DIR);
        let mut journal = Journal::open(&agent, task.clone()).expect("journal");
        journal
            .append_event(at(1), Kind::TaskCreated, serde_json::json!({"title": "t"}))
            .expect("created");
        journal.close().expect("close");
        let verdicts = recorded_verdicts(&project(dir.path()), &task).expect("read");
        assert_eq!(verdicts, Vec::new(), "a journal with no verdicts must yield none");
    }
}
