//! The run cycle: from a plan on disk to a verdict, with no policy of its own.
//!
//! # What was wrong with where this lived
//!
//! Until this milestone the only implementation was in `crates/daemon/tests/simulator.rs`. That
//! made M4's exit criterion provable and the product unusable: there was no code path a daemon
//! could call, and no way to say whether the tested cycle and the shipped one agreed. A test
//! fixture that is the only implementation is a reference document, not a program.
//!
//! So the cycle is here, and the simulator calls **this**. A test that proves the cycle works
//! therefore proves it about the code that runs.
//!
//! # Why there is no policy in this module
//!
//! Every decision belongs to something already tested on its own:
//!
//! * which subtask runs next is [`immortality_core::scheduler::next`];
//! * what an attempt ends with is [`crate::runner::AttemptSession::drive`];
//! * whether a claim closes a subtask is `Runner::settle_claim`, through the session;
//! * whether a split is legal is [`immortality_core::scheduler::apply_split`];
//! * whether the task may move to `ready_for_review` is the lifecycle table via
//!   `TaskState::trigger`.
//!
//! What this module adds is the **order** the design fixes and nothing else. Three orderings are
//! load-bearing and each is written once:
//!
//! * **Tree before journal.** A subtask's state is written to `TASKS.json` before its audit line
//!   is written to the journal. A crash between them reads as "settled, nothing to say"; the other
//!   order announces a closure the tree never received.
//! * **Gates before the verdict.** `ready_for_review` is written only after the task-level gates
//!   are green, and the gates are evaluated before the lifecycle edge is taken.
//! * **Out of the model's way on an outage.** A backend failure ends the attempt as `Abandoned`,
//!   which journals `backend_unavailable` and leaves the task exactly where it was. It does not
//!   block, because an outage is not a verdict about the work (D-027).
//!
//! # Why every subtask needs an attempt number from the tree
//!
//! A subtask resumed after a restart must continue counting attempts, or the restart budget
//! resets on every process start and a hopeless subtask runs forever. The number comes from the
//! attempts the tree already records, not from a counter in memory.
//!
//! # Why the caller says where the run starts
//!
//! A daemon that has been `kill -9`ed mid-verification restarts in `verifying`, with every subtask
//! settled. Re-driving that tree would find nothing to run and then try to take
//! `running → execution_finished`, which the lifecycle table does not allow. [`Start`] carries the
//! state the journal ended in, so the cycle resumes at the right phase instead of inferring it.

use immortality_core::domain::{Dod, Subtask, SubtaskId, SubtaskState, TaskState, TaskTrigger, Timestamp};
use immortality_core::gate::{GateRecord, Reviewer, evaluate};
use immortality_core::journal::{Journal, JournalReader, JournalRecord, Kind, Scope};
use immortality_core::plan::Plan;
use immortality_core::restart::{Allowance, Spent, Stop};
use immortality_core::scheduler;
use immortality_core::scheduler::backend;

use crate::runner::{Ending, Runner};

/// How many subtask passes one cycle may take before it declares itself stuck.
///
/// Not a budget — the real limits are per attempt and per subtask — but a guard against a bug in
/// the loop itself. A correct tree settles or blocks every subtask it visits, so this is
/// unreachable; reaching it is a defect, and returning an error is how the daemon finds out
/// instead of spinning until a human notices.
pub const MAX_SUBTASK_PASSES: usize = 256;

/// Why a cycle ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CycleError {
    /// The tree has open subtasks none of which may run. Validation rejects this shape, so it is
    /// either a plan edited into a cycle while running or a bug in `next_ready`.
    Stuck { open: Vec<String> },
    /// The loop exceeded [`MAX_SUBTASK_PASSES`]. Always a defect.
    DidNotConverge,
    /// The task is in a state this cycle does not start from: `ready_for_review`, `merged`,
    /// `cancelled`. Reported rather than guessed, because guessing here would mean inventing a
    /// lifecycle edge that does not exist.
    Refused { state: TaskState },
    /// A journal write or a tree write failed; the reason is the store's own.
    Store(String),
}

impl std::fmt::Display for CycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stuck { open } => {
                write!(
                    f,
                    "no subtask may run but {} are still open: {}",
                    open.len(),
                    open.join(", ")
                )
            }
            Self::DidNotConverge => write!(
                f,
                "the cycle did not settle within {MAX_SUBTASK_PASSES} subtask passes"
            ),
            Self::Refused { state } => {
                write!(f, "a task in `{state}` has no run ahead of it")
            }
            Self::Store(reason) => write!(f, "{reason}"),
        }
    }
}

impl std::error::Error for CycleError {}

/// Where a run starts, as the durable state reports it.
///
/// Assembled by the caller from the tree and the journal — never inferred inside the loop, which
/// cannot read the journal it is writing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Start {
    /// The state the task is in. `queued` for a fresh task; whatever the last record said for a
    /// resumed one.
    pub state: TaskState,
    /// What makes this pass different from the previous one, quoted by every restart. Empty for a
    /// fresh task.
    pub strategy: String,
    /// When the backend was first found unavailable, from the journal. `None` while healthy.
    ///
    /// Carried in so the outage interval stays continuous: a run that re-detected the outage and
    /// started counting again would re-arm the fifteen-minute escalation on every attempt.
    pub outage_since: Option<Timestamp>,
}

impl Start {
    /// A task that has never run.
    #[must_use]
    pub const fn fresh() -> Self {
        Self {
            state: TaskState::Queued,
            strategy: String::new(),
            outage_since: None,
        }
    }

    /// A task picked up from wherever the last process left it.
    #[must_use]
    pub fn resuming(state: TaskState, strategy: impl Into<String>, outage_since: Option<Timestamp>) -> Self {
        Self {
            state,
            strategy: strategy.into(),
            outage_since,
        }
    }
}

/// What one pass over the tree achieved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cycle {
    /// The state the task reached: `ready_for_review`, `blocked`, `failed_gates`, or `running`
    /// when the backend went away.
    pub state: TaskState,
    /// Why the last subtask stopped, for a caller that wants to explain a blocked task.
    pub stop: Stop,
    /// Subtasks this pass closed, in the order they closed.
    pub closed: Vec<String>,
}

/// The inputs a cycle needs that are not policy.
///
/// A struct rather than six parameters (the workspace's argument limit is seven, and seven
/// positional arguments of which three are `u8`-ish is how a call site gets them backwards).
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Attempts one subtask may spend, from `LoopLimits::max_attempts_per_subtask`.
    pub attempts: u32,
    /// Restarts one subtask may spend, from `LoopLimits::max_restarts_per_subtask`.
    pub restarts: u32,
    /// How deep the tree may grow, from `LoopLimits::max_subtask_depth`.
    pub max_depth: u8,
}

impl Limits {
    const fn allowance(self) -> Allowance {
        Allowance {
            attempts_left: self.attempts,
            restarts_left: self.restarts,
        }
    }
}

/// Run a whole task: the lifecycle edges, every subtask, then the task-level gates.
///
/// `plan` is `&mut` because every settled subtask is written back as the run proceeds — a cycle
/// that buffered the tree in memory would lose the work a crash happened after.
///
/// Every journal record is stamped from [`Runner::clock`] at the moment of writing, not from one
/// timestamp captured at the start: an attempt advances the clock by its own steps, and a cycle
/// that wrote its own records at the start time would put them before the work they describe.
///
/// # Errors
/// [`CycleError`]: the tree is stuck, the loop did not converge, or a store failed.
pub fn run(
    runner: &Runner,
    plan: &mut Plan,
    journal: &mut Journal,
    start: &Start,
    limits: Limits,
) -> Result<Cycle, CycleError> {
    // `verifying` is a resume, not a start: every subtask is settled, and the only work left is
    // the verdict. In every other state the cycle has to enter `running` first.
    let mut state = match start.state {
        TaskState::Verifying => TaskState::Verifying,
        other => enter_running(runner, journal, other)?,
    };

    // A subtask the last process left `running` has no worker now: this process holds the task's
    // run slot, and startup abandoned every attempt it found open. `next_ready` only offers
    // `pending` subtasks, so a tree still saying `running` would read as `Stuck` and the task
    // would idle forever — the exact failure `kill -9` recovery exists to survive.
    adopt_leftover_running(plan, journal, runner.clock.now())?;

    // Read once, before the pass. A subtask is settled or blocked the first time it is visited,
    // so no subtask runs twice here and a snapshot cannot go stale under the loop. Reading at all
    // is what makes attempt numbers and budgets survive a process start: the tree's `attempts`
    // field is never written by production, and a live `kill -9` found two attempts under the same
    // number because of it.
    let history = journal_records(journal)?;

    let mut pass = Pass {
        runner,
        plan: &mut *plan,
        journal: &mut *journal,
        limits,
        last: Stop::Done,
        closed: Vec::new(),
        outage_since: start.outage_since,
        history: &history,
    };
    if state != TaskState::Verifying {
        pass.drive(&start.strategy)?;
    }
    let stop = pass.last.clone();
    let closed = pass.closed.clone();
    // The pass's borrow of the journal and the tree ends here, and the verdict needs both.
    drop(pass);

    // An abandoned task is not blocked: nothing was learned about the work, and blocking would
    // make a temporary outage look like a verdict. `running` is where it was, and `start --task`
    // or the scheduler resumes it when the backend returns.
    if matches!(stop, Stop::Abandoned) {
        return Ok(Cycle { state, stop, closed });
    }
    if matches!(stop, Stop::Blocked(_)) {
        state = take(runner, journal, state, TaskTrigger::Block)?;
        return Ok(Cycle { state, stop, closed });
    }
    if state != TaskState::Verifying {
        state = take(runner, journal, state, TaskTrigger::ExecutionFinished)?;
    }
    let green = accept(runner, journal, plan)?;
    let trigger = if green {
        TaskTrigger::GatesPassed
    } else {
        TaskTrigger::GatesFailed
    };
    state = take(runner, journal, state, trigger)?;
    Ok(Cycle { state, stop, closed })
}

/// Put a subtask the last process left `running` back in the queue.
///
/// # Why the tree cannot be trusted to say this itself
///
/// `TASKS.json` is written by the process that runs, and `kill -9` happens between two writes: a
/// subtask marked `running` on disk has no worker the moment the process is gone. Startup
/// abandons every attempt it finds open, which reopens the *attempt* budget but not the tree, and
/// `Plan::next_ready` offers only `pending` subtasks. A tree still saying `running` therefore
/// reads as [`CycleError::Stuck`] and the task idles forever — a `kill -9` that silently ends the
/// run. This process holds the task's run slot, so ownership is not in question.
fn adopt_leftover_running(plan: &mut Plan, journal: &mut Journal, at: Timestamp) -> Result<(), CycleError> {
    let leftovers: Vec<SubtaskId> = plan
        .open_subtasks()
        .into_iter()
        .filter(|subtask| subtask.state == SubtaskState::Running)
        .map(|subtask| subtask.id.clone())
        .collect();
    for id in leftovers {
        plan.set_state(&id, SubtaskState::Pending, at)
            .map_err(|error| CycleError::Store(error.to_string()))?;
        subtask_state(journal, &id, "pending", at)?;
    }
    Ok(())
}

/// Take one lifecycle edge, writing it to the journal.
///
/// The one place an edge is taken, so every transition in a run is recorded and every refusal is
/// reported with the trigger that was attempted.
fn take(
    runner: &Runner,
    journal: &mut Journal,
    state: TaskState,
    trigger: TaskTrigger,
) -> Result<TaskState, CycleError> {
    let next = state
        .trigger(trigger)
        .map_err(|error| CycleError::Store(format!("the lifecycle refused `{trigger}`: {error}")))?;
    record_state(journal, next, runner.clock.now())?;
    Ok(next)
}

/// Move a task into `running`, taking only the edges its current state needs.
///
/// `blocked` and `failed_gates` are resumable: a human (or the scheduler) decided there is more
/// work. `ready_for_review` and later are not — a run there would be a second claim about a
/// finished task.
fn enter_running(runner: &Runner, journal: &mut Journal, state: TaskState) -> Result<TaskState, CycleError> {
    let mut state = state;
    let edges: &[TaskTrigger] = match state {
        TaskState::Running => &[],
        TaskState::Queued => &[
            TaskTrigger::StartPlanning,
            TaskTrigger::PlanWritten,
            TaskTrigger::PlanApproved,
        ],
        // The plan is on disk — the caller loaded it — so only the remaining edges are owed.
        TaskState::Planning => &[TaskTrigger::PlanWritten, TaskTrigger::PlanApproved],
        TaskState::AwaitingPlanApproval => &[TaskTrigger::PlanApproved],
        TaskState::Blocked => &[TaskTrigger::Unblock],
        TaskState::FailedGates => &[TaskTrigger::Retry],
        other => return Err(CycleError::Refused { state: other }),
    };
    for trigger in edges {
        state = take(runner, journal, state, *trigger)?;
    }
    Ok(state)
}

/// One pass over the tree, with the state a pass accumulates.
struct Pass<'a> {
    runner: &'a Runner,
    plan: &'a mut Plan,
    journal: &'a mut Journal,
    limits: Limits,
    /// The task's records as they stood when the pass started.
    ///
    /// Read once, before the loop, and never re-read: this is what a subtask's attempt number and
    /// remaining budget are derived from, and re-reading inside the loop would both cost a file
    /// scan per subtask and let the answer change under a `&mut` journal that is the caller's
    /// proof of what was written. A subtask is visited once per pass, so one snapshot is enough.
    history: &'a [JournalRecord],
    /// Why the last subtask stopped.
    last: Stop,
    /// Subtasks that closed, in order.
    closed: Vec<String>,
    outage_since: Option<Timestamp>,
}

impl Pass<'_> {
    /// Land this subtask's work on the task branch, as one Conventional Commit.
    ///
    /// `None` when the runner has no checkpoint world: a caller that wired none (a test without a
    /// repository) has no branch to commit to, and refusing to close there would make `gates: None`
    /// the only thing a test could exercise.
    ///
    /// # Errors
    /// Git failed. The caller must not close the subtask: a closure recorded in the tree but not
    /// in the branch is the loss D-011's branch exists to prevent.
    fn commit(&self, title: &str) -> Result<(), CycleError> {
        let Some(checkpoints) = self.runner.checkpoints.as_deref() else {
            return Ok(());
        };
        // Conventional Commits because D-011 makes the branch's history the thing a human reads:
        // one `feat:`/`fix:` per subtask says what landed without opening the plan. `chore:` when
        // the title's own prefix is not one of the types, which keeps a planner-written title from
        // producing a commit message no tool can parse.
        let message = conventional(&subtitle(title));
        checkpoints
            .commit(&message)
            .map(|_| ())
            .map_err(|error| CycleError::Store(format!("cannot commit `{title}`: {error}")))
    }

    /// Run subtasks until none may run.
    fn drive(&mut self, strategy: &str) -> Result<(), CycleError> {
        for _ in 0..MAX_SUBTASK_PASSES {
            let Some(subtask) = self.plan.next_ready().cloned() else {
                return match scheduler::next(self.plan) {
                    scheduler::Next::Finished => Ok(()),
                    scheduler::Next::Stuck { open } => Err(CycleError::Stuck {
                        open: open.iter().map(|subtask| subtask.id.to_string()).collect(),
                    }),
                    // `next` and `next_ready` were asked the same question a microsecond apart; a
                    // disagreement means the plan moved without this loop moving it.
                    scheduler::Next::Run(_) => continue,
                };
            };
            self.run_one(&subtask, strategy)?;
            // `Abandoned` and `Blocked` end the whole task, not this subtask: the first means the
            // backend is gone and every further attempt would spend a full request failing, the
            // second that the tree could not be put back — continuing would build on a state
            // nobody knows.
            if matches!(self.last, Stop::Abandoned | Stop::Blocked(_)) {
                return Ok(());
            }
        }
        Err(CycleError::DidNotConverge)
    }

    /// Run one subtask to a settled state, or record the stop that ends the task.
    fn run_one(&mut self, subtask: &Subtask, strategy: &str) -> Result<(), CycleError> {
        let now = self.runner.clock.now();
        self.plan
            .set_state(&subtask.id, SubtaskState::Running, now)
            .map_err(|error| CycleError::Store(format!("cannot mark {} running: {error}", subtask.id)))?;
        let mut running = self
            .plan
            .subtask(&subtask.id)
            .cloned()
            .ok_or_else(|| CycleError::Store(format!("{} vanished from the tree", subtask.id)))?;
        running.state = SubtaskState::Running;

        // From the journal, not the tree: `Subtask::attempts` is never written by production, and
        // a live `kill -9` proved the consequence — two `attempt_started` records under the same
        // `(subtask, 1)` key, and a resumed run with the whole budget back. `Spent` is what makes
        // the module doc's promise ("a resumed subtask continues counting attempts") true.
        let spent = Spent::read(self.history, self.journal.task(), &running.id);
        let mut session = self
            .runner
            .session(&running, spent.next)
            .with_strategy(strategy)
            // A resumed subtask inherits what its last attempt learned. Without this its first
            // brief in the new process is as blind as the very first one. The report is written
            // to the tree when a subtask closes (`set_report`), and an *open* subtask has none:
            // its accumulated ideas live only in the session, which is the honest limit — a
            // process that died mid-subtask has nothing durable to inherit from, and the journal
            // is where the reason it died is recorded.
            .resuming_from(running.report.clone().unwrap_or_default());
        let end = session
            .drive(spent.left(self.limits.allowance()), self.journal)
            .map_err(|error| CycleError::Store(error.to_string()))?;

        match &end.stop {
            Stop::Done => {
                // The commit comes first, and it is what makes the closure mean something: the
                // tree records `done`, but `agent/<task>` is what a human merges, and a `done`
                // subtask whose work never reached that branch is the loss this whole milestone is
                // about. Measured on three runs — `word-count`, `word-count-fresh`, `kill-test` —
                // where `agent/<task>` sat on `main`'s commit with the work only in the worktree
                // and in `agent-ckpt/<task>`; `immortality merge` then reported `merged` having
                // merged nothing.
                //
                // Before the tree write, so a crash between them leaves work committed under a
                // subtask the tree still lists as open: replaying the subtask commits again (an
                // empty second commit that `commit` answers with `Ok(None)`), which is harmless.
                // The other order would leave a `done` whose work is nowhere, which nothing can
                // repair.
                self.commit(&subtask.title)?;
                // Tree first, then journal. A crash between the two leaves a settled tree with a
                // missing audit line, which reads as "nothing to say"; the other order announces a
                // closure the tree never received.
                //
                // The report goes in with the state, not after it: it is what the next subtask
                // inherits, and `rejected_ideas` is what a resumed attempt of *this* subtask
                // inherits. Leaving it out is what made `resuming_from` below always receive
                // `None`, so a restarted process re-proposed ideas the previous attempts had
                // already ruled out.
                self.plan
                    .set_report(
                        &subtask.id,
                        SubtaskState::Done,
                        end.report.clone(),
                        self.runner.clock.now(),
                    )
                    .map_err(|error| CycleError::Store(error.to_string()))?;
                self.subtask_state(&subtask.id, "done")?;
                self.closed.push(subtask.id.to_string());
                // A closed subtask is proof the backend answered.
                self.restored()?;
            }
            Stop::Split => {
                let pieces = match &end.outcome.ending {
                    Ending::SplitRequested(pieces) => pieces.clone(),
                    other => {
                        return Err(CycleError::Store(format!(
                            "a split stop without pieces: {other:?}"
                        )));
                    }
                };
                // The tree decides whether the split is legal: the model proposes, and a model
                // that could split itself would be choosing the shape of a tree it is not allowed
                // to see.
                scheduler::apply_split(
                    self.plan,
                    self.journal,
                    &subtask.id,
                    &pieces,
                    self.limits.max_depth,
                    self.runner.clock.now(),
                )
                .map_err(|error| CycleError::Store(error.to_string()))?;
            }
            Stop::Exhausted(_) => {
                // The report matters here at least as much as on `done`: a subtask that is out
                // of attempts is the one a human is most likely to reopen by hand, and the ideas
                // that already failed are exactly what its next attempt must not repeat.
                self.plan
                    .set_report(
                        &subtask.id,
                        SubtaskState::Blocked,
                        end.report.clone(),
                        self.runner.clock.now(),
                    )
                    .map_err(|error| CycleError::Store(error.to_string()))?;
                self.subtask_state(&subtask.id, "blocked")?;
            }
            Stop::Abandoned => {
                // Nothing was learned about the work: the model never answered. Leaving the
                // subtask `running` would claim a worker no process owns, and `next_ready`
                // skips a running subtask — so the resume this state exists for would find no
                // work and fail `Stuck`. `pending` is the honest state: the work is not done,
                // the attempt was not spent against the budget, and the worktree still holds
                // whatever the last step left.
                self.plan
                    .set_state(&subtask.id, SubtaskState::Pending, self.runner.clock.now())
                    .map_err(|error| CycleError::Store(error.to_string()))?;
                self.subtask_state(&subtask.id, "pending")?;
                self.unavailable()?;
            }
            Stop::Blocked(_) => {}
        }
        self.last = end.stop;
        Ok(())
    }

    /// Journal that the backend is gone, once per continuous outage.
    ///
    /// `since` is kept from the previous record — or from the journal, through [`Start`] — so the
    /// fifteen-minute escalation measures one outage rather than the last attempt. The
    /// record itself is what `scheduling` reads to put the task on `waiting_backend`, which is why
    /// it is written here and not in a scheduler loop that may never run.
    fn unavailable(&mut self) -> Result<(), CycleError> {
        let now = self.runner.clock.now();
        if self.outage_since.is_some() {
            // Already recorded: a record per attempt is noise, and the hold is already visible.
            return Ok(());
        }
        self.outage_since = Some(now);
        self.journal
            .append_event(
                now,
                Kind::BackendUnavailable,
                backend::unavailable_payload(now, now, 0),
            )
            .map(|_| ())
            .map_err(|error| CycleError::Store(error.to_string()))
    }

    /// Journal that the backend is back, if an outage was open.
    fn restored(&mut self) -> Result<(), CycleError> {
        if self.outage_since.take().is_none() {
            return Ok(());
        }
        let now = self.runner.clock.now();
        self.journal
            .append_event(
                now,
                Kind::BackendRestored,
                // The key is the same one `unavailable_payload` writes, which is what makes the
                // pair read as one outage closed rather than two unrelated facts.
                serde_json::json!({ "service": backend::KEY, "at": immortality_core::time::truncate(now) }),
            )
            .map(|_| ())
            .map_err(|error| CycleError::Store(error.to_string()))
    }

    fn subtask_state(&mut self, id: &SubtaskId, state: &str) -> Result<(), CycleError> {
        subtask_state(self.journal, id, state, self.runner.clock.now())
    }
}

/// The task's acceptance bar, as gates.
///
/// Not `Runner::settle_claim`: that judges an *attempt* against its own `DoD` and writes at
/// attempt scope. This is the bar the whole task must meet before `ready_for_review`, and its
/// records belong to the task, so recovery can tell "the task was verified" from "an attempt was".
fn accept(runner: &Runner, journal: &mut Journal, plan: &Plan) -> Result<bool, CycleError> {
    let Some(gates) = runner.gates.as_deref() else {
        // Fail closed, with a reason that names the wiring bug rather than the work. A task that
        // reported `ready_for_review` because nothing could check it would be the one lie this
        // whole design exists to prevent.
        return Err(CycleError::Store(
            "the task has acceptance gates but the runner was wired without any; \
             nothing could verify the work"
                .to_string(),
        ));
    };
    // Upcast explicitly: `evaluate` wants a plain `&dyn Reviewer`, and a `&dyn Reviewer +
    // Send + Sync` is a different trait object even though it is a superset.
    let reviewer: Option<&dyn Reviewer> = runner
        .reviewer
        .as_ref()
        .map(|reviewer| reviewer.as_ref() as &dyn Reviewer);
    let report = evaluate(&acceptance(plan), gates, reviewer, None)
        .map_err(|error| CycleError::Store(format!("the acceptance gate could not run: {error}")))?;
    for record in GateRecord::of(&report) {
        let payload = serde_json::to_value(&record)
            .map_err(|error| CycleError::Store(format!("cannot encode a gate record: {error}")))?;
        journal
            .append_scoped(runner.clock.now(), Kind::GateResult, payload, &Scope::task())
            .map_err(|error| CycleError::Store(error.to_string()))?;
    }
    Ok(report.is_green())
}

/// The definition of done the task as a whole is judged against.
///
/// Built from the plan, which is the durable statement of what the task promised: every required
/// check any subtask declared, deduplicated by id, plus one review. A fixed two-command list in
/// the cycle would make "the task passed" mean "cargo build and cargo test passed" even when the
/// plan asked for something else — and the plan is what a human approved.
fn acceptance(plan: &Plan) -> Dod {
    let mut checks = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for subtask in plan.subtasks() {
        for check in &subtask.dod.checks {
            if check.required && seen.insert(check.id.clone()) {
                checks.push(check.clone());
            }
        }
    }
    Dod {
        statement: "every required check the plan declared passes".to_string(),
        checks,
        review: true,
    }
}

/// A subtask title as a one-line Conventional Commit subject.
///
/// D-011 makes `agent/<task>` the history a human reviews before merging, so one commit per
/// subtask has to read as what happened. The planner writes titles as short imperatives
/// ("parse the input") and the commit type is not in them, so `chore:` is the honest default: it
/// says "the loop did this", not that the work was a feature. A title that already starts with a
/// type the repository uses keeps it — a planner that wrote `fix: …` said more than `chore:` can.
fn conventional(title: &str) -> String {
    const TYPES: [&str; 11] = [
        "feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert",
    ];
    let looks_typed = title
        .split_once(':')
        .is_some_and(|(head, _)| TYPES.contains(&head.trim()));
    if looks_typed {
        return title.to_string();
    }
    format!("chore: {title}")
}

/// A single line: a commit subject with a newline in it silently truncates at git's hands.
fn subtitle(title: &str) -> String {
    let one_line = title.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.is_empty() {
        "close a subtask".to_string()
    } else {
        one_line
    }
}

/// The records of this task's journal, read from disk without disturbing the writer.
///
/// A second handle rather than a method on [`Journal`]: the open `&mut Journal` is the caller's
/// proof that every effect was written, and reading through it would need a `&mut` borrow that
/// fights the one the pass holds. `JournalReader` reads the same files and closes over nothing.
fn journal_records(journal: &Journal) -> Result<Vec<JournalRecord>, CycleError> {
    JournalReader::open(journal.dir(), journal.task())
        .and_then(|reader| reader.read())
        .map(|outcome| outcome.records)
        .map_err(|error| CycleError::Store(format!("cannot read the journal for the cycle: {error}")))
}

fn record_state(journal: &mut Journal, state: TaskState, at: Timestamp) -> Result<(), CycleError> {
    journal
        .append_event(
            at,
            Kind::TaskState,
            serde_json::json!({ "state": state.as_str() }),
        )
        .map(|_| ())
        .map_err(|error| CycleError::Store(error.to_string()))
}

fn subtask_state(
    journal: &mut Journal,
    id: &SubtaskId,
    state: &str,
    at: Timestamp,
) -> Result<(), CycleError> {
    journal
        .append_scoped(
            at,
            Kind::SubtaskState,
            serde_json::json!({ "subtask": id.to_string(), "state": state }),
            &Scope::subtask(id.clone()),
        )
        .map(|_| ())
        .map_err(|error| CycleError::Store(error.to_string()))
}
