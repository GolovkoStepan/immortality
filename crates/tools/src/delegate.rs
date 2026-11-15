//! `delegate` — hand one brief to a sub-agent that starts with a clean context.
//!
//! # Why a sub-agent at all
//!
//! The design's unit of context is the subtask: when a piece of work closes, its conversation is
//! dropped and only a structured report survives. `delegate` is that mechanism as a tool, usable
//! mid-attempt. It exists because the alternative — doing everything in one conversation — is
//! how a weak model spends its window on a log tail and then forgets the goal. A delegated run
//! begins with only its brief, its scope and the task's files.
//!
//! # What this tool owns, and what it does not
//!
//! It owns the *contract*: the brief, the scope, the step budget, the durable report, and the
//! refusal of nesting. It does not own the loop. [`SubAgent`] is a seam the daemon fills with a
//! real runner (its own model client, its own tool registry); in a test it is a fake that returns
//! a report in microseconds. Everything this module can get wrong on its own — running the same
//! brief twice, letting a sub-agent delegate again, accepting a report stamped by nothing — is
//! enforced here rather than left to the runner.
//!
//! # Why the report is written before the tool returns
//!
//! `delegate` is `Unknown` idempotency: a crash between the sub-agent finishing and the journal
//! recording the result leaves no way to know whether a whole delegated run happened. Recovery
//! probes, and the probe looks for the report file under the id derived from `call_id`
//! (D-068). Writing it first is the direction that makes the probe true: `Applied` means the
//! work really is on disk, and a retry is refused instead of spending the budget twice.
//!
//! # Why a repeat call is answered from the file
//!
//! The tool checks for an existing report *before* invoking the runner. That covers the case
//! the probe does not: recovery decided to retry (or the model repeated the exact call), and the
//! sub-agent must not run a second time. Answering from the file is also how the parent sees the
//! result after a restart, when the original observation is gone.
//!
//! # Why nesting is refused
//!
//! A sub-agent that could delegate is an unbounded recursion with a token bill attached, and the
//! design caps the task tree at depth 2. The daemon marks a sub-agent's context with
//! [`ToolCtx::with_depth`]; this tool refuses at any depth above zero. The check is here, not in
//! the runner, because the runner is the thing being constrained.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use immortality_core::delegation::{
    DelegationId, DelegationReport, DelegationStore, Finish, normalise_scope, within_scope,
};
use immortality_core::domain::{SubtaskReport, Timestamp};
use immortality_core::ids::TaskId;
use immortality_core::protocol::{Idempotency, Probe};
use immortality_core::time;
use serde_json::{Value, json};

use crate::Effect;
use crate::args::Args;
use crate::policy::PathUse;
use crate::tool::{Tool, ToolCtx, ToolError, ToolResult};

/// Steps a sub-agent gets when the model does not say.
///
/// A third of the parent's `max_steps_per_attempt` of 60: a delegated brief is one coherent
/// piece of work, not a second attempt, and a default that could eat the parent's whole budget
/// would make the parent unable to finish what it started.
pub const DEFAULT_STEPS: u32 = 20;

/// Hard ceiling on `max_steps`, whatever the model asks for.
///
/// Two thirds of an attempt. Above this the brief is not a brief; the model should split it into
/// subtasks, where the scheduler's own budgets and retry accounting apply.
pub const MAX_STEPS: u32 = 40;

/// Longest brief accepted, in bytes.
pub const MAX_BRIEF_LEN: usize = 8_000;

/// Most scope entries accepted.
pub const MAX_SCOPE_ENTRIES: usize = 32;

/// One delegated run, as the runner receives it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delegation {
    /// Stable for the life of the call, so the report can be found after a crash.
    pub id: DelegationId,
    /// What to do, in full. The sub-agent sees nothing else from the parent's conversation.
    pub brief: String,
    /// Worktree-relative prefixes the sub-agent should confine itself to.
    pub scope: Vec<String>,
    /// Steps the sub-agent may take. The runner enforces this; it is stated here so the caller
    /// and the report agree on one number.
    pub max_steps: u32,
    /// The task the work belongs to, for the sub-agent's own journal records.
    pub task: TaskId,
    /// The worktree the sub-agent operates in — the *same* one as the parent, by design.
    pub worktree: PathBuf,
    /// Where the sub-agent writes its own journal.
    ///
    /// Not the task's journal: [`immortality_core::journal::Journal`] is single-writer, and the
    /// parent holds its writer for the whole delegated call. A second writer on the same segment
    /// would interleave two chains and break the digest the recovery reader checks, so the
    /// sub-agent gets `delegations/<del_id>/` instead — beside its report, under the same id.
    ///
    /// The consequence is deliberate and worth stating: the parent's recovery sees the `delegate`
    /// call as one unit, never the sub-agent's steps. That is the right unit anyway — re-running
    /// the call re-runs the brief — and it is why the report, not the sub-agent's journal, is what
    /// the probe looks for.
    pub journal_dir: PathBuf,
}

/// What a runner produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubAgentOutcome {
    /// Steps actually used.
    pub steps: u32,
    /// How the run ended.
    pub finish: Finish,
    /// The structured result the parent inherits.
    pub report: SubtaskReport,
    /// Why it failed, when it did.
    pub error: Option<String>,
}

impl SubAgentOutcome {
    /// A run that finished its brief.
    #[must_use]
    pub const fn done(steps: u32, report: SubtaskReport) -> Self {
        Self {
            steps,
            finish: Finish::Done,
            report,
            error: None,
        }
    }

    /// A run that stopped short, with the reason.
    #[must_use]
    pub fn unfinished(steps: u32, finish: Finish, report: SubtaskReport, error: impl Into<String>) -> Self {
        Self {
            steps,
            finish,
            report,
            error: Some(error.into()),
        }
    }
}

/// Something that can carry out a delegated brief.
///
/// A trait, not a function pointer, because the daemon's runner needs the whole tool registry,
/// the model client and a journal, and a test needs none of those. It is also the only place the
/// sub-agent's own policy is enforced: this tool cannot make a runner obey the scope.
pub trait SubAgent: Send + Sync + std::fmt::Debug {
    /// Run `delegation` to completion or to its step budget.
    ///
    /// # Errors
    /// The run could not start at all — no model, no worktree. An error here writes no report,
    /// so recovery's probe answers `NotApplied` and the call may be retried. A run that started
    /// and then failed belongs in [`SubAgentOutcome`] with `Finish::Failed`, which does write a
    /// report.
    fn run(&self, delegation: &Delegation) -> Result<SubAgentOutcome, ToolError>;

    /// Whether this is a real runner rather than the placeholder.
    ///
    /// Defaulted to `true` so a real implementation says nothing; [`NoSubAgent`] overrides it.
    /// It exists for one assertion that is otherwise impossible to make cheaply: that the
    /// production path installed a runner rather than leaving the slot at its refusing default,
    /// which would look exactly like a model that never called `delegate`.
    fn is_wired(&self) -> bool {
        true
    }
}

/// A `SubAgent` that is installed after the tool is built.
///
/// # Why a slot and not a constructor argument
///
/// `delegate` is a tool inside the [`crate::Registry`], and the real runner holds that registry:
/// `Runner { registry: Arc<Registry> }`. Building the runner therefore needs the registry, and
/// building the tool therefore cannot need the runner. Something has to break the cycle, and a
/// slot does it without either side knowing about the other.
///
/// The alternative — passing the runner in and letting `toolset` depend on the daemon — is the
/// dependency this crate exists to avoid.
///
/// # Why `RwLock` and not `OnceLock`
///
/// `OnceLock` would make a second install a silent no-op. A run that re-wires its sub-agent (a
/// resumed task with a fresh model client, a test replacing the fake) would then keep delegating
/// to the first one, and nothing would say so. Replacing is the honest operation, and it is
/// cheap: the lock is taken once per `delegate` call, never per step.
///
/// The slot starts at [`NoSubAgent`], which refuses. A build that never installs one gets an
/// error the model can act on rather than a hang or a fabricated report.
#[derive(Debug)]
pub struct SubAgentSlot {
    inner: std::sync::RwLock<Arc<dyn SubAgent>>,
}

impl Default for SubAgentSlot {
    fn default() -> Self {
        Self::new()
    }
}

impl SubAgentSlot {
    /// A slot with no runner installed yet.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: std::sync::RwLock::new(Arc::new(NoSubAgent)),
        }
    }

    /// A slot holding `runner` from the start.
    #[must_use]
    pub fn filled(runner: Arc<dyn SubAgent>) -> Self {
        Self {
            inner: std::sync::RwLock::new(runner),
        }
    }

    /// Install `runner`, replacing whatever was there.
    ///
    /// Poisoning is recovered from rather than propagated, for the same reason as [`Self::current`]:
    /// the value is one `Arc` assignment, so a panic in some other call cannot have left it torn,
    /// and taking the daemon down over a lock whose contents are still a valid runner would turn a
    /// recoverable panic elsewhere into a dead task.
    pub fn install(&self, runner: Arc<dyn SubAgent>) {
        let mut guard = self
            .inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *guard = runner;
    }

    /// The runner currently installed.
    fn current(&self) -> Arc<dyn SubAgent> {
        // Poisoning is ignored for a read: the value is an `Arc` to a runner, and a panic in a
        // *different* call cannot have left it half-written. Recovering beats taking the whole
        // task down over a lock whose contents are still valid.
        Arc::clone(
            &self
                .inner
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    /// Whether a real runner is installed, for tests and for `status`.
    #[must_use]
    pub fn is_wired(&self) -> bool {
        self.current().is_wired()
    }
}

/// A `SubAgent` that refuses. The honest default before the daemon has wired one.
///
/// Answering with an empty successful report would be worse: the parent would believe a brief
/// was carried out.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoSubAgent;

impl SubAgent for NoSubAgent {
    fn run(&self, _delegation: &Delegation) -> Result<SubAgentOutcome, ToolError> {
        Err(ToolError::Refused(
            json!({
                "error": "no_sub_agent",
                "hint": "This build has no sub-agent runner wired in. Do the work yourself, \
                         or split it with the planner.",
            })
            .to_string(),
        ))
    }

    fn is_wired(&self) -> bool {
        false
    }
}

fn delegate_schema() -> &'static Value {
    static SCHEMA: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "brief": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_BRIEF_LEN,
                    "description": "Everything the sub-agent needs. It starts with an empty context: it does not see this conversation, so name the files, the goal and the definition of done."
                },
                "scope": {
                    "type": "array",
                    "maxItems": MAX_SCOPE_ENTRIES,
                    "items": { "type": "string", "minLength": 1 },
                    "description": "Worktree-relative path prefixes the sub-agent should stay inside, e.g. [\"src/parser\"]. Files outside them are reported back as violations."
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_STEPS,
                    "description": "Step budget for the sub-agent. Defaults to 20. A brief that needs more is too big; split it."
                }
            },
            "required": ["brief", "scope"],
            "additionalProperties": false
        })
    })
}

/// Hand a brief to a sub-agent with a clean context.
pub struct Delegate {
    state_root: PathBuf,
    runner: RunnerSource,
}

/// Where `Delegate` gets its sub-agent: fixed at construction, or installed later.
///
/// Both cases are real. A test builds the tool over a fake and never changes it; the daemon
/// cannot, because the real runner owns the registry this tool lives in, so it installs the
/// runner into a [`SubAgentSlot`] after both exist. Keeping both shapes here rather than making
/// every test build a slot is what keeps the common case honest — `Delegate::new` cannot silently
/// end up unanswered.
enum RunnerSource {
    Fixed(Arc<dyn SubAgent>),
    Slot(Arc<SubAgentSlot>),
}

impl RunnerSource {
    fn get(&self) -> Arc<dyn SubAgent> {
        match self {
            Self::Fixed(runner) => Arc::clone(runner),
            Self::Slot(slot) => slot.current(),
        }
    }
}

impl std::fmt::Debug for Delegate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Delegate")
            .field("state_root", &self.state_root)
            .field("runner", &self.runner.get())
            .finish()
    }
}

impl Delegate {
    /// A tool whose reports live under `state_root/<task>/delegations`.
    #[must_use]
    pub fn new(state_root: impl Into<PathBuf>, runner: Arc<dyn SubAgent>) -> Self {
        Self {
            state_root: state_root.into(),
            runner: RunnerSource::Fixed(runner),
        }
    }

    /// A tool whose runner is installed after the registry is built.
    ///
    /// The daemon's shape: `Runner` holds `Arc<Registry>`, this tool lives in that registry, and
    /// the runner cannot exist first.
    #[must_use]
    pub fn deferred(state_root: impl Into<PathBuf>, slot: Arc<SubAgentSlot>) -> Self {
        Self {
            state_root: state_root.into(),
            runner: RunnerSource::Slot(slot),
        }
    }

    /// A tool wired to [`NoSubAgent`], for a build without a runner.
    #[must_use]
    pub fn disabled(state_root: impl Into<PathBuf>) -> Self {
        Self::new(state_root, Arc::new(NoSubAgent))
    }

    /// The store for a task, which is where the report lives.
    fn store(&self, task: &TaskId) -> DelegationStore {
        DelegationStore::for_task(&self.state_root, task)
    }
}

impl Tool for Delegate {
    fn name(&self) -> &'static str {
        "delegate"
    }

    fn schema(&self) -> &'static Value {
        delegate_schema()
    }

    fn description(&self) -> &'static str {
        "Hand one self-contained brief to a sub-agent that starts with a clean context in the same worktree. It does not see this conversation: put everything it needs in `brief`. Returns a structured report. A sub-agent cannot delegate further."
    }

    fn effect(&self, _args: &Args) -> Effect {
        // `Mutate`, not `External`: the call writes one report under the daemon's state
        // directory and touches nothing outside it. The sub-agent's own file, git and shell
        // calls are gated one by one, as they would be for any other agent.
        Effect::Mutate
    }

    fn idempotency(&self, _args: &Args) -> Idempotency {
        // The effect is a whole delegated run, which cannot be undone by repeating it. The
        // report makes it probeable instead (D-068), so recovery asks before it retries.
        Idempotency::Unknown
    }

    fn paths(&self, _args: &Args) -> Vec<PathUse> {
        // Nothing in the worktree. The report is under the state root, which the gate does not
        // govern and the model cannot name.
        Vec::new()
    }

    /// The first clause of the brief. A delegated run can cost twenty steps, so the log has to
    /// show what the sub-agent was asked to do before it starts, not after its report lands.
    fn expected_effect(&self, args: &Args) -> String {
        let brief = args.str("brief").unwrap_or("");
        crate::tool::one_line(
            &format!("delegate to a sub-agent: {brief}"),
            crate::tool::EFFECT_LIMIT,
        )
    }

    fn execute(&self, ctx: &ToolCtx, args: Args) -> Result<ToolResult, ToolError> {
        refuse_nested(ctx)?;

        let brief = read_brief(&args)?;
        let scope = read_scope(&args)?;
        let max_steps = read_max_steps(&args)?;

        // A report is addressed by the call that asked for it, and the call id is the only part
        // of this call that survives a crash. Without one there is no address, so there is no
        // safe retry — better to refuse than to run something nothing can look up.
        let call_id = ctx.call().ok_or_else(|| {
            ToolError::Refused(
                json!({
                    "error": "no_call_id",
                    "hint": "A delegated run is addressed by its call id so recovery can find \
                             its report. This call arrived without one, which is a daemon bug \
                             rather than something to retry.",
                })
                .to_string(),
            )
        })?;
        let task = ctx.task().ok_or_else(|| {
            ToolError::Refused(
                json!({
                    "error": "no_task",
                    "hint": "A delegation belongs to a task; that is where its report lives. \
                             This call has no task context, which is a daemon bug.",
                })
                .to_string(),
            )
        })?;
        let store = self.store(task);
        let id = DelegationId::for_call(call_id);

        // Already ran: recovery retried, or the model repeated itself. Either way the
        // sub-agent must not run twice, and the parent must still get the report.
        if let Some(report) = store.read(&id).map_err(|error| store_error(&error))? {
            return Ok(completed_result(&report, true));
        }

        let delegation = Delegation {
            id: id.clone(),
            brief: brief.clone(),
            scope: scope.clone(),
            max_steps,
            task: task.clone(),
            worktree: ctx.gate().worktree().to_path_buf(),
            // Beside the report, under the same id: the sub-agent's journal is readable on its
            // own and never competes with the parent's single writer.
            journal_dir: store.dir().join(id.as_str()),
        };

        let outcome = self.runner.get().run(&delegation).map_err(|error| {
            // No report is written: nothing durable happened, so the probe correctly says
            // `NotApplied` and recovery may retry the call.
            tracing::debug!(%id, "delegated run did not start: {error}");
            error
        })?;

        // Any disagreement between the budget and what the runner used is a runner bug, and
        // the durable record must show the truth: the budget is not a suggestion.
        let (finish, error) = if outcome.steps > max_steps {
            (
                Finish::Exhausted,
                Some(match outcome.error {
                    Some(inner) => format!(
                        "the runner used {} steps of a {max_steps}-step budget; {inner}",
                        outcome.steps
                    ),
                    None => format!(
                        "the runner used {} steps of a {max_steps}-step budget",
                        outcome.steps
                    ),
                }),
            )
        } else {
            (outcome.finish, outcome.error)
        };

        let report = DelegationReport::new(id, brief, scope, max_steps, now()).with_outcome(
            outcome.steps,
            finish,
            outcome.report,
            error,
        );
        store.write(&report).map_err(|error| store_error(&error))?;
        Ok(completed_result(&report, false))
    }
}

/// Refuse a delegated run from inside a delegated run.
fn refuse_nested(ctx: &ToolCtx) -> Result<(), ToolError> {
    if ctx.depth() == 0 {
        return Ok(());
    }
    Err(ToolError::Refused(
        json!({
            "error": "nested_delegation",
            "depth": ctx.depth(),
            "hint": "A sub-agent cannot delegate further: the task tree is two levels deep. \
                     Do this part of the brief yourself, and report what did not fit.",
        })
        .to_string(),
    ))
}

/// The brief, trimmed and bounded.
fn read_brief(args: &Args) -> Result<String, ToolError> {
    let brief = args.str("brief")?.trim().to_string();
    if brief.is_empty() {
        return Err(ToolError::Arguments(
            "`brief` is empty. The sub-agent starts with a clean context, so the brief is all it \
             knows: name the goal, the files and the definition of done."
                .to_string(),
        ));
    }
    if brief.len() > MAX_BRIEF_LEN {
        return Err(ToolError::Arguments(format!(
            "`brief` is {} bytes; the limit is {MAX_BRIEF_LEN}. A brief that large is a plan; \
             split it into subtasks.",
            brief.len()
        )));
    }
    Ok(brief)
}

/// The scope, normalised, refusing entries that would widen it.
fn read_scope(args: &Args) -> Result<Vec<String>, ToolError> {
    let Some(Value::Array(items)) = args.raw().get("scope") else {
        return Err(ToolError::Arguments(
            "`scope` must be an array of worktree-relative path prefixes, e.g. [\"src/parser\"]".to_string(),
        ));
    };
    if items.len() > MAX_SCOPE_ENTRIES {
        return Err(ToolError::Arguments(format!(
            "`scope` has {} entries; the limit is {MAX_SCOPE_ENTRIES}",
            items.len()
        )));
    }
    let mut scope = Vec::with_capacity(items.len());
    for item in items {
        let Some(raw) = item.as_str() else {
            return Err(ToolError::Arguments(format!(
                "`scope` must contain strings, got {item}"
            )));
        };
        let Some(normalised) = normalise_scope(raw) else {
            // Refused rather than dropped: a model that wrote `/etc` or `../..` meant
            // something, and silently narrowing it would hide that from the human reading
            // the journal.
            return Err(ToolError::Arguments(format!(
                "`{raw}` is not a usable scope: it must be relative to the worktree and stay \
                 inside it. `~`, absolute paths and `..` are refused because a scope narrows \
                 what a sub-agent may touch."
            )));
        };
        scope.push(normalised);
    }
    Ok(scope)
}

/// The step budget, defaulted and bounded.
fn read_max_steps(args: &Args) -> Result<u32, ToolError> {
    let requested = args.opt_u64("max_steps")?;
    let Some(requested) = requested else {
        return Ok(DEFAULT_STEPS);
    };
    if requested == 0 {
        return Err(ToolError::Arguments(
            "`max_steps` is 0. A sub-agent that may take no steps cannot do anything; use 1 at \
             least."
                .to_string(),
        ));
    }
    if requested > u64::from(MAX_STEPS) {
        return Err(ToolError::Arguments(format!(
            "`max_steps` is {requested}; the limit is {MAX_STEPS}. A brief that needs more steps \
             is a subtask, not a delegation — split it."
        )));
    }
    Ok(u32::try_from(requested).unwrap_or(MAX_STEPS))
}

/// The clock, in one place, so a test sees the value the report carries.
fn now() -> Timestamp {
    time::now()
}

/// The observation for a report on disk.
///
/// Used both for a fresh run and for a repeat answered from the file. `replayed` is what tells
/// the model the difference: the work happened in an earlier step, and it must not be redone.
fn completed_result(report: &DelegationReport, replayed: bool) -> ToolResult {
    use std::fmt::Write as _;

    let mut observation = String::new();
    if replayed {
        let _ = writeln!(
            observation,
            "Delegation {} had already completed; its report follows and the sub-agent was NOT \
             run again.\n",
            report.id
        );
    } else {
        let _ = writeln!(
            observation,
            "Delegation {} finished ({}).\n",
            report.id,
            report.finish.as_str()
        );
    }
    let _ = writeln!(observation, "Outcome: {}", report.report.outcome);

    // Each block is skipped when empty, so a bare report does not print three bare headings.
    write_list(&mut observation, "Facts", &report.report.facts);
    write_list(
        &mut observation,
        "Files touched",
        &report
            .report
            .files_touched
            .iter()
            .map(|file| file.display().to_string())
            .collect::<Vec<_>>(),
    );
    write_list(
        &mut observation,
        "Rejected approaches (do not repeat them)",
        &report.report.rejected_ideas,
    );
    if let Some(error) = &report.error {
        let _ = writeln!(observation, "Problem: {error}");
    }

    let outside = scope_violations(report);
    if !outside.is_empty() {
        // The files are already written; the honest response is to name them, not to pretend
        // the scope held. The parent decides whether to keep, revert or escalate.
        write_list(
            &mut observation,
            "WARNING: the sub-agent touched paths outside the scope it was given",
            &outside,
        );
        observation
            .push_str("Review these changes before committing. Nothing outside the scope was reverted.\n");
    }
    if report.finish == Finish::Exhausted {
        observation.push_str(
            "The step budget ran out before the brief was finished. Do not delegate the same \
             brief again; split it into smaller pieces or finish the rest yourself.\n",
        );
    }

    ToolResult::new(
        observation,
        json!({
            "delegation": report.id,
            "finish": report.finish,
            "steps": report.steps,
            "max_steps": report.max_steps,
            "replayed": replayed,
            "scope": report.scope,
            "scope_violations": outside,
            "report": report.report,
        }),
    )
}

/// Append `heading:` and one `- item` line per entry, or nothing when there are none.
fn write_list(observation: &mut String, heading: &str, items: &[String]) {
    use std::fmt::Write as _;
    if items.is_empty() {
        return;
    }
    let _ = writeln!(observation, "{heading}:");
    for item in items {
        let _ = writeln!(observation, "- {item}");
    }
}

/// The touched files that fall outside the scope, as strings for the observation.
fn scope_violations(report: &DelegationReport) -> Vec<String> {
    report
        .report
        .files_touched
        .iter()
        .filter(|file| !within_scope(file, &report.scope))
        .map(|file| file.display().to_string())
        .collect()
}

/// Turn a storage error into the message the model sees.
fn store_error(error: &immortality_core::Error) -> ToolError {
    ToolError::Refused(
        json!({
            "error": "delegation_store_unavailable",
            "detail": error.to_string(),
            "hint": "The report could not be recorded, so the delegated run is not recoverable. \
                     This is a storage problem, not a policy decision; report it if it persists.",
        })
        .to_string(),
    )
}

/// Probe whether a `delegate` call took effect, for crash recovery.
///
/// Derived from the call id alone, because that is all recovery has. A corrupt report is
/// `Unknown` and not `NotApplied`: the run did happen, and repeating it would spend a whole
/// sub-agent budget twice.
#[must_use]
pub fn probe_delegated(state_root: &Path, task: &TaskId, call_id: &str) -> Probe {
    let store = DelegationStore::for_task(state_root, task);
    let id = DelegationId::for_call(call_id);
    match store.read(&id) {
        Ok(Some(_)) => Probe::Applied,
        Ok(None) => Probe::NotApplied,
        Err(error) => {
            tracing::warn!(%id, "a delegation report exists but cannot be read: {error}");
            Probe::Unknown
        }
    }
}

/// Every report for a task, for a CLI listing.
///
/// # Errors
/// The directory cannot be listed, or a report in it is corrupt.
pub fn listing(state_root: &Path, task: &TaskId) -> immortality_core::Result<Vec<DelegationReport>> {
    DelegationStore::for_task(state_root, task).list()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A runner that records the delegations it saw and answers a scripted outcome.
    #[derive(Debug)]
    struct FakeRunner {
        calls: Mutex<Vec<Delegation>>,
        answer: Mutex<Option<SubAgentOutcome>>,
        error: Mutex<Option<String>>,
    }

    impl FakeRunner {
        fn new() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                answer: Mutex::new(None),
                error: Mutex::new(None),
            }
        }

        fn answering(outcome: SubAgentOutcome) -> Self {
            let runner = Self::new();
            *runner.answer.lock().unwrap() = Some(outcome);
            runner
        }

        fn failing(message: &str) -> Self {
            let runner = Self::new();
            *runner.error.lock().unwrap() = Some(message.to_string());
            runner
        }

        fn runs(&self) -> usize {
            self.calls.lock().unwrap().len()
        }

        fn last(&self) -> Delegation {
            self.calls.lock().unwrap().last().cloned().unwrap()
        }
    }

    impl SubAgent for FakeRunner {
        fn run(&self, delegation: &Delegation) -> Result<SubAgentOutcome, ToolError> {
            self.calls.lock().unwrap().push(delegation.clone());
            let failure = self.error.lock().unwrap().clone();
            if let Some(message) = failure {
                return Err(ToolError::Refused(message));
            }
            Ok(self
                .answer
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| SubAgentOutcome::done(3, SubtaskReport::new("nothing in particular"))))
        }
    }

    struct Fixture {
        dir: tempfile::TempDir,
        task: TaskId,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            // The worktree is a real directory: the gate resolves paths against it.
            std::fs::create_dir_all(dir.path().join("work")).unwrap();
            Self {
                dir,
                task: TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV"),
            }
        }

        fn gate(&self) -> crate::PolicyGate {
            crate::PolicyGate::with_defaults(
                self.dir.path().join("work"),
                immortality_core::config::Policy {
                    workspace_root: self.dir.path().to_path_buf(),
                    ..immortality_core::config::Policy::default()
                },
            )
            .unwrap()
        }

        /// A context with a call id, a task and depth, as the daemon builds one.
        fn ctx(&self) -> ToolCtx {
            ToolCtx::new(Arc::new(self.gate()), crate::OutputBudget::default())
                .with_call(CALL_ID)
                .with_task(self.task.clone())
        }

        fn tool(&self, runner: Arc<FakeRunner>) -> Delegate {
            Delegate::new(self.dir.path(), runner)
        }

        fn store(&self) -> DelegationStore {
            DelegationStore::for_task(self.dir.path(), &self.task)
        }

        /// The id this fixture's call produces. An associated function, not a method: it does
        /// not depend on the fixture, and a method would hide that.
        fn fixture_id() -> DelegationId {
            DelegationId::for_call(CALL_ID)
        }

        fn call(&self, runner: Arc<FakeRunner>, args: Value) -> Result<ToolResult, ToolError> {
            self.tool(runner).execute(&self.ctx(), Args::new(args))
        }
    }

    /// The call id every test uses, so `ctx()` and `id()` cannot drift apart.
    const CALL_ID: &str = "call_01ARZ3NDEKTSV4RRFFQ69G5FAV";

    fn brief() -> Value {
        json!({
            "brief": "Add a trailing-comma test to src/parser.rs and make it pass.",
            "scope": ["src"],
            "max_steps": 10
        })
    }

    #[test]
    fn a_delegation_runs_once_and_returns_the_structured_report() {
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::answering(SubAgentOutcome::done(
            4,
            SubtaskReport {
                outcome: "trailing commas parse".to_string(),
                facts: vec!["the grammar table lives in src/grammar.rs".to_string()],
                files_touched: vec![PathBuf::from("src/parser.rs")],
                rejected_ideas: vec!["rewriting the lexer".to_string()],
            },
        )));

        let result = fixture.call(runner.clone(), brief()).unwrap();
        assert_eq!(runner.runs(), 1);
        assert_eq!(result.data["finish"], "done");
        assert_eq!(result.data["steps"], 4);
        assert_eq!(result.data["max_steps"], 10);
        assert_eq!(result.data["replayed"], false);
        assert_eq!(result.data["scope_violations"], json!([]));
        assert_eq!(result.data["report"]["files_touched"][0], "src/parser.rs");

        // The text the model reads must carry the parts a fresh context needs, since that is
        // all the parent keeps of the sub-agent's work.
        assert!(
            result.observation.contains("trailing commas parse"),
            "{}",
            result.observation
        );
        assert!(
            result.observation.contains("src/grammar.rs"),
            "{}",
            result.observation
        );
        assert!(
            result.observation.contains("rewriting the lexer"),
            "{}",
            result.observation
        );
    }

    #[test]
    fn the_report_is_on_disk_and_names_the_call_it_belongs_to() {
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        fixture.call(runner, brief()).unwrap();

        let report = fixture.store().read(&Fixture::fixture_id()).unwrap().unwrap();
        assert_eq!(report.brief, brief()["brief"].as_str().unwrap());
        assert_eq!(report.scope, vec!["src".to_string()]);
        assert_eq!(report.max_steps, 10);
        assert_eq!(report.steps, 3);
        // Under the task's own directory, not in a global store: a report dies with its task.
        assert!(store_path_contains_task(&fixture));
        assert_eq!(report.id, Fixture::fixture_id());
    }

    /// Whether the report file lives under the task directory of the state root.
    fn store_path_contains_task(fixture: &Fixture) -> bool {
        let path = fixture.store().path_for(&Fixture::fixture_id());
        path.starts_with(fixture.dir.path().join(fixture.task.as_str()).join("delegations"))
    }

    #[test]
    fn the_probe_finds_the_report_by_call_id_and_nothing_else() {
        // The roadmap's first required test. After a crash this is the only evidence recovery
        // has that a whole delegated run happened.
        let fixture = Fixture::new();
        let call_id = "call_01ARZ3NDEKTSV4RRFFQ69G5FAV";
        assert_eq!(
            probe_delegated(fixture.dir.path(), &fixture.task, call_id),
            Probe::NotApplied,
            "nothing has run yet"
        );

        fixture.call(Arc::new(FakeRunner::new()), brief()).unwrap();
        assert_eq!(
            probe_delegated(fixture.dir.path(), &fixture.task, call_id),
            Probe::Applied
        );
        // A different call is a different delegation, even with an identical brief.
        assert_eq!(
            probe_delegated(fixture.dir.path(), &fixture.task, "call_other"),
            Probe::NotApplied
        );
        // And a different task does not inherit another task's report.
        let other = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAW");
        assert_eq!(
            probe_delegated(fixture.dir.path(), &other, call_id),
            Probe::NotApplied
        );
    }

    #[test]
    fn a_corrupt_report_probes_unknown_not_not_applied() {
        // `NotApplied` here would repeat a run that already happened, spending the budget twice.
        let fixture = Fixture::new();
        let store = fixture.store();
        std::fs::create_dir_all(store.dir()).unwrap();
        std::fs::write(store.path_for(&Fixture::fixture_id()), "{not json").unwrap();
        assert_eq!(
            probe_delegated(
                fixture.dir.path(),
                &fixture.task,
                "call_01ARZ3NDEKTSV4RRFFQ69G5FAV"
            ),
            Probe::Unknown
        );
    }

    #[test]
    fn a_repeated_call_runs_the_sub_agent_once_and_replays_the_report() {
        // Recovery retried, or the model repeated itself. Either way the budget must not be
        // spent twice, and the parent must still see the result.
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        let first = fixture.call(runner.clone(), brief()).unwrap();
        let second = fixture.call(runner.clone(), brief()).unwrap();

        assert_eq!(runner.runs(), 1, "the sub-agent must not run twice");
        assert_eq!(first.data["replayed"], false);
        assert_eq!(second.data["replayed"], true);
        assert_eq!(second.data["delegation"], first.data["delegation"]);
        assert!(
            second.observation.contains("NOT run again"),
            "{}",
            second.observation
        );
    }

    #[test]
    fn a_second_call_with_a_different_id_is_a_different_delegation() {
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        fixture.call(runner.clone(), brief()).unwrap();

        let ctx = ToolCtx::new(Arc::new(fixture.gate()), crate::OutputBudget::default())
            .with_call("call_second")
            .with_task(fixture.task.clone());
        fixture
            .tool(runner.clone())
            .execute(&ctx, Args::new(brief()))
            .unwrap();
        assert_eq!(runner.runs(), 2);
        assert_eq!(fixture.store().list().unwrap().len(), 2);
    }

    #[test]
    fn the_step_limit_is_enforced_and_a_longer_run_is_recorded_as_exhausted() {
        // The roadmap's second required test. The budget is a limit, not a suggestion: a runner
        // that overruns is recorded as exhausted whatever it claims, so the parent cannot
        // believe a too-large brief finished.
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::answering(SubAgentOutcome::done(
            11,
            SubtaskReport::new("probably done"),
        )));
        let result = fixture.call(runner, brief()).unwrap();

        assert_eq!(result.data["finish"], "exhausted");
        assert_eq!(result.data["steps"], 11);
        assert_eq!(result.data["max_steps"], 10);
        assert!(
            result.observation.contains("11 steps of a 10-step budget"),
            "{}",
            result.observation
        );
        assert!(
            result.observation.contains("split it into smaller pieces"),
            "{}",
            result.observation
        );
        assert_eq!(
            fixture
                .store()
                .read(&Fixture::fixture_id())
                .unwrap()
                .unwrap()
                .finish,
            Finish::Exhausted
        );
    }

    #[test]
    fn the_runner_receives_the_budget_the_call_asked_for() {
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        fixture
            .call(
                runner.clone(),
                json!({"brief": "b", "scope": ["src"], "max_steps": 7}),
            )
            .unwrap();
        assert_eq!(runner.last().max_steps, 7);

        // Omitted means the documented default, not zero.
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        fixture
            .call(runner.clone(), json!({"brief": "b", "scope": ["src"]}))
            .unwrap();
        assert_eq!(runner.last().max_steps, DEFAULT_STEPS);
    }

    #[test]
    fn a_budget_above_the_ceiling_is_refused_with_the_reason() {
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        let error = fixture
            .call(
                runner.clone(),
                json!({"brief": "b", "scope": ["src"], "max_steps": MAX_STEPS + 1}),
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("is a subtask, not a delegation"),
            "{error}"
        );
        assert_eq!(runner.runs(), 0);

        let error = fixture
            .call(runner, json!({"brief": "b", "scope": ["src"], "max_steps": 0}))
            .unwrap_err();
        assert!(error.to_string().contains("`max_steps` is 0"), "{error}");
    }

    #[test]
    fn a_nested_delegation_is_refused_before_the_runner_is_touched() {
        // The roadmap's third required test. A sub-agent that could delegate is unbounded
        // recursion with a token bill, and the task tree is capped at depth 2.
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        let ctx = fixture.ctx().with_depth(1);
        let error = fixture
            .tool(runner.clone())
            .execute(&ctx, Args::new(brief()))
            .unwrap_err();
        assert!(error.to_string().contains("nested_delegation"), "{error}");
        assert_eq!(runner.runs(), 0);
        assert!(!fixture.store().exists(&Fixture::fixture_id()));
    }

    #[test]
    fn the_top_level_agent_may_delegate() {
        // The mirror of the previous test: the guard must not refuse everyone.
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        fixture
            .tool(runner.clone())
            .execute(&fixture.ctx().with_depth(0), Args::new(brief()))
            .unwrap();
        assert_eq!(runner.runs(), 1);
    }

    #[test]
    fn a_call_without_an_id_is_refused_rather_than_run_unaddressably() {
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        let ctx = ToolCtx::new(Arc::new(fixture.gate()), crate::OutputBudget::default())
            .with_task(fixture.task.clone());
        let error = fixture
            .tool(runner.clone())
            .execute(&ctx, Args::new(brief()))
            .unwrap_err();
        assert!(error.to_string().contains("no_call_id"), "{error}");
        assert_eq!(runner.runs(), 0);
    }

    #[test]
    fn a_call_without_a_task_is_refused() {
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        let ctx = ToolCtx::new(Arc::new(fixture.gate()), crate::OutputBudget::default()).with_call("call_x");
        let error = fixture
            .tool(runner.clone())
            .execute(&ctx, Args::new(brief()))
            .unwrap_err();
        assert!(error.to_string().contains("no_task"), "{error}");
        assert_eq!(runner.runs(), 0);
    }

    #[test]
    fn a_runner_that_cannot_start_writes_no_report_so_the_probe_says_not_applied() {
        // The distinction the whole recovery path rests on: nothing durable happened, so the
        // call may be retried.
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::failing("no model configured"));
        let error = fixture.call(runner, brief()).unwrap_err();
        assert!(error.to_string().contains("no model configured"), "{error}");
        assert!(!fixture.store().exists(&Fixture::fixture_id()));
        assert_eq!(
            probe_delegated(
                fixture.dir.path(),
                &fixture.task,
                "call_01ARZ3NDEKTSV4RRFFQ69G5FAV"
            ),
            Probe::NotApplied
        );
    }

    #[test]
    fn a_run_that_started_and_failed_still_writes_a_report() {
        // The other side of that distinction: the work happened, so a retry would repeat it.
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::answering(SubAgentOutcome::unfinished(
            5,
            Finish::Failed,
            SubtaskReport::new("found the bug but did not fix it"),
            "the gate went red",
        )));
        let result = fixture.call(runner, brief()).unwrap();
        assert_eq!(result.data["finish"], "failed");
        assert!(
            result.observation.contains("the gate went red"),
            "{}",
            result.observation
        );
        assert_eq!(
            probe_delegated(
                fixture.dir.path(),
                &fixture.task,
                "call_01ARZ3NDEKTSV4RRFFQ69G5FAV"
            ),
            Probe::Applied
        );
    }

    #[test]
    fn files_outside_the_scope_are_reported_and_not_hidden() {
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::answering(SubAgentOutcome::done(
            2,
            SubtaskReport {
                outcome: "done, mostly".to_string(),
                files_touched: vec![PathBuf::from("src/parser.rs"), PathBuf::from("Cargo.toml")],
                ..SubtaskReport::default()
            },
        )));
        let result = fixture.call(runner, brief()).unwrap();
        assert_eq!(result.data["scope_violations"], json!(["Cargo.toml"]));
        assert!(
            result.observation.contains("outside the scope"),
            "{}",
            result.observation
        );
        assert!(
            result
                .observation
                .contains("Nothing outside the scope was reverted")
        );
    }

    #[test]
    fn a_prefix_lookalike_is_a_violation() {
        // `srcfoo` must not be read as `src`: a scope check that passes on a prefix
        // comparison is decoration.
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::answering(SubAgentOutcome::done(
            1,
            SubtaskReport {
                outcome: "x".to_string(),
                files_touched: vec![PathBuf::from("srcfoo/main.rs")],
                ..SubtaskReport::default()
            },
        )));
        let result = fixture.call(runner, brief()).unwrap();
        assert_eq!(result.data["scope_violations"], json!(["srcfoo/main.rs"]));
    }

    #[test]
    fn an_empty_brief_is_refused_because_a_clean_context_knows_only_the_brief() {
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        let error = fixture
            .call(runner.clone(), json!({"brief": "   ", "scope": ["src"]}))
            .unwrap_err();
        assert!(error.to_string().contains("`brief` is empty"), "{error}");
        assert_eq!(runner.runs(), 0);
    }

    #[test]
    fn an_oversized_brief_is_refused_without_running_anything() {
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        let error = fixture
            .call(
                runner.clone(),
                json!({"brief": "x".repeat(MAX_BRIEF_LEN + 1), "scope": ["src"]}),
            )
            .unwrap_err();
        assert!(error.to_string().contains("is a plan"), "{error}");
        assert_eq!(runner.runs(), 0);
    }

    #[test]
    fn a_scope_that_would_widen_the_reach_is_refused_by_name() {
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        for bad in ["/etc", "~/secrets", "../outside"] {
            let error = fixture
                .call(runner.clone(), json!({"brief": "b", "scope": [bad]}))
                .unwrap_err();
            assert!(
                error.to_string().contains("is not a usable scope"),
                "{bad}: {error}"
            );
        }
        assert_eq!(runner.runs(), 0);
    }

    #[test]
    fn a_scope_must_be_an_array_of_strings() {
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        let error = fixture
            .call(runner.clone(), json!({"brief": "b", "scope": "src"}))
            .unwrap_err();
        assert!(error.to_string().contains("must be an array"), "{error}");

        let error = fixture
            .call(runner, json!({"brief": "b", "scope": [7]}))
            .unwrap_err();
        assert!(error.to_string().contains("must contain strings"), "{error}");
    }

    #[test]
    fn the_scope_is_normalised_before_it_reaches_the_runner() {
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        fixture
            .call(
                runner.clone(),
                json!({"brief": "b", "scope": [" src//parser/ ", "./docs"]}),
            )
            .unwrap();
        assert_eq!(
            runner.last().scope,
            vec!["src/parser".to_string(), "docs".to_string()]
        );
    }

    #[test]
    fn the_runner_gets_the_callers_worktree_not_one_it_chose() {
        // A sub-agent works in the same worktree as its parent. A runner that could be pointed
        // elsewhere would be a way out of the per-task isolation.
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        fixture.call(runner.clone(), brief()).unwrap();
        let seen = runner.last();
        assert_eq!(
            seen.worktree.canonicalize().unwrap(),
            fixture.dir.path().join("work").canonicalize().unwrap()
        );
        assert_eq!(seen.task, fixture.task);
    }

    #[test]
    fn the_disabled_runner_refuses_instead_of_pretending() {
        let fixture = Fixture::new();
        let tool = Delegate::disabled(fixture.dir.path());
        let error = tool.execute(&fixture.ctx(), Args::new(brief())).unwrap_err();
        assert!(error.to_string().contains("no_sub_agent"), "{error}");
        assert!(!fixture.store().exists(&Fixture::fixture_id()));
    }

    #[test]
    fn the_tool_declares_no_worktree_paths_and_a_mutating_effect() {
        let tool = Delegate::disabled("/tmp");
        assert_eq!(tool.paths(&Args::empty()), Vec::new());
        assert_eq!(tool.effect(&Args::empty()), Effect::Mutate);
        assert_eq!(tool.idempotency(&Args::empty()), Idempotency::Unknown);
        assert!(tool.idempotency(&Args::empty()).requires_probe());
    }

    #[test]
    fn the_schema_is_well_formed_and_has_no_nesting_escape_hatch() {
        let schema = delegate_schema();
        assert_eq!(immortality_llm::schema::schema_defects(schema), Vec::new());
        assert_eq!(
            schema["required"],
            json!(["brief", "scope"]),
            "both are needed: a brief without a scope has no boundary"
        );
        assert!(schema["properties"].get("delegate").is_none());
        assert_eq!(schema["properties"]["max_steps"]["maximum"], json!(MAX_STEPS));
    }

    #[test]
    fn a_corrupt_report_makes_the_call_refuse_rather_than_run_again() {
        let fixture = Fixture::new();
        let store = fixture.store();
        std::fs::create_dir_all(store.dir()).unwrap();
        std::fs::write(store.path_for(&Fixture::fixture_id()), "{not json").unwrap();

        let runner = Arc::new(FakeRunner::new());
        let error = fixture.call(runner.clone(), brief()).unwrap_err();
        assert!(
            error.to_string().contains("delegation_store_unavailable"),
            "{error}"
        );
        assert_eq!(runner.runs(), 0, "a broken record must not spend the budget");
    }

    #[test]
    fn listing_returns_every_report_for_the_task() {
        let fixture = Fixture::new();
        let runner = Arc::new(FakeRunner::new());
        fixture.call(runner.clone(), brief()).unwrap();

        let ctx = ToolCtx::new(Arc::new(fixture.gate()), crate::OutputBudget::default())
            .with_call("call_second")
            .with_task(fixture.task.clone());
        fixture
            .tool(runner)
            .execute(&ctx, Args::new(json!({"brief": "second", "scope": ["docs"]})))
            .unwrap();

        let listed = listing(fixture.dir.path(), &fixture.task).unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listing(fixture.dir.path(), &TaskId::generate()).unwrap(), []);
    }
}
