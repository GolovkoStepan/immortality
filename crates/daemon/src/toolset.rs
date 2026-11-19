//! The production tool registry: every tool the model may call, in one place.
//!
//! # Why a module of its own
//!
//! Two callers need the same set: the recovery executor, which re-issues calls a crash
//! interrupted, and the agent loop, which issues new ones. A re-issued call that found a
//! *different* registry than the original would be a call judged by another policy and dispatched
//! to another implementation of the same name — so the set is built once, here, and both callers
//! take it.
//!
//! # Why the tools are all constructed eagerly
//!
//! Every tool below is either stateless or owns an `Arc` to shared state. Constructing one per run
//! would be cheap, but a registry is what makes the *set* a value: `Registry::new` rejects a
//! duplicated name, and that check belongs at construction, not at the first dispatch of a
//! misconfigured daemon.
//!
//! # What this deliberately does not own
//!
//! The `delegate` tool needs a [`SubAgent`] runner, which is the agent loop itself. Wiring it here
//! would make this module depend on the runner and the runner depend on this module. It is passed
//! in instead, and a caller without a loop (recovery, today) gets [`NoSubAgent`], which reports
//! that delegation is unavailable rather than pretending to run.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use immortality_core::artifacts::ArtifactStore;
use immortality_core::config::Config;
use immortality_core::ids::TaskId;
use immortality_tools::{
    Delegate, EditFile, Escalate, Git, GlobTool, Grep, GrepJournal, KillProcess, Notifier, OutputBudget,
    PolicyGate, ProcessRegistry, ReadFile, ReadProcess, ReadState, Registry, Shell, ShellState, Silent,
    StartProcess, SubAgent, UpdateState, WriteFile,
};

/// Everything a tool needs that the model never names.
///
/// Assembled once per task by the daemon. The policy gate is the authority: it is the same gate
/// the recovery probe reads through, so "what the tool wrote" and "what the probe looks for" are
/// the same path by construction and not by coincidence.
#[derive(Debug, Clone)]
pub struct ToolsetDeps {
    /// The task the tools act for.
    pub task: TaskId,
    /// The task worktree writes are confined to.
    pub worktree: PathBuf,
    /// The gate every call is checked against.
    pub gate: Arc<PolicyGate>,
    /// Content-addressed oversized output.
    pub artifacts: ArtifactStore,
    /// Budget for what a single call may return to the model.
    pub budget: OutputBudget,
    /// The task's machine-local state directory, `<root>/state/<project>/<task>`.
    pub task_dir: PathBuf,
    /// The project's own checkout, which is where `.agent/` lives.
    ///
    /// Not the worktree: the worktree is a sibling directory whose `.agent/` is a checkout from
    /// whenever the branch was cut, and therefore stale by construction. The plan the scheduler
    /// executes, the journal recovery reads and the notes the agent writes are all in *this*
    /// repository's `.agent/`.
    pub repository: PathBuf,
    /// The directory *holding* per-task state directories, `<root>/state/<project>`.
    ///
    /// Not the machine state root, and not the task directory: [`DirectoryQueue`] and
    /// [`immortality_core::delegation::DelegationStore`] both append the task id to whatever they
    /// are given, so the queue for `<root>/state/<project>/<task>` is rooted one level above it.
    /// Passing the state root would queue escalations in `<root>/<task>` — a directory nothing
    /// else reads — and the human would never see the question.
    pub queue_root: PathBuf,
    /// How the human is told a question is waiting.
    pub notifier: Arc<dyn Notifier>,
    /// The sub-agent runner behind `delegate`, installed after the registry exists.
    ///
    /// A slot rather than the runner itself because the real runner holds the registry this tool
    /// lives in: the tool has to be built first. See [`immortality_tools::SubAgentSlot`].
    pub sub_agent: Arc<immortality_tools::SubAgentSlot>,
    /// Where `escalate` collects the journal records it cannot write itself.
    ///
    /// The runner holds the task journal's writer for the whole run, so a queue that opened it
    /// again would be a second writer on one segment — a duplicated `seq` and a broken chain
    /// (D-117). The tool therefore collects and the runner drains, exactly as the judge does.
    pub escalations: Arc<immortality_tools::JournalQueue>,
}

impl ToolsetDeps {
    /// Deps for one task, with the design's defaults.
    ///
    /// `state_root` is where the artifact store lives; `task_dir` is the task's own directory,
    /// which the caller already has from [`immortality_core::registry::Registry::tasks`]. Both are
    /// passed rather than derived from one another because the layout between them is
    /// `<root>/state/<project>/<task>` and re-deriving it here would be a second copy of that rule.
    ///
    /// # Errors
    /// The policy gate's default globs do not compile, or the artifact store cannot be created.
    pub fn for_task(
        config: &Config,
        state_root: impl Into<PathBuf>,
        task_dir: impl Into<PathBuf>,
        worktree: impl Into<PathBuf>,
        repository: impl Into<PathBuf>,
        task: TaskId,
    ) -> immortality_core::Result<Self> {
        let state_root = state_root.into();
        let task_dir = task_dir.into();
        let worktree = worktree.into();
        let repository = repository.into();
        // The gate is told where `.agent/` is, because the state tools write there and the gate is
        // the only thing that hands out paths. Without this the tools fall back to the worktree's
        // stale copy — the split-brain the `word-count` fixture still shows: project `done`,
        // worktree `pending`.
        let gate = PolicyGate::with_defaults(&worktree, config.policy.clone())
            .map_err(|error| {
                immortality_core::Error::Protocol(format!("the policy gate is misconfigured: {error}"))
            })?
            .with_project_agent_dir(repository.join(immortality_core::paths::AGENT_DIR));
        let artifacts = ArtifactStore::open(state_root.join("artifacts"))?;
        // The queue and the delegation store both append the task id themselves. Falling back to
        // the task directory would put a second `<task>` segment under it, which is a directory
        // no reader looks in — visible only as a question that is never answered.
        let queue_root = task_dir
            .parent()
            .map_or_else(|| task_dir.clone(), Path::to_path_buf);
        let escalations = Arc::new(immortality_tools::JournalQueue::new(queue_root.clone()));
        Ok(Self {
            task_dir,
            task,
            worktree,
            repository,
            gate: Arc::new(gate),
            artifacts,
            budget: OutputBudget {
                max_bytes: config.context.max_tool_output_bytes,
                max_context: 3,
            },
            queue_root,
            notifier: Arc::new(Silent),
            sub_agent: Arc::new(immortality_tools::SubAgentSlot::new()),
            escalations,
        })
    }

    /// Use `notifier` for questions that need a human.
    #[must_use]
    pub fn notifying(mut self, notifier: Arc<dyn Notifier>) -> Self {
        self.notifier = notifier;
        self
    }

    /// Use `sub_agent` behind the `delegate` tool.
    #[must_use]
    pub fn delegating(mut self, sub_agent: Arc<dyn SubAgent>) -> Self {
        self.sub_agent = Arc::new(immortality_tools::SubAgentSlot::filled(sub_agent));
        self
    }

    /// Install the judge in the policy gate, and tell it what the task is for.
    ///
    /// # Why this is a method and not a constructor argument
    ///
    /// The judge needs the model, the model is built by the daemon, and this crate must not
    /// depend on the model at all — that is the whole reason [`crate::toolset`] exists as a
    /// separate module. So the gate is built here without a judge and the slot is filled
    /// afterwards, once both exist. The gate is behind an `Arc`, so this replaces the shared
    /// value rather than mutating one somebody else may already hold: a caller that cloned the
    /// gate *before* this call keeps the judge-less one, which is why every caller must install
    /// the judge before handing the deps out.
    #[must_use]
    pub fn with_judge(
        mut self,
        judge: Arc<dyn immortality_tools::judge::Judge>,
        goal: impl Into<String>,
    ) -> Self {
        Arc::make_mut(&mut self.gate).set_judge(judge, goal);
        self
    }

    /// The process registry, opened under the task's own directory.
    ///
    /// Handles are `proc_<n>` local to a task: two tasks both starting a server on port 3000 is a
    /// bug the model has to see, and a shared counter would hide which task started what.
    ///
    /// # Errors
    /// The directory cannot be created or read.
    pub fn processes(&self) -> Result<ProcessRegistry, immortality_tools::ToolError> {
        ProcessRegistry::open(self.task_dir.join("processes"))
    }
}

/// The registry every caller dispatches through.
///
/// # Errors
/// A tool's JSON schema is defective or two tools share a name. Both are programming errors, and
/// failing at construction is what makes them visible in `daemon doctor` rather than at the first
/// call of the first run.
pub fn build(deps: &ToolsetDeps) -> immortality_core::Result<Registry> {
    build_with_processes(deps).map(|(registry, _)| registry)
}

/// [`build`], and the process registry the tools were built over.
///
/// The runner needs the same registry to sweep a task's processes when an attempt ends (D-152).
/// Returning it here rather than opening a second one is the same reason the tools share one:
/// two `ProcessRegistry` values over one directory would allocate `proc_1` twice, and a handle
/// would then name two processes.
///
/// # Errors
/// See [`build`].
pub fn build_with_processes(
    deps: &ToolsetDeps,
) -> immortality_core::Result<(Registry, Arc<ProcessRegistry>)> {
    let shell = Arc::new(ShellState::new());
    // The registry's files live under the task directory; a failure here is the state root
    // being unwritable, which is a reason not to start rather than a reason to omit the process
    // tools and let the model discover their absence at the first call.
    let processes = Arc::new(
        deps.processes()
            .map_err(|error| immortality_core::Error::Protocol(error.to_string()))?,
    );
    let registry = Registry::new(vec![
        Arc::new(ReadFile),
        Arc::new(WriteFile),
        Arc::new(EditFile),
        Arc::new(GlobTool),
        Arc::new(Grep),
        Arc::new(Shell::new(Arc::clone(&shell), Some(deps.artifacts.clone()))),
        Arc::new(Git::new()),
        Arc::new(ReadState),
        Arc::new(UpdateState),
        Arc::new(GrepJournal),
        Arc::new(Escalate::collecting(
            Arc::clone(&deps.escalations),
            Arc::clone(&deps.notifier),
        )),
        Arc::new(Delegate::deferred(&deps.queue_root, Arc::clone(&deps.sub_agent))),
        Arc::new(StartProcess::new(Arc::clone(&processes), Arc::clone(&shell))),
        Arc::new(ReadProcess::new(Arc::clone(&processes))),
        Arc::new(KillProcess::new(Arc::clone(&processes))),
    ])
    .map_err(|error| immortality_core::Error::Protocol(format!("the tool registry is broken: {error}")))?;
    Ok((registry, processes))
}

/// The directory a task's journal lives in, given the repository it belongs to.
///
/// One function rather than a `join` at each call site: the journal is in the project's `.agent/`
/// and the worktree is a sibling directory, and mistaking one for the other is how recovery once
/// deleted its own journal with `reset --hard`.
#[must_use]
pub fn journal_dir(repository: &Path) -> PathBuf {
    repository.join(immortality_core::paths::AGENT_DIR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use immortality_core::config::Config;

    fn deps(dir: &Path) -> ToolsetDeps {
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let worktree = dir.join("worktree");
        std::fs::create_dir_all(&worktree).expect("worktree");
        // The daemon's layout: `<state>/state/<project>/<task>`.
        let task_dir = dir.join("state").join("project").join(task.as_str());
        std::fs::create_dir_all(&task_dir).expect("task dir");
        ToolsetDeps::for_task(&Config::default(), dir, task_dir, worktree, dir, task).expect("deps")
    }

    #[test]
    fn the_gate_is_exclusive_until_the_judge_is_installed() {
        // `with_judge` goes through `Arc::make_mut`, which *clones* instead of mutating once a
        // second strong reference exists. Nothing about that failure is loud: the run keeps
        // escalating every unknown program and the journal just has no verdicts. So the
        // precondition — a gate nobody else has cloned yet — is asserted here rather than
        // trusted, because the clone that breaks it would be added to `for_task` for an
        // unrelated reason and nothing else would notice.
        let dir = tempfile::tempdir().expect("tempdir");
        let deps = deps(dir.path());
        assert_eq!(
            Arc::strong_count(&deps.gate),
            1,
            "`for_task` must hand out a gate with no other owner, or `with_judge` cannot reach it"
        );

        let judge: Arc<dyn immortality_tools::judge::Judge> =
            immortality_tools::judge::Scripted::new(Vec::new());
        let deps = deps.with_judge(Arc::clone(&judge), "parse trailing commas");

        assert!(
            deps.gate.judge().is_some(),
            "the judge was accepted and dropped, so the run would never consult it"
        );
        assert!(
            deps.gate
                .judge()
                .is_some_and(|installed| Arc::ptr_eq(installed, &judge)),
            "a different judge was installed than the one the caller passed"
        );
    }

    #[test]
    fn every_design_tool_is_registered_exactly_once() {
        // The set is a contract with the model: a tool named in the prompt but missing here is an
        // unknown-tool error at the first call, and a name registered twice is a dispatch order
        // that depends on registration order.
        let dir = tempfile::tempdir().expect("tempdir");
        let registry = build(&deps(dir.path())).expect("registry");
        let names: Vec<&str> = registry.names();
        for expected in [
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
            "shell",
            "git",
            "read_state",
            "update_state",
            "grep_journal",
            "escalate",
            "delegate",
            "start_process",
            "read_process",
            "kill_process",
        ] {
            assert!(
                names.contains(&expected),
                "`{expected}` is missing; got {names:?}"
            );
        }
        assert_eq!(names.len(), 15, "an unlisted tool appeared: {names:?}");
    }

    #[test]
    fn the_escalation_queue_is_where_the_reader_looks() {
        // Both `DirectoryQueue` and `DelegationStore` append the task id. Deriving the root from
        // the machine state root instead of the task directory's parent produced
        // `<root>/<task>` — a directory `EscalationStore::open(&task_dir, ..)` never reads — so an
        // escalation would be written and never shown. This pins the path, not the mechanism.
        let dir = tempfile::tempdir().expect("tempdir");
        let deps = deps(dir.path());
        let expected = deps
            .task_dir
            .parent()
            .expect("the task dir has a parent")
            .to_path_buf();
        assert_eq!(deps.queue_root, expected);
        // `Escalate` writes to `<queue_root>/<task>`; the store opens `<task_dir>`.
        assert_eq!(
            deps.queue_root.join(deps.task.as_str()),
            deps.task_dir,
            "the queue's task directory must be the task's state directory"
        );
    }

    #[test]
    fn the_registry_is_the_same_for_recovery_and_the_loop() {
        // The whole reason this module exists: a retried call must find the tool the original
        // call used. Two builds must therefore be identical in name and schema.
        let dir = tempfile::tempdir().expect("tempdir");
        let deps = deps(dir.path());
        let first = build(&deps).expect("first");
        let second = build(&deps).expect("second");
        assert_eq!(first.names(), second.names());
        for name in first.names() {
            assert_eq!(
                first.get(name).map(|tool| tool.schema().to_string()),
                second.get(name).map(|tool| tool.schema().to_string()),
                "`{name}` differs between builds"
            );
        }
    }

    #[test]
    fn the_journal_lives_outside_the_worktree() {
        // The layout a rollback depends on: `reset --hard` runs in the task worktree, which is a
        // sibling of the project (`<parent>/<name>-agent/<task>`), and the journal is in the
        // *project's* `.agent/`. Passing the worktree as the repository would put the journal
        // inside the tree the rollback rewrites — which is exactly the bug this pins.
        let dir = tempfile::tempdir().expect("tempdir");
        let project = dir.path().join("project");
        std::fs::create_dir_all(&project).expect("project");
        let manager = immortality_core::worktree::WorktreeManager::new(&project);
        let worktree = manager.path_for("tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV");
        std::fs::create_dir_all(&worktree).expect("worktree");

        let journal = journal_dir(&project);
        assert!(
            !journal.starts_with(&worktree),
            "the journal would be deleted by a rollback: {}",
            journal.display()
        );
        assert!(journal.ends_with(immortality_core::paths::AGENT_DIR));
    }
}
