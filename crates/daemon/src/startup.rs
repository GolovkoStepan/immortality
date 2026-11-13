//! The startup pass: read, plan, and execute recovery for a task.
//!
//! # Why this is separate from [`crate::recovery`]
//!
//! `recovery` answers two narrow questions — "what does the plan say?" and "how does each step
//! touch the world?" — and both are testable with a fixture and no daemon. This module answers the
//! third: *which* facts the daemon gathers, in *what order*, before either of those can run. That
//! ordering is where the bugs are:
//!
//! 1. **Read the journal before opening it for writing.** `Journal::open` truncates a torn tail
//!    and journals the repair, so a plan built after the open cannot see that there was a tail and
//!    reports a clean start for a crash (D-083).
//! 2. **Do not open a journal the plan is about to block on.** A broken chain makes `open` refuse,
//!    so the block goes to `panic.json` and the escalation queue instead of a record (D-082).
//! 3. **Do not touch a finished task.** A merged or cancelled task has no work ahead, and
//!    rewriting its cache or closing its attempts would edit a history that is over.
//!
//! Getting any of the three wrong produces a daemon that looks healthy while silently losing the
//! account of what happened, which is the one failure this project cannot absorb.

use std::path::PathBuf;
use std::sync::Arc;

use immortality_core::artifacts::ArtifactStore;
use immortality_core::cache::Cache;
use immortality_core::checkpoint::GitRepo;
use immortality_core::domain::TaskId;
use immortality_core::recover::{self, RecoveryInput, Verdict, WorkspaceFacts};
use immortality_core::time::Clock;
use immortality_core::worktree::WorktreeManager;
use immortality_tools::{PolicyGate, Registry};

use crate::recovery::{
    self as exec, BlockReason, Executed, Executor, RecoveryDeps, ToolProbe, daemon_started, expected_branch,
    has_begun, is_unfinished, last_checkpoint, read_journal,
};

/// One task, recovered from wherever the last process left it.
///
/// Public fields, like [`RecoveryDeps`]: eight values assembled once, by a caller that already has
/// all of them. A builder or a `new` that defaults half of them would let `serve` run recovery
/// against an empty tool registry without noticing.
#[derive(Debug)]
pub struct RecoveryRequest {
    /// The task to recover.
    pub task: TaskId,
    /// The project's main checkout, where `git worktree add` runs.
    pub repository: PathBuf,
    /// The project's `.agent/`, where the journal lives.
    pub journal_dir: PathBuf,
    /// The task's machine-local state directory.
    pub task_dir: PathBuf,
    /// The registry every retried call dispatches through.
    pub registry: Arc<Registry>,
    /// The gate those calls are checked against.
    pub gate: Arc<PolicyGate>,
    /// Where oversized output goes.
    pub artifacts: ArtifactStore,
    /// The clock records are stamped from.
    pub clock: Arc<dyn Clock>,
}

/// What a startup pass did for one task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recovered {
    /// The plan's verdict.
    pub verdict: Verdict,
    /// Steps carried out, in order.
    pub executed: Executed,
    /// One line for the daemon's log.
    pub summary: String,
    /// Whether this pass opened the task's journal for writing.
    ///
    /// Answered here rather than re-derived by the caller: the pass is the only thing that knows,
    /// and a caller guessing from the summary text or the journal's last record is a second
    /// implementation of the same decision. `serve` uses it to bracket exactly the sessions that
    /// actually began with `daemon_stopped`.
    pub session_open: bool,
}

impl Recovered {
    /// Whether anything changed on disk.
    #[must_use]
    pub fn repaired(&self) -> bool {
        self.verdict != Verdict::Healthy
    }
}

/// Read, plan, and execute recovery for one task.
///
/// # Errors
/// The journal cannot be read, the plan leaves an open call unaddressed, or a step of the plan
/// could not be carried out. All three are reported with the task they concern: "recovery failed"
/// is not something a human can act on.
pub fn recover_task(request: &RecoveryRequest) -> Result<Recovered, String> {
    let task = &request.task;
    let branch = expected_branch(task);

    // Step 1: read. Nothing has opened the journal for writing in this process yet, so a torn
    // tail is still on disk for the plan to see.
    let outcome = match read_journal(&request.journal_dir, task) {
        Ok(outcome) => outcome,
        // A *parse* failure is not a torn tail: the line is complete and unreadable, which means a
        // record was written by something that is not this schema version or the file was edited.
        // The reader reports that as an error rather than a problem, so there is no `ReadOutcome`
        // to plan over — and blocking is precisely the only safe answer. It is reached here, and
        // not by propagating, because a task that fails to recover on every start is a task a
        // human never hears about.
        Err(error) => {
            let block = BlockReason {
                task: task.clone(),
                task_dir: request.task_dir.clone(),
                clock: Arc::clone(&request.clock),
            };
            block.apply(&format!("the journal cannot be read: {error}"))?;
            return Ok(Recovered {
                verdict: Verdict::Blocked,
                executed: Executed {
                    steps: vec![format!("blocked the task: {error}")],
                },
                summary: format!("blocked: {error}"),
                // A blocked task has no writable journal, so no session was opened — which is why
                // nothing may write `daemon_stopped` for it either.
                session_open: false,
            });
        }
    };
    if !is_unfinished(&outcome.records) {
        return Ok(Recovered {
            verdict: Verdict::Healthy,
            executed: Executed { steps: Vec::new() },
            summary: "finished; not recovered".into(),
            session_open: false,
        });
    }
    // A registered project with no task, or a task whose plan exists but whose first attempt was
    // never journaled, has nothing to repair. Recovering it anyway would bracket a session the
    // task never had and ask git for a branch nobody created.
    if !has_begun(&outcome.records) {
        return Ok(Recovered {
            verdict: Verdict::Healthy,
            executed: Executed { steps: Vec::new() },
            summary: "not started; nothing to recover".into(),
            session_open: false,
        });
    }

    let manager = WorktreeManager::new(&request.repository);
    let workspace = match manager.find(task.as_str()) {
        Ok(Some(found)) => found.facts(),
        // Absent before the first attempt, and after a `git worktree prune`. The plan decides
        // whether to rebuild it and from which commit; guessing here would be a second source of
        // truth about a tree that is gone.
        Ok(None) => WorkspaceFacts::absent(),
        // A directory that exists but is not our worktree is a human problem, and `find` says so;
        // reporting it as absent would make recovery try to recreate over unknown files.
        Err(error) => return Err(format!("cannot inspect the worktree of {task}: {error}")),
    };

    let repo = GitRepo::new(&request.repository);
    let probe = ToolProbe {
        gate: &request.gate,
        repo: &repo,
        // `DelegationStore::for_task` appends the task id, so the root is the directory above it.
        delegation_root: request.task_dir.parent().unwrap_or(&request.task_dir),
        task,
        task_dir: &request.task_dir,
    };
    let input = RecoveryInput {
        journal: &outcome,
        workspace,
        persisted: Cache::new(&request.task_dir).read(),
        last_checkpoint: last_checkpoint(&outcome.records),
        expected_branch: &branch,
    };
    let report = recover::plan(&input, &probe);
    if !recover::all_calls_addressed(&outcome.records, &report) {
        // A planning bug, reported rather than passed on: continuing would leave a call nothing
        // will ever close, and the subtask waits forever for an attempt no process owns.
        return Err(format!(
            "recovery planned {} of the open calls for {task}; refusing to apply a partial repair",
            report.calls.len()
        ));
    }

    // Step 2: open for writing, now that the plan exists. A blocked task has a journal that cannot
    // be opened — that is what blocking means — so the executor gets no writer and `BlockTask` is
    // the only step that can run.
    let mut journal = if report.verdict == Verdict::Blocked {
        None
    } else {
        // `daemon_started` is the open: it truncates the torn tail (which the plan already
        // reported) and records that this process is the one doing the repairing. Passing the
        // clock in is what stamps that first record with the daemon's clock rather than the
        // journal's default.
        let mut journal = daemon_started(&request.journal_dir, task.clone(), request.clock.as_ref())?;
        journal.set_clock(Arc::clone(&request.clock));
        Some(journal)
    };

    let deps = RecoveryDeps {
        task: task.clone(),
        task_dir: request.task_dir.clone(),
        // The repository, not the worktree: recreating a worktree means asking the repository to
        // recreate one of its own, and a manager built on the missing worktree creates nothing.
        repository: request.repository.clone(),
        worktree: manager.path_for(task.as_str()),
        branch,
        gate: Arc::clone(&request.gate),
        registry: Arc::clone(&request.registry),
        artifacts: request.artifacts.clone(),
        clock: Arc::clone(&request.clock),
    };
    let session_open = journal.is_some();
    let mut executor = Executor::new(&deps, journal.as_mut(), outcome, &report);
    let executed = executor.run(&report.steps)?;

    Ok(Recovered {
        verdict: report.verdict,
        executed,
        summary: report.summary(),
        session_open,
    })
}

/// Record `daemon_stopped` for every task the daemon touched, during a clean shutdown.
///
/// Best-effort by construction: a shutdown is not the moment to fail a process. The caller logs
/// what could not be written, and the next start's recovery covers the gap.
pub fn stopped(tasks: &[(TaskId, PathBuf)], at: immortality_core::domain::Timestamp) {
    for (task, journal_dir) in tasks {
        if let Err(error) = exec::daemon_stopped(journal_dir, task.clone(), at) {
            tracing::warn!("cannot record daemon_stopped for {task}: {error}");
        }
    }
}
