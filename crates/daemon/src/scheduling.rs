//! Assembling the scheduler's view of the world from what is on disk.
//!
//! # Why this is not in the scheduler
//!
//! [`immortality_core::scheduler::queue`] decides *who runs next* and is a pure function of
//! values. This module is the other half: it reads the registry, each project's plan and each
//! task's journal to produce those values. Keeping them apart is what makes the ordering rules
//! testable in microseconds and this module testable against a fixture repository instead of a
//! fake queue.
//!
//! # Why the journal is read here and not the cache
//!
//! The scheduling answer must be correct immediately after a crash, and the cache is explicitly
//! allowed to be stale (D-083). Reading the journal costs one file per task per turn — the same
//! read `status` already does — and buys the property that a daemon which just recovered schedules
//! from the same facts its recovery used.
//!
//! # What is *not* inferred
//!
//! A task with no `TASKS.json` has no work, not work that cannot be read: the plan is written by
//! `plan` (M5.7) and a task that has not planned yet has nothing to run. Treating a missing plan as
//! an error would make `status` fail on a freshly started task, which is the one moment a human is
//! watching.

use std::path::Path;

use immortality_core::config::Config;
use immortality_core::domain::{Hold, TaskState};
use immortality_core::journal::JournalReader;
use immortality_core::plan::Plan;
use immortality_core::registry::{Project, Registry};
use immortality_core::scheduler::backend;
use immortality_core::scheduler::queue::{Candidate, Priority};

/// One task, with everything the queue and `status` need.
#[derive(Debug, Clone)]
pub struct Scheduled {
    /// The candidate, ready for the queue.
    pub candidate: Candidate,
    /// The project it belongs to, for rendering and for the worktree rules.
    pub project: Project,
    /// The task's machine-local directory.
    pub task_dir: std::path::PathBuf,
}

/// Every task this daemon knows about, as the scheduler sees it.
///
/// A task whose journal cannot be read is included with `state: Blocked` and no work: it is a fact
/// a human must see (`recovery` will block it on the next start), and omitting it would make
/// `status` report a task that exists as absent.
///
/// # Errors
/// The registry cannot be read. A single task's journal or plan being unreadable is not an error
/// here — it is a state with a reason, which is what `status` is for.
pub fn candidates(
    registry: &Registry,
    state_root: &Path,
    config: &Config,
) -> immortality_core::Result<Vec<Scheduled>> {
    let mut out = Vec::new();
    for (project, task, task_dir) in registry.tasks(state_root)? {
        let facts = facts_for(&project, &task, config);
        out.push(Scheduled {
            candidate: Candidate {
                task,
                priority: Priority::Normal,
                state: facts.state,
                hold: facts.hold,
                created_at: facts.created_at,
                has_work: facts.has_work,
            },
            project,
            task_dir,
        });
    }
    Ok(out)
}

/// The facts about one task.
#[derive(Debug, Clone, Copy)]
struct Facts {
    state: TaskState,
    hold: Option<Hold>,
    created_at: immortality_core::domain::Timestamp,
    has_work: bool,
}

fn facts_for(project: &Project, task: &immortality_core::ids::TaskId, config: &Config) -> Facts {
    let now = immortality_core::time::now();
    let outcome = JournalReader::open(project.agent_dir(), task).and_then(|reader| reader.read());
    let Ok(outcome) = outcome else {
        // Unreadable is not the same as empty, and the difference is what a human acts on. A
        // blocked task is visible and does not take a slot; a skipped one disappears.
        return Facts {
            state: TaskState::Blocked,
            hold: None,
            created_at: now,
            has_work: false,
        };
    };
    let records: Vec<&immortality_core::journal::JournalRecord> = outcome
        .records
        .iter()
        .filter(|record| &record.fields.task == task)
        .collect();
    let owned: Vec<immortality_core::journal::JournalRecord> =
        records.iter().map(|record| (*record).clone()).collect();

    let state = last_task_state(&owned).unwrap_or(if owned.is_empty() {
        // A task with a state directory and no journal has been registered but has not started.
        TaskState::Queued
    } else {
        TaskState::Running
    });
    // The hold is derived from the journal rather than stored: `backend_unavailable` followed by
    // `backend_restored` is the whole record of an outage, and a second copy would be a second
    // thing to keep in step (D-027).
    let hold = if backend::since_from_journal(&owned).is_some() {
        Some(Hold::WaitingBackend)
    } else {
        None
    };
    let created_at = owned.first().map_or(now, |record| record.fields.at);
    // A plan that cannot be read is not "work available": starting an attempt against a tree
    // nobody could parse would fail on the first step, and the task would spend its attempts on a
    // planner bug. It is left to `status`/`plan` to say why.
    let has_work = Plan::load(&project.path, config.loop_.max_subtask_depth)
        .is_ok_and(|plan| plan.next_ready().is_some());

    Facts {
        state,
        hold,
        created_at,
        has_work,
    }
}

/// The state the last `task_state` record announced.
pub fn last_task_state(records: &[immortality_core::journal::JournalRecord]) -> Option<TaskState> {
    records
        .iter()
        .rev()
        .filter(|record| record.fields.kind == immortality_core::journal::Kind::TaskState)
        .find_map(|record| {
            record
                .event
                .payload
                .get("state")
                .and_then(serde_json::Value::as_str)
                .and_then(|name| TaskState::parse(name).ok())
        })
}
