//! The read-mostly commands: `plan`, `log`, `gc` and `merge`.
//!
//! # Why these live in the daemon and not in the client
//!
//! The daemon is the only writer of the journal (D-001) and the only process holding the live
//! scheduler state. Three of these commands need facts the client does not have:
//!
//! * `plan` must read the tree with the daemon's depth limit, or it would refuse a plan the
//!   scheduler happily runs;
//! * `gc` must know which tasks are unfinished, because their artifacts are untouchable;
//! * `merge` must journal `merged`, and a second writer of the journal is exactly what D-001
//!   forbids.
//!
//! `log` is the exception that proves the rule: it only reads, but it reads the journal the same
//! way `attach` does, and putting it here keeps one definition of "this task's records" — a
//! client-side filter would drift from the daemon's and disagree about which records a task owns.
//!
//! # Why `merge` aborts on conflict
//!
//! The human asked for a merge, and a conflicted index is a repository in a state nobody asked
//! for: every later command sees a half-merged tree. Aborting restores the pre-merge state and
//! reports git's own words, so the human can resolve it by hand with the context they have. The
//! alternative — leaving the conflict in place — means `immortality status` and the next `merge`
//! both lie about what is on disk.

use std::path::PathBuf;

use immortality_core::checkpoint::GitRepo;
use immortality_core::domain::{TaskId, TaskState, TaskTrigger};
use immortality_core::gc;
use immortality_core::journal::{Journal, JournalReader, JournalRecord, Kind};
use immortality_core::plan::Plan;
use immortality_core::registry::{Project, Registry};

use crate::rpc::{ErrorCode, RpcError};
use crate::server::ServerState;

/// A refusal that names the thing asked about.
fn not_found(message: impl Into<String>) -> RpcError {
    RpcError::new(ErrorCode::NotFound, message)
}

/// A refused request whose arguments were wrong.
fn bad_params(message: impl Into<String>) -> RpcError {
    RpcError::new(ErrorCode::BadParams, message)
}

/// Everything internal, with the cause attached.
fn internal(error: String) -> RpcError {
    RpcError::new(ErrorCode::Internal, error)
}

/// The project a request names, by task id or by repository path.
///
/// # Errors
/// The task is not registered, or the path is not a registered project. Guessing from the
/// daemon's own directory would make the answer depend on where launchd happened to start it,
/// which is the one directory a human never chose.
fn project_for(state: &ServerState, params: &serde_json::Value) -> Result<Project, RpcError> {
    let registry = Registry::open(state.registry_path()).map_err(|error| internal(error.to_string()))?;
    if let Some(task) = params.get("task").and_then(serde_json::Value::as_str) {
        let task = TaskId::parse(task).map_err(|error| bad_params(error.to_string()))?;
        let owner = registry
            .owner_of(&state.state_root, &task)
            .map_err(|error| internal(error.to_string()))?;
        return owner.cloned().ok_or_else(|| {
            not_found(format!(
                "no project in {} owns task {task}; `immortality status` lists what exists",
                state.registry_path().display()
            ))
        });
    }
    let wanted = params
        .get("project")
        .and_then(serde_json::Value::as_str)
        .map(PathBuf::from);
    // Deliberately *not* `std::env::current_dir()`: that is the daemon's directory, which under
    // launchd is `/` and under a hand-started `serve` is wherever the human happened to start it.
    // Neither is the repository the human is standing in, so a request that names no project is
    // refused and the client sends its own cwd instead.
    let wanted = wanted.ok_or_else(|| {
        bad_params("neither `task` nor `project` was given; the client must send its working directory")
    })?;
    let wanted = std::fs::canonicalize(&wanted).unwrap_or_else(|_| wanted.clone());
    registry
        .projects()
        .iter()
        .find(|project| {
            let path = std::fs::canonicalize(&project.path).unwrap_or_else(|_| project.path.clone());
            path == wanted || wanted.starts_with(&path)
        })
        .cloned()
        .ok_or_else(|| {
            not_found(format!(
                "{} is not a project this daemon knows about; run `immortality init` in it",
                wanted.display()
            ))
        })
}

/// `plan`: the tree as the scheduler reads it, or why it cannot be read.
///
/// The load error is a *result*, not an RPC failure: a plan that fails validation is exactly
/// what a human needs to see, and reporting it as a transport error would hide the message
/// behind `BadParams`.
pub fn plan(state: &ServerState, params: &serde_json::Value) -> Result<serde_json::Value, RpcError> {
    let project = project_for(state, params)?;
    let depth = state.config.loop_.max_subtask_depth;
    let mut body = serde_json::json!({
        "project": project.id,
        "root": project.path,
        "max_subtask_depth": depth,
    });
    match Plan::load(&project.path, depth) {
        Ok(plan) => {
            body["subtasks"] = serde_json::to_value(plan.subtasks())
                .map_err(|error| internal(format!("cannot encode the tree: {error}")))?;
            body["count"] = serde_json::json!(plan.subtasks().len());
        }
        Err(error) => {
            // The file may not exist yet, which is the normal state of a task that has not been
            // planned. `load_error` rather than `error`: the request succeeded and the answer is
            // "there is no readable plan here".
            body["load_error"] = serde_json::json!(error.to_string());
        }
    }
    Ok(body)
}

/// `log`: every record this task owns, oldest first.
///
/// No pagination and no filtering by kind: `immortality log` is the transcript, and a transcript
/// with a page boundary is one a human can read only half of. The volume is bounded by the
/// journal's own rotation, not by this command.
pub fn log(state: &ServerState, params: &serde_json::Value) -> Result<serde_json::Value, RpcError> {
    let task = params
        .get("task")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| bad_params("`task` is required"))?;
    let task = TaskId::parse(task).map_err(|error| bad_params(error.to_string()))?;
    let dir = state
        .journal_dir(&task)
        .map_err(|error| not_found(error.to_string()))?;
    let outcome = JournalReader::open(&dir, &task)
        .and_then(|reader| reader.read())
        .map_err(|error| internal(error.to_string()))?;

    // Same filter and same reason as `status` and `attach`: one journal per repository holds
    // every task of that repository, so an unfiltered transcript would show another task's work.
    let owned: Vec<&JournalRecord> = outcome
        .records
        .iter()
        .filter(|record| record.fields.task == task)
        .collect();
    let records: Vec<serde_json::Value> = owned
        .iter()
        .filter_map(|record| serde_json::to_value(record).ok())
        .collect();
    Ok(serde_json::json!({
        "task": task,
        "records": records,
        "count": records.len(),
        "next_seq": owned.last().map_or(0, |record| record.fields.seq + 1),
        "corrupt": outcome.is_corrupt(),
        "torn_tail": outcome.has_torn_tail,
    }))
}

/// `gc`: collect artifacts, temp files and logs that nothing needs any more.
///
/// The live set is every artifact referenced by a journal whose task is unfinished. That is the
/// one fact this command cannot get from the filesystem, and getting it wrong would delete the
/// body of a tool result the scheduler is about to replay.
pub fn collect_garbage(
    state: &ServerState,
    params: &serde_json::Value,
) -> Result<serde_json::Value, RpcError> {
    // `dry_run` exists because "gc deleted the wrong thing" is not recoverable: an artifact is a
    // content-addressed body with no backup, and a human who wants to see the list first should
    // not have to run `gc` and read the report afterwards.
    let dry_run = params
        .get("dry_run")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    let registry = Registry::open(state.registry_path()).map_err(|error| internal(error.to_string()))?;
    let known = registry
        .tasks(&state.state_root)
        .map_err(|error| internal(error.to_string()))?;

    let mut protected = std::collections::BTreeSet::new();
    let mut unfinished = Vec::new();
    for (project, task, _) in &known {
        let outcome = match JournalReader::open(project.agent_dir(), task).and_then(|reader| reader.read()) {
            Ok(outcome) => outcome,
            // An unreadable journal is not an empty one: treating it as empty would put its
            // artifacts in the deletable set, which is how `gc` destroys the record of the one
            // task a human is debugging.
            Err(error) => {
                return Err(internal(format!(
                    "cannot read the journal of {task}, so nothing may be collected: {error}"
                )));
            }
        };
        if crate::recovery::is_unfinished(&outcome.records) {
            unfinished.push(task.clone());
            protected.extend(gc::referenced_artifacts(&outcome.records));
        }
    }

    if dry_run {
        // A dry run reports what *would* be collected by running the same rules against a store
        // that cannot delete. `is_noop` and the counts come from the same function, so the dry
        // run and the real run cannot disagree about which files are candidates.
        return Ok(serde_json::json!({
            "dry_run": true,
            "unfinished": unfinished,
            "protected": protected.len(),
            "storage": {
                "artifact_ttl_days": state.config.storage.artifact_ttl_days,
                "log_retention_months": state.config.storage.log_retention_months,
            },
        }));
    }

    let store = immortality_core::artifacts::ArtifactStore::open(state.artifacts_dir())
        .map_err(|error| internal(error.to_string()))?;
    let report = gc::collect(
        &store,
        &state.logs_dir(),
        &state.config.storage,
        &protected,
        chrono::Local::now(),
    )
    .map_err(|error| internal(error.to_string()))?;
    Ok(serde_json::json!({
        "dry_run": false,
        "unfinished": unfinished,
        "protected": protected.len(),
        "artifacts_removed": report.artifacts_removed,
        "bytes_reclaimed": report.bytes_reclaimed,
        "artifacts_protected": report.artifacts_protected,
        "temp_removed": report.temp_removed,
        "logs_compressed": report.logs_compressed,
        "logs_deleted": report.logs_deleted,
        "noop": report.is_noop(),
    }))
}

/// `merge`: the human's one git action.
///
/// # The guards, and why each exists
///
/// * the task must be `ready_for_review`: merging a running task would publish work the gates
///   have not seen, and D-011 gives the merge to a human precisely because the gates are not the
///   whole judgement;
/// * the main checkout must be clean: merging into a dirty tree mixes the agent's work with
///   whatever a human was editing, and the result is a commit neither of them wrote;
/// * the branch must exist: a missing branch means the task never committed, and git's own
///   message ("not something we can merge") says that less clearly than naming the task.
pub fn merge(state: &ServerState, params: &serde_json::Value) -> Result<serde_json::Value, RpcError> {
    let task = params
        .get("task")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| bad_params("`task` is required"))?;
    let task = TaskId::parse(task).map_err(|error| bad_params(error.to_string()))?;

    let registry = Registry::open(state.registry_path()).map_err(|error| internal(error.to_string()))?;
    let project = registry
        .owner_of(&state.state_root, &task)
        .map_err(|error| internal(error.to_string()))?
        .ok_or_else(|| not_found(format!("no project owns task {task}")))?;

    let outcome = JournalReader::open(project.agent_dir(), &task)
        .and_then(|reader| reader.read())
        .map_err(|error| internal(error.to_string()))?;
    let owned: Vec<JournalRecord> = outcome
        .records
        .iter()
        .filter(|record| record.fields.task == task)
        .cloned()
        .collect();
    let current = crate::scheduling::last_task_state(&owned).unwrap_or(TaskState::Queued);
    // The lifecycle table is the authority on whether this edge exists, not a list of states
    // repeated here — a second copy would be the one that goes stale.
    if let Err(error) = current.trigger(TaskTrigger::Merge) {
        return Err(bad_params(format!(
            "task {task} is `{current}`; only `ready_for_review` may be merged ({error})"
        )));
    }

    let repo = GitRepo::new(&project.path);
    if repo
        .is_dirty()
        .map_err(|error| internal(format!("cannot read the main checkout: {error}")))?
    {
        return Err(bad_params(format!(
            "{} has uncommitted changes; commit or stash them before merging, so the merge \
             commit contains only the task's work",
            project.path.display()
        )));
    }
    let branch = format!("agent/{}", task.as_str());
    if !repo.branch_exists(&branch) {
        return Err(not_found(format!(
            "branch `{branch}` does not exist in {}; the task never committed",
            project.path.display()
        )));
    }

    let merge = repo.git(&["merge", "--no-ff", "-m", &format!("merge {task}"), &branch]);
    if let Err(error) = merge {
        // Abort so the repository is exactly as it was. Best effort: if `merge --abort` itself
        // fails there is nothing else to try, and the original error is the one worth reporting.
        let _ = repo.git(&["merge", "--abort"]);
        return Err(RpcError::new(
            ErrorCode::Internal,
            format!(
                "merging `{branch}` failed and was aborted; the main checkout is unchanged.\n\
                 git said: {error}\n\
                 Resolve it by hand with `git merge {branch}` in {}.",
                project.path.display()
            ),
        ));
    }

    // Journal last: the merge happened, and this record is what makes the task `merged`
    // everywhere else. Writing it before the merge would announce work that is not on disk.
    let at = immortality_core::time::now();
    let mut journal =
        Journal::open(project.agent_dir(), task.clone()).map_err(|error| internal(error.to_string()))?;
    journal
        .append_event(
            at,
            Kind::TaskState,
            serde_json::json!({ "state": TaskState::Merged.as_str() }),
        )
        .map_err(|error| internal(error.to_string()))?;
    journal.close().map_err(|error| internal(error.to_string()))?;

    Ok(serde_json::json!({
        "task": task,
        "branch": branch,
        "project": project.id,
        "path": project.path,
        "state": TaskState::Merged.as_str(),
    }))
}
