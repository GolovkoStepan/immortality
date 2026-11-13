//! `plan --goal`: turning a sentence into `PLAN.md`, `ACCEPTANCE.md` and `TASKS.json`.
//!
//! # Why this is a daemon method and not client logic
//!
//! Same reason as everything else in [`crate::commands`] (D-088): the depth limit the plan is
//! validated against is the daemon's, and the model call belongs to the process that holds the
//! one inference slot. A client that built its own request would be a second definition of what
//! a plan is, and the two would drift.
//!
//! # Why the model is built inside `spawn_blocking`
//!
//! [`LlmModel`] owns a current-thread tokio runtime, and a runtime **cannot be dropped from an
//! async context** — tokio panics with "Cannot drop a runtime in a context where blocking is not
//! allowed". A daemon that held one in [`ServerState`] would therefore panic on shutdown, in the
//! one moment where nobody is watching. Building it, using it and dropping it all inside a
//! blocking task keeps the drop on a thread that is allowed to block, and costs one request's
//! worth of client construction — plans are made once per task, not once per step.
//!
//! # Why a refusal happens before the model is asked
//!
//! `plan --goal` on a repository that already has a plan is refused unless `--force`, and that
//! refusal must not cost a model call: the answer would be thrown away, and on a local backend a
//! plan takes minutes of prefill. The check is repeated inside [`immortality_core::planner::write`]
//! because that is the authority — the early one is about not paying, not about correctness.
//!
//! # Why the repository context is a listing and not a search
//!
//! The planner has no tools, by design: it writes a plan instead of executing one, and a tool
//! would let it start "looking around" without ever finishing. A sorted listing of the
//! repository's top level is the cheap, deterministic shape that still tells it "this is a Rust
//! workspace" or "this is a text repo" — enough to stop it proposing a `Cargo.toml` change in a
//! project that has none.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use immortality_core::config::Config;
use immortality_core::domain::Subtask;
use immortality_core::plan::{Plan, PlanError};
use immortality_core::planner::{self, Proposal};
use immortality_llm::Backend;
use immortality_llm::wire::ChatRequest;
use serde_json::Value;

use crate::rpc::{ErrorCode, RpcError};
use crate::runner::{LlmModel, Model};
use crate::server::ServerState;

/// Tokens allowed for the planner's answer.
///
/// Deliberately larger than [`immortality_llm::wire::DEFAULT_MAX_TOKENS`]: that default is sized
/// for one step of tool calls, and a plan is ten subtasks each with a goal, a definition of done
/// and a check. A plan cut off at 1024 tokens is not a smaller plan, it is unparseable JSON, and
/// the whole call is wasted.
pub const MAX_PLAN_TOKENS: u32 = 4096;

/// Entries listed in the repository context.
///
/// A monorepo with ten thousand files would fill the window with names the planner cannot use.
/// The top level is what decides the project's shape; recursion is the executor's job, not the
/// planner's.
pub const MAX_CONTEXT_ENTRIES: usize = 200;

/// Directories that say nothing about the project and would crowd out the ones that do.
const NOISE: &[&str] = &[".git", ".agent", "target", "node_modules", ".venv", "vendor"];

fn bad_params(message: impl Into<String>) -> RpcError {
    RpcError::new(ErrorCode::BadParams, message)
}

fn internal(message: impl Into<String>) -> RpcError {
    RpcError::new(ErrorCode::Internal, message)
}

/// The request that asks for a plan.
///
/// No `tools`: that absence is what makes the planner a planner. [`planner::SYSTEM`] says so in
/// words; omitting the field is what enforces it.
#[must_use]
pub fn request(config: &Config, goal: &str, context: &str) -> ChatRequest {
    ChatRequest::new(config.models.planner.clone())
        .with_system(planner::SYSTEM)
        .with_user(planner::prompt(goal, context))
        .with_max_tokens(MAX_PLAN_TOKENS)
        // A planner that reasons about how to decompose a goal is exactly the profile the backend
        // allows thinking on; the executor disables it because there the reasoning is spent on
        // steps the journal already records.
        .with_thinking(config.backend.enable_thinking)
}

/// A one-screen description of the repository: its top-level entries, sorted.
///
/// Names only. Reading file contents would mean choosing which files matter, which is the
/// planner's decision, and it has no tools precisely so it cannot make it.
#[must_use]
pub fn context_of(root: &Path) -> String {
    let Ok(entries) = std::fs::read_dir(root) else {
        return String::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            if NOISE.contains(&name.as_str()) || name.starts_with('.') && name != ".gitignore" {
                return None;
            }
            let is_dir = entry.file_type().is_ok_and(|kind| kind.is_dir());
            Some(if is_dir { format!("{name}/") } else { name })
        })
        .collect();
    names.sort();
    names.truncate(MAX_CONTEXT_ENTRIES);
    let mut out = String::from("Top level of the repository:\n");
    for name in names {
        out.push_str("  ");
        out.push_str(&name);
        out.push('\n');
    }
    out
}

/// Read the model's answer into a validated tree.
///
/// The two steps are kept together because neither is useful alone: a parsed proposal that does
/// not validate is a plan that cannot run, and [`planner::compile`] runs the same
/// [`immortality_core::plan::validate`] the loader does, so a plan the planner accepts is one the
/// scheduler will accept.
///
/// # Errors
/// The answer is not JSON, the tree is invalid, or a dependency names a missing subtask.
pub fn compile_answer(
    goal: &str,
    answer: &str,
    max_depth: u8,
) -> Result<(Proposal, Vec<Subtask>), PlanError> {
    let proposal = planner::parse(answer)?;
    let subtasks = planner::compile(goal, &proposal, max_depth, immortality_core::time::now())?;
    Ok((proposal, subtasks))
}

/// The backend client described by the configuration.
fn backend(config: &Config) -> Result<Backend, RpcError> {
    Backend::new(
        config.backend.base_url.clone(),
        Duration::from_secs(config.backend.connect_timeout_secs),
        Duration::from_secs(config.backend.request_timeout_secs),
    )
    .map_err(|error| internal(format!("cannot build the HTTP client: {error}")))
}

/// `plan` with a `goal`: ask the model, validate, and write the three files.
///
/// # Errors
/// An empty goal, an existing plan without `force`, a model failure, or a proposal that does not
/// validate. Every refusal leaves the directory untouched.
pub async fn write_plan(state: &ServerState, params: &Value) -> Result<Value, RpcError> {
    let project = crate::commands::project_for(state, params)?;
    let goal = params
        .get("goal")
        .and_then(Value::as_str)
        .ok_or_else(|| bad_params("`goal` is required to plan"))?
        .trim()
        .to_string();
    if goal.is_empty() {
        return Err(bad_params(
            "the goal is empty; `immortality plan --goal \"...\"` needs a sentence to plan",
        ));
    }
    let force = params.get("force").and_then(Value::as_bool).unwrap_or(false);

    // Refuse before spending a model call. `planner::write` refuses again, and that is the
    // authority; this one exists because a refused plan that took three minutes of prefill is a
    // refusal a human pays for.
    let present = planner::existing(&project.path);
    if present.any() && !force {
        return Err(bad_params(format!(
            "a plan is already here ({}); edit it by hand, or pass --force to replace it",
            present.names().join(", ")
        )));
    }

    let root = project.path.clone();
    let depth = state.config.loop_.max_subtask_depth;
    let injected = state.model.clone();
    let config = state.config.clone();
    let asked = goal.clone();
    let plan = tokio::task::spawn_blocking(move || {
        let context = context_of(&root);
        let request = request(&config, &asked, &context);
        let model: Arc<dyn Model> = match injected {
            Some(model) => model,
            None => Arc::new(
                LlmModel::new(backend(&config)?)
                    .map_err(|error| internal(format!("cannot start the model client: {error}")))?,
            ),
        };
        // A model failure is `internal`, not `bad_params`: the goal was fine and the repository
        // was fine, and telling the human to check their arguments would send them to the wrong
        // place.
        let answer = model
            .complete(&request)
            .map_err(|error| internal(format!("the planner could not answer: {error}")))?;
        let (proposal, subtasks) = compile_answer(&asked, &answer.content, depth)
            .map_err(|error| internal(format!("the planner's answer is unusable: {error}")))?;
        let files = planner::render(&proposal, &subtasks)
            .map_err(|error| internal(format!("cannot render the plan: {error}")))?;
        planner::write(&root, &files, depth, force)
            .map_err(|error| internal(format!("cannot write the plan: {error}")))
    })
    .await
    .map_err(|error| internal(format!("the planning task ended unexpectedly: {error}")))??;

    Ok(report(&project.id, &project.path, depth, &plan, &goal))
}

/// The answer to a successful `plan --goal`, in the shape `immortality plan` already renders.
///
/// The same fields as the read-only `plan`, plus the goal that produced them: a client that
/// rendered the two differently would make the freshly planned case look like a different
/// command's output.
fn report(project: &str, root: &Path, depth: u8, plan: &Plan, goal: &str) -> Value {
    let subtasks: Vec<Subtask> = plan.subtasks().to_vec();
    serde_json::json!({
        "project": project,
        "root": root,
        "max_subtask_depth": depth,
        "subtasks": subtasks,
        "count": subtasks.len(),
        "goal": goal,
        "written": planner::existing(root).names(),
        "created": true,
    })
}
