//! Turning a goal into a plan: the prompt, the proposal schema, and the compiler.
//!
//! # Why the model proposes names, not identifiers
//!
//! A subtask id is a ULID (`sub_01ARZ3…`). Asking a small local model to invent thirty of them is
//! asking it to be wrong thirty times, and every wrong one is a validation failure that costs a
//! whole request. So the model writes short **names** it can hold in its head (`parse`,
//! `parse_tests`), dependencies refer to those names, and [`compile`] assigns the real ids and
//! rewrites the dependencies. The name is also what makes a broken proposal readable: "subtask
//! `parse_tests` depends on `parser`, which is not in the plan" is a sentence a human can act on,
//! while "subtask `sub_01ARZ…` depends on `sub_01ARZ…`" is not.
//!
//! # Why the prompt is here and not at the call site
//!
//! Two properties are load-bearing and easy to lose in a string assembled at a call site: the
//! planner sees the goal and the repository's shape and nothing else (no conversation, no tools —
//! it is writing a plan, not executing one), and it is asked for JSON in a shape the compiler can
//! reject. [`SYSTEM`] is the single place both are true.
//!
//! # Why an invalid proposal writes nothing
//!
//! A half-written plan is worse than no plan: `TASKS.json` present with a broken sibling is a task
//! that looks planned and cannot start, and the human's next move is unclear. [`write`] validates
//! everything first and then writes all three files, and a refusal leaves the directory exactly as
//! it was.
//!
//! # Why the second `plan` refuses without `--force`
//!
//! The roadmap's rule is that a human may open `TASKS.json` and rewrite it before the run starts.
//! Regenerating the plan would discard exactly that work, silently, and the human would learn about
//! it when the task did something they had removed. `--force` exists because "the plan is wrong"
//! is also a real situation, and then the human is explicitly asking for the regenerated one.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::domain::{Dod, Subtask, SubtaskId};
use crate::plan::{self, Plan, PlanError, SCHEMA_VERSION};

/// The system message: who the planner is, and the shape of the answer.
///
/// No tools are attached to the request that carries this. That is what enforces "it writes a plan
/// instead of doing the work": a sentence in a prompt is a request, and the absence of tools is a
/// fact.
pub const SYSTEM: &str = "\
You are planning work for an autonomous coding agent that will execute the plan without further \
questions. You do not have tools and cannot read the repository; you turn a goal into a task tree.

Answer with JSON and nothing else, in exactly this shape:

{\"plan\":\"markdown\",\"acceptance\":\"markdown\",\"subtasks\":[{\
\"name\":\"unique_short_name\",\"title\":\"one line\",\"goal\":\"what to achieve, for an agent \
that has none of this conversation\",\"dod\":{\"statement\":\"the observable outcome\",\
\"checks\":[{\"id\":\"test\",\"command\":\"cargo test\"}],\"review\":true},\
\"depends_on\":[\"another_name\"],\"children\":[]}]}

Rules, and a violation gets the whole plan rejected:
- `name` is a short identifier, unique across the whole tree. Dependencies refer to names, never \
to ids: you do not write ids.
- Every subtask needs a definition of done with at least one check or `review: true`. A subtask \
with no gate cannot be verified and will not be accepted.
- `goal` must stand alone. The agent executing it will see the goal and the definition of done, \
not this conversation.
- Children run before their parent. Do not make a subtask depend on its own ancestor.
- `acceptance` is the human-readable bar for the whole task, and it must name a command or an \
observable result. A task whose acceptance is \"it works\" has no gate.
- A `command` is one program, not a pipeline. Never write `cargo test | tail -20`: a shell takes \
the exit code of the last command in a pipe, so `tail` succeeds and the check is green on a build \
that did not compile. The plan is rejected if a check contains an unquoted `|`.
- Prefer few, meaningful subtasks. One subtask per file is not a plan.";

/// The user message: the goal, and whatever the caller knows about the repository.
#[must_use]
pub fn prompt(goal: &str, context: &str) -> String {
    let mut out = String::new();
    out.push_str("# Goal\n\n");
    out.push_str(goal.trim());
    out.push_str("\n\n");
    if !context.trim().is_empty() {
        out.push_str("# The repository\n\n");
        out.push_str(context.trim());
        out.push_str("\n\n");
    }
    out.push_str("# Task\n\nProduce the plan as JSON.\n");
    out
}

/// One subtask as the model proposes it: a name, not an id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Proposed {
    /// Unique across the tree; dependencies refer to it.
    pub name: String,
    pub title: String,
    pub goal: String,
    pub dod: Dod,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default)]
    pub children: Vec<Self>,
}

/// The model's whole answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Proposal {
    /// `PLAN.md`: the human-readable description of the approach.
    #[serde(default)]
    pub plan: String,
    /// `ACCEPTANCE.md`: the bar the finished task must meet.
    #[serde(default)]
    pub acceptance: String,
    pub subtasks: Vec<Proposed>,
}

/// Read the model's answer.
///
/// Tolerant about the wrapper — a code fence, a sentence before the JSON — and strict about the
/// body, for the same reason as the reviewer: a parser that accepted prose would make "the planner
/// produced a plan" mean "the planner produced a non-empty string".
///
/// # Errors
/// The answer contains no JSON object, or the object is not a [`Proposal`].
pub fn parse(answer: &str) -> Result<Proposal, PlanError> {
    let json = enclosing_object(answer).ok_or_else(|| PlanError::BadProposal {
        reason: "the answer contains no JSON object".to_string(),
    })?;
    serde_json::from_str(json).map_err(|error| PlanError::BadProposal {
        reason: error.to_string(),
    })
}

/// The outermost `{…}` in `text`, if there is one.
///
/// Brace counting rather than a regex: the JSON contains braces inside strings, and a regular
/// expression that tried to skip them would be a second JSON parser with worse bugs.
fn enclosing_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in text[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..start + offset + ch.len_utf8()]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Turn a proposal into a validated, executable task tree.
///
/// Assigns ids, resolves dependency names, and validates the result with the same
/// [`plan::validate`] a file on disk goes through — so a plan the planner accepts and a plan the
/// scheduler rejects cannot exist.
///
/// # Errors
/// An empty goal, an empty or gateless acceptance, a duplicate name, an unknown dependency name,
/// or any validation failure in the compiled tree.
pub fn compile(
    goal: &str,
    proposal: &Proposal,
    max_depth: u8,
    at: crate::domain::Timestamp,
) -> Result<Vec<Subtask>, PlanError> {
    if goal.trim().is_empty() {
        return Err(PlanError::EmptyGoal);
    }
    if proposal.acceptance.trim().is_empty() {
        return Err(PlanError::EmptyAcceptance);
    }
    if proposal.subtasks.is_empty() {
        return Err(PlanError::NoSubtasks);
    }

    // Names are resolved in a second pass so that a dependency may point forward: a plan whose
    // first subtask waits on the third is a normal shape, and refusing it would make the model's
    // ordering constraint an ordering *requirement*.
    let mut ids: BTreeMap<String, SubtaskId> = BTreeMap::new();
    assign(&proposal.subtasks, &mut ids)?;
    let subtasks = build(&proposal.subtasks, &ids, 1, at)?;
    plan::validate(&subtasks, max_depth)?;
    Ok(subtasks)
}

/// Give every proposed subtask an id, refusing a duplicate name.
fn assign(proposed: &[Proposed], ids: &mut BTreeMap<String, SubtaskId>) -> Result<(), PlanError> {
    for subtask in proposed {
        let name = subtask.name.trim();
        if name.is_empty() {
            return Err(PlanError::BadProposal {
                reason: format!("subtask `{}` has an empty name", subtask.title.trim()),
            });
        }
        if ids.insert(name.to_string(), SubtaskId::generate()).is_some() {
            return Err(PlanError::DuplicateName {
                name: name.to_string(),
            });
        }
        assign(&subtask.children, ids)?;
    }
    Ok(())
}

/// Build the tree with ids and resolved dependencies.
fn build(
    proposed: &[Proposed],
    ids: &BTreeMap<String, SubtaskId>,
    depth: u8,
    at: crate::domain::Timestamp,
) -> Result<Vec<Subtask>, PlanError> {
    proposed
        .iter()
        .map(|subtask| {
            let id = ids
                .get(subtask.name.trim())
                .cloned()
                .ok_or_else(|| PlanError::BadProposal {
                    reason: format!("`{}` was not assigned an id", subtask.name),
                })?;
            let depends_on = subtask
                .depends_on
                .iter()
                .map(|dependency| {
                    ids.get(dependency.trim())
                        .cloned()
                        .ok_or_else(|| PlanError::DanglingDependency {
                            id: subtask.name.trim().to_string(),
                            missing: dependency.trim().to_string(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Subtask {
                id,
                depth,
                title: subtask.title.trim().to_string(),
                goal: subtask.goal.trim().to_string(),
                dod: subtask.dod.clone(),
                state: crate::domain::SubtaskState::Pending,
                depends_on,
                current_attempt: None,
                attempts: Vec::new(),
                report: None,
                children: build(&subtask.children, ids, depth + 1, at)?,
                created_at: at,
                updated_at: at,
            })
        })
        .collect()
}

/// The three files a plan consists of.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Files {
    /// `PLAN.md`.
    pub plan: String,
    /// `ACCEPTANCE.md`.
    pub acceptance: String,
    /// `TASKS.json`, already serialized and validated.
    pub tasks: String,
}

/// Render the three files for a compiled tree.
///
/// # Errors
/// The tree cannot be serialized, which would be a bug in the schema rather than in the proposal.
pub fn render(proposal: &Proposal, subtasks: &[Subtask]) -> Result<Files, PlanError> {
    let body = serde_json::json!({ "v": SCHEMA_VERSION, "subtasks": subtasks });
    let mut tasks = serde_json::to_string_pretty(&body).map_err(|source| PlanError::Encode {
        path: PathBuf::from("TASKS.json"),
        source,
    })?;
    tasks.push('\n');
    Ok(Files {
        plan: ensure_newline(&proposal.plan),
        acceptance: ensure_newline(&proposal.acceptance),
        tasks,
    })
}

fn ensure_newline(text: &str) -> String {
    let trimmed = text.trim_end();
    let mut out = trimmed.to_string();
    out.push('\n');
    out
}

/// Where a plan lives.
#[must_use]
pub fn paths(root: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let agent = root.join(crate::paths::AGENT_DIR);
    (
        agent.join("PLAN.md"),
        agent.join("ACCEPTANCE.md"),
        agent.join("TASKS.json"),
    )
}

/// What [`write`] found on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Existing {
    /// `PLAN.md` is present.
    pub plan: bool,
    /// `ACCEPTANCE.md` is present.
    pub acceptance: bool,
    /// `TASKS.json` is present.
    pub tasks: bool,
}

impl Existing {
    /// Whether anything a plan would overwrite is there.
    #[must_use]
    pub const fn any(self) -> bool {
        self.plan || self.acceptance || self.tasks
    }

    /// The files that exist, for a message naming them.
    #[must_use]
    pub fn names(self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.plan {
            out.push("PLAN.md");
        }
        if self.acceptance {
            out.push("ACCEPTANCE.md");
        }
        if self.tasks {
            out.push("TASKS.json");
        }
        out
    }
}

/// Which of the three files already exist under `root`.
#[must_use]
pub fn existing(root: &Path) -> Existing {
    let (plan, acceptance, tasks) = paths(root);
    Existing {
        plan: plan.exists(),
        acceptance: acceptance.exists(),
        tasks: tasks.exists(),
    }
}

/// The `TASKS.json` body, as [`write`] reads it back before trusting it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Written {
    v: u16,
    subtasks: Vec<Subtask>,
}

/// Write the three files, refusing to overwrite a plan a human may have edited.
///
/// `force` is the human saying "replace what is there". Without it an existing plan is an error
/// rather than a merge: the roadmap's contract is that a human may rewrite these files before the
/// run, and silently regenerating them would discard exactly that.
///
/// Validation reads [`Files::tasks`] back and checks *those* subtasks, not a tree handed in
/// beside it. Two inputs that are supposed to agree are two inputs that can disagree, and the one
/// that lands on disk is the one that must be valid.
///
/// # Errors
/// `TASKS.json` does not parse, its schema version is wrong, the tree is invalid, `ACCEPTANCE.md`
/// is empty, a file exists and `force` is false, or a write fails. Every check happens before
/// anything is written, so a refusal leaves the directory untouched.
pub fn write(root: &Path, files: &Files, max_depth: u8, force: bool) -> Result<Plan, PlanError> {
    // Validate the bytes that are about to land: this is the check `Plan::load` performs, run
    // against the serialized form, so a serializer change that produced something unloadable is
    // caught before a human has a broken file.
    let written: Written = serde_json::from_str(&files.tasks).map_err(|source| PlanError::Json {
        path: PathBuf::from("TASKS.json"),
        source,
    })?;
    if written.v != SCHEMA_VERSION {
        return Err(PlanError::Version {
            found: written.v,
            expected: SCHEMA_VERSION,
        });
    }
    if files.acceptance.trim().is_empty() {
        return Err(PlanError::EmptyAcceptance);
    }
    plan::validate(&written.subtasks, max_depth)?;

    let present = existing(root);
    if present.any() && !force {
        return Err(PlanError::AlreadyPlanned {
            files: present.names().join(", "),
        });
    }

    let (plan_path, acceptance_path, tasks_path) = paths(root);
    let agent = root.join(crate::paths::AGENT_DIR);
    std::fs::create_dir_all(&agent).map_err(|source| PlanError::Io {
        path: agent.clone(),
        source,
    })?;
    for (path, text) in [
        (&plan_path, &files.plan),
        (&acceptance_path, &files.acceptance),
        (&tasks_path, &files.tasks),
    ] {
        crate::atomic::write_atomically(path, text.as_bytes()).map_err(|error| PlanError::Io {
            path: path.clone(),
            source: std::io::Error::other(error.to_string()),
        })?;
    }

    // Load what just landed, with the same limit the loader will use. A write that returned a
    // `Plan` built from the in-memory tree would be reporting on a file it had not read.
    Plan::load(root, max_depth)
}
