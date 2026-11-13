//! The plan: `PLAN.md`, `ACCEPTANCE.md` and `TASKS.json`, and what makes one usable.
//!
//! # Three files, because they are read by three different readers
//!
//! `PLAN.md` is prose for the human and for the model's brief. `ACCEPTANCE.md` is the gate
//! definition, read by the runner. `TASKS.json` is the task tree, read by the scheduler and
//! written back by the runner as subtasks settle. One file would mean a human editing prose
//! risks corrupting the tree the scheduler depends on — and it *would* happen, because the plan
//! is explicitly meant to be edited by a human mid-run.
//!
//! # Why the tree is validated here and not at use time
//!
//! A cycle in `depends_on` does not fail on write; it fails when the scheduler finds no ready
//! subtask while subtasks remain, which is hours later and does not say why. A dangling
//! dependency fails the same way. Both are cheap to detect at load and impossible to diagnose
//! from the symptom, so [`Plan::validate`] is a precondition for executing anything.
//!
//! # Why edits are picked up without a restart
//!
//! The design requires it: a human may rewrite the plan at any moment, including before
//! execution starts. So [`Plan::load`] is called before every subtask, and the file's
//! modification time and length are the cache key. A plan whose mtime and length are unchanged
//! is not re-parsed; anything else is re-read and re-validated. This is not an optimisation:
//! re-parsing on every scheduler tick would also mean a half-written file from an editor's
//! save could be read as the plan, and the mtime changes on the *next* save too, so the
//! recovery is the same read.
//!
//! # What the loader refuses
//!
//! A missing file, invalid JSON, an unknown field, a cycle, a dangling dependency, a depth
//! beyond the configured maximum, a `DoD` without a gate, duplicate ids. Each of those, if
//! accepted, produces a run that cannot be finished or a gate that always passes.
//!
//! A *zero-subtask* plan is refused as well: a plan with nothing to do is the planner failing
//! silently, and the task lifecycle already states that an empty decomposition is not "done".

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::atomic::write_atomically;
use crate::domain::{Dod, Subtask, SubtaskId, SubtaskState, SubtaskTrigger, Timestamp};

/// Schema version of `TASKS.json`.
pub const SCHEMA_VERSION: u16 = 1;

/// Why a plan cannot be executed.
///
/// Not `Clone`/`PartialEq`: it wraps `io::Error` and `serde_json::Error`, neither of which is
/// either. Comparing errors was never the point — tests match on the variant — and a cloneable
/// error would only invite callers to store and re-report one instead of fixing the plan.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{path} is not valid JSON: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("Rust/JSON error in {path}: {source}")]
    Encode {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("unsupported schema version {found}; this build reads {expected}")]
    Version { found: u16, expected: u16 },

    #[error("the plan has no subtasks; an empty plan is a planner failure, not a finished task")]
    NoSubtasks,

    #[error("two subtasks share the id `{id}`")]
    DuplicateId { id: String },

    #[error("subtask `{id}` depends on `{missing}`, which is not in the plan")]
    DanglingDependency { id: String, missing: String },

    #[error("a subtask cannot depend on itself: `{id}`")]
    SelfDependency { id: String },

    #[error("the dependencies form a cycle: {path}")]
    Cycle { path: String },

    #[error("subtask `{id}` depends on its own ancestor `{ancestor}`; children run before parents")]
    AncestorDependency { id: String, ancestor: String },

    #[error("subtask `{id}` is at depth {depth}, beyond the configured maximum of {max}")]
    TooDeep { id: String, depth: u8, max: u8 },

    #[error("subtask `{id}` declares depth {declared} but sits at depth {actual}; the tree is the truth")]
    DepthMismatch { id: String, declared: u8, actual: u8 },

    #[error("subtask `{id}` has an unusable definition of done: {reason}")]
    BadDod { id: String, reason: String },

    #[error("subtask `{id}` has an empty title or goal")]
    EmptyText { id: String },

    #[error("`{id}` is not a subtask identifier; ids look like `sub_<26 characters>`")]
    BadId { id: String },

    #[error("there is no subtask `{id}` in this plan")]
    UnknownSubtask { id: String },

    #[error("`ACCEPTANCE.md` is missing; the task has no gate to pass")]
    NoAcceptance,
}

/// The parsed task tree, ready to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// The repository root the plan was read from.
    root: PathBuf,
    subtasks: Vec<Subtask>,
    /// The file's mtime and length, used to decide whether a reload is needed.
    stamp: Stamp,
}

/// What identifies the version of the file on disk.
///
/// mtime *and* length: some editors write in place and preserve mtime to the second, and a
/// same-size rewrite is exactly what a human fixing a typo produces. Two independent signals
/// make a silently stale plan much less likely than either alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stamp {
    modified: Option<std::time::SystemTime>,
    len: u64,
}

impl Stamp {
    fn of(metadata: &std::fs::Metadata) -> Self {
        Self {
            modified: metadata.modified().ok(),
            len: metadata.len(),
        }
    }
}

/// The on-disk form of `TASKS.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileBody {
    v: u16,
    subtasks: Vec<Subtask>,
}

impl Plan {
    /// Load and validate the plan under `root`.
    ///
    /// # Errors
    /// Any [`PlanError`]: a missing or malformed file, or a tree that cannot execute.
    pub fn load(root: impl AsRef<Path>, max_depth: u8) -> Result<Self, PlanError> {
        let root = root.as_ref().to_path_buf();
        let acceptance = root.join(".agent").join("ACCEPTANCE.md");
        if !acceptance.exists() {
            return Err(PlanError::NoAcceptance);
        }
        let path = tasks_path(&root);
        let metadata = std::fs::metadata(&path).map_err(|source| PlanError::Io {
            path: path.clone(),
            source,
        })?;
        let stamp = Stamp::of(&metadata);
        let text = std::fs::read_to_string(&path).map_err(|source| PlanError::Io {
            path: path.clone(),
            source,
        })?;
        let body: FileBody = serde_json::from_str(&text).map_err(|source| PlanError::Json {
            path: path.clone(),
            source,
        })?;
        if body.v != SCHEMA_VERSION {
            return Err(PlanError::Version {
                found: body.v,
                expected: SCHEMA_VERSION,
            });
        }
        let subtasks = body.subtasks;
        validate(&subtasks, max_depth)?;
        Ok(Self {
            root,
            subtasks,
            stamp,
        })
    }

    /// Reload if the file changed since this plan was loaded.
    ///
    /// Returns `None` when nothing changed, so a caller can keep the plan it already has
    /// without paying for a parse. A file that disappeared is an error, not "unchanged": the
    /// plan is the thing being executed, and its absence is not a no-op.
    ///
    /// # Errors
    /// The file changed and the new version is malformed or unexecutable.
    pub fn refresh(&self, max_depth: u8) -> Result<Option<Self>, PlanError> {
        let path = tasks_path(&self.root);
        let metadata = std::fs::metadata(&path).map_err(|source| PlanError::Io {
            path: path.clone(),
            source,
        })?;
        if Stamp::of(&metadata) == self.stamp {
            return Ok(None);
        }
        Self::load(&self.root, max_depth).map(Some)
    }

    /// The task tree.
    #[must_use]
    pub fn subtasks(&self) -> &[Subtask] {
        &self.subtasks
    }

    /// The repository root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// A subtask by id, at either level of the tree.
    #[must_use]
    pub fn subtask(&self, id: &SubtaskId) -> Option<&Subtask> {
        find(&self.subtasks, id)
    }

    /// The next subtask the scheduler may run.
    ///
    /// A subtask's children are its prerequisites: the children run first, and the parent only
    /// becomes ready once they are all settled. That is what a split means, and it also makes a
    /// planner-written tree behave the way its nesting reads. Otherwise a subtask is ready when
    /// it is `Pending` and every id in `depends_on` is settled. Order is the file's order at
    /// every level, so a human who reorders `TASKS.json` reorders execution.
    #[must_use]
    pub fn next_ready(&self) -> Option<&Subtask> {
        self.next_ready_with(&BTreeSet::new())
    }

    /// What would be ready if `id` were settled, without mutating the tree.
    ///
    /// This is what L5 asks before blocking a subtask: "if I retire this one, is there other work
    /// to do?" Answering it by looking for any `Pending` sibling would be wrong, because a
    /// pending subtask whose dependency is also pending is not independent work — it is a queue
    /// with nothing at its head.
    #[must_use]
    pub fn next_ready_after_retiring(&self, id: &SubtaskId) -> Option<&Subtask> {
        let retired = BTreeSet::from([id.as_str()]);
        self.next_ready_with(&retired)
    }

    fn next_ready_with(&self, retired: &BTreeSet<&str>) -> Option<&Subtask> {
        let mut all: Vec<&Subtask> = Vec::new();
        for subtask in &self.subtasks {
            collect(subtask, &mut all);
        }
        let settled: BTreeSet<&str> = all
            .iter()
            .filter(|subtask| subtask.state.counts_as_settled() || retired.contains(subtask.id.as_str()))
            .map(|subtask| subtask.id.as_str())
            .collect();
        self.subtasks
            .iter()
            .find_map(|subtask| ready_within(subtask, &settled))
    }

    /// Whether every subtask in the tree is settled.
    ///
    /// Unlike [`crate::domain::Task::ready_to_finish_execution`] this cannot be called on an
    /// empty tree: a plan that passed [`Plan::load`] always has at least one subtask, and an
    /// empty one is refused, so `true` here always means "there was work and it is done".
    #[must_use]
    pub fn all_settled(&self) -> bool {
        !self.subtasks.is_empty()
            && self
                .subtasks
                .iter()
                .all(|subtask| subtask.state.counts_as_settled())
    }

    /// Every subtask in the tree that is not settled, deepest first.
    ///
    /// Deepest first because the innermost unfinished work is what a human reading `status`
    /// needs; a parent that is waiting on a child is not the interesting line.
    #[must_use]
    pub fn open_subtasks(&self) -> Vec<&Subtask> {
        let mut all: Vec<&Subtask> = Vec::new();
        for subtask in &self.subtasks {
            collect(subtask, &mut all);
        }
        let mut open: Vec<&Subtask> = all
            .into_iter()
            .filter(|subtask| !subtask.state.counts_as_settled())
            .collect();
        open.sort_by(|left, right| right.depth.cmp(&left.depth).then_with(|| left.id.cmp(&right.id)));
        open
    }

    /// Replace one subtask with its children and write the tree.
    ///
    /// The parent is retired with [`SubtaskState::Skipped`] and keeps its children, so the tree
    /// still records that the work was once one subtask. Validation happens against the tree as
    /// it is *now*, before anything is mutated, so a refused split leaves the file untouched.
    ///
    /// # Errors
    /// A [`crate::scheduler::SplitError`] for a split that would weaken the parent's `DoD` or
    /// exceed the depth limit, or a [`PlanError`] if the parent is unknown or the write fails.
    pub fn split(
        &mut self,
        parent: &SubtaskId,
        children: Vec<Subtask>,
        max_depth: u8,
        at: Timestamp,
    ) -> Result<(), crate::scheduler::SplitError> {
        let existing = self
            .subtask(parent)
            .ok_or_else(|| PlanError::UnknownSubtask {
                id: parent.to_string(),
            })?
            .clone();
        crate::scheduler::validate_split(&existing, &children, max_depth)?;
        let target = find_mut(&mut self.subtasks, parent).ok_or_else(|| PlanError::UnknownSubtask {
            id: parent.to_string(),
        })?;
        target
            .transition(SubtaskTrigger::Split, at)
            .map_err(|_| crate::scheduler::SplitError::Terminal {
                id: parent.to_string(),
                state: target.state,
            })?;
        target.children.extend(children);
        target.updated_at = at;
        self.write()?;
        Ok(())
    }

    /// Replace one subtask's state and write the tree.
    ///
    /// Written atomically: a scheduler that settles a subtask and crashes mid-write must come
    /// back to either the old tree or the new one, never to half of it.
    ///
    /// # Errors
    /// The id is not in the tree, or the write fails.
    pub fn set_state(&mut self, id: &SubtaskId, state: SubtaskState, at: Timestamp) -> Result<(), PlanError> {
        let subtask = find_mut(&mut self.subtasks, id)
            .ok_or_else(|| PlanError::UnknownSubtask { id: id.to_string() })?;
        subtask.state = state;
        subtask.updated_at = at;
        self.write()
    }

    /// Write the tree back to `TASKS.json` and take the new stamp.
    ///
    /// Sorted by id so that a diff shows what changed rather than what reordered.
    ///
    /// Takes `&mut self` because the write changes the file, and the stamp must follow it: a
    /// `refresh` after our own write that saw a difference would re-parse a file this process
    /// just produced, and a scheduler that reloads after every settled subtask is both slower
    /// and racier than one that knows its own writes.
    ///
    /// # Errors
    /// The file cannot be written or serialised.
    pub fn write(&mut self) -> Result<(), PlanError> {
        let path = tasks_path(&self.root);
        let body = FileBody {
            v: SCHEMA_VERSION,
            subtasks: sorted(&self.subtasks),
        };
        let mut text = serde_json::to_string_pretty(&body).map_err(|source| PlanError::Encode {
            path: path.clone(),
            source,
        })?;
        text.push('\n');
        write_atomically(&path, text.as_bytes()).map_err(|error| PlanError::Io {
            path,
            // `write_atomically` already wraps an io error with its path; unwrapping it here
            // would lose the source. The message carries the real cause.
            source: std::io::Error::other(error.to_string()),
        })?;
        let metadata = std::fs::metadata(tasks_path(&self.root)).map_err(|source| PlanError::Io {
            path: tasks_path(&self.root),
            source,
        })?;
        self.stamp = Stamp::of(&metadata);
        Ok(())
    }
}

/// Validate a task tree.
///
/// # Errors
/// A cycle, a dangling or self dependency, a duplicate id, a depth beyond `max_depth`, an empty
/// title or goal, an unusable `DoD`, or an empty tree.
pub fn validate(subtasks: &[Subtask], max_depth: u8) -> Result<(), PlanError> {
    if subtasks.is_empty() {
        return Err(PlanError::NoSubtasks);
    }
    let mut ids: BTreeMap<&str, u8> = BTreeMap::new();
    let mut all: Vec<&Subtask> = Vec::new();
    for subtask in subtasks {
        collect(subtask, &mut all);
    }
    for subtask in &all {
        if subtask.title.trim().is_empty() || subtask.goal.trim().is_empty() {
            return Err(PlanError::EmptyText {
                id: subtask.id.to_string(),
            });
        }
        if ids.insert(subtask.id.as_str(), subtask.depth).is_some() {
            return Err(PlanError::DuplicateId {
                id: subtask.id.to_string(),
            });
        }
        if let Err(error) = subtask.dod.validate() {
            return Err(PlanError::BadDod {
                id: subtask.id.to_string(),
                reason: error.to_string(),
            });
        }
    }
    // `depth` is a declared field, so it can lie. The tree's own nesting is the truth, and the
    // limit is about nesting: a plan that writes `depth: 1` on a node three levels down must be
    // refused, or `max_subtask_depth` is advisory.
    check_nesting(subtasks, 1, max_depth)?;
    check_ancestors(subtasks, &mut Vec::new())?;
    for subtask in &all {
        for dependency in &subtask.depends_on {
            if dependency == &subtask.id {
                return Err(PlanError::SelfDependency {
                    id: subtask.id.to_string(),
                });
            }
            if !ids.contains_key(dependency.as_str()) {
                return Err(PlanError::DanglingDependency {
                    id: subtask.id.to_string(),
                    missing: dependency.to_string(),
                });
            }
        }
    }
    detect_cycle(&all)?;
    Ok(())
}

/// Refuse a declared `depth` that disagrees with where the node actually sits.
fn check_nesting(subtasks: &[Subtask], depth: u8, max_depth: u8) -> Result<(), PlanError> {
    for subtask in subtasks {
        if subtask.depth != depth {
            return Err(PlanError::DepthMismatch {
                id: subtask.id.to_string(),
                declared: subtask.depth,
                actual: depth,
            });
        }
        // Checked after the equality above, so this fires only when the tree really is deeper
        // than the limit and the declared depth agrees with the nesting.
        if depth > max_depth {
            return Err(PlanError::TooDeep {
                id: subtask.id.to_string(),
                depth,
                max: max_depth,
            });
        }
        check_nesting(&subtask.children, depth.saturating_add(1), max_depth)?;
    }
    Ok(())
}

/// Refuse a subtask that depends on one of its own ancestors.
///
/// The dependency DFS cannot see this: containment is not in `depends_on`. At run time it is a
/// deadlock — children run before their parent, so the child waits for the parent that waits
/// for the child — and it would show up as "no ready subtask" hours later.
fn check_ancestors<'a>(subtasks: &'a [Subtask], ancestors: &mut Vec<&'a str>) -> Result<(), PlanError> {
    for subtask in subtasks {
        for dependency in &subtask.depends_on {
            if ancestors.contains(&dependency.as_str()) {
                return Err(PlanError::AncestorDependency {
                    id: subtask.id.to_string(),
                    ancestor: dependency.to_string(),
                });
            }
        }
        ancestors.push(subtask.id.as_str());
        check_ancestors(&subtask.children, ancestors)?;
        ancestors.pop();
    }
    Ok(())
}

/// Depth-first search over the dependency graph.
///
/// Reports the cycle as a path so the message names the loop rather than one node in it: "a
/// cycle exists" leaves the human to find it, and the plan is meant to be edited by a human.
fn detect_cycle(all: &[&Subtask]) -> Result<(), PlanError> {
    let index: BTreeMap<&str, &Subtask> = all.iter().map(|s| (s.id.as_str(), *s)).collect();
    let mut done: BTreeSet<&str> = BTreeSet::new();
    for subtask in all {
        let mut stack: Vec<&str> = Vec::new();
        if !visit(subtask.id.as_str(), &index, &mut done, &mut stack) {
            stack.reverse();
            let mut path = stack.join(" -> ");
            // Close the loop so the message reads `a -> b -> a` rather than leaving the reader
            // to infer that the last node is the first.
            if let Some(first) = stack.first() {
                path.push_str(" -> ");
                path.push_str(first);
            }
            return Err(PlanError::Cycle { path });
        }
    }
    Ok(())
}

/// `false` when a cycle was found; `stack` then holds the loop, in dependency order.
fn visit<'a>(
    node: &'a str,
    index: &BTreeMap<&'a str, &'a Subtask>,
    done: &mut BTreeSet<&'a str>,
    stack: &mut Vec<&'a str>,
) -> bool {
    if stack.contains(&node) {
        // Trim the prefix that led into the loop, so the reported path is the cycle itself and
        // not the tail of a walk that happened to reach it.
        let start = stack.iter().position(|entry| *entry == node).unwrap_or(0);
        stack.drain(..start);
        return false;
    }
    if done.contains(node) {
        return true;
    }
    stack.push(node);
    if let Some(subtask) = index.get(node) {
        for dependency in &subtask.depends_on {
            if !visit(dependency.as_str(), index, done, stack) {
                return false;
            }
        }
    }
    stack.pop();
    done.insert(node);
    true
}

/// The first ready subtask in this subtree, or `None`.
///
/// Children before their parent: a split subtask is a decomposition, so its children are its
/// prerequisites even when `depends_on` does not repeat them.
fn ready_within<'a>(subtask: &'a Subtask, settled: &BTreeSet<&str>) -> Option<&'a Subtask> {
    for child in &subtask.children {
        if let Some(ready) = ready_within(child, settled) {
            return Some(ready);
        }
    }
    let satisfied = subtask
        .depends_on
        .iter()
        .all(|dependency| settled.contains(dependency.as_str()));
    let children_settled = subtask
        .children
        .iter()
        .all(|child| child.state.counts_as_settled());
    // `settled` also holds ids the caller asked to treat as retired, so a `Pending` subtask can
    // appear in it. Checking membership keeps a retired subtask from being offered as ready.
    if subtask.state == SubtaskState::Pending
        && !settled.contains(subtask.id.as_str())
        && satisfied
        && children_settled
    {
        Some(subtask)
    } else {
        None
    }
}

fn collect<'a>(subtask: &'a Subtask, out: &mut Vec<&'a Subtask>) {
    out.push(subtask);
    for child in &subtask.children {
        collect(child, out);
    }
}

fn find<'a>(subtasks: &'a [Subtask], id: &SubtaskId) -> Option<&'a Subtask> {
    for subtask in subtasks {
        if &subtask.id == id {
            return Some(subtask);
        }
        if let Some(found) = find(&subtask.children, id) {
            return Some(found);
        }
    }
    None
}

fn find_mut<'a>(subtasks: &'a mut [Subtask], id: &SubtaskId) -> Option<&'a mut Subtask> {
    for subtask in subtasks {
        if &subtask.id == id {
            return Some(subtask);
        }
        if let Some(found) = find_mut(&mut subtask.children, id) {
            return Some(found);
        }
    }
    None
}

fn sorted(subtasks: &[Subtask]) -> Vec<Subtask> {
    let mut out: Vec<Subtask> = subtasks.to_vec();
    out.sort_by(|left, right| left.id.cmp(&right.id));
    for subtask in &mut out {
        subtask.children = sorted(&subtask.children);
    }
    out
}

fn tasks_path(root: &Path) -> PathBuf {
    root.join(".agent").join("TASKS.json")
}

/// One subtask as the planner writes it, before it becomes a [`Subtask`].
///
/// A separate type because `Subtask` carries `created_at`/`updated_at`, and a planner should
/// not invent timestamps: `Plan::from_draft` fills them in. Keeping one type for both would
/// mean the loader accepted a file without them and guessed, which is how a plan ends up with
/// a `created_at` of the epoch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Draft {
    pub id: String,
    #[serde(default = "one")]
    pub depth: u8,
    pub title: String,
    pub goal: String,
    pub dod: Dod,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default)]
    pub children: Vec<Self>,
}

const fn one() -> u8 {
    1
}

impl Draft {
    /// Turn a validated draft into a `Subtask`.
    ///
    /// # Errors
    /// An id that is not a `sub_` identifier.
    pub fn into_subtask(self, at: Timestamp) -> Result<Subtask, PlanError> {
        let id = SubtaskId::parse(&self.id).map_err(|_| PlanError::BadId { id: self.id.clone() })?;
        let children = self
            .children
            .into_iter()
            .map(|child| child.into_subtask(at))
            .collect::<Result<Vec<_>, _>>()?;
        let depends_on = self
            .depends_on
            .iter()
            .map(|text| SubtaskId::parse(text).map_err(|_| PlanError::BadId { id: text.clone() }))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Subtask {
            id,
            depth: self.depth,
            title: self.title,
            goal: self.goal,
            dod: self.dod,
            state: SubtaskState::Pending,
            depends_on,
            current_attempt: None,
            attempts: Vec::new(),
            report: None,
            children,
            created_at: at,
            updated_at: at,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::DodCheck;

    fn sub(id: &str, depends_on: &[&str]) -> Subtask {
        Subtask {
            id: SubtaskId::from_suffix(id),
            depth: 1,
            title: format!("title {id}"),
            goal: format!("goal {id}"),
            dod: Dod::reviewed("reviewed"),
            state: SubtaskState::Pending,
            depends_on: depends_on.iter().map(|dep| SubtaskId::from_suffix(dep)).collect(),
            current_attempt: None,
            attempts: Vec::new(),
            report: None,
            children: Vec::new(),
            created_at: crate::time::now(),
            updated_at: crate::time::now(),
        }
    }

    fn fixture(subtasks: &[Subtask]) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("repo");
        let agent = root.join(".agent");
        std::fs::create_dir_all(&agent).expect("mkdir");
        std::fs::write(agent.join("ACCEPTANCE.md"), "# acceptance\n").expect("write");
        let body = serde_json::json!({"v": SCHEMA_VERSION, "subtasks": subtasks});
        std::fs::write(
            agent.join("TASKS.json"),
            serde_json::to_string_pretty(&body).expect("json"),
        )
        .expect("write");
        (dir, root)
    }

    #[test]
    fn a_valid_plan_loads_and_exposes_its_tree() {
        let (dir, root) = fixture(&[sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[])]);
        let plan = Plan::load(&root, 2).expect("load");
        assert_eq!(plan.subtasks().len(), 1);
        assert_eq!(plan.root(), root.as_path());
        drop(dir);
    }

    #[test]
    fn a_missing_tasks_file_is_an_error_naming_the_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("repo");
        std::fs::create_dir_all(root.join(".agent")).expect("mkdir");
        std::fs::write(root.join(".agent").join("ACCEPTANCE.md"), "x").expect("write");
        let error = Plan::load(&root, 2).expect_err("missing file");
        assert!(matches!(error, PlanError::Io { .. }), "{error}");
        assert!(error.to_string().contains("TASKS.json"), "{error}");
    }

    #[test]
    fn a_missing_acceptance_file_is_refused_before_anything_executes() {
        // Without it every `DoD` gate would be unenforceable, and the run would "succeed" by
        // doing nothing.
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("repo");
        std::fs::create_dir_all(root.join(".agent")).expect("mkdir");
        std::fs::write(root.join(".agent").join("TASKS.json"), r#"{"v":1,"subtasks":[]}"#).expect("write");
        let error = Plan::load(&root, 2).expect_err("no acceptance");
        assert!(matches!(error, PlanError::NoAcceptance), "{error}");
    }

    #[test]
    fn an_empty_plan_is_refused() {
        let error = validate(&[], 2).expect_err("empty");
        assert!(matches!(error, PlanError::NoSubtasks), "{error}");
    }

    #[test]
    fn a_cycle_is_refused_and_the_message_names_the_loop() {
        let a = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &["01ARZ3NDEKTSV4RRFFQ69G5FAB"]);
        let b = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", &["01ARZ3NDEKTSV4RRFFQ69G5FAA"]);
        let error = validate(&[a, b], 2).expect_err("cycle");
        let text = error.to_string();
        assert!(text.contains("cycle"), "{text}");
        assert!(text.contains("->"), "the loop is named: {text}");
    }

    #[test]
    fn a_self_dependency_is_refused() {
        let a = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &["01ARZ3NDEKTSV4RRFFQ69G5FAA"]);
        let error = validate(&[a], 2).expect_err("self");
        assert!(matches!(error, PlanError::SelfDependency { .. }), "{error}");
    }

    #[test]
    fn a_dangling_dependency_is_refused() {
        let a = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &["01ARZ3NDEKTSV4RRFFQ69G5FAB"]);
        let error = validate(&[a], 2).expect_err("dangling");
        assert!(matches!(error, PlanError::DanglingDependency { .. }), "{error}");
    }

    #[test]
    fn a_duplicate_id_is_refused() {
        let a = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[]);
        let b = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[]);
        let error = validate(&[a, b], 2).expect_err("duplicate");
        assert!(matches!(error, PlanError::DuplicateId { .. }), "{error}");
    }

    #[test]
    fn a_dod_with_no_gate_is_refused() {
        // `checks: []` with `review: false` passes every field-level check and makes the whole
        // gate unconditionally green. The plan is where that has to be caught.
        let mut a = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[]);
        a.dod = Dod {
            statement: "looks fine".to_string(),
            checks: Vec::new(),
            review: false,
        };
        let error = validate(&[a], 2).expect_err("no gate");
        assert!(matches!(error, PlanError::BadDod { .. }), "{error}");
        assert!(error.to_string().contains("nothing would gate"), "{error}");
    }

    #[test]
    fn a_dod_check_with_a_duplicate_id_is_refused() {
        let mut a = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[]);
        a.dod = Dod {
            statement: "builds".to_string(),
            checks: vec![
                DodCheck {
                    id: "test".to_string(),
                    command: "cargo test".to_string(),
                    required: true,
                },
                DodCheck {
                    id: "test".to_string(),
                    command: "cargo build".to_string(),
                    required: true,
                },
            ],
            review: false,
        };
        let error = validate(&[a], 2).expect_err("duplicate check");
        assert!(error.to_string().contains("duplicate"), "{error}");
    }

    #[test]
    fn an_empty_title_or_goal_is_refused() {
        let mut a = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[]);
        a.goal = "   ".to_string();
        let error = validate(&[a], 2).expect_err("empty goal");
        assert!(matches!(error, PlanError::EmptyText { .. }), "{error}");
    }

    #[test]
    fn readiness_respects_dependencies_and_file_order() {
        let a = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[]);
        let b = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", &["01ARZ3NDEKTSV4RRFFQ69G5FAA"]);
        let (dir, root) = fixture(&[a, b]);
        let plan = Plan::load(&root, 2).expect("load");
        let ready = plan.next_ready().expect("something is ready");
        assert_eq!(ready.id, SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAA"));
        drop(dir);
    }

    #[test]
    fn a_settled_dependency_unblocks_the_next_subtask() {
        let mut a = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[]);
        let b = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", &["01ARZ3NDEKTSV4RRFFQ69G5FAA"]);
        let (dir, root) = fixture(&[a.clone(), b]);
        let mut plan = Plan::load(&root, 2).expect("load");
        a.state = SubtaskState::Done;
        plan.set_state(&a.id, SubtaskState::Done, crate::time::now())
            .expect("settle");
        let ready = plan.next_ready().expect("next is ready");
        assert_eq!(ready.id, SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAB"));
        drop(dir);
    }

    #[test]
    fn nothing_is_ready_while_every_candidate_is_blocked_by_dependencies() {
        // The symptom a cycle produces at run time, which is why the cycle is caught at load.
        let a = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &["01ARZ3NDEKTSV4RRFFQ69G5FAB"]);
        let b = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", &["01ARZ3NDEKTSV4RRFFQ69G5FAA"]);
        let (dir, root) = fixture(&[a, b]);
        // Loading it must fail, which is the point; this asserts the loader is what stops it.
        let error = Plan::load(&root, 2).expect_err("cycle refused at load");
        assert!(matches!(error, PlanError::Cycle { .. }), "{error}");
        drop(dir);
    }

    #[test]
    fn a_hand_edited_plan_is_picked_up_without_a_restart() {
        // The design's requirement: a human may rewrite the plan at any moment, including
        // before execution starts, and the run continues with the new version.
        let (dir, root) = fixture(&[sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[])]);
        let plan = Plan::load(&root, 2).expect("load");
        assert_eq!(plan.subtasks().len(), 1);
        assert!(plan.refresh(2).expect("unchanged").is_none());

        // Rewrite with a second subtask.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let body = serde_json::json!({
            "v": SCHEMA_VERSION,
            "subtasks": [
                sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[]),
                sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", &[]),
            ]
        });
        std::fs::write(
            root.join(".agent").join("TASKS.json"),
            serde_json::to_string_pretty(&body).expect("json"),
        )
        .expect("rewrite");

        let refreshed = plan.refresh(2).expect("reload").expect("changed");
        assert_eq!(refreshed.subtasks().len(), 2);
        drop(dir);
    }

    #[test]
    fn a_broken_edit_is_reported_rather_than_silently_ignored() {
        // A human mid-edit can leave invalid JSON; the alternative to failing is running a plan
        // that is not the one on disk.
        let (dir, root) = fixture(&[sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[])]);
        let plan = Plan::load(&root, 2).expect("load");
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(root.join(".agent").join("TASKS.json"), "{not json").expect("write");
        let error = plan.refresh(2).expect_err("broken edit");
        assert!(matches!(error, PlanError::Json { .. }), "{error}");
        drop(dir);
    }

    #[test]
    fn writing_the_plan_sorts_it_and_round_trips() {
        let b = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", &[]);
        let a = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[]);
        let (dir, root) = fixture(&[b, a]);
        let mut plan = Plan::load(&root, 2).expect("load");
        plan.write().expect("write");
        let text = std::fs::read_to_string(root.join(".agent").join("TASKS.json")).expect("read");
        let a_position = text.find("01ARZ3NDEKTSV4RRFFQ69G5FAA").expect("a");
        let b_position = text.find("01ARZ3NDEKTSV4RRFFQ69G5FAB").expect("b");
        assert!(a_position < b_position, "sorted by id");
        let reloaded = Plan::load(&root, 2).expect("reload");
        assert_eq!(reloaded.subtasks().len(), 2);
        drop(dir);
    }

    #[test]
    fn writing_does_not_make_the_next_refresh_reload() {
        // The plan writes itself as subtasks settle; if the write did not update the stamp, the
        // scheduler would re-parse after every state change.
        let (dir, root) = fixture(&[sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[])]);
        let mut plan = Plan::load(&root, 2).expect("load");
        plan.write().expect("write");
        assert!(plan.refresh(2).expect("no error").is_none());
        drop(dir);
    }

    #[test]
    fn a_wrong_schema_version_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("repo");
        std::fs::create_dir_all(root.join(".agent")).expect("mkdir");
        std::fs::write(root.join(".agent").join("ACCEPTANCE.md"), "x").expect("write");
        std::fs::write(
            root.join(".agent").join("TASKS.json"),
            r#"{"v":99,"subtasks":[]}"#,
        )
        .expect("write");
        let error = Plan::load(&root, 2).expect_err("version");
        assert!(matches!(error, PlanError::Version { found: 99, .. }), "{error}");
    }

    #[test]
    fn an_unknown_field_in_the_file_is_refused() {
        // A typo in `depends_on` would otherwise leave the dependency silently absent, and the
        // subtask would run earlier than intended.
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("repo");
        std::fs::create_dir_all(root.join(".agent")).expect("mkdir");
        std::fs::write(root.join(".agent").join("ACCEPTANCE.md"), "x").expect("write");
        std::fs::write(
            root.join(".agent").join("TASKS.json"),
            r#"{"v":1,"subtasks":[],"extra":true}"#,
        )
        .expect("write");
        let error = Plan::load(&root, 2).expect_err("unknown field");
        assert!(matches!(error, PlanError::Json { .. }), "{error}");
    }

    #[test]
    fn a_draft_becomes_a_subtask_with_a_timestamp() {
        let at = crate::time::now();
        // `id` is a full prefixed id, not a suffix: `into_subtask` parses it as one, which is
        // what the file on disk holds.
        let draft = Draft {
            id: SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAA").to_string(),
            depth: 1,
            title: "t".to_string(),
            goal: "g".to_string(),
            dod: Dod::reviewed("r"),
            depends_on: vec![SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAB").to_string()],
            children: Vec::new(),
        };
        let subtask = draft.into_subtask(at).expect("into");
        assert_eq!(subtask.state, SubtaskState::Pending);
        assert_eq!(subtask.created_at, at);
        assert_eq!(subtask.updated_at, at);
        assert_eq!(
            subtask.depends_on,
            vec![SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAB")]
        );
    }

    #[test]
    fn a_draft_with_a_bad_id_is_refused_as_a_bad_id() {
        let draft = Draft {
            id: "not-a-subtask-id".to_string(),
            depth: 1,
            title: "t".to_string(),
            goal: "g".to_string(),
            dod: Dod::reviewed("r"),
            depends_on: Vec::new(),
            children: Vec::new(),
        };
        let at = crate::time::now();
        let error = draft.into_subtask(at).expect_err("bad id");
        assert!(matches!(error, PlanError::BadId { .. }), "{error}");
        assert!(error.to_string().contains("not-a-subtask-id"), "{error}");
    }

    #[test]
    fn a_lying_depth_field_is_refused_against_the_actual_nesting() {
        // `depth` is declared, so it can claim 1 while the node sits three levels down. The
        // limit is about nesting, and an advisory limit is no limit.
        let mut child = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", &[]);
        child.depth = 1;
        let mut parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[]);
        parent.depth = 1;
        parent.children = vec![child];
        let error = validate(&[parent], 2).expect_err("lying depth");
        assert!(
            matches!(
                error,
                PlanError::DepthMismatch {
                    declared: 1,
                    actual: 2,
                    ..
                }
            ),
            "{error}"
        );
    }

    #[test]
    fn a_genuinely_deep_tree_is_refused_even_with_honest_depths() {
        let mut grandchild = sub("01ARZ3NDEKTSV4RRFFQ69G5FAC", &[]);
        grandchild.depth = 3;
        let mut child = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", &[]);
        child.depth = 2;
        child.children = vec![grandchild];
        let mut parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[]);
        parent.depth = 1;
        parent.children = vec![child];
        let error = validate(&[parent], 2).expect_err("too deep");
        assert!(
            matches!(error, PlanError::TooDeep { depth: 3, max: 2, .. }),
            "{error}"
        );
    }

    #[test]
    fn a_child_depending_on_its_own_parent_is_refused() {
        // Children run before the parent, so this is a deadlock the dependency DFS cannot see:
        // containment is not in `depends_on`.
        let mut child = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", &["01ARZ3NDEKTSV4RRFFQ69G5FAA"]);
        child.depth = 2;
        let mut parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[]);
        parent.depth = 1;
        parent.children = vec![child];
        let error = validate(&[parent], 2).expect_err("ancestor dependency");
        assert!(matches!(error, PlanError::AncestorDependency { .. }), "{error}");
    }

    #[test]
    fn children_are_ready_before_their_parent() {
        let mut child = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", &[]);
        child.depth = 2;
        let mut parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[]);
        parent.depth = 1;
        parent.children = vec![child];
        let (dir, root) = fixture(&[parent]);
        let plan = Plan::load(&root, 2).expect("load");
        let ready = plan.next_ready().expect("the child is ready");
        assert_eq!(ready.id, SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAB"));
        drop(dir);
    }

    #[test]
    fn a_parent_is_ready_once_its_children_are_settled() {
        let mut child = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", &[]);
        child.depth = 2;
        let mut parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[]);
        parent.depth = 1;
        parent.children = vec![child];
        let (dir, root) = fixture(&[parent]);
        let mut plan = Plan::load(&root, 2).expect("load");
        plan.set_state(
            &SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAB"),
            SubtaskState::Done,
            crate::time::now(),
        )
        .expect("settle child");
        let ready = plan.next_ready().expect("the parent is ready");
        assert_eq!(ready.id, SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAA"));
        drop(dir);
    }

    #[test]
    fn an_unknown_id_is_reported_as_unknown_rather_than_as_a_dependency_problem() {
        let (dir, root) = fixture(&[sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", &[])]);
        let mut plan = Plan::load(&root, 2).expect("load");
        let error = plan
            .set_state(
                &SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FFF"),
                SubtaskState::Done,
                crate::time::now(),
            )
            .expect_err("unknown");
        assert!(matches!(error, PlanError::UnknownSubtask { .. }), "{error}");
        drop(dir);
    }
}
