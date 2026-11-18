//! Returning a machine to the state it had before this project ever ran.
//!
//! # Why this is not `gc`
//!
//! `gc` is a janitor: it keeps every artifact a live task can still replay, and it deliberately
//! refuses to touch a repository it might need. This module is the opposite question — "what did
//! immortality ever create, so it can all go?" — and the answer is not a subset of what `gc`
//! looks at. A task's worktree, its `agent/<task>` branch, `.agent/` in the repository, the
//! registry and the daemon token are all created by this program and none of them is garbage in
//! `gc`'s sense, because `gc` is not allowed to assume the run is over.
//!
//! # Why the inventory is computed before anything is deleted
//!
//! The deletion order matters and the phases are the reason. A worktree cannot be removed with
//! `rm -rf` and leave a correct repository behind — git keeps an entry for it under
//! `.git/worktrees/`, and a deleted directory with a live entry is the "prunable" state that makes
//! `git worktree add` refuse the same path forever. So worktrees go first, through `git`, while
//! the repository still exists to be told about them. The inventory is computed up front because
//! after the first deletion the facts used to compute it — the registry, the journals — start
//! disappearing, and a plan derived mid-deletion is a plan that can forget the last tenth.
//!
//! # Why the repository is untouched beyond `.agent/`
//!
//! The human's commits are theirs. This module never runs `git reset`, never deletes a branch that
//! is checked out, and never rewrites history: it deletes the branches this program named, and it
//! says which ones it found so a human can see nothing of their own was in the list. A reset that
//! silently removed `main` would be a much worse bug than the state it was cleaning.
//!
//! # Why `dry_run` is the default in the daemon
//!
//! There is no undo. The artifacts are content-addressed bodies with no backup, and the journals
//! are the only record of what a run did. `reset` therefore has to be asked to delete twice: once
//! to list, once to act.

use std::path::{Path, PathBuf};

use crate::checkpoint::GitRepo;
use crate::paths::{RepoPaths, StatePaths};
use crate::registry::Registry;

/// The prefix every task branch has, and the checkpoint branch beside it.
///
/// Both come from the domain rather than being written here: the branch a task gets is decided in
/// one place, and a reset that spelled the prefix a second time would stop finding branches the
/// day it changed.
const TASK_BRANCH_PREFIX: &str = "agent/";

/// Everything the reset would remove, computed before anything is removed.
///
/// Each field is a list of absolute paths or names, never a predicate: a report is what a human
/// approves, and "we will delete everything matching this rule" is not something a human can
/// check.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Inventory {
    /// Task worktrees, each with the repository it belongs to.
    pub worktrees: Vec<WorktreeEntry>,
    /// Task and checkpoint branches, per repository.
    pub branches: Vec<BranchEntry>,
    /// Repositories whose `.agent/` would be removed.
    pub agent_dirs: Vec<PathBuf>,
    /// The state root's own contents: `state/`, `artifacts/`, `logs/`.
    pub state_dirs: Vec<PathBuf>,
    /// The registry file, and the daemon token.
    pub files: Vec<PathBuf>,
    /// Repositories that are registered but no longer exist on disk.
    pub missing: Vec<PathBuf>,
}

/// One worktree that belongs to a task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorktreeEntry {
    /// The repository it is a worktree of.
    pub repository: PathBuf,
    /// The worktree's own directory.
    pub path: PathBuf,
    /// The task id encoded in the directory name, for the report.
    pub task: String,
}

/// The branches this program created in one repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchEntry {
    /// The repository.
    pub repository: PathBuf,
    /// Branch names, `agent/<task>` and `agent-ckpt/<task>`.
    pub names: Vec<String>,
}

impl Inventory {
    /// Whether there is nothing to remove.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.worktrees.is_empty()
            && self.branches.iter().all(|entry| entry.names.is_empty())
            && self.agent_dirs.is_empty()
            && self.state_dirs.is_empty()
            && self.files.is_empty()
    }

    /// Every path this reset would visit, for a report.
    #[must_use]
    pub fn all_paths(&self) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = self.worktrees.iter().map(|entry| entry.path.clone()).collect();
        paths.extend(self.agent_dirs.iter().cloned());
        paths.extend(self.state_dirs.iter().cloned());
        paths.extend(self.files.iter().cloned());
        paths
    }
}

/// What a reset actually did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    /// The plan that was executed, so a dry run and a real run describe themselves the same way.
    pub inventory: Inventory,
    /// Worktrees git removed.
    pub worktrees_removed: usize,
    /// Branches deleted.
    pub branches_deleted: usize,
    /// Directories and files removed.
    pub paths_removed: usize,
    /// Things that could not be removed, with the reason.
    pub failures: Vec<String>,
}

impl Report {
    /// Whether the reset removed anything, or would have.
    #[must_use]
    pub fn is_noop(&self) -> bool {
        self.inventory.is_empty()
    }
}

/// Why a repository cannot be cleaned without a human looking first.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ResetError {
    /// The registry cannot be read, so the set of repositories is unknown.
    #[error("cannot read the registry: {0}")]
    Registry(String),
    /// A `git` command failed. The reset stops rather than deleting around it: a worktree git
    /// would not remove is one whose branch may still be checked out somewhere.
    #[error("`git` in {repository} said: {message}")]
    Git {
        /// The repository that failed.
        repository: String,
        /// git's own message.
        message: String,
    },
    /// A path exists and cannot be removed.
    #[error("cannot remove {path}: {message}")]
    Io {
        /// What could not be removed.
        path: String,
        /// The operating system's message.
        message: String,
    },
}

/// Work out everything a reset of this machine would remove, without removing it.
///
/// # Errors
/// The registry is unreadable, or `git` fails while listing worktrees and branches. Both are
/// refusals by design: a reset that guessed at the set of repositories would be a reset that
/// missed one, and a missing entry in a "delete everything" command is exactly the state it was
/// supposed to reach.
pub fn inventory(state: &StatePaths) -> Result<Inventory, ResetError> {
    let registry =
        Registry::open(state.registry()).map_err(|error| ResetError::Registry(error.to_string()))?;

    // Only paths that exist. A report that lists `artifacts/` on a machine that never ran is a
    // report a human learns to skip, and `is_empty` would be false on a pristine machine — which
    // is exactly the state a reset is supposed to be able to reach.
    let candidates = [
        state.root().join("state"),
        state.artifacts_dir(),
        state.logs_dir(),
    ];
    let mut inventory = Inventory {
        state_dirs: candidates.iter().filter(|path| path.exists()).cloned().collect(),
        files: [state.registry(), state.daemon_token()]
            .iter()
            .filter(|path| path.exists())
            .cloned()
            .collect(),
        ..Inventory::default()
    };

    for project in registry.projects() {
        if !project.path.is_dir() {
            // A registered repository that has been moved or deleted. Its state directory is
            // still under `state/`, which the wholesale `state_dirs` entry covers; there is no
            // git to ask, and reporting it is more useful than failing.
            inventory.missing.push(project.path.clone());
            continue;
        }
        let repo = GitRepo::new(&project.path);

        // Worktrees: every task worktree this repository has, by the same derived path the
        // scheduler uses. Asking git for its own list instead would include worktrees a human
        // created by hand, and those are not this program's to delete.
        let base = crate::worktree::WorktreeManager::new(&project.path).base_dir();
        for task in tasks_of(state, &project.path)? {
            let path = base.join(&task);
            if path.is_dir() {
                inventory.worktrees.push(WorktreeEntry {
                    repository: project.path.clone(),
                    path,
                    task,
                });
            }
        }

        let names = branches_of(&repo, &project.path)?;
        if !names.is_empty() {
            inventory.branches.push(BranchEntry {
                repository: project.path.clone(),
                names,
            });
        }

        let agent = RepoPaths::new(&project.path).agent_dir();
        if agent.is_dir() {
            inventory.agent_dirs.push(agent);
        }
    }

    Ok(inventory)
}

/// The task ids this program has run in `repository`, from the journals.
///
/// The journal is the authority on which tasks existed: `.agent/TASKS.json` describes the *plan*,
/// not the runs, and a repository whose `.agent/` was deleted by hand still has its branch and its
/// worktree. Reading the branch names would work only for repositories this reset has not half
/// cleaned; the journals are the record, and both are consulted below so neither is a single point
/// of failure.
fn tasks_of(state: &StatePaths, repository: &Path) -> Result<Vec<String>, ResetError> {
    let mut tasks: Vec<String> = Vec::new();
    // Branch names are the durable trace that survives a deleted `.agent/`.
    let repo = GitRepo::new(repository);
    for name in branch_names(&repo, repository, TASK_BRANCH_PREFIX)? {
        tasks.push(name.trim_start_matches(TASK_BRANCH_PREFIX).to_string());
    }
    // And the journals name tasks whose branch was never created, because the run failed before
    // its first commit.
    let registry =
        Registry::open(state.registry()).map_err(|error| ResetError::Registry(error.to_string()))?;
    if let Ok(known) = registry.tasks(state.root()) {
        for (project, task, _) in known {
            if project.path == repository {
                let text = task.to_string();
                if !tasks.contains(&text) {
                    tasks.push(text);
                }
            }
        }
    }
    tasks.sort();
    tasks.dedup();
    Ok(tasks)
}

/// The branches this program created in `repository`.
///
/// Both prefixes are listed because a checkpoint branch outlives the task branch it shadows:
/// deleting one and keeping the other leaves a repository that looks clean in `git branch` and
/// still has every commit of the run in it.
fn branches_of(repo: &GitRepo, repository: &Path) -> Result<Vec<String>, ResetError> {
    let mut names = branch_names(repo, repository, TASK_BRANCH_PREFIX)?;
    names.extend(branch_names(
        repo,
        repository,
        crate::checkpoint::CKPT_BRANCH_PREFIX,
    )?);
    names.sort();
    Ok(names)
}

/// Local branches whose name starts with `prefix`.
fn branch_names(repo: &GitRepo, repository: &Path, prefix: &str) -> Result<Vec<String>, ResetError> {
    // `refs/heads/` limits this to local branches: a remote-tracking `origin/agent/<task>` is a
    // different repository's ref and deleting it locally would be a lie about what exists.
    let output = repo
        .git(&[
            "for-each-ref",
            "--format=%(refname:short)",
            &format!("refs/heads/{prefix}"),
        ])
        .map_err(|error| ResetError::Git {
            repository: repository.display().to_string(),
            message: error.to_string(),
        })?;
    Ok(output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect())
}

/// Remove everything in `inventory`, in the order that leaves a consistent machine.
///
/// # Errors
/// A worktree cannot be removed by git, or a branch cannot be deleted. Both stop the reset: the
/// remaining steps assume the earlier ones succeeded, and continuing past a worktree git refused
/// would delete the `.agent/` that still describes it.
pub fn execute(state: &StatePaths, inventory: &Inventory) -> Result<Report, ResetError> {
    let mut report = Report {
        inventory: inventory.clone(),
        ..Report::default()
    };

    // 1. Worktrees, through git, while the repositories still exist to be asked.
    for entry in &inventory.worktrees {
        let repo = GitRepo::new(&entry.repository);
        // `--force` twice is what git requires for a worktree with modifications: the first is
        // "even if dirty", the second "even if the index is locked". A reset is allowed to discard
        // uncommitted work by definition — the whole command is "throw it all away".
        let removed = repo.git(&[
            "worktree",
            "remove",
            "--force",
            "--force",
            &entry.path.display().to_string(),
        ]);
        match removed {
            Ok(_) => report.worktrees_removed += 1,
            // A directory git no longer knows about is not a failure: the entry was pruned or the
            // path was deleted by hand, and the goal — the path is gone — still holds.
            Err(error) if !entry.path.exists() => {
                report.failures.push(format!(
                    "{} was already gone, though git said: {error}",
                    entry.path.display()
                ));
            }
            Err(error) => {
                return Err(ResetError::Git {
                    repository: entry.repository.display().to_string(),
                    message: error.to_string(),
                });
            }
        }
    }
    // Prune the administrative entries left behind by the worktrees above, so `git worktree list`
    // is empty afterwards rather than showing the paths just removed. Deduplicated in place
    // because one repository has one worktree per task, and `prune` is per repository.
    let mut pruned: Vec<PathBuf> = Vec::new();
    for entry in &inventory.worktrees {
        if pruned.contains(&entry.repository) {
            continue;
        }
        pruned.push(entry.repository.clone());
        let _ = GitRepo::new(&entry.repository).git(&["worktree", "prune"]);
    }

    // 2. Branches, after the worktrees that had them checked out.
    for entry in &inventory.branches {
        let repo = GitRepo::new(&entry.repository);
        for name in &entry.names {
            // `-D`, not `-d`: these branches were never merged on purpose — the human decides
            // that with `merge` — so `-d` would refuse every one of them.
            match repo.git(&["branch", "-D", name]) {
                Ok(_) => report.branches_deleted += 1,
                Err(error) => report
                    .failures
                    .push(format!("cannot delete branch {name}: {error}")),
            }
        }
    }

    // 3. Files and directories. Last, because everything above used them.
    for path in inventory
        .agent_dirs
        .iter()
        .chain(inventory.state_dirs.iter())
        .chain(inventory.files.iter())
    {
        match remove_path(path) {
            Ok(()) => report.paths_removed += 1,
            Err(error) => report
                .failures
                .push(format!("cannot remove {}: {error}", path.display())),
        }
    }

    // The token is the one file whose absence would make a still-running daemon refuse every
    // client, so it is removed last and only after the registry: a human who sees a failure here
    // still has a daemon they can talk to.
    let _ = state;
    Ok(report)
}

/// Remove a file or a directory tree, treating "already absent" as success.
fn remove_path(path: &Path) -> std::io::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => std::fs::remove_dir_all(path),
        Ok(_) => std::fs::remove_file(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paths::AGENT_DIR;
    use crate::registry::Project;
    use std::process::Command;

    fn git(dir: &Path, args: &[&str]) {
        let output = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .expect("git");
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    /// A registered repository with one commit and an `agent/<task>` branch.
    fn fixture() -> (tempfile::TempDir, PathBuf, StatePaths) {
        let dir = tempfile::tempdir().expect("tempdir");
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).expect("repo");
        git(&repo, &["init", "-q", "-b", "main"]);
        git(&repo, &["config", "user.email", "t@example.com"]);
        git(&repo, &["config", "user.name", "t"]);
        std::fs::write(repo.join("a.txt"), "a\n").expect("write");
        git(&repo, &["add", "."]);
        git(&repo, &["commit", "-q", "-m", "first"]);

        let root = dir.path().join("state");
        std::fs::create_dir_all(&root).expect("state root");
        std::fs::write(root.join("daemon.token"), "token\n").expect("token");
        let state = StatePaths::new(&root);
        let mut registry = Registry::open(state.registry()).expect("registry");
        registry.register(&repo, None).expect("register");

        (dir, repo, state)
    }

    #[test]
    fn nothing_created_means_nothing_to_remove() {
        // The empty case has to be the empty case: a reset that always reports work would train a
        // human to ignore the word "warning" in its report.
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("state");
        std::fs::create_dir_all(&root).expect("root");
        let state = StatePaths::new(&root);

        let inventory = inventory(&state).expect("inventory");
        assert!(inventory.is_empty(), "{inventory:?}");
        assert_eq!(inventory.worktrees, []);
        assert_eq!(inventory.branches, []);
    }

    #[test]
    fn a_task_branch_is_found_but_a_humans_branch_is_not() {
        // The rule that protects the human's work: only `agent/` and `agent-ckpt/` are ours. A
        // reset that deleted `feature/login` because it started with `f` would be a bug that
        // costs a day.
        let (_dir, repo, state) = fixture();
        git(&repo, &["branch", "agent/tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV"]);
        git(&repo, &["branch", "agent-ckpt/tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV"]);
        git(&repo, &["branch", "feature/login"]);

        let inventory = inventory(&state).expect("inventory");
        let names: Vec<String> = inventory
            .branches
            .iter()
            .flat_map(|entry| entry.names.clone())
            .collect();

        assert!(names.iter().any(|name| name.starts_with("agent/")), "{names:?}");
        assert!(
            names.iter().any(|name| name.starts_with("agent-ckpt/")),
            "{names:?}"
        );
        assert!(
            !names.iter().any(|name| name == "feature/login"),
            "a human's branch was scheduled for deletion: {names:?}"
        );
        assert!(!names.iter().any(|name| name == "main"), "{names:?}");
    }

    #[test]
    fn the_agent_directory_and_the_state_root_contents_are_both_listed() {
        let (_dir, repo, state) = fixture();
        std::fs::create_dir_all(repo.join(AGENT_DIR)).expect("agent dir");
        std::fs::create_dir_all(state.artifacts_dir()).expect("artifacts");
        std::fs::create_dir_all(state.logs_dir()).expect("logs");
        std::fs::create_dir_all(state.root().join("state").join("prj")).expect("state dir");

        let inventory = inventory(&state).expect("inventory");
        // The registry stores the canonical path, so macOS's `/var` → `/private/var` shows up as
        // a difference here; the product compares canonical forms for the same reason.
        let repo = std::fs::canonicalize(&repo).expect("canonical repo");
        assert!(
            inventory.agent_dirs.contains(&repo.join(AGENT_DIR)),
            "{inventory:?}"
        );
        assert!(inventory.state_dirs.contains(&state.artifacts_dir()));
        assert!(inventory.state_dirs.contains(&state.logs_dir()));
        assert!(inventory.state_dirs.contains(&state.root().join("state")));
        assert!(inventory.files.contains(&state.registry()));
        assert!(inventory.files.contains(&state.daemon_token()));
    }

    #[test]
    fn a_worktree_that_git_knows_about_is_removed_and_pruned() {
        // The state that makes `git worktree add` refuse the path forever: a directory deleted
        // with `rm -rf` while `.git/worktrees/<name>/` still exists. Only git can clear it.
        let (_dir, repo, state) = fixture();
        let task = "tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV";
        let path = crate::worktree::WorktreeManager::new(&repo).path_for(task);
        let branch = format!("agent/{task}");
        crate::worktree::WorktreeManager::new(&repo)
            .ensure(task, &branch, None)
            .expect("worktree");

        let inventory = inventory(&state).expect("inventory");
        assert_eq!(inventory.worktrees.len(), 1, "{inventory:?}");
        // macOS resolves `/var` to `/private/var`, and the worktree manager derives its path from
        // the undiscovered spelling. Comparing canonical forms is what the product does too.
        assert_eq!(
            std::fs::canonicalize(&inventory.worktrees[0].path).expect("canonical"),
            std::fs::canonicalize(&path).expect("canonical")
        );

        let report = execute(&state, &inventory).expect("execute");
        assert_eq!(report.worktrees_removed, 1, "{report:?}");
        assert!(!path.exists(), "the worktree directory must be gone");
        // No prunable entry is left: git no longer lists it, so the same path is usable again.
        let list = GitRepo::new(&repo).git(&["worktree", "list"]).expect("list");
        assert!(
            !list.contains(task),
            "a prunable entry survived the reset: {list}"
        );
    }

    #[test]
    fn executing_removes_the_inventory_and_is_idempotent() {
        let (_dir, repo, state) = fixture();
        std::fs::create_dir_all(repo.join(AGENT_DIR)).expect("agent dir");
        std::fs::write(repo.join(AGENT_DIR).join("PLAN.md"), "# Plan\n").expect("plan");
        git(&repo, &["branch", "agent/tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV"]);

        let inventory = inventory(&state).expect("inventory");
        let report = execute(&state, &inventory).expect("execute");
        assert_eq!(report.failures, Vec::<String>::new(), "{report:?}");
        assert!(!repo.join(AGENT_DIR).exists());
        assert!(!state.registry().exists());
        let branches = GitRepo::new(&repo).git(&["branch"]).expect("branch");
        assert!(!branches.contains("agent/"), "{branches}");

        // A second run over the same machine is a no-op, because a script may call it twice.
        let again = super::inventory(&state).expect("second inventory");
        assert!(again.is_empty(), "{again:?}");
    }

    #[test]
    fn a_registered_repository_that_is_gone_is_reported_not_failed_on() {
        // A repository that was moved: there is no git to ask, and the reset of everything else
        // must still be possible. Its state directory is under `state/` and is removed with it.
        let (_dir, repo, state) = fixture();
        let mut registry = Registry::open(state.registry()).expect("registry");
        registry
            .register(Path::new("/nonexistent/immortality/nope"), None)
            .expect("register a missing repo");
        std::fs::remove_dir_all(&repo).expect("delete the repository");

        let inventory = inventory(&state).expect("inventory");
        assert!(
            inventory
                .missing
                .iter()
                .any(|path| path == Path::new("/nonexistent/immortality/nope")),
            "{inventory:?}"
        );
        execute(&state, &inventory).expect("execute");
    }

    #[test]
    fn a_project_is_a_path_and_a_name_and_this_module_uses_both() {
        // Guards the field names the inventory builds `BranchEntry`/`WorktreeEntry` from; a rename
        // here would be a compile error, and this test exists so the intent is written down.
        let project = Project {
            id: "prj_x".to_string(),
            path: PathBuf::from("/tmp/repo"),
            remote: None,
        };
        assert_eq!(project.path, PathBuf::from("/tmp/repo"));
    }
}
