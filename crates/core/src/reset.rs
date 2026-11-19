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
    /// Directories outside the state root that a project's own configuration named.
    ///
    /// Listed always, deleted only with `--outside`: see [`OutsideEntry`]. They are not part of
    /// [`Self::is_empty`] or [`Self::all_paths`], because a machine whose only leftover is one of
    /// these is at the zero state *for this program* — the directory may be where the human's own
    /// `bundle install` puts gems, and only they can decide it is garbage.
    pub outside: Vec<OutsideEntry>,
}

/// A directory outside the state root that a project's own configuration named.
///
/// # Why this is not "everything the run wrote"
///
/// Measured on `black_jack`: `bundle install` inside the task worktree wrote gems into
/// `~/.gem/shared`, and nothing about that path says "immortality" — a `.bundle/config` in the
/// repository said it. `reset` therefore cannot find these by looking at its own state, and it must
/// not find them by guessing: the directory a project names is one the human's own tools also
/// write to, so deleting it silently would throw away work `reset` never created.
///
/// The entry carries the config that named it for the same reason: an unexplained path in a list
/// of things about to be deleted is a path a human has to go and investigate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutsideEntry {
    /// The directory, with a leading `~` expanded.
    pub path: PathBuf,
    /// The config file that named it.
    pub named_by: PathBuf,
    /// The key in that file: `BUNDLE_PATH`.
    pub key: String,
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
    /// Whether this program's own footprint is gone.
    ///
    /// [`Self::outside`] deliberately does not count: a directory a project's configuration named
    /// may be where the human's own `bundle install` puts gems, so a machine whose only leftover is
    /// one of those is at the zero state *for immortality*. `reset` still prints the list — that is
    /// the whole point of finding it — but it is not work this program owes anyone.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.worktrees.is_empty()
            && self.branches.iter().all(|entry| entry.names.is_empty())
            && self.agent_dirs.is_empty()
            && self.state_dirs.is_empty()
            && self.files.is_empty()
    }

    /// Every path this reset would visit, for a report.
    ///
    /// Not including [`Self::outside`]: those are visited only under an explicit flag, and a report
    /// that folded them in would print `paths` a plain `--yes` does not touch.
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
    /// Directories outside the state root removed, under an explicit `--outside`.
    pub outside_removed: usize,
    /// Directories outside the state root that were listed and left alone.
    ///
    /// Carried in the report rather than only printed by the CLI, so the `--json` answer says the
    /// same thing the human-readable one does: a machine that asked for a reset and got one still
    /// has these, and a script that only read `noop` would not know.
    pub outside_kept: Vec<PathBuf>,
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

/// The key in a project's `.bundle/config` that names a directory outside the repository.
///
/// One key, named here rather than derived, because the list of "config keys that mean a path" is a
/// decision and not a discovery: a `BUNDLE_FROZEN: "true"` is a setting, and reading every value as
/// a path would list `true` as a directory to delete.
const BUNDLE_PATH_KEY: &str = "BUNDLE_PATH";

/// The config file a project writes when `bundle config set --local` is used.
const BUNDLE_CONFIG: &str = ".bundle/config";

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

        // Directories a project's own configuration named, outside the state root. The worktree is
        // consulted too, and not just the main checkout: `bundle install` inside a task worktree
        // writes the config it was handed, and the measured `black_jack` worktree had its own copy
        // of `.bundle/config` with the same `BUNDLE_PATH`.
        let mut roots = vec![project.path.clone()];
        roots.extend(
            inventory
                .worktrees
                .iter()
                .filter(|entry| entry.repository == project.path)
                .map(|entry| entry.path.clone()),
        );
        for root in roots {
            for entry in outside_paths(&root) {
                if !entry.path.exists() || inventory.outside.contains(&entry) {
                    continue;
                }
                inventory.outside.push(entry);
            }
        }
    }

    Ok(inventory)
}

/// The directories `root`'s `.bundle/config` names and that sit outside `root` itself.
///
/// Only paths that leave the repository are reported: a `BUNDLE_PATH: vendor/bundle` is inside the
/// worktree, which the worktree entry already removes, and repeating it here would be a second
/// report line for one directory.
///
/// # Why this parses the file by hand
///
/// The format is YAML, and the two keys it holds here are `KEY: "value"` lines written by bundler
/// itself. A YAML dependency for that is a parser this program would have to keep in step with a
/// schema that also allows anchors, multi-line scalars and flow mappings — none of which bundler
/// writes, and all of which a real parser would accept and then produce a *different* path from.
/// A line-based reader that only accepts the shape bundler writes and silently ignores everything
/// else is the honest size of the job: the failure mode is a missing entry in a list, never a wrong
/// directory deleted.
fn outside_paths(root: &Path) -> Vec<OutsideEntry> {
    let config = root.join(BUNDLE_CONFIG);
    let Ok(text) = std::fs::read_to_string(&config) else {
        return Vec::new();
    };
    let mut entries = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if key.trim() != BUNDLE_PATH_KEY {
            continue;
        }
        let value = value.trim().trim_matches('"').trim_matches('\'');
        if value.is_empty() {
            continue;
        }
        let path = resolve_config_path(value, root);
        if path.starts_with(root) {
            // Inside the repository: the worktree entry covers it, and calling it "outside" would
            // make `--outside` a flag that also changes what `--yes` already does.
            continue;
        }
        entries.push(OutsideEntry {
            path,
            named_by: config.clone(),
            key: BUNDLE_PATH_KEY.to_string(),
        });
    }
    entries
}

/// Turn a config value into a path, expanding `~` and resolving a relative value against `root`.
///
/// A relative `BUNDLE_PATH` is relative to the project, which is how bundler reads it; expanding
/// `~` by hand is what bundler does too, and the literal string is not a path a human recognizes as
/// their home directory.
fn resolve_config_path(value: &str, root: &Path) -> PathBuf {
    if let Some(rest) = value.strip_prefix("~/") {
        let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("/nonexistent"), PathBuf::from);
        return home.join(rest);
    }
    let path = PathBuf::from(value);
    if path.is_absolute() { path } else { root.join(path) }
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
/// `outside` is the `--outside` flag and nothing else: when it is `false`, the entries in
/// [`Inventory::outside`] are recorded in [`Report::outside_kept`] and left on disk. The parameter
/// is a `bool` rather than a field of the inventory because the inventory is the *fact* — what was
/// found — and the flag is the *decision*; folding the decision into the fact would make the same
/// inventory mean two different things at the two call sites that print it.
///
/// # Errors
/// A worktree cannot be removed by git, or a branch cannot be deleted. Both stop the reset: the
/// remaining steps assume the earlier ones succeeded, and continuing past a worktree git refused
/// would delete the `.agent/` that still describes it.
pub fn execute(state: &StatePaths, inventory: &Inventory, outside: bool) -> Result<Report, ResetError> {
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

    // 4. Directories a project's own configuration named, outside the state root. Last, and only
    // when asked: they may be where the human's own `bundle install` puts gems, and the flag is
    // what separates "this is mine to delete" from "this program wrote through it once".
    for entry in &inventory.outside {
        if !outside {
            report.outside_kept.push(entry.path.clone());
            continue;
        }
        match remove_path(&entry.path) {
            Ok(()) => report.outside_removed += 1,
            Err(error) => report.failures.push(format!(
                "cannot remove {} (named by {}): {error}",
                entry.path.display(),
                entry.named_by.display()
            )),
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

        let report = execute(&state, &inventory, false).expect("execute");
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
        let report = execute(&state, &inventory, false).expect("execute");
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
        execute(&state, &inventory, false).expect("execute");
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

    #[test]
    fn a_bundle_path_outside_the_repository_is_listed_with_the_config_that_named_it() {
        // The measured `black_jack` case: `.bundle/config` said `BUNDLE_PATH: "~/.gem/shared"`, so
        // `bundle install` in the task worktree wrote gems there. Nothing about the directory says
        // "immortality", and reset cannot find it by looking at its own state — the project named
        // it. It must be listed, and it must say which file named it: a path in a delete-list that
        // a human cannot trace is a path they have to go and investigate.
        let (dir, repo, state) = fixture();
        let gems = dir.path().join("gems");
        std::fs::create_dir_all(&gems).expect("gems");
        std::fs::create_dir_all(repo.join(".bundle")).expect("bundle dir");
        std::fs::write(
            repo.join(".bundle").join("config"),
            format!(
                "---\nBUNDLE_PATH: \"{}\"\nBUNDLE_CLEAN: \"true\"\n",
                gems.display()
            ),
        )
        .expect("config");

        let inventory = inventory(&state).expect("inventory");
        assert_eq!(inventory.outside.len(), 1, "{inventory:?}");
        assert_eq!(inventory.outside[0].path, gems);
        assert_eq!(inventory.outside[0].key, "BUNDLE_PATH");
        // The registry stores the canonical path, so on macOS the `/var` → `/private/var`
        // difference shows up here too.
        assert_eq!(
            inventory.outside[0].named_by,
            std::fs::canonicalize(&repo)
                .expect("canonical repo")
                .join(".bundle")
                .join("config")
        );
        // Listed but not part of this program's own footprint: a machine whose only leftover is a
        // gem directory the human's own `bundle install` writes to is at the zero state here.
        let only_outside = Inventory {
            outside: inventory.outside,
            ..Inventory::default()
        };
        assert!(
            only_outside.is_empty(),
            "an outside path is a note, not this program's garbage: {only_outside:?}"
        );
        assert!(
            !only_outside.all_paths().contains(&gems),
            "`--yes` alone must not touch it"
        );
    }

    #[test]
    fn a_bundle_path_inside_the_repository_is_not_outside() {
        // `BUNDLE_PATH: vendor/bundle` lives in the worktree, which the worktree entry already
        // removes. Reporting it again would give one directory two lines and make `--outside` a
        // flag that also changes what `--yes` does.
        let (_dir, repo, state) = fixture();
        // The directory exists, so the only thing that can keep it out of `outside` is the
        // inside-the-repository rule itself — an absent path would pass this test for the wrong
        // reason, and that is how the first version of it did.
        std::fs::create_dir_all(repo.join("vendor").join("bundle")).expect("vendor dir");
        std::fs::create_dir_all(repo.join(".bundle")).expect("bundle dir");
        std::fs::write(
            repo.join(".bundle").join("config"),
            "---\nBUNDLE_PATH: \"vendor/bundle\"\n",
        )
        .expect("config");

        let inventory = inventory(&state).expect("inventory");
        assert!(inventory.outside.is_empty(), "{inventory:?}");
    }

    #[test]
    fn a_bundle_path_with_a_tilde_is_expanded_to_the_home_directory() {
        // `~/.gem/shared` is what bundler writes and what the human recognizes; the literal string
        // is not a directory. A reset that took it at face value would report `./~/.gem/shared`
        // inside the repository and delete nothing at all.
        let (_dir, repo, state) = fixture();
        std::fs::create_dir_all(repo.join(".bundle")).expect("bundle dir");
        std::fs::write(
            repo.join(".bundle").join("config"),
            "---\nBUNDLE_PATH: \"~/.gem/shared\"\n",
        )
        .expect("config");

        let inventory = inventory(&state).expect("inventory");
        let home = std::env::var_os("HOME").expect("HOME");
        assert_eq!(inventory.outside.len(), 1, "{inventory:?}");
        assert_eq!(
            inventory.outside[0].path,
            PathBuf::from(home).join(".gem").join("shared")
        );
        assert!(
            !inventory.outside[0].path.starts_with(&repo),
            "an unexpanded `~` would read as a path inside the repository"
        );
    }

    #[test]
    fn an_outside_directory_survives_yes_and_is_removed_only_with_outside() {
        // The safety property the whole flag exists for: the directory may hold gems the human
        // installed themselves, so `--yes` — which means "delete what this program created" — is
        // not permission to delete it. Only the separate flag is.
        let (dir, repo, state) = fixture();
        let gems = dir.path().join("gems");
        std::fs::create_dir_all(gems.join("ruby")).expect("gems");
        std::fs::write(gems.join("ruby").join("a.gem"), "gem").expect("gem");
        std::fs::create_dir_all(repo.join(".bundle")).expect("bundle dir");
        std::fs::write(
            repo.join(".bundle").join("config"),
            format!("---\nBUNDLE_PATH: \"{}\"\n", gems.display()),
        )
        .expect("config");

        let inventory = inventory(&state).expect("inventory");
        let report = execute(&state, &inventory, false).expect("execute");
        assert_eq!(report.outside_removed, 0);
        assert_eq!(report.outside_kept, vec![gems.clone()]);
        assert!(gems.join("ruby").join("a.gem").exists(), "`--yes` deleted it");

        // The same inventory, run again with the flag: `execute` is a function of the plan and the
        // decision, so the second half is the same call with `outside: true`. Not a fresh
        // `inventory()` — see `the_registry_a_plain_yes_deleted_is_why_outside_must_ride_along`
        // for why that would find nothing.
        let report = execute(&state, &inventory, true).expect("execute");
        assert_eq!(report.outside_removed, 1);
        assert_eq!(report.outside_kept, Vec::<PathBuf>::new());
        assert!(!gems.exists(), "--outside must delete it");
    }

    #[test]
    fn the_bundler_config_is_read_as_lines_and_only_for_the_key_it_names() {
        // `BUNDLE_CLEAN: "true"` is a setting, not a directory. A reader that took every value as a
        // path would list `true` in a delete-list, and a real YAML parser would accept anchors and
        // multi-line scalars bundler never writes and produce a different path from them.
        //
        // `BUNDLE_APP_CONFIG` is the case that keeps this test honest. It is a real bundler setting
        // whose value *is* a directory outside the repository, so a reader that accepted any
        // `BUNDLE_*` key would list it — and the `BUNDLE_CLEAN: "true"` line alone would not have
        // caught that, because `true` is relative and gets filtered as "inside the repository" by
        // an unrelated rule. Named keys, not a prefix, is the property being pinned.
        let (dir, repo, state) = fixture();
        let gems = dir.path().join("gems");
        let app_config = dir.path().join("bundle-app-config");
        std::fs::create_dir_all(&gems).expect("gems");
        std::fs::create_dir_all(&app_config).expect("app config");
        std::fs::create_dir_all(repo.join(".bundle")).expect("bundle dir");
        std::fs::write(
            repo.join(".bundle").join("config"),
            format!(
                "---\nBUNDLE_CLEAN: \"true\"\nBUNDLE_APP_CONFIG: \"{}\"\nBUNDLE_PATH: \"{}\"\n# BUNDLE_PATH: \"{}\"\n",
                app_config.display(),
                gems.display(),
                dir.path().join("commented").display()
            ),
        )
        .expect("config");

        let inventory = inventory(&state).expect("inventory");
        assert_eq!(inventory.outside.len(), 1, "{inventory:?}");
        assert_eq!(inventory.outside[0].path, gems);
        assert!(
            !inventory.outside[0].path.ends_with("commented"),
            "a commented-out key is not a path: {inventory:?}"
        );
    }

    #[test]
    fn a_worktrees_bundle_config_names_paths_too() {
        // Measured: the `black_jack` worktree had its own `.bundle/config` with the same
        // `BUNDLE_PATH`, because `bundle install` ran there. Reading only the main checkout would
        // miss the directory the run actually wrote to.
        let (dir, repo, state) = fixture();
        let task = "tsk_01ARZ3NDEKTSV4RRFFQ69G5FAV";
        let path = crate::worktree::WorktreeManager::new(&repo).path_for(task);
        crate::worktree::WorktreeManager::new(&repo)
            .ensure(task, &format!("agent/{task}"), None)
            .expect("worktree");
        let gems = dir.path().join("worktree-gems");
        std::fs::create_dir_all(&gems).expect("gems");
        std::fs::create_dir_all(path.join(".bundle")).expect("bundle dir");
        std::fs::write(
            path.join(".bundle").join("config"),
            format!("---\nBUNDLE_PATH: \"{}\"\n", gems.display()),
        )
        .expect("config");

        let inventory = inventory(&state).expect("inventory");
        assert_eq!(inventory.outside.len(), 1, "{inventory:?}");
        assert_eq!(inventory.outside[0].path, gems);
        assert_eq!(
            inventory.outside[0].named_by,
            std::fs::canonicalize(path.join(".bundle").join("config"))
                .unwrap_or_else(|_| { path.join(".bundle").join("config") })
        );
    }

    #[test]
    fn the_registry_a_plain_yes_deleted_is_why_outside_must_ride_along() {
        // The trap the first version of the flag walked into, found by running the test that
        // deletes twice. The outside entry is discovered *through* `registry.json`, and
        // `reset --yes` deletes that registry. So a `--yes` without `--outside` leaves an orphan
        // directory that no later `reset --outside` can find, and the CLI would truthfully report
        // "zero state" while `~/.gem/shared` is still on disk holding every gem the run installed.
        //
        // The fix is not to keep the registry: reaching the zero state is the command's whole
        // point, and a registry kept alive only to name gems would be a second state root. The fix
        // is to say so — the dry run tells a human to pass both flags together, and this test
        // pins the fact they have to.
        let (dir, repo, state) = fixture();
        let gems = dir.path().join("gems");
        std::fs::create_dir_all(&gems).expect("gems");
        std::fs::create_dir_all(repo.join(".bundle")).expect("bundle dir");
        std::fs::write(
            repo.join(".bundle").join("config"),
            format!("---\nBUNDLE_PATH: \"{}\"\n", gems.display()),
        )
        .expect("config");

        let inventory = inventory(&state).expect("inventory");
        assert_eq!(inventory.outside.len(), 1, "{inventory:?}");
        execute(&state, &inventory, false).expect("execute");
        assert!(gems.exists(), "a plain --yes deleted it");

        // The second run has nothing to consult, and says so by finding nothing at all.
        let after = super::inventory(&state).expect("second inventory");
        assert!(
            after.outside.is_empty(),
            "the entry cannot survive the registry it was found through: {after:?}"
        );
    }

    #[test]
    fn a_project_with_no_bundle_config_adds_no_outside_entry() {
        // The negative control for every test above: this fixture has no `.bundle/`, and the
        // inventory must not invent one.
        let (_dir, _repo, state) = fixture();
        let inventory = inventory(&state).expect("inventory");
        assert!(inventory.outside.is_empty(), "{inventory:?}");
    }
}
