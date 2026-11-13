//! One git worktree per task, and the rules for creating, finding and rebuilding it.
//!
//! # Why a worktree rather than the user's checkout
//!
//! The agent runs `git add --all`, `reset --hard` and `clean -fd`. In the directory a human is
//! editing that is not a workflow, it is a hostile surprise — and it makes an accidental
//! `reset` unrecoverable. A worktree is a second checkout of the same repository with its own
//! `HEAD`, so the agent can wreck it as much as it likes without touching the human's branch.
//!
//! # Why the path is derived, not stored
//!
//! `<repo>/../<repo-name>-agent/<task>` is a pure function of the repository and the task id.
//! Recovery runs after a crash with nothing in memory; a stored path would be one more thing
//! that can disagree with reality, and a path that has drifted is a worktree the daemon cannot
//! find. Deriving it means "where is task X's worktree?" always has the same answer.
//!
//! Sibling rather than inside the repository on purpose: a worktree inside a worktree confuses
//! every tool that walks upwards looking for `.git`, including `cargo` and `git` itself.
//!
//! # Why an occupied directory blocks instead of being replaced
//!
//! D-035. If the path exists and is not a worktree of this repository — a directory someone
//! put there, a half-deleted checkout, a worktree left by an older layout — recreating over it
//! destroys files that recovery cannot identify or return. A missing path is rebuilt; an
//! occupied one is a human's decision.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::checkpoint::{GitError, GitRepo};
use crate::recover::WorkspaceFacts;

/// Suffix appended to the repository's own name.
const WORKTREE_SUFFIX: &str = "-agent";

/// Why a worktree could not be produced.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorktreeError {
    /// Git refused, with its own message.
    #[error("{0}")]
    Git(#[from] GitError),
    /// The path exists and is not a worktree of this repository.
    #[error("`{path}` already exists and is not a worktree of this repository: {reason}")]
    Occupied { path: String, reason: String },
    /// The task branch does not exist and there is nothing to base it on.
    #[error("task branch `{branch}` does not exist and no base commit was given")]
    NoBase { branch: String },
}

impl WorktreeError {
    /// Whether a human has to look before anything else happens.
    ///
    /// The distinction matters for recovery: a git failure is often transient and retryable, an
    /// occupied directory never is. The daemon blocks the task in the second case and escalates.
    #[must_use]
    pub const fn needs_human(&self) -> bool {
        matches!(self, Self::Occupied { .. } | Self::NoBase { .. })
    }
}

/// One task worktree, as found on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskWorktree {
    /// Directory the agent works in.
    pub path: PathBuf,
    /// Branch checked out, `None` for a detached `HEAD`.
    pub branch: Option<String>,
    /// `HEAD` SHA, `None` for a repository with no commits.
    pub head: Option<String>,
    /// Whether the working tree has uncommitted changes.
    pub dirty: bool,
}

impl TaskWorktree {
    /// A [`GitRepo`] over this worktree, which is what every git tool takes.
    #[must_use]
    pub fn repo(&self) -> GitRepo {
        GitRepo::new(self.path.clone())
    }

    /// The facts recovery needs, gathered from the worktree itself.
    #[must_use]
    pub fn facts(&self) -> WorkspaceFacts {
        WorkspaceFacts {
            exists: true,
            is_repository: true,
            branch: self.branch.clone(),
            head: self.head.clone(),
            dirty: self.dirty,
        }
    }
}

/// Creates and finds per-task worktrees of one repository.
///
/// Holds the repository's *main* checkout, which is where `git worktree add` has to run: git
/// records the new worktree in `<repo>/.git/worktrees`, and a command run from inside another
/// worktree would still find it, but naming the main checkout makes the intent visible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorktreeManager {
    repository: PathBuf,
}

impl WorktreeManager {
    /// Wrap a repository's main checkout.
    ///
    /// Does not verify it: [`Self::ensure`] and [`Self::find`] report what they find, and a
    /// constructor that can fail makes every caller handle an error it cannot act on.
    #[must_use]
    pub fn new(repository: impl Into<PathBuf>) -> Self {
        Self {
            repository: repository.into(),
        }
    }

    /// The repository this manager belongs to.
    #[must_use]
    pub fn repository(&self) -> &Path {
        &self.repository
    }

    /// The directory holding every task worktree of this repository.
    ///
    /// `<parent>/<name>-agent`: a sibling, so it is inside the same `workspace_root` (which is
    /// what the policy gate confines reads to) and cannot be mistaken for part of the project.
    #[must_use]
    pub fn base_dir(&self) -> PathBuf {
        let name = self.repository.file_name().map_or_else(
            || "repository".to_string(),
            |name| name.to_string_lossy().into_owned(),
        );
        // A repository at the filesystem root has no parent; the worktree has to go somewhere,
        // and a relative path would depend on the daemon's cwd.
        self.repository.parent().map_or_else(
            || PathBuf::from(format!("/{name}{WORKTREE_SUFFIX}")),
            |parent| parent.join(format!("{name}{WORKTREE_SUFFIX}")),
        )
    }

    /// Where the worktree for `task` lives, whether or not it exists.
    #[must_use]
    pub fn path_for(&self, task: &str) -> PathBuf {
        self.base_dir().join(task)
    }

    /// Find an existing worktree for `task`.
    ///
    /// `None` means the directory is absent — the normal state before the first attempt, and the
    /// state that triggers a rebuild. A directory that exists but is not a worktree is an
    /// [`WorktreeError::Occupied`], not `None`, because those two need opposite reactions.
    ///
    /// # Errors
    /// The path is occupied by something that is not this repository's worktree, or git failed
    /// while asking.
    pub fn find(&self, task: &str) -> Result<Option<TaskWorktree>, WorktreeError> {
        let path = self.path_for(task);
        if !path.exists() {
            return Ok(None);
        }
        // `git -C <path> rev-parse --git-common-dir` answers for *any* repository, including one
        // that merely happens to live there. Comparing the common dir against ours is what
        // distinguishes "our worktree" from "somebody's checkout at our path".
        let Ok(common) = Self::git_in(&path, &["rev-parse", "--git-common-dir"]) else {
            return Err(WorktreeError::Occupied {
                path: path.display().to_string(),
                reason: "the directory is not a git work tree".to_string(),
            });
        };
        let common = resolve_relative(&path, common.trim());
        let ours = Self::git_in(&self.repository, &["rev-parse", "--git-common-dir"])?;
        let ours = resolve_relative(&self.repository, ours.trim());
        if common != ours {
            return Err(WorktreeError::Occupied {
                path: path.display().to_string(),
                reason: format!("it belongs to the repository at `{}`", common.display()),
            });
        }
        // Registered in `.git/worktrees` is a separate question from "is a repository": a
        // directory copied from elsewhere has a `.git` file pointing at a repository nobody
        // knows about. Git itself is the authority.
        if !self.registered(&path) {
            return Err(WorktreeError::Occupied {
                path: path.display().to_string(),
                reason: "it is a checkout of this repository but not a registered worktree".to_string(),
            });
        }
        Ok(Some(inspect(Self::git_in, &path)))
    }

    /// Produce a worktree for `task`, creating or keeping it as the situation requires.
    ///
    /// * absent → created from `branch` if it exists, else from `base_ref`, else from `HEAD`;
    /// * present and healthy → returned untouched, with its dirty state reported;
    /// * present, on the wrong branch → the branch is checked out, provided the tree is clean;
    /// * present and not ours → [`WorktreeError::Occupied`], never replaced.
    ///
    /// `base_ref` is the last checkpoint, which is the only known-good commit when the branch
    /// does not exist yet and `HEAD` has moved on since. Passing it rather than letting the
    /// manager default to `HEAD` is what makes D-035's "rebuild from the checkpoint" true.
    ///
    /// # Errors
    /// See [`WorktreeError`]. A dirty worktree on the wrong branch is a git error rather than a
    /// forced checkout: `checkout --force` would discard work the agent has not committed, and
    /// that work is exactly what a human wants to see after a restart.
    pub fn ensure(
        &self,
        task: &str,
        branch: &str,
        base_ref: Option<&str>,
    ) -> Result<TaskWorktree, WorktreeError> {
        if let Some(existing) = self.find(task)? {
            return self.put_on_branch(existing, branch);
        }
        let path = self.path_for(task);
        // A worktree whose *directory* vanished leaves its registration behind in
        // `.git/worktrees/<name>`, and `git worktree add` then refuses the path as "a missing but
        // already registered worktree". That is the normal shape of the crash this function
        // repairs — the task directory was removed, or the whole agent directory was — so the
        // stale record is cleared before the add. `prune` only drops registrations whose gitdir is
        // gone; a live worktree of this repository is untouched.
        Self::git_in(&self.repository, &["worktree", "prune"])?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| WorktreeError::Occupied {
                path: parent.display().to_string(),
                reason: format!("the parent directory could not be created: {error}"),
            })?;
        }
        let path_text = path.display().to_string();
        // `base` is owned here rather than inside the branch below, because `args` borrows it and
        // an argument list that outlives its source is exactly the temporary-lifetime bug that
        // reads as a borrow error three lines away from the cause.
        let base = if self.branch_exists(branch) {
            None
        } else {
            let base = base_ref
                .map(str::to_string)
                .or_else(|| Self::head_sha(&self.repository));
            let Some(base) = base else {
                return Err(WorktreeError::NoBase {
                    branch: branch.to_string(),
                });
            };
            Some(base)
        };

        // Everything `git worktree add` needs is in this one argument list, so a half-created
        // worktree is not a state we have to reason about: git either records it or it does not.
        let mut args: Vec<&str> = vec!["worktree", "add", &path_text];
        match &base {
            Some(base) => {
                // `-b` fails if the branch appeared between the check and the call, which is the
                // race worth failing on: two daemons must not share one task worktree.
                args.extend(["-b", branch, base]);
            }
            None => args.push(branch),
        }
        Self::git_in(&self.repository, &args)?;
        self.find(task)?.ok_or_else(|| {
            WorktreeError::Git(GitError {
                command: format!("worktree add {path_text}"),
                message: "git reported success but the worktree is not present".to_string(),
            })
        })
    }

    /// Remove a task's worktree and prune the record.
    ///
    /// `force` discards uncommitted work. Callers pass it only after a checkpoint of that work
    /// exists — the argument exists so the decision is visible at the call site, not buried in
    /// this function.
    ///
    /// # Errors
    /// Git refused, which for a worktree with modifications and `force = false` is the point.
    pub fn remove(&self, task: &str, force: bool) -> Result<(), WorktreeError> {
        let path = self.path_for(task);
        if !path.exists() {
            // Pruning is still worth doing: a worktree whose directory vanished is exactly the
            // record `git worktree prune` exists to clear, and leaving it makes the next
            // `worktree add` fail for a reason nobody will understand.
            Self::git_in(&self.repository, &["worktree", "prune"])?;
            return Ok(());
        }
        let path_text = path.display().to_string();
        if force {
            Self::git_in(&self.repository, &["worktree", "remove", "--force", &path_text])?;
        } else {
            Self::git_in(&self.repository, &["worktree", "remove", &path_text])?;
        }
        Self::git_in(&self.repository, &["worktree", "prune"])?;
        Ok(())
    }

    /// Every registered worktree, main checkout included.
    ///
    /// # Errors
    /// Git refused.
    pub fn list(&self) -> Result<Vec<TaskWorktree>, WorktreeError> {
        let out = Self::git_in(&self.repository, &["worktree", "list", "--porcelain"])?;
        let mut out_worktrees = Vec::new();
        let mut path: Option<PathBuf> = None;
        for line in out.lines() {
            if let Some(rest) = line.strip_prefix("worktree ") {
                path = Some(PathBuf::from(rest));
            } else if line.starts_with("branch ") && path.is_some() {
                // The porcelain format writes `branch refs/heads/<name>`; a detached worktree
                // writes `detached` instead and no branch line at all.
                let branch = line
                    .trim_start_matches("branch ")
                    .trim_start_matches("refs/heads/")
                    .to_string();
                let path = path.take().unwrap_or_default();
                out_worktrees.push(TaskWorktree {
                    branch: Some(branch),
                    head: Self::head_sha(&path),
                    dirty: Self::is_dirty(&path),
                    path,
                });
            } else if line.trim() == "detached" {
                let path = path.take().unwrap_or_default();
                out_worktrees.push(TaskWorktree {
                    branch: None,
                    head: Self::head_sha(&path),
                    dirty: Self::is_dirty(&path),
                    path,
                });
            }
        }
        // A bare `worktree` line with no branch line is the main checkout of an empty repository;
        // reporting it keeps `list` a complete answer instead of a surprising one.
        if let Some(path) = path {
            out_worktrees.push(TaskWorktree {
                branch: None,
                head: Self::head_sha(&path),
                dirty: Self::is_dirty(&path),
                path,
            });
        }
        Ok(out_worktrees)
    }

    /// Whether a branch already exists in the repository.
    #[must_use]
    pub fn branch_exists(&self, branch: &str) -> bool {
        Self::git_in(&self.repository, &["rev-parse", "--verify", branch]).is_ok()
    }

    /// Move an existing worktree onto `branch`, refusing when that would discard work.
    fn put_on_branch(&self, worktree: TaskWorktree, branch: &str) -> Result<TaskWorktree, WorktreeError> {
        if worktree.branch.as_deref() == Some(branch) {
            return Ok(worktree);
        }
        if worktree.dirty {
            // `checkout --force` here would delete uncommitted work. Recovery's job is to keep
            // what exists until a checkpoint has it, so this is a refusal rather than a repair.
            return Err(WorktreeError::Occupied {
                path: worktree.path.display().to_string(),
                reason: format!(
                    "it is on `{}` with uncommitted changes; moving it to `{branch}` would \
                     discard them",
                    worktree.branch.unwrap_or_else(|| "a detached HEAD".to_string())
                ),
            });
        }
        let path_text = worktree.path.display().to_string();
        Self::git_in(&worktree.path, &["checkout", "--force", branch]).map_err(|error| {
            WorktreeError::Git(GitError {
                command: format!("checkout {branch} in {path_text}"),
                message: error.message,
            })
        })?;
        self.find(
            Path::new(&path_text)
                .file_name()
                .map_or("", |name| name.to_str().unwrap_or_default()),
        )?
        .ok_or_else(|| {
            WorktreeError::Git(GitError {
                command: format!("checkout {branch}"),
                message: "the worktree disappeared after checkout".to_string(),
            })
        })
    }

    /// Whether git has this path recorded in `.git/worktrees`.
    fn registered(&self, path: &Path) -> bool {
        let Ok(list) = Self::git_in(&self.repository, &["worktree", "list", "--porcelain"]) else {
            // `git worktree list` failed; the caller already knows the repository is odd, and a
            // second failure mode would only obscure the first.
            return false;
        };
        let wanted = canonical(path);
        list.lines()
            .filter_map(|line| line.strip_prefix("worktree "))
            .any(|listed| canonical(Path::new(listed)) == wanted)
    }

    fn head_sha(at: &Path) -> Option<String> {
        Self::git_in(at, &["rev-parse", "--verify", "HEAD"])
            .ok()
            .map(|sha| sha.trim().to_string())
            .filter(|sha| !sha.is_empty())
    }

    fn is_dirty(at: &Path) -> bool {
        Self::git_in(at, &["status", "--porcelain", "--untracked-files=all"])
            .is_ok_and(|out| !out.trim().is_empty())
    }

    fn git_in(at: &Path, args: &[&str]) -> Result<String, GitError> {
        let output = Command::new("git")
            .arg("-C")
            .arg(at)
            .args(args)
            // The same guards as `GitRepo::git`, for the same reason: a credential prompt or a
            // pager would hang the daemon, and `LC_ALL` keeps message parsing stable.
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_PAGER", "cat")
            .env("LC_ALL", "C")
            .output()
            .map_err(|error| GitError {
                command: args.join(" "),
                message: error.to_string(),
            })?;
        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).into_owned())
        } else {
            Err(GitError {
                command: args.join(" "),
                message: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            })
        }
    }
}

/// Build a [`TaskWorktree`] by asking git about `path` directly.
///
/// Every field comes from the worktree itself rather than from a caller's belief about it: a
/// `branch` passed in would be the branch the caller *wanted*, and the whole point of `find` is
/// to report what is actually checked out.
fn inspect(git: impl Fn(&Path, &[&str]) -> Result<String, GitError>, path: &Path) -> TaskWorktree {
    TaskWorktree {
        path: path.to_path_buf(),
        branch: git(path, &["rev-parse", "--abbrev-ref", "HEAD"])
            .ok()
            .map(|text| text.trim().to_string())
            // `HEAD` is what git prints for a detached checkout; callers that care about the
            // difference should not have to string-compare it.
            .filter(|text| text != "HEAD"),
        head: git(path, &["rev-parse", "--verify", "HEAD"])
            .ok()
            .map(|sha| sha.trim().to_string())
            .filter(|sha| !sha.is_empty()),
        dirty: git(path, &["status", "--porcelain", "--untracked-files=all"])
            .is_ok_and(|out| !out.trim().is_empty()),
    }
}

/// Resolve a `--git-common-dir` answer, which git prints relative to the queried directory.
fn resolve_relative(base: &Path, reported: &str) -> PathBuf {
    let path = Path::new(reported);
    if path.is_absolute() {
        canonical(path)
    } else {
        canonical(&base.join(path))
    }
}

/// Best-effort canonicalisation, because the directories may be gone and a comparison against a
/// missing path must still not panic.
fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A repository plus a manager, in a tempdir whose parent is the test's own.
    struct Fixture {
        dir: tempfile::TempDir,
        manager: WorktreeManager,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let repo = dir.path().join("project");
            std::fs::create_dir_all(&repo).unwrap();
            run(&repo, &["init", "-q", "-b", "main"]);
            run(&repo, &["config", "user.name", "Test"]);
            run(&repo, &["config", "user.email", "test@example.com"]);
            std::fs::write(repo.join("README.md"), "# project\n").unwrap();
            run(&repo, &["add", "--all"]);
            run(&repo, &["commit", "-q", "-m", "chore: initial"]);
            let manager = WorktreeManager::new(&repo);
            Self { dir, manager }
        }

        fn repo_path(&self) -> PathBuf {
            self.dir.path().join("project")
        }
    }

    fn run(dir: &Path, args: &[&str]) {
        let output = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("LC_ALL", "C")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn the_worktree_path_is_a_sibling_of_the_repository() {
        let fixture = Fixture::new();
        assert_eq!(
            fixture.manager.base_dir(),
            fixture.dir.path().join("project-agent")
        );
        assert_eq!(
            fixture.manager.path_for("tsk_01"),
            fixture.dir.path().join("project-agent").join("tsk_01")
        );
    }

    #[test]
    fn a_missing_worktree_is_created_from_the_branch_base() {
        let fixture = Fixture::new();
        let worktree = fixture.manager.ensure("tsk_01", "agent/tsk_01", None).unwrap();

        assert_eq!(worktree.branch.as_deref(), Some("agent/tsk_01"));
        assert_eq!(worktree.path, fixture.manager.path_for("tsk_01"));
        assert!(
            worktree.path.join("README.md").exists(),
            "the tree is checked out"
        );
        assert!(!worktree.dirty);
        assert!(worktree.head.is_some());
    }

    #[test]
    fn a_worktree_whose_directory_vanished_is_recreated() {
        // The exact shape a crash leaves: the directory is gone but git still has the
        // registration, and a plain `worktree add` fails with "a missing but already registered
        // worktree". Recovery's whole job is to repair this, so `ensure` has to clear the record.
        let fixture = Fixture::new();
        let first = fixture.manager.ensure("tsk_01", "agent/tsk_01", None).unwrap();
        std::fs::remove_dir_all(&first.path).unwrap();
        assert!(!first.path.exists());

        let recreated = fixture
            .manager
            .ensure("tsk_01", "agent/tsk_01", None)
            .expect("a lost worktree must be recreated");
        assert!(recreated.path.join("README.md").exists());
        assert_eq!(recreated.branch.as_deref(), Some("agent/tsk_01"));
    }

    #[test]
    fn creating_twice_returns_the_same_worktree_and_keeps_its_work() {
        // The property that makes `ensure` safe to call at every task start: it must not reset a
        // worktree that already has an attempt's edits in it.
        let fixture = Fixture::new();
        let first = fixture.manager.ensure("tsk_01", "agent/tsk_01", None).unwrap();
        std::fs::write(first.path.join("work.txt"), "attempt one").unwrap();

        let second = fixture.manager.ensure("tsk_01", "agent/tsk_01", None).unwrap();
        assert_eq!(second.path, first.path);
        assert_eq!(
            std::fs::read_to_string(second.path.join("work.txt")).unwrap(),
            "attempt one"
        );
        assert!(second.dirty, "uncommitted work is reported, not discarded");
    }

    #[test]
    fn a_missing_worktree_with_an_existing_branch_starts_from_that_branch() {
        let fixture = Fixture::new();
        // A branch that exists and has a commit the base does not: the rebuilt worktree must
        // contain it, or every restart silently reverts a subtask.
        run(&fixture.repo_path(), &["checkout", "-q", "-b", "agent/tsk_02"]);
        std::fs::write(fixture.repo_path().join("feature.rs"), "fn f() {}\n").unwrap();
        run(&fixture.repo_path(), &["add", "--all"]);
        run(&fixture.repo_path(), &["commit", "-q", "-m", "feat: work"]);
        run(&fixture.repo_path(), &["checkout", "-q", "main"]);

        let worktree = fixture.manager.ensure("tsk_02", "agent/tsk_02", None).unwrap();
        assert!(worktree.path.join("feature.rs").exists(), "{worktree:?}");
    }

    #[test]
    fn a_missing_worktree_can_be_built_from_a_checkpoint_sha() {
        // D-035's second half: when the branch was never created, the last checkpoint is the only
        // known-good commit, and starting from `HEAD` would replay work the checkpoint undid.
        let fixture = Fixture::new();
        let repo = GitRepo::new(fixture.repo_path());
        std::fs::write(fixture.repo_path().join("keep.txt"), "checkpointed\n").unwrap();
        let checkpoint = repo
            .hidden_checkpoint("agent/tsk_03", "before risky")
            .unwrap()
            .unwrap();
        // Move the tree on, so `HEAD` no longer describes the checkpoint's state. The `reset`
        // is not decoration: `hidden_checkpoint` snapshots through `git add --all` and leaves the
        // index populated, so committing straight afterwards would put `keep.txt` back into
        // `HEAD` and the test would pass without ever proving `base_ref` was used.
        run(&fixture.repo_path(), &["reset", "--hard", "-q", "HEAD"]);
        std::fs::write(fixture.repo_path().join("later.txt"), "moved on\n").unwrap();
        run(&fixture.repo_path(), &["add", "--all"]);
        run(&fixture.repo_path(), &["commit", "-q", "-m", "chore: move on"]);

        let worktree = fixture
            .manager
            .ensure("tsk_03", "agent/tsk_03", Some(&checkpoint.sha))
            .unwrap();
        assert_eq!(worktree.head.as_deref(), Some(checkpoint.sha.as_str()));
        assert!(
            worktree.path.join("keep.txt").exists(),
            "the checkpoint's tree must be what is checked out"
        );
    }

    #[test]
    fn an_occupied_directory_blocks_and_is_left_alone() {
        // D-035: never recreate over files nobody can identify afterwards.
        let fixture = Fixture::new();
        let path = fixture.manager.path_for("tsk_04");
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("somebody-elses.txt"), "do not delete me").unwrap();

        let error = fixture
            .manager
            .ensure("tsk_04", "agent/tsk_04", None)
            .unwrap_err();
        assert!(matches!(error, WorktreeError::Occupied { .. }), "{error}");
        assert!(error.needs_human());
        assert_eq!(
            std::fs::read_to_string(path.join("somebody-elses.txt")).unwrap(),
            "do not delete me"
        );
    }

    #[test]
    fn a_worktree_of_another_repository_at_our_path_blocks() {
        let fixture = Fixture::new();
        let path = fixture.manager.path_for("tsk_05");
        std::fs::create_dir_all(&path).unwrap();
        run(&path, &["init", "-q", "-b", "main"]);

        let error = fixture
            .manager
            .find("tsk_05")
            .expect_err("a foreign repository is not our worktree");
        assert!(matches!(error, WorktreeError::Occupied { .. }), "{error}");
    }

    #[test]
    fn a_detached_moving_worktree_is_put_back_on_its_branch() {
        let fixture = Fixture::new();
        let worktree = fixture.manager.ensure("tsk_06", "agent/tsk_06", None).unwrap();
        run(&worktree.path, &["checkout", "-q", "--detach"]);

        let repaired = fixture.manager.ensure("tsk_06", "agent/tsk_06", None).unwrap();
        assert_eq!(repaired.branch.as_deref(), Some("agent/tsk_06"));
    }

    #[test]
    fn a_dirty_worktree_on_the_wrong_branch_is_refused_rather_than_forced() {
        // `checkout --force` would delete work the agent has not committed. That work is what a
        // human needs after a restart, so the manager refuses and lets recovery decide.
        let fixture = Fixture::new();
        let worktree = fixture.manager.ensure("tsk_07", "agent/tsk_07", None).unwrap();
        std::fs::write(worktree.path.join("uncommitted.txt"), "precious").unwrap();

        let error = fixture.manager.ensure("tsk_07", "agent/other", None).unwrap_err();
        assert!(matches!(error, WorktreeError::Occupied { .. }), "{error}");
        assert_eq!(
            std::fs::read_to_string(worktree.path.join("uncommitted.txt")).unwrap(),
            "precious"
        );
    }

    #[test]
    fn removing_a_worktree_leaves_the_repository_intact() {
        let fixture = Fixture::new();
        let worktree = fixture.manager.ensure("tsk_08", "agent/tsk_08", None).unwrap();
        let path = worktree.path;

        fixture.manager.remove("tsk_08", true).unwrap();
        assert!(!path.exists());
        assert!(fixture.repo_path().join("README.md").exists());
        assert!(
            fixture.manager.find("tsk_08").unwrap().is_none(),
            "a removed worktree must not be reported as present"
        );
    }

    #[test]
    fn removing_an_absent_worktree_prunes_and_succeeds() {
        // The directory of a real worktree can vanish under the daemon; `remove` must clean up
        // git's record of it rather than failing on a path that is already gone.
        let fixture = Fixture::new();
        let worktree = fixture.manager.ensure("tsk_09", "agent/tsk_09", None).unwrap();
        std::fs::remove_dir_all(&worktree.path).unwrap();

        fixture.manager.remove("tsk_09", false).unwrap();
        assert!(fixture.manager.find("tsk_09").unwrap().is_none());
    }

    #[test]
    fn list_includes_the_main_checkout_and_its_worktrees() {
        let fixture = Fixture::new();
        fixture.manager.ensure("tsk_10", "agent/tsk_10", None).unwrap();
        let all = fixture.manager.list().unwrap();
        let paths: Vec<_> = all.iter().map(|worktree| worktree.path.clone()).collect();
        assert!(paths.contains(&canonical(&fixture.repo_path())), "{paths:?}");
        assert!(
            paths.contains(&canonical(&fixture.manager.path_for("tsk_10"))),
            "git prints canonical paths, so the comparison has to canonise too: {paths:?}"
        );
    }

    #[test]
    fn the_facts_of_a_healthy_worktree_say_so() {
        let fixture = Fixture::new();
        let worktree = fixture.manager.ensure("tsk_11", "agent/tsk_11", None).unwrap();
        let facts = worktree.facts();
        assert!(facts.exists);
        assert!(facts.is_repository);
        assert_eq!(facts.branch.as_deref(), Some("agent/tsk_11"));
        assert!(!facts.dirty);
        assert!(facts.head.is_some());

        std::fs::write(worktree.path.join("new.txt"), "x").unwrap();
        let dirty = fixture
            .manager
            .find("tsk_11")
            .unwrap()
            .expect("still there")
            .facts();
        assert!(dirty.dirty);
    }

    #[test]
    fn no_base_and_no_head_is_reported_as_needing_a_human() {
        // An empty repository with no commits: `git worktree add -b branch <sha>` has nothing to
        // point at, and inventing one would produce a worktree on no branch.
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("empty");
        std::fs::create_dir_all(&repo).unwrap();
        run(&repo, &["init", "-q", "-b", "main"]);
        let manager = WorktreeManager::new(&repo);

        let error = manager.ensure("tsk_12", "agent/tsk_12", None).unwrap_err();
        assert!(matches!(error, WorktreeError::NoBase { .. }), "{error}");
        assert!(error.needs_human());
    }
}
