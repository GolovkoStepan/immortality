//! The production [`Checkpoints`]: a real worktree and a real `git`.
//!
//! # Why the snapshot is journalled by the caller
//!
//! [`GitRepo::hidden_checkpoint`] is reached through [`Checkpoints::snapshot`], and the runner
//! writes `checkpoint_created` for the result *before* it asks git to reset. That ordering is the
//! only one in which a crash between the two leaves something a human can act on: the branch
//! already points at the discarded tree, so a journal that names it is enough to find it again.
//! Reversing the order would leave a reset tree and a journal that never said a snapshot existed.
//!
//! # Why `restore_tree` and not `rollback_to`
//!
//! `GitRepo::rollback_to` takes its own safety snapshot. The runner takes one first — because it
//! also has to journal it — so [`Checkpoints::restore`] uses the half that only moves the tree
//! (`restore_tree`), and the snapshot is taken once, by the caller.

use std::path::{Path, PathBuf};

use immortality_core::checkpoint::GitRepo;
use immortality_core::restart::{Checkpoints, Snapshot};

/// A checkpoint world backed by one worktree.
#[derive(Debug, Clone)]
pub struct GitCheckpoints {
    repo: GitRepo,
    /// The branch snapshots are written to, `agent/<task>`.
    branch: String,
    /// The branch the worktree should be on after a rollback, when the caller knows it.
    ///
    /// `None` leaves the worktree on whatever branch it is on: a repository that has no task
    /// branch yet has no restore step, and forcing a checkout of a branch that does not exist
    /// would turn a working attempt into a failed restart.
    restore: Option<String>,
}

impl GitCheckpoints {
    /// Checkpoints for `worktree`, snapshots on `branch`.
    #[must_use]
    pub fn new(worktree: impl Into<PathBuf>, branch: impl Into<String>) -> Self {
        Self {
            repo: GitRepo::new(worktree),
            branch: branch.into(),
            restore: None,
        }
    }

    /// Also check `branch` out again after a rollback.
    ///
    /// Needed when the model may have left the worktree detached or on another branch: `reset
    /// --hard` moves whatever `HEAD` points at, so a rollback without a checkout can restore the
    /// right tree onto the wrong branch.
    #[must_use]
    pub fn restoring(mut self, branch: impl Into<String>) -> Self {
        self.restore = Some(branch.into());
        self
    }

    /// The worktree these checkpoints belong to.
    #[must_use]
    pub fn worktree(&self) -> &Path {
        self.repo.worktree()
    }

    /// The repository, for a caller that wants to journal what the checkpoint was against.
    #[must_use]
    pub const fn repo(&self) -> &GitRepo {
        &self.repo
    }
}

impl Checkpoints for GitCheckpoints {
    fn snapshot(&self, reason: &str) -> Result<Option<Snapshot>, String> {
        self.repo
            .hidden_checkpoint(&self.branch, reason)
            .map(|checkpoint| {
                checkpoint.map(|checkpoint| Snapshot {
                    sha: checkpoint.sha,
                    branch: checkpoint.branch,
                })
            })
            .map_err(|error| error.to_string())
    }

    fn restore(&self, sha: &str) -> Result<(), String> {
        // Validated before anything moves. Discovering a bad SHA after the checkout would leave
        // the worktree on the wrong branch with nothing to show for it.
        if !self.repo.has_commit(sha) {
            return Err(format!(
                "checkpoint commit {sha} does not exist, so there is nothing to roll back to"
            ));
        }
        // The checkout is forced and comes before the read: a worktree left detached by a
        // previous attempt would otherwise have its tree restored onto a detached `HEAD`, and the
        // next commit would land nowhere.
        //
        // A branch that does not exist is created at the snapshot's **parent**, not at the
        // snapshot itself: the snapshot is the hidden commit named `checkpoint`, and making it the
        // first commit of `agent/<task>` would put that name at the root of the history a human
        // merges. Its parent is the tip the snapshot was taken from, and the tree on disk comes
        // from `restore_tree` either way.
        if let Some(branch) = &self.restore {
            let base = self
                .repo
                .first_parent(sha)
                .map_err(|error| error.to_string())?
                .unwrap_or_else(|| sha.to_string());
            self.repo
                .ensure_branch(branch, &base)
                .map_err(|error| error.to_string())?;
        }
        self.repo.restore_tree(sha).map_err(|error| error.to_string())
    }

    /// Commit the tree to the task branch, as one closed subtask's work.
    ///
    /// `commit_to_branch` rather than `commit_with_trailers`: the latter starts with
    /// `git checkout --force <branch>`, which reverts every tracked modification when the worktree
    /// is already on that branch. Measured — `M src/count.rs` plus an untracked `PLAN.md` came
    /// back identical to `HEAD` — and it is why the dogfood rehearsal went from green to
    /// `failed_gates`: the commit that was supposed to preserve the work destroyed it before the
    /// task-level gate read the diff.
    ///
    /// `Ok(None)` means the tree matched the branch's tip; a subtask can close without touching a
    /// file when an earlier one already landed the change.
    fn commit(&self, message: &str) -> Result<Option<String>, String> {
        self.repo
            .commit_to_branch(&self.branch, message)
            .map(|checkpoint| checkpoint.map(|checkpoint| checkpoint.sha))
            .map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn repo() -> (tempfile::TempDir, GitRepo) {
        let dir = tempfile::tempdir().expect("tempdir");
        for args in [
            vec!["init", "-q", "-b", "main"],
            vec!["config", "user.email", "test@example.com"],
            vec!["config", "user.name", "test"],
        ] {
            let status = Command::new("git")
                .arg("-C")
                .arg(dir.path())
                .args(&args)
                .status()
                .expect("git");
            assert!(status.success(), "git {args:?}");
        }
        std::fs::write(dir.path().join("a.txt"), "base\n").expect("write");
        for args in [vec!["add", "-A"], vec!["commit", "-q", "-m", "base"]] {
            let status = Command::new("git")
                .arg("-C")
                .arg(dir.path())
                .args(&args)
                .status()
                .expect("git");
            assert!(status.success(), "git {args:?}");
        }
        let repo = GitRepo::new(dir.path());
        (dir, repo)
    }

    fn checkpoints(dir: &Path) -> GitCheckpoints {
        GitCheckpoints::new(dir, "agent/tsk_1").restoring("agent/tsk_1")
    }

    #[test]
    fn a_checkpoint_is_a_commit_that_the_tree_can_be_reset_to() {
        let (dir, _) = repo();
        let world = checkpoints(dir.path());
        std::fs::write(dir.path().join("b.txt"), "attempt\n").expect("write");
        let snapshot = world
            .snapshot("first attempt")
            .expect("checkpoint")
            .expect("a sha");
        assert!(snapshot.branch.contains("agent-ckpt"), "{}", snapshot.branch);

        std::fs::write(dir.path().join("b.txt"), "thrown away\n").expect("write");
        world.restore(&snapshot.sha).expect("rollback");
        let text = std::fs::read_to_string(dir.path().join("b.txt")).expect("read");
        assert_eq!(text, "attempt\n", "the tree must be back at the checkpoint");
    }

    #[test]
    fn a_rollback_does_not_put_a_checkpoint_commit_on_the_task_branch() {
        // The daemon-level shape of the same defect: `restore` is what the runner calls, and it
        // used to end in `reset --hard` while `HEAD` was on `agent/<task>`.
        let (dir, repo) = repo();
        let world = checkpoints(dir.path());
        repo.ensure_branch("agent/tsk_1", "HEAD").expect("branch");
        std::fs::write(dir.path().join("a.txt"), "landed\n").expect("write");
        let landed = repo
            .commit_to_branch("agent/tsk_1", "chore: landed")
            .expect("commit")
            .expect("a commit");
        assert_eq!(
            repo.git(&["rev-parse", "agent/tsk_1"]).expect("tip").trim(),
            landed.sha
        );

        // A dirty tree makes the snapshot a child of the tip, which is what `reset --hard` moved
        // the branch onto.
        std::fs::write(dir.path().join("b.txt"), "uncommitted\n").expect("write");
        let base = world.snapshot("base").expect("checkpoint").expect("a sha");
        std::fs::write(dir.path().join("b.txt"), "lost\n").expect("write");
        world.restore(&base.sha).expect("rollback");

        assert_eq!(
            repo.git(&["rev-parse", "agent/tsk_1"]).expect("tip").trim(),
            landed.sha,
            "the history a human merges must not gain a `checkpoint` commit"
        );
        // And the rollback still did its job.
        assert_eq!(
            std::fs::read_to_string(dir.path().join("b.txt")).expect("read"),
            "uncommitted\n"
        );
    }

    #[test]
    fn a_rollback_returns_the_work_it_discarded() {
        // The property the whole milestone rests on: throwing work away is allowed, losing it is
        // not. The returned snapshot must contain the file the reset deletes.
        let (dir, _) = repo();
        let world = checkpoints(dir.path());
        let base = world.snapshot("base").expect("checkpoint").expect("a sha");

        std::fs::write(dir.path().join("only-here.txt"), "precious\n").expect("write");
        let safety = world
            .snapshot("L3 needs the old work")
            .expect("snapshot")
            .expect("a safety snapshot");
        world.restore(&base.sha).expect("rollback");
        assert!(!dir.path().join("only-here.txt").exists());

        let out = Command::new("git")
            .arg("-C")
            .arg(dir.path())
            .args(["show", &format!("{}:only-here.txt", safety.sha)])
            .output()
            .expect("git show");
        assert_eq!(String::from_utf8_lossy(&out.stdout), "precious\n");
    }

    #[test]
    fn a_rollback_puts_the_worktree_back_on_the_task_branch() {
        // A model that detached `HEAD` with a git command must not leave the next attempt
        // committing into nowhere.
        let (dir, repo) = repo();
        let world = checkpoints(dir.path());
        let base = world.snapshot("base").expect("checkpoint").expect("a sha");
        repo.ensure_branch("agent/tsk_1", "HEAD").expect("branch");
        repo.git(&["checkout", "--detach"]).expect("detach");
        assert_eq!(
            repo.git(&["rev-parse", "--abbrev-ref", "HEAD"]).expect("branch"),
            "HEAD\n"
        );

        world.restore(&base.sha).expect("rollback");
        assert_eq!(
            repo.git(&["rev-parse", "--abbrev-ref", "HEAD"]).expect("branch"),
            "agent/tsk_1\n"
        );
        assert!(!repo.is_dirty().expect("dirty"), "the tree must be clean");
    }

    #[test]
    fn a_missing_sha_is_an_error_and_leaves_the_work_in_place() {
        let (dir, _) = repo();
        let world = checkpoints(dir.path());
        std::fs::write(dir.path().join("keep.txt"), "keep\n").expect("write");
        let error = world
            .restore("0000000000000000000000000000000000000000")
            .expect_err("missing");
        assert!(error.contains("does not exist"), "{error}");
        assert!(dir.path().join("keep.txt").exists());
    }

    #[test]
    fn a_clean_tree_snapshots_head_rather_than_failing() {
        let (dir, _) = repo();
        let snapshot = checkpoints(dir.path()).snapshot("clean").expect("checkpoint");
        assert!(
            snapshot.is_some(),
            "a repository with a commit always has a SHA to reset to"
        );
    }
}
