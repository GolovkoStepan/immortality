//! Git checkpoints: the thing that makes L3 ("roll back and try again") mean something.
//!
//! Two independent kinds, as the design requires:
//!
//! - **Hidden** (`agent-ckpt/<task>`): a snapshot taken before every risky action. It never
//!   appears in the task's history, so an attempt that produces nothing leaves no trace.
//! - **Meaningful** (`agent/<task>`): a Conventional Commit when a subtask closes.
//!
//! The difference is not cosmetic. A hidden checkpoint is written on a branch nobody reads, so
//! "checkpoint every 30 seconds" costs nothing in review quality; a meaningful commit is what a
//! human merges, so it must be one per subtask and nothing else.
//!
//! # How a snapshot is taken
//!
//! `git stash create` is not used. It writes to the *global* stash and can be raced by the user;
//! moreover it refuses to run when there is nothing to stash, which makes the common "nothing
//! changed" case look like an error. Instead the tree is snapshotted with
//! `git add --all` + `git write-tree` + `git commit-tree`, and the resulting commit is pointed at
//! by a ref. This works with dirty trees, never touches the index's visible state for longer
//! than the command sequence, and produces a real SHA that `reset --hard` accepts.
//!
//! # Why the rollback preserves uncommitted work
//!
//! `reset --hard` destroys whatever the model was doing. The work may be exactly what the next
//! attempt needs to build on, and losing it after a crash makes the whole "days of autonomous
//! work" claim a lie. So a rollback first writes a **safety checkpoint** of the current tree, and
//! only then resets. If the safety checkpoint cannot be written, the rollback is refused: an
//! operator can always delete the worktree by hand, but the tool must not do it silently.
//!
//! # Why every command is `git -C <worktree>`
//!
//! The daemon serves several projects and the user's working directory is arbitrary. Relying on
//! the process's cwd would break the moment two tasks run in two worktrees, and would make a
//! stray `git` invocation operate on the wrong repository — the kind of mistake that is
//! discovered only after it has committed to `main`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::domain::NonceTag;

/// Prefix of the hidden checkpoint branch. One per task, force-updated on every snapshot.
pub const CKPT_BRANCH_PREFIX: &str = "agent-ckpt/";

/// Prefix of the meaningful task branch.
pub const TASK_BRANCH_PREFIX: &str = "agent/";

/// How a checkpoint was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckpointKind {
    /// Hidden snapshot: no message, no history, force-moved.
    Hidden,
    /// Meaningful commit: Conventional Commit message, appended to the branch.
    Meaningful,
}

impl CheckpointKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hidden => "hidden",
            Self::Meaningful => "meaningful",
        }
    }
}

/// A recorded checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    pub sha: String,
    pub kind: CheckpointKind,
    /// Branch the checkpoint was written to.
    pub branch: String,
    /// What it was taken for, shown in `immortality status`.
    pub reason: String,
}

/// A failure to talk to git.
///
/// Carries the command's stderr: `git`'s own message is better than any paraphrase, and it is
/// what an operator will search for.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("git {command} failed: {message}")]
pub struct GitError {
    pub command: String,
    pub message: String,
}

impl GitError {
    fn new(command: impl Into<String>, stderr: &str) -> Self {
        Self {
            command: command.into(),
            message: stderr.trim().to_string(),
        }
    }
}

/// A checkout of the repository that checkpoints are taken in.
///
/// Usually a dedicated worktree per task, never the user's own checkout: the agent runs
/// `git add --all` and `reset --hard`, and doing that in the directory a human is editing would
/// be a hostile surprise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitRepo {
    worktree: PathBuf,
}

/// Identity used for agent commits.
///
/// Set through `-c` on every invocation rather than written to `.git/config`: the user's global
/// identity must not leak into commits attributed to the agent, and editing repository config is
/// exactly what the policy layer forbids.
const AGENT_NAME: &str = "immortality";
const AGENT_EMAIL: &str = "agent@immortality.local";

impl GitRepo {
    /// Wrap a worktree path. Does not verify that it is a repository: [`Self::is_repository`]
    /// exists for that, and a caller that skips it gets a clear git error rather than a panic.
    #[must_use]
    pub fn new(worktree: impl Into<PathBuf>) -> Self {
        Self {
            worktree: worktree.into(),
        }
    }

    #[must_use]
    pub fn worktree(&self) -> &Path {
        &self.worktree
    }

    /// Whether the path is inside a git work tree.
    #[must_use]
    pub fn is_repository(&self) -> bool {
        self.git(&["rev-parse", "--git-dir"]).is_ok()
    }

    /// Current `HEAD` SHA, or `None` in an empty repository with no commits.
    pub fn head(&self) -> Result<Option<String>, GitError> {
        match self.git(&["rev-parse", "--verify", "HEAD"]) {
            Ok(out) => Ok(Some(out.trim().to_string())),
            Err(error) if error.message.contains("Needed a single revision") => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Whether the working tree has any modification, staged or not, including untracked files.
    ///
    /// Untracked files count: a new file the model wrote is work, and a rollback that keeps
    /// "modifications" but silently deletes new files would be surprising.
    pub fn is_dirty(&self) -> Result<bool, GitError> {
        let out = self.git(&["status", "--porcelain", "--untracked-files=all"])?;
        Ok(!out.trim().is_empty())
    }

    /// Take a hidden snapshot and move `agent-ckpt/<task>` to it.
    ///
    /// Returns `None` when the tree is clean and `HEAD` is already on the checkpoint branch:
    /// there is nothing new to record, and creating an identical commit on every risky action
    /// would fill the object store with duplicates for no benefit.
    pub fn hidden_checkpoint(&self, task_branch: &str, reason: &str) -> Result<Option<Checkpoint>, GitError> {
        let ckpt_branch = format!("{CKPT_BRANCH_PREFIX}{}", strip_agent_prefix(task_branch));
        let parent = self.head()?;

        if !self.is_dirty()? {
            // A clean tree still needs a checkpoint pointing at HEAD the first time, otherwise
            // `reset --hard <ckpt>` has nothing to target.
            let Some(parent) = parent else {
                return Ok(None);
            };
            self.force_branch(&ckpt_branch, &parent)?;
            return Ok(Some(Checkpoint {
                sha: parent,
                kind: CheckpointKind::Hidden,
                branch: ckpt_branch,
                reason: reason.to_string(),
            }));
        }

        // Snapshot the index into a tree object without moving HEAD or the user-visible index
        // for longer than this command sequence.
        let tree = self.snapshot_tree()?;
        let tree = tree.trim().to_string();
        let identity = identity_args();
        let mut args: Vec<&str> = vec!["-c", &identity[0], "-c", &identity[1], "commit-tree", &tree];
        if let Some(parent) = &parent {
            args.extend(["-p", parent]);
        }
        // `-m` must come after the parent flags: git parses options positionally in practice.
        args.extend(["-m", "checkpoint"]);
        let sha = self.git(&args)?.trim().to_string();

        self.force_branch(&ckpt_branch, &sha)?;
        Ok(Some(Checkpoint {
            sha,
            kind: CheckpointKind::Hidden,
            branch: ckpt_branch,
            reason: reason.to_string(),
        }))
    }

    /// Commit the working tree to `task_branch` without moving `HEAD` or touching the tree.
    ///
    /// The same sequence a hidden checkpoint uses — `add --all`, `write-tree`, `commit-tree`,
    /// `update-ref` — pointed at the task branch instead of `agent-ckpt/<task>`. That is the
    /// point: `commit_with_trailers` starts with `git checkout --force <branch>`, and a forced
    /// checkout of the branch the worktree is already on **reverts every tracked modification**.
    /// Measured: a worktree with `M src/count.rs` and `A PLAN.md` came back identical to its
    /// `HEAD` after `git checkout --force agent/<task>`, so the work a subtask had just passed its
    /// gates with was thrown away before it could be committed.
    ///
    /// `Ok(None)` when the tree is identical to the branch's tip — a subtask whose `DoD` was
    /// already satisfied by earlier work. A commit with no change would be noise in a history a
    /// human reviews before merging.
    ///
    /// # Errors
    /// Git failed. The caller must not close the subtask.
    pub fn commit_to_branch(&self, task_branch: &str, message: &str) -> Result<Option<Checkpoint>, GitError> {
        let branch = ensure_agent_prefix(task_branch);
        self.git(&["add", "--all"])?;
        let tree = self.git(&["write-tree"])?.trim().to_string();
        // The parent is the branch's tip when it exists, `HEAD` otherwise: a task branch that
        // nobody has created yet starts where the worktree is, which is the commit the worktree
        // was built from.
        let parent = if self.branch_exists(&branch) {
            Some(self.git(&["rev-parse", &branch])?.trim().to_string())
        } else {
            self.head()?
        };
        if let Some(parent) = &parent {
            let parent_tree = self
                .git(&["rev-parse", &format!("{parent}^{{tree}}")])?
                .trim()
                .to_string();
            if parent_tree == tree {
                return Ok(None);
            }
        }

        let identity = identity_args();
        let mut args: Vec<&str> = vec!["-c", &identity[0], "-c", &identity[1], "commit-tree", &tree];
        if let Some(parent) = &parent {
            args.extend(["-p", parent]);
        }
        args.extend(["-m", message]);
        let sha = self.git(&args)?.trim().to_string();
        // The ref moves; when `HEAD` is a symref to it the worktree's view follows, and the index
        // already holds this tree, so the tree reads clean afterwards. When `HEAD` is elsewhere
        // the worktree keeps its changes, which is what the untracked-file branch of the gate diff
        // expects.
        self.force_branch(&branch, &sha)?;
        Ok(Some(Checkpoint {
            sha,
            kind: CheckpointKind::Meaningful,
            branch,
            reason: message.to_string(),
        }))
    }

    /// Commit the current tree to the task branch with a Conventional Commit message.
    ///
    /// `nonce` is appended as a trailer so that a crash-time probe can find this commit by
    /// searching the log. Without it, "did the commit happen?" is unanswerable after a crash
    /// between `commit` and writing the journal record.
    pub fn meaningful_commit(
        &self,
        task_branch: &str,
        message: &str,
        nonce: &NonceTag,
    ) -> Result<Checkpoint, GitError> {
        self.commit_with_trailers(task_branch, message, &[("Attempt", nonce.as_str())])
    }

    /// Commit the current tree to the task branch, with arbitrary trailers in the body.
    ///
    /// One commit path for both callers. The attempt path records `Attempt: <nonce>` and the
    /// model-driven `git` tool records `Call: <call-id>` — and the second exists because a probe
    /// runs after a crash with nothing in memory but the journal, which carries the call id and
    /// *not* the arguments. A probe that had to guess the subcommand would be a probe that
    /// answers `Applied` for a commit nobody made.
    pub fn commit_with_trailers(
        &self,
        task_branch: &str,
        message: &str,
        trailers: &[(&str, &str)],
    ) -> Result<Checkpoint, GitError> {
        let branch = ensure_agent_prefix(task_branch);
        // Detach onto the branch first so the commit lands in the right history even when the
        // worktree was left on another branch by a previous failed attempt.
        self.git(&["checkout", "--force", &branch])?;
        self.git(&["add", "--all"])?;

        let mut body = message.to_string();
        if !trailers.is_empty() {
            // A blank line, because git only recognises trailers in the last paragraph. Without
            // it `Attempt: …` is prose, and `log --grep` still finds it while a human reading
            // `git log` sees a message that looks unfinished.
            body.push_str("\n\n");
            for (index, (key, value)) in trailers.iter().enumerate() {
                if index > 0 {
                    body.push('\n');
                }
                let _ = write!(body, "{key}: {value}");
            }
        }
        let identity = identity_args();
        self.git(&[
            "-c",
            &identity[0],
            "-c",
            &identity[1],
            // `--no-verify`: a repository hook written by the model must not be able to veto or
            // silently rewrite the agent's own commits.
            "commit",
            "--no-verify",
            "-m",
            &body,
        ])?;
        let sha = self
            .head()?
            .ok_or_else(|| GitError::new("rev-parse HEAD", "commit produced no HEAD"))?;
        Ok(Checkpoint {
            sha,
            kind: CheckpointKind::Meaningful,
            branch,
            reason: message.to_string(),
        })
    }

    /// Create the task branch from `base` if it does not exist and check it out.
    pub fn ensure_branch(&self, task_branch: &str, base: &str) -> Result<(), GitError> {
        let branch = ensure_agent_prefix(task_branch);
        if self.git(&["rev-parse", "--verify", &branch]).is_ok() {
            self.git(&["checkout", "--force", &branch])?;
        } else {
            self.git(&["checkout", "--force", "-B", &branch, base])?;
        }
        Ok(())
    }

    /// Roll the worktree back to `sha`, keeping the current work in a safety checkpoint.
    ///
    /// Returns the safety checkpoint so the caller can journal it: a human who is told "your
    /// work was rolled back" must be able to find what was rolled back.
    pub fn rollback_to(
        &self,
        sha: &str,
        task_branch: &str,
        reason: &str,
    ) -> Result<Option<Checkpoint>, GitError> {
        let safety = self.hidden_checkpoint(task_branch, reason)?;
        self.reset_hard(sha)?;
        Ok(safety)
    }

    /// Move the tree to `sha` and delete what the reset would otherwise leave behind.
    ///
    /// No safety checkpoint: this is the second half of [`Self::rollback_to`], and the caller
    /// that takes its own snapshot first — the daemon's L3 path does, because the snapshot has to
    /// be journaled *before* the tree is destroyed — would otherwise take it twice.
    ///
    /// # Errors
    /// `sha` is not a commit, or git refuses the reset. A refusal is reported and not retried:
    /// the tree may be half-moved, and continuing is how a run builds on a state nobody knows.
    pub fn reset_hard(&self, sha: &str) -> Result<(), GitError> {
        if self
            .git(&["cat-file", "-e", &format!("{sha}^{{commit}}")])
            .is_err()
        {
            return Err(GitError::new(
                format!("cat-file {sha}"),
                "checkpoint commit does not exist",
            ));
        }
        self.git(&["reset", "--hard", sha])?;
        // Untracked files are not touched by `reset --hard`, and they are exactly the leftovers
        // that make the next attempt fail for reasons nobody understands.
        self.git(&["clean", "-fd"])?;
        Ok(())
    }

    /// Commit exists and is reachable from any ref, used by the post-crash probe for `git commit`.
    #[must_use]
    pub fn has_commit_with_nonce(&self, nonce: &NonceTag) -> bool {
        let pattern = format!("Attempt: {nonce}");
        self.git(&[
            "log",
            "--all",
            "--fixed-strings",
            "--grep",
            &pattern,
            "--format=%H",
        ])
        .is_ok_and(|out| !out.trim().is_empty())
    }

    /// List commits whose message contains the nonce, newest first.
    pub fn commits_with_nonce(&self, nonce: &NonceTag) -> Result<Vec<String>, GitError> {
        let pattern = format!("Attempt: {nonce}");
        let out = self.git(&[
            "log",
            "--all",
            "--fixed-strings",
            "--grep",
            &pattern,
            "--format=%H",
        ])?;
        Ok(out
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect())
    }

    /// Whether a branch exists.
    #[must_use]
    pub fn branch_exists(&self, branch: &str) -> bool {
        self.git(&["rev-parse", "--verify", branch]).is_ok()
    }

    /// Whether a SHA exists in this repository.
    #[must_use]
    pub fn has_commit(&self, sha: &str) -> bool {
        self.git(&["cat-file", "-e", &format!("{sha}^{{commit}}")])
            .is_ok()
    }

    /// Files changed between two commits, for the reviewer subagent and diff sanity.
    pub fn changed_files(&self, from: &str, to: &str) -> Result<Vec<PathBuf>, GitError> {
        let out = self.git(&["diff", "--name-only", from, to])?;
        Ok(out
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(PathBuf::from)
            .collect())
    }

    /// Commit the current index on whatever branch is checked out.
    ///
    /// Separate from [`Self::meaningful_commit`] because that one checks the task branch out
    /// first: correct for the daemon closing a subtask, wrong for a model that asked to commit
    /// its work *where it is*. A commit that silently moved `HEAD` would make the next `git
    /// status` describe a tree the model never touched.
    ///
    /// Returns `None` when the index has nothing to commit. That is a normal answer — "working
    /// tree clean" — and not an error a model should have to parse out of stderr.
    ///
    /// # Errors
    /// Git failed for a reason other than an empty index.
    pub fn commit_on_current(
        &self,
        message: &str,
        trailers: &[(&str, &str)],
    ) -> Result<Option<String>, GitError> {
        // Checked before committing, not inferred from stderr afterwards: `git commit` exits 1
        // both for "nothing to commit" and for a real failure, and the two need different
        // answers. `--quiet` makes this a status question with no output.
        if self.git(&["diff", "--cached", "--quiet"]).is_ok() {
            return Ok(None);
        }
        let mut body = message.to_string();
        if !trailers.is_empty() {
            body.push_str("\n\n");
            for (index, (key, value)) in trailers.iter().enumerate() {
                if index > 0 {
                    body.push('\n');
                }
                let _ = write!(body, "{key}: {value}");
            }
        }
        let identity = identity_args();
        self.git(&[
            "-c",
            &identity[0],
            "-c",
            &identity[1],
            "commit",
            "--no-verify",
            "-m",
            &body,
        ])?;
        self.head()
    }

    /// Stage everything, deletions and untracked files included.
    ///
    /// # Errors
    /// Git failed.
    pub fn add_all(&self) -> Result<(), GitError> {
        self.git(&["add", "--all"]).map(|_| ())
    }

    /// Stage one path.
    ///
    /// # Errors
    /// The path is not tracked and does not exist, which git reports as a fatal.
    pub fn add_path(&self, path: &str) -> Result<(), GitError> {
        self.git(&["add", "--", path]).map(|_| ())
    }

    /// Whether any commit's message contains `needle`, in any branch or the stash.
    ///
    /// Used by the `git` tool's commit probe. `--all` covers the task branch and the checkpoint
    /// branch, and `refs/stash` is searched separately because `--all` does not include it —
    /// measured, not assumed.
    #[must_use]
    pub fn has_trailer(&self, needle: &str) -> bool {
        let pattern = needle.to_string();
        if self
            .git(&[
                "log",
                "--all",
                "--fixed-strings",
                "--grep",
                &pattern,
                "--format=%H",
            ])
            .is_ok_and(|out| !out.trim().is_empty())
        {
            return true;
        }
        self.git(&[
            "log",
            "refs/stash",
            "-n",
            "50",
            "--fixed-strings",
            "--grep",
            &pattern,
            "--format=%H",
        ])
        .is_ok_and(|out| !out.trim().is_empty())
    }

    fn snapshot_tree(&self) -> Result<String, GitError> {
        // `add --all` stages deletions and untracked files alike, which is what a snapshot of
        // "the state of the working tree" means. The index is left populated; every caller
        // either resets or commits immediately afterwards.
        self.git(&["add", "--all"])?;
        self.git(&["write-tree"])
    }

    fn force_branch(&self, branch: &str, sha: &str) -> Result<(), GitError> {
        self.git(&["update-ref", &format!("refs/heads/{branch}"), sha])?;
        Ok(())
    }

    /// Run a git command in this worktree and return its stdout.
    ///
    /// Public because the `git_*` tools are a thin surface over exactly this: a tool that built
    /// its own `Command` would have to remember `-C`, `GIT_TERMINAL_PROMPT`, the pager and
    /// `LC_ALL`, and the one that forgot would hang the daemon on a credential prompt. One
    /// spawner, one set of guards, one place to audit.
    pub fn git(&self, args: &[&str]) -> Result<String, GitError> {
        let output = Command::new("git")
            .arg("-C")
            .arg(&self.worktree)
            .args(args)
            // Never let a credential prompt or a pager hang the daemon.
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
            Err(GitError::new(
                args.join(" "),
                &String::from_utf8_lossy(&output.stderr),
            ))
        }
    }
}

/// `-c` pairs that attribute commits to the agent rather than to the user.
///
/// Returned as owned strings because `Command::args` borrows them across the call; building
/// them inline produces temporaries that die before `git` runs.
fn identity_args() -> [String; 2] {
    [
        format!("user.name={AGENT_NAME}"),
        format!("user.email={AGENT_EMAIL}"),
    ]
}

/// Strip a leading `agent/` so a full branch name can be passed where a task part is expected.
fn strip_agent_prefix(branch: &str) -> &str {
    branch.strip_prefix(TASK_BRANCH_PREFIX).unwrap_or(branch)
}

/// Add `agent/` unless the caller already supplied a prefixed branch.
fn ensure_agent_prefix(branch: &str) -> String {
    if branch.starts_with(TASK_BRANCH_PREFIX) || branch.starts_with(CKPT_BRANCH_PREFIX) {
        branch.to_string()
    } else {
        format!("{TASK_BRANCH_PREFIX}{branch}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    /// A throwaway repository with one commit and a `main` branch.
    struct Fixture {
        dir: tempfile::TempDir,
        repo: GitRepo,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let repo = GitRepo::new(dir.path());
            for args in [
                vec!["init", "-q", "-b", "main"],
                vec!["config", "user.name", "Test"],
                vec!["config", "user.email", "test@example.com"],
            ] {
                run(dir.path(), &args);
            }
            std::fs::write(dir.path().join("README.md"), "# fixture\n").unwrap();
            run(dir.path(), &["add", "--all"]);
            run(dir.path(), &["commit", "-q", "-m", "chore: initial"]);
            Self { dir, repo }
        }

        fn write(&self, name: &str, contents: &str) {
            let path = self.dir.path().join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, contents).unwrap();
        }

        fn read(&self, name: &str) -> String {
            std::fs::read_to_string(self.dir.path().join(name)).unwrap()
        }
    }

    fn nonce() -> NonceTag {
        NonceTag::new(crate::ids::Nonce::generate().to_string())
    }

    fn run(dir: &Path, args: &[&str]) {
        let output = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .env("GIT_TERMINAL_PROMPT", "0")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn a_repository_is_recognised() {
        let fixture = Fixture::new();
        assert!(fixture.repo.is_repository());
        assert!(!GitRepo::new(fixture.dir.path().join("nope")).is_repository());
    }

    #[test]
    fn a_hidden_checkpoint_snapshots_a_dirty_tree() {
        let fixture = Fixture::new();
        fixture.write("work.txt", "attempt one");
        let checkpoint = fixture
            .repo
            .hidden_checkpoint("agent/tsk_1", "before shell")
            .unwrap()
            .expect("a checkpoint");

        assert_eq!(checkpoint.kind, CheckpointKind::Hidden);
        assert_eq!(checkpoint.branch, "agent-ckpt/tsk_1");
        // The working tree is untouched and the field is on disk.
        assert_eq!(fixture.read("work.txt"), "attempt one");
        assert!(fixture.repo.has_commit(&checkpoint.sha));
    }

    #[test]
    fn a_hidden_checkpoint_captures_untracked_files() {
        let fixture = Fixture::new();
        fixture.write("new/deep/file.txt", "payload");
        let checkpoint = fixture
            .repo
            .hidden_checkpoint("agent/tsk_1", "before risky")
            .unwrap()
            .unwrap();
        // The SHA is a real commit whose tree contains the untracked file.
        let out = Command::new("git")
            .arg("-C")
            .arg(fixture.dir.path())
            .args(["show", "--name-only", "--format=", &checkpoint.sha])
            .output()
            .unwrap();
        let listed = String::from_utf8_lossy(&out.stdout);
        assert!(listed.contains("new/deep/file.txt"), "{listed}");
    }

    #[test]
    fn a_checkpoint_with_nothing_to_snapshot_points_at_head() {
        let fixture = Fixture::new();
        let head = fixture.repo.head().unwrap().unwrap();
        let checkpoint = fixture
            .repo
            .hidden_checkpoint("agent/tsk_1", "clean tree")
            .unwrap()
            .unwrap();
        assert_eq!(checkpoint.sha, head);
    }

    #[test]
    fn rollback_restores_the_exact_tree() {
        let fixture = Fixture::new();
        fixture.write("work.txt", "good");
        let checkpoint = fixture
            .repo
            .hidden_checkpoint("agent/tsk_1", "good state")
            .unwrap()
            .unwrap();

        fixture.write("work.txt", "broken");
        fixture.write("junk.txt", "leftover");
        fixture
            .repo
            .rollback_to(&checkpoint.sha, "agent/tsk_1", "L3")
            .unwrap();

        assert_eq!(fixture.read("work.txt"), "good");
        assert!(
            !fixture.dir.path().join("junk.txt").exists(),
            "untracked leftovers must be cleaned"
        );
    }

    #[test]
    fn rollback_preserves_discarded_work_in_a_safety_checkpoint() {
        let fixture = Fixture::new();
        fixture.write("work.txt", "good");
        let checkpoint = fixture
            .repo
            .hidden_checkpoint("agent/tsk_1", "good")
            .unwrap()
            .unwrap();

        fixture.write("only-in-broken.txt", "precious");
        let safety = fixture
            .repo
            .rollback_to(&checkpoint.sha, "agent/tsk_1", "L3 needs the old work")
            .unwrap()
            .expect("a safety checkpoint");

        assert!(!fixture.dir.path().join("only-in-broken.txt").exists());
        assert!(fixture.repo.has_commit(&safety.sha));
        let out = Command::new("git")
            .arg("-C")
            .arg(fixture.dir.path())
            .args(["show", &format!("{}:only-in-broken.txt", safety.sha)])
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout), "precious");
    }

    #[test]
    fn reset_hard_moves_the_tree_without_taking_a_second_snapshot() {
        // The daemon snapshots first (so the discarded work can be journaled before it is gone)
        // and then calls this. A second safety checkpoint here would journal two SHAs for one
        // rollback and leave the caller reading a branch pointer that moved twice.
        let fixture = Fixture::new();
        fixture.write("work.txt", "good");
        let checkpoint = fixture
            .repo
            .hidden_checkpoint("agent/tsk_1", "good state")
            .unwrap()
            .unwrap();
        let branch_before = fixture.repo.git(&["rev-parse", "agent-ckpt/tsk_1"]).unwrap();

        fixture.write("work.txt", "broken");
        fixture.repo.reset_hard(&checkpoint.sha).unwrap();

        assert_eq!(fixture.read("work.txt"), "good");
        let branch_after = fixture.repo.git(&["rev-parse", "agent-ckpt/tsk_1"]).unwrap();
        assert_eq!(branch_before, branch_after, "the checkpoint ref must not move");
    }

    #[test]
    fn reset_hard_refuses_a_missing_sha_and_leaves_the_tree_alone() {
        let fixture = Fixture::new();
        fixture.write("work.txt", "keep me");
        assert!(
            fixture
                .repo
                .reset_hard("0000000000000000000000000000000000000000")
                .is_err()
        );
        assert_eq!(fixture.read("work.txt"), "keep me");
    }

    #[test]
    fn rollback_to_a_missing_sha_is_an_error_and_changes_nothing() {
        let fixture = Fixture::new();
        fixture.write("work.txt", "keep me");
        let error = fixture
            .repo
            .rollback_to("0000000000000000000000000000000000000000", "agent/tsk_1", "L3")
            .unwrap_err();
        assert!(error.message.contains("does not exist"), "{error}");
        assert_eq!(fixture.read("work.txt"), "keep me");
    }

    #[test]
    fn committing_to_the_branch_keeps_the_tree_it_committed() {
        // The failure this replaces was measured on the dogfood rehearsal: `git checkout --force`
        // of the branch the worktree was already on reverted every tracked modification, so the
        // work a subtask had just passed its gates with disappeared, and the task-level gate then
        // read an empty diff and went red. A commit is supposed to preserve work, not destroy it.
        let fixture = Fixture::new();
        fixture.repo.ensure_branch("agent/tsk_keep", "main").unwrap();
        fixture.write("README.md", "# changed by the subtask\n");
        fixture.write("src/new.rs", "pub fn new() {}\n");

        let checkpoint = fixture
            .repo
            .commit_to_branch("agent/tsk_keep", "chore: keep the work")
            .unwrap()
            .expect("a commit");
        assert_eq!(checkpoint.kind, CheckpointKind::Meaningful);

        // Both files are still there, with their contents: the tracked edit was not reverted, and
        // the untracked file was committed rather than left behind.
        assert_eq!(fixture.read("README.md"), "# changed by the subtask\n");
        assert_eq!(fixture.read("src/new.rs"), "pub fn new() {}\n");
        // And the branch carries them, which is what a human merges.
        let branch = fixture.repo.git(&["rev-parse", "agent/tsk_keep"]).unwrap();
        assert_eq!(branch.trim(), checkpoint.sha);
        let on_branch = fixture
            .repo
            .git(&["show", "--name-only", "--format=", &checkpoint.sha])
            .unwrap();
        assert!(on_branch.contains("README.md"), "{on_branch}");
        assert!(on_branch.contains("src/new.rs"), "{on_branch}");
    }

    #[test]
    fn committing_an_unchanged_tree_reports_nothing_to_commit() {
        // A subtask can close without touching a file — its `DoD` may be satisfied by work an
        // earlier subtask landed. An empty commit in a history a human reviews before merging is
        // noise, and `Ok(None)` is the honest answer.
        let fixture = Fixture::new();
        fixture.repo.ensure_branch("agent/tsk_same", "main").unwrap();
        let first = fixture
            .repo
            .commit_to_branch("agent/tsk_same", "chore: nothing new")
            .unwrap();
        assert!(first.is_none(), "an unchanged tree has nothing to commit");
    }

    #[test]
    fn a_meaningful_commit_lands_on_the_task_branch_with_its_nonce() {
        let fixture = Fixture::new();
        fixture.repo.ensure_branch("agent/tsk_2", "main").unwrap();
        fixture.write("feature.rs", "fn main() {}\n");
        let nonce = nonce();
        let checkpoint = fixture
            .repo
            .meaningful_commit("agent/tsk_2", "feat: add the feature", &nonce)
            .unwrap();

        assert_eq!(checkpoint.kind, CheckpointKind::Meaningful);
        assert_eq!(checkpoint.branch, "agent/tsk_2");
        assert!(fixture.repo.has_commit_with_nonce(&nonce));
        assert_eq!(
            fixture.repo.commits_with_nonce(&nonce).unwrap(),
            vec![checkpoint.sha]
        );
    }

    #[test]
    fn a_nonce_from_another_attempt_is_not_found() {
        let fixture = Fixture::new();
        fixture.repo.ensure_branch("agent/tsk_2", "main").unwrap();
        fixture.write("a.rs", "x\n");
        let mine = nonce();
        fixture
            .repo
            .meaningful_commit("agent/tsk_2", "feat: one", &mine)
            .unwrap();

        let other = NonceTag::new("NOT-A-REAL-NONCE");
        assert!(!fixture.repo.has_commit_with_nonce(&other));
        assert_eq!(fixture.repo.commits_with_nonce(&other).unwrap().len(), 0);
    }

    #[test]
    fn the_probe_search_is_by_substring_not_by_prefix() {
        // A commit made by a *different* attempt must not satisfy this attempt's probe, and a
        // substring search is the only shape that works for a git message.
        let fixture = Fixture::new();
        fixture.repo.ensure_branch("agent/tsk_3", "main").unwrap();
        fixture.write("a.rs", "x\n");
        let mine = nonce();
        fixture
            .repo
            .meaningful_commit("agent/tsk_3", "feat: real", &mine)
            .unwrap();
        assert!(
            !fixture
                .repo
                .has_commit_with_nonce(&NonceTag::new(format!("{}x", mine.as_str()))),
            "a longer nonce must not match"
        );
    }

    #[test]
    fn ensure_branch_creates_then_reuses() {
        let fixture = Fixture::new();
        fixture.repo.ensure_branch("agent/tsk_4", "main").unwrap();
        assert!(fixture.repo.branch_exists("agent/tsk_4"));
        fixture.write("a.rs", "x\n");
        fixture
            .repo
            .meaningful_commit("agent/tsk_4", "feat: first", &nonce())
            .unwrap();
        let head = fixture.repo.head().unwrap().unwrap();

        // Re-entering must not recreate the branch from main and lose the commit.
        fixture.repo.ensure_branch("agent/tsk_4", "main").unwrap();
        assert_eq!(fixture.repo.head().unwrap().unwrap(), head);
    }

    #[test]
    fn a_full_branch_name_is_accepted_where_a_task_part_is_expected() {
        let fixture = Fixture::new();
        fixture.repo.ensure_branch("agent/tsk_5", "main").unwrap();
        fixture.write("a.rs", "x\n");
        let checkpoint = fixture
            .repo
            .meaningful_commit("agent/tsk_5", "feat: x", &nonce())
            .unwrap();
        assert_eq!(checkpoint.branch, "agent/tsk_5");
    }

    #[test]
    fn changed_files_lists_the_diff() {
        let fixture = Fixture::new();
        let base = fixture.repo.head().unwrap().unwrap();
        fixture.write("one.rs", "1\n");
        fixture.write("two/two.rs", "2\n");
        let from = fixture
            .repo
            .hidden_checkpoint("agent/tsk_6", "one")
            .unwrap()
            .unwrap();
        fixture.write("three.rs", "3\n");
        let to = fixture
            .repo
            .hidden_checkpoint("agent/tsk_6", "two")
            .unwrap()
            .unwrap();

        let changed = fixture.repo.changed_files(&from.sha, &to.sha).unwrap();
        assert_eq!(changed, vec![PathBuf::from("three.rs")]);
        let _ = base;
    }

    #[test]
    fn a_clean_tree_produces_no_new_commit() {
        let fixture = Fixture::new();
        let first = fixture
            .repo
            .hidden_checkpoint("agent/tsk_7", "one")
            .unwrap()
            .unwrap();
        let second = fixture
            .repo
            .hidden_checkpoint("agent/tsk_7", "two")
            .unwrap()
            .unwrap();
        assert_eq!(
            first.sha, second.sha,
            "an unchanged tree must not create a duplicate commit"
        );
    }

    #[test]
    fn is_dirty_tracks_untracked_files() {
        let fixture = Fixture::new();
        assert!(!fixture.repo.is_dirty().unwrap());
        fixture.write("untracked.txt", "x");
        assert!(fixture.repo.is_dirty().unwrap(), "untracked files are work");
    }

    #[test]
    fn git_errors_carry_the_stderr() {
        let fixture = Fixture::new();
        let error = fixture.repo.git(&["checkout", "no-such-branch"]).unwrap_err();
        assert!(error.message.contains("no-such-branch"), "{error}");
        assert!(error.command.contains("checkout"), "{error}");
    }
}
