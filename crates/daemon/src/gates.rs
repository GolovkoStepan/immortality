//! The world the gate evaluation talks to: git, the policy gate, and the command runner.
//!
//! # Where the diff comes from
//!
//! Against `HEAD`, and including untracked files. `HEAD` rather than "the last committed state"
//! because the agent commits its own checkpoints: a subtask's work may already be committed, and a
//! diff against `HEAD` would then be empty and the reviewer would approve nothing. The three
//! commands below build the change out of what is actually on disk:
//!
//! - `git diff HEAD` for tracked changes (staged and unstaged alike),
//! - `git ls-files --others --exclude-standard` for files the agent created and never added,
//! - the binary sniff for those untracked files, because git has no blob to compare them against.
//!
//! Untracked files are the common case for an agent that wrote a new module and did not commit,
//! and dropping them would let exactly the files nobody has reviewed through the gate.
//!
//! # Why the base is configurable
//!
//! The daemon's checkpoint machinery knows which commit an attempt started from. When it passes
//! one, the diff is scoped to this attempt's work instead of the whole task; when it does not, the
//! diff is against `HEAD`, which is a correct — if broader — answer. Broadening is the safe
//! direction: an over-wide diff costs a reviewer some reading, an under-wide one approves unseen
//! work.
//!
//! The task-level verdict has the opposite problem and so the opposite base. Every closed subtask
//! is committed to `agent/<task>` (D-122), which means the work is *in* `HEAD` and the diff against
//! it is empty. [`task_base`] gives the fork point instead, so the reviewer reads what the task
//! actually added rather than nothing at all (D-123).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use immortality_core::checkpoint::GitRepo;
use immortality_core::gate::{CommandRun, Diff, Gates, ReviewRequest, Reviewer, parse_diff};
use immortality_tools::PolicyGate;

use crate::runner::Model;

/// How a gate command is run.
#[derive(Debug, Clone)]
pub struct CheckRunner {
    /// Where the command runs.
    pub worktree: PathBuf,
    /// Per-command deadline.
    pub timeout: Duration,
    /// Where a check's whole output is kept, when one is wired.
    ///
    /// A gate check is the one command in the system whose output nothing else records: the tail
    /// is bounded, the process exits, and a post-mortem has no way to ask what it printed. The
    /// live `black_jack` run left two red `rspec` gates that nobody could inspect afterwards,
    /// while the same command run through `shell` had a full artifact.
    pub artifacts: Option<std::sync::Arc<immortality_core::artifacts::ArtifactStore>>,
}

impl CheckRunner {
    /// A runner for one worktree, with the tool layer's default deadline.
    #[must_use]
    pub fn new(worktree: impl Into<PathBuf>) -> Self {
        Self {
            worktree: worktree.into(),
            timeout: Duration::from_secs(immortality_tools::shell::DEFAULT_TIMEOUT_SECS),
            artifacts: None,
        }
    }

    /// Keep every check's whole output in `artifacts`.
    #[must_use]
    pub fn spooling_to(
        mut self,
        artifacts: std::sync::Arc<immortality_core::artifacts::ArtifactStore>,
    ) -> Self {
        self.artifacts = Some(artifacts);
        self
    }
}

/// The production [`Gates`].
#[derive(Debug, Clone)]
pub struct RealGates {
    repo: GitRepo,
    gate: Arc<PolicyGate>,
    checks: CheckRunner,
    /// The commit the diff is against, when the caller knows the attempt's base.
    base: Option<String>,
}

impl RealGates {
    /// Build the world for one evaluation.
    #[must_use]
    pub fn new(
        worktree: impl Into<PathBuf>,
        gate: Arc<PolicyGate>,
        checks: CheckRunner,
        base: Option<String>,
    ) -> Self {
        let worktree = worktree.into();
        Self {
            repo: GitRepo::new(&worktree),
            gate,
            checks,
            base,
        }
    }

    /// The repository, for the caller that wants to record what the diff was against.
    #[must_use]
    pub const fn repo(&self) -> &GitRepo {
        &self.repo
    }
}

impl Gates for RealGates {
    fn diff(&self, base: Option<&str>) -> Result<Diff, String> {
        // The caller's base wins: it is the checkpoint this attempt rolled back to, and only the
        // caller knows it. `Self::base` is the task-level fallback for an attempt that never
        // restarted.
        let base = base
            .map(str::to_string)
            .or_else(|| self.base.clone())
            .unwrap_or_else(|| "HEAD".to_string());
        let tracked = self
            .repo
            .git(&["diff", "--no-color", &base])
            .map_err(|error| error.to_string())?;
        let untracked = self
            .repo
            .git(&["ls-files", "--others", "--exclude-standard"])
            .map_err(|error| error.to_string())?;

        let mut text = tracked;
        for path in untracked.lines().map(str::trim).filter(|line| !line.is_empty()) {
            text.push_str(&untracked_diff(&self.repo, path));
        }
        Ok(parse_diff(&text, false))
    }

    fn authorise(&self, command: &str) -> Result<(), String> {
        let verdict = self.gate.check_command(command);
        if verdict.is_allowed() {
            return Ok(());
        }
        // An escalation is not a yes: a gate command waits for nobody, so an external command
        // fails the check rather than pausing the verification until a human answers.
        Err(verdict
            .observation()
            .unwrap_or_else(|| "the command needs a human".to_string()))
    }

    fn run(&self, command: &str, timeout: Option<Duration>) -> Result<CommandRun, String> {
        // The check's own deadline wins: a plan that declares nine minutes for `cargo test` means
        // it, and a runner that clamped it to the default would fail a check the author knowingly
        // made slow. `run_command` applies the outer cap, so a plan cannot ask for an hour either.
        let timeout = timeout.unwrap_or(self.checks.timeout);
        let (output, artifact) = match &self.checks.artifacts {
            Some(store) => {
                let (output, digest) =
                    immortality_tools::run_command_spooled(command, &self.checks.worktree, timeout, store)
                        .map_err(|error| error.to_string())?;
                (output, digest)
            }
            None => (
                immortality_tools::run_command(command, &self.checks.worktree, timeout)
                    .map_err(|error| error.to_string())?,
                None,
            ),
        };
        Ok(CommandRun {
            exit_code: output.exit_code,
            timed_out: output.timed_out,
            duration_ms: u64::try_from(output.duration.as_millis()).unwrap_or(u64::MAX),
            output_tail: output.tail,
            stdout_lines: output.stdout_lines,
            stderr_lines: output.stderr_lines,
            artifact,
        })
    }
}

/// A synthetic diff for a file git does not know about.
///
/// `git diff --no-index` against `/dev/null` is the honest way to get one: it produces the same
/// headers as a real addition, including the binary marker, without this module having to
/// reimplement unified-diff generation. It exits 1 for a difference, which is the normal case.
fn untracked_diff(repo: &GitRepo, path: &str) -> String {
    let null = if cfg!(windows) { "NUL" } else { "/dev/null" };
    // Exit 1 means "there is a difference", which `git()` reports as an error because it only
    // returns stdout on success. Asking for the diff again through a form that succeeds keeps the
    // file in the review instead of dropping it over a status code.
    repo.git(&["diff", "--no-color", "--no-index", null, path])
        .unwrap_or_else(|_| fallback_addition(repo, path))
}

/// The addition as a minimal diff, for when `--no-index` reported a difference as an error.
fn fallback_addition(repo: &GitRepo, path: &str) -> String {
    use std::fmt::Write as _;

    let full = repo.worktree().join(path);
    let Ok(bytes) = std::fs::read(&full) else {
        return String::new();
    };
    let mut out = String::new();
    let _ = writeln!(out, "diff --git a/{path} b/{path}");
    let _ = writeln!(out, "new file mode 100644");
    if immortality_tools::looks_binary(&bytes) {
        let _ = writeln!(out, "Binary files /dev/null and b/{path} differ");
        return out;
    }
    let _ = writeln!(out, "--- /dev/null");
    let _ = writeln!(out, "+++ b/{path}");
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    let _ = writeln!(out, "@@ -0,0 +1,{} @@", lines.len());
    for line in lines {
        let _ = writeln!(out, "+{line}");
    }
    out
}

/// The default answer budget for [`ModelReviewer`].
///
/// # The defect this replaces
///
/// The reviewer used to cap itself at 1024 tokens, the number that fits a tool call. A reviewer is
/// not a tool call: it is handed the definition of done and a diff and asked to explain itself, so
/// a large subtask produces a long answer. Measured on `qna_rails` (2026-11-16): the reviewer
/// stopped at `finish_reason=length` after 3727 characters of content, the gate turned that into
/// `review_rejected: the reviewer could not be called`, and the subtask was blocked — with its
/// document written and its checks green. The truncation was invisible in the verdict: an input
/// that was merely too long to review reads exactly like a change that failed review.
///
/// 4096 rather than the executor's 8192: a review is one argument, not a file being written, and a
/// reviewer that needs eight thousand tokens is describing a change nobody wants merged.
pub const REVIEW_MAX_TOKENS: u32 = 4096;

/// The production [`Reviewer`]: one model call, no tools.
///
/// The absence of tools is the enforcement of "the reviewer cannot edit code". A sentence in the
/// prompt asking it not to is a request; a request with no tool definitions is a fact, and the
/// difference matters because a reviewer that *could* edit could approve a change by making it.
#[derive(Debug)]
pub struct ModelReviewer {
    model: Arc<dyn Model>,
    model_name: String,
    /// Ceiling on the answer. A verdict is a few tokens; the budget exists so a model that starts
    /// explaining cannot spend a pass on it.
    max_tokens: u32,
}

impl ModelReviewer {
    /// A reviewer over an existing model client.
    #[must_use]
    pub fn new(model: Arc<dyn Model>, model_name: impl Into<String>) -> Self {
        Self {
            model,
            model_name: model_name.into(),
            max_tokens: REVIEW_MAX_TOKENS,
        }
    }

    /// A reviewer whose answer is capped at `max_tokens`.
    ///
    /// The cap is the operator's, not this type's: a reviewer asked to judge a large diff writes
    /// free prose, and the number that fits a tool call does not fit it. See
    /// [`REVIEW_MAX_TOKENS`] for the defect.
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

impl Reviewer for ModelReviewer {
    fn review(&self, request: &ReviewRequest) -> Result<String, String> {
        let prompt = immortality_core::gate::review_prompt(request);
        let chat = immortality_llm::wire::ChatRequest::new(self.model_name.clone())
            .with_system(immortality_core::gate::REVIEW_SYSTEM)
            .with_max_tokens(self.max_tokens)
            .with(immortality_llm::wire::Message::user(prompt));
        self.model.complete(&chat).map(|reply| reply.content)
    }
}

/// The commit a task's work forked from, used as the task-level gate's diff base.
///
/// The task-level review used to diff against `HEAD` and see the whole task's work, because the
/// work was uncommitted. Committing each closed subtask to `agent/<task>` moved that work into
/// `HEAD`, and the same diff became empty — measured on a live `word-count` run: every subtask
/// closed green, `agent/<task>` carried three commits, and the task verdict was
/// `review_rejected: the diff is empty`. A reviewer that approves an empty diff approves nothing,
/// so the task-level gate needs the fork point instead.
///
/// `None` when either endpoint is unknown — a repository with no commits, or a branch git cannot
/// resolve. The caller then keeps `HEAD`, which is what the gate did before and is never wider
/// than safe.
#[must_use]
pub fn task_base(worktree: &GitRepo, repository: &Path, branch: &str) -> Option<String> {
    let fork = GitRepo::new(repository).head().ok().flatten()?;
    let merged = worktree
        .git(&["merge-base", &fork, branch])
        .ok()?
        .trim()
        .to_string();
    (!merged.is_empty()).then_some(merged)
}

/// Read the base commit of an attempt, if the journal knows one.
///
/// Kept as a separate function so the caller decides what "no base" means. `None` here makes the
/// diff against `HEAD`, which is broader and therefore safe.
#[must_use]
pub fn attempt_base(checkpoint: Option<&str>) -> Option<String> {
    checkpoint.map(str::to_string)
}

/// The worktree path a gate evaluation runs in.
#[must_use]
pub fn worktree_of(gate: &PolicyGate) -> &Path {
    gate.worktree()
}

#[cfg(test)]
mod tests {
    use super::*;
    use immortality_core::config::Policy;
    use std::process::Command;

    fn repo() -> (tempfile::TempDir, GitRepo) {
        let dir = tempfile::tempdir().expect("tempdir");
        let repo = GitRepo::new(dir.path());
        for args in [
            vec!["init", "-q"],
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
        std::fs::write(dir.path().join("a.txt"), "one\n").expect("write");
        let status = Command::new("git")
            .arg("-C")
            .arg(dir.path())
            .args(["add", "-A"])
            .status()
            .expect("git add");
        assert!(status.success());
        let status = Command::new("git")
            .arg("-C")
            .arg(dir.path())
            .args(["commit", "-q", "-m", "base"])
            .status()
            .expect("git commit");
        assert!(status.success());
        (dir, repo)
    }

    /// Run git in `dir`, failing the test with git's own words if it refuses.
    fn git(dir: &Path, args: &[&str]) {
        let output = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .env("GIT_TERMINAL_PROMPT", "0")
            .output()
            .expect("git");
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn gates(worktree: &Path) -> RealGates {
        let gate = PolicyGate::new(
            worktree,
            &["**".to_string()],
            &[],
            Policy {
                workspace_root: worktree.to_path_buf(),
                ..Policy::default()
            },
        )
        .expect("gate");
        RealGates::new(
            worktree,
            Arc::new(gate),
            CheckRunner {
                worktree: worktree.to_path_buf(),
                timeout: Duration::from_secs(10),
                // No artifact store in a test: `None` is the honest answer, and the check still
                // runs and reports its exit code.
                artifacts: None,
            },
            None,
        )
    }

    #[test]
    fn a_tracked_modification_reaches_the_diff() {
        let (dir, _) = repo();
        std::fs::write(dir.path().join("a.txt"), "one\ntwo\n").expect("write");
        let diff = gates(dir.path()).diff(None).expect("diff");
        assert_eq!(diff.files.len(), 1);
        assert_eq!(diff.files[0].path, PathBuf::from("a.txt"));
        assert!(diff.files[0].added_lines.iter().any(|line| line == "two"));
    }

    #[test]
    fn an_untracked_file_reaches_the_diff_as_an_addition() {
        // The common case for an agent that wrote a module and has not committed: a diff that
        // ignored it would approve exactly the file nobody reviewed.
        let (dir, _) = repo();
        std::fs::write(dir.path().join("new.rs"), "fn new() {}\n").expect("write");
        let diff = gates(dir.path()).diff(None).expect("diff");
        let file = diff
            .files
            .iter()
            .find(|file| file.path == Path::new("new.rs"))
            .expect("the new file");
        assert_eq!(file.status, immortality_core::gate::FileStatus::Added);
        assert!(
            file.added_lines.iter().any(|line| line.contains("fn new")),
            "{file:?}"
        );
    }

    #[test]
    fn an_untracked_binary_is_flagged_rather_than_dropped() {
        let (dir, _) = repo();
        std::fs::write(dir.path().join("logo.png"), b"\x89PNG\r\n\x1a\n\0\0").expect("write");
        let diff = gates(dir.path()).diff(None).expect("diff");
        let file = diff
            .files
            .iter()
            .find(|file| file.path == Path::new("logo.png"))
            .expect("the binary");
        assert!(file.binary, "{file:?}");
    }

    #[test]
    fn a_clean_tree_has_an_empty_diff() {
        let (dir, _) = repo();
        let diff = gates(dir.path()).diff(None).expect("diff");
        assert!(diff.is_empty(), "{diff:?}");
    }

    #[test]
    fn the_base_commit_scopes_the_diff() {
        // With a base, the diff shows what happened since it — which is what a reviewer of *this
        // attempt* should read, not the whole task.
        let (dir, repo) = repo();
        let base = repo.head().expect("head").expect("a commit");
        std::fs::write(dir.path().join("a.txt"), "one\ntwo\n").expect("write");
        let mut scoped = gates(dir.path());
        scoped.base = Some(base);
        assert_eq!(scoped.diff(None).expect("diff").files.len(), 1);
    }

    #[test]
    fn the_task_base_shows_a_reviewer_the_work_that_was_committed() {
        // Measured on a live `word-count` run: every subtask closed green, `agent/<task>` carried
        // three commits, and the task-level verdict was `review_rejected` because the diff against
        // `HEAD` was empty — the work was *in* `HEAD`. A reviewer approving an empty diff approves
        // nothing, so the task gate has to diff from the fork point.
        //
        // Two checkouts, as in production: the repository a human merges into, and the task's
        // worktree. One directory would put both `HEAD`s on the same branch and answer the wrong
        // question.
        let dir = tempfile::tempdir().expect("tempdir");
        let repository = dir.path().join("repo");
        let worktree = dir.path().join("wt");
        std::fs::create_dir_all(&repository).expect("repo dir");
        git(&repository, &["init", "-q", "-b", "main"]);
        git(&repository, &["config", "user.email", "test@example.com"]);
        git(&repository, &["config", "user.name", "test"]);
        std::fs::write(repository.join("a.txt"), "one\n").expect("write");
        git(&repository, &["add", "-A"]);
        git(&repository, &["commit", "-q", "-m", "chore: base"]);
        git(
            &repository,
            &[
                "worktree",
                "add",
                "-q",
                worktree.to_str().expect("utf-8"),
                "-b",
                "agent/tsk_base",
            ],
        );
        std::fs::write(worktree.join("a.txt"), "one\ntwo\n").expect("write");
        git(&worktree, &["add", "-A"]);
        git(&worktree, &["commit", "-q", "-m", "chore: the task's work"]);

        // Against `HEAD` there is nothing to review, which is the bug.
        assert!(
            gates(&worktree).diff(None).expect("diff").is_empty(),
            "the work is committed, so `HEAD` alone shows nothing"
        );

        // The fork point is where the task forked from, so the committed work is the diff.
        let base = task_base(&GitRepo::new(&worktree), &repository, "agent/tsk_base").expect("a fork point");
        let mut scoped = gates(&worktree);
        scoped.base = Some(base);
        let diff = scoped.diff(None).expect("diff");
        assert_eq!(diff.files.len(), 1, "{diff:?}");
        assert_eq!(diff.files[0].path, PathBuf::from("a.txt"));
        assert!(diff.files[0].added_lines.iter().any(|line| line == "two"));
    }

    #[test]
    fn a_command_runs_in_the_worktree() {
        let (dir, _) = repo();
        let run = gates(dir.path())
            .run("test -f a.txt && echo ok", None)
            .expect("run");
        assert_eq!(run.exit_code, Some(0));
        assert!(run.output_tail.contains("ok"), "{run:?}");
    }

    #[test]
    fn a_check_deadline_overrides_the_runner_default_for_that_command_only() {
        // The `black_jack` defect end to end: the runner's default is the 120 s tool default, and a
        // check that declares its own longer deadline has to get it while its neighbours keep the
        // default. Here the direction is inverted on purpose — a one-second deadline against a
        // command that sleeps — because a short deadline is observable in a test and a nine-minute
        // one is not.
        let (dir, _) = repo();
        let runner = gates(dir.path());
        let configured = runner.checks.timeout;
        assert_eq!(configured, Duration::from_secs(10), "the fixture's default");

        // No declared deadline: the configured default applies and a quick command passes.
        let plain = runner.run("true", None).expect("run");
        assert!(!plain.timed_out, "{plain:?}");
        assert_eq!(plain.exit_code, Some(0));

        // A declared deadline wins over the default, even when it is far shorter.
        let run = runner.run("sleep 5", Some(Duration::from_secs(1))).expect("run");
        assert!(run.timed_out, "{run:?}");
        assert_eq!(run.exit_code, None);
        // And the override did not mutate the runner: the next check gets the default back.
        assert_eq!(runner.checks.timeout, configured);
    }

    #[test]
    fn a_denied_command_is_not_authorised() {
        // `sudo` is external, and external is not a yes: a gate command waits for nobody.
        let (dir, _) = repo();
        let error = gates(dir.path())
            .authorise("sudo cargo test")
            .expect_err("external");
        assert!(error.contains("sudo") || error.contains("approval"), "{error}");
    }

    #[test]
    fn an_ordinary_build_command_is_authorised() {
        let (dir, _) = repo();
        gates(dir.path())
            .authorise("cargo test --workspace")
            .expect("allowed");
    }

    #[test]
    fn no_base_means_head() {
        assert_eq!(attempt_base(None), None);
        assert_eq!(attempt_base(Some("abc123")), Some("abc123".to_string()));
    }

    /// A model that answers "approved" and remembers what it was asked.
    #[derive(Debug)]
    struct Recorder {
        asked: std::sync::Mutex<Vec<immortality_llm::wire::ChatRequest>>,
    }

    impl Model for Recorder {
        fn complete(
            &self,
            request: &immortality_llm::wire::ChatRequest,
        ) -> Result<crate::runner::Reply, String> {
            self.asked.lock().expect("lock").push(request.clone());
            Ok(crate::runner::Reply::text("APPROVED"))
        }
    }

    #[test]
    fn the_reviewers_budget_is_the_operators_not_a_tool_calls() {
        // The defect: the reviewer capped itself at 1024 tokens, the number that fits a tool call.
        // On `qna_rails` it stopped at `finish_reason=length` after 3727 characters, the gate read
        // that as `review_rejected: the reviewer could not be called`, and a subtask with its
        // document written and its checks green was blocked. A too-long input and a change that
        // failed review must not produce the same verdict.
        let recorder = Arc::new(Recorder {
            asked: std::sync::Mutex::new(Vec::new()),
        });

        // The default is the reviewer's own, and it is bigger than a tool call's.
        let reviewer = ModelReviewer::new(Arc::clone(&recorder) as Arc<dyn Model>, "utility");
        assert!(
            reviewer.max_tokens > immortality_llm::wire::DEFAULT_MAX_TOKENS,
            "a review is prose and cannot be capped at a tool call's budget"
        );
        reviewer
            .review(&ReviewRequest {
                statement: "docs/services.md exists".to_string(),
                diff: immortality_core::gate::Diff::default(),
            })
            .expect("review");
        {
            let asked = recorder.asked.lock().expect("lock");
            assert_eq!(asked.len(), 1);
            assert_eq!(asked[0].max_tokens, Some(REVIEW_MAX_TOKENS));
        }

        // And the operator's field reaches this type, so one number sizes every completion.
        ModelReviewer::new(Arc::clone(&recorder) as Arc<dyn Model>, "utility")
            .with_max_tokens(20_000)
            .review(&ReviewRequest {
                statement: "docs/services.md exists".to_string(),
                diff: immortality_core::gate::Diff::default(),
            })
            .expect("review");
        let asked = recorder.asked.lock().expect("lock");
        assert_eq!(asked[1].max_tokens, Some(20_000));
    }
}
