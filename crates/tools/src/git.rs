//! `git` — one tool, one subcommand argument, and no second git layer.
//!
//! # Why a single tool rather than eight
//!
//! The design lists `git status/diff/log/add/commit/branch/worktree/stash` as separate rows, but
//! they share one argument shape, one effect table and one failure mode. Eight tools would mean
//! eight schemas for a model to get wrong and eight places to keep the "no `push`" rule. One
//! tool with a `subcommand` enum keeps the rule in one match arm.
//!
//! # Why `push` and `fetch` are not subcommands
//!
//! They are `Effect::External` (the classifier already says so) and the design's rule is that a
//! human does the pushing and merging. A tool that could reach a remote would make the rule a
//! convention; leaving them out of the schema makes it structural. A model that wants them has
//! `shell`, where the classifier escalates.
//!
//! # Why `commit` is `Unknown` and the others are not
//!
//! A crash between `git commit` and the journal's `completed` record leaves a commit nobody
//! recorded. Replaying it makes a second commit, so recovery must probe. The probe is a trailer:
//! the commit message carries `Immortality-Call: <call-id>`, and finding that string proves the
//! commit happened. `add` and `branch` are idempotent — running them twice is running them once.
//!
//! # Why the worktree path is checked before the verb runs
//!
//! Every command is `git -C <worktree>`, and the worktree comes from the gate, never from the
//! arguments. A model that could name the directory could operate on the user's own checkout,
//! which is the one thing the per-task worktree exists to prevent.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::OnceLock;

use immortality_core::checkpoint::GitRepo;
use immortality_core::protocol::Idempotency;
use serde_json::{Value, json};

use crate::Effect;
use crate::args::Args;
use crate::policy::PathUse;
use crate::tool::{Tool, ToolCtx, ToolError, ToolResult};

/// Largest diff shown before the model is told to narrow it.
///
/// A `diff` of a whole dependency update is not information, it is a wall. The cap is in lines
/// rather than bytes because a model reads lines and a single minified line would otherwise blow
/// through any byte budget as one enormous row.
pub const MAX_DIFF_LINES: usize = 400;

/// Stash subcommands that delete an entry, refused because `refs/stash` is shared with the
/// human's checkout. `pop` is here because it is `apply` + `drop`.
const STASH_REMOVERS: &[&str] = &["clear", "drop", "pop"];

/// Subcommands a model may call.
///
/// Deliberately closed. `push`, `fetch`, `pull`, `clone`, `remote` and `submodule` are absent
/// because they reach the network, and `config` is absent because `.git/config` is on the
/// built-in deny list — a tool that could write it would be a hole in every other rule at once.
pub const SUBCOMMANDS: &[&str] = &[
    "status",
    "diff",
    "log",
    "show",
    "add",
    "commit",
    "branch",
    "worktree",
    "stash",
    "rev_parse",
];

fn git_schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "subcommand": {
                    "type": "string",
                    "enum": SUBCOMMANDS,
                    "description": "Which git verb to run. `push` and `fetch` are not available: the human does those."
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arguments passed to the verb, in order. No shell quoting: each entry is one argv element."
                },
                "message": {
                    "type": "string",
                    "description": "Commit message. Required for `commit`; must start with a Conventional Commit type (`feat:`, `fix:`, `chore:` …)."
                },
                "path": {
                    "type": "string",
                    "description": "Worktree-relative path to stage. Defaults to the whole tree for `add`."
                },
                "reason": {
                    "type": "string",
                    "description": "Why the operation is being done. Recorded on hidden checkpoints and for the human reading the log."
                }
            },
            "required": ["subcommand"],
            "additionalProperties": false
        })
    })
}

/// One git verb.
///
/// The mapping from the model's `subcommand` string is explicit rather than a direct passthrough,
/// so a name that is not in the list cannot become an argument to git. Validating against the
/// schema would be enough in the normal path, but a tool that takes a path straight from
/// `Args` is a tool that ran without a check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verb {
    Status,
    Diff,
    Log,
    Show,
    Add,
    Commit,
    Branch,
    Worktree,
    Stash,
    RevParse,
}

impl Verb {
    /// Parse the model's subcommand name.
    ///
    /// # Errors
    /// The name is not one of [`SUBCOMMANDS`].
    fn parse(name: &str) -> Result<Self, ToolError> {
        Ok(match name {
            "status" => Self::Status,
            "diff" => Self::Diff,
            "log" => Self::Log,
            "show" => Self::Show,
            "add" => Self::Add,
            "commit" => Self::Commit,
            "branch" => Self::Branch,
            "worktree" => Self::Worktree,
            "stash" => Self::Stash,
            "rev_parse" => Self::RevParse,
            other => {
                return Err(ToolError::Arguments(format!(
                    "`{other}` is not an available git subcommand. Available: {}. \
                     `push` and `fetch` are deliberately absent: a human performs those.",
                    SUBCOMMANDS.join(", ")
                )));
            }
        })
    }

    /// What this verb can change, for the policy gate.
    const fn effect(self) -> Effect {
        match self {
            // Listing is a read; the model's `args` cannot promote it, because the promotion is
            // decided here rather than parsed out of the arguments.
            Self::Status | Self::Diff | Self::Log | Self::Show | Self::RevParse => Effect::Read,
            Self::Add | Self::Commit | Self::Branch | Self::Worktree | Self::Stash => Effect::Mutate,
        }
    }
}

/// Run a git command in the task worktree.
///
/// Carries nothing. The directory comes from [`ToolCtx::gate`] at execution time, so there is no
/// stored path that could disagree with the one the policy checked — the same reason
/// [`ToolCtx`] hands out resolved paths rather than letting tools build their own.
#[derive(Debug, Clone, Copy, Default)]
pub struct Git;

impl Git {
    /// The tool. It has no state, so this exists only so `Registry::new` can take a value.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    fn repo(ctx: &ToolCtx) -> GitRepo {
        GitRepo::new(ctx.gate().worktree().to_path_buf())
    }
}

impl Tool for Git {
    fn name(&self) -> &'static str {
        "git"
    }

    fn schema(&self) -> &'static Value {
        git_schema()
    }

    fn description(&self) -> &'static str {
        "Run a git command in the task worktree. `push`/`fetch` are not available. `commit` needs a Conventional Commit `message`."
    }

    fn effect(&self, args: &Args) -> Effect {
        // An unreadable subcommand is treated as the most dangerous case rather than the safest:
        // a policy that defaults to `Read` for a typo is a policy that lets the typo through.
        args.str("subcommand")
            .ok()
            .and_then(|name| Verb::parse(name).ok())
            .map_or(Effect::External, Verb::effect)
    }

    /// `Unknown` for `commit` only, and that is the whole reason the trailer exists.
    fn idempotency(&self, args: &Args) -> Idempotency {
        match args
            .str("subcommand")
            .ok()
            .and_then(|name| Verb::parse(name).ok())
        {
            // A second `add` is a first `add`; a second `branch -b` fails; a second `stash` is a
            // second stash, but it is also harmless in the sense that no work is lost. Only
            // `commit` produces something a replay would duplicate and nothing could undo.
            Some(Verb::Commit) => Idempotency::Unknown,
            // An unknown verb is never executed at all, so its idempotency decides nothing; it
            // lands here rather than in an arm of its own.
            Some(_) | None => Idempotency::Idempotent,
        }
    }

    fn paths(&self, args: &Args) -> Vec<PathUse> {
        // `add` and `commit` touch the tree; the rest touch refs, which the gate cannot see and
        // which the effect classification covers. Naming a path only when one was given keeps
        // `git status` from being rejected for a path it does not have.
        args.opt_str("path")
            .ok()
            .flatten()
            .map(|path| vec![PathUse::write(path)])
            .unwrap_or_default()
    }

    fn execute(&self, ctx: &ToolCtx, args: Args) -> Result<ToolResult, ToolError> {
        let verb = Verb::parse(args.str("subcommand")?)?;
        let extra = read_args(&args)?;

        // The gate sees the effect of the verb, not the verb's arguments: `git branch -D x` and
        // `git branch` are both `Branch`, and one of them deletes refs. Both are `Mutate`, which
        // is the conservative answer for the pair.
        match ctx.gate().check(verb.effect(), &self.paths(&args)) {
            crate::Verdict::Allowed => {}
            crate::Verdict::Denied(denial) => return Err(ToolError::Refused(denial.observation())),
            crate::Verdict::Escalated(escalation) => {
                return Err(ToolError::Refused(escalation.observation()));
            }
        }

        let repo = Self::repo(ctx);
        if !repo.is_repository() {
            return Err(ToolError::Refused(
                json!({
                    "error": "not_a_repository",
                    "worktree": repo.worktree().display().to_string(),
                    "hint": "The task worktree is not a git repository. Report this; do not run \
                             `git init` over it."
                })
                .to_string(),
            ));
        }

        match verb {
            Verb::Status => status(ctx, &repo, &extra),
            Verb::Diff => diff(ctx, &repo, &extra),
            Verb::Log => log(ctx, &repo, &extra),
            Verb::Show => show(ctx, &repo, &extra),
            Verb::Add => add(ctx, &repo, &args, &extra),
            Verb::Commit => commit(ctx, &repo, &args, &extra),
            Verb::Branch => branch(&repo, &extra),
            Verb::Worktree => worktree(&repo, &extra),
            Verb::Stash => stash(ctx, &repo, &args, &extra),
            Verb::RevParse => rev_parse(&repo, &extra),
        }
    }
}

/// The model's `args` array, as borrowed strings.
fn read_args(args: &Args) -> Result<Vec<String>, ToolError> {
    let Some(value) = args.raw().get("args") else {
        return Ok(Vec::new());
    };
    let Some(items) = value.as_array() else {
        return Err(ToolError::Arguments(
            "`args` must be an array of strings".to_string(),
        ));
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let Some(text) = item.as_str() else {
            return Err(ToolError::Arguments(
                "`args` must contain only strings; each entry is one argv element and needs no \
                 quoting"
                    .to_string(),
            ));
        };
        out.push(text.to_string());
    }
    Ok(out)
}

/// Turn a git failure into a refusal the model can act on.
///
/// A refusal rather than an `Io` error: git failing is the normal case — the branch is gone, the
/// path is untracked — and the model needs the message, not a crash.
fn refused(verb: &str, error: &immortality_core::checkpoint::GitError) -> ToolError {
    ToolError::Refused(
        json!({
            "error": "git_failed",
            "subcommand": verb,
            "git_command": error.command,
            "message": error.message,
        })
        .to_string(),
    )
}

/// Cap a block of text at [`MAX_DIFF_LINES`], saying how much was dropped.
fn cap(text: &str, limit: usize) -> (String, usize) {
    let all: Vec<&str> = text.lines().collect();
    if all.len() <= limit {
        return (text.to_string(), 0);
    }
    let kept = all.get(..limit).unwrap_or_default().join("\n");
    (kept, all.len() - limit)
}

fn status(ctx: &ToolCtx, repo: &GitRepo, extra: &[String]) -> Result<ToolResult, ToolError> {
    let mut args = vec!["status", "--porcelain=v1", "--untracked-files=all", "--branch"];
    args.extend(extra.iter().map(String::as_str));
    let out = repo.git(&args).map_err(|error| refused("status", &error))?;
    let branch = out
        .lines()
        .find_map(|line| line.strip_prefix("## "))
        .map(str::to_string);
    let changed: Vec<&str> = out
        .lines()
        .filter(|line| !line.starts_with("## "))
        .filter(|line| !line.trim().is_empty())
        .collect();

    let mut observation = String::new();
    match &branch {
        Some(branch) => {
            let _ = writeln!(observation, "On {branch}.");
        }
        None => observation.push_str("No branch reported.\n"),
    }
    if changed.is_empty() {
        observation.push_str("Working tree clean.");
    } else {
        let shown: Vec<&str> = changed
            .iter()
            .take(ctx.budget().max_bytes / 40)
            .copied()
            .collect();
        for line in &shown {
            let _ = writeln!(observation, "{line}");
        }
        if changed.len() > shown.len() {
            let _ = writeln!(
                observation,
                "… {} more changed path(s) not shown.",
                changed.len() - shown.len()
            );
        }
    }

    Ok(ToolResult::new(
        observation,
        json!({
            "branch": branch,
            "clean": changed.is_empty(),
            "changed": changed,
        }),
    ))
}

fn diff(ctx: &ToolCtx, repo: &GitRepo, extra: &[String]) -> Result<ToolResult, ToolError> {
    let mut args = vec!["diff", "--no-color"];
    args.extend(extra.iter().map(String::as_str));
    let out = repo.git(&args).map_err(|error| refused("diff", &error))?;
    Ok(output_capped(ctx, "diff", &out, json!({"paths": extra})))
}

fn log(ctx: &ToolCtx, repo: &GitRepo, extra: &[String]) -> Result<ToolResult, ToolError> {
    // `--no-pager` is already implied by `GIT_PAGER=cat` in the shared spawner; the explicit
    // format is what makes the output one line per commit instead of a multi-line block whose
    // width depends on the terminal git thinks it has.
    let mut args = vec!["log", "--no-color", "--format=%h %ad %s", "--date=short"];
    if extra.is_empty() {
        args.push("-n");
        args.push("20");
    }
    args.extend(extra.iter().map(String::as_str));
    let out = repo.git(&args).map_err(|error| refused("log", &error))?;
    Ok(output_capped(ctx, "log", &out, json!({"args": extra})))
}

fn show(ctx: &ToolCtx, repo: &GitRepo, extra: &[String]) -> Result<ToolResult, ToolError> {
    let mut args = vec!["show", "--no-color", "--stat"];
    args.extend(extra.iter().map(String::as_str));
    let out = repo.git(&args).map_err(|error| refused("show", &error))?;
    Ok(output_capped(ctx, "show", &out, json!({"args": extra})))
}

fn add(ctx: &ToolCtx, repo: &GitRepo, args: &Args, extra: &[String]) -> Result<ToolResult, ToolError> {
    // A `path` and extra args are not both accepted: `git add -- <path> <other>` would stage a
    // path the gate never checked.
    if args.opt_str("path")?.is_some() && !extra.is_empty() {
        return Err(ToolError::Arguments(
            "pass either `path` or `args`, not both: the path is policy-checked and extra \
             arguments are not"
                .to_string(),
        ));
    }
    if let Some(raw) = args.opt_str("path")? {
        let resolved = ctx.write_path(raw)?;
        let relative = relative_to(repo.worktree(), &resolved);
        repo.add_path(&relative).map_err(|error| refused("add", &error))?;
        return Ok(ToolResult::new(
            format!("Staged `{relative}`."),
            json!({"staged": "path", "path": relative}),
        ));
    }
    if extra.is_empty() {
        repo.add_all().map_err(|error| refused("add", &error))?;
        return Ok(ToolResult::new(
            "Staged every change, deletions and untracked files included.",
            json!({"staged": "all"}),
        ));
    }
    let mut all = vec!["add", "--"];
    all.extend(extra.iter().map(String::as_str));
    repo.git(&all).map_err(|error| refused("add", &error))?;
    Ok(ToolResult::new(
        format!("Staged {}.", extra.join(" ")),
        json!({"staged": extra}),
    ))
}

fn commit(ctx: &ToolCtx, repo: &GitRepo, args: &Args, extra: &[String]) -> Result<ToolResult, ToolError> {
    let message = args.str("message")?.trim().to_string();
    if message.is_empty() {
        return Err(ToolError::Arguments(
            "`message` is required for `commit`; an empty message makes the commit ".to_string(),
        ));
    }
    if !is_conventional(&message) {
        return Err(ToolError::Arguments(format!(
            "`message` must start with a Conventional Commit type, e.g. `feat: …`, `fix: …`, \
             `chore: …`. Got: {message:?}"
        )));
    }
    // Any extra arguments are refused rather than forwarded: `commit --amend` and `commit -a`
    // bypass the index the gate just reasoned about, and `--no-verify` is already applied.
    if !extra.is_empty() {
        return Err(ToolError::Arguments(
            "`commit` takes no extra `args`; use `add` first. `--amend` and `-a` are not \
             available because they change what a probe can prove."
                .to_string(),
        ));
    }

    // Stage everything, so a model's `write_file` + `git commit` sequence works without a
    // separate `add`. The alternative — commit an empty index — fails for a reason that has
    // nothing to do with what the model asked for.
    repo.add_all().map_err(|error| refused("commit", &error))?;

    // The trailer the probe searches for. Without a call id (a direct test call, or a tool used
    // outside the daemon) there is nothing to stamp, and the commit is still correct — it is
    // only its *recoverability* that needs the id.
    let call = ctx.call().map(str::to_string);
    let trailers: Vec<(&str, &str)> = call
        .as_deref()
        .map(|id| vec![("Immortality-Call", id)])
        .unwrap_or_default();
    let sha = repo
        .commit_on_current(&message, &trailers)
        .map_err(|error| refused("commit", &error))?;

    let Some(sha) = sha else {
        return Ok(ToolResult::new(
            "Nothing to commit: the working tree matches the index.",
            // `committed: false` rather than an error: an agent that stages nothing and commits
            // nothing has learned something true, and turning it into a failure would make it
            // retry.
            json!({"committed": false, "reason": "nothing_to_commit"}),
        ));
    };
    Ok(ToolResult::new(
        format!("Committed {sha}: {message}"),
        json!({
            "committed": true,
            "sha": sha,
            "message": message,
            "call": call,
        }),
    ))
}

fn branch(repo: &GitRepo, extra: &[String]) -> Result<ToolResult, ToolError> {
    refuse_branch_hazards(extra)?;
    let mut args = vec!["branch", "--no-color"];
    args.extend(extra.iter().map(String::as_str));
    let out = repo.git(&args).map_err(|error| refused("branch", &error))?;
    let branches: Vec<&str> = out
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    Ok(ToolResult::new(
        if branches.is_empty() {
            "No branches.".to_string()
        } else {
            branches.join("\n")
        },
        json!({"branches": branches}),
    ))
}

/// Refuse the branch operations that reach outside this task's own ref.
///
/// A worktree shares `refs/heads` with the repository it was added to — measured, not assumed:
/// `git branch -D feature-x` run from a task worktree deleted the human's `feature-x`, and
/// `git stash clear` emptied a stash list the human also sees. Git protects only branches that
/// are *checked out* somewhere, so `main` is safe by accident while everything else is not.
///
/// Listing and creating are kept. Creating writes one new ref that nothing else points at, which
/// is what a task branch needs and what recovery can identify. Everything that moves, deletes or
/// repoints an existing ref is refused, because an agent that can delete a branch can destroy
/// work that no checkpoint in this task ever recorded.
fn refuse_branch_hazards(extra: &[String]) -> Result<(), ToolError> {
    const HAZARDS: &[&str] = &[
        "-d",
        "-D",
        "--delete",
        "-m",
        "-M",
        "--move",
        "-f",
        "--force",
        "-u",
        "--set-upstream-to",
    ];
    let Some(hazard) = extra
        .iter()
        .find(|arg| HAZARDS.contains(&arg.as_str()) || arg.starts_with("--set-upstream-to="))
    else {
        return Ok(());
    };
    Err(ToolError::Refused(
        json!({
            "error": "branch_ref_out_of_scope",
            "flag": hazard,
            "hint": "Branches are shared with the human's checkout. Only listing and creating a \
                     new branch are allowed from a task worktree; deleting or moving a ref is \
                     an operator action.",
        })
        .to_string(),
    ))
}

fn worktree(repo: &GitRepo, extra: &[String]) -> Result<ToolResult, ToolError> {
    // `worktree add` from inside a worktree would create a worktree of a worktree, which is how
    // a repository acquires a directory nobody can find again. Listing is safe; anything else
    // is the daemon's job, through `WorktreeManager`.
    let listing = extra.is_empty() || extra.first().is_some_and(|arg| arg == "list");
    if !listing {
        return Err(ToolError::Refused(
            json!({
                "error": "worktree_managed_by_daemon",
                "hint": "Only `worktree list` is available. Creating and removing task worktrees \
                         is the daemon's responsibility, so that recovery can reason about them."
            })
            .to_string(),
        ));
    }
    let out = repo
        .git(&["worktree", "list", "--porcelain"])
        .map_err(|error| refused("worktree", &error))?;
    let entries: Vec<&str> = out
        .lines()
        .filter_map(|line| line.strip_prefix("worktree "))
        .collect();
    let mut observation = String::new();
    for entry in &entries {
        let _ = writeln!(observation, "{entry}");
    }
    Ok(ToolResult::new(observation, json!({"worktrees": entries})))
}

fn stash(ctx: &ToolCtx, repo: &GitRepo, args: &Args, extra: &[String]) -> Result<ToolResult, ToolError> {
    // The stash message carries the call id, so a crash between `stash push` and its journal
    // record leaves something a probe can find. `git stash list` is a read and takes none.
    let reason = args.opt_str("reason")?.unwrap_or("").trim();
    let call = ctx.call();
    let mut message = String::new();
    if !reason.is_empty() {
        message.push_str(reason);
    }
    if let Some(call) = call {
        if !message.is_empty() {
            message.push(' ');
        }
        let _ = write!(message, "[{call}]");
    }
    // Every stash entry lives in `refs/stash`, which the human's checkout sees too — measured:
    // a stash pushed from a task worktree appears in `git stash list` run in the main checkout.
    // So the operations that *remove* an entry are refused and only those that leave it behind
    // are allowed. `pop` is in the refused set because it is `apply` + `drop`, and bare `pop`
    // applies whichever entry is on top — which may not be the agent's.
    if let Some(verb) = extra.iter().find(|arg| STASH_REMOVERS.contains(&arg.as_str())) {
        return Err(ToolError::Refused(
            json!({
                "error": "stash_entry_out_of_scope",
                "verb": verb,
                "hint": "The stash is shared with the human's checkout and nothing here records \
                         what an entry held. `apply` restores the work and keeps the entry; \
                         removing one is an operator action.",
            })
            .to_string(),
        ));
    }
    let mut all: Vec<String> = vec!["stash".to_string()];
    all.extend(extra.iter().cloned());
    let keeps_work = !all
        .iter()
        .any(|arg| arg == "list" || arg == "show" || arg == "drop");
    if !message.is_empty() && keeps_work {
        all.push("-m".to_string());
        // Cloned because the message is also reported back in the structured result, and the
        // caller of a `stash push` needs to know the exact string a probe would search for.
        all.push(message.clone());
    }
    let borrowed: Vec<&str> = all.iter().map(String::as_str).collect();
    let out = repo.git(&borrowed).map_err(|error| refused("stash", &error))?;
    Ok(output_capped(
        ctx,
        "stash",
        &out,
        json!({"args": extra, "message": message}),
    ))
}

fn rev_parse(repo: &GitRepo, extra: &[String]) -> Result<ToolResult, ToolError> {
    let mut args = vec!["rev-parse"];
    args.extend(extra.iter().map(String::as_str));
    let out = repo.git(&args).map_err(|error| refused("rev-parse", &error))?;
    let value = out.trim().to_string();
    Ok(ToolResult::new(value.clone(), json!({"value": value})))
}

/// A capped text result, with the same shape for every read-only verb.
fn output_capped(ctx: &ToolCtx, verb: &str, text: &str, data: Value) -> ToolResult {
    if text.trim().is_empty() {
        let mut observation = format!("`git {verb}` produced no output.");
        if matches!(verb, "diff" | "show") {
            observation.push_str(" The working tree probably has no changes against that commit.");
        }
        return ToolResult::new(observation, data);
    }
    // The line cap and the byte budget are both real: a 2 000-line diff is a wall regardless of
    // how few bytes each line is.
    let limit = MAX_DIFF_LINES.min(ctx.budget().max_bytes.max(1) / 40).max(20);
    let (capped, dropped) = cap(text, limit);
    let mut observation = capped;
    if dropped > 0 {
        let _ = writeln!(
            observation,
            "… {dropped} more line(s) not shown; narrow the request or pass `args` to limit it."
        );
    }
    ToolResult::new(observation, data)
}

/// Whether a message begins with a Conventional Commit type.
fn is_conventional(message: &str) -> bool {
    const TYPES: &[&str] = &[
        "build", "chore", "ci", "docs", "feat", "fix", "perf", "refactor", "revert", "style", "test",
    ];
    let Some((head, _)) = message.split_once(':') else {
        return false;
    };
    // `feat(scope)!` is legal; the scope and the bang are not part of the type word.
    let head = head.trim();
    let head = head.strip_suffix('!').unwrap_or(head);
    let head = head.split('(').next().unwrap_or(head).trim();
    TYPES.contains(&head)
}

/// `path` relative to the worktree, with forward slashes.
///
/// Both sides are canonicalised first, because the gate resolves the path it authorises while
/// [`PolicyGate::worktree`] is the path the human and the journal use. On macOS those differ for
/// everything under `/tmp`, which is a symlink to `/private/tmp` — a `strip_prefix` against the
/// unresolved root fails for every legitimate path in a tempdir, and the fallback would hand git
/// an absolute path where it expected a worktree-relative one.
fn relative_to(root: &Path, path: &Path) -> String {
    let canonical_root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let canonical_path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let relative = canonical_path
        .strip_prefix(&canonical_root)
        .or_else(|_| path.strip_prefix(root))
        .unwrap_or(path);
    relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{PathUse, PolicyGate};
    use std::sync::Arc;

    /// A repository in a tempdir, with a gate whose worktree is that directory.
    struct Fixture {
        dir: tempfile::TempDir,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
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
            Self { dir }
        }

        fn own_path(&self) -> std::path::PathBuf {
            self.dir.path().to_path_buf()
        }
    }

    fn run(dir: &Path, args: &[&str]) {
        let output = std::process::Command::new("git")
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

    fn ctx(dir: &Path) -> ToolCtx {
        let gate = PolicyGate::with_defaults(
            dir,
            immortality_core::config::Policy {
                workspace_root: dir.parent().unwrap().to_path_buf(),
                ..immortality_core::config::Policy::default()
            },
        )
        .unwrap();
        ToolCtx::new(Arc::new(gate), crate::OutputBudget::default())
    }

    fn call(dir: &Path, args: Value) -> Result<ToolResult, ToolError> {
        Git::new().execute(&ctx(dir), Args::new(args))
    }

    #[test]
    fn status_reports_the_branch_and_cleanliness() {
        let fixture = Fixture::new();
        let result = call(&fixture.own_path(), json!({"subcommand": "status"})).unwrap();
        assert_eq!(result.data["clean"], true);
        assert!(result.observation.contains("main"), "{}", result.observation);
    }

    #[test]
    fn status_lists_a_new_file() {
        let fixture = Fixture::new();
        std::fs::write(fixture.dir.path().join("new.txt"), "x").unwrap();
        let result = call(&fixture.own_path(), json!({"subcommand": "status"})).unwrap();
        assert_eq!(result.data["clean"], false);
        assert!(result.observation.contains("new.txt"), "{}", result.observation);
    }

    #[test]
    fn diff_shows_a_change_and_an_empty_diff_says_so() {
        let fixture = Fixture::new();
        let empty = call(&fixture.own_path(), json!({"subcommand": "diff"})).unwrap();
        assert!(empty.observation.contains("no output"), "{}", empty.observation);

        std::fs::write(fixture.dir.path().join("README.md"), "# changed\n").unwrap();
        let changed = call(&fixture.own_path(), json!({"subcommand": "diff"})).unwrap();
        assert!(changed.observation.contains("changed"), "{}", changed.observation);
    }

    #[test]
    fn log_lists_commits_newest_first() {
        let fixture = Fixture::new();
        std::fs::write(fixture.dir.path().join("b.txt"), "b").unwrap();
        run(fixture.dir.path(), &["add", "--all"]);
        run(fixture.dir.path(), &["commit", "-q", "-m", "feat: second"]);

        let result = call(&fixture.own_path(), json!({"subcommand": "log"})).unwrap();
        let lines: Vec<&str> = result.observation.lines().collect();
        assert!(lines[0].contains("feat: second"), "{lines:?}");
        assert!(result.observation.contains("chore: initial"), "{lines:?}");
    }

    #[test]
    fn add_stages_everything_by_default() {
        let fixture = Fixture::new();
        std::fs::write(fixture.dir.path().join("a.txt"), "a").unwrap();
        let result = call(&fixture.own_path(), json!({"subcommand": "add"})).unwrap();
        assert_eq!(result.data["staged"], "all");

        let status = call(&fixture.own_path(), json!({"subcommand": "status"})).unwrap();
        assert!(status.observation.contains("a.txt"), "{}", status.observation);
    }

    #[test]
    fn add_of_one_path_leaves_the_others_untracked() {
        let fixture = Fixture::new();
        std::fs::write(fixture.dir.path().join("one.txt"), "1").unwrap();
        std::fs::write(fixture.dir.path().join("two.txt"), "2").unwrap();
        let result = call(
            &fixture.own_path(),
            json!({"subcommand": "add", "path": "one.txt"}),
        )
        .unwrap();
        assert_eq!(result.data["path"], "one.txt");

        let porcelain = std::process::Command::new("git")
            .arg("-C")
            .arg(fixture.dir.path())
            .args(["status", "--porcelain"])
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&porcelain.stdout);
        assert!(text.contains("A  one.txt"), "{text}");
        assert!(text.contains("?? two.txt"), "{text}");
    }

    #[test]
    fn add_refuses_a_path_outside_the_worktree() {
        let fixture = Fixture::new();
        let error = call(
            &fixture.own_path(),
            json!({"subcommand": "add", "path": "../escape.txt"}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("denied_by_policy"), "{error}");
    }

    #[test]
    fn a_commit_requires_a_conventional_message() {
        let fixture = Fixture::new();
        std::fs::write(fixture.dir.path().join("a.txt"), "a").unwrap();
        let error = call(
            &fixture.own_path(),
            json!({"subcommand": "commit", "message": "did some stuff"}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("Conventional Commit"), "{error}");
    }

    #[test]
    fn a_commit_records_the_call_id_so_a_probe_can_find_it() {
        // The whole crash-safety story of this tool: after a crash between `commit` and the
        // journal record, recovery has the call id and nothing else.
        let fixture = Fixture::new();
        std::fs::write(fixture.dir.path().join("a.txt"), "a").unwrap();
        let ctx = ctx(&fixture.own_path()).with_call("call_abc123");
        let result = Git::new()
            .execute(
                &ctx,
                Args::new(json!({"subcommand": "commit", "message": "feat: add a"})),
            )
            .unwrap();
        assert_eq!(result.data["committed"], true);
        let sha = result.data["sha"].as_str().unwrap();

        let repo = GitRepo::new(fixture.own_path());
        let message = repo.git(&["log", "-1", "--format=%B", sha]).unwrap();
        assert!(
            message.contains("Immortality-Call: call_abc123"),
            "the probe marker must be in the message: {message}"
        );
        assert!(repo.has_trailer("Immortality-Call: call_abc123"));
    }

    #[test]
    fn a_commit_with_nothing_to_commit_is_a_result_not_an_error() {
        let fixture = Fixture::new();
        let result = call(
            &fixture.own_path(),
            json!({"subcommand": "commit", "message": "feat: nothing"}),
        )
        .unwrap();
        assert_eq!(result.data["committed"], false);
        assert_eq!(result.data["reason"], "nothing_to_commit");
    }

    #[test]
    fn a_commit_stages_unstaged_work_first() {
        // `write_file` followed by `git commit` is the most common pair of calls a coder model
        // makes, and requiring `add` between them would make the common path fail.
        let fixture = Fixture::new();
        std::fs::write(fixture.dir.path().join("written.txt"), "content").unwrap();
        let result = call(
            &fixture.own_path(),
            json!({"subcommand": "commit", "message": "feat: write it"}),
        )
        .unwrap();
        assert_eq!(result.data["committed"], true);
    }

    #[test]
    fn commit_refuses_amend_because_it_would_break_the_probe() {
        let fixture = Fixture::new();
        let error = call(
            &fixture.own_path(),
            json!({"subcommand": "commit", "message": "feat: x", "args": ["--amend"]}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("--amend"), "{error}");
    }

    #[test]
    fn branch_lists_and_creates() {
        let fixture = Fixture::new();
        let listed = call(&fixture.own_path(), json!({"subcommand": "branch"})).unwrap();
        assert!(listed.observation.contains("main"), "{}", listed.observation);

        let created = call(
            &fixture.own_path(),
            json!({"subcommand": "branch", "args": ["agent/tsk_1"]}),
        )
        .unwrap();
        // Creating a branch is silent on success, so the test asserts the branch exists rather
        // than that the observation quotes it: measuring `git branch` showed an empty stdout.
        assert!(
            created.observation.contains("No branches."),
            "{}",
            created.observation
        );
        let now = call(&fixture.own_path(), json!({"subcommand": "branch"})).unwrap();
        assert!(now.observation.contains("agent/tsk_1"), "{}", now.observation);
    }

    #[test]
    fn deleting_or_moving_a_branch_is_refused_because_refs_are_shared() {
        // Measured: from a task worktree, `git branch -D feature-x` deleted a branch in the
        // human's repository. Git protects only branches checked out somewhere, so this guard is
        // the only thing standing between the agent and the human's other branches.
        let fixture = Fixture::new();
        run(fixture.dir.path(), &["branch", "feature-x"]);

        for args in [
            json!({"subcommand": "branch", "args": ["-D", "feature-x"]}),
            json!({"subcommand": "branch", "args": ["--delete", "feature-x"]}),
            json!({"subcommand": "branch", "args": ["-m", "feature-x", "renamed"]}),
            json!({"subcommand": "branch", "args": ["-f", "feature-x", "HEAD"]}),
        ] {
            let error = call(&fixture.own_path(), args.clone()).unwrap_err();
            assert!(
                error.to_string().contains("branch_ref_out_of_scope"),
                "{args} -> {error}"
            );
        }

        let listed = call(&fixture.own_path(), json!({"subcommand": "branch"})).unwrap();
        assert!(
            listed.observation.contains("feature-x"),
            "the branch must still be there: {}",
            listed.observation
        );
    }

    #[test]
    fn creating_a_branch_is_still_allowed() {
        // The guard must not be so broad that a task cannot make its own branch.
        let fixture = Fixture::new();
        let result = call(
            &fixture.own_path(),
            json!({"subcommand": "branch", "args": ["agent/tsk_2"]}),
        );
        assert!(result.is_ok(), "{:?}", result.err());
        let listed = call(&fixture.own_path(), json!({"subcommand": "branch"})).unwrap();
        assert!(
            listed.observation.contains("agent/tsk_2"),
            "{}",
            listed.observation
        );
    }

    #[test]
    fn stash_operations_that_remove_an_entry_are_refused() {
        let fixture = Fixture::new();
        std::fs::write(fixture.dir.path().join("README.md"), "# dirty\n").unwrap();
        run(fixture.dir.path(), &["stash", "push", "-m", "human work"]);

        for verb in ["clear", "drop", "pop"] {
            let error = call(
                &fixture.own_path(),
                json!({"subcommand": "stash", "args": [verb]}),
            )
            .unwrap_err();
            assert!(
                error.to_string().contains("stash_entry_out_of_scope"),
                "{verb} -> {error}"
            );
        }

        let list = std::process::Command::new("git")
            .arg("-C")
            .arg(fixture.dir.path())
            .args(["stash", "list"])
            .output()
            .unwrap();
        assert!(
            String::from_utf8_lossy(&list.stdout).contains("human work"),
            "the stash must survive"
        );
    }

    #[test]
    fn stash_apply_is_allowed_because_it_keeps_the_entry() {
        let fixture = Fixture::new();
        std::fs::write(fixture.dir.path().join("keep.txt"), "saved\n").unwrap();
        run(fixture.dir.path(), &["add", "--all"]);
        run(fixture.dir.path(), &["stash", "push", "-m", "saved"]);

        call(
            &fixture.own_path(),
            json!({"subcommand": "stash", "args": ["apply"]}),
        )
        .unwrap();
        assert!(
            fixture.dir.path().join("keep.txt").exists(),
            "apply must restore the work"
        );
        let list = std::process::Command::new("git")
            .arg("-C")
            .arg(fixture.dir.path())
            .args(["stash", "list"])
            .output()
            .unwrap();
        assert!(
            String::from_utf8_lossy(&list.stdout).contains("saved"),
            "apply must keep the entry"
        );
    }

    #[test]
    fn worktree_list_is_allowed_and_add_is_refused() {
        let fixture = Fixture::new();
        let listed = call(
            &fixture.own_path(),
            json!({"subcommand": "worktree", "args": ["list"]}),
        )
        .unwrap();
        assert_ne!(listed.data["worktrees"].as_array().unwrap().len(), 0);

        let refused = call(
            &fixture.own_path(),
            json!({"subcommand": "worktree", "args": ["add", "../elsewhere"]}),
        )
        .unwrap_err();
        assert!(
            refused.to_string().contains("worktree_managed_by_daemon"),
            "{refused}"
        );
    }

    #[test]
    fn stash_carries_the_call_id_in_its_message() {
        let fixture = Fixture::new();
        std::fs::write(fixture.dir.path().join("README.md"), "# dirty\n").unwrap();
        let ctx = ctx(&fixture.own_path()).with_call("call_stash1");
        let result = Git::new()
            .execute(
                &ctx,
                Args::new(json!({"subcommand": "stash", "args": ["push"], "reason": "before risky"})),
            )
            .unwrap();
        assert_eq!(result.data["message"], "before risky [call_stash1]");

        let list = std::process::Command::new("git")
            .arg("-C")
            .arg(fixture.dir.path())
            .args(["stash", "list"])
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&list.stdout);
        assert!(text.contains("call_stash1"), "{text}");
    }

    #[test]
    fn rev_parse_returns_the_head() {
        let fixture = Fixture::new();
        let result = call(
            &fixture.own_path(),
            json!({"subcommand": "rev_parse", "args": ["HEAD"]}),
        )
        .unwrap();
        assert_eq!(result.data["value"].as_str().unwrap().len(), 40);
    }

    #[test]
    fn push_is_not_a_subcommand() {
        let fixture = Fixture::new();
        let error = call(&fixture.own_path(), json!({"subcommand": "push"})).unwrap_err();
        assert!(
            error.to_string().contains("not an available git subcommand"),
            "{error}"
        );
        assert!(error.to_string().contains("push"), "{error}");
    }

    #[test]
    fn a_directory_that_is_not_a_repository_is_refused_rather_than_initialised() {
        let dir = tempfile::tempdir().unwrap();
        let error = call(dir.path(), json!({"subcommand": "status"})).unwrap_err();
        assert!(error.to_string().contains("not_a_repository"), "{error}");
        assert!(
            !dir.path().join(".git").exists(),
            "the tool must never run `git init`"
        );
    }

    #[test]
    fn the_effect_follows_the_verb() {
        let git = Git::new();
        assert_eq!(
            git.effect(&Args::new(json!({"subcommand": "status"}))),
            Effect::Read
        );
        assert_eq!(
            git.effect(&Args::new(json!({"subcommand": "commit"}))),
            Effect::Mutate
        );
        // An unknown name is the most dangerous class, not the safest.
        assert_eq!(
            git.effect(&Args::new(json!({"subcommand": "frobnicate"}))),
            Effect::External
        );
    }

    #[test]
    fn only_commit_is_unknown_for_idempotency() {
        let git = Git::new();
        assert_eq!(
            git.idempotency(&Args::new(json!({"subcommand": "commit", "message": "feat: x"}))),
            Idempotency::Unknown
        );
        assert_eq!(
            git.idempotency(&Args::new(json!({"subcommand": "add"}))),
            Idempotency::Idempotent
        );
    }

    #[test]
    fn args_are_argv_elements_not_a_shell_string() {
        // `--` is needed for a path with a leading dash, and it can only survive if the model's
        // array is passed element by element.
        let fixture = Fixture::new();
        std::fs::write(fixture.dir.path().join("dash.txt"), "x").unwrap();
        let result = call(
            &fixture.own_path(),
            json!({"subcommand": "log", "args": ["-n", "1", "--format=%s"]}),
        )
        .unwrap();
        assert_eq!(
            result.observation.trim().lines().count(),
            1,
            "{}",
            result.observation
        );
    }

    #[test]
    fn a_bad_args_type_is_a_usable_refusal() {
        let fixture = Fixture::new();
        let error = call(&fixture.own_path(), json!({"subcommand": "log", "args": [1, 2]})).unwrap_err();
        assert!(
            error.to_string().contains("`args` must contain only strings"),
            "{error}"
        );
    }

    #[test]
    fn a_huge_diff_is_capped_and_says_so() {
        let fixture = Fixture::new();
        let mut body = String::new();
        for n in 0..2_000 {
            let _ = writeln!(body, "line {n}");
        }
        std::fs::write(fixture.dir.path().join("README.md"), body).unwrap();
        let result = call(&fixture.own_path(), json!({"subcommand": "diff"})).unwrap();
        assert!(
            result.observation.contains("more line(s) not shown"),
            "{}",
            result.observation
        );
        assert!(
            result.observation.lines().count() < 2_000,
            "the cap must actually cap"
        );
    }

    #[test]
    fn the_schema_is_well_formed() {
        assert_eq!(
            immortality_llm::schema::schema_defects(Git::new().schema()),
            Vec::new()
        );
    }

    #[test]
    fn the_schema_does_not_offer_push() {
        let schema = Git::new().schema();
        let offered = schema["properties"]["subcommand"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|value| value.as_str())
            .collect::<Vec<_>>();
        assert!(!offered.contains(&"push"), "{offered:?}");
        assert!(!offered.contains(&"fetch"), "{offered:?}");
        assert!(offered.contains(&"commit"), "{offered:?}");
    }

    #[test]
    fn paths_declares_a_write_only_when_one_was_named() {
        let git = Git::new();
        assert_eq!(git.paths(&Args::new(json!({"subcommand": "status"}))), vec![]);
        assert_eq!(
            git.paths(&Args::new(json!({"subcommand": "add", "path": "a.txt"}))),
            vec![PathUse::write("a.txt")]
        );
    }

    #[test]
    fn an_empty_message_is_refused() {
        let fixture = Fixture::new();
        let error = call(
            &fixture.own_path(),
            json!({"subcommand": "commit", "message": "   "}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("empty message"), "{error}");
    }

    /// A repository plus a real per-task worktree, arranged the way the daemon arranges them.
    struct TaskFixture {
        dir: tempfile::TempDir,
        human: std::path::PathBuf,
        task: immortality_core::worktree::TaskWorktree,
    }

    impl TaskFixture {
        fn new() -> Self {
            use immortality_core::worktree::WorktreeManager;

            let dir = tempfile::tempdir().unwrap();
            let human = dir.path().join("project");
            std::fs::create_dir_all(&human).unwrap();
            for args in [
                vec!["init", "-q", "-b", "main"],
                vec!["config", "user.name", "Test"],
                vec!["config", "user.email", "test@example.com"],
            ] {
                run(&human, &args);
            }
            std::fs::write(human.join("README.md"), "# human checkout\n").unwrap();
            run(&human, &["add", "--all"]);
            run(&human, &["commit", "-q", "-m", "chore: initial"]);

            let task = WorktreeManager::new(&human)
                .ensure("tsk_01", "agent/tsk_01", None)
                .unwrap();
            Self { dir, human, task }
        }

        /// A gate over the *task worktree*, with the repository's parent as workspace root —
        /// exactly the arrangement the daemon builds.
        fn ctx(&self) -> ToolCtx {
            let gate = PolicyGate::with_defaults(
                &self.task.path,
                immortality_core::config::Policy {
                    workspace_root: self.dir.path().to_path_buf(),
                    ..immortality_core::config::Policy::default()
                },
            )
            .unwrap();
            ToolCtx::new(Arc::new(gate), crate::OutputBudget::default())
        }
    }

    #[test]
    fn the_tool_operates_on_the_task_worktree_not_the_human_checkout() {
        // The whole point of the per-task worktree. If this test can be made to pass by writing
        // to the human's checkout, the isolation the design promises does not exist.
        let fixture = TaskFixture::new();
        std::fs::write(fixture.task.path.join("work.txt"), "agent work\n").unwrap();

        let ctx = fixture.ctx();
        let status = Git::new()
            .execute(&ctx, Args::new(json!({"subcommand": "status"})))
            .unwrap();
        assert_eq!(status.data["branch"], "agent/tsk_01");
        assert!(status.observation.contains("work.txt"), "{}", status.observation);

        let commit = Git::new()
            .execute(
                &ctx,
                Args::new(json!({"subcommand": "commit", "message": "feat: agent work"})),
            )
            .unwrap();
        assert_eq!(commit.data["committed"], true);

        // The agent's commit is on its own branch, and the human's checkout has not moved.
        let human_head = std::process::Command::new("git")
            .arg("-C")
            .arg(&fixture.human)
            .args(["log", "-1", "--format=%s"])
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&human_head.stdout).trim(),
            "chore: initial",
            "the agent's commit must not land on the human's branch"
        );
        assert!(!fixture.human.join("work.txt").exists());
    }

    #[test]
    fn a_write_path_outside_the_worktree_is_refused_by_the_gate() {
        // Reads may reach the workspace root, writes may not: `..` out of the worktree is a
        // denial, not a staged file in the human's checkout.
        let fixture = TaskFixture::new();
        let ctx = fixture.ctx();
        let error = Git::new()
            .execute(
                &ctx,
                Args::new(json!({
                    "subcommand": "add",
                    "path": "../project/README.md"
                })),
            )
            .unwrap_err();
        assert!(error.to_string().contains("denied_by_policy"), "{error}");

        let porcelain = std::process::Command::new("git")
            .arg("-C")
            .arg(&fixture.human)
            .args(["status", "--porcelain"])
            .output()
            .unwrap();
        assert!(
            String::from_utf8_lossy(&porcelain.stdout).trim().is_empty(),
            "the human's checkout must stay clean"
        );
    }

    #[test]
    fn conventional_types_are_recognised_with_scope_and_bang() {
        assert!(is_conventional("feat: add"));
        assert!(is_conventional("fix(parser): handle eof"));
        assert!(is_conventional("refactor!: drop the old path"));
        assert!(!is_conventional("add stuff"));
        assert!(!is_conventional("feature: not a type"));
    }
}
