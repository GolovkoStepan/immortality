//! Executing a [`immortality_core::recover`] plan against the real world.
//!
//! # Why the executor is not in `core`
//!
//! `core::recover` is a pure planner: it reads facts and returns steps. That split is what lets
//! every crash point be tested without a crash. This module is the other half — the only place
//! allowed to touch git, the filesystem, the escalation queue and the tool registry on recovery's
//! behalf — and it exists so that "the plan is right" and "the plan was carried out" are two
//! separate claims with two separate tests.
//!
//! # Why the prober lives here
//!
//! Answering "did this `write_file` land?" means reading the file through the *same* policy gate
//! the tool used. A prober in `core` would have to join the worktree itself, and two resolutions
//! of one path is how a probe judges a file the tool never wrote. [`ToolProbe`] therefore borrows
//! the gate, the repository and the task id from the daemon.
//!
//! # The rule that shapes every arm
//!
//! `Unknown` is always allowed and is never a failure of the probe. It means "I cannot tell", and
//! the planner turns it into a rollback rather than a second execution. A probe that guessed
//! `NotApplied` for an effect that happened would duplicate it; a probe that guessed `Applied`
//! would silently drop it. Both are worse than an honest rollback.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use immortality_core::artifacts::ArtifactStore;
use immortality_core::cache::{self, Cache};
use immortality_core::checkpoint::GitRepo;
use immortality_core::context::truncate_output;
use immortality_core::domain::{TaskId, Timestamp};
use immortality_core::escalation::{Entry, EscalationKind, EscalationStore};
use immortality_core::gate::GateVerdict;
use immortality_core::ids::SubtaskId;
use immortality_core::journal::{Journal, JournalRecord, Kind, ReadOutcome, Scope};
use immortality_core::protocol::{CallId, Intent, Probe};
use immortality_core::recover::{Divergence, Prober, RecoveryReport, RecoveryStep, Verdict};
use immortality_core::time::Clock;
use immortality_tools::state::StateFile;
use immortality_tools::{
    Args, OutputBudget, PolicyGate, Registry, ToolCtx, ToolError, probe_delegated, probe_started,
    process_start_digest,
};

/// The call id trailer a model-driven `git commit` carries.
///
/// Spelled once. The `git` tool writes it and this probe reads it; a divergence in case or
/// spacing would make every commit probe answer `NotApplied` and re-run a commit.
pub const COMMIT_TRAILER: &str = "Immortality-Call";

/// Asks the world whether an interrupted call took effect.
#[derive(Debug)]
pub struct ToolProbe<'a> {
    /// The same gate the tool ran under, so paths resolve identically.
    pub gate: &'a PolicyGate,
    /// The task worktree, for `git` probes.
    pub repo: &'a GitRepo,
    /// The machine-local state root, for `delegate` reports.
    pub state_root: &'a Path,
    /// The task the call belongs to.
    pub task: &'a TaskId,
    /// The task's state directory, for the escalation queue.
    pub task_dir: &'a Path,
}

impl Prober for ToolProbe<'_> {
    fn probe(&self, intent: &Intent, arguments: Option<&serde_json::Value>) -> Probe {
        let Some(arguments) = arguments else {
            // Every probe below needs the arguments: a path to re-read, a command to hash, a key
            // to look up. Without them the only honest answer is that the effect is undetermined.
            return Probe::Unknown;
        };
        match intent.tool.as_str() {
            "write_file" => self.probe_content(arguments, "content"),
            "update_state" => self.probe_state(arguments),
            "edit_file" => self.probe_edit(arguments),
            "git" => self.probe_git(intent, arguments),
            "start_process" => self.probe_start(arguments),
            "delegate" => probe_delegated(self.state_root, self.task, intent.call.as_str()),
            "escalate" => self.probe_escalation(arguments),
            // Reads are `Idempotent` and never reach a probe; `read_process` is `ByContent` but
            // re-reading a process's output loses nothing, so re-running it is safe.
            "read_process" => Probe::NotApplied,
            // A mutating shell command is the class this whole mechanism exists for, and it is
            // exactly the one with no cheap evidence: the tree may or may not have changed, and
            // `git status` cannot tell "the command did nothing" from "the command did what it
            // was asked". The planner rolls back — and so it does for every unrecognised tool.
            _ => Probe::Unknown,
        }
    }
}

impl ToolProbe<'_> {
    /// The path a tool writing to `raw` would have opened, or `None` if the gate refuses it.
    ///
    /// This reproduces `ToolCtx::write_path` rather than calling `resolve` alone: the gate's
    /// *check* is what refuses a path outside the worktree or one the deny list protects, and a
    /// probe that skipped it would judge `/etc/passwd` as a file the task legitimately wrote.
    /// A refused path is not evidence of anything, which makes the honest answer `Unknown`.
    fn resolve_write(&self, raw: &str) -> Option<PathBuf> {
        match self.gate.check(
            immortality_tools::Effect::Mutate,
            std::slice::from_ref(&immortality_tools::PathUse::write(raw)),
        ) {
            immortality_tools::Verdict::Allowed => self.gate.resolve(raw),
            // Denied and escalated are both "no such call could have run": nothing to compare.
            _ => None,
        }
    }

    /// The path a tool reading or entering `raw` would have used.
    fn resolve_read(&self, raw: &str) -> Option<PathBuf> {
        match self.gate.check(
            immortality_tools::Effect::Read,
            std::slice::from_ref(&immortality_tools::PathUse::read(raw)),
        ) {
            immortality_tools::Verdict::Allowed => self.gate.resolve(raw),
            _ => None,
        }
    }

    /// A whole-file write: does the file already hold exactly `field`?
    fn probe_content(&self, arguments: &serde_json::Value, field: &str) -> Probe {
        let Some(raw) = arguments.get("path").and_then(|path| path.as_str()) else {
            return Probe::Unknown;
        };
        let Some(content) = arguments.get(field).and_then(|value| value.as_str()) else {
            return Probe::Unknown;
        };
        let Some(path) = self.resolve_write(raw) else {
            return Probe::Unknown;
        };
        std::fs::read(&path).map_or_else(
            |error| {
                // Absent is evidence: the write did not land. Any other read error is not.
                if error.kind() == std::io::ErrorKind::NotFound {
                    Probe::NotApplied
                } else {
                    Probe::Unknown
                }
            },
            |bytes| {
                if bytes == content.as_bytes() {
                    Probe::Applied
                } else {
                    Probe::NotApplied
                }
            },
        )
    }

    /// `update_state`: the same content check, with the state file's own path.
    fn probe_state(&self, arguments: &serde_json::Value) -> Probe {
        let Some(name) = arguments.get("file").and_then(|file| file.as_str()) else {
            return Probe::Unknown;
        };
        let Ok(file) = StateFile::parse(name) else {
            return Probe::Unknown;
        };
        let Some(content) = arguments.get("content").and_then(|value| value.as_str()) else {
            return Probe::Unknown;
        };
        let Some(path) = self.resolve_write(&file.relative_path()) else {
            return Probe::Unknown;
        };
        match std::fs::read(&path) {
            Ok(bytes) if bytes == content.as_bytes() => Probe::Applied,
            Ok(_) => Probe::NotApplied,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Probe::NotApplied,
            Err(_) => Probe::Unknown,
        }
    }

    /// `edit_file`: `new` in and `old` gone means the edit landed.
    ///
    /// Three states are distinguishable and one is not. A file that no longer holds `old` but
    /// holds `new` was edited. A file that still holds `old` was not. A file holding neither —
    /// or both, because `old` also occurs somewhere the edit did not touch — cannot be judged
    /// from the text alone, and the answer is `Unknown` rather than a coin flip: a wrong
    /// `Applied` leaves a half-edited file in place.
    fn probe_edit(&self, arguments: &serde_json::Value) -> Probe {
        let (Some(raw), Some(old), Some(new)) = (
            arguments.get("path").and_then(|value| value.as_str()),
            arguments.get("old").and_then(|value| value.as_str()),
            arguments.get("new").and_then(|value| value.as_str()),
        ) else {
            return Probe::Unknown;
        };
        let Some(path) = self.gate.resolve(raw) else {
            return Probe::Unknown;
        };
        let Ok(bytes) = std::fs::read(&path) else {
            return Probe::NotApplied;
        };
        let Ok(text) = String::from_utf8(bytes) else {
            return Probe::Unknown;
        };
        let has_old = text.contains(old);
        let has_new = text.contains(new);
        match (has_old, has_new) {
            (false, true) => Probe::Applied,
            (true, _) => Probe::NotApplied,
            (false, false) => Probe::Unknown,
        }
    }

    /// `git commit` is found by the trailer the tool writes; any other verb is not probed.
    fn probe_git(&self, intent: &Intent, arguments: &serde_json::Value) -> Probe {
        if arguments.get("subcommand").and_then(|value| value.as_str()) != Some("commit") {
            return Probe::Unknown;
        }
        let needle = format!("{COMMIT_TRAILER}: {}", intent.call.as_str());
        if self.repo.has_trailer(&needle) {
            Probe::Applied
        } else {
            // No commit carrying this call id exists. That is evidence for a repository that is
            // readable; `has_trailer` already swallowed the failure case, so an unreadable
            // repository reports the same answer as a clean one. Acceptable: a commit that cannot
            // be found cannot be duplicated either, and the retry would fail loudly.
            Probe::NotApplied
        }
    }

    /// `start_process` is found by the marker its environment carries.
    ///
    /// The marker is derived from `(command, resolved cwd)`, not from the whole argument object,
    /// so it is recomputed here exactly as the tool computed it. `ToolCtx::enter_path` is what
    /// resolves `cwd` for the tool, and it uses the gate — the same call this uses.
    fn probe_start(&self, arguments: &serde_json::Value) -> Probe {
        let Some(command) = arguments.get("command").and_then(|value| value.as_str()) else {
            return Probe::Unknown;
        };
        let raw = arguments
            .get("cwd")
            .and_then(|value| value.as_str())
            .unwrap_or(".");
        let Some(directory) = self.resolve_read(raw) else {
            return Probe::Unknown;
        };
        probe_started(&process_start_digest(command, &directory))
    }

    /// `escalate` deduplicates by key: a queue entry proves the question was written.
    fn probe_escalation(&self, arguments: &serde_json::Value) -> Probe {
        let Some(key) = arguments.get("key").and_then(|value| value.as_str()) else {
            return Probe::Unknown;
        };
        EscalationStore::open(self.task_dir, self.task.clone()).map_or(Probe::Unknown, |store| {
            if store.by_key(key).is_some() {
                Probe::Applied
            } else {
                Probe::NotApplied
            }
        })
    }
}

/// Everything the executor needs that does not change while it runs.
///
/// Public fields rather than a constructor: there are eight of them, the workspace lints cap a
/// function at seven arguments, and a builder for a value that is assembled once is ceremony.
#[derive(Debug)]
pub struct RecoveryDeps {
    /// The task being recovered.
    pub task: TaskId,
    /// Its directory under the state root.
    pub task_dir: PathBuf,
    /// The machine-local state root.
    pub state_root: PathBuf,
    /// The task worktree.
    pub worktree: PathBuf,
    /// The branch the worktree must be on, `agent/<task>`.
    pub branch: String,
    /// The policy gate every retried tool call runs under.
    pub gate: Arc<PolicyGate>,
    /// The tools a retry may dispatch to.
    pub registry: Arc<Registry>,
    /// Where oversized tool output goes.
    pub artifacts: ArtifactStore,
    /// The clock every record is stamped from.
    pub clock: Arc<dyn Clock>,
}

/// What executing a plan did, for the daemon's log line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Executed {
    /// Steps carried out, in order.
    pub steps: Vec<String>,
}

impl Executed {
    /// A one-line description, for a log message a human reads.
    #[must_use]
    pub fn summary(&self) -> String {
        if self.steps.is_empty() {
            return "nothing to do".into();
        }
        self.steps.join("; ")
    }
}

/// Carries out a recovery plan, in order.
///
/// One step at a time and fail-fast: a plan is ordered because a later step depends on an earlier
/// one (a rollback before a retry, the state rewrite after everything). Continuing past a failure
/// would apply steps to a world that no longer matches the assumptions the plan was built on.
pub struct Executor<'a> {
    deps: &'a RecoveryDeps,
    journal: &'a mut Journal,
    /// The journal as recovery read it, for deriving caches and for scope lookup.
    outcome: ReadOutcome,
    /// The scope each unfinished call was written in.
    scopes: Vec<(CallId, Scope)>,
    /// The last checkpoint named by the plan, reused for the `attempt_restarted` payload.
    base: Option<String>,
    clock: Arc<dyn Clock>,
}

impl std::fmt::Debug for Executor<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Executor")
            .field("task", &self.deps.task)
            .field("records", &self.outcome.records.len())
            .finish_non_exhaustive()
    }
}

impl<'a> Executor<'a> {
    /// An executor writing to `journal`, carrying out `report`.
    #[must_use]
    pub fn new(
        deps: &'a RecoveryDeps,
        journal: &'a mut Journal,
        outcome: ReadOutcome,
        report: &RecoveryReport,
    ) -> Self {
        let scopes = report
            .calls
            .iter()
            .map(|call| (call.call.clone(), call.scope.clone()))
            .collect();
        // The clock lives in the journal, which is where every other writer reads it from. A
        // second `Arc` here would be a second source that has to be kept in step by hand.
        let clock = Arc::clone(&deps.clock);
        Self {
            deps,
            journal,
            outcome,
            scopes,
            base: None,
            clock,
        }
    }

    /// Run every step.
    ///
    /// # Errors
    /// A step could not be carried out. The message names the step, because "recovery failed" is
    /// not something a human can act on.
    pub fn run(&mut self, steps: &[RecoveryStep]) -> Result<Executed, String> {
        let mut done = Vec::with_capacity(steps.len());
        for step in steps {
            let description = self.step(step)?;
            done.push(description);
        }
        Ok(Executed { steps: done })
    }

    fn step(&mut self, step: &RecoveryStep) -> Result<String, String> {
        match step {
            // Already applied by `Journal::open`, which truncated the file and journalled the
            // repair. Reporting it keeps the plan's account honest about what changed on disk.
            RecoveryStep::DropTornTail { bytes } => Ok(format!("dropped {bytes} bytes of torn tail")),
            RecoveryStep::BlockTask { reason } => self.block(reason),
            RecoveryStep::RecreateWorktree { branch, sha } => self.recreate(branch, sha.as_deref()),
            RecoveryStep::RestoreBranch { branch } => self.restore_branch(branch),
            RecoveryStep::PreserveDirtyWork { reason } => self.preserve(reason),
            RecoveryStep::SynthesizeApplied { call, observation } => self.synthesize(call, observation),
            RecoveryStep::RetryCall {
                call,
                tool,
                arguments,
            } => self.retry(call, tool, arguments.as_ref()),
            RecoveryStep::AbandonCall { call, reason } => self.abandon(call, reason),
            RecoveryStep::RollbackToCheckpoint { sha, reason } => self.rollback(sha, reason),
            RecoveryStep::AbandonAttempt { subtask, number } => self.abandon_attempt(subtask, *number),
            RecoveryStep::RewriteStateFile { divergence } => self.rewrite(divergence),
            RecoveryStep::RecordRecovery { summary } => self.record_recovery(summary),
        }
    }

    /// Stop the task: the journal is not trustworthy.
    ///
    /// # Why this does not write to the journal
    ///
    /// The one condition that blocks a task is a journal whose chain is broken or whose sequence
    /// has a gap. A [`Journal`] cannot even be *opened* over such a file — `open` refuses rather
    /// than appending after a break, which is the whole point — so a `block` that required a write
    /// handle could never run in the only case it exists for.
    ///
    /// The durable record of the block therefore goes to `panic.json` in the task's state
    /// directory, the file the design already reserves for an emergency stop, and the blocking
    /// escalation is the record a human acts on. Both are what `immortality status` reads, and the
    /// journal is left exactly as the crash left it: silently repairing a broken chain would erase
    /// the evidence of whatever broke it.
    fn block(&self, reason: &str) -> Result<String, String> {
        let marker = self.deps.task_dir.join("panic.json");
        let body = serde_json::json!({
            "v": 1,
            "state": "blocked",
            "cause": "journal_corrupt",
            "reason": reason,
            "blocked_at": self.clock.now().to_rfc3339(),
            "task": self.deps.task.as_str(),
        });
        let mut text = serde_json::to_string_pretty(&body)
            .map_err(|error| format!("cannot encode the block marker: {error}"))?;
        text.push('\n');
        immortality_core::atomic::write_atomically(&marker, text.as_bytes())
            .map_err(|error| format!("cannot write {}: {error}", marker.display()))?;

        let mut store = EscalationStore::open(&self.deps.task_dir, self.deps.task.clone())
            .map_err(|error| format!("cannot open the escalation queue: {error}"))?;
        let entry = Entry::new(
            self.deps.task.clone(),
            "recovery:journal_corrupt",
            EscalationKind::JournalCorrupt,
            format!(
                "The journal for {} cannot be trusted: {reason}. The task is blocked. Inspect \
                 the journal and the state directory before it runs again.",
                self.deps.task
            ),
            Vec::new(),
            self.clock.now(),
        );
        store
            .ask(entry)
            .map_err(|error| format!("cannot open the escalation: {error}"))?;
        Ok(format!("blocked the task: {reason}"))
    }

    fn recreate(&self, branch: &str, sha: Option<&str>) -> Result<String, String> {
        let manager = immortality_core::worktree::WorktreeManager::new(&self.deps.worktree);
        // `ensure` refuses to guess a base: when the branch does not exist and no SHA was
        // recorded, it reports that rather than creating the branch from whatever `HEAD` is.
        manager
            .ensure(self.deps.task.as_str(), branch, sha)
            .map_err(|error| format!("cannot recreate the worktree: {error}"))?;
        Ok(sha.map_or_else(
            || format!("recreated the worktree on {branch}"),
            |sha| format!("recreated the worktree on {branch} at {sha}"),
        ))
    }

    /// Put a detached worktree back on its branch.
    ///
    /// Forced, because the plan guarantees `PreserveDirtyWork` ran first when the tree has
    /// uncommitted changes: the work has a checkpoint named in the journal before anything moves.
    /// Without that guarantee this would be a data-loss bug, which is why the plan's ordering is
    /// a rule and not a detail.
    fn restore_branch(&self, branch: &str) -> Result<String, String> {
        let repo = GitRepo::new(&self.deps.worktree);
        let head = repo
            .head()
            .map_err(|error| format!("cannot read HEAD: {error}"))?
            .unwrap_or_else(|| branch.to_string());
        repo.ensure_branch(branch, &head)
            .map_err(|error| format!("cannot check out {branch}: {error}"))?;
        Ok(format!("checked out {branch} again"))
    }

    /// Snapshot uncommitted work before anything can roll over it.
    fn preserve(&mut self, reason: &str) -> Result<String, String> {
        let repo = GitRepo::new(&self.deps.worktree);
        let checkpoint = repo
            .hidden_checkpoint(&self.deps.branch, reason)
            .map_err(|error| format!("cannot snapshot the dirty worktree: {error}"))?;
        let Some(checkpoint) = checkpoint else {
            return Ok("nothing to preserve".into());
        };
        self.journal
            .append_scoped(
                self.clock.now(),
                Kind::CheckpointCreated,
                serde_json::json!({
                    "sha": checkpoint.sha,
                    "branch": checkpoint.branch,
                    "role": "preserved",
                    "reason": reason,
                }),
                &Scope::task(),
            )
            .map_err(|error| format!("cannot journal the snapshot: {error}"))?;
        Ok(format!("snapshotted uncommitted work at {}", checkpoint.sha))
    }

    /// The effect happened: tell the model so, without running the call again.
    fn synthesize(&mut self, call: &CallId, observation: &str) -> Result<String, String> {
        let scope = self.scope_of(call);
        let mut log = immortality_core::protocol::ToolCallLog::recovering(self.journal, scope);
        log.effect(call, Probe::Applied, serde_json::json!({ "by": "recovery" }))
            .map_err(|error| format!("cannot record the probe verdict for {call}: {error}"))?;
        log.completed(
            call,
            serde_json::json!({ "observation": observation, "recovered": true }),
            None,
        )
        .map_err(|error| format!("cannot close {call}: {error}"))?;
        Ok(format!("closed {call} as already applied"))
    }

    /// Re-issue a call the plan decided is safe to repeat.
    fn retry(
        &mut self,
        call: &CallId,
        tool_name: &str,
        arguments: Option<&serde_json::Value>,
    ) -> Result<String, String> {
        let Some(arguments) = arguments else {
            // The intent record predates the `arguments` field. Inventing them would be running a
            // command nobody asked for; the model gets an observation instead.
            return self.abandon(
                call,
                "the interrupted call's arguments are not in the journal, so it was not re-run",
            );
        };
        let Some(tool) = self.deps.registry.get(tool_name).cloned() else {
            return self.abandon(
                call,
                &format!("`{tool_name}` is not registered, so the call was not re-run"),
            );
        };
        let scope = self.scope_of(call);
        let args = Args::new(arguments.clone());
        let ctx = ToolCtx::new(Arc::clone(&self.deps.gate), OutputBudget::default())
            .with_call(call.as_str().to_string())
            .with_task(self.deps.task.clone());

        let mut log = immortality_core::protocol::ToolCallLog::recovering(self.journal, scope);
        let result = tool.execute(&ctx, args);
        match result {
            Ok(result) => {
                let truncated = truncate_output(&self.deps.artifacts, &result.observation, 8192)
                    .map_err(|error| format!("cannot store the output of {call}: {error}"))?;
                log.completed(
                    call,
                    serde_json::json!({ "observation": truncated.text, "recovered": true }),
                    truncated.artifact,
                )
                .map_err(|error| format!("cannot close {call}: {error}"))?;
                Ok(format!("re-ran {tool_name} for {call}"))
            }
            Err(error) => {
                let observation = refusal(&error);
                log.failed(
                    call,
                    serde_json::json!({ "error": observation, "recovered": true }),
                )
                .map_err(|error| format!("cannot close {call}: {error}"))?;
                Ok(format!("re-ran {tool_name} for {call}: it refused"))
            }
        }
    }

    /// Close a call that must not be re-run, with a reason the model can read.
    fn abandon(&mut self, call: &CallId, reason: &str) -> Result<String, String> {
        let scope = self.scope_of(call);
        let mut log = immortality_core::protocol::ToolCallLog::recovering(self.journal, scope);
        log.failed(
            call,
            serde_json::json!({
                "error": reason,
                "abandoned": true,
                "hint": "This call was interrupted by a restart and cannot be replayed. Do not \
                         repeat it blindly; inspect the workspace first.",
            }),
        )
        .map_err(|error| format!("cannot close {call}: {error}"))?;
        Ok(format!("abandoned {call}"))
    }

    /// Move the tree to a known-good SHA.
    ///
    /// The runner's own ordering — snapshot, journal, move — is kept. The role is `recovery`
    /// rather than `rollback` because `rollback` means "the runner decided this and the matching
    /// `attempt_restarted` will confirm it"; recovery has no such confirmation record, and
    /// pretending it did would make the next start believe a second rollback was interrupted.
    fn rollback(&mut self, sha: &str, reason: &str) -> Result<String, String> {
        let repo = GitRepo::new(&self.deps.worktree);
        // The plan may ask for the rollback after a preserve step, but the tree can still have
        // changed since: `hidden_checkpoint` is cheap and a rollback is destructive, so the
        // snapshot is taken unconditionally here too. `None` means the tree was clean.
        let safety = repo
            .hidden_checkpoint(&self.deps.branch, reason)
            .map_err(|error| format!("cannot snapshot before the rollback: {error}"))?;
        if let Some(safety) = &safety {
            self.journal
                .append_scoped(
                    self.clock.now(),
                    Kind::CheckpointCreated,
                    serde_json::json!({
                        "sha": safety.sha,
                        "branch": safety.branch,
                        "role": "discarded",
                        "reason": reason,
                    }),
                    &Scope::task(),
                )
                .map_err(|error| format!("cannot journal the safety snapshot: {error}"))?;
        }
        self.journal
            .append_scoped(
                self.clock.now(),
                Kind::CheckpointCreated,
                serde_json::json!({ "sha": sha, "branch": self.deps.branch, "role": "recovery", "reason": reason }),
                &Scope::task(),
            )
            .map_err(|error| format!("cannot journal the rollback marker: {error}"))?;
        repo.ensure_branch(&self.deps.branch, sha)
            .map_err(|error| format!("cannot check out {}: {error}", self.deps.branch))?;
        repo.reset_hard(sha)
            .map_err(|error| format!("cannot reset to {sha}: {error}"))?;
        self.base = Some(sha.to_string());
        Ok(format!("rolled the tree back to {sha}"))
    }

    /// Close an attempt that the crash left open.
    ///
    /// Written as `attempt_restarted`, because a `Running` attempt can never be superseded and
    /// there is no `attempt_abandoned` kind: the record that ends an attempt is the one that
    /// names its replacement. `cause: "recovery"` says a restart, not a failure, ended it — the
    /// model did nothing wrong, and spending its failure budget on a reboot is how a task dies
    /// of an unrelated crash.
    fn abandon_attempt(&mut self, subtask: &SubtaskId, number: u32) -> Result<String, String> {
        self.journal
            .append_scoped(
                self.clock.now(),
                Kind::AttemptRestarted,
                serde_json::json!({
                    "number": number,
                    "next": number.saturating_add(1),
                    "cause": "recovery",
                    "why": "the daemon restarted while this attempt was running",
                    "base": self.base.clone().unwrap_or_default(),
                }),
                &Scope::attempt(subtask.clone(), number),
            )
            .map_err(|error| format!("cannot close attempt {number} of {subtask}: {error}"))?;
        Ok(format!("closed attempt {number} of {subtask}"))
    }

    /// Rewrite the derived cache from the journal.
    ///
    /// Both files are rewritten, not just the one the divergence named: they are a single view of
    /// one journal, and a second divergence found by the same pass would otherwise be repaired by
    /// the next restart instead of this one.
    fn rewrite(&self, divergence: &Divergence) -> Result<String, String> {
        let state = cache::derived(&self.outcome.records);
        Cache::new(&self.deps.task_dir)
            .write(&state)
            .map_err(|error| format!("cannot rewrite the state cache: {error}"))?;
        Ok(format!("rewrote the state cache ({divergence:?})"))
    }

    /// The last step: audit what was repaired, and leave the cache consistent.
    fn record_recovery(&mut self, summary: &str) -> Result<String, String> {
        let state = cache::derived(&self.outcome.records);
        Cache::new(&self.deps.task_dir)
            .write(&state)
            .map_err(|error| format!("cannot rewrite the state cache: {error}"))?;
        self.journal
            .append_event(
                self.clock.now(),
                Kind::DaemonRecovered,
                serde_json::json!({ "summary": summary }),
            )
            .map_err(|error| format!("cannot journal the recovery: {error}"))?;
        // Flushed before the caller returns: the whole point of the record is that a *later*
        // crash finds it.
        self.journal
            .sync()
            .map_err(|error| format!("cannot flush the journal: {error}"))?;
        Ok("recorded the recovery".into())
    }

    /// The scope a call was written in, or the task scope when the plan does not name it.
    fn scope_of(&self, call: &CallId) -> Scope {
        self.scopes
            .iter()
            .find(|(candidate, _)| candidate == call)
            .map_or_else(Scope::task, |(_, scope)| scope.clone())
    }
}

/// A tool's failure, as text the model can act on.
fn refusal(error: &ToolError) -> String {
    match error {
        ToolError::Refused(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Record `daemon_started` in a task's journal, before recovery runs.
///
/// Written first so the journal's own lifecycle brackets the repair: a reader can see the
/// process that is about to fix things was already running when it decided to.
///
/// # Errors
/// The journal cannot be opened or written.
pub fn daemon_started(
    journal_dir: &Path,
    task: TaskId,
    clock: &dyn Clock,
) -> std::result::Result<Journal, String> {
    let mut journal = Journal::open(journal_dir, task)
        .map_err(|error| format!("cannot open the journal in {}: {error}", journal_dir.display()))?;
    journal
        .append_event(clock.now(), Kind::DaemonStarted, serde_json::json!({}))
        .map_err(|error| format!("cannot record daemon_started: {error}"))?;
    Ok(journal)
}

/// Record `daemon_stopped` in a task's journal during a clean shutdown.
///
/// # Errors
/// The journal cannot be opened or written. A shutdown is best-effort: the caller logs this.
pub fn daemon_stopped(journal_dir: &Path, task: TaskId, at: Timestamp) -> std::result::Result<(), String> {
    let mut journal = Journal::open(journal_dir, task)
        .map_err(|error| format!("cannot open the journal in {}: {error}", journal_dir.display()))?;
    journal
        .append_event(at, Kind::DaemonStopped, serde_json::json!({}))
        .map_err(|error| format!("cannot record daemon_stopped: {error}"))?;
    journal
        .sync()
        .map_err(|error| format!("cannot flush the journal: {error}"))
}

/// Whether a journal's task is still unfinished.
///
/// The last `task_state` record decides. `merged` and `cancelled` are terminal, and a task that
/// reached one is not recovered: rewriting its cache or closing its attempts would touch a
/// history that is over. A journal with no state record at all is unfinished — a task created but
/// never started still has work ahead of it.
#[must_use]
pub fn is_unfinished(records: &[JournalRecord]) -> bool {
    let last = records
        .iter()
        .rev()
        .find(|record| record.fields.kind == Kind::TaskState)
        .and_then(|record| record.event.payload.get("state").and_then(|s| s.as_str()));
    !matches!(last, Some("merged" | "cancelled"))
}

/// Whether executing the plan did any repair.
#[must_use]
pub fn repaired(report: &RecoveryReport) -> bool {
    report.verdict != Verdict::Healthy
}

/// The last checkpoint a journal names, as recovery's fallback target.
///
/// Reads the newest `checkpoint_created` with a SHA, in any role. The plan decides what to do with
/// it; this only answers "which tree was known good last".
#[must_use]
pub fn last_checkpoint(records: &[JournalRecord]) -> Option<immortality_core::recover::LastCheckpoint> {
    records.iter().rev().find_map(|record| {
        if record.fields.kind != Kind::CheckpointCreated {
            return None;
        }
        let sha = record.event.payload.get("sha").and_then(|sha| sha.as_str())?;
        // Every `checkpoint_created` record names a hidden snapshot: meaningful commits are the
        // task branch itself and are not what a rollback targets.
        Some(immortality_core::recover::LastCheckpoint {
            sha: sha.to_string(),
            kind: immortality_core::checkpoint::CheckpointKind::Hidden,
        })
    })
}

/// The branch a task's worktree must be on.
#[must_use]
pub fn expected_branch(task: &TaskId) -> String {
    format!("{}{task}", immortality_core::checkpoint::TASK_BRANCH_PREFIX)
}

/// Read a journal directory into the shape recovery plans over.
///
/// # Errors
/// The directory cannot be read.
pub fn read_journal(journal_dir: &Path, task: &TaskId) -> std::result::Result<ReadOutcome, String> {
    immortality_core::journal::JournalReader::open(journal_dir, task)
        .map_err(|error| format!("cannot open the journal in {}: {error}", journal_dir.display()))?
        .read()
        .map_err(|error| format!("cannot read the journal in {}: {error}", journal_dir.display()))
}

/// The gate verdict type, re-exported so a caller does not have to reach into core for it.
pub type GateOutcome = GateVerdict;

// The executor and the probe are tested in `tests/recovery.rs`: they need a real git repository
// and a real journal, and a unit test here would build both through `Command` anyway.
