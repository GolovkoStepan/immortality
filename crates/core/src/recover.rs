//! Crash recovery: turn "the process died somewhere" into a concrete, ordered plan.
//!
//! # Recovery is a planning problem, not an execution problem
//!
//! This module does not touch git, does not write the journal and does not run shell commands.
//! It reads the facts — a journal scan, a worktree inspection, a state-file snapshot — and
//! returns a [`RecoveryReport`] whose `steps` the daemon executes in order. The daemon owns the
//! tool registry, the git handle and the escalation queue; this module owns the decisions.
//!
//! The split exists for a practical reason: a planner over plain data can be driven to every
//! crash point in a test without a real crash, and those tests are what make "no data loss on
//! `kill -9`" a claim rather than a hope. A function that both decided and acted would need a
//! repository, a process table and a backend for every assertion.
//!
//! The one exception is [`Prober`]: answering "did this effect happen?" requires looking at the
//! world (`git log`, the filesystem), so the caller injects that knowledge. [`NoProbe`] keeps
//! the module usable with no I/O at all.
//!
//! # The five rules
//!
//! 1. **The journal wins.** `attempts.json` and `subtasks.json` are caches for a fast `status`;
//!    when they disagree with the journal they are rewritten from it. A divergence is reported,
//!    never resolved by majority vote.
//! 2. **A torn tail is dropped, a broken chain is not repaired.** A torn tail is provably the
//!    last thing written and nothing can follow it. A chain break means records were edited,
//!    reordered or deleted, and continuing would launder corruption into history.
//! 3. **An unfinished `Unknown` call is probed, never retried.** See [`crate::protocol`].
//! 4. **Every unfinished call produces an observation.** A resumed conversation carries a
//!    `tool_calls` entry the backend expects an answer for. A missing answer is a rejected
//!    request, and a rejected request after a crash is how a task gets stuck forever.
//! 5. **An attempt the journal left open is abandoned, not failed.** A `Running` attempt can never
//!    be superseded, so leaving it open parks the subtask forever; and marking it failed would
//!    spend the failure budget on a reboot the model had nothing to do with.
//!
//! # Ordering
//!
//! Steps are ordered by what would make the next step lie if it ran first: the journal is
//! trusted before the worktree is inspected (a corrupt journal means nothing else is knowable),
//! uncommitted work is snapshotted before any probe can run over it, the worktree is restored
//! before probes inspect it, every rollback runs before any call is retried or answered (a reset
//! would otherwise throw the retry away), state files are rewritten last (they derive from
//! everything above), and `record_recovery` is always the final step so the audit trail is
//! written after the repair it describes.

use std::collections::{BTreeMap, BTreeSet};

use crate::checkpoint::CheckpointKind;
use crate::domain::{SubtaskId, SubtaskState};
use crate::journal::{JournalRecord, Kind, ReadOutcome, ReadProblem};
use crate::protocol::{CallId, FoundOpenCall, Intent, Probe, RecoveryAction};

/// Facts about the task's worktree, gathered by the caller.
///
/// Gathered rather than fetched: recovery runs while the repository may be half-broken, and a
/// `git` call that hangs is a good reason to keep this module free of subprocesses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceFacts {
    /// Whether the worktree directory exists at all.
    pub exists: bool,
    /// Whether it is a git work tree, as opposed to a directory that was moved or emptied.
    pub is_repository: bool,
    /// Branch currently checked out. `None` means a detached `HEAD`.
    pub branch: Option<String>,
    /// `HEAD` SHA, if the repository has commits.
    pub head: Option<String>,
    /// Whether there are uncommitted changes, untracked files included.
    pub dirty: bool,
}

impl WorkspaceFacts {
    /// A worktree that was never created: the normal state before the first attempt.
    #[must_use]
    pub const fn absent() -> Self {
        Self {
            exists: false,
            is_repository: false,
            branch: None,
            head: None,
            dirty: false,
        }
    }

    /// A healthy worktree on `branch` at `head`.
    #[must_use]
    pub fn healthy(branch: impl Into<String>, head: impl Into<String>) -> Self {
        Self {
            exists: true,
            is_repository: true,
            branch: Some(branch.into()),
            head: Some(head.into()),
            dirty: false,
        }
    }

    /// Same worktree, with uncommitted changes.
    #[must_use]
    pub const fn dirty(mut self) -> Self {
        self.dirty = true;
        self
    }

    /// Same repository, with a detached `HEAD`.
    #[must_use]
    pub fn detached(mut self) -> Self {
        self.branch = None;
        self
    }
}

/// What the task's state files claim, as read before recovery.
///
/// Only the fields recovery acts on. This is deliberately not the full `Task`: the whole point
/// is that this data is untrusted, and a type that could be mistaken for the durable state would
/// invite exactly the confusion the journal exists to prevent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PersistedState {
    /// Attempts described by `attempts.json`.
    pub attempts: Vec<PersistedAttempt>,
    /// Subtask states described by `subtasks.json`.
    pub subtasks: Vec<PersistedSubtask>,
}

/// One attempt as the cache file describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedAttempt {
    pub subtask: SubtaskId,
    pub number: u32,
    pub finished: bool,
}

/// One subtask as the cache file describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedSubtask {
    pub id: SubtaskId,
    pub state: SubtaskState,
}

/// A place where a cache file disagrees with the journal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Divergence {
    /// The file says the attempt finished and the journal never closed it.
    AttemptOpenInJournal { subtask: SubtaskId, number: u32 },
    /// The journal closed the attempt and the file still lists it as running.
    AttemptClosedInJournal { subtask: SubtaskId, number: u32 },
    /// The subtask's state in the file is not the state the journal implies.
    SubtaskState {
        subtask: SubtaskId,
        file: SubtaskState,
        journal: SubtaskState,
    },
    /// The file describes a subtask the journal never created.
    SubtaskUnknown { subtask: SubtaskId },
    /// The file describes an attempt the journal never started.
    ///
    /// Worth reporting rather than dropping silently: the attempt number feeds
    /// `max_attempts_per_subtask`, so an attempt that exists only in the cache consumes budget
    /// for work that never happened.
    AttemptUnknown { subtask: SubtaskId, number: u32 },
}

/// What recovery decided about one unfinished tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallRecovery {
    pub call: CallId,
    pub action: RecoveryAction,
    /// The scope the call's `intent` was written in.
    ///
    /// The executor writes the closing record — `tool_call_completed`, `tool_call_failed`,
    /// `effect_verified` — in *this* scope. `(subtask, attempt)` is the key `attempts.json` is
    /// rebuilt from, so a closing record in a scope of recovery's own invention would leave the
    /// attempt looking open while the model's `tool_calls` entry is answered.
    pub scope: crate::journal::Scope,
    /// The probe's answer, when one was run. `None` for calls that did not need a probe.
    ///
    /// The executor reads this to write `effect_verified` / `effect_absent` before running the
    /// matching step, so the journal records the evidence and not just the conclusion.
    pub effect: Option<Probe>,
}

impl CallRecovery {
    /// Whether the executor must synthesize an observation for this call.
    ///
    /// A retried call answers its own `tool_calls` entry when it runs again, so nothing needs to
    /// be invented for it. Every other disposition can never run the tool again, and the model
    /// must still be told something; leaving the id unanswered makes the next request invalid.
    #[must_use]
    pub const fn needs_synthesized_observation(&self) -> bool {
        !matches!(self.action, RecoveryAction::Retry)
    }
}

/// One executable step in the recovery plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryStep {
    /// Drop a torn tail. Already applied when the journal was opened; listed so the report is
    /// honest about what changed on disk.
    DropTornTail { bytes: u64 },

    /// Stop: the journal cannot be trusted.
    BlockTask { reason: String },

    /// Recreate the worktree from `branch`, at `sha` when known.
    RecreateWorktree { branch: String, sha: Option<String> },

    /// The worktree is intact but parked on a detached `HEAD`; check the task branch out again.
    RestoreBranch { branch: String },

    /// Snapshot uncommitted work before anything can roll over it.
    ///
    /// A crash leaves the model's half-finished edits in the tree. They may be exactly what the
    /// next attempt builds on, so they are checkpointed rather than reset — and checkpointed
    /// first, because the probe or the rollback that follows may destroy them.
    PreserveDirtyWork { reason: String },

    /// The effect is already present: write `effect_verified` and hand the model this
    /// observation instead of running the tool again.
    SynthesizeApplied { call: CallId, observation: String },

    /// Nothing happened (or the call was safe to repeat): run it.
    ///
    /// `arguments` are the stored arguments, because re-issuing a call means re-issuing *that*
    /// call: without them the executor would have the tool name and nothing to run. `None` is a
    /// record that lost its payload, and the executor must close the call with an observation
    /// rather than guess a second time — running a `shell` with invented arguments is exactly the
    /// duplicate this project cannot have.
    RetryCall {
        call: CallId,
        tool: String,
        arguments: Option<serde_json::Value>,
    },

    /// The call can neither be probed nor retried; close it with an observation explaining why.
    AbandonCall { call: CallId, reason: String },

    /// The probe could not decide. Roll back to the last green checkpoint and start over.
    RollbackToCheckpoint { sha: String, reason: String },

    /// Close an attempt that is being abandoned, so a fresh one can start.
    ///
    /// Emitted only when the plan rolls back: L3 is defined as "roll back to the last green
    /// checkpoint, fresh context, a different strategy in the brief", so the attempt that was
    /// interrupted by the crash is over. A plain restart does *not* emit this — the same attempt
    /// resumes with its conversation replayed from the journal, which is the whole point of
    /// recording every tool call.
    ///
    /// Recorded as `attempt_restarted`; there is no `attempt_abandoned` kind and the list is
    /// closed. Its `attempt` field names the attempt being **closed**, not its replacement, so
    /// that every record's `attempt` field means "the attempt this record is about" without
    /// exception. The replacement number goes in the payload.
    AbandonAttempt { subtask: SubtaskId, number: u32 },

    /// Rewrite a derived state file from the journal.
    RewriteStateFile { divergence: Divergence },

    /// Write `daemon_recovered` with the summary. Always the last step.
    RecordRecovery { summary: String },
}

/// The top-level answer recovery gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Nothing to fix.
    Healthy,
    /// Repairs are planned; the task can continue.
    Repaired,
    /// The task must stop and wait for a human.
    Blocked,
}

/// What recovery found and what it would do about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryReport {
    pub verdict: Verdict,
    /// Bytes dropped from a torn tail, if any.
    pub dropped_tail_bytes: u64,
    /// Chain breaks and sequence gaps. Never repaired automatically.
    pub journal_problems: Vec<ReadProblem>,
    /// Unfinished calls and their dispositions, in journal order.
    pub calls: Vec<CallRecovery>,
    /// Divergences between cache files and the journal.
    pub divergences: Vec<Divergence>,
    /// Why the task is blocked, when it is.
    pub blocked_reason: Option<String>,
    /// Steps to execute, in order.
    pub steps: Vec<RecoveryStep>,
}

impl RecoveryReport {
    /// Human-readable one-liner for `daemon_recovered` and `immortality status`.
    #[must_use]
    pub fn summary(&self) -> String {
        if self.verdict == Verdict::Healthy {
            return "clean".into();
        }
        let mut parts = Vec::new();
        if self.dropped_tail_bytes > 0 {
            parts.push(format!(
                "dropped {} bytes of torn journal tail",
                self.dropped_tail_bytes
            ));
        }
        if !self.journal_problems.is_empty() {
            parts.push(format!("{} journal problems", self.journal_problems.len()));
        }
        if !self.calls.is_empty() {
            parts.push(format!("{} unfinished tool calls", self.calls.len()));
        }
        if !self.divergences.is_empty() {
            parts.push(format!("{} state divergences", self.divergences.len()));
        }
        if let Some(reason) = &self.blocked_reason {
            parts.push(format!("blocked: {reason}"));
        }
        // A preserve-only repair explains nothing above, and an empty `daemon_recovered` summary
        // reads as "nothing happened" for the one repair a human most needs to find: the agent's
        // uncommitted work was snapshotted and is sitting on a checkpoint branch.
        if self
            .steps
            .iter()
            .any(|step| matches!(step, RecoveryStep::PreserveDirtyWork { .. }))
        {
            parts.push("preserved uncommitted work".into());
        }
        if parts.is_empty() {
            // Every repair counted above; this arm exists so a future step kind cannot make the
            // audit record silent.
            parts.push(format!("{} repairs", self.steps.len()));
        }
        parts.join("; ")
    }
}

/// Ask a tool whether its effect happened.
///
/// Implemented by the tool layer, which knows that a `git commit` is found by its nonce and that
/// a `shell` command is judged by the tree it produced. Core only needs the verdict.
///
/// `arguments` is the call's stored argument object, exactly as the `intent` record holds it.
/// It is not part of [`Intent`] because that type is compared and stored, and a multi-kilobyte
/// string there would make an accidental `PartialEq` the basis of a decision. It is a separate
/// parameter rather than extra fields on `Intent` for the same reason — but the probe genuinely
/// needs it: a `write_file` probe answers "the file already has this content" only by knowing
/// which content. `None` means the record did not carry it, and a probe that cannot work
/// without the arguments must answer [`Probe::Unknown`] rather than guess.
pub trait Prober {
    /// `Unknown` is a valid and expected answer, not a failure of the probe.
    fn probe(&self, intent: &Intent, arguments: Option<&serde_json::Value>) -> Probe;
}

/// A prober that refuses to guess. The default for callers that cannot probe yet.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoProbe;

impl Prober for NoProbe {
    fn probe(&self, _intent: &Intent, _arguments: Option<&serde_json::Value>) -> Probe {
        Probe::Unknown
    }
}

/// Everything recovery needs to decide, gathered by the caller.
#[derive(Debug)]
pub struct RecoveryInput<'a> {
    pub journal: &'a ReadOutcome,
    pub workspace: WorkspaceFacts,
    pub persisted: PersistedState,
    /// Last green checkpoint, the target when a probe cannot decide.
    pub last_checkpoint: Option<LastCheckpoint>,
    /// Branch this task's worktree must be on.
    pub expected_branch: &'a str,
}

/// The fallback target for L3.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LastCheckpoint {
    pub sha: String,
    pub kind: CheckpointKind,
}

/// Build the recovery plan.
///
/// `prober` is consulted only for calls whose idempotency class requires evidence; a probe that
/// answers `Unknown` is not an error, because `Unknown` is a legitimate verdict and leads to a
/// rollback rather than a guess.
#[must_use]
pub fn plan(input: &RecoveryInput<'_>, prober: &impl Prober) -> RecoveryReport {
    let journal_problems = input.journal.problems.clone();
    let dropped_tail_bytes = journal_problems
        .iter()
        .find_map(|problem| match problem {
            ReadProblem::TornTail { bytes, .. } => Some(*bytes),
            _ => None,
        })
        .unwrap_or(0);

    let mut steps = Vec::new();
    let mut verdict = if dropped_tail_bytes > 0 {
        steps.push(RecoveryStep::DropTornTail {
            bytes: dropped_tail_bytes,
        });
        Verdict::Repaired
    } else {
        Verdict::Healthy
    };

    if let Some(fatal) = journal_problems.iter().find(|problem| problem.is_fatal()) {
        let reason = describe_problem(fatal);
        steps.push(RecoveryStep::BlockTask {
            reason: reason.clone(),
        });
        // Nothing else is knowable: every later decision reads the journal, and a journal with a
        // broken chain describes a history that may not have happened. Stop here.
        return finish(RecoveryReport {
            verdict: Verdict::Blocked,
            dropped_tail_bytes,
            journal_problems,
            calls: Vec::new(),
            divergences: Vec::new(),
            blocked_reason: Some(reason),
            steps,
        });
    }

    // ── 2. Worktree ────────────────────────────────────────────────────────────────────────
    match worktree_action(input) {
        WorktreeAction::Intact => {}
        WorktreeAction::Repair(step) => {
            steps.push(step);
            verdict = Verdict::Repaired;
        }
        WorktreeAction::Block(reason) => {
            steps.push(RecoveryStep::BlockTask {
                reason: reason.clone(),
            });
            return finish(RecoveryReport {
                verdict: Verdict::Blocked,
                dropped_tail_bytes,
                journal_problems,
                calls: Vec::new(),
                divergences: Vec::new(),
                blocked_reason: Some(reason),
                steps,
            });
        }
    }

    // ── 3. Uncommitted work ────────────────────────────────────────────────────────────────
    // Before any probe: a probe inspects the tree, and a rollback destroys it.
    if input.workspace.dirty {
        steps.push(RecoveryStep::PreserveDirtyWork {
            reason: "uncommitted changes found after a restart".into(),
        });
        verdict = Verdict::Repaired;
    }

    // ── 4. Unfinished tool calls ───────────────────────────────────────────────────────────
    let reclaimed = reclaim_calls(input, prober, &mut verdict);
    let calls = reclaimed.calls;
    steps.extend(reclaimed.rollbacks);
    steps.extend(reclaimed.actions);

    // ── 5. Attempts ───────────────────────────────────────────────────────────────────────
    //
    // A plain restart resumes the interrupted attempt: its conversation is rebuilt from the
    // journal, which is why every tool result is journaled before the model sees it, and why
    // every unfinished call above gets an observation. Nothing is abandoned.
    //
    // A rollback is different. L3 is defined as a *fresh context with a different strategy*, so
    // the attempt that the crash interrupted is over and must be closed — otherwise it stays
    // `Running` forever and the subtask waits for a process that no longer exists.
    let rolling_back = steps
        .iter()
        .any(|step| matches!(step, RecoveryStep::RollbackToCheckpoint { .. }));
    if rolling_back {
        for (subtask, number) in open_attempts(input) {
            steps.push(RecoveryStep::AbandonAttempt { subtask, number });
        }
    }

    // ── 6. State-file divergence ───────────────────────────────────────────────────────────
    let divergences = reconcile(input);
    for divergence in &divergences {
        steps.push(RecoveryStep::RewriteStateFile {
            divergence: divergence.clone(),
        });
        verdict = Verdict::Repaired;
    }

    finish(RecoveryReport {
        verdict,
        dropped_tail_bytes,
        journal_problems,
        calls,
        divergences,
        blocked_reason: None,
        steps,
    })
}

/// What reclaiming the journal's unfinished calls decided.
struct Reclaimed {
    calls: Vec<CallRecovery>,
    /// Steps that rewrite the tree. They run before anything that reads or finishes it.
    rollbacks: Vec<RecoveryStep>,
    /// Steps that answer or re-issue a call.
    actions: Vec<RecoveryStep>,
}

/// Give every open call a disposition, in two ordered lists.
///
/// The order is not the journal's order. A rollback rewrites the tree, so a retry placed before it
/// would have its effect thrown away, and a synthesized observation would describe a world that no
/// longer exists. Every rollback therefore comes before every retry, whichever order the calls
/// appear in the journal.
fn reclaim_calls(input: &RecoveryInput<'_>, prober: &impl Prober, verdict: &mut Verdict) -> Reclaimed {
    let mut calls = Vec::new();
    let mut actions = Vec::new();
    let mut rollbacks = Vec::new();
    // Goes first in the rollback list because it describes a rollback the runner had already
    // decided on before the crash: it is not speculation about an unfinished call.
    if let Some(step) = interrupted_rollback(input, verdict) {
        rollbacks.push(step);
    }
    for found in crate::protocol::open_calls(&input.journal.records) {
        let recovered = reclaim_call(&found, prober, input, &mut actions, &mut rollbacks, verdict);
        calls.push(recovered);
    }
    // Two unknown calls often answer `Unknown` for the same reason and point at the same
    // checkpoint. Resetting twice rewrites the same tree, so duplicates are dropped: one reset is
    // enough, and the second would only obscure which call forced it.
    let mut seen = BTreeSet::new();
    rollbacks.retain(|step| match step {
        RecoveryStep::RollbackToCheckpoint { sha, .. } => seen.insert(sha.clone()),
        _ => true,
    });
    Reclaimed {
        calls,
        rollbacks,
        actions,
    }
}

/// Append the `record_recovery` step, which must come after the repair it describes.
///
/// # Why a blocked report gets no `RecordRecovery`
///
/// The only way to block is a journal whose chain is broken or whose sequence has a gap, and a
/// journal in that state cannot be *opened* for writing — `Journal::open` refuses rather than
/// appending after a break. A `RecordRecovery` step would therefore be a step whose only possible
/// outcome is an error, queued after the block it is meant to describe. The reason is carried in
/// `blocked_reason`, which the executor persists to `panic.json` and the escalation queue: places
/// that are readable without the journal, which is the property the audit record was wanted for.
fn finish(mut report: RecoveryReport) -> RecoveryReport {
    if report.verdict != Verdict::Healthy && report.verdict != Verdict::Blocked {
        let summary = report.summary();
        report.steps.push(RecoveryStep::RecordRecovery { summary });
    }
    report
}

/// What to do about the worktree.
enum WorktreeAction {
    Intact,
    Repair(RecoveryStep),
    Block(String),
}

/// Decide whether the worktree needs repair.
///
/// The four failure modes are kept apart because the safe reaction differs: a missing worktree is
/// rebuilt, a detached `HEAD` is checked out again, a worktree belonging to *another* task is left
/// alone, and a directory that exists but is not a repository stops the task. Forcing a checkout
/// on the wrong branch would destroy somebody else's work, and recreating over a non-repository
/// directory would destroy files nobody can identify afterwards — those are operator errors that
/// need a human, not a repair.
fn worktree_action(input: &RecoveryInput<'_>) -> WorktreeAction {
    let facts = &input.workspace;
    if !facts.exists {
        // `head` is what the worktree last checked out, if the caller could still read it. When it
        // is gone, the last checkpoint is the only known-good commit: restarting from the branch
        // tip would replay work that the checkpoint explicitly superseded.
        let sha = facts
            .head
            .clone()
            .or_else(|| input.last_checkpoint.as_ref().map(|c| c.sha.clone()));
        return WorktreeAction::Repair(RecoveryStep::RecreateWorktree {
            branch: input.expected_branch.to_string(),
            sha,
        });
    }
    if !facts.is_repository {
        return WorktreeAction::Block(format!(
            "`{}` exists but is not a git work tree; inspect it before recovery recreates it",
            input.expected_branch
        ));
    }
    match facts.branch.as_deref() {
        Some(branch) if branch == input.expected_branch => WorktreeAction::Intact,
        Some(other) => WorktreeAction::Block(format!(
            "worktree is on `{other}`, expected `{}`",
            input.expected_branch
        )),
        None => WorktreeAction::Repair(RecoveryStep::RestoreBranch {
            branch: input.expected_branch.to_string(),
        }),
    }
}

/// Decide the fate of one unfinished call and append the matching steps.
///
/// `Probe` and `VerifyContent` share a body on purpose: both ask the tool for evidence and then
/// branch on the answer. They differ only in *why* the evidence is needed — an unknown effect
/// versus a write whose content must be compared — and the difference is already carried by
/// [`CallRecovery::action`] for the executor to report.
///
/// Steps go to one of two lists: `rollbacks` collects the tree-rewriting steps that must run
/// before anything else, `actions` collects the rest. See the caller for why.
fn reclaim_call(
    found: &FoundOpenCall,
    prober: &impl Prober,
    input: &RecoveryInput<'_>,
    actions: &mut Vec<RecoveryStep>,
    rollbacks: &mut Vec<RecoveryStep>,
    verdict: &mut Verdict,
) -> CallRecovery {
    let open = &found.open;
    let action = open.required_action();
    let call = open.intent.call.clone();
    let tool = open.intent.tool.clone();
    let mut effect = None;

    match action {
        RecoveryAction::Retry => {
            // Safe to repeat, and no evidence exists to gather: `intent` without `started` on an
            // idempotent call means at worst it ran twice with the same result.
            //
            // This still counts as a repair: the journal holds a call the backend expects an
            // answer for, and re-running it is an action taken on the durable state. Leaving the
            // verdict `Healthy` here would skip `record_recovery` and make the restart invisible.
            *verdict = Verdict::Repaired;
            actions.push(RecoveryStep::RetryCall {
                call: call.clone(),
                tool,
                arguments: found.arguments.clone(),
            });
        }
        // `required_action` cannot return `Abandon` today; handled so that giving it a
        // non-recoverable class later does not silently fall through to a no-op.
        RecoveryAction::Abandon => {
            *verdict = Verdict::Repaired;
            actions.push(RecoveryStep::AbandonCall {
                call: call.clone(),
                reason: "not idempotent and no probe is available".into(),
            });
        }
        RecoveryAction::Probe | RecoveryAction::VerifyContent => {
            let answer = prober.probe(&open.intent, found.arguments.as_ref());
            effect = Some(answer);
            *verdict = Verdict::Repaired;
            match answer {
                Probe::Applied => actions.push(RecoveryStep::SynthesizeApplied {
                    call: call.clone(),
                    observation: format!(
                        "`{}` had already taken effect before the crash; it was not run again",
                        open.intent.expected_effect
                    ),
                }),
                Probe::NotApplied => actions.push(RecoveryStep::RetryCall {
                    call: call.clone(),
                    tool,
                    arguments: found.arguments.clone(),
                }),
                Probe::Unknown => match &input.last_checkpoint {
                    // L3. Guessing here is how a destructive command runs twice.
                    Some(checkpoint) => rollbacks.push(RecoveryStep::RollbackToCheckpoint {
                        sha: checkpoint.sha.clone(),
                        reason: format!(
                            "cannot determine whether `{}` had any effect",
                            open.intent.expected_effect
                        ),
                    }),
                    None => actions.push(RecoveryStep::AbandonCall {
                        call: call.clone(),
                        reason: "effect cannot be determined and there is no checkpoint to fall \
                                 back to"
                            .into(),
                    }),
                },
            }
        }
    }

    CallRecovery {
        call,
        action,
        scope: found.scope.clone(),
        effect,
    }
}

/// The rollback an interrupted L3 restart had already committed to, if there is one.
///
/// An L3 restart writes a marker before it moves the tree and `attempt_restarted` after. A crash
/// between the two leaves the tree in one of two states — the failed attempt's work, or the base —
/// and both are acceptable only if the next attempt starts from the base. Returns `None` when
/// there is nothing to restore, which is not an error: it is the normal case.
fn interrupted_rollback(input: &RecoveryInput<'_>, verdict: &mut Verdict) -> Option<RecoveryStep> {
    let sha = interrupted_restart(input)?;
    *verdict = Verdict::Repaired;
    Some(RecoveryStep::RollbackToCheckpoint {
        sha,
        reason: "an L3 restart was interrupted before it recorded the rollback".to_string(),
    })
}

/// The base of an L3 restart that was journalled but never completed.
///
/// The runner writes `checkpoint_created {role: "rollback", sha: <base>}` before it moves the
/// tree and `attempt_restarted` after. A crash between the two leaves a `reset` whose outcome
/// nobody knows, and the pair is the only evidence of it: the marker names the tree to restore,
/// and the missing `attempt_restarted` is what says the move was never confirmed.
///
/// Only the latest marker for an attempt can be interrupted — an earlier one whose restart *did*
/// complete has its `attempt_restarted` record after it — so the search walks backwards and stops
/// at the first `attempt_restarted`.
fn interrupted_restart(input: &RecoveryInput<'_>) -> Option<String> {
    // Walked backwards, so an `attempt_restarted` is seen before the marker it confirms.
    let mut confirmed = BTreeSet::new();
    for record in input.journal.records.iter().rev() {
        let (Some(subtask), Some(attempt)) = (record.fields.subtask.clone(), record.fields.attempt) else {
            continue;
        };
        let key = (subtask, attempt);
        match record.fields.kind {
            Kind::AttemptRestarted => {
                confirmed.insert(key);
            }
            Kind::CheckpointCreated
                if record.event.payload.get("role").and_then(|role| role.as_str()) == Some("rollback") =>
            {
                if confirmed.contains(&key) {
                    continue;
                }
                // A marker without a SHA names nothing to restore, so it is not evidence of a
                // rollback: the runner writes the SHA in the same record.
                if let Some(sha) = record.event.payload.get("sha").and_then(|sha| sha.as_str()) {
                    return Some(sha.to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Attempts the journal started and never closed.
///
/// Not a divergence: it is the journal's own account of a crash. Recovery must close them, or the
/// subtask waits forever for a `Running` attempt that no process owns.
fn open_attempts(input: &RecoveryInput<'_>) -> Vec<(SubtaskId, u32)> {
    let sets = attempt_sets(input);
    sets.started.difference(&sets.closed).cloned().collect()
}

/// The two attempt sets recovery derives from the journal.
struct AttemptSets {
    started: BTreeSet<(SubtaskId, u32)>,
    closed: BTreeSet<(SubtaskId, u32)>,
}

/// Attempts the journal started, and attempts it closed.
///
/// Closed means *ended*: failed, restarted, passed its gates, or implicitly succeeded when the
/// subtask was marked done. There is no `attempt_succeeded` kind, because success is a property
/// of the gates rather than of the attempt: the runner writes its `gate_result` verdict and stops,
/// and the scheduler closes the subtask afterwards.
///
/// The green-verdict arm is load-bearing rather than decorative. A verified attempt has no
/// `attempt_failed` record — it did not fail — so without this rule it would look open to
/// recovery, and the next start would abandon an attempt whose work is exactly what the tree is
/// about to close.
fn attempt_sets(input: &RecoveryInput<'_>) -> AttemptSets {
    let mut started = BTreeSet::new();
    let mut closed = BTreeSet::new();
    for record in &input.journal.records {
        let (Some(subtask), Some(attempt)) = (record.fields.subtask.clone(), record.fields.attempt) else {
            continue;
        };
        let key = (subtask, attempt);
        match record.fields.kind {
            Kind::AttemptStarted => {
                started.insert(key);
            }
            Kind::AttemptFailed | Kind::AttemptRestarted => {
                closed.insert(key);
            }
            Kind::GateResult if gate_verdict_passed(record) => {
                closed.insert(key);
            }
            Kind::SubtaskState if subtask_state_of(record) == Some(SubtaskState::Done) => {
                closed.insert(key);
            }
            _ => {}
        }
    }
    AttemptSets { started, closed }
}

/// Whether a record is the `gate_result` verdict that says the claim passed.
///
/// Anything unreadable is not a pass: a malformed verdict record must not be the reason an
/// attempt is considered closed, because closing means "this work is verified".
fn gate_verdict_passed(record: &JournalRecord) -> bool {
    let payload = &record.event.payload;
    payload.get("gate").and_then(|gate| gate.as_str()) == Some("verdict")
        && payload.get("passed").and_then(serde_json::Value::as_bool) == Some(true)
}

/// Compare the cache files with what the journal actually says.
///
/// Only two things are checked, because they are the two that change behaviour: whether an
/// attempt is over, and whether a subtask is closed. The rest of `subtasks.json` is presentation
/// and is rewritten wholesale from the journal anyway.
fn reconcile(input: &RecoveryInput<'_>) -> Vec<Divergence> {
    let mut divergences = Vec::new();
    let sets = attempt_sets(input);
    let (started, closed) = (sets.started, sets.closed);

    for persisted in &input.persisted.attempts {
        let key = (persisted.subtask.clone(), persisted.number);
        let in_journal = started.contains(&key) || closed.contains(&key);
        if !in_journal {
            // Nothing in the journal mentions this attempt: the cache is ahead of the record, or
            // an attempt counter was incremented before the journal was written. Either way the
            // entry must go, because it inflates the attempt budget for work that never began.
            divergences.push(Divergence::AttemptUnknown {
                subtask: persisted.subtask.clone(),
                number: persisted.number,
            });
            continue;
        }
        if persisted.finished && !closed.contains(&key) {
            divergences.push(Divergence::AttemptOpenInJournal {
                subtask: persisted.subtask.clone(),
                number: persisted.number,
            });
        }
        if !persisted.finished && closed.contains(&key) {
            divergences.push(Divergence::AttemptClosedInJournal {
                subtask: persisted.subtask.clone(),
                number: persisted.number,
            });
        }
    }

    // Subtasks: the last `subtask_state` record wins, because it is the last word written.
    let mut journal_states: BTreeMap<SubtaskId, SubtaskState> = BTreeMap::new();
    for record in &input.journal.records {
        if record.fields.kind != Kind::SubtaskState {
            continue;
        }
        let Some(subtask) = record.fields.subtask.clone() else {
            continue;
        };
        if let Some(state) = subtask_state_of(record) {
            journal_states.insert(subtask, state);
        }
    }

    for persisted in &input.persisted.subtasks {
        match journal_states.get(&persisted.id) {
            Some(journal) if journal != &persisted.state => {
                divergences.push(Divergence::SubtaskState {
                    subtask: persisted.id.clone(),
                    file: persisted.state,
                    journal: *journal,
                });
            }
            // A subtask the journal never mentioned exists only in the cache. Rewriting the file
            // from the journal drops it, which is the honest outcome: the journal is the record.
            None => divergences.push(Divergence::SubtaskUnknown {
                subtask: persisted.id.clone(),
            }),
            Some(_) => {}
        }
    }

    divergences
}

/// The state a `subtask_state` record announces, if the payload says so.
fn subtask_state_of(record: &JournalRecord) -> Option<SubtaskState> {
    record
        .event
        .payload
        .get("state")
        .and_then(|value| serde_json::from_value(value.clone()).ok())
}

fn describe_problem(problem: &ReadProblem) -> String {
    match problem {
        ReadProblem::TornTail { bytes, .. } => format!("torn journal tail of {bytes} bytes"),
        ReadProblem::ChainBreak { line, seq, .. } => {
            format!("journal chain broken at line {line} (seq {seq})")
        }
        ReadProblem::SequenceGap {
            line,
            expected,
            found,
            ..
        } => format!("journal sequence gap at line {line}: expected {expected}, found {found}"),
    }
}

/// Whether every unfinished call in the journal has a planned disposition.
///
/// Called by the daemon before executing a plan: a plan that leaves a `tool_calls` entry
/// unanswered produces a conversation the backend rejects, and the failure then surfaces hours
/// later as an inexplicable 400. Equality is checked in both directions, so a plan that invents
/// a call for a finished one is caught too.
#[must_use]
pub fn all_calls_addressed(journal: &[JournalRecord], report: &RecoveryReport) -> bool {
    let unfinished = crate::protocol::open_calls(journal);
    let unfinished: BTreeSet<&str> = unfinished
        .iter()
        .map(|found| found.open.intent.call.as_str())
        .collect();
    let planned: BTreeSet<&str> = report.calls.iter().map(|call| call.call.as_str()).collect();
    unfinished == planned
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{TaskId, Timestamp};
    use crate::ids::SubtaskId;
    use crate::journal::{JournalEvent, RecordFields};
    use chrono::TimeZone as _;

    fn task() -> TaskId {
        TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV")
    }

    fn subtask() -> SubtaskId {
        SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV")
    }

    fn at() -> Timestamp {
        chrono::Utc.with_ymd_and_hms(2026, 11, 12, 10, 14, 3).unwrap()
    }

    fn record(seq: u64, kind: Kind, payload: serde_json::Value) -> JournalRecord {
        JournalRecord::new(
            RecordFields::new(seq, at(), task(), kind),
            JournalEvent::with_payload(payload),
            None,
        )
    }

    fn scoped(seq: u64, attempt: u32, kind: Kind, payload: serde_json::Value) -> JournalRecord {
        JournalRecord::new(
            RecordFields {
                seq,
                at: at(),
                task: task(),
                subtask: Some(subtask()),
                attempt: Some(attempt),
                kind,
            },
            JournalEvent::with_payload(payload),
            None,
        )
    }

    fn outcome(records: Vec<JournalRecord>) -> ReadOutcome {
        ReadOutcome {
            records,
            problems: Vec::new(),
            has_torn_tail: false,
            last_digest: None,
        }
    }

    fn clean_journal() -> ReadOutcome {
        outcome(vec![record(0, Kind::TaskCreated, serde_json::json!({}))])
    }

    fn input(
        journal: &ReadOutcome,
        workspace: WorkspaceFacts,
        persisted: PersistedState,
    ) -> RecoveryInput<'_> {
        RecoveryInput {
            journal,
            workspace,
            persisted,
            last_checkpoint: None,
            expected_branch: "agent/tsk_01",
        }
    }

    /// A prober that always answers the same way, so each test names its scenario once.
    struct Fixed(Probe);

    impl Prober for Fixed {
        fn probe(&self, _intent: &Intent, _arguments: Option<&serde_json::Value>) -> Probe {
            self.0
        }
    }

    fn intent_payload(call: &str, idempotency: &str) -> serde_json::Value {
        serde_json::json!({
            "call": call,
            "tool": "shell",
            "idempotency": idempotency,
            "expected_effect": "remove build/",
        })
    }

    fn started_payload(call: &str) -> serde_json::Value {
        serde_json::json!({ "call": call, "pid": 4242 })
    }

    fn call_with(idempotency: &str, name: &str, started: bool) -> ReadOutcome {
        let mut records = vec![record(0, Kind::ToolCallIntent, intent_payload(name, idempotency))];
        if started {
            records.push(record(1, Kind::ToolCallStarted, started_payload(name)));
        }
        outcome(records)
    }

    fn step_of<'a>(report: &'a RecoveryReport, name: &str) -> Option<&'a RecoveryStep> {
        report
            .steps
            .iter()
            .find(|step| format!("{step:?}").starts_with(name))
    }

    // ── healthy and journal problems ───────────────────────────────────────────────────────

    #[test]
    fn a_clean_shutdown_needs_nothing() {
        let journal = clean_journal();
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc123"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert_eq!(report.verdict, Verdict::Healthy);
        assert_eq!(report.steps.len(), 0, "{:?}", report.steps);
        assert_eq!(report.summary(), "clean");
    }

    #[test]
    fn a_torn_tail_is_dropped_counted_and_recorded() {
        let mut journal = clean_journal();
        journal.problems.push(ReadProblem::TornTail {
            segment: "JOURNAL.jsonl".into(),
            bytes: 137,
        });
        journal.has_torn_tail = true;
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert_eq!(report.verdict, Verdict::Repaired);
        assert_eq!(report.dropped_tail_bytes, 137);
        assert!(matches!(
            report.steps.first(),
            Some(RecoveryStep::DropTornTail { bytes: 137 })
        ));
        assert!(report.summary().contains("137 bytes"), "{}", report.summary());
        assert!(matches!(
            report.steps.last(),
            Some(RecoveryStep::RecordRecovery { .. })
        ));
    }

    #[test]
    fn a_chain_break_blocks_and_stops_the_plan() {
        let mut journal = clean_journal();
        journal.problems.push(ReadProblem::ChainBreak {
            segment: "JOURNAL.jsonl".into(),
            line: 4,
            seq: 3,
        });
        let report = plan(
            &input(&journal, WorkspaceFacts::absent(), PersistedState::default()),
            &NoProbe,
        );
        assert_eq!(report.verdict, Verdict::Blocked);
        assert!(
            report
                .blocked_reason
                .as_deref()
                .unwrap_or_default()
                .contains("chain broken"),
            "{:?}",
            report.blocked_reason
        );
        // No worktree repair, no probes: an untrustworthy journal makes every later decision a
        // guess about a history that may not have happened.
        //
        // Only the block: a `RecordRecovery` step would target the journal that was just declared
        // unopenable, so the reason travels in `blocked_reason` instead.
        assert_eq!(report.steps.len(), 1, "block only, got {:?}", report.steps);
        assert!(matches!(report.steps[0], RecoveryStep::BlockTask { .. }));
        assert_eq!(report.divergences.len(), 0);
    }

    #[test]
    fn a_sequence_gap_blocks_the_task() {
        let mut journal = clean_journal();
        journal.problems.push(ReadProblem::SequenceGap {
            segment: "JOURNAL.jsonl".into(),
            line: 2,
            expected: 1,
            found: 5,
        });
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert_eq!(report.verdict, Verdict::Blocked);
        assert!(
            report
                .blocked_reason
                .as_deref()
                .unwrap_or_default()
                .contains("sequence gap")
        );
    }

    #[test]
    fn a_torn_tail_does_not_block() {
        let mut journal = clean_journal();
        journal.problems.push(ReadProblem::TornTail {
            segment: "JOURNAL.jsonl".into(),
            bytes: 10,
        });
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert_ne!(report.verdict, Verdict::Blocked);
    }

    // ── worktree ───────────────────────────────────────────────────────────────────────────

    #[test]
    fn a_missing_worktree_is_recreated() {
        let journal = clean_journal();
        let report = plan(
            &input(&journal, WorkspaceFacts::absent(), PersistedState::default()),
            &NoProbe,
        );
        assert_eq!(report.verdict, Verdict::Repaired);
        assert!(matches!(
            report.steps.first(),
            Some(RecoveryStep::RecreateWorktree { branch, sha: None }) if branch == "agent/tsk_01"
        ));
    }

    #[test]
    fn a_lost_worktree_restarts_from_the_last_checkpoint() {
        let journal = clean_journal();
        let mut input = input(&journal, WorkspaceFacts::absent(), PersistedState::default());
        input.last_checkpoint = Some(LastCheckpoint {
            sha: "feedface".into(),
            kind: CheckpointKind::Hidden,
        });
        let report = plan(&input, &NoProbe);
        assert!(matches!(
            report.steps.first(),
            Some(RecoveryStep::RecreateWorktree { sha: Some(sha), .. }) if sha == "feedface"
        ));
    }

    #[test]
    fn a_directory_that_is_not_a_repository_blocks_instead_of_being_overwritten() {
        // Recreating over it would delete files that recovery cannot identify or restore.
        let journal = clean_journal();
        let facts = WorkspaceFacts {
            exists: true,
            is_repository: false,
            branch: None,
            head: None,
            dirty: false,
        };
        let report = plan(&input(&journal, facts, PersistedState::default()), &NoProbe);
        assert_eq!(report.verdict, Verdict::Blocked);
        assert!(
            !report
                .steps
                .iter()
                .any(|step| matches!(step, RecoveryStep::RecreateWorktree { .. })),
            "{:?}",
            report.steps
        );
        assert!(
            report
                .blocked_reason
                .as_deref()
                .unwrap_or_default()
                .contains("not a git work tree")
        );
    }

    #[test]
    fn a_detached_worktree_restores_its_branch() {
        let journal = clean_journal();
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc").detached(),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert_eq!(report.verdict, Verdict::Repaired);
        assert!(matches!(
            report.steps.first(),
            Some(RecoveryStep::RestoreBranch { branch }) if branch == "agent/tsk_01"
        ));
    }

    #[test]
    fn a_worktree_on_a_foreign_branch_blocks_and_is_not_forced() {
        let journal = clean_journal();
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_OTHER", "abc"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert_eq!(report.verdict, Verdict::Blocked);
        assert!(
            report
                .blocked_reason
                .as_deref()
                .unwrap_or_default()
                .contains("agent/tsk_OTHER")
        );
        // No checkout step: forcing it would destroy another task's work.
        assert!(
            !report
                .steps
                .iter()
                .any(|step| matches!(step, RecoveryStep::RestoreBranch { .. }))
        );
    }

    #[test]
    fn uncommitted_work_is_preserved_before_any_probe() {
        let journal = call_with("unknown", "call_dirty", true);
        let mut input = input(
            &journal,
            WorkspaceFacts::healthy("agent/tsk_01", "abc").dirty(),
            PersistedState::default(),
        );
        input.last_checkpoint = Some(LastCheckpoint {
            sha: "deadbeef".into(),
            kind: CheckpointKind::Hidden,
        });
        let report = plan(&input, &Fixed(Probe::Unknown));
        assert_eq!(report.verdict, Verdict::Repaired);
        let index = report
            .steps
            .iter()
            .position(|step| matches!(step, RecoveryStep::PreserveDirtyWork { .. }))
            .expect("a preserve step");
        let rollback = report
            .steps
            .iter()
            .position(|step| matches!(step, RecoveryStep::RollbackToCheckpoint { .. }))
            .expect("a rollback step");
        assert!(
            index < rollback,
            "the snapshot must come before the rollback that destroys the tree: {:?}",
            report.steps
        );
    }

    // ── unfinished tool calls ─────────────────────────────────────────────────────────────

    #[test]
    fn an_unstarted_idempotent_call_is_retried() {
        let journal = call_with("idempotent", "call_a", false);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert_eq!(report.calls.len(), 1);
        assert!(matches!(
            step_of(&report, "RetryCall"),
            Some(RecoveryStep::RetryCall { call, .. }) if call.as_str() == "call_a"
        ));
        assert!(
            !report.calls[0].needs_synthesized_observation(),
            "a retried call answers its own tool_calls entry"
        );
        // A retry is a repair, not a clean start: otherwise the restart leaves no journal trace
        // and a later investigation cannot tell why a command ran twice.
        assert_eq!(report.verdict, Verdict::Repaired);
        assert!(matches!(
            report.steps.last(),
            Some(RecoveryStep::RecordRecovery { .. })
        ));
    }

    #[test]
    fn a_retried_call_carries_the_arguments_it_was_issued_with() {
        // Re-issuing a call means re-issuing *that* call. A plan that said "run `shell` again"
        // without the command would either force the executor to invent one — a second, possibly
        // destructive, command nobody asked for — or to give up on an idempotent call that is
        // safe to repeat. The stored arguments travel with the step.
        let journal = outcome(vec![record(
            0,
            Kind::ToolCallIntent,
            serde_json::json!({
                "call": "call_shell",
                "tool": "shell",
                "idempotency": "idempotent",
                "expected_effect": "run cargo fmt",
                "arguments": {"command": "cargo fmt --all"},
            }),
        )]);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert_eq!(
            step_of(&report, "RetryCall"),
            Some(&RecoveryStep::RetryCall {
                call: CallId::from_model("call_shell"),
                tool: "shell".into(),
                arguments: Some(serde_json::json!({"command": "cargo fmt --all"})),
            })
        );
    }

    #[test]
    fn a_retried_call_without_stored_arguments_says_so() {
        // An older `intent` record may not carry `arguments`. `None` is the honest answer and the
        // executor must close the call with an observation; nothing here may guess.
        let journal = call_with("idempotent", "call_bare", false);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert!(matches!(
            step_of(&report, "RetryCall"),
            Some(RecoveryStep::RetryCall { arguments: None, .. })
        ));
    }

    #[test]
    fn an_unknown_call_with_a_positive_probe_is_not_rerun() {
        let journal = call_with("unknown", "call_b", true);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &Fixed(Probe::Applied),
        );
        assert_eq!(report.calls[0].effect, Some(Probe::Applied));
        assert!(report.calls[0].needs_synthesized_observation());
        assert!(matches!(
            step_of(&report, "SynthesizeApplied"),
            Some(RecoveryStep::SynthesizeApplied { call, .. }) if call.as_str() == "call_b"
        ));
        assert!(
            step_of(&report, "RetryCall").is_none(),
            "a present effect must never be re-run: {:?}",
            report.steps
        );
    }

    #[test]
    fn an_unknown_call_with_a_negative_probe_is_retried() {
        let journal = call_with("unknown", "call_c", true);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &Fixed(Probe::NotApplied),
        );
        assert_eq!(report.calls[0].effect, Some(Probe::NotApplied));
        assert!(matches!(
            step_of(&report, "RetryCall"),
            Some(RecoveryStep::RetryCall { call, .. }) if call.as_str() == "call_c"
        ));
    }

    #[test]
    fn an_indeterminate_probe_rolls_back_to_the_checkpoint() {
        let journal = call_with("unknown", "call_d", true);
        let mut input = input(
            &journal,
            WorkspaceFacts::healthy("agent/tsk_01", "abc"),
            PersistedState::default(),
        );
        input.last_checkpoint = Some(LastCheckpoint {
            sha: "deadbeef".into(),
            kind: CheckpointKind::Hidden,
        });
        let report = plan(&input, &NoProbe);
        assert_eq!(report.verdict, Verdict::Repaired);
        assert!(matches!(
            step_of(&report, "RollbackToCheckpoint"),
            Some(RecoveryStep::RollbackToCheckpoint { sha, .. }) if sha == "deadbeef"
        ));
    }

    #[test]
    fn an_indeterminate_probe_without_a_checkpoint_abandons_the_call() {
        let journal = call_with("unknown", "call_e", true);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert!(matches!(
            step_of(&report, "AbandonCall"),
            Some(RecoveryStep::AbandonCall { call, .. }) if call.as_str() == "call_e"
        ));
        assert!(
            step_of(&report, "RetryCall").is_none(),
            "an undecidable effect must never be retried"
        );
    }

    #[test]
    fn by_content_calls_are_verified_with_the_probe() {
        let journal = call_with("by_content", "call_f", true);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &Fixed(Probe::Applied),
        );
        assert_eq!(report.calls[0].action, RecoveryAction::VerifyContent);
        assert_eq!(report.calls[0].effect, Some(Probe::Applied));
    }

    #[test]
    fn the_prober_receives_the_stored_arguments() {
        // `VerifyContent` exists for writes, and a write's content is the arguments. A planner
        // that handed the prober only the intent would force every `ByContent` probe to answer
        // `Unknown`, which rolls the tree back to the last checkpoint and discards a file that
        // may be exactly what the crash left complete.
        use std::sync::Mutex;

        #[derive(Debug)]
        struct Recording(Mutex<Vec<Option<serde_json::Value>>>);

        impl Prober for Recording {
            fn probe(&self, _intent: &Intent, arguments: Option<&serde_json::Value>) -> Probe {
                self.0.lock().unwrap().push(arguments.cloned());
                Probe::Applied
            }
        }

        let journal = outcome(vec![
            record(
                0,
                Kind::ToolCallIntent,
                serde_json::json!({
                    "call": "call_write",
                    "tool": "write_file",
                    "idempotency": "by_content",
                    "expected_effect": "write src/lib.rs",
                    "arguments": {"path": "src/lib.rs", "content": "fn main() {}\n"},
                }),
            ),
            record(1, Kind::ToolCallStarted, started_payload("call_write")),
        ]);
        let recorder = Recording(Mutex::new(Vec::new()));
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &recorder,
        );

        assert_eq!(report.calls[0].action, RecoveryAction::VerifyContent);
        assert_eq!(
            recorder.0.lock().unwrap().as_slice(),
            [Some(serde_json::json!({
                "path": "src/lib.rs",
                "content": "fn main() {}\n",
            }))]
        );
    }

    #[test]
    fn an_unstarted_unknown_call_is_still_probed() {
        // The launch window: `intent` is durable, the process may already be running, `started`
        // was never written. Nothing about the missing record proves the command did not run.
        let journal = call_with("unknown", "call_window", false);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert_eq!(report.calls[0].action, RecoveryAction::Probe);
    }

    #[test]
    fn every_unfinished_call_gets_a_disposition() {
        let journal = outcome(vec![
            record(0, Kind::ToolCallIntent, intent_payload("call_g", "unknown")),
            record(1, Kind::ToolCallStarted, started_payload("call_g")),
            record(2, Kind::ToolCallIntent, intent_payload("call_h", "idempotent")),
        ]);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &Fixed(Probe::NotApplied),
        );
        assert_eq!(report.calls.len(), 2);
        assert!(all_calls_addressed(&journal.records, &report));
        // One is retried, one had a probe: exactly one needs an invented observation.
        assert_eq!(
            report
                .calls
                .iter()
                .filter(|call| call.needs_synthesized_observation())
                .count(),
            1,
            "{:?}",
            report.calls
        );
    }

    #[test]
    fn a_missed_call_is_caught_by_all_calls_addressed() {
        // Structural guard: if a future edit stops emitting a disposition for some class of
        // call, the daemon's precondition fails loudly instead of resuming a conversation with a
        // dangling `tool_calls` entry.
        let journal = call_with("unknown", "call_i", true);
        let mut report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &Fixed(Probe::Unknown),
        );
        assert!(all_calls_addressed(&journal.records, &report));
        report.calls.clear();
        assert!(!all_calls_addressed(&journal.records, &report));
    }

    #[test]
    fn a_rollback_runs_before_any_retry_whatever_the_journal_order() {
        // The idempotent call comes first in the journal, so a naive plan would retry it and only
        // then roll the tree back — throwing the retry's effect away, or worse, reporting an
        // observation for a file that no longer exists.
        let journal = outcome(vec![
            record(
                0,
                Kind::ToolCallIntent,
                intent_payload("call_early", "idempotent"),
            ),
            record(1, Kind::ToolCallIntent, intent_payload("call_late", "unknown")),
            record(2, Kind::ToolCallStarted, started_payload("call_late")),
        ]);
        let mut input = input(
            &journal,
            WorkspaceFacts::healthy("agent/tsk_01", "abc"),
            PersistedState::default(),
        );
        input.last_checkpoint = Some(LastCheckpoint {
            sha: "cafe".into(),
            kind: CheckpointKind::Hidden,
        });
        let report = plan(&input, &NoProbe);
        let rollback = report
            .steps
            .iter()
            .position(|step| matches!(step, RecoveryStep::RollbackToCheckpoint { .. }))
            .expect("a rollback step");
        let retry = report
            .steps
            .iter()
            .position(|step| {
                matches!(
                    step,
                    RecoveryStep::RetryCall { call, .. } if call.as_str() == "call_early"
                )
            })
            .expect("a retry for the idempotent call");
        assert!(
            rollback < retry,
            "a retry must never run before the rollback that rewrites the tree: {:?}",
            report.steps
        );
    }

    #[test]
    fn two_unknown_calls_sharing_a_checkpoint_roll_back_once() {
        let journal = outcome(vec![
            record(0, Kind::ToolCallIntent, intent_payload("call_one", "unknown")),
            record(1, Kind::ToolCallStarted, started_payload("call_one")),
            record(2, Kind::ToolCallIntent, intent_payload("call_two", "unknown")),
            record(3, Kind::ToolCallStarted, started_payload("call_two")),
        ]);
        let mut input = input(
            &journal,
            WorkspaceFacts::healthy("agent/tsk_01", "abc"),
            PersistedState::default(),
        );
        input.last_checkpoint = Some(LastCheckpoint {
            sha: "cafe".into(),
            kind: CheckpointKind::Hidden,
        });
        let report = plan(&input, &Fixed(Probe::Unknown));
        let rollbacks = report
            .steps
            .iter()
            .filter(|step| matches!(step, RecoveryStep::RollbackToCheckpoint { .. }))
            .count();
        assert_eq!(rollbacks, 1, "{:?}", report.steps);
        // Both calls are still accounted for; only the redundant reset is gone.
        assert_eq!(report.calls.len(), 2);
    }

    #[test]
    fn an_attempt_unknown_to_the_journal_is_reported() {
        // A cache entry with no journal counterpart inflates `max_attempts_per_subtask` for work
        // that never began, so it is a divergence even though nothing is "open" in the journal.
        let journal = clean_journal();
        let persisted = PersistedState {
            attempts: vec![PersistedAttempt {
                subtask: subtask(),
                number: 7,
                finished: false,
            }],
            subtasks: Vec::new(),
        };
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                persisted,
            ),
            &NoProbe,
        );
        assert_eq!(
            report.divergences.as_slice(),
            [Divergence::AttemptUnknown {
                subtask: subtask(),
                number: 7
            }]
        );
        assert_eq!(report.verdict, Verdict::Repaired);
    }

    #[test]
    fn an_invented_call_is_caught_by_all_calls_addressed() {
        let journal = clean_journal();
        let mut report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        report.calls.push(CallRecovery {
            call: CallId::from_model("call_ghost"),
            action: RecoveryAction::Retry,
            scope: crate::journal::Scope::task(),
            effect: None,
        });
        assert!(!all_calls_addressed(&journal.records, &report));
    }

    #[test]
    fn call_ids_in_the_plan_are_the_call_ids_in_the_journal() {
        let journal = outcome(vec![
            record(0, Kind::ToolCallIntent, intent_payload("call_x", "idempotent")),
            record(1, Kind::ToolCallIntent, intent_payload("call_y", "idempotent")),
        ]);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        let ids: Vec<&str> = report.calls.iter().map(|call| call.call.as_str()).collect();
        assert_eq!(ids, ["call_x", "call_y"]);
    }

    #[test]
    fn a_journal_written_to_disk_reads_back_into_the_same_plan() {
        // The unit tests build `ReadOutcome` by hand, so a change in what the writer actually
        // stores (field names, `call` placement) would slip through. This one goes through a real
        // journal, a real torn tail and the real reader.
        let dir = tempfile::tempdir().unwrap();
        let task = task();
        {
            let mut journal = crate::journal::Journal::open(dir.path(), task.clone()).unwrap();
            journal
                .append_event(at(), Kind::ToolCallIntent, intent_payload("call_disk", "unknown"))
                .unwrap();
            journal
                .append_event(
                    at(),
                    Kind::ToolCallStarted,
                    serde_json::json!({ "call": "call_disk", "pid": 1 }),
                )
                .unwrap();
            journal.close().unwrap();
        }
        // A torn tail, exactly as a crash would leave it.
        let path = dir.path().join("JOURNAL.jsonl");
        let torn = b"{\"v\":1,\"seq\":2,\"ts\":";
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(torn);
        std::fs::write(&path, &bytes).unwrap();

        let journal = crate::journal::JournalReader::open(dir.path(), &task)
            .unwrap()
            .read()
            .unwrap();
        assert!(journal.has_torn_tail);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert_eq!(report.dropped_tail_bytes, torn.len() as u64);
        assert_eq!(report.calls.len(), 1);
        assert!(all_calls_addressed(&journal.records, &report));
        assert_eq!(
            report.calls[0].action,
            RecoveryAction::Probe,
            "the idempotency class must survive the round trip through JSON"
        );
        assert!(matches!(
            report.steps.last(),
            Some(RecoveryStep::RecordRecovery { .. })
        ));
    }

    // ── state-file divergence: the journal always wins ────────────────────────────────────

    #[test]
    fn a_rollback_abandons_the_interrupted_attempt() {
        // L3: fresh context. The interrupted attempt must be closed, or it stays `Running`
        // forever and the subtask waits for a process that no longer exists.
        let journal = outcome(vec![
            scoped(0, 3, Kind::AttemptStarted, serde_json::json!({})),
            record(1, Kind::ToolCallIntent, intent_payload("call_l3", "unknown")),
            record(2, Kind::ToolCallStarted, started_payload("call_l3")),
        ]);
        let mut input = input(
            &journal,
            WorkspaceFacts::healthy("agent/tsk_01", "abc"),
            PersistedState::default(),
        );
        input.last_checkpoint = Some(LastCheckpoint {
            sha: "deadbeef".into(),
            kind: CheckpointKind::Hidden,
        });
        let report = plan(&input, &NoProbe);
        let rollback = report
            .steps
            .iter()
            .position(|step| matches!(step, RecoveryStep::RollbackToCheckpoint { .. }))
            .expect("a rollback");
        let abandon = report
            .steps
            .iter()
            .position(|step| matches!(step, RecoveryStep::AbandonAttempt { number: 3, .. }))
            .expect("the attempt must be closed for a fresh one to start");
        // Closed after the reset that supersedes its work, never before.
        assert!(rollback < abandon, "{:?}", report.steps);
    }

    #[test]
    fn a_plain_restart_resumes_the_attempt_instead_of_abandoning_it() {
        // No rollback means the work stands: the same attempt continues with its conversation
        // replayed from the journal. Abandoning here would throw the context away for free on
        // every daemon restart.
        let journal = outcome(vec![scoped(0, 2, Kind::AttemptStarted, serde_json::json!({}))]);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert_eq!(report.verdict, Verdict::Healthy);
        assert!(
            !report
                .steps
                .iter()
                .any(|step| matches!(step, RecoveryStep::AbandonAttempt { .. })),
            "{:?}",
            report.steps
        );
    }

    #[test]
    fn a_closed_attempt_is_not_abandoned() {
        let journal = outcome(vec![
            scoped(0, 1, Kind::AttemptStarted, serde_json::json!({})),
            scoped(1, 1, Kind::AttemptFailed, serde_json::json!({})),
        ]);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert!(
            !report
                .steps
                .iter()
                .any(|step| matches!(step, RecoveryStep::AbandonAttempt { .. })),
            "{:?}",
            report.steps
        );
    }

    #[test]
    fn an_attempt_closed_only_in_the_file_is_reported_as_open() {
        let journal = outcome(vec![scoped(0, 1, Kind::AttemptStarted, serde_json::json!({}))]);
        let persisted = PersistedState {
            attempts: vec![PersistedAttempt {
                subtask: subtask(),
                number: 1,
                finished: true,
            }],
            subtasks: Vec::new(),
        };
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                persisted,
            ),
            &NoProbe,
        );
        assert_eq!(report.verdict, Verdict::Repaired);
        assert!(matches!(
            report.divergences.as_slice(),
            [Divergence::AttemptOpenInJournal { number: 1, .. }]
        ));
        assert!(matches!(
            step_of(&report, "RewriteStateFile"),
            Some(RecoveryStep::RewriteStateFile { .. })
        ));
    }

    #[test]
    fn an_attempt_closed_in_the_journal_wins_over_the_file() {
        let journal = outcome(vec![
            scoped(0, 1, Kind::AttemptStarted, serde_json::json!({})),
            scoped(1, 1, Kind::AttemptFailed, serde_json::json!({})),
        ]);
        let persisted = PersistedState {
            attempts: vec![PersistedAttempt {
                subtask: subtask(),
                number: 1,
                finished: false,
            }],
            subtasks: Vec::new(),
        };
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                persisted,
            ),
            &NoProbe,
        );
        assert!(matches!(
            report.divergences.as_slice(),
            [Divergence::AttemptClosedInJournal { number: 1, .. }]
        ));
    }

    #[test]
    fn a_successful_attempt_is_closed_by_the_subtask_state_record() {
        // There is no `attempt_succeeded` kind: success is announced by `subtask_state = done`.
        let journal = outcome(vec![
            scoped(0, 1, Kind::AttemptStarted, serde_json::json!({})),
            scoped(1, 1, Kind::SubtaskState, serde_json::json!({ "state": "done" })),
        ]);
        let persisted = PersistedState {
            attempts: vec![PersistedAttempt {
                subtask: subtask(),
                number: 1,
                finished: false,
            }],
            subtasks: Vec::new(),
        };
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                persisted,
            ),
            &NoProbe,
        );
        assert!(matches!(
            report.divergences.as_slice(),
            [Divergence::AttemptClosedInJournal { number: 1, .. }]
        ));
    }

    #[test]
    fn a_green_gate_verdict_closes_the_attempt_without_a_subtask_state() {
        // A verified attempt has no `attempt_failed`: it did not fail. If only `attempt_failed`
        // and `subtask_state = done` closed attempts, a crash between the verdict and the
        // scheduler's write would make recovery abandon work the tree is about to close.
        let journal = outcome(vec![
            scoped(0, 1, Kind::AttemptStarted, serde_json::json!({})),
            scoped(
                1,
                1,
                Kind::GateResult,
                serde_json::json!({ "gate": "verdict", "passed": true }),
            ),
        ]);
        let persisted = PersistedState {
            attempts: vec![PersistedAttempt {
                subtask: subtask(),
                number: 1,
                finished: false,
            }],
            subtasks: Vec::new(),
        };
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                persisted,
            ),
            &NoProbe,
        );
        assert!(matches!(
            report.divergences.as_slice(),
            [Divergence::AttemptClosedInJournal { number: 1, .. }]
        ));
    }

    #[test]
    fn a_red_gate_verdict_does_not_close_the_attempt() {
        // A red claim is closed by `attempt_failed`, which the runner writes. Treating a red
        // verdict as a closure here would make a failed attempt look finished and successful.
        let journal = outcome(vec![
            scoped(0, 1, Kind::AttemptStarted, serde_json::json!({})),
            scoped(
                1,
                1,
                Kind::GateResult,
                serde_json::json!({ "gate": "verdict", "passed": false }),
            ),
        ]);
        let persisted = PersistedState {
            attempts: vec![PersistedAttempt {
                subtask: subtask(),
                number: 1,
                finished: true,
            }],
            subtasks: Vec::new(),
        };
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                persisted,
            ),
            &NoProbe,
        );
        assert!(matches!(
            report.divergences.as_slice(),
            [Divergence::AttemptOpenInJournal { number: 1, .. }]
        ));
    }

    #[test]
    fn a_malformed_gate_record_is_not_a_pass() {
        // `passed: "yes"` and a missing `gate` tag are both "not the verdict". A lenient read
        // here would let a corrupt line close an attempt nobody verified.
        for payload in [
            serde_json::json!({ "gate": "verdict", "passed": "yes" }),
            serde_json::json!({ "passed": true }),
            serde_json::json!({ "gate": "check", "passed": true }),
            serde_json::json!({}),
        ] {
            let journal = outcome(vec![
                scoped(0, 1, Kind::AttemptStarted, serde_json::json!({})),
                scoped(1, 1, Kind::GateResult, payload.clone()),
            ]);
            let persisted = PersistedState {
                attempts: vec![PersistedAttempt {
                    subtask: subtask(),
                    number: 1,
                    finished: true,
                }],
                subtasks: Vec::new(),
            };
            let report = plan(
                &input(
                    &journal,
                    WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                    persisted,
                ),
                &NoProbe,
            );
            assert!(
                matches!(
                    report.divergences.as_slice(),
                    [Divergence::AttemptOpenInJournal { number: 1, .. }]
                ),
                "{payload} was read as a closure"
            );
        }
    }

    #[test]
    fn a_subtask_unknown_to_the_journal_is_reported() {
        let journal = clean_journal();
        let persisted = PersistedState {
            attempts: Vec::new(),
            subtasks: vec![PersistedSubtask {
                id: subtask(),
                state: SubtaskState::Done,
            }],
        };
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                persisted,
            ),
            &NoProbe,
        );
        assert!(matches!(
            report.divergences.as_slice(),
            [Divergence::SubtaskUnknown { .. }]
        ));
    }

    #[test]
    fn a_subtask_state_mismatch_names_both_sides() {
        let journal = outcome(vec![scoped(
            0,
            1,
            Kind::SubtaskState,
            serde_json::json!({ "state": "done" }),
        )]);
        let persisted = PersistedState {
            attempts: Vec::new(),
            subtasks: vec![PersistedSubtask {
                id: subtask(),
                state: SubtaskState::Running,
            }],
        };
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                persisted,
            ),
            &NoProbe,
        );
        assert!(matches!(
            report.divergences.as_slice(),
            [Divergence::SubtaskState {
                file: SubtaskState::Running,
                journal: SubtaskState::Done,
                ..
            }]
        ));
    }

    #[test]
    fn agreeing_state_files_produce_no_divergence() {
        let journal = outcome(vec![
            scoped(0, 1, Kind::AttemptStarted, serde_json::json!({})),
            scoped(1, 1, Kind::AttemptFailed, serde_json::json!({})),
            scoped(2, 1, Kind::SubtaskState, serde_json::json!({ "state": "done" })),
        ]);
        let persisted = PersistedState {
            attempts: vec![PersistedAttempt {
                subtask: subtask(),
                number: 1,
                finished: true,
            }],
            subtasks: vec![PersistedSubtask {
                id: subtask(),
                state: SubtaskState::Done,
            }],
        };
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                persisted,
            ),
            &NoProbe,
        );
        assert_eq!(report.verdict, Verdict::Healthy);
        assert_eq!(report.divergences.len(), 0);
    }

    #[test]
    fn a_payload_without_a_state_field_does_not_close_anything() {
        // A malformed `subtask_state` record must not be read as some default state.
        let journal = outcome(vec![
            scoped(0, 1, Kind::AttemptStarted, serde_json::json!({})),
            scoped(1, 1, Kind::SubtaskState, serde_json::json!({})),
        ]);
        let persisted = PersistedState {
            attempts: Vec::new(),
            subtasks: vec![PersistedSubtask {
                id: subtask(),
                state: SubtaskState::Running,
            }],
        };
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc"),
                persisted,
            ),
            &NoProbe,
        );
        assert_eq!(
            report.divergences.as_slice(),
            [Divergence::SubtaskUnknown { subtask: subtask() }]
        );
    }

    // ── report shape ───────────────────────────────────────────────────────────────────────

    #[test]
    fn the_record_step_states_everything_that_was_repaired() {
        let mut journal = call_with("unknown", "call_z", true);
        journal.problems.push(ReadProblem::TornTail {
            segment: "JOURNAL.jsonl".into(),
            bytes: 4,
        });
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc").dirty(),
                PersistedState::default(),
            ),
            &Fixed(Probe::Applied),
        );
        let Some(RecoveryStep::RecordRecovery { summary }) = report.steps.last() else {
            panic!("last step must be record_recovery: {:?}", report.steps);
        };
        assert!(summary.contains("4 bytes"), "{summary}");
        assert!(summary.contains("1 unfinished tool calls"), "{summary}");
    }

    #[test]
    fn a_blocked_task_carries_its_reason_outside_the_journal() {
        // The journal that blocks a task is the one that cannot be written, so the reason has to
        // be carried on the report. This test pins that the reason survives, and that no step
        // targets the broken journal.
        let mut journal = clean_journal();
        journal.problems.push(ReadProblem::ChainBreak {
            segment: "JOURNAL.jsonl".into(),
            line: 2,
            seq: 1,
        });
        let report = plan(
            &input(&journal, WorkspaceFacts::absent(), PersistedState::default()),
            &NoProbe,
        );
        let reason = report.blocked_reason.as_deref().unwrap_or_default();
        assert!(reason.contains("chain broken"), "{reason}");
        assert!(
            !report
                .steps
                .iter()
                .any(|step| matches!(step, RecoveryStep::RecordRecovery { .. })),
            "a record_recovery step would target the unopenable journal: {:?}",
            report.steps
        );
    }

    // ── interrupted L3 restarts ───────────────────────────────────────────────────────────

    fn checkpoint_record(seq: u64, attempt: u32, role: &str, sha: &str) -> JournalRecord {
        scoped(
            seq,
            attempt,
            Kind::CheckpointCreated,
            serde_json::json!({ "sha": sha, "branch": "agent-ckpt/tsk_01", "role": role }),
        )
    }

    fn restarted(seq: u64, attempt: u32, next: u32) -> JournalRecord {
        scoped(
            seq,
            attempt,
            Kind::AttemptRestarted,
            serde_json::json!({ "number": attempt, "next": next, "cause": "d3" }),
        )
    }

    #[test]
    fn a_marker_without_its_restart_record_plans_the_rollback() {
        // The crash window the marker exists for: the journal names the tree to restore, and the
        // missing `attempt_restarted` says the move was never confirmed.
        let journal = outcome(vec![
            scoped(0, 1, Kind::AttemptStarted, serde_json::json!({})),
            checkpoint_record(1, 1, "base", "base-1"),
            checkpoint_record(2, 1, "rollback", "base-1"),
        ]);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc123").dirty(),
                PersistedState::default(),
            ),
            &NoProbe,
        );

        let sha = report
            .steps
            .iter()
            .find_map(|step| match step {
                RecoveryStep::RollbackToCheckpoint { sha, .. } => Some(sha.clone()),
                _ => None,
            })
            .expect("a rollback");
        assert_eq!(sha, "base-1");
        assert_eq!(report.verdict, Verdict::Repaired);
    }

    #[test]
    fn a_confirmed_restart_is_not_planned_again() {
        // The record after the move is the confirmation. Rolling back a second time would take the
        // new attempt's own work away from it, from a SHA it has already been reset to.
        let journal = outcome(vec![
            scoped(0, 1, Kind::AttemptStarted, serde_json::json!({})),
            checkpoint_record(1, 1, "rollback", "base-1"),
            restarted(2, 1, 2),
        ]);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc123"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert!(
            !report
                .steps
                .iter()
                .any(|step| matches!(step, RecoveryStep::RollbackToCheckpoint { .. })),
            "{:?}",
            report.steps
        );
    }

    #[test]
    fn only_the_newest_unconfirmed_marker_is_restored() {
        // Two restarts, the first confirmed. The second attempt's marker is the one the crash
        // left behind, and restoring the first one would replay work an attempt has already
        // superseded.
        let journal = outcome(vec![
            scoped(0, 1, Kind::AttemptStarted, serde_json::json!({})),
            checkpoint_record(1, 1, "rollback", "base-1"),
            restarted(2, 1, 2),
            scoped(3, 2, Kind::AttemptStarted, serde_json::json!({})),
            checkpoint_record(4, 2, "rollback", "base-2"),
        ]);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc123"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        let shas: Vec<String> = report
            .steps
            .iter()
            .filter_map(|step| match step {
                RecoveryStep::RollbackToCheckpoint { sha, .. } => Some(sha.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(shas, vec!["base-2".to_string()]);
    }

    #[test]
    fn a_marker_without_a_sha_is_not_a_rollback() {
        // A malformed record must not become a reason to reset the tree to nowhere.
        let journal = outcome(vec![scoped(
            0,
            1,
            Kind::CheckpointCreated,
            serde_json::json!({ "role": "rollback" }),
        )]);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc123"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert!(
            !report
                .steps
                .iter()
                .any(|step| matches!(step, RecoveryStep::RollbackToCheckpoint { .. })),
            "{:?}",
            report.steps
        );
    }

    #[test]
    fn an_interrupted_restart_also_closes_the_attempt_it_closed() {
        // The attempt was ended by the restart, so the rollback implies abandoning it. Leaving it
        // `Running` is exactly the state recovery exists to clear.
        let journal = outcome(vec![
            scoped(0, 1, Kind::AttemptStarted, serde_json::json!({})),
            checkpoint_record(1, 1, "rollback", "base-1"),
        ]);
        let report = plan(
            &input(
                &journal,
                WorkspaceFacts::healthy("agent/tsk_01", "abc123"),
                PersistedState::default(),
            ),
            &NoProbe,
        );
        assert!(
            report
                .steps
                .iter()
                .any(|step| matches!(step, RecoveryStep::AbandonAttempt { number: 1, .. })),
            "{:?}",
            report.steps
        );
    }
}
