//! The production [`SubAgent`]: one brief, one attempt, one journal of its own.
//!
//! # Why this lives in the daemon
//!
//! A sub-agent is the agent loop: a model client, a tool registry, gates and a journal. All of
//! that is the daemon's, and [`immortality_tools::SubAgent`] is only the seam the tool layer can
//! name. `toolset` cannot build this — it exists precisely so that crate does not depend on the
//! loop.
//!
//! # Why the sub-agent gets its own journal
//!
//! [`immortality_core::journal::Journal`] is single-writer and the parent holds its writer for the
//! whole `delegate` call, so a second writer on the same segment would interleave two digest
//! chains and break the reader. The delegated run therefore writes to
//! `delegations/<del_id>/JOURNAL.*.jsonl`, beside its report and under the same id, where a human
//! can read exactly what the sub-agent did without the parent's steps in between.
//!
//! The unit of recovery is consequently the `delegate` call, not the sub-agent's step: re-running
//! the call re-runs the brief. That is what D-068 already promised, and it is why the probe looks
//! for the report rather than for a journal record.
//!
//! # Why the sub-agent's gate world is empty rather than absent
//!
//! [`Runner::gates`] being `None` means "nothing can verify a claim", and for a top-level subtask
//! that makes the claim a failed gate: nothing may close work it did not check. That is the right
//! fail-closed default and the wrong answer for a brief, because a brief has nothing of its own to
//! prove — and `None` would make every delegated claim come back rejected with a message about a
//! wiring bug the sub-agent cannot do anything about.
//!
//! A brief is judged differently, and the difference is not a loophole. The sub-agent leaves its
//! work in the parent's worktree, and the parent's `DoD` runs against that worktree when the
//! parent claims. Giving the brief its own `RealGates` would mean running the parent's acceptance
//! commands twice per delegation — once against a half-finished tree, and once more when it
//! matters — and the first run would fail for a reason that is not a verdict about the work.
//!
//! So [`BriefGates`] answers "an empty change, nothing refused, nothing to run", which makes
//! `evaluate` return `Passed` on the brief's empty `DoD` without touching the world. `Finish::Done`
//! then means exactly "the brief says it finished", never "the work is verified": the parent is
//! still the one who has to prove it, and a green `SubAgentOutcome` buys the parent nothing.

use std::sync::Arc;
use std::time::Duration;

use immortality_core::delegation::Finish;
use immortality_core::domain::{Dod, Subtask};
use immortality_core::gate::{CommandRun, Diff, Gates};
use immortality_core::journal::Journal;
use immortality_tools::delegate::{Delegation, SubAgent, SubAgentOutcome};
use immortality_tools::{SubAgentSlot, ToolError};

use crate::runner::{AttemptLimits, Runner};

/// The gate world of a brief: nothing to look at, nothing to refuse, nothing to run.
///
/// # Why this type exists instead of `None`
///
/// `gates: None` is the runner's marker for "nothing can verify a claim", and it fails a claim
/// closed. For a top-level subtask that is exactly right. For a brief it is a false accusation:
/// the report would read "the gates rejected it: no gate runner is wired", which describes the
/// daemon's wiring rather than the work, and the model reading it has no move to make.
///
/// So the brief is given a world that is *honestly* empty. `Diff::default()` is no files and no
/// text, so the sanity scan has nothing to find; `authorise` allows what `run` will never be
/// asked to execute, because the brief's `DoD` has no checks. If that `DoD` ever grew a check,
/// `run` would refuse it and the brief would fail with a message naming this type — the failure
/// stays loud, it just stops lying about what happened.
///
/// The real verdict still comes from the parent: the sub-agent works in the parent's worktree and
/// the parent's `DoD` runs against it. A green verdict here certifies nothing on its own.
#[derive(Debug, Clone, Copy)]
struct BriefGates;

impl Gates for BriefGates {
    fn diff(&self, _base: Option<&str>) -> Result<Diff, String> {
        // Reading the real diff would cost a git call per brief and change nothing: the only
        // gates that consult it are sanity and review, and a brief has neither.
        Ok(Diff::default())
    }

    fn authorise(&self, _command: &str) -> Result<(), String> {
        Ok(())
    }

    fn run(&self, command: &str) -> Result<CommandRun, String> {
        // Unreachable while the brief's `DoD` is empty. Refusing rather than passing is the
        // difference between a loud wiring change and a brief that silently "passes" a check it
        // never ran.
        Err(format!(
            "a delegated brief has no gate world to run `{command}` in; the parent's `DoD` is \
             where this work gets checked"
        ))
    }
}

/// A sub-agent over this daemon's model, tools and gate.
///
/// Cheap to clone: everything inside is an `Arc`.
#[derive(Clone)]
pub struct BriefRunner {
    /// What runs the brief: the parent's runner, shared rather than rebuilt.
    ///
    /// Shared so that a second `delegate` call reuses one model client, one registry and one
    /// artifact store. The differences a brief needs — its step budget, no gate world, no
    /// checkpoints — are applied to a *clone* of this in [`Self::runner_for`], so the parent's own
    /// limits are never touched by a delegated run.
    runner: Arc<Runner>,
}

impl std::fmt::Debug for BriefRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BriefRunner")
            .field("task", &self.runner.task)
            .field("tools", &self.runner.registry.len())
            .finish_non_exhaustive()
    }
}

impl BriefRunner {
    /// A sub-agent that runs briefs with `runner`'s model, registry and gate.
    ///
    /// The registry is the **same** one the parent uses — deliberately. A sub-agent with a
    /// narrower toolset would be a second definition of what the agent can do, and a sub-agent
    /// with a *wider* one would be a way around the parent's policy. Its own `delegate` tool
    /// refuses at depth one, which is what keeps the sharing from being recursion.
    #[must_use]
    pub const fn new(runner: Arc<Runner>) -> Self {
        Self { runner }
    }

    /// The runner this sub-agent was built over.
    #[must_use]
    pub const fn runner(&self) -> &Arc<Runner> {
        &self.runner
    }

    /// The runner for one brief: the parent's, with the brief's own limits.
    ///
    /// A fresh `Runner` rather than a mutated one, because the parent's is shared: a `&mut` here
    /// would either need exclusive access for the whole call or leave the parent running with the
    /// brief's step budget when the delegation returned.
    fn runner_for(&self, delegation: &Delegation) -> Runner {
        let parent = &self.runner;
        Runner {
            model: Arc::clone(&parent.model),
            model_name: parent.model_name.clone(),
            registry: Arc::clone(&parent.registry),
            artifacts: parent.artifacts.clone(),
            gate: Arc::clone(&parent.gate),
            system_prompt: parent.system_prompt.clone(),
            max_tool_output_bytes: parent.max_tool_output_bytes,
            max_completion_tokens: parent.max_completion_tokens,
            enable_thinking: parent.enable_thinking,
            budget: parent.budget,
            limits: AttemptLimits {
                max_steps: delegation.max_steps.max(1),
                max_tokens: parent.limits.max_tokens,
                stall_steps: parent.limits.stall_steps,
                // A brief is a fraction of an attempt, and a wall-clock ceiling of its own would
                // be a second number nobody tunes. It borrows the parent's, capped so that one
                // delegation cannot hold the task's whole thread for the parent's full window.
                max_wall_time: parent.limits.max_wall_time.min(Duration::from_mins(20)),
            },
            counter: Arc::clone(&parent.counter),
            task: delegation.task.clone(),
            detectors: parent.detectors.clone(),
            gates: Some(Arc::new(BriefGates)),
            reviewer: None,
            // Depth one: `split_subtask` is offered only where the tree may still grow, and a
            // brief's pieces are the parent's business.
            max_subtask_depth: 1,
            // Every tool call this runner issues is stamped depth one, which is what makes the
            // shared registry's `delegate` refuse: a sub-agent cannot delegate onward.
            delegation_depth: 1,
            // No checkpoints: the sub-agent must not roll the worktree back. It works in the
            // parent's tree, and a rollback from inside a brief would undo work the parent
            // believes is on disk.
            checkpoints: None,
            clock: Arc::clone(&parent.clock),
            // A brief does not drain the escalation collector, and that is deliberate rather
            // than an omission. `escalate` inside a brief queues the question in the parent's
            // queue root and under the parent's task id, so the record belongs in the *parent's*
            // journal; the brief's own journal lives in `delegations/<id>/` and nothing reads it
            // for metrics. Writing it here would put the record in a file no reader looks at.
            //
            // The records therefore stay pending until the parent's `run_tool` returns from the
            // `delegate` call, which is the next thing that happens and which drains into the
            // journal a human actually reads.
            escalations: Vec::new(),
        }
    }
}

impl SubAgent for BriefRunner {
    fn run(&self, delegation: &Delegation) -> Result<SubAgentOutcome, ToolError> {
        // The brief becomes a subtask so that the loop, the prompt builder and the report take the
        // shape they take for ordinary work. Depth one marks it as delegated: `delegate` refuses
        // there, so the sub-agent cannot hand the brief onward.
        let subtask = Subtask::new(
            1,
            "delegated brief",
            delegation.brief.clone(),
            // An empty `DoD` with review off is the one shape that gates nothing. `Dod::validate`
            // refuses this for a *planned* subtask, and that is right: a plan may not ship a
            // subtask nobody checks. A brief is not a plan — the check happens one level up.
            Dod {
                statement: delegation.brief.clone(),
                checks: Vec::new(),
                review: false,
            },
            self.runner.clock.now(),
        );

        // Its own journal, in its own directory: the parent holds the task journal's writer for
        // the whole call, and this must not become a second writer on that segment.
        let mut journal =
            Journal::open(&delegation.journal_dir, delegation.task.clone()).map_err(|error| {
                // No report is written, so the probe says `NotApplied` and the call may be
                // retried. That is the correct answer here: nothing ran.
                ToolError::Refused(
                    serde_json::json!({
                        "error": "sub_agent_journal_unavailable",
                        "detail": error.to_string(),
                        "hint": "The sub-agent's journal could not be opened, so the brief did \
                                 not start. This is a storage problem, not a policy decision.",
                    })
                    .to_string(),
                )
            })?;

        let runner = self.runner_for(delegation);
        let outcome = runner
            .run(&subtask, 1, "delegated", &mut journal)
            .map_err(|error| {
                // The loop itself failed to record what it did. That is not a report about the
                // work, so writing one would be inventing an outcome.
                ToolError::Refused(
                    serde_json::json!({
                        "error": "sub_agent_failed",
                        "detail": error.to_string(),
                        "hint": "The sub-agent started but its journal could not be written, so \
                                 nothing can be said about what it did. Do not assume the work is \
                                 done.",
                    })
                    .to_string(),
                )
            })?;

        // No `close()`. `Journal` syncs every append, so the segment is already durable and its
        // final state needs no separate declaration; the parent closes because it owns a journal
        // that outlives the call, and here the journal is dropped instead. Adding a close would
        // add a failure branch that cannot fail for any reason an error could name.
        Ok(outcome_of(&outcome, delegation.max_steps))
    }
}

/// Translate what the loop did into what the delegating tool records.
///
/// The mapping is not one-to-one on purpose. `Done` means the model claimed the brief finished —
/// the gates that matter run in the parent — so a claim with no gate world is still `Done`. The
/// step budget is the one distinction that survives: running out of steps is `Exhausted`, which
/// the parent reads as "the brief was too big" rather than "the work was wrong".
fn outcome_of(outcome: &crate::runner::RunOutcome, max_steps: u32) -> SubAgentOutcome {
    let steps = outcome.steps;
    let finish = if outcome.claimed_done() {
        Finish::Done
    } else if steps >= max_steps {
        Finish::Exhausted
    } else {
        Finish::Failed
    };
    SubAgentOutcome {
        steps,
        finish,
        report: outcome.report.clone(),
        error: match finish {
            Finish::Done => None,
            Finish::Exhausted => Some(format!(
                "the brief used all {max_steps} of its step(s) without claiming the work done"
            )),
            Finish::Failed => Some(format!(
                "the brief ended as `{}` after {steps} step(s)",
                outcome.ending.as_str()
            )),
        },
    }
}

/// Build the real sub-agent and install it into the slot the registry already holds.
///
/// # Why this is one function
///
/// The order is load-bearing: `toolset::build` creates the registry — and with it the `delegate`
/// tool over `slot` — then this call builds the [`BriefRunner`] over that registry, then installs
/// it. Doing it in a different order would be a cycle the compiler cannot see, because both sides
/// sit behind an `Arc`.
///
/// Returns the runner so a caller may also keep it for a `status` report; installing is the part
/// that matters, and it happens before the return.
#[must_use]
pub fn install(runner: &Runner, slot: &SubAgentSlot) -> Arc<BriefRunner> {
    let brief = Arc::new(BriefRunner::new(Arc::new(runner.clone())));
    slot.install(Arc::clone(&brief) as Arc<dyn SubAgent>);
    brief
}
