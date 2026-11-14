//! The agent loop: one attempt at one subtask, with a fresh context.
//!
//! # What a runner is, and what it is not
//!
//! A runner turns a brief into a sequence of steps — one model call, then the tool calls it
//! asked for — and ends with a [`SubtaskReport`]. It is the only component that spends tokens.
//! It does not decide *which* subtask runs, whether to split one, what a red gate means, or
//! whether to compact: those are the scheduler's, the gates' and the compactor's jobs. A runner
//! that also scheduled would make both impossible to test in isolation.
//!
//! # Why the model client is behind a trait
//!
//! [`Model`] is the seam. In production [`LlmModel`] wraps [`Backend`]; in a test a fake returns
//! scripted turns in microseconds. Without it every test of the loop would need a server, which
//! means either the suite depends on the network or the loop is untested — and an untested loop
//! is exactly where a run spends a day looping.
//!
//! # Why the context is built once per attempt
//!
//! `docs/design.md` §8 and D-074: the stable layers are rendered into an immutable prefix and
//! history grows only at the end. The runner therefore owns exactly one [`Context`] per attempt
//! and never edits its front. A restart is a *new attempt* with a new context, and that is the
//! only operation allowed to change the prefix.
//!
//! # The three-record protocol
//!
//! Every tool call goes through [`ToolCallLog`]: `intent`, `started`, then
//! `completed`/`failed`. The rule that makes recovery possible is that `completed` is written
//! **before** the model sees the observation. This module is where that ordering lives, because
//! it is the only place that both runs a tool and talks to the model.
//!
//! # Why a bad call is not a bad attempt
//!
//! A tool error, a refusal, or arguments that do not parse are *information*: the model gets the
//! message and may fix the call. Only the budgets end an attempt. The alternative — ending on
//! the first malformed call — makes a weak model unusable, and a weak model is the target.
//!
//! # What ends an attempt
//!
//! The model calling `finish_subtask`, or a budget: steps, tokens, or wall time. `finish_subtask`
//! is a *claim*, not a verdict: the gates decide ([`Runner::settle_claim`]). A green verdict closes
//! the attempt `Succeeded`; a red one closes it `Failed` with the gate's reason, and the report
//! carries what the gate said so the next attempt can act on it instead of re-guessing.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use immortality_core::artifacts::ArtifactStore;
use immortality_core::config::Detectors;
use immortality_core::context::{
    Budget, Context, Entry, StableLayers, TokenCounter, ToolCall, Verdict, artifact_hint, truncate_output,
};
use immortality_core::detect::{Detector, Level, Reaction, Signals, Stream, StreamVerdict, Thresholds};
use immortality_core::domain::Timestamp;
use immortality_core::domain::{Attempt, AttemptTrigger, FailureReason, Subtask, SubtaskId, SubtaskReport};
use immortality_core::gate::{GateRecord, GateVerdict, Gates, Reviewer, evaluate};
use immortality_core::ids::TaskId;
use immortality_core::journal::{Journal, Kind, Scope};
use immortality_core::protocol::{CallId, Intent, ToolCallLog};
use immortality_core::restart::{self, Allowance, Cause, Checkpoints, Decision, Ended, Limit, Stop};
use immortality_core::scheduler::Piece;
use immortality_core::time::Clock;
use immortality_llm::turn::AssistantTurn;
use immortality_llm::wire::ChatRequest;

/// The request type [`Model`] is given, re-exported.
///
/// A test fake in another crate has to name this type in its signature. Without a re-export, every
/// such crate would add `immortality-llm` as a dependency purely to spell a parameter it never
/// reads — and a dependency added for a signature is one more thing to keep in step.
pub use immortality_llm::wire::ChatRequest as ModelRequest;
use immortality_llm::{Backend, ToolDef};
use immortality_tools::{Args, PolicyGate, Registry, ToolCtx, ToolError};

/// The name the model calls to claim the subtask is done.
///
/// Spelled here once and compared against a name that came from a model: a typo would turn
/// "done" into "unknown tool" and run the attempt to its step budget.
pub const FINISH_SUBTASK: &str = "finish_subtask";

/// The name the model calls to say the subtask is too large and propose pieces.
///
/// Like [`FINISH_SUBTASK`] it is declared but never dispatched to a tool: it has no effect on
/// its own. The model *proposes*, the scheduler decides — a model that could split itself would
/// be choosing the shape of a tree it is not allowed to see, and its pieces still have to keep
/// the parent's gates, which only [`immortality_core::scheduler::validate_split`] can check.
pub const SPLIT_SUBTASK: &str = "split_subtask";

/// The largest number of pieces a split may propose.
///
/// Not a limit on decomposition but on a model's habit of listing ten headings instead of two
/// pieces of work: past this, a split is a plan rewrite, and a plan rewrite needs a human.
pub const MAX_SPLIT_PIECES: usize = 8;

/// Step, token and time limits for one attempt.
///
/// Taken from `LoopLimits` at construction rather than read inside the loop, so a test can run
/// a three-step budget without building a config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttemptLimits {
    pub max_steps: u32,
    pub max_tokens: u32,
    /// Steps without a successful tool call before the attempt is considered stalled.
    pub stall_steps: u32,
    pub max_wall_time: Duration,
}

impl Default for AttemptLimits {
    fn default() -> Self {
        // The design's defaults: 60 steps, 120K tokens, ~45 minutes.
        Self {
            max_steps: 60,
            max_tokens: 120_000,
            stall_steps: 15,
            max_wall_time: Duration::from_mins(45),
        }
    }
}

/// A model turn, as the loop needs it.
///
/// Narrower than [`AssistantTurn`] on purpose: the loop acts on content, tool calls and usage,
/// and a fake model should not have to invent a `finish_reason` it does not care about.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Reply {
    pub content: String,
    pub tool_calls: Vec<immortality_llm::turn::ToolCall>,
    /// Prompt and completion tokens, when the backend reported them.
    pub usage: Option<(u32, u32)>,
}

impl Reply {
    /// A reply that only speaks.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            ..Self::default()
        }
    }

    /// A reply that calls one tool.
    #[must_use]
    pub fn call(id: impl Into<String>, name: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self {
            tool_calls: vec![immortality_llm::turn::ToolCall::function(id, name, arguments)],
            ..Self::default()
        }
    }

    /// Whether the reply asks for tools.
    #[must_use]
    pub const fn is_tool_call(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

/// Something that can answer one step.
///
/// A trait rather than a concrete client so the loop is testable without a server. The error is
/// a `String` and not [`immortality_llm::Failure`] because the loop does not retry: backoff lives
/// in the client, which is the thing that knows what is retryable.
pub trait Model: Send + Sync + std::fmt::Debug {
    /// Answer one step.
    ///
    /// # Errors
    /// The request failed in a way the loop cannot act on; the attempt ends.
    fn complete(&self, request: &ChatRequest) -> Result<Reply, String>;

    /// Answer one step, letting a watcher see the text as it arrives.
    ///
    /// `watch` is called with each piece of content; returning `false` aborts the request. An
    /// aborted request returns [`Streamed::Aborted`] with the text seen so far, which is what
    /// D4 needs: the detector has decided the model is looping, and the loop must be able to
    /// journal the abort and react without having paid for the rest of the message.
    ///
    /// The default implementation calls [`Model::complete`] and watches the finished text in one
    /// go. That is correct for a fake and useless for D4 — a stream has to be interruptible
    /// mid-message to save anything — so [`LlmModel`] overrides it. A test that asserts the abort
    /// path must therefore use a model that overrides it too.
    ///
    /// # Errors
    /// See [`Model::complete`].
    fn complete_watched(
        &self,
        request: &ChatRequest,
        watch: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Streamed, String> {
        let reply = self.complete(request)?;
        if watch(&reply.content) {
            Ok(Streamed::Complete(reply))
        } else {
            Ok(Streamed::Aborted { seen: reply.content })
        }
    }
}

/// What a watched request produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Streamed {
    /// The turn finished.
    Complete(Reply),
    /// The watcher aborted it; this is the text that arrived before the abort.
    Aborted {
        /// The content seen so far, for the journal and for the next brief.
        seen: String,
    },
}

/// The real client, blocking.
///
/// The daemon runs on a tokio runtime and the loop is synchronous: making it `async` would push
/// `async` into every detector and every test for no benefit, because steps are strictly
/// sequential and the global semaphore admits one inference at a time anyway.
pub struct LlmModel {
    backend: Backend,
    runtime: tokio::runtime::Runtime,
}

impl std::fmt::Debug for LlmModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmModel")
            .field("base_url", &self.backend.base_url())
            .field("runtime", &"current-thread")
            .finish()
    }
}

impl LlmModel {
    /// Wrap the HTTP client.
    ///
    /// Owns a single-threaded runtime because the loop is synchronous and the client is not.
    /// The runner must therefore be driven from a dedicated thread — `spawn_blocking` in the
    /// daemon — not from inside an async task: `block_on` on a runtime worker panics, and
    /// catching that at runtime would be worse than documenting it.
    ///
    /// # Errors
    /// The runtime could not be created.
    pub fn new(backend: Backend) -> std::io::Result<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        Ok(Self { backend, runtime })
    }
}

impl Model for LlmModel {
    fn complete(&self, request: &ChatRequest) -> Result<Reply, String> {
        let turn = self
            .runtime
            .block_on(self.backend.turn_blocking(request))
            .map_err(|failure| failure.to_string())?;
        Ok(reply_from_turn(&turn))
    }

    fn complete_watched(
        &self,
        request: &ChatRequest,
        watch: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Streamed, String> {
        // `ChatRequest::new` already asks for a stream; `turn_blocking` is the one that turns it
        // off. Going through the streaming path is the only way to abort mid-message.
        let mut seen = String::new();
        let outcome = self
            .runtime
            .block_on(self.backend.turn_streaming(request, |delta| {
                seen.push_str(delta);
                watch(delta)
            }));
        match outcome {
            Ok(turn) => Ok(Streamed::Complete(reply_from_turn(&turn))),
            // A cancelled stream is not a backend failure: the watcher ended it. The partial text
            // is what the journal records and what the next brief calls "what you were writing".
            Err(immortality_llm::Failure::Cancelled) => Ok(Streamed::Aborted { seen }),
            Err(failure) => Err(failure.to_string()),
        }
    }
}

/// Convert a client turn into the loop's reply.
fn reply_from_turn(turn: &AssistantTurn) -> Reply {
    Reply {
        content: turn.content.clone(),
        tool_calls: turn.tool_calls.clone(),
        usage: turn
            .usage
            .as_ref()
            .map(|usage| (usage.prompt_tokens, usage.completion_tokens)),
    }
}

/// Why a run ended, for the journal and for the next attempt's brief.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ending {
    /// The model claimed the subtask done.
    Claimed,
    /// The step budget ran out.
    Steps,
    /// The token or context budget ran out.
    Tokens,
    /// The wall-clock budget ran out.
    Time,
    /// The model answered with nothing at all.
    Empty,
    /// A tool call named a tool that is not registered.
    ///
    /// Not retried: the model was shown the tool list and asked for something else, so an
    /// identical second step produces the same answer.
    UnknownTool { name: String },
    /// The backend failed.
    Backend { detail: String },
    /// Nothing progressed for `stall_steps` steps.
    Stalled,
    /// A detector ended the attempt and asked for a new one (L3).
    ///
    /// Distinct from the budget endings above: those say *which* budget ran out, and this says
    /// *which signal* concluded that another step in this attempt was worthless. The strategy of
    /// the next attempt is written from the detector's name and metric.
    Looped(Reaction),
    /// The model proposed a decomposition of the subtask (L4).
    ///
    /// The runner does not own the tree, so it stops and hands the pieces up. Validating them
    /// — depth, inherited gates, no duplicate work — is the scheduler's job, and doing it here
    /// would make the runner the authority on a tree it cannot see the neighbours of.
    SplitRequested(Vec<Piece>),
}

impl Ending {
    /// The failure reason this maps to, when it is a failure.
    ///
    /// `None` for [`Ending::Claimed`]: a claim is not a failure yet, because the gates have not
    /// spoken. Deciding otherwise would be the runner grading its own work.
    #[must_use]
    pub const fn failure_reason(&self) -> Option<FailureReason> {
        match self {
            Self::Claimed => None,
            Self::Steps | Self::Stalled => Some(FailureReason::StepsExhausted),
            Self::Tokens => Some(FailureReason::TokensExhausted),
            Self::Time => Some(FailureReason::TimeExhausted),
            Self::Backend { .. } => Some(FailureReason::BackendUnavailable),
            Self::Empty | Self::UnknownTool { .. } => Some(FailureReason::ToolError),
            // D-006: a task that loops is not a tool error and not a budget. It has its own
            // reason, so a metric query can separate "the model gave up" from "the model ran out
            // of room".
            Self::Looped(_) => Some(FailureReason::Looped),
            // A split is not a failure of the model's work: the subtask was too large, which is
            // what the split says. Its own reason keeps that apart from "too hard".
            Self::SplitRequested(_) => Some(FailureReason::Split),
        }
    }

    /// The wire name, for a journal record.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Claimed => "claimed",
            Self::Steps => "steps",
            Self::Tokens => "tokens",
            Self::Time => "time",
            Self::Empty => "empty",
            Self::UnknownTool { .. } => "unknown_tool",
            Self::Backend { .. } => "backend",
            Self::Stalled => "stalled",
            Self::Looped(_) => "looped",
            Self::SplitRequested(_) => "split",
        }
    }
}

/// What one finished attempt produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOutcome {
    /// The attempt, with counters and final state filled in.
    pub attempt: Attempt,
    /// Why it ended.
    pub ending: Ending,
    /// The report the next attempt inherits. Always present: even an exhausted run knows what
    /// it did, and a missing report would read as "nothing happened".
    pub report: SubtaskReport,
    /// Steps used.
    pub steps: u32,
    /// What the gates said about the claim, when there was a claim to judge.
    ///
    /// `None` for every ending that is not a claim. `Some(Passed)` is the only value that licenses
    /// closing the subtask, and closing it is the scheduler's job: the runner owns the attempt,
    /// the tree belongs to the plan, and a runner that wrote `done` into a tree it cannot see
    /// would be deciding what the next subtask is.
    pub gates: Option<GateVerdict>,
    /// The checkpoint the attempt's tree came from, if a restart rolled it back.
    pub base_checkpoint: Option<String>,
}

impl RunOutcome {
    /// Whether the model claimed the subtask done.
    #[must_use]
    pub const fn claimed_done(&self) -> bool {
        matches!(self.ending, Ending::Claimed)
    }

    /// Whether the claim passed every gate, and the subtask may be closed.
    #[must_use]
    pub fn verified(&self) -> bool {
        self.gates.as_ref().is_some_and(GateVerdict::is_passed)
    }

    /// What ended the attempt, in the restart vocabulary.
    ///
    /// The translation lives here rather than in [`Cause`] because only the outcome knows both
    /// why the loop stopped and what the gates said about it — and the claim is not a failure
    /// until the gates speak.
    #[must_use]
    pub fn ended(&self) -> Ended {
        match (&self.ending, &self.gates) {
            // A claim is only a failure when the gates say so. `None` means nothing could check
            // it, which is also a closure: the runner never certifies a claim it did not see.
            (Ending::Claimed, None | Some(GateVerdict::Passed)) => Ended::Verified,
            (Ending::Claimed, Some(GateVerdict::Failed { reason, why })) => Ended::Failed(Cause::Gate {
                reason: *reason,
                why: why.clone(),
            }),
            (Ending::Steps, _) => Ended::Failed(Cause::Limit(Limit::Steps)),
            (Ending::Tokens, _) => Ended::Failed(Cause::Limit(Limit::Tokens)),
            (Ending::Time, _) => Ended::Failed(Cause::Limit(Limit::Time)),
            (Ending::Stalled, _) => Ended::Failed(Cause::Stalled),
            (Ending::Empty, _) => Ended::Failed(Cause::EmptyReply),
            (Ending::UnknownTool { name }, _) => Ended::Failed(Cause::UnknownTool { name: name.clone() }),
            (Ending::Backend { .. }, _) => Ended::Abandoned,
            (Ending::Looped(reaction), _) => Ended::Failed(Cause::Detector {
                detector: reaction.detector,
                metric: reaction.metric,
            }),
            (Ending::SplitRequested(_), _) => Ended::Split,
        }
    }
}

/// Everything one run needs that is not the subtask itself.
#[derive(Clone)]
pub struct Runner {
    /// The model, and the name it is asked under.
    pub model: Arc<dyn Model>,
    /// Model name for the request body.
    pub model_name: String,
    /// Every tool the model may call.
    pub registry: Arc<Registry>,
    /// Where oversized tool output goes.
    pub artifacts: ArtifactStore,
    /// The policy gate every tool call passes through.
    pub gate: Arc<PolicyGate>,
    /// What the model is told before the history starts.
    pub system_prompt: String,
    /// Bytes of one tool result that may enter the prompt.
    pub max_tool_output_bytes: usize,
    /// Soft and hard context budgets.
    pub budget: Budget,
    /// Step, token and time limits.
    pub limits: AttemptLimits,
    /// Counts tokens for the context budget.
    pub counter: Arc<dyn TokenCounter>,
    /// The task the subtask belongs to, for journal records.
    pub task: TaskId,
    /// Detector thresholds, read from config once.
    pub detectors: Detectors,
    /// The world the gates read: git, the policy gate, the command runner.
    ///
    /// `None` is not "no gates needed" — the `DoD` decides that. It is "nothing can verify a
    /// claim", and a claim that nothing verified does not close the subtask. An `Option` rather
    /// than a required field so a test that exercises the loop without a repository does not
    /// have to invent one, and fail-closed so that decision cannot become a bypass.
    pub gates: Option<Arc<dyn Gates + Send + Sync>>,
    /// The reviewer, wired only when the daemon has a model to call.
    ///
    /// Separate from [`Self::gates`] because a `DoD` with `review: false` never needs it, and
    /// because the reviewer is a model call: a caller that wants the command gates without
    /// paying for a review passes `None` and gets exactly that.
    pub reviewer: Option<Arc<dyn Reviewer + Send + Sync>>,
    /// How deep the tree may grow, from `LoopLimits::max_subtask_depth`.
    ///
    /// The runner does not enforce it — the scheduler does — but the brief has to know whether
    /// offering a split would be an offer the tree is going to refuse.
    pub max_subtask_depth: u8,
    /// The tree a restart rolls back to.
    ///
    /// `None` means "a restart cannot be done": the decision to roll back is still made (it is
    /// pure), but acting on it without this would claim an L3 that never happened. The session
    /// then stops instead of starting an attempt on a tree nobody put back.
    pub checkpoints: Option<Arc<dyn Checkpoints + Send + Sync>>,
    /// Where wall-clock time comes from.
    ///
    /// The attempt budget and every journal timestamp read it. A simulator passes a fixed clock
    /// so that a 45-minute attempt ends in microseconds and a golden journal has one spelling per
    /// timestamp; production passes [`SystemClock`].
    pub clock: Arc<dyn Clock>,
}

impl std::fmt::Debug for Runner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runner")
            .field("model", &self.model)
            .field("model_name", &self.model_name)
            .field("tools", &self.registry.len())
            .field("task", &self.task)
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl Runner {
    /// Run one attempt at `subtask`, writing to `journal`.
    ///
    /// `strategy` says what makes this pass different from the previous one; empty for a first
    /// attempt.
    ///
    /// # Errors
    /// The journal cannot be written, or the attempt cannot be recorded. A model or tool
    /// failure is an [`Ending`], not an error: the run still produces a report, because a run
    /// that failed is exactly when the next attempt most needs to know why.
    pub fn run(
        &self,
        subtask: &Subtask,
        attempt_number: u32,
        strategy: &str,
        journal: &mut Journal,
    ) -> immortality_core::Result<RunOutcome> {
        self.run_with_base(subtask, attempt_number, strategy, None, journal)
    }

    /// [`Self::run`], recording which checkpoint the attempt's tree came from.
    ///
    /// `base_checkpoint` is the SHA a restart rolled back to, and it is written onto the attempt so
    /// that a later gate evaluation can diff against it instead of against `HEAD`: a reviewer that
    /// reads the whole task on every claim cannot tell this attempt's work from the previous
    /// one's, and approving the sum of two attempts is not approving either.
    ///
    /// # Errors
    /// See [`Self::run`].
    pub fn run_with_base(
        &self,
        subtask: &Subtask,
        attempt_number: u32,
        strategy: &str,
        base_checkpoint: Option<&str>,
        journal: &mut Journal,
    ) -> immortality_core::Result<RunOutcome> {
        // One clock, not two kept in step by hand: the records this call writes and the records
        // `ToolCallLog` writes inside it have to be ordered by the same source, or a golden
        // journal is unreproducible in exactly the way that hides an ordering bug.
        journal.set_clock(std::sync::Arc::clone(&self.clock));
        let mut attempt = Attempt::start(subtask.id.clone(), attempt_number, strategy, self.clock.now());
        if let Some(sha) = base_checkpoint {
            attempt.base_checkpoint = Some(sha.to_string());
        }
        journal.append_scoped(
            self.clock.now(),
            Kind::AttemptStarted,
            serde_json::json!({
                "attempt": attempt.id.to_string(),
                "number": attempt_number,
                "strategy": strategy,
                "subtask": subtask.id.to_string(),
            }),
            &Scope::attempt(subtask.id.clone(), attempt_number),
        )?;

        let mut state = StepLoop::new(self, subtask, &attempt);
        let ending = state.drive(journal)?;

        attempt.steps_used = state.steps;
        attempt.tokens_used = state.tokens;
        let base = attempt.base_checkpoint.clone();
        let report = state.report(&ending);
        if ending == Ending::Claimed {
            return self.settle_claim(subtask, attempt, report, state.steps, base, journal);
        }

        let trigger = match &ending {
            // A backend outage says nothing about the model's work and must not count against
            // the subtask's failure budget: after a reboot a subtask should not be one attempt
            // closer to `blocked` (D-027).
            Ending::Backend { .. } => AttemptTrigger::Abandon,
            // Step, token and wall-clock budgets are the attempt's own limits; they are
            // failures the ladder is supposed to react to, not external interruptions.
            _ => AttemptTrigger::Fail,
        };
        attempt
            .finish(trigger, ending.failure_reason(), self.clock.now())
            .map_err(|error| immortality_core::Error::Protocol(error.to_string()))?;

        journal.append_scoped(
            self.clock.now(),
            Kind::AttemptFailed,
            serde_json::json!({
                "attempt": attempt.id.to_string(),
                "state": attempt.state.as_str(),
                "ending": ending.as_str(),
                "steps": attempt.steps_used,
                "tokens": attempt.tokens_used,
                // For `looped` this is the strategy the restart exists for; every other ending
                // leaves the field out rather than writing `null`, which reads as "unknown".
                "detector": state.stopped_by.map(|reaction| reaction.detector.as_str()),
                "level": state.stopped_by.map(|reaction| reaction.level.as_str()),
            }),
            &Scope::attempt(subtask.id.clone(), attempt_number),
        )?;

        Ok(RunOutcome {
            attempt,
            ending,
            report,
            steps: state.steps,
            gates: None,
            base_checkpoint: base,
        })
    }

    /// A session that runs attempts at `subtask` until one ends it.
    ///
    /// The entry point the daemon uses. [`Self::run`] stays public because a single attempt is
    /// what most tests need, and because a caller that already applied the ladder itself — the
    /// recovery path, which has no allowance to spend — should not have to fake a session.
    #[must_use]
    pub fn session<'a>(&'a self, subtask: &'a Subtask, attempt_number: u32) -> AttemptSession<'a> {
        AttemptSession::new(self, subtask, attempt_number)
    }

    /// Turn a claim into a closed attempt, if and only if the gates are green.
    ///
    /// # Why this is a separate step
    ///
    /// A claim is one model sentence; closing a subtask is a verdict about work. Putting the
    /// evaluation here rather than inline in [`Self::run`] keeps the loop free of git, the
    /// policy gate and the reviewer, and it keeps the one rule that matters visible in one
    /// place: `Succeeded` is written **after** the verdict, never before it.
    ///
    /// # Why fail-closed on a missing gate world
    ///
    /// [`Self::gates`] is `None` when a caller wired no way to verify anything. Treating that as
    /// "no gates needed" would make every anemic caller a bypass, so it is a failed gate with a
    /// reason that names the wiring bug. The subtask does not close, and the report says why.
    ///
    /// # Errors
    /// The journal cannot be written.
    fn settle_claim(
        &self,
        subtask: &Subtask,
        mut attempt: Attempt,
        mut report: SubtaskReport,
        steps: u32,
        base: Option<String>,
        journal: &mut Journal,
    ) -> immortality_core::Result<RunOutcome> {
        let verdict = self.evaluate_gates(subtask, attempt.number, base.as_deref(), journal)?;
        match &verdict {
            GateVerdict::Passed => {
                attempt
                    .finish(AttemptTrigger::Succeed, None, self.clock.now())
                    .map_err(|error| immortality_core::Error::Protocol(error.to_string()))?;
                // Nothing is written about the subtask here. Closing it means writing `done` into
                // `TASKS.json` and then the journal (D-078's order), and only the scheduler holds
                // the tree. A runner that wrote the record without the tree would leave a journal
                // claiming a closure the plan never received.
            }
            GateVerdict::Failed { reason, why } => {
                attempt
                    .finish(AttemptTrigger::Fail, Some(*reason), self.clock.now())
                    .map_err(|error| immortality_core::Error::Protocol(error.to_string()))?;
                journal.append_scoped(
                    self.clock.now(),
                    Kind::AttemptFailed,
                    serde_json::json!({
                        "attempt": attempt.id.to_string(),
                        "state": attempt.state.as_str(),
                        "ending": "gate",
                        "failure": reason,
                        "why": why,
                        "steps": attempt.steps_used,
                        "tokens": attempt.tokens_used,
                    }),
                    &Scope::attempt(subtask.id.clone(), attempt.number),
                )?;
                // The claim was rejected, so the report must not read as an achievement: the
                // next attempt's brief is where the reason does its work.
                report
                    .rejected_ideas
                    .push(format!("the claim was rejected: {why}"));
                report.outcome = format!(
                    "claimed `{}` done, but the gates rejected it: {why}",
                    subtask.title
                );
            }
        }
        Ok(RunOutcome {
            attempt,
            ending: Ending::Claimed,
            report,
            steps,
            gates: Some(verdict),
            base_checkpoint: base,
        })
    }

    /// Run every gate the subtask's `DoD` asks for, journalling each observation.
    ///
    /// One `gate_result` per gate, in the order it ran, plus the verdict last. Written before
    /// the attempt is closed so that a crash between the evaluation and the close leaves a
    /// journal that says "the gates ran and here is what they saw" — recovery then closes the
    /// attempt like any other open one, and the next pass has real evidence.
    fn evaluate_gates(
        &self,
        subtask: &Subtask,
        attempt_number: u32,
        base: Option<&str>,
        journal: &mut Journal,
    ) -> immortality_core::Result<GateVerdict> {
        let scope = Scope::attempt(subtask.id.clone(), attempt_number);
        let Some(gates) = self.gates.as_deref() else {
            // No way to verify. Reported through the same channel as a red gate so nothing
            // downstream has to distinguish "unwired" from "failed": both mean "do not close".
            let verdict = GateVerdict::Failed {
                reason: FailureReason::GateFailed,
                why: "no gate runner is wired, so the claim cannot be verified".to_string(),
            };
            record_gate(
                self.clock.now(),
                journal,
                &scope,
                &GateRecord::Verdict {
                    passed: false,
                    reason: verdict.failure(),
                    why: verdict.why().map(str::to_string),
                },
            )?;
            return Ok(verdict);
        };
        // `Option<&dyn A + B>` does not coerce to `Option<&dyn A>`; the map is what lets the
        // `Send + Sync` bounds stay on the field without leaking into the pure signature.
        let reviewer = self.reviewer.as_deref().map(|it| it as &dyn Reviewer);
        // The base narrows the diff to this attempt's work. `Gates` decides what a base means;
        // the runner only knows which checkpoint it rolled back to.
        let report = evaluate(&subtask.dod, gates, reviewer, base).map_err(|why| {
            immortality_core::Error::Protocol(format!("the gate diff could not be produced: {why}"))
        })?;
        for record in GateRecord::of(&report) {
            record_gate(self.clock.now(), journal, &scope, &record)?;
        }
        Ok(report.verdict())
    }
}

/// Append one `gate_result` record.
fn record_gate(
    at: Timestamp,
    journal: &mut Journal,
    scope: &Scope,
    record: &GateRecord,
) -> immortality_core::Result<()> {
    let payload =
        serde_json::to_value(record).map_err(|error| immortality_core::Error::Protocol(error.to_string()))?;
    journal.append_scoped(at, Kind::GateResult, payload, scope)?;
    Ok(())
}

/// Run a subtask until it closes, splits, or spends its budget.
///
/// # Why this is a session and not one call to [`Runner::run`]
///
/// The design's attempt budget is not one number: attempts under one strategy, then a change of
/// strategy and, when the ladder says the attempt is going nowhere, a restart with a rollback
/// (D-080). Putting that in the scheduler would put the ladder into the one component that must
/// not know *why* an attempt failed. So the loop lives next to the thing it loops over:
/// [`Runner::run`] runs one attempt, [`AttemptSession::drive`] decides what the next one is.
///
/// # What a session owns, and what it does not
///
/// It owns the attempt counter, the accumulated report and the checkpoint the tree started from.
/// It does not own the tree: rolling back is a [`Checkpoints`] call, journalled here because the
/// journal is where a restart's safety snapshot has to appear *before* git deletes it.
///
/// # The ordering rule
///
/// Snapshot, journal, then move. A crash after the journal record leaves a SHA a human can find
/// and a tree that was never touched; a crash after the reset leaves a tree that is back at the
/// base and a record saying so. The other order — reset, then journal — loses the name of the work
/// the reset destroyed, which is the thing the milestone forbids.
pub struct AttemptSession<'a> {
    runner: &'a Runner,
    subtask: &'a Subtask,
    /// Number of the next attempt to start; 1-based, and the caller chooses the first because a
    /// resumed subtask does not start at 1.
    next: u32,
    /// The most recent report. Each attempt starts from it and replaces it, which is how a
    /// rejected idea reaches the following brief.
    report: SubtaskReport,
    /// The checkpoint the current attempt's tree came from.
    base: Option<restart::Snapshot>,
    /// Why no base could be taken, when one could not.
    ///
    /// Kept so a restart stops with git's own words instead of taking a late snapshot of the
    /// failed attempt's tree — the bug this field exists to make unrepeatable.
    base_error: Option<String>,
    /// The strategy the caller started the subtask with, quoted by every restart.
    ///
    /// Deliberately the *initial* one and never the generated text: quoting a generated strategy
    /// back at the model nests the same advice inside itself on every attempt. What actually
    /// changes between attempts is the accumulated list of failures below.
    seed: String,
    /// Why the last attempt ended. Kept beside the report because the strategy is built from
    /// both, and rebuilding it from the `RunOutcome` after the fact re-derives what is here.
    last: Option<Cause>,
}

impl std::fmt::Debug for AttemptSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttemptSession")
            .field("subtask", &self.subtask.id)
            .field("next", &self.next)
            .field("base", &self.base)
            .finish_non_exhaustive()
    }
}

impl<'a> AttemptSession<'a> {
    /// A session for `subtask`, whose next attempt is `attempt_number`.
    #[must_use]
    pub fn new(runner: &'a Runner, subtask: &'a Subtask, attempt_number: u32) -> Self {
        Self {
            runner,
            subtask,
            next: attempt_number,
            report: SubtaskReport::default(),
            base: None,
            base_error: None,
            seed: String::new(),
            last: None,
        }
    }

    /// Start from the strategy the caller chose for this subtask.
    ///
    /// Empty for "no strategy given", which is the common case: the plan says what to do and the
    /// runner only reacts when it has to.
    #[must_use]
    pub fn with_strategy(mut self, strategy: impl Into<String>) -> Self {
        self.seed = strategy.into();
        self
    }

    /// Start from the report a previous process left behind.
    ///
    /// A resumed subtask inherits what its last attempt learned; without this its first new brief
    /// would be as blind as the very first one, and the rejected ideas the restart exists to carry
    /// would be lost with the process.
    #[must_use]
    pub fn resuming_from(mut self, report: SubtaskReport) -> Self {
        self.report = report;
        self
    }

    /// Run attempts until one ends the session.
    ///
    /// `allowance` is read once and decremented here: attempts and restarts are two budgets, and a
    /// caller that recomputed them from config per attempt would count a restart as an attempt
    /// and stop a subtask after one rollback.
    ///
    /// # Errors
    /// The journal cannot be written. Everything else — a failed attempt, a red gate, a refused
    /// rollback — is a [`SessionEnd`], because a run that failed is exactly when the next attempt
    /// most needs to know why.
    pub fn drive(
        &mut self,
        allowance: Allowance,
        journal: &mut Journal,
    ) -> immortality_core::Result<SessionEnd> {
        // The base is the state the subtask *started* from, so it has to be taken before the
        // first attempt touches the tree. Taking it inside the rollback — which is what this did
        // — snapshots a tree that already contains the failed attempt's work, journals that as
        // "base", and then resets the branch onto it: the rollback discards nothing, and the
        // attempt's own work becomes the commit its gates are reviewing, so `git diff HEAD` is
        // empty and the reviewer rejects work that is on disk. Measured in a live run: the three
        // checkpoint records of one restart all named the same SHA.
        if self.base.is_none() && self.base_error.is_none() {
            match self.capture_base(journal) {
                Ok(Some(base)) => self.base = Some(base),
                // Nothing to roll back to. Kept as a reason rather than a stop of its own: a
                // session that never restarts does not need a base, and refusing to start it
                // would turn a missing commit into a task that cannot run at all.
                Ok(None) => {
                    self.base_error = Some(
                        "the worktree has no commits, so a restart has nothing to roll back to".to_string(),
                    );
                }
                Err(why) => self.base_error = Some(why),
            }
        }
        let mut left = allowance;
        loop {
            let number = self.next;
            let strategy = self.strategy();
            let outcome = self.runner.run_with_base(
                self.subtask,
                number,
                &strategy,
                self.base.as_ref().map(|base| base.sha.as_str()),
                journal,
            )?;
            self.absorb(&outcome.report);
            let cause = Self::cause_of(&outcome);
            self.remember(number, &cause);
            self.last = Some(cause);

            // The attempt just run has spent one of the budget, whichever way it ended: a stop is
            // evaluated against what is left *after* it, or a two-attempt budget would run three.
            left.attempts_left = left.attempts_left.saturating_sub(1);
            match restart::decide(outcome.ended(), left) {
                // Nothing is started, so nothing is written: the stop is the caller's to act on,
                // and the scheduler owns what a subtask becomes.
                Decision::Stop(stop) => return Ok(SessionEnd { outcome, stop }),
                Decision::Retry(_) => {
                    // The tree stays: a rejected claim is evidence about the work on disk, and
                    // rolling it away throws away the closest thing to a finished subtask.
                    self.next = self.next.saturating_add(1);
                }
                Decision::Restart(cause) => {
                    left.restarts_left = left.restarts_left.saturating_sub(1);
                    match self.roll_back(&outcome, &cause, journal)? {
                        RollBack::Done => {}
                        // A tree that could not be put back is not a tree to build on. The
                        // attempt is already closed, so the caller gets a stop naming git
                        // rather than a restart that lied about rolling back.
                        RollBack::Blocked(reason) => {
                            return Ok(SessionEnd {
                                outcome,
                                stop: Stop::Blocked(reason),
                            });
                        }
                    }
                    self.next = self.next.saturating_add(1);
                }
            }
        }
    }

    /// The strategy for the attempt about to start.
    fn strategy(&self) -> String {
        let Some(cause) = self.last.clone() else {
            return self.seed.clone();
        };
        restart::strategy(&cause, &self.seed, &self.report.rejected_ideas)
    }

    /// Take the attempt's report, keeping the failures the previous attempts named.
    ///
    /// The attempt's own report replaces the summary — it is the newest account of the work — but
    /// `rejected_ideas` is cumulative: the whole reason it exists is to outlive the context that
    /// produced it (D-074), and a restart that dropped the earlier failures would let a model
    /// rediscover them one attempt at a time.
    fn absorb(&mut self, report: &SubtaskReport) {
        let inherited = std::mem::take(&mut self.report.rejected_ideas);
        self.report = report.clone();
        for idea in inherited {
            if !self.report.rejected_ideas.contains(&idea) {
                self.report.rejected_ideas.insert(0, idea);
            }
        }
    }

    /// Remember why an attempt failed, for the briefs that follow it.
    ///
    /// The gate path already records its own reason in `settle_claim`, so only the other endings
    /// are added here: it is the same sentence the cause prints, and a brief that listed it twice
    /// would look like two failures.
    fn remember(&mut self, number: u32, cause: &Cause) {
        if matches!(cause, Cause::Gate { .. }) {
            return;
        }
        self.report
            .rejected_ideas
            .push(format!("attempt {number}: {}", cause.observation()));
    }

    /// Why an outcome ended, as the strategy text needs it.
    fn cause_of(outcome: &RunOutcome) -> Cause {
        match outcome.ended() {
            Ended::Failed(cause) => cause,
            // A verified claim, a split or an abandonment end the session; the cause is only read
            // when a restart happens, so this value is never used for a brief.
            Ended::Verified | Ended::Split | Ended::Abandoned => Cause::Stalled,
        }
    }

    /// Take the checkpoint a restart would roll back to.
    ///
    /// Called once, before the first attempt, so the base is the tree the subtask *started* from.
    /// `Ok(None)` is a repository with no commits: there is nothing to return to, which is a fact
    /// about the repository rather than an error.
    fn capture_base(&self, journal: &mut Journal) -> Result<Option<restart::Snapshot>, String> {
        let Some(checkpoints) = self.runner.checkpoints.as_deref() else {
            return Err("a restart needs a checkpoint world, and none is wired".to_string());
        };
        let taken = checkpoint_snapshot(
            checkpoints,
            &format!("base for attempt {} of `{}`", self.next, self.subtask.title),
        )?;
        let Some(taken) = taken else {
            return Ok(None);
        };
        record_checkpoint(
            self.runner.clock.now(),
            journal,
            &self.subtask.id,
            self.next,
            &taken,
            "base",
        )
        .map_err(|error| error.to_string())?;
        Ok(Some(taken))
    }

    /// Snapshot, journal, then move the tree back to the base.
    ///
    /// The base is the checkpoint taken before the *first* attempt of the session, not the last
    /// one: L3 gets back to the last state that was green, and an attempt that started on a broken
    /// tree must not make that broken tree the new base.
    fn roll_back(
        &self,
        outcome: &RunOutcome,
        cause: &Cause,
        journal: &mut Journal,
    ) -> immortality_core::Result<RollBack> {
        let Some(checkpoints) = self.runner.checkpoints.as_deref() else {
            return Ok(RollBack::Blocked(
                "a restart needs a checkpoint world, and none is wired".to_string(),
            ));
        };
        let closed = outcome.attempt.number;
        let reason = format!("L3 restart after attempt {closed}: {}", cause.observation());
        // No late snapshot here. A base taken now would contain the failed attempt's own work —
        // the branch would be reset onto it and the rollback would discard nothing. `drive` takes
        // it before the first attempt; a missing one is a stop, not something to invent.
        if self.base.is_none() {
            return Ok(RollBack::Blocked(self.base_error.clone().unwrap_or_else(|| {
                "no base checkpoint was taken before the attempt, so there is nothing to roll \
                 back to"
                    .to_string()
            })));
        }
        let Some(base) = self.base.clone() else {
            // Unreachable: the branch above fills it or returns. Written as a branch rather than
            // an `expect` because the workspace forbids panicking in library code.
            return Ok(RollBack::Blocked(
                "no base checkpoint was recorded, so there is nothing to roll back to".to_string(),
            ));
        };
        // Taken first so the discarded work has a name before git deletes it. `None` means the
        // tree was clean: there was nothing to preserve.
        match checkpoint_snapshot(checkpoints, &reason) {
            Ok(Some(safety)) => {
                record_checkpoint(
                    self.runner.clock.now(),
                    journal,
                    &self.subtask.id,
                    closed,
                    &safety,
                    "discarded",
                )?;
            }
            Ok(None) => {}
            Err(why) => return Ok(RollBack::Blocked(why)),
        }
        // The marker is what makes the crash window between the move and the record below
        // recoverable: it says "the tree is about to be reset to this SHA", and a journal that
        // stops after it is a rollback recovery can finish (see `recover`, plan step 4).
        journal.append_scoped(
            self.runner.clock.now(),
            Kind::CheckpointCreated,
            serde_json::json!({
                "sha": base.sha,
                "branch": base.branch,
                "role": "rollback",
                "attempt": closed,
            }),
            &Scope::attempt(self.subtask.id.clone(), closed),
        )?;
        if let Err(why) = checkpoints.restore(&base.sha) {
            return Ok(RollBack::Blocked(why));
        }
        journal.append_scoped(
            self.runner.clock.now(),
            Kind::AttemptRestarted,
            serde_json::json!({
                "number": closed,
                "next": closed.saturating_add(1),
                "cause": cause.as_str(),
                "why": cause.observation(),
                "base": base.sha,
            }),
            // D-027: `attempt` names the attempt being **closed**; the replacement's number is in
            // the payload, so every record's `attempt` field means the same thing.
            &Scope::attempt(self.subtask.id.clone(), closed),
        )?;
        Ok(RollBack::Done)
    }
}

/// What a session stopped on.
///
/// Both fields are mandatory: every stop is reached after an attempt ran, and a stop without the
/// attempt that caused it is a message a human cannot act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionEnd {
    /// The attempt that ended the loop.
    pub outcome: RunOutcome,
    /// Why no further attempt is started.
    pub stop: Stop,
}

/// What putting the tree back produced.
enum RollBack {
    /// The tree is at the base checkpoint.
    Done,
    /// The tree could not be put back, and the string is why.
    Blocked(String),
}

/// Take a snapshot through the seam, turning a git failure into a reason to stop.
///
/// A restart whose snapshot failed must not proceed: the tree is about to be reset, and a reset
/// that nothing recorded is the one outcome the milestone forbids.
fn checkpoint_snapshot(
    checkpoints: &(dyn Checkpoints + Send + Sync),
    reason: &str,
) -> Result<Option<restart::Snapshot>, String> {
    checkpoints.snapshot(reason)
}

/// Append a `checkpoint_created` record for one attempt's tree state.
fn record_checkpoint(
    at: Timestamp,
    journal: &mut Journal,
    subtask: &SubtaskId,
    attempt: u32,
    snapshot: &restart::Snapshot,
    role: &str,
) -> immortality_core::Result<()> {
    journal.append_scoped(
        at,
        Kind::CheckpointCreated,
        serde_json::json!({
            "sha": snapshot.sha,
            "branch": snapshot.branch,
            "role": role,
            "attempt": attempt,
        }),
        &Scope::attempt(subtask.clone(), attempt),
    )?;
    Ok(())
}

/// The mutable part of one run: context, counters, and the last thing that happened.
struct StepLoop<'a> {
    runner: &'a Runner,
    context: Context,
    subtask: &'a Subtask,
    /// The scope every record of this run carries. Recovery keys attempts by
    /// `(subtask, attempt)`, so a step record without it belongs to no attempt.
    scope: Scope,
    steps: u32,
    tokens: u64,
    /// The largest prompt this attempt has been billed for.
    ///
    /// The prompt is re-sent in full on every step, and its prefix is the same bytes the previous
    /// step already paid for. Charging `prompt + completion` each time would make
    /// `max_tokens_per_attempt` a disguised *step* limit whose real value depends on how large the
    /// context happened to be: at ~10K a step, 120000 is spent after twelve steps of the sixty the
    /// config allows, which is what a live run measured. Only the part of the prompt that is new
    /// since the last step is charged, so the budget counts the work an attempt added rather than
    /// the work it re-read.
    charged_prompt: u32,
    /// Steps since the last successful tool call. Any success resets it; talking does not.
    stalled: u32,
    /// Paths tools said they touched, for the report.
    touched: Vec<PathBuf>,
    /// The last text the model or a tool produced, for a claim's report.
    last_text: String,
    /// What the model said in the `finish_subtask` call, if it said anything.
    ///
    /// Preferred over `last_text` for the report because it is a deliberate summary rather than
    /// whatever happened to be observed last — the next attempt inherits it.
    claim_summary: Option<String>,
    /// What D1–D3 have seen so far, and what they have already announced.
    signals: Signals,
    /// The thresholds, read from config once.
    thresholds: Thresholds,
    /// The detector that asked for the restart, if one did.
    ///
    /// A restarted attempt is not a failed one in the same way as a step budget: the report has
    /// to say *which* signal gave up, because the next attempt's strategy is written from it.
    stopped_by: Option<Reaction>,
    /// When the attempt started, on the runner's clock.
    ///
    /// A wall-clock instant rather than `std::time::Instant` because the budget it measures is
    /// 45 minutes of *wall* time, and because a fake clock has to be able to move it: a budget
    /// that can only be spent by waiting is a budget no test covers.
    started: Timestamp,
}

impl<'a> StepLoop<'a> {
    fn new(runner: &'a Runner, subtask: &'a Subtask, attempt: &Attempt) -> Self {
        let layers = StableLayers {
            system: runner.system_prompt.clone(),
            brief: brief_for(subtask, &attempt.strategy, runner.max_subtask_depth),
            // Working memory and the task-level digest are the scheduler's to supply. This
            // milestone runs one subtask, so both are empty and render as `(nothing yet)`
            // rather than vanishing, keeping the layer shape constant.
            notes: String::new(),
            state: format!("Attempt {} of this subtask.", attempt.number),
        };
        Self {
            runner,
            context: Context::new(&layers),
            subtask,
            scope: Scope::attempt(subtask.id.clone(), attempt.number),
            steps: 0,
            tokens: 0,
            charged_prompt: 0,
            stalled: 0,
            touched: Vec::new(),
            last_text: String::new(),
            claim_summary: None,
            signals: Signals::default(),
            thresholds: Thresholds::from_config(&runner.detectors),
            stopped_by: None,
            started: runner.clock.now(),
        }
    }

    /// Run steps until something ends the attempt.
    fn drive(&mut self, journal: &mut Journal) -> immortality_core::Result<Ending> {
        loop {
            if let Some(ending) = self.exhausted() {
                // D5 is a detector like the others, and the journal is where the metric lives:
                // without this record a budget ending is indistinguishable from the loop quietly
                // stopping at a step count someone has to look up in config.
                self.record_reaction(journal, ending_reaction(&ending), None)?;
                return Ok(ending);
            }
            let reply = match self.step(journal)? {
                Step::Reply(reply) => reply,
                // D4 aborts the stream and applies L2: the attempt continues, but the next step
                // is a forced strategy rather than another tool call. It counts as an idle step
                // so that a model which loops in every message still reaches D3's restart.
                Step::Looping => {
                    self.steps = self.steps.saturating_add(1);
                    self.stalled = self.stalled.saturating_add(1);
                    self.signals.record_idle_step();
                    self.inject_l2();
                    if let Some(ending) = self.react(journal)? {
                        return Ok(ending);
                    }
                    continue;
                }
                Step::Backend(detail) => return Ok(Ending::Backend { detail }),
            };
            self.steps = self.steps.saturating_add(1);
            let (charge, charged_prompt) = attempt_charge(reply.usage, self.charged_prompt);
            self.tokens = self.tokens.saturating_add(charge);
            self.charged_prompt = charged_prompt;
            self.record_response(journal, &reply)?;

            if reply.content.trim().is_empty() && reply.tool_calls.is_empty() {
                return Ok(Ending::Empty);
            }

            let calls = calls_of(&reply);
            self.context.push_assistant(reply.content.clone(), calls.clone());

            if calls.is_empty() {
                // Prose with no calls: the model may be thinking aloud, and the next step may
                // carry the call, so this counts against the stall budget rather than ending.
                self.last_text.clone_from(&reply.content);
                self.stalled = self.stalled.saturating_add(1);
                self.signals.record_idle_step();
                if let Some(ending) = self.react(journal)? {
                    return Ok(ending);
                }
                if self.stalled >= self.runner.limits.stall_steps {
                    return Ok(Ending::Stalled);
                }
                continue;
            }

            match self.handle_calls(journal, &calls) {
                Ok(Calls::Claimed) => return Ok(Ending::Claimed),
                Ok(Calls::Split(pieces)) => return Ok(Ending::SplitRequested(pieces)),
                Ok(Calls::Continue) => {}
                Err(ending) => return Ok(ending),
            }
            if let Some(ending) = self.react(journal)? {
                return Ok(ending);
            }
            if self.stalled >= self.runner.limits.stall_steps {
                return Ok(Ending::Stalled);
            }
        }
    }

    /// Run every call in one reply, and say whether the reply ended the attempt.
    ///
    /// Two of the three call kinds the model can name are not tools: `finish_subtask` is a
    /// claim and `split_subtask` is a proposal, and neither has an effect to execute. They are
    /// intercepted here because this is the only place that sees both the call and the attempt,
    /// and because a split has to stop the loop before the calls planned for the retired
    /// subtask run.
    fn handle_calls(&mut self, journal: &mut Journal, calls: &[ToolCall]) -> Result<Calls, Ending> {
        let mut claimed = false;
        for call in calls {
            if call.name == FINISH_SUBTASK {
                claimed = true;
                self.claim_summary = summary_of(call);
                continue;
            }
            if call.name == SPLIT_SUBTASK {
                // A proposed split ends the attempt: the rest of the tool calls in this reply
                // were planned for a subtask that is about to stop existing, and running them
                // would put work on disk under a parent the tree is retiring.
                match self.proposed_split(journal, call) {
                    Ok(Some(pieces)) => return Ok(Calls::Split(pieces)),
                    Ok(None) => continue,
                    Err(ending) => return Err(ending),
                }
            }
            let step = self.run_tool(journal, call)?;
            match step {
                ToolStep::Ran => self.stalled = 0,
                ToolStep::NotRun => self.stalled = self.stalled.saturating_add(1),
            }
        }
        Ok(if claimed { Calls::Claimed } else { Calls::Continue })
    }

    /// Read a `split_subtask` call, or tell the model why it cannot be honoured.
    ///
    /// `Ok(None)` means the call was answered with an observation and the attempt continues: a
    /// model that described the decomposition badly has still told us the subtask is too large,
    /// which is worth keeping, and may still produce a usable answer.
    fn proposed_split(
        &mut self,
        journal: &mut Journal,
        call: &ToolCall,
    ) -> Result<Option<Vec<Piece>>, Ending> {
        let detail = if self.subtask.depth.saturating_add(1) > self.runner.max_subtask_depth {
            // The tool is not offered at this depth, so reaching here means the model asked for
            // something it was never shown. Telling it why is cheaper than ending the attempt.
            Some(format!(
                "`{SPLIT_SUBTASK}` was not offered: a subtask at depth {} cannot be split, \
                 because the tree may not grow past depth {}",
                self.subtask.depth, self.runner.max_subtask_depth
            ))
        } else if let Some(pieces) = pieces_of(call) {
            return Ok(Some(pieces));
        } else {
            Some(format!(
                "`{SPLIT_SUBTASK}` needs at least two pieces, each with a non-empty title and goal"
            ))
        };
        let Some(detail) = detail else {
            return Ok(None);
        };
        journal
            .append_scoped(
                self.runner.clock.now(),
                Kind::InvalidToolcall,
                serde_json::json!({
                    "call": call.call_id,
                    "tool": call.name,
                    "detail": detail,
                }),
                &self.scope,
            )
            .map_err(|error| backend(&error))?;
        self.last_text.clone_from(&detail);
        self.context
            .push_entry(Entry::tool(call.call_id.clone(), call.name.clone(), detail));
        self.stalled = self.stalled.saturating_add(1);
        Ok(None)
    }

    /// Ask the model one step, watching the stream for D4.
    ///
    /// D4 aborts the stream and applies L2: the attempt continues, but the next step is a forced
    /// strategy rather than another tool call. `Looping` is reported separately from a backend
    /// error because the two leave the attempt in different states.
    fn step(&self, journal: &mut Journal) -> immortality_core::Result<Step> {
        let request = self.request();
        let mut stream = Stream::new(&self.runner.detectors);
        let mut seen = String::new();
        let mut looping = None;
        let outcome = self.runner.model.complete_watched(&request, &mut |delta| {
            seen.push_str(delta);
            let verdict = stream.push(delta);
            if let StreamVerdict::Looping { occurrences } = verdict {
                looping = Some(occurrences);
            }
            !verdict.must_abort()
        });
        if let Some(occurrences) = looping {
            let reaction = Reaction {
                detector: Detector::ReasoningLoop,
                level: Level::ForceStrategy,
                metric: occurrences,
            };
            journal.append_scoped(
                self.runner.clock.now(),
                Kind::ModelStreamAborted,
                serde_json::json!({
                    "step": self.steps,
                    "detector": reaction.detector.as_str(),
                    "level": reaction.level.as_str(),
                    "chars_seen": seen.chars().count(),
                }),
                &self.scope,
            )?;
            self.record_reaction(journal, reaction, Some(&seen))?;
            return Ok(Step::Looping);
        }
        match outcome {
            Ok(Streamed::Complete(reply)) => Ok(Step::Reply(reply)),
            // A real client reports an abort only through `looping` above, because its watcher
            // returns `false` exactly when D4 fires. An abort without a verdict means the watcher
            // stopped for a reason the loop does not know, and inventing a reply from the text
            // seen so far would put a half-written sentence in the history as if the model had
            // finished it.
            Ok(Streamed::Aborted { .. }) => Ok(Step::Backend(
                "the response stream was aborted without a detector firing".to_string(),
            )),
            Err(detail) => Ok(Step::Backend(detail)),
        }
    }

    /// Apply L2: tell the model to stop calling tools and state a strategy.
    /// Apply L2: tell the model to stop calling tools and state a strategy.
    ///
    /// The journal record is written by `record_reaction` before this is called; writing another
    /// here would count one firing twice, and the metric exists precisely to be trusted.
    fn inject_l2(&mut self) {
        self.signals.strategy_delivered();
        let text = Level::ForceStrategy
            .instruction(Detector::ReasoningLoop)
            .unwrap_or_default();
        self.context.push_entry(Entry::note(text));
    }

    /// The request as the context now stands.
    fn request(&self) -> ChatRequest {
        let mut request = ChatRequest::new(self.runner.model_name.clone());
        request.messages = immortality_llm::prompt::messages(&self.context);
        let mut tools: Vec<ToolDef> = self
            .runner
            .registry
            .iter()
            .map(|tool| ToolDef::function(tool.name(), tool.description(), tool.schema().clone()))
            .collect();
        // `finish_subtask` is not a tool: it has nothing to execute and no effect on the tree.
        // It has to be *declared* all the same, because a model cannot call what it was never
        // offered — the runner would recognise the name and the model would never produce it,
        // so every run would end at the step budget instead of on a claim.
        tools.push(finish_def());
        // Declared for the same reason, and intercepted by name in `drive` for the same reason:
        // a split is a proposal about a tree the runner cannot see, so it has no side effect to
        // execute here. A model cannot call what it was not offered.
        tools.push(split_def());
        request.tools = Some(tools);
        request
    }

    /// Journal a detector's reaction and, for L1, put the instruction in the context.
    ///
    /// L1 speaks and continues; L2 for a *non*-D4 detector is recorded but not injected, because
    /// the forced strategy step belongs to the next model call and `react` returns `None` at L2
    /// only when the caller is already driving one. `react` handles that ordering; this function
    /// records what was decided.
    fn record_reaction(
        &self,
        journal: &mut Journal,
        reaction: Reaction,
        seen: Option<&str>,
    ) -> immortality_core::Result<()> {
        let mut fields = serde_json::json!({
            "detector": reaction.detector.as_str(),
            "level": reaction.level.as_str(),
            "metric": reaction.metric,
            "step": self.steps,
        });
        if let Some(text) = seen {
            fields["chars_seen"] = serde_json::json!(text.chars().count());
        }
        journal.append_scoped(self.runner.clock.now(), Kind::DetectorFired, fields, &self.scope)?;
        Ok(())
    }

    /// Consult the ladder and act on what it says.
    ///
    /// Returns an `Ending` when the attempt is over. L1 appends the nudge to the history — the
    /// prefix stays untouched, so the cache survives — and L2 returns an ending only for a
    /// detector whose escalation is "stop this attempt"; the runner injects L2 itself at the top
    /// of the next step.
    fn react(&mut self, journal: &mut Journal) -> immortality_core::Result<Option<Ending>> {
        let Some(reaction) = self.signals.react(&self.thresholds) else {
            return Ok(None);
        };
        self.record_reaction(journal, reaction, None)?;
        match reaction.level {
            Level::Nudge => {
                if let Some(text) = reaction.level.instruction(reaction.detector) {
                    self.context.push_entry(Entry::note(text));
                }
                Ok(None)
            }
            Level::ForceStrategy => {
                // L2 is a forced *step*, not the end of the attempt: the model is told to stop
                // calling tools and state a strategy, and gets the chance to comply. Only L3
                // gives up on the attempt. A model that ignores this too will reach L3 by the
                // same detector's stop threshold, which is where the restart lives.
                if let Some(text) = reaction.level.instruction(reaction.detector) {
                    self.signals.strategy_delivered();
                    self.context.push_entry(Entry::note(text));
                }
                Ok(None)
            }
            Level::Restart => {
                self.stopped_by = Some(reaction);
                Ok(Some(Ending::Looped(reaction)))
            }
        }
    }

    /// The budget that has been exhausted, if any.
    fn exhausted(&self) -> Option<Ending> {
        if self.steps >= self.runner.limits.max_steps {
            return Some(Ending::Steps);
        }
        if self.tokens >= u64::from(self.runner.limits.max_tokens) {
            return Some(Ending::Tokens);
        }
        let elapsed = self.runner.clock.now() - self.started;
        let limit =
            chrono::Duration::from_std(self.runner.limits.max_wall_time).unwrap_or(chrono::Duration::MAX);
        if elapsed >= limit {
            return Some(Ending::Time);
        }
        // The hard context budget is a token budget too. `Compact` is not acted on: compaction
        // needs a model call of its own and a place to write the summary, and a runner that
        // silently dropped history would break the append-only prefix the cache depends on.
        if self
            .context
            .verdict(self.runner.counter.as_ref(), self.runner.budget)
            == Verdict::Restart
        {
            return Some(Ending::Tokens);
        }
        None
    }

    /// Journal the response's shape and cost.
    fn record_response(&self, journal: &mut Journal, reply: &Reply) -> immortality_core::Result<()> {
        journal.append_scoped(
            self.runner.clock.now(),
            Kind::ModelResponse,
            serde_json::json!({
                "step": self.steps,
                "chars": reply.content.len(),
                "tool_calls": reply.tool_calls.len(),
                "prompt_tokens": reply.usage.map(|(prompt, _)| prompt),
                "completion_tokens": reply.usage.map(|(_, completion)| completion),
            }),
            &self.scope,
        )?;
        Ok(())
    }

    /// Find the tool the model named.
    ///
    /// A name that is not registered is a protocol failure and ends the attempt: the model was
    /// shown the tool list, so a second identical step would produce the same answer.
    fn lookup(
        &self,
        journal: &mut Journal,
        call: &ToolCall,
    ) -> Result<Arc<dyn immortality_tools::Tool>, Ending> {
        let Some(tool) = self.runner.registry.get(&call.name).cloned() else {
            journal
                .append_scoped(
                    self.runner.clock.now(),
                    Kind::InvalidToolcall,
                    serde_json::json!({ "call": call.call_id, "tool": call.name, "detail": "unknown tool" }),
                    &self.scope,
                )
                .map_err(|error| backend(&error))?;
            return Err(Ending::UnknownTool {
                name: call.name.clone(),
            });
        };
        Ok(tool)
    }

    /// Parse and validate the arguments.
    ///
    /// `Ok(None)` means the model was told what was wrong and the attempt continues. Arguments
    /// that do not parse or do not fit the schema are a *recoverable* mistake; nothing runs, so
    /// nothing is journaled as a call.
    fn arguments(
        &mut self,
        journal: &mut Journal,
        call: &ToolCall,
        schema: &serde_json::Value,
    ) -> Result<Option<Args>, Ending> {
        let value = match serde_json::from_str::<serde_json::Value>(&call.arguments) {
            Ok(value) => value,
            Err(error) => {
                let detail = format!("the arguments are not JSON: {error}");
                return self.reject_arguments(journal, call, &detail);
            }
        };
        if let Err(problems) = immortality_llm::schema::validate(schema, &value) {
            let detail = problems.first().map_or_else(
                || "the arguments do not fit the tool's schema".to_string(),
                immortality_llm::schema::Problem::render,
            );
            return self.reject_arguments(journal, call, &detail);
        }
        Ok(Some(Args::new(value)))
    }

    /// Run one tool call, through the journal and the policy gate.
    ///
    /// Returns an `Ending` in the error position only for a protocol failure that ends the
    /// attempt; a tool that refused is a `ToolStep::NotRun` observation.
    fn run_tool(&mut self, journal: &mut Journal, call: &ToolCall) -> Result<ToolStep, Ending> {
        let tool = self.lookup(journal, call)?;
        let Some(args) = self.arguments(journal, call, tool.schema())? else {
            return Ok(ToolStep::NotRun);
        };

        let mut log = ToolCallLog::scoped(journal, self.scope.clone());
        let intent = Intent::new(
            CallId::from_model(call.call_id.clone()),
            call.name.clone(),
            args.raw(),
            tool.idempotency(&args),
            // What *this* call is expected to do, not what the tool is: the journal quote goes to
            // the operator reading `immortality log` and into the observation recovery hands the
            // model, and a tool description there says "Write a complete file…" for every write.
            tool.expected_effect(&args),
        );
        // The arguments are the second argument, not the record: `intent` writes the metadata
        // (`tool`, `idempotency`, digest, expected effect) from the typed value and nests these
        // under `arguments`, so a tool argument named `tool` cannot impersonate a record field.
        log.intent(&intent, args.raw()).map_err(|error| backend(&error))?;
        log.started(&intent.call, serde_json::Value::Null)
            .map_err(|error| backend(&error))?;

        let ctx = ToolCtx::new(
            self.runner.gate.clone(),
            immortality_tools::OutputBudget {
                max_bytes: self.runner.max_tool_output_bytes,
                max_context: 3,
            },
        )
        .with_call(call.call_id.clone())
        .with_task(self.runner.task.clone());

        let declared = tool.paths(&args);
        match tool.execute(&ctx, args) {
            Ok(result) => {
                for path in &declared {
                    if path.write {
                        self.touched.push(PathBuf::from(&path.path));
                    }
                }
                let truncated = truncate_output(
                    &self.runner.artifacts,
                    &result.observation,
                    self.runner.max_tool_output_bytes,
                )
                .map_err(|error| backend(&error))?;
                let observation = if truncated.was_cut() {
                    let hint = truncated
                        .artifact
                        .as_ref()
                        .map(|digest| artifact_hint(digest, truncated.total))
                        .unwrap_or_default();
                    format!("{}\n\n{hint}", truncated.text)
                } else {
                    truncated.text
                };
                // Before the model sees it: this is what makes the journal authoritative.
                log.completed(
                    &intent.call,
                    serde_json::json!({ "observation": observation }),
                    truncated.artifact,
                )
                .map_err(|error| backend(&error))?;
                drop(log);
                self.signals.record_call(
                    &call.name,
                    &call.arguments,
                    Some(&observation),
                    Some(self.runner.gate.worktree()),
                );
                self.last_text.clone_from(&observation);
                self.context
                    .push_entry(Entry::tool(call.call_id.clone(), call.name.clone(), observation));
                Ok(ToolStep::Ran)
            }
            Err(error) => {
                let observation = match &error {
                    ToolError::Refused(text) => text.clone(),
                    other => other.to_string(),
                };
                log.failed(&intent.call, serde_json::json!({ "error": observation }))
                    .map_err(|error| backend(&error))?;
                drop(log);
                // A refusal is a step that produced no evidence about the task, so it feeds the
                // D1 run but not D2: the model is being told what was wrong, and that is neither
                // progress nor a repeated experiment.
                self.signals.record_call(
                    &call.name,
                    &call.arguments,
                    None,
                    Some(self.runner.gate.worktree()),
                );
                self.last_text.clone_from(&observation);
                self.context
                    .push_entry(Entry::tool(call.call_id.clone(), call.name.clone(), observation));
                Ok(ToolStep::NotRun)
            }
        }
    }

    /// Tell the model its arguments were wrong, without inventing a call in the journal.
    fn reject_arguments(
        &mut self,
        journal: &mut Journal,
        call: &ToolCall,
        detail: &str,
    ) -> Result<Option<Args>, Ending> {
        journal
            .append_scoped(
                self.runner.clock.now(),
                Kind::InvalidToolcall,
                serde_json::json!({
                    "call": call.call_id,
                    "tool": call.name,
                    "detail": detail,
                }),
                &self.scope,
            )
            .map_err(|error| backend(&error))?;
        let observation = format!(
            "`{}` was not run: {detail}. Fix the arguments and call again.",
            call.name
        );
        self.last_text.clone_from(&observation);
        self.context
            .push_entry(Entry::tool(call.call_id.clone(), call.name.clone(), observation));
        Ok(None)
    }

    /// The report the next attempt inherits.
    fn report(&self, ending: &Ending) -> SubtaskReport {
        let outcome = match ending {
            Ending::Claimed => self
                .claim_summary
                .clone()
                .unwrap_or_else(|| self.last_text.trim().to_string()),
            other => format!(
                "did not finish `{}`: {} after {} step(s)",
                self.subtask.title,
                other.as_str(),
                self.steps
            ),
        };
        let outcome = if outcome.is_empty() {
            match ending {
                Ending::Claimed => format!(
                    "claimed `{}` done; the gates have not run yet",
                    self.subtask.title
                ),
                other => format!("`{}` ended: {}", self.subtask.title, other.as_str()),
            }
        } else {
            outcome
        };
        let mut report = SubtaskReport::new(outcome);
        report.files_touched = dedup(&self.touched);
        report
    }
}

/// What one model reply did.
///
/// Separate from [`Step`] because a reply can end the attempt without being an answer to run:
/// a claim and a split proposal are both calls, and both stop the loop before the next step.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Calls {
    /// The reply's tools ran; the loop continues.
    Continue,
    /// The model claimed the subtask done.
    Claimed,
    /// The model proposed a decomposition; the scheduler decides whether it is legal.
    Split(Vec<Piece>),
}

/// What one tool call did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolStep {
    /// The tool ran and produced an observation.
    Ran,
    /// Nothing ran: refused by the policy gate, or the arguments were refused.
    NotRun,
}

/// What one model step produced.
#[derive(Debug, Clone)]
enum Step {
    /// A finished turn.
    Reply(Reply),
    /// D4 aborted the stream and L2 was applied; the attempt continues.
    Looping,
    /// The request failed in a way the loop cannot act on.
    Backend(String),
}

/// The brief: what a fresh context is told about the subtask.
///
/// No history and no parent conversation — that is the point of a fresh context. It names the
/// goal, the definition of done, the strategy, and the one rule the model cannot infer: the
/// gates decide, not the claim.
#[must_use]
pub fn brief_for(subtask: &Subtask, strategy: &str, max_depth: u8) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let _ = writeln!(out, "# Subtask: {}", subtask.title);
    out.push('\n');
    let _ = writeln!(out, "## Goal");
    let _ = writeln!(out, "{}", subtask.goal.trim());
    out.push('\n');
    let _ = writeln!(out, "## Definition of done");
    let _ = writeln!(out, "{}", subtask.dod.statement.trim());
    if !subtask.dod.checks.is_empty() {
        out.push('\n');
        let _ = writeln!(out, "These commands are run by the system, not by you:");
        for check in &subtask.dod.checks {
            let required = if check.required {
                "required"
            } else {
                "informational"
            };
            let _ = writeln!(out, "- `{}` ({required}): {}", check.id, check.command);
        }
    }
    if subtask.dod.review {
        out.push('\n');
        let _ = writeln!(out, "A reviewer will read the diff and must approve it.");
    }
    if !strategy.trim().is_empty() {
        out.push('\n');
        let _ = writeln!(out, "## Strategy for this attempt");
        let _ = writeln!(out, "{}", strategy.trim());
    }
    out.push('\n');
    let _ = writeln!(out, "## How to finish");
    let _ = writeln!(
        out,
        "Call `{FINISH_SUBTASK}` when you believe the definition of done is met. That is a \
         claim: the system runs the checks and a reviewer may reject the diff. Do not claim it \
         before the work is on disk."
    );
    if subtask.depth < max_depth {
        let _ = writeln!(
            out,
            "If the goal is clear but the work does not fit in one pass, call `{SPLIT_SUBTASK}` \
             with two or more pieces instead of burning the attempt on it. Each piece inherits \
             this subtask's definition of done, and the system decides whether the split is \
             legal."
        );
    }
    out
}

/// The declaration of `finish_subtask`, which the model is offered but the runner never runs.
///
/// The schema takes a required `summary`: the report the next subtask inherits is written from
/// the model's own words, and a claim with nothing to say leaves the next attempt starting from
/// a blank. Fields the gates use are deliberately absent — a model that could describe its own
/// verification would be grading itself.
fn finish_def() -> ToolDef {
    ToolDef::function(
        FINISH_SUBTASK,
        "Claim that this subtask's definition of done is met. The system then runs its checks \
         and a reviewer may reject the diff, so claim it only when the work is on disk.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "What you achieved, in the past tense, and anything the next \
                                    subtask should know."
                }
            },
            "required": ["summary"],
            "additionalProperties": false
        }),
    )
}

/// The declaration of `split_subtask`, which the model is offered but the runner never runs.
///
/// The pieces are words only: title and goal. The `DoD` is not a parameter because it is
/// inherited from the parent — a model that could write its children's gates could quietly
/// weaken the guarantee the split exists to preserve, and validating that after the fact would
/// be checking a promise it had no power to keep.
fn split_def() -> ToolDef {
    ToolDef::function(
        SPLIT_SUBTASK,
        "Say that this subtask is too large to finish in one pass and propose two or more \
         pieces of it. Use it when the goal is clear but the work does not fit in one attempt — \
         not when you are stuck. Each piece keeps this subtask's definition of done, and the \
         system decides whether the decomposition is legal.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the subtask does not fit in one pass."
                },
                "pieces": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": MAX_SPLIT_PIECES,
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": { "type": "string", "description": "A short title." },
                            "goal": {
                                "type": "string",
                                "description": "What to achieve, written for a fresh context \
                                                that has not seen this conversation."
                            }
                        },
                        "required": ["title", "goal"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["pieces"],
            "additionalProperties": false
        }),
    )
}

/// The `pieces` argument of a `split_subtask` call.
///
/// `None` when the arguments do not name between two and [`MAX_SPLIT_PIECES`] pieces with text in
/// them. Neither bound is left to the schema: `split_subtask` is intercepted before the generic
/// argument validation, precisely so it never reaches a tool, so the schema's `minItems` and
/// `maxItems` would otherwise be advice nobody enforces. The caller turns `None` into an
/// observation and the attempt continues: a model that described a decomposition badly has still
/// told us the subtask is too large, and that is not a protocol failure.
fn pieces_of(call: &ToolCall) -> Option<Vec<Piece>> {
    let value: serde_json::Value = serde_json::from_str(&call.arguments).ok()?;
    let pieces: Vec<Piece> = value
        .get("pieces")?
        .as_array()?
        .iter()
        .map(|piece| {
            Some(Piece::new(
                piece.get("title")?.as_str()?.trim(),
                piece.get("goal")?.as_str()?.trim(),
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    // A single piece is not a decomposition but the same work under a new id with a fresh
    // attempts budget; past the cap a "split" is a plan rewrite, which needs a human.
    ((2..=MAX_SPLIT_PIECES).contains(&pieces.len())
        && pieces.iter().all(|p| !p.title.is_empty() && !p.goal.is_empty()))
    .then_some(pieces)
}

/// The `summary` argument of a `finish_subtask` call, if it parsed and was not blank.
///
/// Tolerant on purpose: the claim itself is what ends the attempt, and losing it because the
/// model wrote `{}` would trade a usable report for an unused one. A blank summary reads as
/// "no summary", which the report then fills from the last observation.
fn summary_of(call: &ToolCall) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(&call.arguments).ok()?;
    let summary = value.get("summary")?.as_str()?.trim();
    if summary.is_empty() {
        None
    } else {
        Some(summary.to_string())
    }
}

/// What one step costs an attempt's token budget, and the prompt it has now charged for.
///
/// The prompt is re-sent in full on every step and its stable prefix is the same bytes the
/// previous step already paid for. Charging `prompt + completion` every time turns
/// `max_tokens_per_attempt` into a step limit whose real size depends on the context: a live run
/// billed ~10K a step, so the 120000 budget was gone after twelve of the sixty steps the config
/// allows, and two of three subtasks restarted for a reason that had nothing to do with the model
/// looping.
///
/// So the charge is what this step *added*: the completion, plus the part of the prompt that is
/// larger than the previous prompt. A prompt that shrank (a compaction, a rolled-back attempt)
/// charges nothing extra — the tokens are already spent, and billing a negative number would let a
/// shrinking history buy back budget it never had.
///
/// `usage: None` charges nothing, which is the honest answer for a backend that reported no
/// counts: the alternative is an estimate that over- or under-charges every step.
fn attempt_charge(usage: Option<(u32, u32)>, already_charged: u32) -> (u64, u32) {
    let Some((prompt, completion)) = usage else {
        return (0, already_charged);
    };
    let new_prompt = prompt.saturating_sub(already_charged);
    (
        u64::from(new_prompt) + u64::from(completion),
        prompt.max(already_charged),
    )
}

fn calls_of(reply: &Reply) -> Vec<ToolCall> {
    reply
        .tool_calls
        .iter()
        .map(|call| ToolCall {
            call_id: call.id.clone(),
            name: call.function.name.clone(),
            arguments: call.function.arguments.clone(),
        })
        .collect()
}

/// A journal or artifact-store failure. Always the end of the attempt.
fn backend(error: &immortality_core::Error) -> Ending {
    Ending::Backend {
        detail: error.to_string(),
    }
}

/// D5's reaction for a budget ending.
///
/// A budget running out is always L3: the attempt cannot be talked down, because the steps or the
/// tokens it would take to do so are what ran out.
const fn ending_reaction(ending: &Ending) -> Reaction {
    Reaction {
        detector: Detector::Budget,
        level: Level::Restart,
        metric: match ending {
            // The metric is the budget that ran out, not its value: the value is in the attempt's
            // counters, and repeating it here would be a second number to keep in sync.
            Ending::Steps => 1,
            Ending::Tokens => 2,
            Ending::Time => 3,
            _ => 0,
        },
    }
}

fn dedup(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for path in paths {
        if !out.contains(path) {
            out.push(path.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use immortality_core::domain::{AttemptState, Dod, SubtaskId, SubtaskState};
    use immortality_core::protocol::Idempotency;
    use immortality_tools::tool::RegistryError;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    /// A model that answers from a script, then refuses to answer again.
    #[derive(Debug)]
    struct Scripted {
        replies: Mutex<VecDeque<Result<Reply, String>>>,
        seen: Mutex<Vec<ChatRequest>>,
    }

    impl Scripted {
        fn new(replies: Vec<Result<Reply, String>>) -> Arc<Self> {
            Arc::new(Self {
                replies: Mutex::new(replies.into()),
                seen: Mutex::new(Vec::new()),
            })
        }

        fn requests(&self) -> Vec<ChatRequest> {
            self.seen.lock().map(|guard| guard.clone()).unwrap_or_default()
        }
    }

    impl Model for Scripted {
        fn complete(&self, request: &ChatRequest) -> Result<Reply, String> {
            if let Ok(mut seen) = self.seen.lock() {
                seen.push(request.clone());
            }
            self.replies
                .lock()
                .ok()
                .and_then(|mut queue| queue.pop_front())
                .unwrap_or_else(|| Err("the script has no reply for this step".to_string()))
        }

        /// Feeds the scripted text through the watcher one word at a time.
        ///
        /// The default implementation hands the watcher the finished message in one call, which
        /// cannot abort mid-message and therefore cannot exercise D4 the way a real stream does.
        /// Word-by-word is the smallest fidelity that still crosses delta boundaries, which is
        /// where a repetition detector can go wrong.
        fn complete_watched(
            &self,
            request: &ChatRequest,
            watch: &mut dyn FnMut(&str) -> bool,
        ) -> Result<Streamed, String> {
            let reply = self.complete(request)?;
            let mut seen = String::new();
            for piece in reply.content.split_inclusive(' ') {
                seen.push_str(piece);
                if !watch(piece) {
                    return Ok(Streamed::Aborted { seen });
                }
            }
            // A content-free reply (pure tool calls) still has to reach the watcher once, so a
            // watcher that counts calls sees the turn it actually got.
            if reply.content.is_empty() && !watch("") {
                return Ok(Streamed::Aborted { seen });
            }
            Ok(Streamed::Complete(reply))
        }
    }

    /// A model that spends wall time: each reply advances the shared clock.
    ///
    /// The 45-minute budget is the one limit no test can wait for, so the only way to cover it
    /// is a clock the loop reads and a step that moves it. This is not a fake of the budget — it
    /// is the budget, measured through the seam that production reads too.
    #[derive(Debug)]
    struct Slow {
        replies: Mutex<VecDeque<Result<Reply, String>>>,
        clock: Arc<immortality_core::time::FixedClock>,
        per_step: chrono::Duration,
    }

    impl Slow {
        fn new(
            replies: Vec<Result<Reply, String>>,
            clock: Arc<immortality_core::time::FixedClock>,
            per_step: chrono::Duration,
        ) -> Arc<Self> {
            Arc::new(Self {
                replies: Mutex::new(replies.into()),
                clock,
                per_step,
            })
        }
    }

    impl Model for Slow {
        fn complete(&self, _request: &ChatRequest) -> Result<Reply, String> {
            self.clock.advance(self.per_step);
            self.replies
                .lock()
                .ok()
                .and_then(|mut queue| queue.pop_front())
                .unwrap_or_else(|| Err("the script has no reply for this step".to_string()))
        }
    }

    /// A tool that always succeeds and touches one path.
    #[derive(Debug)]
    struct Touch;
    impl immortality_tools::Tool for Touch {
        fn name(&self) -> &'static str {
            "touch"
        }

        fn schema(&self) -> &'static serde_json::Value {
            static SCHEMA: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
            SCHEMA.get_or_init(|| {
                serde_json::json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"],
                    "additionalProperties": false
                })
            })
        }

        fn description(&self) -> &'static str {
            "Write content to a path."
        }

        fn effect(&self, _args: &Args) -> immortality_tools::Effect {
            immortality_tools::Effect::Mutate
        }

        fn idempotency(&self, _args: &Args) -> Idempotency {
            Idempotency::ByContent
        }

        fn paths(&self, args: &Args) -> Vec<immortality_tools::PathUse> {
            args.opt_str("path")
                .ok()
                .flatten()
                .map(|path| vec![immortality_tools::PathUse::write(path)])
                .unwrap_or_default()
        }

        fn execute(&self, _ctx: &ToolCtx, args: Args) -> Result<immortality_tools::ToolResult, ToolError> {
            let path = args.str("path")?;
            Ok(immortality_tools::ToolResult::text(format!("wrote {path}")))
        }
    }

    fn subtask() -> Subtask {
        Subtask {
            id: SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAA"),
            depth: 1,
            title: "add the parser".to_string(),
            goal: "parse trailing commas".to_string(),
            dod: Dod::reviewed("trailing commas parse"),
            state: SubtaskState::Pending,
            depends_on: Vec::new(),
            current_attempt: None,
            attempts: Vec::new(),
            report: None,
            children: Vec::new(),
            created_at: immortality_core::time::now(),
            updated_at: immortality_core::time::now(),
        }
    }

    fn runner(model: Arc<dyn Model>, registry: Registry) -> (tempfile::TempDir, Runner) {
        let dir = tempfile::tempdir().expect("tempdir");
        let worktree = dir.path().join("worktree");
        std::fs::create_dir_all(&worktree).expect("mkdir");
        let gate = PolicyGate::new(
            &worktree,
            &["**".to_string()],
            &["**/.git/config".to_string()],
            immortality_core::config::Policy {
                workspace_root: dir.path().to_path_buf(),
                ..immortality_core::config::Policy::default()
            },
        )
        .expect("gate");
        let runner = Runner {
            model,
            model_name: "test-model".to_string(),
            registry: Arc::new(registry),
            artifacts: ArtifactStore::open(dir.path().join("artifacts")).expect("store"),
            gate: Arc::new(gate),
            system_prompt: "You are an agent.".to_string(),
            max_tool_output_bytes: 8192,
            budget: Budget {
                soft: 32_000,
                hard: 64_000,
            },
            limits: AttemptLimits {
                max_steps: 10,
                max_tokens: 100_000,
                stall_steps: 3,
                max_wall_time: Duration::from_secs(60),
            },
            counter: Arc::new(immortality_core::context::CharsCounter::default()),
            task: TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAA"),
            detectors: Detectors::default(),
            max_subtask_depth: 2,
            // A green world by default: these tests are about the loop, and a fixture that
            // failed its own gates would make every claim test assert a failure instead.
            gates: Some(Arc::new(FakeGates::green())),
            reviewer: Some(Arc::new(FakeReviewer::approving())),
            // No git repository in the loop fixtures: a restart that needs a rollback now stops
            // with a named reason instead of silently pretending it rolled back.
            checkpoints: None,
            clock: Arc::new(immortality_core::time::SystemClock),
        };
        (dir, runner)
    }

    fn journal(dir: &tempfile::TempDir) -> Journal {
        Journal::open(
            dir.path().join("journal"),
            TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAJ"),
        )
        .expect("journal")
    }

    /// The gate world a test sees: a diff it returns and command results it scripts.
    ///
    /// `Mutex` rather than `RefCell` because [`Runner`] holds `Arc<dyn Gates + Send + Sync>`: a
    /// fake that is not `Sync` would force every runner test onto a different trait object than
    /// production uses, and then the seam would not be the seam.
    #[derive(Debug, Default)]
    struct FakeGates {
        diff: Mutex<immortality_core::gate::Diff>,
        refused: Mutex<Vec<String>>,
        runs: Mutex<Vec<String>>,
        failing: Mutex<Vec<String>>,
    }

    impl FakeGates {
        fn green() -> Self {
            Self::default()
        }

        fn with_diff(diff: immortality_core::gate::Diff) -> Self {
            Self {
                diff: Mutex::new(diff),
                ..Self::default()
            }
        }

        fn refusing(command: &str) -> Self {
            let gates = Self::default();
            gates.refused.lock().expect("lock").push(command.to_string());
            gates
        }

        fn failing(command: &str) -> Self {
            let gates = Self::default();
            gates.failing.lock().expect("lock").push(command.to_string());
            gates
        }

        fn ran(&self) -> Vec<String> {
            self.runs.lock().map(|runs| runs.clone()).unwrap_or_default()
        }
    }

    impl Gates for FakeGates {
        fn diff(&self, _base: Option<&str>) -> Result<immortality_core::gate::Diff, String> {
            Ok(self.diff.lock().map(|diff| diff.clone()).unwrap_or_default())
        }

        fn authorise(&self, command: &str) -> Result<(), String> {
            let refused = self.refused.lock().map(|r| r.clone()).unwrap_or_default();
            if refused.iter().any(|item| item == command) {
                return Err("the command needs a human".to_string());
            }
            Ok(())
        }

        fn run(&self, command: &str) -> Result<immortality_core::gate::CommandRun, String> {
            if let Ok(mut runs) = self.runs.lock() {
                runs.push(command.to_string());
            }
            let failed = self
                .failing
                .lock()
                .is_ok_and(|failing| failing.iter().any(|item| item == command));
            Ok(immortality_core::gate::CommandRun {
                exit_code: Some(i32::from(failed)),
                timed_out: false,
                duration_ms: 1,
                output_tail: if failed { "boom".to_string() } else { String::new() },
            })
        }
    }

    /// A reviewer with a scripted answer, and no model behind it.
    #[derive(Debug)]
    struct FakeReviewer {
        answer: Result<String, String>,
    }

    impl FakeReviewer {
        fn approving() -> Self {
            Self {
                answer: Ok(r#"{"verdict":"approve"}"#.to_string()),
            }
        }

        fn rejecting(issue: &str) -> Self {
            Self {
                answer: Ok(format!(r#"{{"verdict":"request_changes","issues":["{issue}"]}}"#)),
            }
        }

        fn broken() -> Self {
            Self {
                answer: Err("the backend is down".to_string()),
            }
        }
    }

    impl Reviewer for FakeReviewer {
        fn review(&self, _request: &immortality_core::gate::ReviewRequest) -> Result<String, String> {
            self.answer.clone()
        }
    }

    fn registry(tools: Vec<Arc<dyn immortality_tools::Tool>>) -> Registry {
        Registry::new(tools).expect("registry")
    }

    fn finish() -> Reply {
        Reply::call("call_finish", FINISH_SUBTASK, r#"{"summary":"done"}"#)
    }

    #[test]
    fn a_claim_ends_the_attempt_and_produces_a_report() {
        let model = Scripted::new(vec![
            Ok(Reply::call("call_1", "touch", r#"{"path":"src/parser.rs"}"#)),
            Ok(finish()),
        ]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert!(outcome.claimed_done());
        assert_eq!(outcome.steps, 2);
        assert_eq!(outcome.report.files_touched, vec![PathBuf::from("src/parser.rs")]);
        assert_eq!(outcome.report.outcome, "done");
    }

    #[test]
    fn a_claim_with_green_gates_closes_the_attempt_and_the_subtask() {
        // The one path that writes `Succeeded`: the model claimed, and every gate the `DoD`
        // asked for actually passed.
        let mut subtask = subtask();
        subtask.dod = Dod::reviewed("trailing commas parse");
        let gates = Arc::new(FakeGates::green());
        let model = Scripted::new(vec![Ok(finish())]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.gates = Some(gates);
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask, 1, "", &mut journal).expect("run");

        assert_eq!(outcome.attempt.state, AttemptState::Succeeded);
        assert_eq!(outcome.attempt.failure, None);
        assert!(outcome.attempt.finished_at.is_some());
        // The runner closes the attempt and hands the verdict up. Writing `done` into the tree is
        // the scheduler's job (D-078's order is tree first, then journal), so a `subtask_state`
        // record here would be a closure the plan never received.
        assert!(outcome.verified(), "{:?}", outcome.gates);
        assert_eq!(
            subtask_states(&dir),
            Vec::new(),
            "the runner must not announce the subtask done"
        );
    }

    #[test]
    fn a_claim_without_a_gate_runner_does_not_close_anything() {
        // The wiring bug this closes: a runner that could not verify anything treated "no gates"
        // as "no gates needed", and every caller that forgot to wire them was a bypass.
        let model = Scripted::new(vec![Ok(finish())]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.gates = None;
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert_eq!(outcome.attempt.state, AttemptState::Failed);
        assert_eq!(outcome.attempt.failure, Some(FailureReason::GateFailed));
        assert!(
            outcome.report.outcome.contains("cannot be verified"),
            "{:?}",
            outcome.report.outcome
        );
        assert!(
            !subtask_states(&dir)
                .iter()
                .any(|(_, state)| *state == SubtaskState::Done)
        );
    }

    #[test]
    fn a_red_required_check_leaves_the_attempt_failed_and_names_it() {
        let mut subtask = subtask();
        subtask.dod = Dod {
            statement: "the build is green".to_string(),
            checks: vec![immortality_core::domain::DodCheck {
                id: "build".to_string(),
                command: "cargo build".to_string(),
                required: true,
            }],
            review: false,
        };
        let gates = Arc::new(FakeGates::failing("cargo build"));
        let model = Scripted::new(vec![Ok(finish())]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.gates = Some(gates);
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask, 1, "", &mut journal).expect("run");

        assert_eq!(outcome.attempt.state, AttemptState::Failed);
        assert_eq!(outcome.attempt.failure, Some(FailureReason::GateFailed));
        assert!(
            outcome.report.outcome.contains("build"),
            "the next brief must name the check: {:?}",
            outcome.report.outcome
        );
        assert!(
            outcome
                .report
                .rejected_ideas
                .iter()
                .any(|idea| idea.contains("rejected")),
            "{:?}",
            outcome.report.rejected_ideas
        );
        assert!(
            !subtask_states(&dir)
                .iter()
                .any(|(_, state)| *state == SubtaskState::Done)
        );
    }

    #[test]
    fn a_red_optional_check_does_not_block_the_close() {
        // `cargo outdated` printing something is information. The flag is what distinguishes it
        // from a gate, and this is the test that says so end to end.
        let mut subtask = subtask();
        subtask.dod = Dod {
            statement: "the build is green".to_string(),
            checks: vec![
                immortality_core::domain::DodCheck {
                    id: "outdated".to_string(),
                    command: "cargo outdated".to_string(),
                    required: false,
                },
                immortality_core::domain::DodCheck {
                    id: "build".to_string(),
                    command: "cargo build".to_string(),
                    required: true,
                },
            ],
            review: false,
        };
        let model = Scripted::new(vec![Ok(finish())]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.gates = Some(Arc::new(FakeGates::failing("cargo outdated")));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask, 1, "", &mut journal).expect("run");

        assert_eq!(
            outcome.attempt.state,
            AttemptState::Succeeded,
            "{:?}",
            outcome.attempt
        );
    }

    #[test]
    fn a_refused_check_is_not_a_passing_one_end_to_end() {
        // The policy said no, so nothing ran. A gate that counted a refusal as success would be
        // the exact bypass this milestone exists to close.
        let mut subtask = subtask();
        subtask.dod = Dod {
            statement: "the tests pass".to_string(),
            checks: vec![immortality_core::domain::DodCheck {
                id: "test".to_string(),
                command: "sudo cargo test".to_string(),
                required: true,
            }],
            review: false,
        };
        let gates = Arc::new(FakeGates::refusing("sudo cargo test"));
        let model = Scripted::new(vec![Ok(finish())]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.gates = Some(gates.clone());
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask, 1, "", &mut journal).expect("run");

        assert_eq!(outcome.attempt.state, AttemptState::Failed);
        assert_eq!(outcome.attempt.failure, Some(FailureReason::GateFailed));
        assert_eq!(gates.ran(), Vec::<String>::new(), "nothing may run");
    }

    #[test]
    fn a_reviewer_rejection_is_its_own_failure_reason() {
        let model = Scripted::new(vec![Ok(finish())]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.reviewer = Some(Arc::new(FakeReviewer::rejecting(
            "src/parser.rs still drops the trailing comma",
        )));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert_eq!(outcome.attempt.state, AttemptState::Failed);
        assert_eq!(outcome.attempt.failure, Some(FailureReason::ReviewRejected));
        assert!(
            outcome
                .report
                .rejected_ideas
                .iter()
                .any(|idea| idea.contains("trailing comma")),
            "{:?}",
            outcome.report.rejected_ideas
        );
    }

    #[test]
    fn a_missing_reviewer_with_a_reviewing_dod_does_not_close() {
        // `DoD::reviewed` asks for a review. No reviewer means the gate could not run, and a gate
        // that could not run is not an approval.
        let model = Scripted::new(vec![Ok(finish())]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.reviewer = None;
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert_eq!(outcome.attempt.state, AttemptState::Failed);
        assert_eq!(outcome.attempt.failure, Some(FailureReason::ReviewRejected));
    }

    #[test]
    fn a_reviewer_outage_is_not_an_approval() {
        let model = Scripted::new(vec![Ok(finish())]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.reviewer = Some(Arc::new(FakeReviewer::broken()));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert_eq!(outcome.attempt.state, AttemptState::Failed);
        assert_eq!(outcome.attempt.failure, Some(FailureReason::ReviewRejected));
    }

    #[test]
    fn a_sanity_finding_stops_the_commands_and_the_reviewer() {
        // Cheap-first is a token budget: a diff that adds a private key is already failed, and
        // paying for a two-minute suite and a model call before saying so is waste.
        let mut subtask = subtask();
        subtask.dod = Dod {
            statement: "the build is green".to_string(),
            checks: vec![immortality_core::domain::DodCheck {
                id: "build".to_string(),
                command: "cargo build".to_string(),
                required: true,
            }],
            review: true,
        };
        let mut diff = immortality_core::gate::Diff::default();
        diff.files.push(immortality_core::gate::ChangedFile {
            path: PathBuf::from("src/config.rs"),
            status: immortality_core::gate::FileStatus::Modified,
            binary: false,
            added_lines: vec!["-----BEGIN RSA PRIVATE KEY-----".to_string()],
        });
        let gates = Arc::new(FakeGates::with_diff(diff));
        let model = Scripted::new(vec![Ok(finish())]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.gates = Some(gates.clone());
        // A reviewer that rejects everything: if it were called, the failure reason would be
        // `ReviewRejected`, so the assertion below also proves it was not.
        runner.reviewer = Some(Arc::new(FakeReviewer::rejecting("anything")));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask, 1, "", &mut journal).expect("run");

        assert_eq!(outcome.attempt.failure, Some(FailureReason::GateFailed));
        assert_eq!(gates.ran(), Vec::<String>::new(), "no command may run");
        assert!(
            outcome.report.outcome.contains("private key"),
            "{:?}",
            outcome.report.outcome
        );
    }

    #[test]
    fn the_gate_records_are_written_in_the_order_they_ran() {
        // The journal is the post-mortem. A verdict written before the check it decided would
        // read as a cause after its effect.
        let mut subtask = subtask();
        subtask.dod = Dod::reviewed("trailing commas parse");
        let model = Scripted::new(vec![Ok(finish())]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.gates = Some(Arc::new(FakeGates::green()));
        let mut journal = journal(&dir);
        runner.run(&subtask, 1, "", &mut journal).expect("run");

        let gates: Vec<String> = journal_records(&dir)
            .into_iter()
            .filter(|record| record.fields.kind == Kind::GateResult)
            .filter_map(|record| {
                record
                    .event
                    .payload
                    .get("gate")
                    .and_then(|gate| gate.as_str())
                    .map(str::to_string)
            })
            .collect();
        assert_eq!(gates, vec!["sanity", "review", "verdict"], "{gates:?}");
    }

    #[test]
    fn a_gate_record_carries_the_attempt_scope() {
        // Recovery keys attempts by `(subtask, attempt)`. A gate record without the scope is
        // invisible to it, which is how a rejected claim would look like an open attempt.
        let model = Scripted::new(vec![Ok(finish())]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Touch)]));
        let subtask = subtask();
        let mut journal = journal(&dir);
        runner.run(&subtask, 3, "", &mut journal).expect("run");

        let gate_records: Vec<_> = journal_records(&dir)
            .into_iter()
            .filter(|record| record.fields.kind == Kind::GateResult)
            .collect();
        assert_ne!(gate_records, Vec::new());
        for record in gate_records {
            assert_eq!(record.fields.subtask.as_ref(), Some(&subtask.id));
            assert_eq!(record.fields.attempt, Some(3));
        }
    }

    #[test]
    fn the_verdict_record_is_last_and_says_whether_the_claim_passed() {
        let model = Scripted::new(vec![Ok(finish())]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        runner.run(&subtask(), 1, "", &mut journal).expect("run");

        let verdict = journal_records(&dir)
            .into_iter()
            .rfind(|record| record.fields.kind == Kind::GateResult)
            .expect("a gate record")
            .event
            .payload;
        assert_eq!(verdict["gate"], "verdict");
        assert_eq!(verdict["passed"], true);
    }

    /// The subtask-state records in the journal, as `(kind, state)` pairs.
    fn subtask_states(dir: &tempfile::TempDir) -> Vec<(Kind, SubtaskState)> {
        journal_records(dir)
            .into_iter()
            .filter(|record| record.fields.kind == Kind::SubtaskState)
            .filter_map(|record| {
                let state: SubtaskState =
                    serde_json::from_value(record.event.payload.get("state")?.clone()).ok()?;
                Some((record.fields.kind, state))
            })
            .collect()
    }

    fn journal_records(dir: &tempfile::TempDir) -> Vec<immortality_core::journal::JournalRecord> {
        immortality_core::journal::JournalReader::open(
            dir.path().join("journal"),
            &TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAJ"),
        )
        .expect("reader")
        .read()
        .expect("records")
        .records
    }

    #[test]
    fn the_claim_summary_is_what_the_next_attempt_inherits() {
        // The report is the only channel between attempts. The model's own summary is a
        // deliberate statement; the last observation is whatever happened to come last.
        let model = Scripted::new(vec![
            Ok(Reply::call("call_1", "touch", r#"{"path":"a"}"#)),
            Ok(Reply::call(
                "call_finish",
                FINISH_SUBTASK,
                r#"{"summary":"parser handles trailing commas; tests added"}"#,
            )),
        ]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert_eq!(
            outcome.report.outcome,
            "parser handles trailing commas; tests added"
        );
    }

    #[test]
    fn a_claim_without_a_summary_falls_back_to_the_last_observation() {
        // The schema requires `summary`, but a model can still send `{}`. Losing the claim over
        // that would trade a usable report for none.
        let model = Scripted::new(vec![
            Ok(Reply::call("call_1", "touch", r#"{"path":"a"}"#)),
            Ok(Reply::call("call_finish", FINISH_SUBTASK, "{}")),
        ]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert!(outcome.claimed_done());
        assert!(
            outcome.report.outcome.contains("wrote a"),
            "{:?}",
            outcome.report.outcome
        );
    }

    #[test]
    fn the_model_is_offered_finish_subtask() {
        // A model cannot call what it was never offered. Without the declaration the runner
        // recognises the name and the model never produces it, so every run ends at the step
        // budget instead of on a claim.
        let model = Scripted::new(vec![Ok(finish())]);
        let (dir, runner) = runner(model.clone(), registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        runner.run(&subtask(), 1, "", &mut journal).expect("run");

        let request = model.requests().pop().expect("a request");
        let tools = request.tools.expect("tools are offered");
        let names: Vec<&str> = tools.iter().map(|tool| tool.function.name.as_str()).collect();
        assert!(names.contains(&FINISH_SUBTASK), "{names:?}");
        // The real tools are still there: the declaration is an addition, not a replacement.
        assert!(names.contains(&"touch"), "{names:?}");
    }

    #[test]
    fn the_model_is_offered_split_subtask() {
        // Same reason as `finish_subtask`: a proposal the model cannot name is a proposal it can
        // never make, and L4 would then depend on luck.
        let model = Scripted::new(vec![Ok(finish())]);
        let (dir, runner) = runner(model.clone(), registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        runner.run(&subtask(), 1, "", &mut journal).expect("run");

        let request = model.requests().pop().expect("a request");
        let tools = request.tools.expect("tools are offered");
        let names: Vec<&str> = tools.iter().map(|tool| tool.function.name.as_str()).collect();
        assert!(names.contains(&SPLIT_SUBTASK), "{names:?}");
    }

    #[test]
    fn a_split_proposal_ends_the_attempt_without_touching_the_tree() {
        // The runner has no tree to touch: it stops and reports, and the scheduler decides. An
        // attempt that kept going would keep spending steps on a subtask about to disappear.
        let model = Scripted::new(vec![Ok(Reply::call(
            "call_split",
            SPLIT_SUBTASK,
            r#"{"reason":"too large","pieces":[{"title":"parse","goal":"parse the header"},{"title":"emit","goal":"emit the body"}]}"#,
        ))]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        match &outcome.ending {
            Ending::SplitRequested(pieces) => {
                assert_eq!(pieces.len(), 2);
                assert_eq!(pieces[0].title, "parse");
                assert_eq!(pieces[1].goal, "emit the body");
            }
            other => panic!("expected a split request, got {other:?}"),
        }
        assert_eq!(outcome.steps, 1);
        assert_eq!(outcome.attempt.failure, Some(FailureReason::Split));
        assert_eq!(outcome.attempt.state, AttemptState::Failed);
    }

    #[test]
    fn a_split_with_one_piece_is_an_observation_not_an_ending() {
        // The schema says `minItems: 2`, but a schema is advice to a model. A single piece would
        // be the same work under a new id with a fresh attempts budget, so it is refused here and
        // the model is told why instead of the attempt ending on a bad call.
        let model = Scripted::new(vec![
            Ok(Reply::call(
                "call_split",
                SPLIT_SUBTASK,
                r#"{"pieces":[{"title":"all of it","goal":"do the whole thing"}]}"#,
            )),
            Ok(finish()),
        ]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert!(outcome.claimed_done(), "{:?}", outcome.ending);
        let records = immortality_core::journal::JournalReader::open(
            dir.path().join("journal"),
            &TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAJ"),
        )
        .expect("reader")
        .read()
        .expect("records")
        .records;
        assert!(
            records.iter().any(|record| {
                record.fields.kind == Kind::InvalidToolcall && record.event.payload["tool"] == SPLIT_SUBTASK
            }),
            "the model must be told what was wrong"
        );
    }

    #[test]
    fn a_split_with_too_many_pieces_is_an_observation() {
        // The schema says `maxItems: 8`, but the split call never reaches the generic argument
        // validation, so an unenforced bound would be advice. Ten headings are a plan rewrite,
        // not a decomposition, and a plan rewrite needs a human.
        let pieces: Vec<String> = (0..=MAX_SPLIT_PIECES)
            .map(|index| format!(r#"{{"title":"p{index}","goal":"do {index}"}}"#))
            .collect();
        let arguments = format!(r#"{{"pieces":[{}]}}"#, pieces.join(","));
        let model = Scripted::new(vec![
            Ok(Reply::call("call_split", SPLIT_SUBTASK, arguments)),
            Ok(finish()),
        ]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert!(outcome.claimed_done(), "{:?}", outcome.ending);
    }

    #[test]
    fn a_split_at_the_depth_limit_is_refused_as_not_offered() {
        // The tool is withheld at `max_depth`, so a model that calls it anyway is answered with
        // the reason rather than silently obeyed into a tree the loader would reject.
        let mut deep = subtask();
        deep.depth = 2;
        let model = Scripted::new(vec![
            Ok(Reply::call(
                "call_split",
                SPLIT_SUBTASK,
                r#"{"pieces":[{"title":"a","goal":"do a"},{"title":"b","goal":"do b"}]}"#,
            )),
            Ok(finish()),
        ]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        let outcome = runner.run(&deep, 1, "", &mut journal).expect("run");

        assert!(outcome.claimed_done(), "{:?}", outcome.ending);
    }

    #[test]
    fn the_brief_offers_a_split_only_where_the_tree_has_room() {
        let shallow = brief_for(&subtask(), "", 2);
        assert!(shallow.contains(SPLIT_SUBTASK), "{shallow}");
        let mut deep = subtask();
        deep.depth = 2;
        let deep_brief = brief_for(&deep, "", 2);
        assert!(!deep_brief.contains(SPLIT_SUBTASK), "{deep_brief}");
    }

    #[test]
    fn the_attempt_records_carry_the_subtask_and_attempt_number() {
        // Recovery keys attempts by `(subtask, attempt)` to find the ones a crash left running.
        // An `attempt_started` without the scope is invisible to it, so the attempt would stay
        // open forever and the subtask would wait for a process that no longer exists.
        let model = Scripted::new(vec![Ok(Reply::call("call_1", "touch", r#"{"path":"a"}"#))]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.limits.max_steps = 1;
        let subtask = subtask();
        let mut journal = journal(&dir);
        runner.run(&subtask, 2, "try again", &mut journal).expect("run");

        let reader = immortality_core::journal::JournalReader::open(
            dir.path().join("journal"),
            &TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAJ"),
        )
        .expect("reader");
        let records = reader.read().expect("records").records;
        assert_ne!(records.len(), 0);
        // Every record the runner writes belongs to this attempt: the step records, the tool
        // protocol records and the attempt's own two. A single unscoped one is invisible to
        // recovery's attempt grouping, which is how an attempt stays `Running` after a crash.
        for record in &records {
            assert_eq!(
                record.fields.subtask.as_ref(),
                Some(&subtask.id),
                "{:?}: {record:?}",
                record.fields.kind
            );
            assert_eq!(record.fields.attempt, Some(2), "{:?}", record.fields.kind);
        }
    }

    #[test]
    fn the_next_attempt_gets_a_context_without_the_previous_history() {
        // The design's requirement: a fresh context per subtask, and the report is the only
        // channel between them.
        let model = Scripted::new(vec![
            Ok(Reply::call("call_1", "touch", r#"{"path":"src/a.rs"}"#)),
            Ok(Reply::text("secret intermediate reasoning")),
            Ok(finish()),
            Ok(finish()),
        ]);
        let (dir, runner) = runner(model.clone(), registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        let first = runner.run(&subtask(), 1, "", &mut journal).expect("first");
        let second = runner
            .run(&subtask(), 2, "try another way", &mut journal)
            .expect("second");

        assert!(first.claimed_done());
        assert!(second.claimed_done());

        let requests = model.requests();
        let second_context = &requests[3];
        let system = second_context.messages[0]
            .content
            .as_deref()
            .expect("a system message");
        assert!(
            !system.contains("secret intermediate reasoning"),
            "the previous step history leaked into the next attempt: {system}"
        );
        assert!(
            system.contains("try another way"),
            "the new strategy must be in the brief: {system}"
        );
        // Two attempts, two histories: the second request of the second run is the first step
        // of a new context, not a continuation. Two messages, not one: the constant user turn
        // is part of the prefix and the backend rejects a request without it.
        assert_eq!(
            second_context.messages.len(),
            2,
            "a fresh context has only the prefix: a system message and the user turn"
        );
        assert_eq!(second_context.messages[1].role, "user");
    }

    #[test]
    fn the_prefix_is_byte_identical_across_steps_of_one_attempt() {
        // The cache is the whole reason the prefix is immutable; a runner that rebuilt it per
        // step would defeat D-074 while every `Context` test stayed green.
        let model = Scripted::new(vec![
            Ok(Reply::call("call_1", "touch", r#"{"path":"a"}"#)),
            Ok(Reply::call("call_2", "touch", r#"{"path":"b"}"#)),
            Ok(finish()),
        ]);
        let (dir, runner) = runner(model.clone(), registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        runner.run(&subtask(), 1, "", &mut journal).expect("run");

        let requests = model.requests();
        assert_eq!(requests.len(), 3);
        let prefixes: Vec<&str> = requests
            .iter()
            .map(|request| request.messages[0].content.as_deref().unwrap_or(""))
            .collect();
        assert_eq!(prefixes[0], prefixes[1]);
        assert_eq!(prefixes[1], prefixes[2]);
    }

    #[test]
    fn a_growing_prompt_is_charged_for_what_it_added_not_for_what_it_reread() {
        // The live run's first defect. Every step re-sends the whole prompt, and its prefix is the
        // bytes the previous step already paid for. Charging `prompt + completion` each time made
        // `max_tokens_per_attempt` a disguised step limit: at ~10K a step, the 120000 budget was
        // spent after twelve of the sixty steps the config allows, and two of three subtasks
        // restarted for "it ran out of tokens" while the context was nowhere near its 64K budget.
        //
        // Fourteen steps with a 10K prompt and a 100-token completion cost 14 * 100 + the prompt's
        // growth, not 14 * 10100.
        let replies = (0..14)
            .map(|index| {
                Ok(Reply {
                    content: String::new(),
                    // A distinct path each step, so D1's exact-repeat ladder is not what ends this.
                    tool_calls: vec![immortality_llm::turn::ToolCall::function(
                        format!("call_{index}"),
                        "touch",
                        format!(r#"{{"path":"a{index}"}}"#),
                    )],
                    usage: Some((10_000 + index * 100, 100)),
                })
            })
            .collect();
        let model = Scripted::new(replies);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.limits.max_steps = 14;
        runner.limits.max_tokens = 120_000;
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert_eq!(
            outcome.ending,
            Ending::Steps,
            "the step budget must be what ends this, not the token budget"
        );
        assert_eq!(outcome.steps, 14);
        // Completion is 100 a step; the prompt grows 100 a step; so the honest total is
        // 14 * 100 + (10_000 + ... + 11_300) - 10_000, well under 120000.
        assert!(
            outcome.attempt.tokens_used < 120_000,
            "the attempt spent {} tokens, which is the double-counted number",
            outcome.attempt.tokens_used
        );
    }

    #[test]
    fn a_prompt_that_does_not_grow_costs_only_its_completion() {
        // The other half: a repeated prompt of identical size is one prompt paid once. A model
        // reading a cached prefix does not re-bill the whole context.
        assert_eq!(attempt_charge(Some((10_000, 120)), 10_000), (120, 10_000));
        assert_eq!(attempt_charge(Some((10_000, 120)), 0), (10_120, 10_000));
        // A prompt that shrank charges nothing extra rather than a negative amount.
        assert_eq!(attempt_charge(Some((4_000, 50)), 10_000), (50, 10_000));
        // A backend that reported nothing charges nothing, rather than guessing.
        assert_eq!(attempt_charge(None, 10_000), (0, 10_000));
    }

    #[test]
    fn the_step_budget_ends_the_attempt_and_says_so() {
        let replies = (0..20)
            .map(|index| Ok(Reply::call(format!("call_{index}"), "touch", r#"{"path":"x"}"#)))
            .collect();
        let model = Scripted::new(replies);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.limits.max_steps = 3;
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert_eq!(outcome.ending, Ending::Steps);
        assert_eq!(outcome.steps, 3);
        assert_eq!(outcome.attempt.failure, Some(FailureReason::StepsExhausted));
        assert_eq!(outcome.attempt.state, AttemptState::Failed);
        assert!(outcome.report.outcome.contains("did not finish"));
    }

    #[test]
    fn the_time_budget_fails_the_attempt_rather_than_abandoning_it() {
        // `max_wall_time_per_attempt` is the attempt's own limit, like steps and tokens: it is
        // what D5 exists to react to, so it must count against the failure budget. `Abandoned`
        // is for causes that say nothing about the work — a backend outage, a daemon restart.
        let model = Scripted::new(vec![Ok(Reply::call("call_1", "touch", r#"{"path":"a"}"#))]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.limits.max_wall_time = Duration::ZERO;
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert_eq!(outcome.ending, Ending::Time);
        assert_eq!(outcome.attempt.state, AttemptState::Failed);
        assert_eq!(outcome.attempt.failure, Some(FailureReason::TimeExhausted));
        assert!(outcome.attempt.state.counts_as_failure());
    }

    #[test]
    fn the_wall_time_budget_is_measured_on_the_injected_clock() {
        // The design's 45 minutes. Waiting for them is not a test, so the loop reads a clock the
        // test can move: one step is allowed to spend 46 of them in zero real time.
        let clock = Arc::new(immortality_core::time::FixedClock::fixture());
        let model = Slow::new(
            vec![Ok(Reply::call("call_1", "touch", r#"{"path":"a"}"#))],
            Arc::clone(&clock),
            chrono::Duration::minutes(46),
        );
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.clock = Arc::clone(&clock) as Arc<dyn Clock>;
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert_eq!(outcome.ending, Ending::Time);
        assert_eq!(outcome.attempt.failure, Some(FailureReason::TimeExhausted));
        // One step ran; the second iteration found the budget gone. A budget checked only
        // before the first step would have run all ten.
        assert_eq!(outcome.steps, 1);
    }

    #[test]
    fn a_step_that_fits_the_budget_does_not_end_the_attempt() {
        // The other half of the guard: the clock seam must not make every attempt look expired.
        let clock = Arc::new(immortality_core::time::FixedClock::fixture());
        let model = Slow::new(
            vec![Ok(Reply::call("call_1", "touch", r#"{"path":"a"}"#))],
            Arc::clone(&clock),
            chrono::Duration::minutes(44),
        );
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.clock = Arc::clone(&clock) as Arc<dyn Clock>;
        runner.limits.max_steps = 1;
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert_eq!(outcome.ending, Ending::Steps);
    }

    #[test]
    fn every_journal_timestamp_comes_from_the_runner_clock() {
        // This is why the clock exists: a golden journal is only golden if its `ts` fields are
        // the seeded ones rather than whatever the machine happened to say. A single record read
        // from `Utc::now()` would make the whole file uncomparable.
        let clock = Arc::new(immortality_core::time::FixedClock::fixture());
        let model = Scripted::new(vec![Ok(Reply::call("call_1", "touch", r#"{"path":"a"}"#))]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner.clock = Arc::clone(&clock) as Arc<dyn Clock>;
        runner.limits.max_steps = 1;
        let mut journal = journal(&dir);
        runner.run(&subtask(), 1, "", &mut journal).expect("run");

        let records = journal_records(&dir);
        assert!(
            !records.is_empty(),
            "an attempt writes at least its own start and end"
        );
        let start = immortality_core::time::FixedClock::fixture().now();
        for record in &records {
            assert_eq!(
                record.fields.at,
                start,
                "{} was not stamped from the clock",
                record.fields.kind.as_str()
            );
        }
    }

    #[test]
    fn a_talking_model_stalls_out_rather_than_looping_forever() {
        // The failure mode the stall budget exists for: a model that answers prose and never
        // calls a tool would otherwise run to the step budget, spending every step's tokens.
        let model = Scripted::new(vec![
            Ok(Reply::text("thinking")),
            Ok(Reply::text("still thinking")),
            Ok(Reply::text("thinking harder")),
        ]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert_eq!(outcome.ending, Ending::Stalled);
        assert_eq!(outcome.steps, 3);
    }

    #[test]
    fn malformed_arguments_are_told_to_the_model_and_do_not_end_the_attempt() {
        // A recoverable mistake: the model must get the chance to fix the call, and the step
        // it spends doing so must not be counted as the end of the run.
        let model = Scripted::new(vec![
            Ok(Reply::call("call_1", "touch", "{not json")),
            Ok(Reply::call("call_2", "touch", r#"{"path":"fixed"}"#)),
            Ok(finish()),
        ]);
        let (dir, runner) = runner(model.clone(), registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert!(outcome.claimed_done(), "{:?}", outcome.ending);
        let requests = model.requests();
        let second = &requests[1];
        let tool_message = second
            .messages
            .iter()
            .find(|message| message.role == "tool")
            .expect("the model was told");
        let content = tool_message.content.as_deref().unwrap_or("");
        assert!(content.contains("not JSON"), "{content}");
        assert!(content.contains("call again"), "{content}");
    }

    #[test]
    fn arguments_that_do_not_fit_the_schema_are_refused_without_running_the_tool() {
        let model = Scripted::new(vec![
            Ok(Reply::call("call_1", "touch", r#"{"wrong":"field"}"#)),
            Ok(finish()),
        ]);
        let (dir, runner) = runner(model.clone(), registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert!(outcome.claimed_done());
        // Nothing was written, so no path was touched.
        assert_eq!(outcome.report.files_touched, Vec::<PathBuf>::new());

        let requests = model.requests();
        let tool_message = requests[1]
            .messages
            .iter()
            .find(|message| message.role == "tool")
            .expect("the model was told");
        assert!(
            tool_message
                .content
                .as_deref()
                .unwrap_or("")
                .contains("was not run"),
            "{:?}",
            tool_message.content
        );
    }

    #[test]
    fn an_unknown_tool_ends_the_attempt_instead_of_being_retried() {
        let model = Scripted::new(vec![Ok(Reply::call("call_1", "no_such_tool", "{}"))]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert_eq!(
            outcome.ending,
            Ending::UnknownTool {
                name: "no_such_tool".to_string()
            }
        );
    }

    #[test]
    fn a_backend_failure_abandons_the_attempt_rather_than_failing_it() {
        // A crash or an outage says nothing about the model's work and must not count against
        // the subtask's failure budget.
        let model = Scripted::new(vec![Err("connection refused".to_string())]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert!(matches!(outcome.ending, Ending::Backend { .. }));
        assert_eq!(outcome.attempt.state, AttemptState::Abandoned);
    }

    #[test]
    fn an_empty_reply_ends_the_attempt() {
        let model = Scripted::new(vec![Ok(Reply::default())]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert_eq!(outcome.ending, Ending::Empty);
    }

    #[test]
    fn each_tool_call_is_journaled_intent_started_completed() {
        // The three-record protocol is what makes recovery possible; a runner that skipped a
        // record would leave a call that recovery cannot close.
        let model = Scripted::new(vec![
            Ok(Reply::call("call_1", "touch", r#"{"path":"a"}"#)),
            Ok(finish()),
        ]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        runner.run(&subtask(), 1, "", &mut journal).expect("run");

        let reader = immortality_core::journal::JournalReader::open(
            dir.path().join("journal"),
            &TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAJ"),
        )
        .expect("reader");
        let kinds: Vec<Kind> = reader
            .read()
            .expect("records")
            .records
            .into_iter()
            .map(|record| record.fields.kind)
            .collect();
        let intent = kinds.iter().position(|kind| *kind == Kind::ToolCallIntent);
        let started = kinds.iter().position(|kind| *kind == Kind::ToolCallStarted);
        let completed = kinds.iter().position(|kind| *kind == Kind::ToolCallCompleted);
        assert!(intent < started, "{kinds:?}");
        assert!(started < completed, "{kinds:?}");
    }

    #[test]
    fn the_intent_records_what_this_call_does_not_what_the_tool_is() {
        // The record used to carry `tool.description()`, so every `write_file` in the journal said
        // "Write a complete file, creating parent directories" and never named the path. Recovery
        // quotes this sentence to the operator when it cannot tell whether a call took effect, and
        // `immortality log` shows it before the call runs; both need the specific call.
        let model = Scripted::new(vec![
            Ok(Reply::call("call_1", "touch", r#"{"path":"src/parser.rs"}"#)),
            Ok(finish()),
        ]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        runner.run(&subtask(), 1, "", &mut journal).expect("run");

        let reader = immortality_core::journal::JournalReader::open(
            dir.path().join("journal"),
            &TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAJ"),
        )
        .expect("reader");
        let intent = reader
            .read()
            .expect("records")
            .records
            .into_iter()
            .find(|record| record.fields.kind == Kind::ToolCallIntent)
            .expect("an intent");
        let effect = intent.event.payload["expected_effect"]
            .as_str()
            .unwrap_or_default();
        assert!(
            effect.contains("src/parser.rs"),
            "the effect must name the subject of the call, not the tool: {effect}"
        );
        assert_ne!(
            effect, "Write content to a path.",
            "the tool's description is not an expected effect: {effect}"
        );
        // And the metadata is in the record at all: a journal that only holds arguments cannot be
        // recovered, because `tool` and `idempotency` are what decide whether a replay is safe.
        assert_eq!(intent.event.payload["tool"], serde_json::json!("touch"));
        assert_eq!(
            intent.event.payload["idempotency"],
            serde_json::json!("by_content")
        );
        assert_eq!(
            intent.event.payload["arguments"],
            serde_json::json!({"path": "src/parser.rs"}),
            "the tool's arguments live under `arguments`, so they cannot impersonate a field"
        );
    }

    #[test]
    fn the_brief_names_the_goal_the_gate_and_the_finish_call() {
        let text = brief_for(&subtask(), "roll back and narrow the change", 2);
        assert!(text.contains("parse trailing commas"), "{text}");
        assert!(text.contains("trailing commas parse"), "{text}");
        assert!(text.contains(FINISH_SUBTASK), "{text}");
        assert!(text.contains("roll back and narrow"), "{text}");
        assert!(text.contains("reviewer"), "{text}");
    }

    #[test]
    fn the_brief_lists_the_commands_the_system_will_run() {
        let mut subtask = subtask();
        subtask.dod = Dod {
            statement: "builds".to_string(),
            checks: vec![immortality_core::domain::DodCheck {
                id: "build".to_string(),
                command: "cargo build".to_string(),
                required: true,
            }],
            review: false,
        };
        let text = brief_for(&subtask, "", 2);
        assert!(text.contains("cargo build"), "{text}");
        assert!(text.contains("required"), "{text}");
    }

    #[test]
    fn a_registry_with_a_broken_tool_is_not_constructible() {
        // Guards the test fixture, not the runner: if this ever passes, the tests above are
        // exercising a tool whose schema would have been rejected in production.
        #[derive(Debug)]
        struct Broken;
        impl immortality_tools::Tool for Broken {
            fn name(&self) -> &'static str {
                "broken"
            }
            fn schema(&self) -> &'static serde_json::Value {
                static SCHEMA: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
                SCHEMA.get_or_init(|| serde_json::json!({ "type": "nonsense" }))
            }
            fn description(&self) -> &'static str {
                "broken"
            }
            fn effect(&self, _args: &Args) -> immortality_tools::Effect {
                immortality_tools::Effect::Read
            }
            fn idempotency(&self, _args: &Args) -> Idempotency {
                Idempotency::Idempotent
            }
            fn paths(&self, _args: &Args) -> Vec<immortality_tools::PathUse> {
                Vec::new()
            }
            fn execute(
                &self,
                _ctx: &ToolCtx,
                _args: Args,
            ) -> Result<immortality_tools::ToolResult, ToolError> {
                Ok(immortality_tools::ToolResult::text("never"))
            }
        }
        let error: RegistryError = Registry::new(vec![Arc::new(Broken)]).expect_err("broken schema");
        assert_eq!(error.tool, "broken");
    }

    // ── the detectors ──────────────────────────────────────────────────────────

    /// A tool that answers with whatever `path` names, so a test can script output hashes.
    #[derive(Debug)]
    struct Echo;

    impl immortality_tools::Tool for Echo {
        fn name(&self) -> &'static str {
            "echo"
        }

        fn schema(&self) -> &'static serde_json::Value {
            static SCHEMA: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
            SCHEMA.get_or_init(|| {
                serde_json::json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"],
                    "additionalProperties": false
                })
            })
        }

        fn description(&self) -> &'static str {
            "Echo the path back."
        }

        fn effect(&self, _args: &Args) -> immortality_tools::Effect {
            immortality_tools::Effect::Read
        }

        fn idempotency(&self, _args: &Args) -> Idempotency {
            Idempotency::Idempotent
        }

        fn paths(&self, _args: &Args) -> Vec<immortality_tools::PathUse> {
            Vec::new()
        }

        fn execute(&self, _ctx: &ToolCtx, args: Args) -> Result<immortality_tools::ToolResult, ToolError> {
            Ok(immortality_tools::ToolResult::text(args.str("path")?))
        }
    }

    fn echo_reply(id: &str, path: &str) -> Reply {
        Reply::call(id, "echo", format!(r#"{{"path":"{path}"}}"#))
    }

    /// A tool that always prints the same thing, so two different calls can share an output.
    #[derive(Debug)]
    struct Constant;

    impl immortality_tools::Tool for Constant {
        fn name(&self) -> &'static str {
            "constant"
        }

        fn schema(&self) -> &'static serde_json::Value {
            static SCHEMA: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
            SCHEMA.get_or_init(|| {
                serde_json::json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"],
                    "additionalProperties": false
                })
            })
        }

        fn description(&self) -> &'static str {
            "Print the same text every time, whatever it is asked for."
        }

        fn effect(&self, _args: &Args) -> immortality_tools::Effect {
            immortality_tools::Effect::Read
        }

        fn idempotency(&self, _args: &Args) -> Idempotency {
            Idempotency::Idempotent
        }

        fn paths(&self, _args: &Args) -> Vec<immortality_tools::PathUse> {
            Vec::new()
        }

        fn execute(&self, _ctx: &ToolCtx, _args: Args) -> Result<immortality_tools::ToolResult, ToolError> {
            Ok(immortality_tools::ToolResult::text("the same text"))
        }
    }

    /// Detector records in the journal, as `(detector, level)` pairs.
    ///
    /// Panics rather than skipping a record whose detector or level is missing. That is the
    /// difference between a metric and a guess: the v0.1 criterion is "every firing is journaled
    /// with its type **and** level", and the obvious `filter_map` — which this used to be — makes
    /// a record without a level simply disappear. Every test written on top of it would still
    /// pass, and the journal would be quietly missing the half that says what was done about the
    /// loop. A strict helper turns the omission into a failure in the five tests that read it.
    fn reactions(dir: &tempfile::TempDir) -> Vec<(String, String)> {
        let reader = immortality_core::journal::JournalReader::open(
            dir.path().join("journal"),
            &TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAJ"),
        )
        .expect("reader");
        reader
            .read()
            .expect("records")
            .records
            .into_iter()
            .filter(|record| record.fields.kind == Kind::DetectorFired)
            .map(|record| {
                let detector = record
                    .event
                    .payload
                    .get("detector")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_else(|| {
                        panic!(
                            "a detector_fired record has no `detector`: {}",
                            record.event.payload
                        )
                    });
                let level = record
                    .event
                    .payload
                    .get("level")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_else(|| {
                        panic!("a detector_fired record has no `level`: {}", record.event.payload)
                    });
                (detector.to_string(), level.to_string())
            })
            .collect()
    }

    #[test]
    fn an_exact_repeat_is_nudged_then_forced_then_ends_the_attempt() {
        // The design's ladder: 3 → L1, 5 → L2, 8 → L3. Each rung is a record and the last one
        // ends the attempt, so a model stuck on one call does not spend the whole step budget.
        let replies = (0..20)
            .map(|index| Ok(echo_reply(&format!("call_{index}"), "same")))
            .collect();
        let (dir, runner) = runner(Scripted::new(replies), registry(vec![Arc::new(Echo)]));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert!(
            matches!(outcome.ending, Ending::Looped(_)),
            "{:?}",
            outcome.ending
        );
        assert_eq!(outcome.attempt.failure, Some(FailureReason::Looped));
        let fired = reactions(&dir);
        assert!(fired.contains(&("d1".to_string(), "l1".to_string())), "{fired:?}");
        assert!(fired.contains(&("d1".to_string(), "l2".to_string())), "{fired:?}");
        // Eight identical calls: L1 at the third, L2 at the fifth, L3 at the eighth. The attempt
        // ends there instead of running to the step budget.
        assert_eq!(outcome.steps, 8, "{:?}", outcome.ending);
    }

    #[test]
    fn one_reasoning_loop_is_recorded_once() {
        // The live run found this: `step` recorded D4 and `inject_l2` recorded it again, so one
        // firing looked like two. The metric exists to be trusted, and a doubled count would make
        // D4 look twice as common as it is.
        let repeated = "the parser is fine ".repeat(60);
        let model = Scripted::new(vec![Ok(Reply::text(repeated)), Ok(finish())]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Echo)]));
        runner.detectors.reasoning_repeat_ngram = 4;
        runner.detectors.reasoning_repeat_count = 3;
        let mut journal = journal(&dir);
        runner.run(&subtask(), 1, "", &mut journal).expect("run");

        let fired = reactions(&dir)
            .into_iter()
            .filter(|(detector, _)| detector == "d4")
            .count();
        assert_eq!(fired, 1, "{:?}", reactions(&dir));
    }

    #[test]
    fn a_normal_run_does_not_fire_a_detector() {
        // The failure that matters more than a missed loop: a detector that fires on ordinary
        // work teaches everyone to ignore the metric.
        let model = Scripted::new(vec![
            Ok(echo_reply("call_1", "a")),
            Ok(echo_reply("call_2", "b")),
            Ok(echo_reply("call_3", "c")),
            Ok(finish()),
        ]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Echo)]));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert!(outcome.claimed_done(), "{:?}", outcome.ending);
        assert_eq!(reactions(&dir), Vec::<(String, String)>::new());
    }

    #[test]
    fn two_different_calls_with_identical_output_are_a_semantic_repeat() {
        // `Echo` prints its argument, so two different arguments print different things. The
        // detector's signal is the opposite: the model changes the call and the answer is the
        // same, which is what a `cat a` / `cat ./a` pair looks like.
        let model = Scripted::new(vec![
            Ok(Reply::call("call_1", "constant", r#"{"path":"a"}"#)),
            Ok(Reply::call("call_2", "constant", r#"{"path":"b"}"#)),
            Ok(finish()),
        ]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Constant)]));
        let mut journal = journal(&dir);
        runner.run(&subtask(), 1, "", &mut journal).expect("run");

        let fired = reactions(&dir);
        assert!(fired.contains(&("d2".to_string(), "l1".to_string())), "{fired:?}");
    }

    #[test]
    fn a_nudge_reaches_the_model_and_the_prefix_does_not_move() {
        // L1 is only useful if the model actually reads it, and it must not cost a cold prefill
        // by being put in the stable prefix.
        let model = Scripted::new(vec![
            Ok(echo_reply("call_1", "same")),
            Ok(echo_reply("call_2", "same")),
            Ok(echo_reply("call_3", "same")),
            Ok(finish()),
        ]);
        let (dir, runner) = runner(model.clone(), registry(vec![Arc::new(Echo)]));
        let mut journal = journal(&dir);
        runner.run(&subtask(), 1, "", &mut journal).expect("run");

        let requests = model.requests();
        let prefixes: Vec<&str> = requests
            .iter()
            .map(|request| request.messages[0].content.as_deref().unwrap_or(""))
            .collect();
        // The whole point of appending: every step's prefix is the same bytes.
        assert!(
            prefixes.windows(2).all(|pair| pair.first() == pair.last()),
            "{prefixes:?}"
        );
        let nudged = requests.iter().any(|request| {
            request.messages.iter().any(|message| {
                message.role == "user"
                    && message
                        .content
                        .as_deref()
                        .is_some_and(|text| text.contains("repeating"))
            })
        });
        assert!(nudged, "the model was never told it was repeating");
    }

    #[test]
    fn a_stalled_model_is_forced_to_state_a_strategy_and_then_restarted() {
        // D3: 15 unproductive steps → L2, 25 → L3. The thresholds are lowered here so the test
        // does not have to script twenty-five turns to prove a ladder that core already asserts
        // in detail.
        let replies = (0..40)
            .map(|index| Ok(Reply::text(format!("thinking {index}"))))
            .collect();
        let (dir, mut runner) = runner(Scripted::new(replies), registry(vec![Arc::new(Echo)]));
        runner.detectors.progress_stall_warn = 4;
        runner.detectors.progress_stall_stop = 5;
        runner.limits.stall_steps = 100;
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert!(
            matches!(outcome.ending, Ending::Looped(_)),
            "{:?}",
            outcome.ending
        );
        let fired = reactions(&dir);
        assert!(fired.contains(&("d3".to_string(), "l2".to_string())), "{fired:?}");
        assert_eq!(outcome.steps, 5, "{:?}", outcome.ending);
    }

    #[test]
    fn a_reasoning_loop_is_cut_mid_stream_and_the_attempt_continues() {
        // D4 fires inside one message: the stream is aborted before the model finishes, and the
        // attempt goes on with a forced strategy rather than ending. Without the abort this test
        // would need a message long enough to reach the step budget, which is the cost D4 exists
        // to avoid.
        let repeated = "the parser is fine ".repeat(60);
        let model = Scripted::new(vec![
            Ok(Reply::text(repeated)),
            Ok(echo_reply("call_1", "a")),
            Ok(finish()),
        ]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Echo)]));
        runner.detectors.reasoning_repeat_ngram = 4;
        runner.detectors.reasoning_repeat_count = 3;
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert!(outcome.claimed_done(), "{:?}", outcome.ending);
        let fired = reactions(&dir);
        assert!(fired.contains(&("d4".to_string(), "l2".to_string())), "{fired:?}");
        let aborted = {
            let reader = immortality_core::journal::JournalReader::open(
                dir.path().join("journal"),
                &TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAJ"),
            )
            .expect("reader");
            reader
                .read()
                .expect("records")
                .records
                .into_iter()
                .filter(|record| record.fields.kind == Kind::ModelStreamAborted)
                .count()
        };
        assert_eq!(aborted, 1);
    }

    #[test]
    fn the_reasoning_loop_is_cut_before_the_message_finishes() {
        // Aborting is the point. A detector that waits for the whole message and then reports a
        // loop saves nothing — the cost D4 exists to avoid has already been paid. The journal's
        // `chars_seen` is the observable proof: it must be a fraction of the scripted 1 200
        // characters, not all of them.
        let repeated = "the parser is fine ".repeat(60);
        assert!(repeated.chars().count() > 1_000);
        let model = Scripted::new(vec![Ok(Reply::text(repeated))]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Echo)]));
        runner.detectors.reasoning_repeat_ngram = 4;
        runner.detectors.reasoning_repeat_count = 3;
        runner.limits.max_steps = 1;
        let mut journal = journal(&dir);
        runner.run(&subtask(), 1, "", &mut journal).expect("run");

        let reader = immortality_core::journal::JournalReader::open(
            dir.path().join("journal"),
            &TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAJ"),
        )
        .expect("reader");
        let seen: Vec<u64> = reader
            .read()
            .expect("records")
            .records
            .into_iter()
            .filter(|record| record.fields.kind == Kind::ModelStreamAborted)
            .filter_map(|record| record.event.payload.get("chars_seen")?.as_u64())
            .collect();
        assert_eq!(seen.len(), 1, "one abort");
        assert!(
            seen[0] < 300,
            "the stream was not cut early: {} characters arrived",
            seen[0]
        );
    }

    #[test]
    fn a_reasoning_loop_that_repeats_forever_still_ends_at_the_step_budget() {
        // D4 alone does not stop a model that loops in every message; it counts as an idle step,
        // so D3 and the step budget still get their turn.
        let repeated = "the parser is fine ".repeat(60);
        let replies = (0..40).map(|_| Ok(Reply::text(repeated.clone()))).collect();
        let (dir, mut runner) = runner(Scripted::new(replies), registry(vec![Arc::new(Echo)]));
        runner.detectors.reasoning_repeat_ngram = 4;
        runner.detectors.reasoning_repeat_count = 3;
        runner.limits.max_steps = 6;
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert!(matches!(outcome.ending, Ending::Steps), "{:?}", outcome.ending);
        assert_eq!(outcome.steps, 6);
    }

    #[test]
    fn a_step_budget_ending_is_recorded_as_d5() {
        // The metric query "how often did the budget stop a run" has to be answerable from the
        // journal, not from the attempt counters alone.
        let replies = (0..20)
            .map(|index| Ok(echo_reply(&format!("call_{index}"), &format!("p{index}"))))
            .collect();
        let (dir, mut runner) = runner(Scripted::new(replies), registry(vec![Arc::new(Echo)]));
        runner.limits.max_steps = 2;
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert_eq!(outcome.ending, Ending::Steps);
        assert!(
            reactions(&dir).contains(&("d5".to_string(), "l3".to_string())),
            "{:?}",
            reactions(&dir)
        );
    }

    /// The distinct `(detector, level)` pairs one scripted run journaled.
    ///
    /// The registry is a parameter because the scenarios need different tools: D2 is about two
    /// distinct calls sharing one output, so it needs a tool that ignores its arguments.
    fn fired_by(
        model: Arc<dyn Model>,
        tools: Registry,
        tune: impl FnOnce(&mut Runner),
    ) -> Vec<(String, String)> {
        let (dir, mut runner) = runner(model, tools);
        tune(&mut runner);
        let mut journal = journal(&dir);
        runner.run(&subtask(), 1, "", &mut journal).expect("run");
        let mut pairs = reactions(&dir);
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    #[test]
    fn every_detector_reaches_the_journal_with_a_level() {
        // The v0.1 criterion is "every detector firing is journaled with its type **and** level".
        // Each detector has a test of its own above, and that is what let this gap survive: every
        // test asserted about the detector it was written for, and nothing asserted that the five
        // together were all there. A detector that stopped being reachable — a threshold raised
        // past the budget, a rung removed, a reaction replaced by a log line that only an operator
        // reads — would leave all five tests green and the metric short of a row.
        //
        // So the set is checked as a set. Every scenario is the smallest one that reaches its
        // detector, and the assertion is that all five names appear with a level the ladder
        // actually has.
        let repeated = "the parser is fine ".repeat(60);
        let scenarios = [
            (
                "d1",
                fired_by(
                    Scripted::new(
                        (0..20)
                            .map(|index| Ok(echo_reply(&format!("call_{index}"), "same")))
                            .collect(),
                    ),
                    registry(vec![Arc::new(Echo)]),
                    |_| {},
                ),
            ),
            (
                "d2",
                fired_by(
                    Scripted::new(vec![
                        Ok(Reply::call("call_1", "constant", r#"{"path":"a"}"#)),
                        Ok(Reply::call("call_2", "constant", r#"{"path":"b"}"#)),
                        Ok(finish()),
                    ]),
                    registry(vec![Arc::new(Constant)]),
                    |_| {},
                ),
            ),
            (
                "d3",
                fired_by(
                    Scripted::new(
                        (0..40)
                            .map(|index| Ok(Reply::text(format!("thinking {index}"))))
                            .collect(),
                    ),
                    registry(vec![Arc::new(Echo)]),
                    |runner| {
                        runner.detectors.progress_stall_warn = 4;
                        runner.detectors.progress_stall_stop = 5;
                        runner.limits.stall_steps = 100;
                    },
                ),
            ),
            (
                "d4",
                fired_by(
                    Scripted::new(vec![
                        Ok(Reply::text(repeated)),
                        Ok(echo_reply("call_1", "a")),
                        Ok(finish()),
                    ]),
                    registry(vec![Arc::new(Echo)]),
                    |runner| {
                        runner.detectors.reasoning_repeat_ngram = 4;
                        runner.detectors.reasoning_repeat_count = 3;
                    },
                ),
            ),
            (
                "d5",
                fired_by(
                    Scripted::new(
                        (0..20)
                            .map(|index| Ok(echo_reply(&format!("call_{index}"), &format!("p{index}"))))
                            .collect(),
                    ),
                    registry(vec![Arc::new(Echo)]),
                    |runner| runner.limits.max_steps = 2,
                ),
            ),
        ];

        let mut journaled: Vec<String> = Vec::new();
        for (detector, pairs) in scenarios {
            assert!(
                pairs.iter().any(|(found, _)| found == detector),
                "`{detector}` never reached the journal, so the metric would report it as never \
                 having fired: {pairs:?}"
            );
            for (_, level) in pairs {
                assert!(
                    matches!(level.as_str(), "l1" | "l2" | "l3"),
                    "a detector record carries `{level}`, which is not a rung of the ladder"
                );
                journaled.push(level);
            }
        }
        // All three rungs are reached by the five scenarios, so "with a level" is not satisfied by
        // a field that is always the same value.
        for rung in ["l1", "l2", "l3"] {
            assert!(
                journaled.iter().any(|level| level == rung),
                "no scenario ever journaled `{rung}`: {journaled:?}"
            );
        }
    }

    #[test]
    fn a_detector_firing_reaches_the_metric_through_the_journal() {
        // The criterion is "every firing is journaled with its type and level", and the reason it
        // matters is `immortality metrics`: that command is how a reader of `docs/metrics-v0.1.md`
        // gets the number. But the writer and the reader are different modules — the runner builds
        // the payload, `core::metrics` picks keys out of it — and every test above checks only one
        // side. Rename `detector` to `kind` in the writer, or read `name` in the reader, and both
        // suites stay green while the report shows nothing.
        //
        // Worse, the reader is deliberately forgiving: a missing key becomes `unknown`. So the
        // breakage would not be an error — it would be a plausible-looking row that says a
        // detector fired without saying which, or no row at all. This test drives a real run, feeds
        // the journal it wrote to the production reader, and requires the loop to be counted.
        let replies = (0..20)
            .map(|index| Ok(echo_reply(&format!("call_{index}"), "same")))
            .collect();
        let (dir, runner) = runner(Scripted::new(replies), registry(vec![Arc::new(Echo)]));
        let mut journal = journal(&dir);
        runner.run(&subtask(), 1, "", &mut journal).expect("run");

        let metrics =
            immortality_core::metrics::compute("01ARZ3NDEKTSV4RRFFQ69G5FAJ", &journal_records(&dir));
        assert!(
            metrics.detectors.get("d1/l1").is_some_and(|count| *count > 0),
            "the run journaled D1 but the metric does not count it; it saw {:?}. A reader of \
             `metrics-v0.1.md` would report this run as having no detector noise at all",
            metrics.detectors
        );
        assert!(
            !metrics.detectors.contains_key("unknown/unknown"),
            "a record reached the metric without a detector or without a level: {:?}",
            metrics.detectors
        );
    }

    #[test]
    fn every_detector_record_carries_the_attempt_scope() {
        // Same invariant as every other record: a `detector_fired` without a scope belongs to no
        // attempt, and the metric would be attributed to the task rather than to the run.
        let replies = (0..20)
            .map(|index| Ok(echo_reply(&format!("call_{index}"), "same")))
            .collect();
        let (dir, runner) = runner(Scripted::new(replies), registry(vec![Arc::new(Echo)]));
        let subtask = subtask();
        let mut journal = journal(&dir);
        runner.run(&subtask, 3, "again", &mut journal).expect("run");

        let reader = immortality_core::journal::JournalReader::open(
            dir.path().join("journal"),
            &TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAJ"),
        )
        .expect("reader");
        let fired: Vec<_> = reader
            .read()
            .expect("records")
            .records
            .into_iter()
            .filter(|record| record.fields.kind == Kind::DetectorFired)
            .collect();
        assert_ne!(fired.len(), 0);
        for record in &fired {
            assert_eq!(record.fields.subtask.as_ref(), Some(&subtask.id));
            assert_eq!(record.fields.attempt, Some(3));
        }
    }

    // ── M4.7: the session, restarts, and the strategy they carry ──────────────────────────

    /// A checkpoint world that records what happened instead of running git.
    ///
    /// The point of the fake is that the *decisions* of a restart — what is snapshotted, what is
    /// journalled, in which order — are testable without a repository. The real git path has its
    /// own tests in `checkpoints.rs`, so a bug in `git reset` cannot hide behind this.
    #[derive(Debug, Default)]
    struct FakeCheckpoints {
        /// Shas handed out, in order.
        taken: Mutex<Vec<String>>,
        /// Shas the tree was moved to, in order.
        restored: Mutex<Vec<String>>,
        /// Every call, in the order it was made, as `snapshot:<n>` / `restore:<sha>`.
        calls: Mutex<Vec<String>>,
        /// When set, `restore` fails with this reason.
        refuse: Mutex<Option<String>>,
        /// When true, `snapshot` reports a clean tree (no SHA to record).
        clean: Mutex<bool>,
        /// Journal directory to read at `restore` time, when the test is about durability.
        ///
        /// The ordering rule is "journal, then move", and the only way to test it is to look at
        /// what is *on disk* at the instant the move happens. Reading the in-memory `Journal`
        /// would be blind to the `fsync` the rule is actually about.
        watching: Mutex<Option<std::path::PathBuf>>,
        /// Whether the rollback marker was already on disk when `restore` was called.
        marker_seen: Mutex<Option<bool>>,
    }

    impl FakeCheckpoints {
        fn new() -> Arc<Self> {
            Arc::new(Self::default())
        }

        fn calls(&self) -> Vec<String> {
            self.calls.lock().map(|calls| calls.clone()).unwrap_or_default()
        }

        fn restored(&self) -> Vec<String> {
            self.restored.lock().map(|shas| shas.clone()).unwrap_or_default()
        }

        fn refusing(reason: &str) -> Arc<Self> {
            let fake = Self::new();
            if let Ok(mut refuse) = fake.refuse.lock() {
                *refuse = Some(reason.to_string());
            }
            fake
        }

        fn watching(journal_dir: impl Into<std::path::PathBuf>) -> Arc<Self> {
            let fake = Self::new();
            if let Ok(mut watching) = fake.watching.lock() {
                *watching = Some(journal_dir.into());
            }
            fake
        }

        fn marker_seen(&self) -> Option<bool> {
            self.marker_seen.lock().ok().and_then(|seen| *seen)
        }

        fn snapshot_count(&self) -> usize {
            self.calls.lock().map_or(0, |calls| {
                calls.iter().filter(|call| call.starts_with("snapshot:")).count()
            })
        }
    }

    impl Checkpoints for FakeCheckpoints {
        fn snapshot(&self, reason: &str) -> Result<Option<restart::Snapshot>, String> {
            let clean = self.clean.lock().is_ok_and(|clean| *clean);
            if let Ok(mut calls) = self.calls.lock() {
                calls.push(format!("snapshot:{reason}"));
            }
            if clean {
                return Ok(None);
            }
            let sha = self.taken.lock().map_or(0, |taken| taken.len());
            let sha = format!("ckpt-{sha}");
            if let Ok(mut taken) = self.taken.lock() {
                taken.push(sha.clone());
            }
            Ok(Some(restart::Snapshot {
                sha,
                branch: "agent-ckpt/tsk".to_string(),
            }))
        }

        fn restore(&self, sha: &str) -> Result<(), String> {
            if let Ok(mut calls) = self.calls.lock() {
                calls.push(format!("restore:{sha}"));
            }
            // Snapshot the journal as it is right now, before anything else changes.
            if let Ok(watching) = self.watching.lock()
                && let Some(dir) = watching.as_ref()
            {
                let seen = std::fs::read_dir(dir).is_ok_and(|entries| {
                    entries.filter_map(Result::ok).any(|entry| {
                        std::fs::read_to_string(entry.path()).is_ok_and(|text| {
                            text.contains("\"role\":\"rollback\"") || text.contains("\"role\": \"rollback\"")
                        })
                    })
                });
                if let Ok(mut marker) = self.marker_seen.lock() {
                    *marker = Some(seen);
                }
            }
            if let Ok(refuse) = self.refuse.lock()
                && let Some(reason) = refuse.as_ref()
            {
                return Err(reason.clone());
            }
            if let Ok(mut restored) = self.restored.lock() {
                restored.push(sha.to_string());
            }
            Ok(())
        }
    }

    fn with_checkpoints(mut runner: Runner, fake: Arc<FakeCheckpoints>) -> Runner {
        runner.checkpoints = Some(fake);
        runner
    }

    fn allowance(attempts: u32, restarts: u32) -> Allowance {
        Allowance {
            attempts_left: attempts,
            restarts_left: restarts,
        }
    }

    /// A model that spends its whole step budget without claiming anything.
    fn looping_replies(count: usize) -> Vec<Result<Reply, String>> {
        (0..count)
            .map(|index| Ok(Reply::call(format!("call_{index}"), "touch", r#"{"path":"a"}"#)))
            .collect()
    }

    #[test]
    fn a_restart_rolls_the_tree_back_and_journals_both_checkpoints() {
        // The milestone's core claim: rollback plus retry happens, and what was thrown away has a
        // name in the journal *before* the tree moves. The call order is the assertion — a
        // snapshot after the restore would name a tree that no longer exists.
        let checkpoints = FakeCheckpoints::new();
        let mut replies = looping_replies(2);
        replies.push(Ok(finish()));
        let (dir, mut runner) = runner(Scripted::new(replies), registry(vec![Arc::new(Touch)]));
        runner.limits.max_steps = 2;
        runner = with_checkpoints(runner, Arc::clone(&checkpoints));
        let mut journal = journal(&dir);
        let end = runner
            .session(&subtask(), 1)
            .drive(allowance(3, 1), &mut journal)
            .expect("session");

        assert_eq!(checkpoints.restored(), vec!["ckpt-0".to_string()]);
        // The base is taken *before* the first attempt, then the discarded work, then the move:
        // no snapshot after the restore. The base coming first is the whole point — taken inside
        // the rollback it would contain the failed attempt's own work, and the "restore" would
        // put the branch back onto exactly what it was meant to throw away.
        let calls = checkpoints.calls();
        assert_eq!(calls.len(), 3, "{calls:?}");
        assert!(calls[0].starts_with("snapshot:base for attempt 1"), "{calls:?}");
        assert!(calls[1].starts_with("snapshot:L3 restart"), "{calls:?}");
        assert_eq!(calls[2], "restore:ckpt-0");
        // And the second attempt inherited the base it was rolled back to.
        let outcome = end.outcome;
        assert_eq!(outcome.attempt.number, 2, "{:?}", outcome.attempt);
        assert!(end.stop == Stop::Done || matches!(end.stop, Stop::Done));
    }

    #[test]
    fn the_restart_record_names_the_closed_attempt_and_its_replacement() {
        // D-027's convention: `attempt` is the attempt being closed, the replacement is in the
        // payload. Recovery keys attempts by that field, so a swapped pair would close the wrong
        // one and leave the real attempt running forever.
        let checkpoints = FakeCheckpoints::new();
        let mut replies = looping_replies(2);
        replies.push(Ok(finish()));
        let (dir, mut runner) = runner(Scripted::new(replies), registry(vec![Arc::new(Touch)]));
        runner.limits.max_steps = 2;
        runner = with_checkpoints(runner, Arc::clone(&checkpoints));
        let mut journal = journal(&dir);
        runner
            .session(&subtask(), 1)
            .drive(allowance(3, 1), &mut journal)
            .expect("session");

        let restart = journal_records(&dir)
            .into_iter()
            .find(|record| record.fields.kind == Kind::AttemptRestarted)
            .expect("a restart record");
        assert_eq!(restart.fields.attempt, Some(1), "the attempt being closed");
        assert_eq!(restart.event.payload["next"], serde_json::json!(2));
        assert_eq!(restart.event.payload["cause"], serde_json::json!("steps"));
        assert!(
            restart.event.payload["why"]
                .as_str()
                .is_some_and(|why| !why.is_empty())
        );
    }

    #[test]
    fn the_checkpoint_records_name_the_base_and_the_discarded_work() {
        // A human told "your work was rolled back" must be able to find it. The two records carry
        // different roles for exactly that reason.
        let checkpoints = FakeCheckpoints::new();
        let mut replies = looping_replies(2);
        replies.push(Ok(finish()));
        let (dir, mut runner) = runner(Scripted::new(replies), registry(vec![Arc::new(Touch)]));
        runner.limits.max_steps = 2;
        runner = with_checkpoints(runner, Arc::clone(&checkpoints));
        let mut journal = journal(&dir);
        runner
            .session(&subtask(), 1)
            .drive(allowance(3, 1), &mut journal)
            .expect("session");

        let roles: Vec<String> = journal_records(&dir)
            .into_iter()
            .filter(|record| record.fields.kind == Kind::CheckpointCreated)
            .filter_map(|record| record.event.payload["role"].as_str().map(str::to_string))
            .collect();
        // The third record is the marker written *before* the move, which is what makes a crash
        // between the move and the `attempt_restarted` record recoverable.
        assert_eq!(
            roles,
            vec![
                "base".to_string(),
                "discarded".to_string(),
                "rollback".to_string(),
            ],
            "{roles:?}"
        );
        let shas: Vec<String> = journal_records(&dir)
            .into_iter()
            .filter(|record| record.fields.kind == Kind::CheckpointCreated)
            .filter_map(|record| record.event.payload["sha"].as_str().map(str::to_string))
            .collect();
        assert_eq!(
            shas,
            vec!["ckpt-0".to_string(), "ckpt-1".to_string(), "ckpt-0".to_string(),],
            "the marker repeats the base it is about to move to"
        );
    }

    #[test]
    fn the_base_checkpoint_is_taken_before_the_first_attempt_touches_the_tree() {
        // The bug a live run exposed: the base was taken *inside* the rollback, so it snapshotted
        // a tree that already held the failed attempt's work, journalled that as "base", and reset
        // the branch onto it. The rollback then discarded nothing, and the attempt's own work
        // became the commit the reviewer diffed against — an empty diff, and a rejected subtask
        // whose checks were green.
        //
        // The assertion is on *ordering*, not on the SHAs: the fake hands out a fresh SHA per
        // snapshot, so a late base looks identical in the journal payload. What differs is when it
        // happens, and the first model call is the first thing that touches the tree.
        #[derive(Debug)]
        struct Watching {
            dir: Arc<Mutex<Option<std::path::PathBuf>>>,
            base_at_first_step: Arc<Mutex<Option<bool>>>,
            replies: Mutex<VecDeque<Result<Reply, String>>>,
        }

        impl Model for Watching {
            fn complete(&self, _request: &ChatRequest) -> Result<Reply, String> {
                let seen = self
                    .dir
                    .lock()
                    .ok()
                    .and_then(|dir| dir.clone())
                    .is_some_and(|dir| {
                        immortality_core::journal::JournalReader::open(
                            dir,
                            &TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAJ"),
                        )
                        .and_then(|reader| reader.read())
                        .is_ok_and(|outcome| {
                            outcome.records.iter().any(|record| {
                                record.fields.kind == Kind::CheckpointCreated
                                    && record.event.payload["role"] == serde_json::json!("base")
                            })
                        })
                    });
                if let Ok(mut slot) = self.base_at_first_step.lock()
                    && slot.is_none()
                {
                    *slot = Some(seen);
                }
                self.replies
                    .lock()
                    .ok()
                    .and_then(|mut queue| queue.pop_front())
                    .unwrap_or_else(|| Err("the script has no reply for this step".to_string()))
            }
        }

        let checkpoints = FakeCheckpoints::new();
        let mut replies = looping_replies(2);
        replies.push(Ok(finish()));
        let dir_slot: Arc<Mutex<Option<std::path::PathBuf>>> = Arc::new(Mutex::new(None));
        let base_at_first_step = Arc::new(Mutex::new(None));
        let watcher = Arc::new(Watching {
            dir: Arc::clone(&dir_slot),
            base_at_first_step: Arc::clone(&base_at_first_step),
            replies: Mutex::new(replies.into()),
        });
        let (dir, mut runner) = runner(watcher, registry(vec![Arc::new(Touch)]));
        if let Ok(mut slot) = dir_slot.lock() {
            *slot = Some(dir.path().join("journal"));
        }
        runner.limits.max_steps = 2;
        runner = with_checkpoints(runner, Arc::clone(&checkpoints));
        let mut journal = journal(&dir);
        runner
            .session(&subtask(), 1)
            .drive(allowance(3, 1), &mut journal)
            .expect("session");

        assert_eq!(
            base_at_first_step.lock().ok().and_then(|slot| *slot),
            Some(true),
            "the base record must be in the journal before the first model step"
        );
    }

    #[test]
    fn a_restarted_attempt_gets_a_strategy_that_forbids_the_previous_one() {
        // The whole reason a restart is worth a rollback: the next brief is not the same brief.
        // The assertion reads the brief the model actually received, not the function's output.
        let checkpoints = FakeCheckpoints::new();
        let mut replies = looping_replies(2);
        replies.push(Ok(finish()));
        let model = Scripted::new(replies);
        let (dir, mut runner) = runner(
            Arc::clone(&model) as Arc<dyn Model>,
            registry(vec![Arc::new(Touch)]),
        );
        runner.limits.max_steps = 2;
        runner = with_checkpoints(runner, Arc::clone(&checkpoints));
        let mut journal = journal(&dir);
        runner
            .session(&subtask(), 1)
            .drive(allowance(3, 1), &mut journal)
            .expect("session");

        let requests = model.requests();
        let second_attempt = requests.last().expect("a second attempt ran");
        let system = second_attempt.messages[0].content.as_deref().unwrap_or("");
        assert!(system.contains("## Strategy for this attempt"), "{system}");
        assert!(system.contains("fresh attempt"), "{system}");
        assert!(system.contains("ran out of steps"), "{system}");
        // The first attempt had no strategy section at all.
        let first = requests[1].messages[0].content.as_deref().unwrap_or("");
        assert!(!first.contains("## Strategy for this attempt"), "{first}");
    }

    #[test]
    fn a_rejected_claim_retries_without_rolling_the_tree_back() {
        // The gate read the diff and named what is wrong. Rolling it away throws away the closest
        // thing to a finished subtask; the retry must keep the tree.
        let checkpoints = FakeCheckpoints::new();
        let model = Scripted::new(vec![Ok(finish()), Ok(finish()), Ok(finish())]);
        let (dir, mut runner) = runner(model, registry(vec![Arc::new(Touch)]));
        runner = with_checkpoints(runner, Arc::clone(&checkpoints));
        runner.gates = Some(Arc::new(FakeGates::default()));
        runner.reviewer = Some(Arc::new(FakeReviewer::rejecting("the error branch is untested")));
        let mut journal = journal(&dir);
        let end = runner
            .session(&subtask(), 1)
            .drive(allowance(2, 2), &mut journal)
            .expect("session");

        assert_eq!(end.outcome.attempt.number, 2, "{:?}", end.outcome.attempt);
        assert_eq!(
            checkpoints.restored(),
            Vec::<String>::new(),
            "a rejected claim must not roll back"
        );
        // The second attempt's brief quotes the gate's own words.
        assert_eq!(end.outcome.attempt.failure, Some(FailureReason::ReviewRejected));
    }

    #[test]
    fn a_restart_that_cannot_move_the_tree_stops_instead_of_lying() {
        // A git failure after the attempt was closed cannot be papered over: the next attempt
        // would build on a tree nobody put back.
        let checkpoints = FakeCheckpoints::refusing("the tree is locked by another process");
        let mut replies = looping_replies(2);
        replies.push(Ok(finish()));
        let (dir, mut runner) = runner(Scripted::new(replies), registry(vec![Arc::new(Touch)]));
        runner.limits.max_steps = 2;
        runner = with_checkpoints(runner, Arc::clone(&checkpoints));
        let mut journal = journal(&dir);
        let end = runner
            .session(&subtask(), 1)
            .drive(allowance(3, 2), &mut journal)
            .expect("session");

        let Stop::Blocked(reason) = &end.stop else {
            panic!("expected a block, got {:?}", end.stop);
        };
        assert!(reason.contains("locked"), "{reason}");
        // The failed attempt is the last one that ran; nothing was started on top of it.
        assert_eq!(end.outcome.attempt.number, 1);
    }

    #[test]
    fn the_restart_is_journalled_before_the_replacement_attempt_starts() {
        // Recovery reads this order: the marker, then the confirmation, then the next attempt. A
        // crash between any two of these is a case the journal has to be able to describe, and an
        // `attempt_restarted` written after the new attempt started would leave a window in which
        // two attempts look live at once.
        let checkpoints = FakeCheckpoints::new();
        let mut replies = looping_replies(2);
        replies.push(Ok(finish()));
        let (dir, mut runner) = runner(Scripted::new(replies), registry(vec![Arc::new(Touch)]));
        runner.limits.max_steps = 2;
        runner = with_checkpoints(runner, Arc::clone(&checkpoints));
        let mut journal = journal(&dir);
        runner
            .session(&subtask(), 1)
            .drive(allowance(3, 1), &mut journal)
            .expect("session");

        let kinds: Vec<&'static str> = journal_records(&dir)
            .into_iter()
            .map(|record| record.fields.kind.as_str())
            .collect();
        let marker = kinds
            .iter()
            .position(|kind| *kind == "checkpoint_created")
            .expect("a checkpoint");
        let restarted = kinds
            .iter()
            .position(|kind| *kind == "attempt_restarted")
            .expect("a restart");
        let second = kinds
            .iter()
            .rposition(|kind| *kind == "attempt_started")
            .expect("the second attempt");
        assert!(marker < restarted, "{kinds:?}");
        assert!(restarted < second, "{kinds:?}");
    }

    #[test]
    fn a_second_restart_does_not_quote_the_generated_strategy_back_at_the_model() {
        // The seed is quoted, never the generated text. Quoting the generated text nests the same
        // advice inside itself once per attempt, and after three restarts the brief is mostly its
        // own previous advice — the exact token waste the restart was meant to avoid.
        let checkpoints = FakeCheckpoints::new();
        let model = Scripted::new(looping_replies(3));
        let (dir, mut runner) = runner(
            Arc::clone(&model) as Arc<dyn Model>,
            registry(vec![Arc::new(Touch)]),
        );
        runner.limits.max_steps = 1;
        runner = with_checkpoints(runner, Arc::clone(&checkpoints));
        let mut journal = journal(&dir);
        runner
            .session(&subtask(), 1)
            .with_strategy("write the parser bottom-up")
            .drive(allowance(3, 3), &mut journal)
            .expect("session");

        let last = model.requests().last().expect("a third attempt").clone();
        let system = last.messages[0].content.as_deref().unwrap_or("");
        assert_eq!(
            system.matches("This is a fresh attempt").count(),
            1,
            "the generated strategy must not be nested: {system}"
        );
        assert_eq!(
            system.matches("write the parser bottom-up").count(),
            1,
            "the seed is quoted once: {system}"
        );
        // Both earlier attempts' failures are still carried: the accumulated list is the part
        // that changes between attempts.
        assert!(system.contains("attempt 1:"), "{system}");
        assert!(system.contains("attempt 2:"), "{system}");
    }

    #[test]
    fn the_rollback_marker_is_on_disk_before_the_tree_moves() {
        // The durability half of the ordering rule. A marker buffered in memory would leave a
        // crash between the reset and the flush with a moved tree and no record of where it came
        // from; the fake reads the journal directory at `restore` time to prove it is not.
        let mut replies = looping_replies(2);
        replies.push(Ok(finish()));
        let (dir, mut runner) = runner(Scripted::new(replies), registry(vec![Arc::new(Touch)]));
        runner.limits.max_steps = 2;
        let watcher = FakeCheckpoints::watching(dir.path().join("journal"));
        runner = with_checkpoints(runner, Arc::clone(&watcher));
        let mut journal = journal(&dir);
        runner
            .session(&subtask(), 1)
            .drive(allowance(3, 1), &mut journal)
            .expect("session");

        assert_eq!(
            watcher.marker_seen(),
            Some(true),
            "the rollback marker must be on disk before git is asked to move the tree"
        );
        let markers = journal_records(&dir)
            .into_iter()
            .filter(|record| record.fields.kind == Kind::CheckpointCreated)
            .filter(|record| record.event.payload["role"] == serde_json::json!("rollback"))
            .count();
        assert_eq!(markers, 1, "one marker per restart");
    }

    #[test]
    fn a_restart_without_a_checkpoint_world_stops_with_a_named_reason() {
        // `checkpoints: None` is "nothing can put the tree back". Claiming an L3 anyway would be
        // the exact bypass the seam exists to prevent.
        let mut replies = looping_replies(2);
        replies.push(Ok(finish()));
        let (dir, mut runner) = runner(Scripted::new(replies), registry(vec![Arc::new(Touch)]));
        runner.limits.max_steps = 2;
        let mut journal = journal(&dir);
        let end = runner
            .session(&subtask(), 1)
            .drive(allowance(3, 2), &mut journal)
            .expect("session");

        assert!(
            matches!(&end.stop, Stop::Blocked(reason) if reason.contains("checkpoint world")),
            "{:?}",
            end.stop
        );
    }

    #[test]
    fn the_attempt_budget_stops_the_session_and_burns_no_tokens_on_a_last_try() {
        // Three attempts exist, the fourth must not start: the session may not spend a step after
        // the budget is gone, and the stop must say the budget ran out.
        let checkpoints = FakeCheckpoints::new();
        let replies = looping_replies(10);
        let model = Scripted::new(replies);
        let (dir, mut runner) = runner(
            Arc::clone(&model) as Arc<dyn Model>,
            registry(vec![Arc::new(Touch)]),
        );
        runner.limits.max_steps = 1;
        runner = with_checkpoints(runner, Arc::clone(&checkpoints));
        let mut journal = journal(&dir);
        let end = runner
            .session(&subtask(), 1)
            .drive(allowance(2, 5), &mut journal)
            .expect("session");

        assert_eq!(end.outcome.attempt.number, 2);
        assert!(matches!(end.stop, Stop::Exhausted(_)), "{:?}", end.stop);
        // Two attempts ran, so the third step was never asked for.
        assert_eq!(model.requests().len(), 2, "{:?}", end.stop);
    }

    #[test]
    fn a_verified_claim_ends_the_session_without_a_restart() {
        let checkpoints = FakeCheckpoints::new();
        let (dir, mut runner) = runner(Scripted::new(vec![Ok(finish())]), registry(vec![Arc::new(Touch)]));
        runner = with_checkpoints(runner, Arc::clone(&checkpoints));
        let mut journal = journal(&dir);
        let end = runner
            .session(&subtask(), 1)
            .drive(allowance(3, 2), &mut journal)
            .expect("session");

        assert_eq!(end.stop, Stop::Done);
        assert!(end.outcome.verified(), "{:?}", end.outcome.gates);
        // One snapshot, and it is the base taken before the first attempt: a session that never
        // restarts never restores, but the base still has to exist for the first attempt to know
        // what its gates should diff against. It is not the "discarded work" snapshot, which only
        // a rollback takes.
        assert_eq!(checkpoints.snapshot_count(), 1);
        assert_eq!(checkpoints.restored(), Vec::<String>::new());
    }

    #[test]
    fn an_interrupted_attempt_ends_the_session_rather_than_spending_a_restart() {
        // A backend outage is D-027's case: the same attempt resumes later, with its conversation
        // replayed from the journal, and no restart budget is spent.
        let checkpoints = FakeCheckpoints::new();
        let (dir, mut runner) = runner(
            Scripted::new(vec![Err("connection refused".to_string())]),
            registry(vec![Arc::new(Touch)]),
        );
        runner = with_checkpoints(runner, Arc::clone(&checkpoints));
        let mut journal = journal(&dir);
        let end = runner
            .session(&subtask(), 1)
            .drive(allowance(3, 2), &mut journal)
            .expect("session");

        assert_eq!(end.stop, Stop::Abandoned);
        assert_eq!(end.outcome.attempt.state, AttemptState::Abandoned);
        assert_eq!(checkpoints.restored(), Vec::<String>::new());
    }

    #[test]
    fn a_restarted_attempt_records_the_checkpoint_it_was_rolled_back_to() {
        // The gate evaluation narrows its diff to this attempt's base, so the attempt has to
        // carry the SHA. Without it every restarted attempt is reviewed on the whole task.
        let checkpoints = FakeCheckpoints::new();
        let mut replies = looping_replies(2);
        replies.push(Ok(finish()));
        let (dir, mut runner) = runner(Scripted::new(replies), registry(vec![Arc::new(Touch)]));
        runner.limits.max_steps = 2;
        runner = with_checkpoints(runner, Arc::clone(&checkpoints));
        let mut journal = journal(&dir);
        let end = runner
            .session(&subtask(), 1)
            .drive(allowance(3, 1), &mut journal)
            .expect("session");

        assert_eq!(end.outcome.base_checkpoint.as_deref(), Some("ckpt-0"));
        assert_eq!(end.outcome.attempt.base_checkpoint.as_deref(), Some("ckpt-0"));
    }

    #[test]
    fn a_resumed_subtask_starts_from_the_report_it_inherited() {
        // A restart after a daemon crash must not make the model re-propose what was already
        // rejected: the report is the only channel between attempts.
        let checkpoints = FakeCheckpoints::new();
        let mut replies = looping_replies(2);
        replies.push(Ok(finish()));
        let model = Scripted::new(replies);
        let (dir, mut runner) = runner(
            Arc::clone(&model) as Arc<dyn Model>,
            registry(vec![Arc::new(Touch)]),
        );
        runner.limits.max_steps = 2;
        runner = with_checkpoints(runner, Arc::clone(&checkpoints));
        let mut journal = journal(&dir);
        let mut report = SubtaskReport::new("nothing yet");
        report
            .rejected_ideas
            .push("regenerating the parser from scratch".to_string());
        runner
            .session(&subtask(), 2)
            .resuming_from(report)
            .drive(allowance(3, 1), &mut journal)
            .expect("session");

        let requests = model.requests();
        let second = requests.last().expect("a second attempt");
        let system = second.messages[0].content.as_deref().unwrap_or("");
        assert!(
            system.contains("regenerating the parser from scratch"),
            "{system}"
        );
        assert!(system.contains("Do not propose them again"), "{system}");
    }
}
