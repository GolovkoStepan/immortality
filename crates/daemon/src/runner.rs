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
//! is a *claim*, not a verdict — the gates decide, and they are M4.6. Until then the runner
//! records the claim and returns the report, which is the honest contract.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use immortality_core::artifacts::ArtifactStore;
use immortality_core::config::Detectors;
use immortality_core::context::{
    Budget, Context, Entry, StableLayers, TokenCounter, ToolCall, Verdict, artifact_hint, truncate_output,
};
use immortality_core::detect::{Detector, Level, Reaction, Signals, Stream, StreamVerdict, Thresholds};
use immortality_core::domain::{Attempt, AttemptTrigger, FailureReason, Subtask, SubtaskReport};
use immortality_core::ids::TaskId;
use immortality_core::journal::{Journal, Kind, Scope};
use immortality_core::protocol::{CallId, Intent, ToolCallLog};
use immortality_llm::turn::AssistantTurn;
use immortality_llm::wire::ChatRequest;
use immortality_llm::{Backend, ToolDef};
use immortality_tools::{Args, PolicyGate, Registry, ToolCtx, ToolError};

/// The name the model calls to claim the subtask is done.
///
/// Spelled here once and compared against a name that came from a model: a typo would turn
/// "done" into "unknown tool" and run the attempt to its step budget.
pub const FINISH_SUBTASK: &str = "finish_subtask";

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
}

impl RunOutcome {
    /// Whether the model claimed the subtask done.
    #[must_use]
    pub const fn claimed_done(&self) -> bool {
        matches!(self.ending, Ending::Claimed)
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
        let mut attempt = Attempt::start(
            subtask.id.clone(),
            attempt_number,
            strategy,
            immortality_core::time::now(),
        );
        journal.append_scoped(
            immortality_core::time::now(),
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
        let report = state.report(&ending);
        // A claim does not close the attempt. `AttemptState::Succeeded` means "the model claimed
        // done and every gate passed", and the gates are M4.6 — a runner that wrote `Succeeded`
        // here would be grading its own work, and `status` would show a success that no check
        // ever confirmed. The attempt stays `Running`, which is exactly true while the gates run,
        // and the scheduler closes it. A crash in that window is closed by recovery like any
        // other open attempt.
        if ending == Ending::Claimed {
            return Ok(RunOutcome {
                attempt,
                ending,
                report,
                steps: state.steps,
            });
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
            .finish(trigger, ending.failure_reason(), immortality_core::time::now())
            .map_err(|error| immortality_core::Error::Protocol(error.to_string()))?;

        journal.append_scoped(
            immortality_core::time::now(),
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
        })
    }
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
    started: Instant,
}

impl<'a> StepLoop<'a> {
    fn new(runner: &'a Runner, subtask: &'a Subtask, attempt: &Attempt) -> Self {
        let layers = StableLayers {
            system: runner.system_prompt.clone(),
            brief: brief_for(subtask, &attempt.strategy),
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
            stalled: 0,
            touched: Vec::new(),
            last_text: String::new(),
            claim_summary: None,
            signals: Signals::default(),
            thresholds: Thresholds::from_config(&runner.detectors),
            stopped_by: None,
            started: Instant::now(),
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
            self.tokens = self
                .tokens
                .saturating_add(reply.usage.map_or(0, |(prompt, completion)| {
                    u64::from(prompt) + u64::from(completion)
                }));
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

            let mut claimed = false;
            for call in &calls {
                if call.name == FINISH_SUBTASK {
                    claimed = true;
                    self.claim_summary = summary_of(call);
                    continue;
                }
                // A protocol failure ends the attempt, but it is still an `Ending` and not an
                // error: the run has to produce a report saying why.
                let step = match self.run_tool(journal, call) {
                    Ok(step) => step,
                    Err(ending) => return Ok(ending),
                };
                match step {
                    ToolStep::Ran => self.stalled = 0,
                    ToolStep::NotRun => self.stalled = self.stalled.saturating_add(1),
                }
            }
            if claimed {
                return Ok(Ending::Claimed);
            }
            if let Some(ending) = self.react(journal)? {
                return Ok(ending);
            }
            if self.stalled >= self.runner.limits.stall_steps {
                return Ok(Ending::Stalled);
            }
        }
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
                immortality_core::time::now(),
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
        journal.append_scoped(
            immortality_core::time::now(),
            Kind::DetectorFired,
            fields,
            &self.scope,
        )?;
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
        if self.started.elapsed() >= self.runner.limits.max_wall_time {
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
            immortality_core::time::now(),
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
                    immortality_core::time::now(),
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
            tool.description(),
        );
        log.intent(&intent, args.raw().clone())
            .map_err(|error| backend(&error))?;
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
                immortality_core::time::now(),
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
pub fn brief_for(subtask: &Subtask, strategy: &str) -> String {
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
    fn a_claim_leaves_the_attempt_running_because_the_gates_have_not_run() {
        // `Succeeded` means "claimed done and every gate passed". The runner has no gates, so
        // writing it here would report a success no check ever confirmed — and `status` would
        // show a finished attempt the scheduler never verified.
        let model = Scripted::new(vec![Ok(finish())]);
        let (dir, runner) = runner(model, registry(vec![Arc::new(Touch)]));
        let mut journal = journal(&dir);
        let outcome = runner.run(&subtask(), 1, "", &mut journal).expect("run");

        assert_eq!(outcome.attempt.state, AttemptState::Running);
        assert_eq!(outcome.attempt.failure, None);
        assert_eq!(outcome.attempt.finished_at, None);
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
    fn the_brief_names_the_goal_the_gate_and_the_finish_call() {
        let text = brief_for(&subtask(), "roll back and narrow the change");
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
        let text = brief_for(&subtask, "");
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
            .filter_map(|record| {
                Some((
                    record.event.payload.get("detector")?.as_str()?.to_string(),
                    record.event.payload.get("level")?.as_str()?.to_string(),
                ))
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
}
