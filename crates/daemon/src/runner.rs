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
use immortality_core::context::{
    Budget, Context, Entry, StableLayers, TokenCounter, ToolCall, Verdict, artifact_hint, truncate_output,
};
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
            started: Instant::now(),
        }
    }

    /// Run steps until something ends the attempt.
    fn drive(&mut self, journal: &mut Journal) -> immortality_core::Result<Ending> {
        loop {
            if let Some(ending) = self.exhausted() {
                return Ok(ending);
            }
            let reply = match self.runner.model.complete(&self.request()) {
                Ok(reply) => reply,
                Err(detail) => return Ok(Ending::Backend { detail }),
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
                self.last_text = reply.content;
                self.stalled = self.stalled.saturating_add(1);
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
            if self.stalled >= self.runner.limits.stall_steps {
                return Ok(Ending::Stalled);
            }
        }
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
}
