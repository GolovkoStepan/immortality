//! Five detectors and the reaction ladder, as pure logic.
//!
//! # Why this is not inside the runner
//!
//! The runner is where a detector's verdict is *acted on*: a nudge is injected, a restart ends
//! the attempt. But the verdict itself is a function of the step history, and a function of the
//! history is testable in microseconds — no model, no journal, no clock. Putting both in the loop
//! would mean every threshold change needs a scripted model to test, and a threshold change is
//! exactly the kind of edit this project makes weekly.
//!
//! # Why the thresholds are separate from the signals
//!
//! [`Signals`] is what the loop measures and [`Ladder`] is what it concludes. The two are joined
//! by [`Ladder::react`], so a test can assert "three exact repeats is a nudge" without
//! constructing a run, and a test of the run can assert that the nudge reached the context.
//!
//! # Why D4 is separate from D1–D3
//!
//! D1–D3 look at *finished* steps: a call was made, an output came back, the tree changed.
//! D4 looks at a *stream in progress* and has to abort it before the model writes five thousand
//! tokens of the same paragraph. It is therefore a stateful scanner fed one delta at a time, not
//! a function over history, and it lives in [`Stream`].

use crate::config::Detectors;

mod normalize;
mod stream;

pub use normalize::normalize;
pub use stream::{Stream, StreamVerdict};

/// Which detector spoke.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Detector {
    /// D1: the same call, normalised, again.
    ExactRepeat,
    /// D2: different calls whose results are indistinguishable.
    SemanticRepeat,
    /// D3: steps passed without the tree, the gates or the facts changing.
    NoProgress,
    /// D4: the model is repeating a phrase inside one message.
    ReasoningLoop,
    /// D5: a budget is spent.
    Budget,
}

impl Detector {
    /// The wire name, for a `detector_fired` record.
    ///
    /// `d1`…`d5` rather than the prose name: the documentation, the config keys and the journal
    /// all refer to the detectors by number, and a metric query that has to translate
    /// `exact_repeat` back to `d1` is a query someone will get wrong.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExactRepeat => "d1",
            Self::SemanticRepeat => "d2",
            Self::NoProgress => "d3",
            Self::ReasoningLoop => "d4",
            Self::Budget => "d5",
        }
    }
}

/// How hard the reaction is.
///
/// L4 (split the subtask) and L5 (block it) are absent: both are decisions about the *task tree*,
/// which is the scheduler's to make in M4.5. A detector that returned `Split` would be claiming
/// authority it does not have, and the runner would have nothing to do with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Level {
    /// L1: inject a nudge and keep going in the same attempt.
    Nudge,
    /// L2: force a step that lists alternatives before another tool call.
    ForceStrategy,
    /// L3: the attempt is not worth continuing; start a new one with a different strategy.
    Restart,
}

impl Level {
    /// The wire name, for a `detector_fired` record.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nudge => "l1",
            Self::ForceStrategy => "l2",
            Self::Restart => "l3",
        }
    }

    /// The instruction this level adds to the model's context, if any.
    ///
    /// `Restart` returns `None`: it ends the attempt instead of talking to a model that has
    /// already stopped listening. The text for the other two is deliberately short and specific:
    /// a long lecture spends the tokens it was meant to save, and a generic "try harder" is
    /// indistinguishable from what the model is already telling itself.
    #[must_use]
    pub const fn instruction(self, detector: Detector) -> Option<&'static str> {
        match (self, detector) {
            (Self::Nudge, Detector::ExactRepeat) => Some(
                "You are repeating an identical tool call. Its result will not change. Read the \
                 output you already have and take a different step.",
            ),
            (Self::Nudge, Detector::SemanticRepeat) => Some(
                "Your last calls produced the same result. Running another variant of the same \
                 command will not tell you anything new.",
            ),
            (Self::ForceStrategy, _) => Some(
                "Stop calling tools for one step. Write down three genuinely different hypotheses \
                 about what is wrong, choose one, and say what evidence would confirm it. Then act \
                 on that instead of repeating the previous approach.",
            ),
            (Self::Nudge | Self::Restart, _) => None,
        }
    }
}

/// What one reaction did, for the journal and for a test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reaction {
    pub detector: Detector,
    pub level: Level,
    /// The measurement that tripped the threshold: a repeat count, a stretch of steps, a budget.
    /// Kept next to the verdict so `detector_fired` records a number and not only a name.
    pub metric: u32,
}

/// One detector's thresholds.
///
/// A zero `stop` means "never escalate past L2": D2 has no stop in the design, because an
/// identical output from two different commands is a hint, not proof that the attempt is spent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Ladder {
    warn: u32,
    strategy: u32,
    stop: u32,
}

impl Ladder {
    /// The level `count` reaches, if any.
    ///
    /// Checked from harshest to mildest so a count that leaps several thresholds at once — a
    /// detector that was not consulted for a few steps — reports the harshest level it has
    /// earned, not the first one that happens to be listed.
    const fn level(self, count: u32) -> Option<Level> {
        if self.stop != 0 && count >= self.stop {
            Some(Level::Restart)
        } else if self.strategy != 0 && count >= self.strategy {
            Some(Level::ForceStrategy)
        } else if count >= self.warn {
            Some(Level::Nudge)
        } else {
            None
        }
    }
}

/// The detector thresholds, in the form the ladder needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Thresholds {
    exact: Ladder,
    semantic: Ladder,
    progress: Ladder,
}

impl Thresholds {
    /// Read the thresholds from a validated config.
    #[must_use]
    pub const fn from_config(detectors: &Detectors) -> Self {
        Self {
            exact: Ladder {
                warn: detectors.exact_repeat_warn,
                strategy: detectors.exact_repeat_strategy,
                stop: detectors.exact_repeat_stop,
            },
            semantic: Ladder {
                warn: detectors.semantic_repeat_warn,
                strategy: detectors.semantic_repeat_strategy,
                stop: 0,
            },
            progress: Ladder {
                warn: u32::MAX,
                strategy: detectors.progress_stall_warn,
                stop: detectors.progress_stall_stop,
            },
        }
    }

    /// The level an exact-repeat count reaches.
    #[must_use]
    pub const fn exact(&self, count: u32) -> Option<Level> {
        self.exact.level(count)
    }

    /// The level a semantic-repeat count reaches.
    #[must_use]
    pub const fn semantic(&self, count: u32) -> Option<Level> {
        self.semantic.level(count)
    }

    /// The level a stretch of unproductive steps reaches.
    #[must_use]
    pub const fn progress(&self, steps: u32) -> Option<Level> {
        self.progress.level(steps)
    }
}

/// What the loop has observed so far in one attempt.
///
/// Fed one step at a time; every field is a counter, so a detector's answer changes only when a
/// step arrives and never as a function of how often it was asked.
#[derive(Debug, Default)]
pub struct Signals {
    /// Canonical `(tool, arguments)` of each call, in order.
    calls: Vec<(String, String)>,
    /// Canonical `(tool, arguments)` of the immediately preceding call, by call order.
    last_call: Option<(String, String)>,
    /// How many times in a row the last call has been sent.
    repeat_run: u32,
    /// The highest repeat run this attempt has seen, kept so a nudge is not re-sent every step:
    /// after a nudge the run continues, and the model already has the message.
    exact_announced: Option<Level>,
    /// The harshest semantic-repeat level already announced.
    semantic_announced: Option<Level>,
    /// The most recent output digest, so only a *consecutive* repeat counts.
    last_output: Option<String>,
    /// Consecutive semantic repeats of the same digest.
    semantic_run: u32,
    /// Steps since the last observable progress.
    unproductive: u32,
    /// The harshest unproductive-stretch level already announced.
    progress_announced: Option<Level>,
    /// Whether the model was just told to stop and think, so the forced step is not forced twice.
    strategy_forced: bool,
}

impl Signals {
    /// Record one tool call and the digest of what it produced.
    ///
    /// `output` is `None` when the tool did not run (refused, bad arguments): that is neither
    /// progress nor a repeat, and treating a refusal as a repeat would nudge a model that is
    /// already being told what was wrong.
    ///
    /// The two repeat signals are kept apart on purpose. An identical call is D1's business and
    /// says nothing about D2: the same command twice is expected to print the same thing, and
    /// counting that as a semantical repeat as well would report one stuck step as two problems.
    /// D2 counts what happens when the model *changes* the call and gets the same answer back.
    pub fn record_call(
        &mut self,
        tool: &str,
        arguments: &str,
        output: Option<&str>,
        worktree: Option<&std::path::Path>,
    ) {
        let canonical = (tool.to_string(), normalize(arguments, worktree));
        let same_call = self.last_call.as_ref() == Some(&canonical);
        if same_call {
            self.repeat_run = self.repeat_run.saturating_add(1);
        } else {
            self.repeat_run = 1;
        }
        self.last_call = Some(canonical.clone());
        self.calls.push(canonical);

        let Some(text) = output else {
            // Nothing ran, so there is no evidence either way about semantics.
            return;
        };
        let digest = crate::journal::digest_text(&crate::artifacts::digest_of(text.as_bytes()));
        if same_call {
            // The identical call is D1's; do not also feed D2.
            return;
        }
        if self.last_output.as_deref() == Some(digest.as_str()) {
            self.semantic_run = self.semantic_run.saturating_add(1);
        } else {
            self.semantic_run = 1;
            // An output this attempt has not seen is progress.
            self.unproductive = 0;
        }
        self.last_output = Some(digest);
    }

    /// Record a step that produced nothing that addressed the task: prose with no calls, or a
    /// call that did not run.
    ///
    /// The last call is deliberately kept. A model that alternates "call A, talk, call A, talk"
    /// is still repeating itself, and clearing the run here would make the exact-repeat detector
    /// blind to the most common shape of a stuck model.
    pub const fn record_idle_step(&mut self) {
        self.unproductive = self.unproductive.saturating_add(1);
    }

    /// Record a step that changed the working tree or the recorded facts.
    ///
    /// The hash itself is not stored: the loop only needs to know *that* it changed, and the
    /// caller has already paid for the hash.
    pub const fn record_progress(&mut self) {
        self.unproductive = 0;
    }

    /// How many times in a row the most recent call has been repeated.
    #[must_use]
    pub const fn repeat_run(&self) -> u32 {
        self.repeat_run
    }

    /// How many consecutive calls produced identical output.
    #[must_use]
    pub const fn semantic_run(&self) -> u32 {
        self.semantic_run
    }

    /// How many steps have passed without progress.
    #[must_use]
    pub const fn unproductive(&self) -> u32 {
        self.unproductive
    }

    /// How many distinct call shapes were made.
    #[must_use]
    pub const fn calls(&self) -> usize {
        self.calls.len()
    }

    /// What the ladder says now, or `None` when nothing has tripped.
    ///
    /// Announced levels are not repeated: after L1 on an exact repeat, the next identical call
    /// raises the count but returns `None` until L2 is earned. Without that, the model would get
    /// the same nudge on every step of the run, which spends context on the message that is
    /// already there.
    pub fn react(&mut self, thresholds: &Thresholds) -> Option<Reaction> {
        // Ordered by importance: an exact repeat is the cheapest and most certain signal, and
        // reporting it first means one record per step rather than three describing one step.
        let decision = raise(
            Detector::ExactRepeat,
            thresholds.exact(self.repeat_run),
            self.repeat_run,
            &mut self.exact_announced,
        );
        if decision.is_some() {
            return decision;
        }
        let decision = raise(
            Detector::SemanticRepeat,
            thresholds.semantic(self.semantic_run),
            self.semantic_run,
            &mut self.semantic_announced,
        );
        if decision.is_some() {
            return decision;
        }
        raise(
            Detector::NoProgress,
            thresholds.progress(self.unproductive),
            self.unproductive,
            &mut self.progress_announced,
        )
    }

    /// Mark that the forced strategy step has been delivered, so it is not repeated.
    pub const fn strategy_delivered(&mut self) {
        self.strategy_forced = true;
    }

    /// Whether the model has already been forced to state a strategy in this attempt.
    #[must_use]
    pub const fn strategy_was_forced(&self) -> bool {
        self.strategy_forced
    }
}

/// Report `level` unless a level at least as harsh was already announced for this detector.
///
/// A count can trip several thresholds at once when a detector was not consulted for a while;
/// [`Ladder::level`] already returns the harshest, so recording it is enough.
fn raise(
    detector: Detector,
    level: Option<Level>,
    metric: u32,
    announced: &mut Option<Level>,
) -> Option<Reaction> {
    let level = level?;
    if announced.is_some_and(|previous| level_rank(level) <= level_rank(previous)) {
        return None;
    }
    *announced = Some(level);
    Some(Reaction {
        detector,
        level,
        metric,
    })
}

/// Ordering of the levels, for "is this harsher than what was already announced".
const fn level_rank(level: Level) -> u8 {
    match level {
        Level::Nudge => 1,
        Level::ForceStrategy => 2,
        Level::Restart => 3,
    }
}

#[cfg(test)]
mod tests;
