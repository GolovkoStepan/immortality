//! Six detectors and the reaction ladder, as pure logic.
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
//! D1–D3 and D6 look at *finished* steps: a call was made, an output came back, the tree
//! changed.
//! D4 looks at a *stream in progress* and has to abort it before the model writes five thousand
//! tokens of the same paragraph. It is therefore a stateful scanner fed one delta at a time, not
//! a function over history, and it lives in [`Stream`].

use crate::config::Detectors;

mod normalize;
mod stream;

pub use normalize::normalize;
pub use stream::{Stream, StreamVerdict};

/// One path, in the form the repeat detectors compare.
///
/// The same rewriting [`normalize`] applies inside a JSON argument object: an absolute path under
/// the worktree becomes relative, so `./src/a.rs`, `src/a.rs` and `/work/tree/src/a.rs` are one
/// file. Exposed separately because D6 sees the path outside the argument JSON and must not
/// re-derive the rule — two implementations of "the same file" would disagree eventually.
#[must_use]
pub fn normalize_path(path: &str, worktree: &std::path::Path) -> String {
    normalize::strip_root(path, &worktree.to_string_lossy())
}

/// Which detector spoke.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Detector {
    /// D1: the same call, normalised, again.
    ExactRepeat,
    /// D2: different calls whose results are indistinguishable.
    SemanticRepeat,
    /// D3: steps produced no observation this attempt had not already seen.
    NoProgress,
    /// D4: the model is repeating a phrase inside one message.
    ReasoningLoop,
    /// D5: a budget is spent.
    Budget,
    /// D6: the same file is being rewritten over and over.
    FileChurn,
}

impl Detector {
    /// The wire name, for a `detector_fired` record.
    ///
    /// `d1`…`d6` rather than the prose name: the documentation, the config keys and the journal
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
            Self::FileChurn => "d6",
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
            (Self::Nudge, Detector::FileChurn) => Some(
                "You have rewritten the same file several times without the check changing its \
                 verdict. Another rewrite is a guess, not a fix: state what the check actually \
                 complains about, or whether the check itself is the thing that is wrong.",
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
    churn: Ladder,
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
            churn: Ladder {
                warn: detectors.file_churn_warn,
                strategy: detectors.file_churn_strategy,
                stop: detectors.file_churn_stop,
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

    /// The level a run of rewrites to one file reaches.
    #[must_use]
    pub const fn churn(&self, writes: u32) -> Option<Level> {
        self.churn.level(writes)
    }
}

/// The tools whose purpose is editing a file.
///
/// Listed here rather than inferred from the tool name: a tool called `patch` or `write_state`
/// would otherwise be counted or missed by a guess. D6 is about a file being rewritten, so only
/// the tools that exist to rewrite one are in the list.
const FILE_EDITORS: &[&str] = &["write_file", "edit_file"];

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
    /// Every output digest this attempt has produced, so a repeat is recognisable even when the
    /// steps are not adjacent.
    ///
    /// D2 needs adjacency and D3 must not: the measured A-B-A-B loop alternated two commands whose
    /// two outputs were both known, so every digest differed from its predecessor and a window of
    /// one saw nothing. The memory is the whole attempt and therefore bounded by its step budget,
    /// and each entry is a 32-byte digest, not the output itself.
    seen_outputs: std::collections::BTreeSet<String>,
    /// Consecutive steps that produced no observation this attempt had not already seen.
    unproductive: u32,
    /// The harshest unproductive-stretch level already announced.
    progress_announced: Option<Level>,
    /// The file most recently rewritten, and how many times it has been rewritten since a write
    /// to some other file.
    churn_path: Option<String>,
    churn_run: u32,
    /// The harshest churn level already announced.
    churn_announced: Option<Level>,
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
    ///
    /// D3 is not D2 and does not care about adjacency: its question is whether this step told the
    /// attempt anything it did not already know. An output digest the attempt has seen before is
    /// therefore a step without progress *wherever* it appeared earlier, which is what makes an
    /// A-B-A-B alternation visible. D2's run stays adjacent on purpose — two identical results
    /// with real work in between are not a loop.
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
        // D3: a digest this attempt has not produced before is a new fact, and any position is
        // enough. This is fed for identical calls too — a repeat of `ls` answers with output the
        // model already has, and a step that learns nothing is exactly what D3 counts.
        if self.seen_outputs.insert(digest.clone()) {
            self.unproductive = 0;
        } else {
            self.unproductive = self.unproductive.saturating_add(1);
        }
        if same_call {
            // The identical call is D1's; do not also feed D2.
            return;
        }
        if self.last_output.as_deref() == Some(digest.as_str()) {
            self.semantic_run = self.semantic_run.saturating_add(1);
        } else {
            self.semantic_run = 1;
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
    /// D3's own counter is reset by a *new observation* as well, so this is for progress a digest
    /// cannot see: a write that changed the tree while printing the same acknowledgement, or a
    /// gate that went green.
    pub fn record_progress(&mut self) {
        self.unproductive = 0;
        // Progress ends a churn cycle: rewriting a file after something *worked* is the next
        // piece of work, not a continuation of the guess that failed.
        self.churn_path = None;
        self.churn_run = 0;
    }

    /// Record a write to `path` by `tool`.
    ///
    /// D6's signal is deliberately about the *destination*, not the content: D1 compares calls and
    /// is blind to a model that rewrites one file nineteen times, because each body is a fresh
    /// guess. Measured on the first live run: nineteen successive writes of `spec/probe_spec.rb`,
    /// each with a different body and each followed by the same failing check, and no detector
    /// fired until the step budget ended the attempt.
    ///
    /// Only the tools whose purpose is editing a file count ([`FILE_EDITORS`]). A shell redirect
    /// that names a path is not one: `cargo test > /tmp/out` repeated eight times is an ordinary
    /// run, and counting it would nudge a model that is making progress.
    ///
    /// Non-editing calls do not reset the run: the measured sequence interleaved the rewrite with
    /// the check that judged it, and a counter that only looked at adjacent calls would have been
    /// reset by every check.
    pub fn record_write(&mut self, tool: &str, path: &str) {
        if !FILE_EDITORS.contains(&tool) {
            return;
        }
        if self.churn_path.as_deref() == Some(path) {
            self.churn_run = self.churn_run.saturating_add(1);
        } else {
            self.churn_path = Some(path.to_string());
            self.churn_run = 1;
            self.churn_announced = None;
        }
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

    /// How many steps have passed without an observation this attempt had not already seen.
    #[must_use]
    pub const fn unproductive(&self) -> u32 {
        self.unproductive
    }

    /// How many times the same file has been rewritten in a row.
    #[must_use]
    pub const fn churn_run(&self) -> u32 {
        self.churn_run
    }

    /// The file the churn run is about, if any.
    #[must_use]
    pub fn churn_path(&self) -> Option<&str> {
        self.churn_path.as_deref()
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
        let decision = raise(
            Detector::FileChurn,
            thresholds.churn(self.churn_run),
            self.churn_run,
            &mut self.churn_announced,
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
