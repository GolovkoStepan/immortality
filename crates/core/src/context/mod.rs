//! Assembling the prompt, built so that its prefix never moves.
//!
//! # Six layers, and why the order is load-bearing
//!
//! The prompt is built from six layers (`docs/design.md` §8):
//!
//! 1. **System prompt** — versioned, unchanged for the whole task.
//! 2. **Brief** — goal, definition of done, constraints, current strategy.
//! 3. **Working memory** — `.agent/NOTES.md`: facts about the codebase, constraints found, and
//!    the *rejected hypotheses*, so the model stops walking in circles.
//! 4. **State digest** — an excerpt of the plan, the task tree's status, the last checkpoint.
//! 5. **Step history** — the last N tool calls with truncated results.
//! 6. **On demand** — `read_state`, `grep_journal`, `read_file`; not part of assembly at all.
//!
//! The order is not stylistic. On this backend an identical prefix gives `cache_n ≈ 30k` and a
//! prefill of 0.24 s instead of 95 s — 400×. Any edit at the *front* invalidates everything
//! after it. So layers 1–4 are rendered into one string exactly once per attempt, and layers
//! 5 grows only at the end. That is the entire caching strategy, and it is a property of the
//! data structures rather than a rule someone has to remember:
//!
//! * [`Context::new`] takes the stable layers and renders them immediately;
//! * the only mutating methods append to a `Vec` of history entries;
//! * there is no setter for the stable layers, because changing them mid-attempt is a new
//!   attempt, not an edit.
//!
//! A test asserts the property directly: assembling two different histories from the same
//! layers produces byte-identical system text.
//!
//! # Why the layers always emit their headings
//!
//! An empty layer is rendered as a placeholder rather than omitted. Two reasons. Omitting it
//! would silently drop a layer whose file exists but is empty — the model would have no way to
//! tell "no notes yet" from "notes were not shown". And the rendered shape stays constant, so
//! the prefix does not move when a layer gains its first line at a subtask boundary.
//!
//! # Budgets
//!
//! The soft budget (32K) triggers [`Verdict::Compact`]; the hard budget (64K) triggers
//! [`Verdict::Restart`]. The server window is 256K and is not the working budget: a cold 64K
//! prefill measured 166 s, and quality degrades long before the window fills.
//!
//! The count must exist *before* the request goes out, because it decides whether to send that
//! request at all. It comes from a [`TokenCounter`]; [`CharsCounter`] is the dependency-free
//! default, and `immortality-llm`'s calibrated estimator implements the same trait.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::config::ContextBudget;
use crate::journal::{Digest, digest_text};

mod compact;
mod truncate;

pub use compact::{
    Compaction, CompactionError, INSTRUCTION as COMPACTION_INSTRUCTION, NONE_MARKER, REQUIRED_SECTIONS,
    request as compaction_request,
};
pub use truncate::{SEPARATOR, Truncated, artifact_hint, digest_if_cut, digest_of_text, truncate_output};

/// Headings of the stable layers, in the order they must appear.
///
/// Exposed because "the order is a guarantee, not a convention" is a claim that needs a test,
/// and a test that hard-codes the strings would pass after the rendering changed underneath it.
pub const LAYER_HEADINGS: &[&str] = &["# Role", "# Task brief", "# Working memory", "# State"];

/// Placeholder for a layer with no content.
const NOTHING_YET: &str = "(nothing yet)";

/// What counts as a token, for a caller that has no calibrated estimator.
///
/// The trait exists so that `core` stays free of the backend client while still letting the
/// calibrated estimator decide the budget: `immortality-llm` depends on `core`, never the
/// other way round.
pub trait TokenCounter {
    /// Estimated tokens in `text`, rounded up.
    fn count(&self, text: &str) -> u32;
}

/// Characters divided by a fixed divisor.
///
/// The design's starting point of 3.5 characters per token is right for English prose and
/// optimistic for JSON tool output. It is good enough for a threshold with 2× headroom, and it
/// is what a test uses when the point is the *decision*, not the arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharsCounter {
    chars_per_token: f64,
}

impl CharsCounter {
    /// Default divisor from the design.
    pub const DEFAULT: f64 = 3.5;

    /// A counter with a given divisor.
    ///
    /// A non-positive or non-finite divisor is replaced by [`Self::DEFAULT`] rather than
    /// rejected: this is a measuring device, and a device that refuses to measure would stop a
    /// run over a configuration typo. The fallback is visible through
    /// [`Self::chars_per_token`].
    #[must_use]
    pub fn new(chars_per_token: f64) -> Self {
        let valid = chars_per_token.is_finite() && chars_per_token > 0.0;
        Self {
            chars_per_token: if valid { chars_per_token } else { Self::DEFAULT },
        }
    }

    /// The divisor actually used.
    #[must_use]
    pub const fn chars_per_token(self) -> f64 {
        self.chars_per_token
    }
}

impl Default for CharsCounter {
    fn default() -> Self {
        Self::new(Self::DEFAULT)
    }
}

impl TokenCounter for CharsCounter {
    fn count(&self, text: &str) -> u32 {
        if text.is_empty() {
            return 0;
        }
        let chars = u64::try_from(text.chars().count()).unwrap_or(u64::MAX);
        // Going through `f64` and saturating, like the calibrated estimator: a count above
        // `u32::MAX` is not meaningful, and every comparison is against a far smaller budget.
        let tokens = (chars as f64 / self.chars_per_token).ceil();
        if !tokens.is_finite() || tokens >= f64::from(u32::MAX) {
            u32::MAX
        } else {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let out = tokens as u32;
            out
        }
    }
}

/// The four layers that must not change between steps of one attempt.
///
/// Taken as text rather than built from `Task`/`Subtask`: composing a brief from the plan and
/// the strategy is the planner's job (M4.2), and a context assembler that also decided what the
/// brief says would make both harder to test.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StableLayers {
    pub system: String,
    pub brief: String,
    pub notes: String,
    pub state: String,
}

/// One tool call the model asked for, echoed into history.
///
/// The arguments are kept as the JSON text the model produced rather than as a parsed value:
/// a re-serialised value would reorder keys, and the prefix cache notices.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

/// One entry of the step history, append-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    /// What the model said, plus the calls it asked for.
    Assistant {
        /// Prose, possibly empty when the turn was pure tool calls.
        text: String,
        calls: Vec<ToolCall>,
    },
    /// A tool's result, already truncated by [`truncate_output`].
    Tool {
        call_id: String,
        name: String,
        output: String,
    },
}

impl Entry {
    /// A tool result with no truncation applied, for callers that already know it is small.
    #[must_use]
    pub fn tool(call_id: impl Into<String>, name: impl Into<String>, output: impl Into<String>) -> Self {
        Self::Tool {
            call_id: call_id.into(),
            name: name.into(),
            output: output.into(),
        }
    }

    /// Text of this entry as the model sees it, for counting.
    ///
    /// Tool results are counted with their text only: the call id and the name are a few dozen
    /// bytes, and counting them would make the estimate disagree with a backend that counts
    /// what the template actually renders — neither is knowable here, and the simpler rule is
    /// the one that can be tested.
    #[must_use]
    pub fn text(&self) -> &str {
        match self {
            Self::Assistant { text, .. } => text,
            Self::Tool { output, .. } => output,
        }
    }
}

/// What the budget says to do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Verdict {
    /// Send the step.
    Fits,
    /// Summarise the history first, then continue in the same attempt.
    Compact,
    /// Abandon the attempt: the context is too large for a summary to be a reasonable
    /// projection, and the next attempt starts from state with a different strategy.
    Restart,
}

impl Verdict {
    /// The wire name, for a journal record or a log line.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fits => "fits",
            Self::Compact => "compact",
            Self::Restart => "restart",
        }
    }
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The two thresholds, taken from config so that the number lives in one place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    pub soft: u32,
    pub hard: u32,
}

impl Budget {
    /// Read the thresholds from a validated config.
    ///
    /// No `Result`: `Config::validate` already refuses `soft >= hard`, and re-checking here
    /// would create a second authority that can disagree with the first.
    #[must_use]
    pub const fn from_config(config: &ContextBudget) -> Self {
        Self {
            soft: config.soft_budget_tokens,
            hard: config.hard_budget_tokens,
        }
    }

    /// Decide from a measured total.
    ///
    /// The comparison at the boundary is `>=` for both: a prompt exactly at the soft budget is
    /// already as large as the budget allows, and treating it as fitting would let the next
    /// step, always a little bigger, cross the line without a compaction having been offered.
    #[must_use]
    pub const fn verdict(&self, tokens: u32) -> Verdict {
        if tokens >= self.hard {
            Verdict::Restart
        } else if tokens >= self.soft {
            Verdict::Compact
        } else {
            Verdict::Fits
        }
    }
}

/// The assembled prompt for one step.
///
/// Cloning it is cheap and meaningless; it exists so that a caller can hold the submitted
/// prompt next to the answer without the next step's append disturbing it.
#[derive(Debug, Clone)]
pub struct Context {
    system: String,
    history: Vec<Entry>,
}

impl Context {
    /// Render the stable layers once and start an empty history.
    #[must_use]
    pub fn new(layers: &StableLayers) -> Self {
        Self {
            system: render_layers(layers),
            history: Vec::new(),
        }
    }

    /// The stable prefix: layers 1–4.
    ///
    /// Byte-identical for every step of an attempt, which is what the prefix cache keys on.
    #[must_use]
    pub fn system(&self) -> &str {
        &self.system
    }

    /// Everything after the prefix, oldest first.
    #[must_use]
    pub fn history(&self) -> &[Entry] {
        &self.history
    }

    /// Append the model's turn.
    pub fn push_assistant(&mut self, text: impl Into<String>, calls: Vec<ToolCall>) {
        self.history.push(Entry::Assistant {
            text: text.into(),
            calls,
        });
    }

    /// Append a tool result.
    ///
    /// Unlike `push_assistant` this takes an [`Entry`], so a caller that went through
    /// [`truncate_output`] cannot accidentally pass the untruncated text.
    pub fn push_entry(&mut self, entry: Entry) {
        self.history.push(entry);
    }

    /// Estimated tokens in the whole prompt.
    #[must_use]
    pub fn estimate(&self, counter: &dyn TokenCounter) -> u32 {
        let mut total = counter.count(&self.system);
        for entry in &self.history {
            // Saturating: a prompt beyond `u32::MAX` tokens is not a prompt any server
            // accepts, and wrapping would report a small number for an absurd one.
            total = total.saturating_add(counter.count(entry.text()));
        }
        total
    }

    /// What to do before sending this prompt.
    #[must_use]
    pub fn verdict(&self, counter: &dyn TokenCounter, budget: Budget) -> Verdict {
        budget.verdict(self.estimate(counter))
    }

    /// Drop the history, keeping the prefix.
    ///
    /// Called after a compaction has been committed to state: the summary becomes part of the
    /// *next* attempt's layers, not an edit to this attempt's prefix.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// How many entries the history holds, for a log line and for a test.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.history.len()
    }
}

/// Render the four stable layers into one string.
///
/// The headings are always emitted, in [`LAYER_HEADINGS`] order, even when a layer is empty.
fn render_layers(layers: &StableLayers) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    for (heading, content) in [
        (LAYER_HEADINGS[0], &layers.system),
        (LAYER_HEADINGS[1], &layers.brief),
        (LAYER_HEADINGS[2], &layers.notes),
        (LAYER_HEADINGS[3], &layers.state),
    ] {
        // `LAYER_HEADINGS` has exactly four entries, so the indexing above cannot fail.
        let _ = writeln!(out, "{heading}");
        if content.trim().is_empty() {
            let _ = writeln!(out, "{NOTHING_YET}");
        } else {
            let _ = writeln!(out, "{}", content.trim_end());
        }
        out.push('\n');
    }
    // One trailing blank line removed so the prefix ends exactly at the last layer: a caller
    // comparing prefixes then compares content rather than whitespace.
    while out.ends_with('\n') {
        out.pop();
    }
    out
}

/// A digest in the form it appears in prompts and journals.
///
/// Re-exported so a caller formatting a truncation hint does not have to know where the digest
/// type lives.
#[must_use]
pub fn artifact_name(digest: &Digest) -> String {
    digest_text(digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layers() -> StableLayers {
        StableLayers {
            system: "You are an agent.".to_string(),
            brief: "Add a `serve` subcommand.".to_string(),
            notes: "The socket is 0600.".to_string(),
            state: "1/3 subtasks done.".to_string(),
        }
    }

    #[test]
    fn the_stable_prefix_is_byte_identical_when_history_grows() {
        // The whole caching strategy in one assertion. A reordering of fields, or a setter that
        // rebuilt the prefix, would break it while every other test stayed green.
        let layers = layers();
        let first = Context::new(&layers);
        let prefix = first.system().to_string();

        let mut second = Context::new(&layers);
        second.push_assistant("looking", vec![]);
        second.push_entry(Entry::tool("call_1", "read_file", "fn main() {}"));
        second.push_assistant("done", vec![]);

        assert_eq!(second.system(), prefix);
        assert_eq!(first.system(), second.system());
        assert_eq!(first.steps(), 0);
        assert_eq!(second.steps(), 3);
    }

    #[test]
    fn layers_appear_in_the_documented_order() {
        let context = Context::new(&layers());
        let mut last = 0usize;
        let mut positions = Vec::new();
        for heading in LAYER_HEADINGS {
            let position = context
                .system()
                .find(heading)
                .unwrap_or_else(|| panic!("missing layer {heading}"));
            assert!(position >= last, "{heading} is out of order");
            last = position;
            positions.push(position);
        }
        assert_eq!(positions.len(), LAYER_HEADINGS.len());
    }

    #[test]
    fn an_empty_layer_is_named_rather_than_silently_dropped() {
        // "No notes yet" and "notes were withheld" look the same to a model if the heading is
        // omitted, and only one of them is true.
        let context = Context::new(&StableLayers {
            system: "S".to_string(),
            ..StableLayers::default()
        });
        assert!(context.system().contains("# Working memory"));
        assert!(context.system().contains(NOTHING_YET));
    }

    #[test]
    fn a_layer_that_gains_content_keeps_the_prefix_shape() {
        // Not the same bytes — the content changed — but *which* headings appear and in what
        // order is the invariant a caller can rely on across a subtask boundary. Positions are
        // not, because a longer layer pushes the headings after it to the right.
        let empty = Context::new(&StableLayers {
            system: "S".to_string(),
            ..StableLayers::default()
        });
        let full = Context::new(&layers());
        for context in [&empty, &full] {
            let mut last = 0usize;
            for heading in LAYER_HEADINGS {
                let position = context
                    .system()
                    .find(heading)
                    .unwrap_or_else(|| panic!("missing {heading}"));
                assert!(position >= last, "{heading} is out of order");
                last = position;
            }
        }
        assert!(full.system().len() > empty.system().len());
    }

    #[test]
    fn the_budget_decides_at_the_boundaries() {
        let budget = Budget { soft: 100, hard: 200 };
        assert_eq!(budget.verdict(0), Verdict::Fits);
        assert_eq!(budget.verdict(99), Verdict::Fits);
        // Exactly at the soft budget is too large: the next append always makes it bigger.
        assert_eq!(budget.verdict(100), Verdict::Compact);
        assert_eq!(budget.verdict(199), Verdict::Compact);
        assert_eq!(budget.verdict(200), Verdict::Restart);
        assert_eq!(budget.verdict(u32::MAX), Verdict::Restart);
    }

    #[test]
    fn the_budget_reads_its_numbers_from_config() {
        let config = ContextBudget::default();
        let budget = Budget::from_config(&config);
        assert_eq!(budget.soft, 32_000);
        assert_eq!(budget.hard, 64_000);
    }

    #[test]
    fn a_chars_counter_rounds_up_and_saturates() {
        let counter = CharsCounter::new(4.0);
        assert_eq!(counter.count(""), 0);
        assert_eq!(counter.count("abcd"), 1);
        // Five characters at four per token is 1.25, and a budget check that rounded down
        // would let the prompt drift past the limit one character at a time.
        assert_eq!(counter.count("abcde"), 2);
        assert_eq!(CharsCounter::new(0.0).chars_per_token(), CharsCounter::DEFAULT);
        assert_eq!(
            CharsCounter::new(f64::NAN).chars_per_token(),
            CharsCounter::DEFAULT
        );
    }

    #[test]
    fn the_estimate_counts_the_prefix_and_every_entry() {
        let mut context = Context::new(&StableLayers {
            system: "x".repeat(400),
            ..StableLayers::default()
        });
        let empty = context.estimate(&CharsCounter::new(4.0));
        context.push_entry(Entry::tool("c", "t", "y".repeat(400)));
        let after = context.estimate(&CharsCounter::new(4.0));
        assert_eq!(after - empty, 100, "400 characters at 4 per token");
    }

    #[test]
    fn verdict_uses_the_estimate_of_the_whole_prompt() {
        let mut context = Context::new(&StableLayers {
            system: "x".repeat(400),
            ..StableLayers::default()
        });
        let counter = CharsCounter::new(4.0);
        // Derived from the measured prefix rather than guessed: the rendered layers include
        // headings and placeholders, and a hard-coded budget would break the moment the
        // rendering changed for an unrelated reason.
        let empty = context.estimate(&counter);
        let budget = Budget {
            soft: empty,
            hard: empty + 100,
        };
        assert_eq!(context.verdict(&counter, budget), Verdict::Compact);
        // 400 characters at four per token is exactly the remaining room to the hard budget.
        context.push_entry(Entry::tool("c", "t", "y".repeat(400)));
        assert_eq!(context.verdict(&counter, budget), Verdict::Restart);
    }

    #[test]
    fn clearing_the_history_keeps_the_prefix_and_reclaims_the_budget() {
        let mut context = Context::new(&layers());
        let prefix = context.system().to_string();
        context.push_entry(Entry::tool("c", "t", "z".repeat(4000)));
        context.clear_history();
        assert_eq!(context.system(), prefix);
        assert_eq!(context.steps(), 0);
    }

    #[test]
    fn verdict_names_are_stable() {
        // They go into journal records, so a rename is a schema change.
        assert_eq!(Verdict::Fits.as_str(), "fits");
        assert_eq!(Verdict::Compact.as_str(), "compact");
        assert_eq!(Verdict::Restart.as_str(), "restart");
        assert_eq!(Verdict::Compact.to_string(), "compact");
    }
}
