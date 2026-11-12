//! Counting tokens when the backend will not.
//!
//! # Why a count is needed at all
//!
//! The context budget decides compaction: the soft limit triggers a summary, the hard limit
//! abandons the attempt. Both are computed from the prompt we are about to send, so the count must
//! exist *before* the request goes out — a count that arrives with the answer describes the request
//! that already happened.
//!
//! # Why `chars / 3.5` alone is not enough
//!
//! The design's starting point is `chars / 3.5`, which is roughly right for English prose and
//! wrong for most of what an agent sends. Rust source is nearer 3.0 characters per token, JSON tool
//! output nearer 2.5, Cyrillic closer to 1.5. On a 64K hard limit a 20 % error is three steps of
//! budget — enough to compact too late or to abandon a healthy attempt.
//!
//! So the divisor is **learned**. Every response that reports `usage.prompt_tokens` is a labelled
//! example of "this many characters cost this many tokens", and the ratio is folded into a running
//! estimate. The design asks for exactly this: `chars / 3.5` with a correction coefficient from
//! history.
//!
//! # Why median-of-ratios and not a mean
//!
//! A measurement against a prompt that differed from what was sent — a retry that added a warning
//! message, say — produces a wild ratio. A mean is dragged by it; the median ignores it. The
//! estimator keeps a bounded ring of recent ratios, so one anomaly cannot dominate and an old
//! calibration eventually falls out instead of pinning the estimate forever.
//!
//! # What this module does not do
//!
//! It does not tokenise. A real tokeniser would be exact and would be a 10 MB table that must match
//! the backend's vocabulary exactly or be wrong in a way nobody can see. The job is an estimate good
//! enough for a threshold with 2× headroom, and measuring the error is the honest way to get there.
//!
//! It also does not split the divisor by character class. Doing that would need a per-class sample
//! count to calibrate against, and with a handful of samples per class the estimate would fit noise
//! rather than content. One measured divisor over mixed text is the version that is defensible
//! today; a test pins the behaviour so a future refinement is a visible change.

use std::collections::VecDeque;

use crate::turn::Usage;

/// Divisor used when nothing has been measured yet.
///
/// 3.5 is the design's number: measured on English prose and Rust source, which is most of what a
/// first step contains. It is a starting point, not a claim — [`Estimator`] replaces it with
/// measured ratios as soon as the backend reports usage.
pub const DEFAULT_CHARS_PER_TOKEN: f64 = 3.5;

/// How many recent ratios the estimator keeps.
///
/// Small enough that the profile follows a change of task (Rust source after a long session on
/// documentation) within a few turns, large enough that one anomaly is not the median.
pub const WINDOW: usize = 16;

/// How far an estimate may differ from a measurement before it is called a divergence.
///
/// 15 % is the design's figure, and it is the threshold for "the estimator has stopped describing
/// the content", not for "the estimate is unusable": a threshold check with 2× headroom tolerates a
/// much larger error. Crossing it is worth reporting because it means something changed — a new
/// kind of content, or a bug in what was measured.
pub const DIVERGENCE_THRESHOLD: f64 = 0.15;

/// A calibrated token estimate for prompts and completions.
///
/// # Two ratios, not one
///
/// The prompt and the completion are different distributions: the prompt is mostly source code and
/// tool output, the completion is mostly prose the model writes. Sharing one ratio means every
/// answer drags the prompt estimate toward the model's writing style. They are measured and kept
/// apart.
#[derive(Debug, Clone)]
pub struct Estimator {
    /// Recent measured prompt ratios, in characters per token.
    prompt: VecDeque<f64>,
    /// Recent measured completion ratios.
    completion: VecDeque<f64>,
    /// Measurements folded in, for the report and for a caller deciding whether to trust it.
    samples: u64,
}

impl Default for Estimator {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator {
    /// A fresh estimator with no measurements.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prompt: VecDeque::with_capacity(WINDOW),
            completion: VecDeque::with_capacity(WINDOW),
            samples: 0,
        }
    }

    /// Characters per prompt token currently believed.
    ///
    /// The median of the measured ratios, or [`DEFAULT_CHARS_PER_TOKEN`] before any measurement.
    #[must_use]
    pub fn chars_per_prompt_token(&self) -> f64 {
        median(&self.prompt).unwrap_or(DEFAULT_CHARS_PER_TOKEN)
    }

    /// Characters per completion token currently believed.
    #[must_use]
    pub fn chars_per_completion_token(&self) -> f64 {
        median(&self.completion).unwrap_or(DEFAULT_CHARS_PER_TOKEN)
    }

    /// How many measurements have been folded in.
    #[must_use]
    pub const fn samples(&self) -> u64 {
        self.samples
    }

    /// Estimate the tokens in a prompt of `chars` characters.
    ///
    /// Rounds up: a budget check that rounds down lets the prompt cross the limit by a little on
    /// every step, and the limit is the thing that decides whether an attempt is abandoned.
    #[must_use]
    pub fn estimate_chars(&self, chars: u64) -> u32 {
        tokens_from(chars, self.chars_per_prompt_token())
    }

    /// Estimate the tokens in a prompt, counting characters rather than bytes.
    ///
    /// Characters and not bytes: a Cyrillic comment is twice as many bytes as characters, and the
    /// budget is in tokens, which track characters.
    #[must_use]
    pub fn estimate(&self, text: &str) -> u32 {
        self.estimate_chars(char_count(text))
    }

    /// Estimate the tokens in a completion of `chars` characters.
    #[must_use]
    pub fn estimate_completion_chars(&self, chars: u64) -> u32 {
        tokens_from(chars, self.chars_per_completion_token())
    }

    /// Fold a measured usage block into the calibration.
    ///
    /// The character counts are those of the request and of the answer as they were sent. Either
    /// may be omitted: a streamed response reports usage even when the caller did not keep the whole
    /// answer, and a token count with no character count is not a measurement.
    ///
    /// A ratio is recorded only when it is plausible. `prompt_tokens: 0` for a non-empty prompt
    /// means the backend reported something other than what we asked for, and folding it in would
    /// make the ratio infinite-ish for the next sixteen samples.
    pub fn observe(&mut self, prompt_chars: Option<u64>, completion_chars: Option<u64>, usage: &Usage) {
        if usage.is_absent() {
            return;
        }
        let prompt_ratio = prompt_chars
            .filter(|chars| *chars > 0 && usage.prompt_tokens > 0)
            .map(|chars| ratio(chars, usage.prompt_tokens));
        let completion_ratio = completion_chars
            .filter(|chars| *chars > 0 && usage.completion_tokens > 0)
            .map(|chars| ratio(chars, usage.completion_tokens));

        if let Some(ratio) = prompt_ratio {
            push_bounded(&mut self.prompt, ratio);
        }
        if let Some(ratio) = completion_ratio {
            push_bounded(&mut self.completion, ratio);
        }
        if prompt_ratio.is_some() || completion_ratio.is_some() {
            self.samples += 1;
        }
    }

    /// Compare an estimate against a measurement, as a relative error.
    ///
    /// `None` when the measurement is absent or the estimate is zero: there is nothing to compare,
    /// and reporting a divergence against zero would fire on every first request.
    #[must_use]
    pub fn divergence(estimated: u32, usage: &Usage) -> Option<f64> {
        if usage.is_absent() || estimated == 0 {
            return None;
        }
        let estimated = f64::from(estimated);
        let actual = f64::from(usage.prompt_tokens);
        Some((estimated - actual).abs() / actual.max(1.0))
    }

    /// Whether the estimate has stopped describing the content.
    #[must_use]
    pub fn is_diverging(estimated: u32, usage: &Usage) -> bool {
        Self::divergence(estimated, usage).is_some_and(|error| error > DIVERGENCE_THRESHOLD)
    }

    /// A one-line report for `doctor` and for the per-step log line.
    ///
    /// Always says how many measurements stand behind the numbers: an estimate calibrated on three
    /// samples and one calibrated on three hundred deserve different trust, and a report that hides
    /// the difference invites acting on the third sample.
    #[must_use]
    pub fn report(&self) -> String {
        format!(
            "prompt {:.2} chars/token, completion {:.2} chars/token, {} measurement(s)",
            self.chars_per_prompt_token(),
            self.chars_per_completion_token(),
            self.samples
        )
    }
}

/// Characters in `text`, saturating rather than wrapping.
#[must_use]
pub fn char_count(text: &str) -> u64 {
    u64::try_from(text.chars().count()).unwrap_or(u64::MAX)
}

fn ratio(chars: u64, tokens: u32) -> f64 {
    // Zero is guarded by the caller, but making the function total is cheaper than reasoning about
    // it at every call site.
    if tokens == 0 {
        return DEFAULT_CHARS_PER_TOKEN;
    }
    // A cast that saturates is still better than one that wraps.
    f64::from(u32::try_from(chars).unwrap_or(u32::MAX)) / f64::from(tokens)
}

fn tokens_from(chars: u64, divisor: f64) -> u32 {
    if chars == 0 {
        return 0;
    }
    let divisor = if divisor.is_finite() && divisor > 0.0 {
        divisor
    } else {
        DEFAULT_CHARS_PER_TOKEN
    };
    // Going through `f64` rather than through `u32` for the character count: clamping the count to
    // `u32::MAX` first would make a 4-billion-character prompt report 1.2 billion tokens instead of
    // saturating, and the saturated value is the one every budget check needs.
    let tokens = (chars as f64 / divisor).ceil();
    // `ceil` can round up past the ceiling, and a count above `u32::MAX` is not meaningful; every
    // caller compares it against a budget far below it, so saturating is the safe answer.
    if !tokens.is_finite() || tokens >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let out = tokens as u32;
        out
    }
}

fn push_bounded(window: &mut VecDeque<f64>, value: f64) {
    if !value.is_finite() || value <= 0.0 {
        return;
    }
    if window.len() == WINDOW {
        window.pop_front();
    }
    window.push_back(value);
}

/// The median of a window, or `None` when it is empty.
///
/// Copies into a `Vec` to sort: the window is sixteen elements, and a selection algorithm would be
/// more code than the copy costs.
fn median(window: &VecDeque<f64>) -> Option<f64> {
    if window.is_empty() {
        return None;
    }
    let mut values: Vec<f64> = window.iter().copied().collect();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        let left = *values.get(middle.saturating_sub(1))?;
        let right = *values.get(middle)?;
        Some(f64::midpoint(left, right))
    } else {
        values.get(middle).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(prompt: u32, completion: u32) -> Usage {
        Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cache_n: 0,
        }
    }

    #[test]
    fn a_fresh_estimator_uses_the_designs_divisor() {
        let estimator = Estimator::new();
        assert!((estimator.chars_per_prompt_token() - 3.5).abs() < 1e-9);
        assert_eq!(estimator.samples(), 0);
    }

    #[test]
    fn an_estimate_rounds_up() {
        // A budget check that rounds down lets the prompt cross the limit a little on every step.
        let estimator = Estimator::new();
        assert_eq!(estimator.estimate_chars(4), 2, "4/3.5 = 1.14");
        assert_eq!(estimator.estimate_chars(0), 0);
    }

    #[test]
    fn characters_are_counted_not_bytes() {
        // A Cyrillic comment is twice as many bytes as characters, and the budget is in tokens,
        // which track characters.
        let estimator = Estimator::new();
        assert_eq!(estimator.estimate("абв"), estimator.estimate("abc"));
        assert_eq!(char_count("абв"), 3);
        assert_eq!(char_count(""), 0);
    }

    #[test]
    fn one_measurement_moves_the_ratio() {
        let mut estimator = Estimator::new();
        // 1000 characters cost 250 tokens: 4 chars/token, not the default 3.5.
        estimator.observe(Some(1000), None, &usage(250, 0));
        assert_eq!(estimator.samples(), 1);
        assert!((estimator.chars_per_prompt_token() - 4.0).abs() < 1e-9);
        // The completion ratio is untouched: the two distributions are kept apart on purpose.
        assert!((estimator.chars_per_completion_token() - 3.5).abs() < 1e-9);
    }

    #[test]
    fn the_median_ignores_one_anomalous_measurement() {
        // The retry that added a warning message: the characters and the tokens describe different
        // requests and the ratio is wild. A mean would be dragged; the median not.
        let mut estimator = Estimator::new();
        for _ in 0..4 {
            estimator.observe(Some(1000), None, &usage(250, 0));
        }
        estimator.observe(Some(10_000), None, &usage(10, 0));
        assert!(
            (estimator.chars_per_prompt_token() - 4.0).abs() < 1e-9,
            "got {}",
            estimator.chars_per_prompt_token()
        );
    }

    #[test]
    fn an_even_window_takes_the_midpoint() {
        let mut estimator = Estimator::new();
        estimator.observe(Some(1000), None, &usage(250, 0));
        estimator.observe(Some(1000), None, &usage(200, 0));
        // 4.0 and 5.0 → 4.5.
        assert!((estimator.chars_per_prompt_token() - 4.5).abs() < 1e-9);
    }

    #[test]
    fn old_measurements_fall_out_of_the_window() {
        // A session that starts on prose and moves to Rust must follow, not average forever.
        let mut estimator = Estimator::new();
        for _ in 0..WINDOW {
            estimator.observe(Some(1000), None, &usage(250, 0));
        }
        assert!((estimator.chars_per_prompt_token() - 4.0).abs() < 1e-9);
        for _ in 0..WINDOW {
            estimator.observe(Some(1000), None, &usage(334, 0));
        }
        assert!(
            (estimator.chars_per_prompt_token() - 2.994).abs() < 0.01,
            "got {}",
            estimator.chars_per_prompt_token()
        );
    }

    #[test]
    fn an_absent_usage_block_changes_nothing() {
        // `Some(Usage::default())` would look like a measured zero and poison the calibration.
        let mut estimator = Estimator::new();
        estimator.observe(Some(5), Some(2), &Usage::default());
        assert_eq!(estimator.samples(), 0);
        assert!((estimator.chars_per_prompt_token() - 3.5).abs() < 1e-9);
    }

    #[test]
    fn a_zero_token_count_is_not_folded_in() {
        // A backend reporting `prompt_tokens: 0` for a large prompt is not saying the ratio is
        // infinite; it is saying nothing.
        let mut estimator = Estimator::new();
        estimator.observe(Some(1000), None, &usage(0, 5));
        assert_eq!(estimator.samples(), 0);
        assert!((estimator.chars_per_prompt_token() - 3.5).abs() < 1e-9);
    }

    #[test]
    fn a_completion_can_be_measured_alone() {
        let mut estimator = Estimator::new();
        estimator.observe(None, Some(900), &usage(0, 300));
        assert!((estimator.chars_per_completion_token() - 3.0).abs() < 1e-9);
        assert!((estimator.chars_per_prompt_token() - 3.5).abs() < 1e-9);
        assert_eq!(estimator.estimate_completion_chars(900), 300);
    }

    #[test]
    fn a_cache_hit_still_counts_as_a_measurement() {
        // The cached prefix is still a prefix: `prompt_tokens` describes the whole prompt, and the
        // ratio is about characters and tokens, not about what was recomputed.
        let mut estimator = Estimator::new();
        let mut measured = usage(250, 0);
        measured.cache_n = 240;
        estimator.observe(Some(1000), None, &measured);
        assert_eq!(estimator.samples(), 1);
        assert!((estimator.chars_per_prompt_token() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn divergence_is_the_relative_error() {
        let measured = usage(100, 0);
        let divergence = Estimator::divergence(120, &measured).unwrap();
        assert!((divergence - 0.2).abs() < 1e-9);
        let divergence = Estimator::divergence(80, &measured).unwrap();
        assert!((divergence - 0.2).abs() < 1e-9, "sign does not matter");
    }

    #[test]
    fn divergence_needs_a_measurement_and_an_estimate() {
        assert_eq!(Estimator::divergence(100, &Usage::default()), None);
        assert_eq!(Estimator::divergence(0, &usage(100, 0)), None);
    }

    #[test]
    fn the_fifteen_percent_threshold_is_the_boundary() {
        let measured = usage(100, 0);
        assert!(!Estimator::is_diverging(114, &measured));
        assert!(Estimator::is_diverging(116, &measured));
        assert!(!Estimator::is_diverging(86, &measured));
        assert!(Estimator::is_diverging(84, &measured));
    }

    #[test]
    fn the_report_says_how_many_measurements_stand_behind_it() {
        let mut estimator = Estimator::new();
        assert!(
            estimator.report().contains("0 measurement(s)"),
            "{}",
            estimator.report()
        );
        estimator.observe(Some(1000), None, &usage(250, 0));
        assert!(
            estimator.report().contains("1 measurement(s)"),
            "{}",
            estimator.report()
        );
        assert!(
            estimator.report().contains("4.00 chars/token"),
            "{}",
            estimator.report()
        );
    }

    #[test]
    fn a_huge_prompt_saturates_instead_of_wrapping() {
        // A wrapping cast would turn a 4-billion-character prompt into a small token count and pass
        // the budget check, which is the exact bug saturation prevents. The ceiling here is
        // `u32::MAX / 3.5`, so the assertion is on the magnitude rather than on `u32::MAX` itself.
        let estimator = Estimator::new();
        let tokens = estimator.estimate_chars(u64::from(u32::MAX));
        assert!(tokens > 1_000_000_000, "{tokens}");

        // Past `u32::MAX / 3.5` the count would not fit, and saturation is the answer.
        assert_eq!(estimator.estimate_chars(u64::MAX), u32::MAX, "not a wrap");
    }

    #[test]
    fn a_nonsense_divisor_falls_back_to_the_default() {
        // Only reachable through a corrupted estimator, but `tokens_from` is cheap to make total.
        assert_eq!(tokens_from(7, 0.0), 2);
        assert_eq!(tokens_from(7, f64::NAN), 2);
        assert_eq!(tokens_from(0, 0.0), 0);
    }

    #[test]
    fn a_realistic_prompt_estimate_is_within_the_designs_tolerance() {
        // The design's acceptance: after calibration the heuristic and the measured usage agree
        // within 15 %. The content is Cyrillic, where the default 3.5 is wrong by more than a
        // factor of two — which is exactly the case the calibration exists for.
        let mut estimator = Estimator::new();
        let prompt = "функция возвращает результат вычисления\n".repeat(100);
        let chars = char_count(&prompt);
        // Say the backend charged 2400 tokens: 3600 characters / 2400 = 1.5 chars/token.
        for _ in 0..5 {
            estimator.observe(Some(chars), None, &usage(2400, 0));
        }
        let estimated = estimator.estimate(&prompt);
        assert!(
            !Estimator::is_diverging(estimated, &usage(2400, 0)),
            "estimated {estimated}, measured 2400"
        );
        // And on a fresh estimator the same prompt is off by more than the threshold, which is the
        // whole reason the calibration exists.
        assert!(Estimator::is_diverging(
            Estimator::new().estimate(&prompt),
            &usage(2400, 0)
        ));
    }

    #[test]
    fn a_catastrophic_anomaly_cannot_poison_the_window() {
        // A ratio of infinity or a negative one comes from an arithmetic bug or a corrupted
        // reading; recording it would make the next sixteen estimates nonsense.
        let mut estimator = Estimator::new();
        push_bounded(&mut estimator.prompt, f64::INFINITY);
        push_bounded(&mut estimator.prompt, -3.0);
        push_bounded(&mut estimator.prompt, f64::NAN);
        push_bounded(&mut estimator.prompt, 0.0);
        assert_eq!(estimator.prompt.len(), 0);
    }
}
