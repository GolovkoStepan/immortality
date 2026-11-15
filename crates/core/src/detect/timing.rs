//! Measured durations, removed before the output digest is taken.
//!
//! # The defect this exists for
//!
//! D3 asks "did this step tell the attempt anything it did not already know?", and answers by
//! comparing output digests. A command that reports how long it took makes every answer unique
//! even when nothing else changed. Measured on the third live `black_jack` run, first attempt
//! `BZFCV9MR`: 32 repetitions of the same action in a row, and D3 saw a series of 1 — because
//! `files took 0.04258 seconds` and `[exited 0 in 0.3 s, …]` differed on every run.
//!
//! # Why the forms are listed instead of matched as "number and a unit"
//!
//! A general rule would eat `4 examples` and `148 examples`, which are *results*, not timings:
//! two `rspec` runs with different example counts are different outcomes, and merging them is
//! exactly the general number normalisation D-104 and D-108 rejected on measured data. So the
//! rule matches only the two forms a run actually produced, and adding a third is a decision
//! backed by a run — not a convenience.
//!
//! # Where this applies, and where it does not
//!
//! Only to the digest. The text a model reads stays raw, because `[exited 0 in 5.0 s, …]` is
//! precisely how the D-112 regression announced itself (a false "still alive" made every command
//! wait out `KILL_GRACE`), and a normalised observation would hide that from the model and from
//! whoever reads the run afterwards.

/// What a measured duration collapses to.
const DURATION: &str = "<duration>";

/// Replace measured durations with [`DURATION`], leaving every other byte alone.
///
/// Two forms are recognised, both taken from real output:
///
/// - `<number> seconds` / `<number> second`, which is what `rspec` prints as `files took 0.04258
///   seconds` and `Finished in 0.04 seconds`;
/// - `<number> s,`, which is the `shell` tool's own summary: `[exited 0 in 0.3 s, 12 line(s) of
///   stdout, 0 of stderr]`.
///
/// The rest of the line is untouched, including numbers that are not durations: `148 examples`
/// and `4 examples, 0 failures` keep their counts, which is what keeps D3 from firing on a
/// legitimate test run whose example count changed.
#[must_use]
pub fn strip_durations(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < bytes.len() {
        if starts_word(bytes, index)
            && let Some(number_end) = number_at(bytes, index)
            && let Some(unit_end) = duration_unit_at(bytes, number_end)
        {
            out.push_str(DURATION);
            index = unit_end;
            continue;
        }
        // One character, not one byte: the output is arbitrary UTF-8 and slicing mid-character
        // would panic on the first non-ASCII line a command prints. The `while` guard keeps the
        // index in bounds, so a missing character is unreachable rather than merely unlikely.
        let mut characters = text[index..].chars();
        let Some(character) = characters.next() else {
            break;
        };
        out.push(character);
        index += character.len_utf8();
    }
    out
}

/// Whether a number may start here: the beginning, or after something that is not a digit or a
/// dot. Without this, `0.04258` would also be tried as `04258` and `258`.
fn starts_word(bytes: &[u8], index: usize) -> bool {
    index
        .checked_sub(1)
        .map(|before| bytes[before])
        .is_none_or(|before| !before.is_ascii_alphanumeric() && before != b'.' && before != b'_')
}

/// The end of a decimal number starting exactly at `index`, or `None` when there is none.
///
/// Digits, with at most one dot and at least one digit on each side of it. `0.` and `.5` are not
/// numbers here: neither shape appears in a duration, and accepting them would widen the matcher
/// for no measured reason.
fn number_at(bytes: &[u8], index: usize) -> Option<usize> {
    let mut cursor = index;
    let mut digits = 0_u32;
    let mut dots = 0_u32;
    while let Some(&byte) = bytes.get(cursor) {
        if byte.is_ascii_digit() {
            digits = digits.saturating_add(1);
            cursor += 1;
        } else if byte == b'.' && dots == 0 {
            dots = dots.saturating_add(1);
            cursor += 1;
        } else {
            break;
        }
    }
    // `0.` is not a number: a trailing dot with nothing after it belongs to prose, and the loop
    // above would otherwise accept `1.` as a duration of one second.
    if digits == 0 || bytes.get(cursor.wrapping_sub(1)) == Some(&b'.') {
        return None;
    }
    Some(cursor)
}

/// The end of a duration unit starting at `index` (just past the number), or `None`.
///
/// Whitespace between the number and its unit is part of the duration, so `0.3 s` and `0.3 s`
/// with a stray space collapse the same way.
fn duration_unit_at(bytes: &[u8], index: usize) -> Option<usize> {
    let mut cursor = index;
    while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
        cursor += 1;
    }
    let rest = bytes.get(cursor..)?;
    for word in ["seconds", "second"] {
        if rest.starts_with(word.as_bytes()) {
            let after = cursor.saturating_add(word.len());
            // A longer word that begins with `second` (`seconds`, handled above; `secondary`) is
            // not a unit. Without this, `2 secondary` would be read as two seconds.
            if bytes.get(after).is_none_or(|byte| !byte.is_ascii_alphanumeric()) {
                return Some(after);
            }
        }
    }
    // The `shell` summary's unit: `in 0.3 s, 12 line(s)`. The comma is what makes this narrow —
    // a bare `s` after a number appears in ordinary prose, and `0.3 s,` does not. The comma is
    // checked but not consumed: it belongs to the summary, and eating it would make the
    // normalised line differ from the raw one in a way nothing asked for.
    if rest.starts_with(b"s,") {
        return Some(cursor.saturating_add(1));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rspec_duration_is_the_same_whatever_it_measured() {
        // The measured case: two runs of one suite that differ only in how long the files took.
        let first = strip_durations("Randomized with seed 4123\nfiles took 0.04258 seconds to load");
        let second = strip_durations("Randomized with seed 4123\nfiles took 1.9 seconds to load");
        assert_eq!(first, second);
        assert!(first.contains("<duration>"), "{first}");
        assert!(
            first.contains("4123"),
            "the seed is a result, not a timing: {first}"
        );
    }

    #[test]
    fn the_shell_summary_duration_is_the_same_whatever_it_measured() {
        let first = strip_durations("[exited 0 in 0.3 s, 12 line(s) of stdout, 0 of stderr]");
        let second = strip_durations("[exited 0 in 5.0 s, 12 line(s) of stdout, 0 of stderr]");
        assert_eq!(first, second);
        assert_eq!(
            first,
            "[exited 0 in <duration>, 12 line(s) of stdout, 0 of stderr]"
        );
    }

    #[test]
    fn a_different_number_of_examples_is_a_different_digest() {
        // The guard D-104 and D-108 demand. `4 examples` and `148 examples` are two different
        // results; folding them together would make a healthy run look like a loop, which is the
        // failure that got general number normalisation rejected in the first place.
        let few = strip_durations("4 examples, 0 failures");
        let many = strip_durations("148 examples, 0 failures");
        assert_ne!(few, many);
        assert_eq!(few, "4 examples, 0 failures");
        assert_eq!(many, "148 examples, 0 failures");
    }

    #[test]
    fn a_line_count_and_a_seed_survive() {
        // Everything in the shell summary except the duration has to stay, or the summary becomes
        // a constant and D3 stops seeing a command that started printing stderr.
        let text = "[exited 1 in 2.0 s, 3 line(s) of stdout, 7 of stderr]";
        assert_eq!(
            strip_durations(text),
            "[exited 1 in <duration>, 3 line(s) of stdout, 7 of stderr]"
        );
    }

    #[test]
    fn a_word_that_only_begins_with_second_is_not_a_unit() {
        // `2 secondary` is not two seconds. The unit check has to look at the character after the
        // word, not just its prefix.
        assert_eq!(strip_durations("2 secondary"), "2 secondary");
    }

    #[test]
    fn a_number_in_a_sentence_is_left_alone() {
        // No unit, no duration: this is the ordinary case, and the normaliser must be invisible
        // in it rather than trimming whitespace or rewriting numbers it does not understand.
        let text = "the build ran 42 steps and printed 3 warnings";
        assert_eq!(strip_durations(text), text);
    }

    #[test]
    fn non_ascii_output_does_not_panic() {
        // The scanner copies one character at a time precisely so a Cyrillic or emoji line cannot
        // be sliced mid-character.
        let text = "проверка 0.5 seconds — готово ✅";
        assert_eq!(strip_durations(text), "проверка <duration> — готово ✅");
    }

    #[test]
    fn a_trailing_dot_is_not_a_number() {
        // `1.` in prose is not a duration of one second, and accepting it would eat the sentence.
        assert_eq!(
            strip_durations("see step 1. seconds later"),
            "see step 1. seconds later"
        );
    }
}
