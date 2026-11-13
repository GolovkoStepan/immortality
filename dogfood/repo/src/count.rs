//! The counting rules.
//!
//! Kept in their own module so each rule can be tested without running the binary.

/// Number of lines in `text`.
///
/// A file that ends with a newline has as many lines as it has newlines; the empty string has
/// none.
///
/// # The bug this fixture ships with
///
/// `str::split` yields a trailing empty field after a final separator, so `lines("a\nb\n")` is 3
/// and not 2. The task `line-count` exists to fix exactly this, and its gate is red until it is
/// fixed — which is what makes the task a fair test rather than a no-op.
pub fn lines(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    text.split('\n').count()
}

/// Number of whitespace-separated words.
pub fn words(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Number of distinct words, compared case-insensitively.
pub fn unique_words(text: &str) -> usize {
    let mut seen: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        let normalised = normalise(word);
        if !seen.contains(&normalised) {
            seen.push(normalised);
        }
    }
    seen.len()
}

/// Strip punctuation and lowercase, so `The,` and `the` count as one word.
fn normalise(word: &str) -> String {
    word.trim_matches(|character: char| !character.is_alphanumeric())
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_has_no_lines_and_no_words() {
        assert_eq!(lines(""), 0);
        assert_eq!(words(""), 0);
        assert_eq!(unique_words(""), 0);
    }

    #[test]
    fn words_are_split_on_whitespace() {
        assert_eq!(words("one two\tthree\nfour"), 4);
    }

    #[test]
    fn unique_words_ignore_case_and_punctuation() {
        assert_eq!(unique_words("The the THE, the."), 1);
    }

    #[test]
    fn distinct_words_are_counted_separately() {
        assert_eq!(unique_words("alpha beta alpha"), 2);
    }
}
