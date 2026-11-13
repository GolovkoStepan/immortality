//! Behaviour the `word-count` task must deliver.
//!
//! Red in the repository as committed, because `count::once_words` does not exist yet. That is the
//! point: a gate has to distinguish "the work was done" from "the work was skipped", and a gate
//! that passes on the untouched repository cannot.
//!
//! They live in `tests/` rather than inside `src/count.rs` for two reasons. An integration test
//! can only call public items, which is the surface a reader cares about anyway; and a test that
//! does not exist yet is not a gate at all — `cargo test` with a filter that matches nothing exits
//! zero, so "the test the agent has not written" would be green for free.

use tally::count;

#[test]
fn a_word_seen_once_is_counted_once() {
    // A word that appears exactly once, as opposed to the number of distinct words the tool
    // already prints.
    assert_eq!(count::once_words("a b a"), 1);
    assert_eq!(count::once_words("a b c"), 3);
}

#[test]
fn repeated_spellings_of_one_word_are_not_once_words() {
    // The normalisation rule the existing counts already use has to apply here too, or `The the`
    // would count as two once-words.
    assert_eq!(count::once_words("The the"), 0);
    assert_eq!(count::once_words("The, the."), 0);
}

#[test]
fn empty_text_has_no_once_words() {
    assert_eq!(count::once_words(""), 0);
}
