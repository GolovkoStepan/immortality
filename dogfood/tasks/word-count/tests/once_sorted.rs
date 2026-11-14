//! The accessor the `word-count` task's third subtask must deliver.
//!
//! Red in the repository as committed, because `count::once_words_sorted` does not exist yet. It
//! is a third test target because `word-count` is the task that has to demonstrate three
//! consecutive subtasks, and each of those subtasks needs a gate that is red **when it starts**.
//! `cargo test --test <name>` compiles only its own target, so the first subtask's green cannot
//! make this one green for free.
//!
//! # Why a separate capability rather than "write a test"
//!
//! Asking the subtask to write a regression test would make the gate "a test target with that name
//! exists", which a model satisfies with an empty test. Shipping a failing specification instead
//! makes the gate mean what the other two mean: a named function exists and answers correctly.
//! `once_words_sorted` is a different function from `once_words` — it returns the words, not their
//! count — so it is work, not a repeat.
//!
//! # Why it must sort
//!
//! The order is the observable part. `once_words` answers "how many"; this answers "which", and a
//! collection that preserved insertion order would still be correct for the first and wrong for
//! the second, which gives the reviewer something to check.

use tally::count;

#[test]
fn the_once_words_come_back_sorted() {
    // The order is part of the contract, not incidental: a `BTreeSet` gives it for free, and a
    // `Vec` of first-seen words does not.
    assert_eq!(
        count::once_words_sorted("gamma alpha beta gamma delta"),
        vec!["alpha".to_string(), "beta".to_string(), "delta".to_string()]
    );
}

#[test]
fn the_sorted_words_are_normalised_the_same_way_the_count_is() {
    // The count and the listing must agree about what "the same word" means, or a reader would
    // see `once: 2` next to a list of three entries.
    assert_eq!(
        count::once_words_sorted("The the, THE; other"),
        vec!["other".to_string()]
    );
    assert!(count::once_words_sorted("").is_empty());
}

#[test]
fn the_listing_agrees_with_the_count() {
    // The two functions describe one set twice. An implementation that got one right and the
    // other wrong would pass each test above in isolation and fail this one.
    for text in ["a b a", "a b c", "The, the. Other", "", "x"] {
        assert_eq!(
            count::once_words_sorted(text).len(),
            count::once_words(text),
            "the listing and the count disagree about `{text}`"
        );
    }
}
