//! Behaviour the `dedupe` refactor must preserve and the one thing it must add.
//!
//! The refactor's requirement is that the *existing* rules keep their answers after the container
//! change, so this file pins them from the outside, where an edited implementation cannot quietly
//! satisfy it. It is red as committed because `unique_words_sorted` does not exist yet — and that
//! absence is what makes the gate a gate: a test for behaviour that already works would pass on
//! the untouched repository, and a subtask could close without changing anything.

use tally::count;

#[test]
fn distinct_words_are_still_counted_case_insensitively() {
    // The pre-existing behaviour, pinned from outside the module that implements it. A refactor
    // that changed normalisation would still pass the module's own tests (which a careless
    // refactor would have edited too) but fail here.
    assert_eq!(count::unique_words("The the THE, the."), 1);
    assert_eq!(count::unique_words("alpha beta alpha"), 2);
    assert_eq!(count::unique_words("alpha beta gamma"), 3);
    assert_eq!(count::unique_words(""), 0);
}

#[test]
fn the_sorted_words_are_unique_and_in_order() {
    // Sorted output is the capability a set buys for free and a `Vec` plus `contains` does not.
    // Asking for it is what makes "use a set" an observable requirement rather than a comment.
    assert_eq!(
        count::unique_words_sorted("beta ALPHA beta gamma"),
        vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]
    );
    assert_eq!(count::unique_words_sorted("The the"), vec!["the".to_string()]);
    assert!(count::unique_words_sorted("").is_empty());
}
