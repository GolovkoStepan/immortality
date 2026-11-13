//! Behaviour the `line-count` fix must deliver.
//!
//! Red in the repository as committed: the fixture ships the defect, and the gate is
//! `cargo test --offline`, so a subtask that claims success without touching `count::lines` runs a
//! test that still fails.

use tally::count;

#[test]
fn a_trailing_newline_does_not_add_a_line() {
    // `split` yields an empty field after the final separator, so every file that ends with a
    // newline — which is every file a text editor writes — reads one line too many.
    assert_eq!(count::lines("a\nb\n"), 2);
}

#[test]
fn the_two_spellings_of_a_two_line_file_agree() {
    // The sharper form of the same rule: the presence of a final newline must not change the
    // count, because it does not change the number of lines.
    assert_eq!(count::lines("a\nb\n"), count::lines("a\nb"));
}

#[test]
fn one_line_with_a_newline_is_one_line() {
    assert_eq!(count::lines("a\n"), 1);
    assert_eq!(count::lines(""), 0);
}
