//! Cutting tool output down to what fits, without losing the ends that matter.
//!
//! # Why head and tail, and not just head
//!
//! A build log starts with the command and ends with the verdict; the middle is the noise. A
//! diff starts with the file name and ends with the hunk that matters. Cutting only the head
//! throws away the exit status, which is the single most useful line in any tool result; cutting
//! only the tail throws away what the model asked about. So the kept text is
//! `head + separator + tail`, and the separator states how much was dropped.
//!
//! # Why the separator is not empty
//!
//! A silently concatenated head and tail reads as one continuous output. A model then reasons
//! about line `head.length + 1` as if it followed the last line of the head, which it does not.
//! The marker is inside the text the model sees, on its own line, and it names the byte count —
//! enough to tell "a little was cut" from "almost everything was cut".
//!
//! # What the hint may promise
//!
//! The full body goes to the content-addressed store, and the hint names its digest. It does
//! **not** say "read the artifact" as an instruction: the store lives outside the task
//! worktree, and the read policy confines reads to the workspace root, so a model told to read
//! it would be sent to a path it will be refused. The digest is there for `immortality log`
//! and for the human; the model is told to narrow the request instead, which is the one action
//! it can actually take.
//!
//! # Why the byte budget and not a line budget
//!
//! A line budget assumes lines of similar length. One minified bundle or one base64 blob breaks
//! it: a hundred lines can be a hundred megabytes. The budget is in bytes, and the cutting is
//! done at character boundaries so the result is always valid UTF-8.

use std::fmt;

use crate::artifacts::{ArtifactStore, digest_of};
use crate::journal::{Digest, digest_text};

/// The beginning of the line that stands in for what was dropped.
///
/// Starts with a Unicode ellipsis so that a reader scanning for the pattern does not have to
/// know the wording, and contains the marker word in ASCII so that `grep` for it works. The
/// counts follow it; this prefix is what a reader or a test can match without knowing them.
pub const SEPARATOR: &str = "⋯ [truncated:";

/// Output after the byte budget was applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Truncated {
    /// The text the model sees: the original, or head + separator + tail.
    pub text: String,
    /// The whole output, when it was too large for the prompt.
    pub artifact: Option<Digest>,
    /// Bytes dropped. Zero means the text is the original.
    pub dropped: usize,
    /// Bytes in the original.
    pub total: usize,
}

impl Truncated {
    /// Whether anything was cut.
    #[must_use]
    pub const fn was_cut(&self) -> bool {
        self.dropped > 0
    }
}

impl fmt::Display for Truncated {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

/// Cut `output` to at most `max_bytes` of visible text, spilling the rest to `store`.
///
/// Splits the budget evenly between head and tail so that neither end is privileged; a build log
/// and a diff want opposite ends, and half each is the only split that serves both without
/// knowing which tool produced the output.
///
/// # Errors
///
/// The artifact could not be written. Not papered over: the caller's alternative is to send a
/// truncated result with no way to recover the rest, which is worse than failing the step.
pub fn truncate_output(store: &ArtifactStore, output: &str, max_bytes: usize) -> crate::Result<Truncated> {
    let total = output.len();
    if total <= max_bytes {
        return Ok(Truncated {
            text: output.to_string(),
            artifact: None,
            dropped: 0,
            total,
        });
    }

    let artifact = store.put(output.as_bytes())?;
    // The separator's own length depends on the numbers inside it, and those depend on how much
    // is kept, so it is built after the split. Reserving its worst case up front would waste
    // budget on every cut; instead the head and tail each get half of what is left once the
    // marker is at its final size, which is one more than the longest possible marker only in
    // pathological cases where the numbers grow — impossible here, because `dropped` and `total`
    // are bounded by `total`.
    let marker = marker_for(total, total);
    let budget = max_bytes.saturating_sub(marker.len());
    let head_budget = budget / 2;
    let tail_budget = budget - head_budget;

    let head_end = floor_char_boundary(output, head_budget);
    let tail_start = ceil_char_boundary(output, total.saturating_sub(tail_budget));

    // A split that overlaps would duplicate text; a split that inverted would panic on a range.
    // Both can happen when the budget is smaller than the marker, and in that case the honest
    // answer is the marker alone.
    let (head, tail) = if head_end >= tail_start {
        ("", "")
    } else {
        (&output[..head_end], &output[tail_start..])
    };
    let dropped = total - head.len() - tail.len();
    let text = format!("{head}{}{tail}", marker_for(dropped, total));

    Ok(Truncated {
        text,
        artifact: Some(artifact),
        dropped,
        total,
    })
}

/// The hint appended to a truncated result, naming the artifact.
///
/// Separate from [`truncate_output`] because the caller decides whether to show it: an
/// observation goes to the model, a log line goes to a human, and they want different words.
#[must_use]
pub fn artifact_hint(digest: &Digest, total: usize) -> String {
    format!(
        "The full output ({total} bytes) is sha256 {}. Narrow the request with `offset`/`limit` \
         or a tighter `grep` pattern to see the rest.",
        digest_text(digest)
    )
}

/// The digest of an output that fits, without storing it.
///
/// `None` when the output was not cut, because a digest of something the model can already see
/// in full is a reference to nothing.
#[must_use]
pub fn digest_if_cut(truncated: &Truncated) -> Option<String> {
    truncated.artifact.as_ref().map(digest_text)
}

fn marker_for(dropped: usize, total: usize) -> String {
    format!("{SEPARATOR} {dropped} of {total} bytes not shown] ⋯")
}

/// The largest index `<= index` that is a character boundary.
const fn floor_char_boundary(text: &str, index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }
    let mut index = index;
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// The smallest index `>= index` that is a character boundary.
const fn ceil_char_boundary(text: &str, index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }
    let mut index = index;
    while index < text.len() && !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

/// The digest of `text`, for a caller that wants one without a store.
#[must_use]
pub fn digest_of_text(text: &str) -> Digest {
    digest_of(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, ArtifactStore) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = ArtifactStore::open(dir.path().join("artifacts")).expect("store");
        (dir, store)
    }

    #[test]
    fn output_that_fits_is_returned_untouched_and_not_stored() {
        let (_dir, store) = store();
        let result = truncate_output(&store, "short", 100).expect("truncate");
        assert_eq!(result.text, "short");
        assert!(!result.was_cut());
        assert_eq!(result.artifact, None);
        assert_eq!(result.dropped, 0);
        assert_eq!(result.total, 5);
    }

    #[test]
    fn the_ends_are_kept_and_the_middle_is_named() {
        let (_dir, store) = store();
        let output: String = (0..1000)
            .map(|i| char::from(b'a' + u8::try_from(i % 26).unwrap()))
            .collect();
        let result = truncate_output(&store, &output, 200).expect("truncate");

        assert!(result.was_cut());
        assert!(result.text.starts_with(&output[..10]), "the head survives");
        assert!(
            result.text.ends_with(&output[output.len() - 10..]),
            "the tail survives"
        );
        assert!(result.text.contains("[truncated:"), "{}", result.text);
        assert!(
            result.text.len() <= 200 + 64,
            "the marker is not counted in the budget"
        );
        assert_eq!(result.total, 1000);
        // Every byte is either shown or reported as dropped.
        assert_eq!(
            result.text.len(),
            result.total - result.dropped + marker_len(&result)
        );
    }

    /// The bytes the marker itself contributes, so a test can account for every byte.
    fn marker_len(result: &Truncated) -> usize {
        result.text.len() - (result.total - result.dropped)
    }

    #[test]
    fn the_whole_output_is_stored_under_its_digest() {
        let (_dir, store) = store();
        let output = "x".repeat(5000);
        let result = truncate_output(&store, &output, 100).expect("truncate");
        let digest = result.artifact.expect("an artifact");
        assert_eq!(store.read_text(&digest).expect("read"), output);
        assert_eq!(digest_text(&digest), digest_if_cut(&result).expect("a name"));
    }

    #[test]
    fn cutting_never_splits_a_utf8_character() {
        let (_dir, store) = store();
        // Three bytes per character, so every byte budget lands inside a character.
        let output = "я".repeat(500);
        let result = truncate_output(&store, &output, 100).expect("truncate");
        // The failure mode this catches is a panic from slicing at a non-boundary, which would
        // happen above. The positive assertion is stronger than "it did not panic": the kept
        // head and tail must be genuine substrings of the original, so no partial character was
        // pasted in, and their byte lengths must be whole multiples of the character width
        // (`я` is two bytes in UTF-8).
        let first = result.text.find('⋯').expect("a marker");
        let last = result.text.rfind('⋯').expect("a closing marker");
        let head = &result.text[..first];
        let tail = &result.text[last + '⋯'.len_utf8()..];
        assert!(output.starts_with(head), "head is not a prefix of the original");
        assert!(output.ends_with(tail), "tail is not a suffix of the original");
        assert_eq!(head.len() % 2, 0, "head ends mid-character");
        assert_eq!(tail.len() % 2, 0, "tail starts mid-character");
        assert!(result.was_cut());
    }

    #[test]
    fn a_budget_smaller_than_the_marker_yields_the_marker_alone() {
        let (_dir, store) = store();
        let output = "z".repeat(1000);
        let result = truncate_output(&store, &output, 1).expect("truncate");
        assert!(result.text.starts_with('⋯'), "{}", result.text);
        assert_eq!(result.dropped, 1000);
        assert!(result.artifact.is_some(), "the body is still recoverable");
    }

    #[test]
    fn the_hint_does_not_tell_the_model_to_read_a_path_it_cannot_reach() {
        // The store is outside the worktree and reads are confined to the workspace root, so a
        // hint saying "now read it" would be an instruction the model cannot follow.
        let digest = digest_of_text("body");
        let hint = artifact_hint(&digest, 42);
        assert!(hint.contains("sha256:"), "{hint}");
        assert!(hint.contains("42 bytes"), "{hint}");
        assert!(hint.contains("Narrow the request"), "{hint}");
        assert!(!hint.to_lowercase().contains("open the file"), "{hint}");
        assert!(!hint.contains(".agent/"), "{hint}");
    }

    #[test]
    fn a_digest_is_only_offered_for_output_that_was_cut() {
        let (_dir, store) = store();
        let small = truncate_output(&store, "small", 100).expect("truncate");
        assert_eq!(digest_if_cut(&small), None);
        let large = truncate_output(&store, &"q".repeat(1000), 100).expect("truncate");
        assert!(digest_if_cut(&large).is_some());
    }
}
