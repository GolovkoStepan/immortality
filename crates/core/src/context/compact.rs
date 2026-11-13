//! Compaction: turning a long history into a summary that survives a restart.
//!
//! # Why a fixed template rather than a free-form summary
//!
//! A model asked to "summarise the conversation" writes prose that reads well and loses the two
//! things that matter: which approaches were *rejected* and where the run stopped. Both are
//! exactly what a model re-derives — and re-proposes — when they are missing, which is the
//! commonest way a long run burns a day. So the output shape is fixed, the sections are
//! mandatory, and a summary missing a section is a failure rather than a summary.
//!
//! # Why `Rejected` is never allowed to be absent
//!
//! "Nothing was rejected" is a real answer, and an omitted section is indistinguishable from a
//! model that skipped it. The parser accepts an explicit `- none` marker, so the section is
//! always present and "none" can be told from "not filled in".
//!
//! # Why the summary is written to state, not just into the context
//!
//! The design is explicit (`docs/design.md` §8): the result is written into state. A summary
//! that lives only in the next prompt is lost on the next restart, and the recovery pass would
//! replay the raw history it was supposed to replace. The compaction becomes part of the layer
//! set of the *next* attempt — never an edit to the current attempt's stable prefix, which
//! would invalidate the cache.

use serde::{Deserialize, Serialize};

use crate::domain::Timestamp;

/// Sections every compaction must contain, in the order they must appear.
///
/// ASCII in the output because a model reproducing Cyrillic headings is a coin flip, and the
/// parser has to be exact. The human-facing wording is in the docs.
pub const REQUIRED_SECTIONS: &[&str] = &[
    "Facts",
    "Decisions",
    "Open questions",
    "Rejected",
    "Where we stopped",
];

/// The instruction for the compaction call.
///
/// `enable_thinking: false` is not part of this text: it is a request parameter, and a prompt
/// that also argued for it would be doing the client's job.
pub const INSTRUCTION: &str = "\
Summarise the work below for a fresh context that will continue it.

Use exactly these five sections, in this order, with the headings verbatim:

## Facts
What is true about the codebase now. One fact per line, each one checkable.

## Decisions
What was decided and why, one per line. Include the alternative that lost.

## Open questions
What is still unknown or unresolved, one per line.

## Rejected
Approaches that were tried or considered and did not work, one per line, with the reason. \
Write `- none` if nothing was rejected. Never omit this section.

## Where we stopped
The exact state of the work: what is done, what is in progress, and what the next step is.

Be specific. A summary that says \"made progress\" is worthless; name files, commands and \
outcomes. Do not invent anything that is not in the text below.";

/// Marker for an explicitly empty `Rejected` section.
pub const NONE_MARKER: &str = "none";

/// A parsed compaction summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Compaction {
    pub facts: Vec<String>,
    pub decisions: Vec<String>,
    pub open_questions: Vec<String>,
    /// Approaches that did not work. Never empty when parsed: `- none` becomes a single
    /// [`NONE_MARKER`] line, so "nothing was rejected" is recorded explicitly.
    pub rejected: Vec<String>,
    pub next_step: String,
    /// How many history entries this summary replaced, for the journal record.
    pub replaced_entries: usize,
    pub at: Timestamp,
}

/// Why a model's answer is not a usable compaction.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CompactionError {
    #[error("the summary is missing the `{section}` section")]
    MissingSection { section: String },

    #[error("the `{section}` section appears out of order")]
    OutOfOrder { section: String },

    #[error("the `{section}` section is empty; write `- none` if there is nothing")]
    EmptySection { section: String },

    #[error("the summary has no `Where we stopped` content")]
    NoNextStep,
}

impl Compaction {
    /// Parse a model's answer, refusing anything the template requires and did not get.
    ///
    /// # Errors
    /// A section is missing, out of order, empty, or the closing section is blank.
    pub fn parse(answer: &str, replaced_entries: usize, at: Timestamp) -> Result<Self, CompactionError> {
        let found = split_sections(answer)?;
        let rejected = bullets(section_text(answer, &found, 3));
        if rejected.is_empty() {
            return Err(CompactionError::EmptySection {
                section: REQUIRED_SECTIONS[3].to_string(),
            });
        }
        let next_step = section_text(answer, &found, 4).trim().to_string();
        if next_step.is_empty() {
            return Err(CompactionError::NoNextStep);
        }
        Ok(Self {
            facts: bullets(section_text(answer, &found, 0)),
            decisions: bullets(section_text(answer, &found, 1)),
            open_questions: bullets(section_text(answer, &found, 2)),
            rejected,
            next_step,
            replaced_entries,
            at,
        })
    }

    /// Whether the model explicitly said nothing was rejected.
    #[must_use]
    pub fn rejected_is_none(&self) -> bool {
        self.rejected.len() == 1 && self.rejected[0].eq_ignore_ascii_case(NONE_MARKER)
    }

    /// The summary as it is written into state and into the next prompt.
    ///
    /// Round-trips through [`Self::parse`] — a test asserts it — so what is written back is the
    /// same shape as what was just produced.
    #[must_use]
    pub fn render(&self) -> String {
        use std::fmt::Write as _;

        let mut out = String::new();
        for (heading, lines) in [
            (REQUIRED_SECTIONS[0], &self.facts),
            (REQUIRED_SECTIONS[1], &self.decisions),
            (REQUIRED_SECTIONS[2], &self.open_questions),
            (REQUIRED_SECTIONS[3], &self.rejected),
        ] {
            let _ = writeln!(out, "## {heading}");
            if lines.is_empty() {
                // Only reachable for a `Compaction` built by hand; `parse` refuses this. The
                // marker keeps the rendered form parseable, which is the property that matters.
                let _ = writeln!(out, "- {NONE_MARKER}");
            } else {
                for line in lines {
                    // A line that already starts with a bullet would render as `- - x`.
                    let _ = writeln!(out, "- {}", line.trim_start_matches('-').trim());
                }
            }
            out.push('\n');
        }
        let _ = writeln!(out, "## {}", REQUIRED_SECTIONS[4]);
        out.push_str(self.next_step.trim_end());
        out.push('\n');
        out
    }
}

/// The instruction and the text to summarise, as separate pieces.
///
/// Separate because the caller puts them in two messages, and joining them here would bake a
/// message layout into the core crate.
#[must_use]
pub const fn request(history: &str) -> (&'static str, &str) {
    (INSTRUCTION, history)
}

/// Where each required heading starts, in document order.
///
/// Errors when one is missing or when the positions are not increasing. The second check is not
/// paranoia: if the model shuffled the sections, the slicing in [`section_text`] would attribute
/// one section's text to another's heading, and the summary would be quietly wrong.
fn split_sections(answer: &str) -> Result<Vec<usize>, CompactionError> {
    let mut positions = Vec::with_capacity(REQUIRED_SECTIONS.len());
    for name in REQUIRED_SECTIONS {
        let needle = format!("## {name}");
        let Some(position) = answer.find(&needle) else {
            return Err(CompactionError::MissingSection {
                section: (*name).to_string(),
            });
        };
        positions.push(position);
    }
    for (index, pair) in positions.windows(2).enumerate() {
        let [earlier, later] = pair else { continue };
        if earlier >= later {
            return Err(CompactionError::OutOfOrder {
                section: REQUIRED_SECTIONS
                    .get(index + 1)
                    .copied()
                    .unwrap_or("?")
                    .to_string(),
            });
        }
    }
    Ok(positions)
}

/// The body of section `index`: text after its heading up to the next heading or the end.
///
/// `index` and the headings are checked against each other in tests; the arithmetic cannot fail
/// because every position in `found` lies within `answer` and the headings are ASCII.
fn section_text<'a>(answer: &'a str, found: &[usize], index: usize) -> &'a str {
    let Some(&start) = found.get(index) else {
        return "";
    };
    let heading = REQUIRED_SECTIONS.get(index).copied().unwrap_or("");
    let body_start = start + heading.len() + "## ".len();
    let body_end = found.get(index + 1).copied().unwrap_or(answer.len());
    if body_start >= body_end {
        return "";
    }
    answer.get(body_start..body_end).unwrap_or("")
}

fn bullets(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| line.trim_start_matches('-').trim().to_string())
        .filter(|line| !line.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at() -> Timestamp {
        crate::time::now()
    }

    fn good() -> String {
        "\
## Facts
- The socket is mode 0600.
- `serve` holds `daemon.lock` before binding.

## Decisions
- Take the lock before `bind`, because `bind` deletes the socket.
- Rejected alternative: bind first and check the error.

## Open questions
- Whether `flock` is needed on NFS.

## Rejected
- `crossterm`: a dependency for six escape codes.
- `lock()` instead of `try_lock()`: hangs instead of reporting.

## Where we stopped
The command builds and 1121 tests pass. Next: wire the context assembler into the runner."
            .to_string()
    }

    #[test]
    fn a_complete_answer_parses_into_every_section() {
        let compaction = Compaction::parse(&good(), 7, at()).expect("parse");
        assert_eq!(compaction.facts.len(), 2);
        assert_eq!(compaction.decisions.len(), 2);
        assert_eq!(compaction.open_questions.len(), 1);
        assert_eq!(compaction.rejected.len(), 2, "rejected ideas survive");
        assert!(compaction.next_step.contains("Next:"));
        assert_eq!(compaction.replaced_entries, 7);
        assert!(!compaction.rejected_is_none());
    }

    #[test]
    fn a_missing_section_is_refused() {
        let without = good().replace("## Open questions", "## Other");
        let error = Compaction::parse(&without, 0, at()).expect_err("must refuse");
        assert!(matches!(error, CompactionError::MissingSection { .. }), "{error}");
        assert!(error.to_string().contains("Open questions"), "{error}");
    }

    #[test]
    fn a_missing_rejected_section_is_refused_rather_than_assumed_empty() {
        // The single most damaging omission: a summary without it is what makes the next
        // attempt re-propose a rejected idea.
        let good_text = good();
        let lines: Vec<&str> = good_text
            .lines()
            .filter(|line| *line != "## Rejected")
            .filter(|line| !line.starts_with("- `crossterm`") && !line.starts_with("- `lock()`"))
            .collect();
        let error = Compaction::parse(&lines.join("\n"), 0, at()).expect_err("must refuse");
        assert!(error.to_string().contains("Rejected"), "{error}");
    }

    #[test]
    fn shuffled_sections_are_refused() {
        // Slice order is implied by heading order; a shuffled answer would attribute text to
        // the wrong heading instead of failing.
        let shuffled = "\
## Decisions
- d

## Facts
- f

## Open questions
- q

## Rejected
- r

## Where we stopped
next";
        let error = Compaction::parse(shuffled, 0, at()).expect_err("must refuse");
        assert!(matches!(error, CompactionError::OutOfOrder { .. }), "{error}");
    }

    #[test]
    fn an_explicit_none_is_distinguishable_from_an_empty_section() {
        let answer = good().replace(
            "- `crossterm`: a dependency for six escape codes.\n- `lock()` instead of `try_lock()`: hangs instead of reporting.",
            "- none",
        );
        let compaction = Compaction::parse(&answer, 0, at()).expect("parse");
        assert!(compaction.rejected_is_none());
        assert_eq!(compaction.rejected, vec![NONE_MARKER.to_string()]);
    }

    #[test]
    fn a_blank_rejected_section_is_an_error_not_an_empty_list() {
        let answer = good().replace(
            "- `crossterm`: a dependency for six escape codes.\n- `lock()` instead of `try_lock()`: hangs instead of reporting.",
            "   ",
        );
        let error = Compaction::parse(&answer, 0, at()).expect_err("must refuse");
        assert!(matches!(error, CompactionError::EmptySection { .. }), "{error}");
    }

    #[test]
    fn a_blank_closing_section_is_refused() {
        let mut answer = good();
        // Keep the heading, drop everything under it, so the failure is a blank section rather
        // than a missing one.
        let cut = answer.find("## Where we stopped").expect("heading");
        answer.truncate(cut + "## Where we stopped".len());
        let error = Compaction::parse(&answer, 0, at()).expect_err("must refuse");
        assert!(matches!(error, CompactionError::NoNextStep), "{error}");
    }

    #[test]
    fn rendering_round_trips_so_state_and_prompt_agree() {
        let first = Compaction::parse(&good(), 3, at()).expect("parse");
        let rendered = first.render();
        let second = Compaction::parse(&rendered, 3, first.at).expect("re-parse its own output");
        assert_eq!(first, second);
    }

    #[test]
    fn rendering_a_hand_built_empty_summary_stays_parseable() {
        let compaction = Compaction {
            facts: Vec::new(),
            decisions: Vec::new(),
            open_questions: Vec::new(),
            rejected: vec!["none".to_string()],
            next_step: "nothing".to_string(),
            replaced_entries: 0,
            at: at(),
        };
        let parsed = Compaction::parse(&compaction.render(), 0, compaction.at).expect("parse");
        assert!(parsed.rejected_is_none());
    }

    #[test]
    fn the_instruction_names_every_required_section() {
        // The prompt and the parser must not drift apart: a section the instruction does not ask
        // for is one the model will not produce, and every compaction would fail.
        for section in REQUIRED_SECTIONS {
            assert!(
                INSTRUCTION.contains(&format!("## {section}")),
                "missing {section}"
            );
        }
    }

    #[test]
    fn the_request_returns_the_text_unchanged() {
        let (instruction, body) = request("the history");
        assert_eq!(instruction, INSTRUCTION);
        assert_eq!(body, "the history");
    }
}
