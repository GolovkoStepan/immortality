//! The review gate: what the reviewer is shown, and how its answer is read.
//!
//! # Why the prompt is built here
//!
//! The reviewer is the one model call in the system whose quality decides whether "done" means
//! anything. Two properties are load-bearing and both are easy to lose in a string built at the
//! call site: it sees **only** the diff and the definition of done (a reviewer that also saw the
//! conversation would be reviewing the model's intentions instead of its output), and it is never
//! given a tool (a reviewer that could edit code would be able to approve a change by making it).
//! [`prompt`] is the single place both are true, and its test asserts the shape of the message
//! rather than the exact words.
//!
//! # Why an unreadable answer is a rejection
//!
//! The reviewer is asked for JSON and a small local model will sometimes answer with prose, or
//! with JSON wrapped in a code fence, or with nothing. Treating any of those as approval would
//! make "the reviewer approved" mean "the reviewer produced a non-empty string", and a gate that
//! opens on noise is worse than no gate: it reports success. So the rule is closed by default —
//! only an explicit `approve` approves, and everything else is `request_changes` carrying what was
//! actually said. The next attempt then gets a real reason to work with.

use serde::{Deserialize, Serialize};

use super::{Diff, Finding, ReviewVerdict};

/// Everything the reviewer is shown. Nothing else reaches it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewRequest {
    /// The `DoD` statement: the only specification of intent the reviewer gets.
    pub statement: String,
    /// The change under review.
    pub diff: Diff,
}

/// System message: who the reviewer is and what it must not do.
///
/// No tool definitions are attached to the request that carries this, which is what enforces "it
/// cannot edit code" — a sentence in a prompt is a request, and the absence of tools is a fact.
pub const SYSTEM: &str = "\
You are reviewing a diff produced by another agent. You see only the definition of done and the \
diff itself. You have no tools and cannot modify anything.

Decide whether the diff satisfies the definition of done. Judge the diff, not the intentions \
behind it: if the change does not do what the definition says, request changes even if the \
writing is good.

Answer with JSON and nothing else, in exactly one of these two shapes:
{\"verdict\":\"approve\"}
{\"verdict\":\"request_changes\",\"issues\":[\"one concrete problem\",\"another\"]}

A request_changes verdict must name at least one concrete, fixable problem with a file and what \
is wrong. \"Not good enough\" is not a reason; \"src/parser.rs still drops the trailing comma\" \
is.";

/// The user message: the definition, then the diff.
///
/// Built as one string rather than a second message because the reviewer sees exactly two
/// messages, and a third would be a place for something to leak in later.
#[must_use]
pub fn prompt(request: &ReviewRequest) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    out.push_str("# Definition of done\n\n");
    out.push_str(request.statement.trim());
    out.push_str("\n\n# Diff\n\n");
    if request.diff.files.is_empty() {
        out.push_str("(the diff is empty: nothing changed)\n");
    } else {
        for file in &request.diff.files {
            let _ = writeln!(
                out,
                "- {} ({}{})",
                file.path.display(),
                file.status.as_str(),
                if file.binary { ", binary" } else { "" }
            );
        }
        out.push_str("\n```diff\n");
        out.push_str(request.diff.text.trim_end());
        out.push_str("\n```\n");
    }
    if request.diff.truncated {
        out.push_str(
            "\nThe diff was truncated before you saw it. You cannot approve a change you have \
             not seen in full: request changes and say the diff must be split.\n",
        );
    }
    out
}

/// The on-wire shape of an approval.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Approval {
    verdict: String,
}

/// The on-wire shape of a rejection.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Rejection {
    verdict: String,
    #[serde(default)]
    issues: Vec<String>,
}

/// Read the reviewer's answer.
///
/// Tolerant about the wrapper — a code fence, a sentence before the JSON — and strict about the
/// verdict: anything that is not an explicit `approve` is a rejection. `extra` is where the reader
/// puts what it could not use, so the journal can record the model's actual answer when the
/// verdict was unreadable.
#[must_use]
pub fn parse_review(text: &str) -> (ReviewVerdict, Option<String>) {
    let Some(json) = json_object(text) else {
        let said = collapse(text);
        return (
            ReviewVerdict::RequestChanges {
                issues: vec![unreadable(&said)],
            },
            Some(said),
        );
    };
    if let Ok(approval) = serde_json::from_str::<Approval>(&json) {
        if approval.verdict.trim().eq_ignore_ascii_case("approve") {
            return (ReviewVerdict::Approve, None);
        }
        return (
            ReviewVerdict::RequestChanges {
                issues: vec![format!(
                    "the reviewer answered `{}`, which is not one of the two verdicts",
                    approval.verdict.trim()
                )],
            },
            None,
        );
    }
    match serde_json::from_str::<Rejection>(&json) {
        Ok(rejection) if rejection.verdict.trim().eq_ignore_ascii_case("request_changes") => {
            let issues: Vec<String> = rejection
                .issues
                .into_iter()
                .map(|issue| issue.trim().to_string())
                .filter(|issue| !issue.is_empty())
                .collect();
            if issues.is_empty() {
                (
                    ReviewVerdict::RequestChanges {
                        issues: vec!["the reviewer requested changes without naming any".to_string()],
                    },
                    None,
                )
            } else {
                (ReviewVerdict::RequestChanges { issues }, None)
            }
        }
        Ok(_) | Err(_) => {
            let said = collapse(text);
            (
                ReviewVerdict::RequestChanges {
                    issues: vec![unreadable(&said)],
                },
                Some(said),
            )
        }
    }
}

/// The message for a reviewer that did not answer in the expected shape.
fn unreadable(said: &str) -> String {
    if said.is_empty() {
        "the reviewer returned nothing".to_string()
    } else {
        format!("the reviewer's answer was not a verdict: {said}")
    }
}

/// The first balanced JSON object in `text`, if there is one.
///
/// Scanned rather than searched for `{`/`}` with a regex: a diff in the prompt has braces in it,
/// and the reviewer may echo a fragment. Tracking string state is what keeps `{` inside a quoted
/// path from being read as a nesting level.
fn json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (index, ch) in text[start..].char_indices() {
        if in_string {
            match ch {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(text[start..=start + index].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// One line of what the reviewer said, short enough for a journal field.
fn collapse(text: &str) -> String {
    let joined = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if joined.chars().count() <= 400 {
        return joined;
    }
    let cut: String = joined.chars().take(400).collect();
    format!("{cut}…")
}

/// The findings the sanity gate would fail on, as one message for a reviewer.
///
/// Not shown to the reviewer: the sanity gate runs first and a red one stops the evaluation
/// before the review is paid for. The function exists for the journal and for the next attempt's
/// brief, which want the same sentences.
#[must_use]
pub fn describe(findings: &[Finding]) -> Vec<String> {
    findings.iter().map(Finding::observation).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::{ChangedFile, FileStatus};
    use std::path::PathBuf;

    fn request() -> ReviewRequest {
        ReviewRequest {
            statement: "trailing commas parse".to_string(),
            diff: Diff {
                files: vec![ChangedFile::new("src/parser.rs", FileStatus::Modified)],
                text: "+++ b/src/parser.rs\n+    parse_comma();".to_string(),
                truncated: false,
            },
        }
    }

    #[test]
    fn the_prompt_carries_the_definition_and_the_diff() {
        let text = prompt(&request());
        assert!(text.contains("trailing commas parse"), "{text}");
        assert!(text.contains("parse_comma"), "{text}");
        assert!(text.contains("src/parser.rs"), "{text}");
    }

    #[test]
    fn the_system_message_says_the_reviewer_cannot_edit() {
        // The absence of tools enforces it; the sentence is what makes the model not try.
        assert!(SYSTEM.contains("no tools"), "{SYSTEM}");
        assert!(SYSTEM.contains("request_changes"), "{SYSTEM}");
    }

    #[test]
    fn an_empty_diff_is_named_instead_of_rendered() {
        // Otherwise the reviewer sees a heading followed by nothing and may infer that the diff
        // was fine.
        let mut request = request();
        request.diff.files.clear();
        request.diff.text.clear();
        assert!(prompt(&request).contains("nothing changed"));
    }

    #[test]
    fn a_truncated_diff_tells_the_reviewer_it_cannot_approve() {
        let mut request = request();
        request.diff.truncated = true;
        assert!(prompt(&request).contains("must be split"));
    }

    #[test]
    fn an_explicit_approval_is_the_only_approval() {
        assert_eq!(parse_review(r#"{"verdict":"approve"}"#).0, ReviewVerdict::Approve);
        assert_eq!(parse_review(r#"{"verdict":"Approve"}"#).0, ReviewVerdict::Approve);
    }

    #[test]
    fn a_fenced_json_answer_is_read() {
        let text = "Sure. Here is my verdict:\n```json\n{\"verdict\":\"approve\"}\n```";
        assert_eq!(parse_review(text).0, ReviewVerdict::Approve);
    }

    #[test]
    fn a_request_for_changes_keeps_the_issues() {
        let text = r#"{"verdict":"request_changes","issues":["src/a.rs drops the comma","no test"]}"#;
        assert_eq!(
            parse_review(text).0,
            ReviewVerdict::RequestChanges {
                issues: vec!["src/a.rs drops the comma".to_string(), "no test".to_string()]
            }
        );
    }

    #[test]
    fn prose_is_a_rejection_rather_than_an_approval() {
        // The failure mode this closes: "the reviewer said something, so it must be fine".
        let text = "The change seems reasonable overall, though I would like more tests.";
        let (verdict, said) = parse_review(text);
        assert!(matches!(verdict, ReviewVerdict::RequestChanges { .. }));
        assert!(said.is_some_and(|said| said.contains("reasonable")));
    }

    #[test]
    fn silence_is_a_rejection() {
        let (verdict, said) = parse_review("   ");
        match verdict {
            ReviewVerdict::RequestChanges { issues } => {
                assert!(issues[0].contains("returned nothing"), "{issues:?}");
            }
            ReviewVerdict::Approve => panic!("silence was read as approval"),
        }
        assert!(said.is_some_and(|said| said.is_empty()));
    }

    #[test]
    fn an_unknown_verdict_word_is_a_rejection_that_names_it() {
        let text = r#"{"verdict":"lgtm"}"#;
        match parse_review(text).0 {
            ReviewVerdict::RequestChanges { issues } => {
                assert!(issues[0].contains("lgtm"), "{issues:?}");
            }
            ReviewVerdict::Approve => panic!("an unknown word was read as approval"),
        }
    }

    #[test]
    fn an_empty_issue_list_is_still_a_rejection() {
        let text = r#"{"verdict":"request_changes","issues":[]}"#;
        match parse_review(text).0 {
            ReviewVerdict::RequestChanges { issues } => {
                assert!(issues[0].contains("without naming any"), "{issues:?}");
            }
            ReviewVerdict::Approve => panic!("an empty rejection was read as approval"),
        }
    }

    #[test]
    fn a_brace_inside_a_quoted_string_does_not_end_the_object() {
        // The reviewer may echo the diff path; a naive brace counter would cut the JSON short.
        let text = r#"{"verdict":"request_changes","issues":["src/{weird}.rs is wrong"]}"#;
        match parse_review(text).0 {
            ReviewVerdict::RequestChanges { issues } => {
                assert!(issues[0].contains("{weird}"), "{issues:?}");
            }
            ReviewVerdict::Approve => panic!("a quoted brace confused the parser"),
        }
    }

    #[test]
    fn a_very_long_prose_answer_is_collapsed_for_the_journal() {
        let text = "word ".repeat(500);
        let (_, said) = parse_review(&text);
        let said = said.expect("the raw answer");
        assert!(said.chars().count() <= 401, "{}", said.chars().count());
        assert!(said.ends_with('…'));
    }

    #[test]
    fn findings_are_described_as_sentences() {
        let sentences = describe(&[Finding::Binary {
            path: PathBuf::from("a.png"),
        }]);
        assert_eq!(sentences, vec!["`a.png` is a binary file".to_string()]);
    }
}
