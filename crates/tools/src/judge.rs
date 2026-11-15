//! The judge: one model reply about one command, before that command runs.
//!
//! # What this is for
//!
//! The policy gate is a table, and a table has a hole: a program nobody listed arrives as
//! [`crate::Effect::External`] and becomes a question for a human. Measured on live runs, that
//! hole is hit immediately — `bundle exec rspec` is not in any table, and every attempt that
//! tried it stopped and waited. The judge answers that one question without a human.
//!
//! It answers **one** question and no other. Every other way to reach `External` is a statement
//! by the operator — a denied program, a network subcommand, a path outside the worktree — and a
//! model allowed to revisit those is a policy bypass with extra steps. That boundary is
//! [`crate::classify::External::is_a_question`], which is a variant match and not a phrase, so
//! rewording a message cannot move it.
//!
//! # Why failure is `escalate` and not `deny`
//!
//! `deny` looks safer and is the worse answer. A task that hits a dead end with no explanation
//! starts looking for a way round the gate; "I could not decide" is a different fact from
//! "you may not", and only one of them is true when the backend timed out. So a timeout, an
//! unreadable reply, or a missing model all become a question for the human, which is where the
//! attempt would have been without the judge at all.
//!
//! # Why the answer is parsed strictly
//!
//! The judge's reply is model output, so it is untrusted in the way every model output is: it can
//! be prose, it can be fenced, it can be `{"verdict":"maybe"}`. Only an explicit
//! `allow`/`deny`/`escalate` counts, and anything else is a failure of the judge rather than a
//! fourth verdict. Guessing `allow` out of a paragraph is the one error that cannot be undone.

use std::sync::Arc;

/// What the judge decided about one command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Judgement {
    /// Run it. Paths are still checked by the gate; this waives only the "unknown program" part.
    Allow,
    /// Do not run it, and say why in the model's own words.
    Deny(String),
    /// Ask the human. Also the answer to every failure of the judge itself.
    Escalate(String),
}

/// A model reply, as a single string.
///
/// A trait rather than a concrete client so `tools` does not depend on the agent loop: the daemon
/// owns the model and hands it in through a slot, and a test hands in a queue of replies. It
/// takes the prompt and returns the answer, which is all a judge needs — no tools, no history, no
/// streaming.
pub trait Judge: Send + Sync + std::fmt::Debug {
    /// One reply, with no tools attached.
    fn judge(&self, prompt: &str) -> Result<String, String>;
}

/// The system message. Fixed, so the test does not have to duplicate the wording.
pub const SYSTEM: &str = "\
You decide whether one shell command may run. You have no tools and cannot look at the repository.

Answer with one JSON object and nothing else: {\"verdict\":\"allow\"} or {\"verdict\":\"deny\",\
\"why\":\"one sentence\"} or {\"verdict\":\"escalate\",\"why\":\"one sentence\"}.

Rules:
- `allow` when the command does the kind of work the task describes and its effect stays inside \
the worktree. Ordinary build and test commands are allowed.
- `deny` when it reaches the network, writes outside the worktree, escalates privileges, or is \
unrelated to the task.
- `escalate` when you cannot tell. That is not a failure — a human reads your reason and decides.
- `why` is read by a model that has none of this conversation, so name the fact, not the feeling.";

/// Build the prompt for one command.
#[must_use]
pub fn prompt(command: &str, worktree: &str, goal: &str) -> String {
    let mut out = String::new();
    out.push_str("# Command\n\n```sh\n");
    out.push_str(command.trim());
    out.push_str("\n```\n\n# Worktree\n\n");
    out.push_str(worktree);
    out.push_str("\n\n# Task\n\n");
    out.push_str(goal.trim());
    out.push('\n');
    out
}

/// Read a judge reply, closed by default.
///
/// Anything that is not an explicit verdict is an error rather than a third guess: the caller
/// turns an error into `Escalate`, which is the only safe direction.
///
/// # Errors
/// The reply is not JSON, has no `verdict`, has an unknown verdict, or is a bare `deny` with no
/// reason — a denial the model cannot explain is a denial the next step cannot act on.
pub fn parse_reply(reply: &str) -> Result<Judgement, String> {
    let text = strip_fence(reply);
    let value: serde_json::Value =
        serde_json::from_str(text.trim()).map_err(|error| format!("not JSON: {error}"))?;
    let verdict = value
        .get("verdict")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "no `verdict` field".to_string())?
        .trim()
        .to_ascii_lowercase();
    let why = value
        .get("why")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    match verdict.as_str() {
        "allow" => Ok(Judgement::Allow),
        "deny" => {
            if why.is_empty() {
                return Err("`deny` with no reason".to_string());
            }
            Ok(Judgement::Deny(why))
        }
        "escalate" => Ok(Judgement::Escalate(if why.is_empty() {
            "the judge could not decide".to_string()
        } else {
            why
        })),
        other => Err(format!("unknown verdict `{other}`")),
    }
}

/// Strip one surrounding markdown fence, which models add despite being told not to.
///
/// Only a fence that opens the reply and closes it: removing fences that appear mid-text would
/// turn a command containing the three-backtick sequence into a different document.
fn strip_fence(reply: &str) -> &str {
    let text = reply.trim();
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    // Skip the language tag on the opening fence.
    let Some(body) = rest.split_once('\n').map(|(_, body)| body) else {
        return text;
    };
    body.strip_suffix("```").map_or(text, str::trim_end)
}

/// A judge that answers with a fixed sequence of replies, for tests.
///
/// Not `#[cfg(test)]`: the daemon's own tests and the escalation path both need a judge whose
/// answer is known, and a test-only constructor in `tools` would leave `daemon` unable to build
/// one.
#[derive(Debug, Default)]
pub struct Scripted {
    replies: std::sync::Mutex<std::collections::VecDeque<Result<String, String>>>,
    prompts: std::sync::Mutex<Vec<String>>,
}

impl Scripted {
    /// A judge that answers with these replies in order.
    #[must_use]
    pub fn new(replies: Vec<Result<String, String>>) -> Arc<Self> {
        Arc::new(Self {
            replies: std::sync::Mutex::new(replies.into()),
            prompts: std::sync::Mutex::new(Vec::new()),
        })
    }

    /// The prompts it was asked, in order.
    #[must_use]
    pub fn prompts(&self) -> Vec<String> {
        self.prompts.lock().map(|guard| guard.clone()).unwrap_or_default()
    }
}

impl Judge for Scripted {
    fn judge(&self, prompt: &str) -> Result<String, String> {
        if let Ok(mut seen) = self.prompts.lock() {
            seen.push(prompt.to_string());
        }
        self.replies
            .lock()
            .ok()
            .and_then(|mut queue| queue.pop_front())
            // Running out is a failure of the test's own script, and `Escalate` is what a real
            // failure produces — so a test that under-scripts its replies fails its own assertion
            // rather than silently allowing a command.
            .unwrap_or_else(|| Err("the scripted judge ran out of replies".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_explicit_allow_is_an_allow() {
        assert_eq!(
            parse_reply(r#"{"verdict":"allow"}"#).expect("parse"),
            Judgement::Allow
        );
    }

    #[test]
    fn a_deny_carries_its_reason() {
        assert_eq!(
            parse_reply(r#"{"verdict":"deny","why":"it writes /etc/passwd"}"#).expect("parse"),
            Judgement::Deny("it writes /etc/passwd".into())
        );
    }

    #[test]
    fn prose_is_not_a_verdict() {
        // The one error that cannot be undone: reading a paragraph as permission. A judge that
        // says something unexpected must not be interpreted charitably.
        for reply in [
            "Sure, that looks fine to me.",
            r#"{"verdict":"probably"}"#,
            r#"{"verdict":"allow""#,
            r#"{"answer":"allow"}"#,
            "",
        ] {
            assert!(parse_reply(reply).is_err(), "`{reply}` was read as a verdict");
        }
    }

    #[test]
    fn a_deny_without_a_reason_is_not_a_denial() {
        // The model has to act on the reason. A denial with nothing in it is a dead end, which is
        // the failure mode `escalate` exists to avoid.
        assert!(parse_reply(r#"{"verdict":"deny"}"#).is_err());
        assert!(parse_reply(r#"{"verdict":"deny","why":"   "}"#).is_err());
    }

    #[test]
    fn an_escalate_without_a_reason_still_escalates() {
        // Unlike `deny`, the direction is already safe, and losing the explanation is not a reason
        // to turn a question into a refusal.
        assert_eq!(
            parse_reply(r#"{"verdict":"escalate"}"#).expect("parse"),
            Judgement::Escalate("the judge could not decide".into())
        );
    }

    #[test]
    fn a_fence_around_the_json_is_tolerated_but_not_required() {
        // Models fence JSON even when told not to. The fence is stripped only when it surrounds
        // the whole reply, so a command containing backticks is not rewritten.
        assert_eq!(
            parse_reply("```json\n{\"verdict\":\"allow\"}\n```").expect("parse"),
            Judgement::Allow
        );
        assert_eq!(
            parse_reply("{\"verdict\":\"allow\"}").expect("parse"),
            Judgement::Allow
        );
    }

    #[test]
    fn the_prompt_names_the_command_and_the_worktree() {
        let text = prompt("bundle exec rspec", "/work/tree", "make the suite pass");
        assert!(text.contains("bundle exec rspec"), "{text}");
        assert!(text.contains("/work/tree"), "{text}");
        assert!(text.contains("make the suite pass"), "{text}");
    }

    #[test]
    fn the_system_message_forbids_tools_and_asks_for_one_object() {
        // These are the two properties the parser relies on, so they are asserted rather than
        // assumed: a system message that invited prose would make strict parsing a trap.
        assert!(SYSTEM.contains("no tools"), "{SYSTEM}");
        assert!(SYSTEM.contains("\"verdict\""), "{SYSTEM}");
    }
}
