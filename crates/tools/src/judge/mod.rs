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

pub mod cache;

pub use cache::{Cache, Verdict, may_judge, policy_fingerprint, signature};

/// The map key for one signature under one policy.
///
/// A function rather than a `format!` at each call site: the cache and the journal must agree on
/// the key exactly, and two `format!` calls that drift would make every lookup a miss — which is
/// invisible, because a miss is not an error.
#[must_use]
pub fn key(signature: &str, policy: &str) -> String {
    format!("{policy}\u{1}{signature}")
}

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

impl Judgement {
    /// The wire name, as the journal stores it.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Deny(_) => "deny",
            Self::Escalate(_) => "escalate",
        }
    }

    /// The reason, for `deny` and `escalate`.
    #[must_use]
    pub fn why(&self) -> Option<&str> {
        match self {
            Self::Allow => None,
            Self::Deny(why) | Self::Escalate(why) => Some(why),
        }
    }
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

/// A judge that remembers its own answers, so one question is paid for once.
///
/// Wraps another [`Judge`] rather than being one, which is what lets the model, the scripted test
/// double and any future judge share this behaviour instead of each reimplementing it.
///
/// # Why it is a `Judge` and also hands its verdicts out
///
/// The gate calls [`Judge::judge`] with a prompt, and the prompt is the only thing it has. The cache
/// needs the command's *signature*, which is computed from the command, so the wrapper computes it
/// from the prompt's own `# Command` section — the two are built together by [`prompt`], so the
/// wrapper cannot disagree with what the model was asked.
///
/// The verdicts are pulled out with [`Recorded::take_fresh`] by whoever holds the journal. The gate
/// cannot write them itself: `run_tool` holds `&mut Journal` for the whole call, and the journal is
/// single-writer by construction.
#[derive(Debug)]
pub struct Recorded {
    inner: Arc<dyn Judge>,
    cache: Cache,
    /// The policy fingerprint, fixed at construction.
    ///
    /// Fixed rather than read per call because the gate's policy is fixed for the run: a policy
    /// that could change mid-run would need every cached verdict reconsidered, and nothing in the
    /// daemon can change it.
    policy: String,
}

impl Recorded {
    /// A caching judge over `inner`, for the policy fingerprinted by `policy`.
    #[must_use]
    pub fn new(inner: Arc<dyn Judge>, policy: impl Into<String>) -> Self {
        Self {
            inner,
            cache: Cache::default(),
            policy: policy.into(),
        }
    }

    /// The same wrapper, seeded with what the journal remembers.
    #[must_use]
    pub fn seeded(inner: Arc<dyn Judge>, policy: impl Into<String>, known: Vec<Verdict>) -> Self {
        Self {
            inner,
            cache: Cache::new(known),
            policy: policy.into(),
        }
    }

    /// The answers given during this process that are not yet in the journal.
    #[must_use]
    pub fn take_fresh(&self) -> Vec<Verdict> {
        self.cache.take_fresh()
    }

    /// The cache, for a caller that wants to inspect it.
    #[must_use]
    pub const fn cache(&self) -> &Cache {
        &self.cache
    }
}

impl Judge for Recorded {
    fn judge(&self, prompt: &str) -> Result<String, String> {
        let command = command_of(prompt);
        let signature = signature(&command);
        if let Some(judgement) = self.cache.get(&signature, &self.policy) {
            return Ok(render(&judgement));
        }
        let reply = self.inner.judge(prompt)?;
        // A reply nobody can read is not cached, and not remembered: it is the failure that turns
        // into an escalation at the call site, and caching "the judge was confused" would make one
        // bad reply permanent.
        if let Ok(judgement) = parse_reply(&reply) {
            self.cache.insert(Verdict {
                signature,
                policy: self.policy.clone(),
                command,
                judgement,
            });
        }
        Ok(reply)
    }
}

/// Render a cached judgement back into the reply shape [`parse_reply`] reads.
///
/// Not the model's original words: `allow` has none, and a cached `deny` is not stored at all. This
/// is why the round trip through text is safe — the only verdict the cache holds is the one that
/// carries no free-form text.
fn render(judgement: &Judgement) -> String {
    let mut body = serde_json::Map::new();
    body.insert(
        "verdict".to_string(),
        serde_json::Value::String(judgement.as_str().to_string()),
    );
    if let Some(why) = judgement.why() {
        body.insert("why".to_string(), serde_json::Value::String(why.to_string()));
    }
    serde_json::Value::Object(body).to_string()
}

/// The command a prompt asks about, read back out of the section [`prompt`] wrote.
///
/// The prompt is the only input a judge gets, and this is the inverse of the one part of it that
/// matters for the cache key. A prompt without a `# Command` section yields an empty string, whose
/// signature nothing else shares — so an unexpected prompt is judged again rather than sharing a
/// verdict with something else.
fn command_of(prompt: &str) -> String {
    let Some(rest) = prompt.split_once("# Command\n\n```sh\n").map(|(_, rest)| rest) else {
        return String::new();
    };
    rest.split_once("\n```").map_or_else(
        || rest.trim().to_string(),
        |(command, _)| command.trim().to_string(),
    )
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

    #[test]
    fn a_repeated_command_shape_is_paid_for_once() {
        // The measured cost Driver: `bundle exec rspec` is an unknown program, and without this a
        // suite of a hundred files would ask the same question a hundred times.
        let inner = Scripted::new(vec![
            Ok(r#"{"verdict":"allow"}"#.into()),
            // No second reply: if the wrapper asks again, `Scripted` answers with an error, which
            // becomes an escalation and fails the assertion below.
        ]);
        let recorded = Recorded::new(Arc::clone(&inner) as Arc<dyn Judge>, "policy-a");
        for command in [
            "bundle exec rspec spec/a_spec.rb",
            "bundle exec rspec spec/b_spec.rb",
        ] {
            let reply = recorded
                .judge(&prompt(command, "/wt", "make the suite pass"))
                .expect("the first judgement answers both");
            assert_eq!(parse_reply(&reply).expect("parse"), Judgement::Allow);
        }
        assert_eq!(
            inner.prompts().len(),
            1,
            "the second command was asked separately"
        );
        assert_eq!(recorded.take_fresh().len(), 1, "one verdict to journal, not two");
    }

    #[test]
    fn a_denial_is_not_remembered_and_is_still_reported() {
        // Two facts at once. A cached `deny` would be an answer a changed world could make wrong,
        // so the next identical command must ask again; and the denial must still reach the
        // journal, or a human reading the run back cannot see why the step failed.
        let inner = Scripted::new(vec![
            Ok(r#"{"verdict":"deny","why":"it writes the system gem directory"}"#.into()),
            Ok(r#"{"verdict":"allow"}"#.into()),
        ]);
        let recorded = Recorded::new(Arc::clone(&inner) as Arc<dyn Judge>, "policy-a");
        let text = prompt("bundle exec rspec a.rb", "/wt", "goal");
        assert!(matches!(
            parse_reply(&recorded.judge(&text).expect("first")),
            Ok(Judgement::Deny(_))
        ));
        assert!(matches!(
            parse_reply(&recorded.judge(&text).expect("second")),
            Ok(Judgement::Allow)
        ));
        assert_eq!(inner.prompts().len(), 2, "a denial must not be cached");
        let fresh = recorded.take_fresh();
        assert_eq!(fresh.len(), 2, "both answers are reported to the journal");
        assert_eq!(fresh[0].judgement.as_str(), "deny");
    }

    #[test]
    fn an_unreadable_reply_is_neither_cached_nor_remembered() {
        // The failure that becomes an escalation. Caching "the judge was confused" would make one
        // bad reply permanent, and the next attempt would never get a real answer.
        let inner = Scripted::new(vec![
            Ok("I think it is fine, actually.".into()),
            Ok(r#"{"verdict":"allow"}"#.into()),
        ]);
        let recorded = Recorded::new(Arc::clone(&inner) as Arc<dyn Judge>, "policy-a");
        let text = prompt("bundle exec rspec a.rb", "/wt", "goal");
        let first = recorded.judge(&text).expect("the reply itself is returned");
        assert!(
            parse_reply(&first).is_err(),
            "the caller turns this into an escalation"
        );
        assert!(
            recorded.take_fresh().is_empty(),
            "nothing unreadable is journalled"
        );
        assert!(matches!(
            parse_reply(&recorded.judge(&text).expect("second")),
            Ok(Judgement::Allow)
        ));
        assert_eq!(inner.prompts().len(), 2, "an unreadable reply must not be cached");
    }

    #[test]
    fn a_seeded_judge_answers_without_calling_the_model() {
        // What a restart reads back from the journal. Calling the model here would re-pay for every
        // unknown program after a crash, which is the cost D-111 says must not exist.
        let inner = Scripted::new(vec![]);
        let known = vec![Verdict {
            signature: signature("bundle exec rspec spec/a_spec.rb"),
            policy: "policy-a".to_string(),
            command: "bundle exec rspec spec/a_spec.rb".to_string(),
            judgement: Judgement::Allow,
        }];
        let recorded = Recorded::seeded(Arc::clone(&inner) as Arc<dyn Judge>, "policy-a", known);
        let reply = recorded
            .judge(&prompt("bundle exec rspec spec/z_spec.rb", "/wt", "goal"))
            .expect("answered from the journal");
        assert_eq!(parse_reply(&reply).expect("parse"), Judgement::Allow);
        assert!(inner.prompts().is_empty(), "the model was called after a restart");
        assert!(recorded.take_fresh().is_empty(), "a seeded verdict is not new");
    }

    #[test]
    fn the_command_is_read_back_out_of_the_prompt() {
        // The wrapper computes the signature from the prompt's own command section, because the
        // prompt is all a `Judge` receives. If this inverse ever disagreed with `prompt`, every
        // lookup would miss — silently, since a miss is not an error.
        let text = prompt("bundle exec rspec a.rb", "/wt", "goal");
        assert_eq!(command_of(&text), "bundle exec rspec a.rb");
        assert_eq!(command_of("no sections here"), "");
    }
}
