//! The production [`Judge`]: one model call, no tools.
//!
//! # Why this lives in the daemon and not in `tools`
//!
//! `immortality-tools` owns the judge's *question* — the system message, the prompt and the
//! parser — and deliberately does not own the model: a crate that could call one would need the
//! backend client, the retry policy and the token accounting, which is the daemon's business and
//! would drag `reqwest` into the policy layer. So the trait lives with the question and the one
//! implementation lives here, where [`Model`] already is.
//!
//! # Why there are no tools on the request
//!
//! The same reason as [`ModelReviewer`](crate::gates::ModelReviewer): the absence of tools is the
//! enforcement of "the judge cannot look at the repository". A system message asking it not to is
//! a request; an empty tool list is a fact, and the difference matters because a judge that could
//! read files could approve a command by checking whether it happens to be safe today.

use std::sync::Arc;

use immortality_llm::wire::{ChatRequest, Message};

use crate::runner::Model;

/// Ceiling on the answer.
///
/// A verdict is a few tokens. The budget exists so a model that starts reasoning about the command
/// cannot spend an attempt's context on it; the reply is parsed strictly, so a long answer is
/// refused rather than summarised.
pub const MAX_TOKENS: u32 = 256;

/// The judge, over the daemon's model client.
#[derive(Debug)]
pub struct ModelJudge {
    model: Arc<dyn Model>,
    model_name: String,
    max_tokens: u32,
}

impl ModelJudge {
    /// A judge over an existing model client, asking the model named by `model_name`.
    ///
    /// The name is separate from the client because one client serves the planner, the executor
    /// and the reviewers, and which of them a request is for lives in the request body. A judge
    /// is a short classification, so the caller passes the utility profile alongside the
    /// reviewer's.
    #[must_use]
    pub fn new(model: Arc<dyn Model>, model_name: impl Into<String>) -> Self {
        Self {
            model,
            model_name: model_name.into(),
            max_tokens: MAX_TOKENS,
        }
    }

    /// The request this judge would send for `prompt`.
    ///
    /// Exposed for the test that checks the two properties the design depends on — no tools, and
    /// the judge's own system message — without a second copy of the request builder that could
    /// drift from the one used here.
    #[must_use]
    pub fn request(&self, prompt: &str) -> ChatRequest {
        ChatRequest::new(self.model_name.clone())
            .with_system(immortality_tools::judge::SYSTEM)
            .with_max_tokens(self.max_tokens)
            .with(Message::user(prompt))
    }
}

impl immortality_tools::judge::Judge for ModelJudge {
    fn judge(&self, prompt: &str) -> Result<String, String> {
        self.model
            .complete(&self.request(prompt))
            .map(|reply| reply.content)
    }
}

/// Nothing here reads the reply: `Recorded` and `PolicyGate` do, and they are covered where the
/// parser is. The tests below are about the request shape and the error path, which are this
/// module's own decisions.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::Reply;
    use std::sync::Mutex;

    /// A model that records what it was asked and answers with a fixed text.
    #[derive(Debug, Default)]
    struct Recorder {
        seen: Mutex<Vec<ChatRequest>>,
        reply: Mutex<Option<Result<Reply, String>>>,
    }

    impl Recorder {
        fn answering(reply: Reply) -> Arc<Self> {
            let recorder = Self::default();
            *recorder.reply.lock().expect("lock") = Some(Ok(reply));
            Arc::new(recorder)
        }

        fn failing(detail: &str) -> Arc<Self> {
            let recorder = Self::default();
            *recorder.reply.lock().expect("lock") = Some(Err(detail.to_string()));
            Arc::new(recorder)
        }

        fn requests(&self) -> Vec<ChatRequest> {
            self.seen.lock().map(|seen| seen.clone()).unwrap_or_default()
        }
    }

    impl Model for Recorder {
        fn complete(&self, request: &ChatRequest) -> Result<Reply, String> {
            if let Ok(mut seen) = self.seen.lock() {
                seen.push(request.clone());
            }
            self.reply
                .lock()
                .expect("lock")
                .clone()
                .unwrap_or_else(|| Err("no reply scripted".to_string()))
        }
    }

    #[test]
    fn the_judge_asks_without_tools_and_with_its_own_system_message() {
        // Two properties the design leans on, and neither is visible from the verdict alone: a
        // judge with tools could approve a command by reading the repository, and a judge with
        // the executor's system message would be answering a different question.
        let model = Recorder::answering(Reply::text(r#"{"verdict":"allow"}"#));
        let judge = ModelJudge::new(Arc::clone(&model) as Arc<dyn Model>, "utility-model");
        let request = judge.request("a prompt");

        assert!(request.tools.is_none(), "the judge was given tools");
        assert_eq!(request.model, "utility-model");
        assert!(request.max_tokens.is_some_and(|cap| cap <= MAX_TOKENS));
        let system = request
            .messages
            .first()
            .and_then(|message| message.content.clone())
            .unwrap_or_default();
        assert_eq!(system, immortality_tools::judge::SYSTEM);
    }

    #[test]
    fn the_reply_is_returned_unparsed_and_the_failure_is_a_string() {
        // The split with `Recorded`: this layer does not decide what a reply means, it only gets
        // one. A backend failure has to survive as the reason, because `PolicyGate` turns it into
        // an escalation and the operator has to read why.
        use immortality_tools::judge::Judge as _;

        let model = Recorder::answering(Reply::text("  not json  "));
        let judge = ModelJudge::new(Arc::clone(&model) as Arc<dyn Model>, "m");
        assert_eq!(judge.judge("p").expect("reply"), "  not json  ");

        let failing = Recorder::failing("connection refused");
        let judge = ModelJudge::new(Arc::clone(&failing) as Arc<dyn Model>, "m");
        assert_eq!(judge.judge("p").expect_err("failure"), "connection refused");

        // And the prompt reaches the model as the user turn, not the system one.
        let request = failing.requests().into_iter().next().expect("one request");
        assert_eq!(request.messages.len(), 2, "{:?}", request.messages);
        let user = request.messages[1].content.clone().unwrap_or_default();
        assert_eq!(user, "p");
    }
}
