//! Turning an assembled [`Context`] into the messages that go on the wire.
//!
//! # Why this lives in `immortality-llm` and not in `core`
//!
//! `core` owns the context and the token budget, but it must not know `Message`: the dependency
//! arrow is `llm → core`, and a `Message` type in `core` would make the protocol a dependency of
//! the domain. So the conversion sits here, next to the wire types it produces.
//!
//! # Why the append-only property is asserted *through* this function
//!
//! `Context` keeping its prefix stable is necessary but not sufficient: a converter that rebuilt
//! the system message from the history — or that put the tool results before the assistant turn
//! that asked for them — would defeat the cache while every `Context` test stayed green. The
//! test here compares the *serialised* prefix, which is what the server actually keys on.
//!
//! # Message order the protocol requires
//!
//! `assistant` with `tool_calls` must be followed by one `tool` message per call id, in the same
//! order. Servers vary in how strictly they check, but the ones that are lenient render a wrong
//! order as a result belonging to the wrong call, which the model cannot tell from the truth.

use immortality_core::context::{Context, Entry, ToolCall};

use crate::turn;
use crate::wire::Message;

/// The one user turn every attempt starts with.
///
/// The stable prefix — layers 1–4, brief included — goes into the system message, so the model
/// already knows what to do; what it lacks is a *turn to answer*. Chat templates require at
/// least one user message, and strata rejects the request outright with `No user query found in
/// messages`, which turns every step into a `Backend` ending before any work happens.
///
/// A constant rather than the brief, deliberately: it sits inside the cacheable prefix, so the
/// bytes must not change between steps. Moving the brief here would work too and would be the
/// more conventional layout, but it would split the prefix across two messages for no gain —
/// the model reads both, and the cache keys on the whole prefix either way.
const BEGIN: &str = "Begin. Work step by step, using the tools you have been given.";

/// Build the message list for one step.
///
/// One system message carrying the stable prefix, one constant user turn, then the history in
/// order.
#[must_use]
pub fn messages(context: &Context) -> Vec<Message> {
    let mut messages = Vec::with_capacity(2 + context.steps());
    messages.push(Message::system(context.system()));
    messages.push(Message::user(BEGIN));
    for entry in context.history() {
        match entry {
            Entry::Assistant { text, calls } => {
                if calls.is_empty() {
                    messages.push(Message::assistant(text.clone()));
                } else {
                    // The prose and the calls can coexist in one assistant turn. Dropping the
                    // prose would lose the model's stated reasoning; dropping the calls would
                    // make the following tool messages reference nothing.
                    let mut message = Message::assistant_tool_calls(calls.iter().map(to_wire_call).collect());
                    if !text.is_empty() {
                        message.content = Some(text.clone());
                    }
                    messages.push(message);
                }
            }
            Entry::Tool {
                call_id,
                name,
                output,
            } => {
                messages.push(Message::tool_result(
                    call_id.clone(),
                    name.clone(),
                    output.clone(),
                ));
            }
            // A nudge from the detector ladder is the system talking, and the wire protocol has
            // no third role. `user` is the honest choice: the model is being asked a question
            // ("what will you do differently?"), not handed a tool result it could mistake for
            // output it asked for. A `system` message in the middle of a conversation is
            // accepted by some servers and silently ignored by others, which would make the
            // ladder a no-op on exactly the backend this project targets.
            Entry::Note { text } => messages.push(Message::user(text.clone())),
        }
    }
    messages
}

/// How many bytes of messages precede the history, for a cache probe.
///
/// Only the system message is stable across steps, so this is the length the cache can be
/// expected to cover on a warm prefix. Used by `doctor` and by the cache-stats assertion.
#[must_use]
pub fn stable_prefix_len(messages: &[Message]) -> usize {
    messages
        .first()
        .and_then(|message| message.content.as_ref())
        .map_or(0, String::len)
}

fn to_wire_call(call: &ToolCall) -> turn::ToolCall {
    turn::ToolCall::function(call.call_id.clone(), call.name.clone(), call.arguments.clone())
}

#[cfg(test)]
mod tests {
    use immortality_core::context::StableLayers;

    use super::*;

    fn context() -> Context {
        Context::new(&StableLayers {
            system: "You are an agent.".to_string(),
            brief: "Add `serve`.".to_string(),
            notes: "The socket is 0600.".to_string(),
            state: "0/1 done.".to_string(),
        })
    }

    #[test]
    fn the_system_message_is_the_whole_prefix() {
        let context = context();
        let messages = messages(&context);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].content.as_deref(), Some(context.system()));
        assert_eq!(stable_prefix_len(&messages), context.system().len());
    }

    #[test]
    fn every_request_carries_a_user_turn_because_the_backend_requires_one() {
        // Measured on the live backend: a request whose messages are all `system` is rejected
        // with `No user query found in messages`, before the model sees anything. A fake model
        // never says so, so the run would fail on its first real step.
        let context = context();
        let messages = messages(&context);
        let users: Vec<&Message> = messages.iter().filter(|m| m.role == "user").collect();
        assert_eq!(users.len(), 1, "{messages:?}");
        assert_ne!(users[0].content.as_deref().unwrap_or(""), "");
    }

    #[test]
    fn the_user_turn_is_stable_across_steps() {
        // It sits inside the prefix the cache keys on, so it must not vary: a user turn that
        // quoted the last observation would look helpful and cost a 400× prefill every step.
        let mut context = context();
        let first = {
            let messages = messages(&context);
            serde_json::to_string(&messages[1]).expect("serialise")
        };
        context.push_assistant("thinking", vec![]);
        context.push_entry(Entry::tool("call", "read_file", "fn main() {}"));
        let later = serde_json::to_string(&messages(&context)[1]).expect("serialise");
        assert_eq!(later, first);
    }

    #[test]
    fn the_serialised_prefix_is_unchanged_as_the_history_grows() {
        // The property that decides whether the backend's prefix cache hits. Asserted on the
        // JSON, which is what is actually sent.
        let mut context = context();
        let prefix = serde_json::to_string(&messages(&context)[0]).expect("serialise");
        for step in 0..5 {
            context.push_assistant(format!("thinking {step}"), vec![]);
            context.push_entry(Entry::tool("call", "read_file", "fn main() {}"));
        }
        let later = serde_json::to_string(&messages(&context)[0]).expect("serialise");
        assert_eq!(later, prefix);
    }

    #[test]
    fn an_assistant_turn_with_calls_and_prose_keeps_both() {
        let mut context = context();
        context.push_assistant(
            "checking the config",
            vec![ToolCall {
                call_id: "call_1".to_string(),
                name: "read_file".to_string(),
                arguments: r#"{"path":"Cargo.toml"}"#.to_string(),
            }],
        );
        context.push_entry(Entry::tool("call_1", "read_file", "[package]"));

        let messages = messages(&context);
        assert_eq!(messages.len(), 4);
        let assistant = &messages[2];
        assert_eq!(assistant.role, "assistant");
        assert_eq!(assistant.content.as_deref(), Some("checking the config"));
        let calls = assistant.tool_calls.as_ref().expect("calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments, r#"{"path":"Cargo.toml"}"#);

        let tool = &messages[3];
        assert_eq!(tool.role, "tool");
        assert_eq!(tool.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(tool.name.as_deref(), Some("read_file"));
        assert_eq!(tool.content.as_deref(), Some("[package]"));
    }

    #[test]
    fn a_prose_only_turn_has_no_tool_calls_field() {
        let mut context = context();
        context.push_assistant("done", vec![]);
        let messages = messages(&context);
        assert_eq!(messages[2].role, "assistant");
        assert_eq!(messages[2].tool_calls, None);
    }

    #[test]
    fn a_note_becomes_a_user_message_after_the_history_it_comments_on() {
        // The ladder's nudge cannot go into the prefix — that would invalidate the cache — so it
        // is appended, and it must arrive after the calls it is about.
        let mut context = context();
        context.push_assistant("looking", vec![]);
        context.push_entry(Entry::note("you are repeating yourself"));
        let messages = messages(&context);
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[2].role, "assistant");
        assert_eq!(messages[3].role, "user");
        assert_eq!(messages[3].content.as_deref(), Some("you are repeating yourself"));
    }

    #[test]
    fn a_note_does_not_move_the_prefix() {
        // The point of appending it: the cache keys on the prefix, and a nudge that changed the
        // prefix would cost a cold prefill exactly when a run is already going badly.
        let mut context = context();
        let prefix = serde_json::to_string(&messages(&context)[0]).expect("serialise");
        context.push_entry(Entry::note("you are repeating yourself"));
        let later = serde_json::to_string(&messages(&context)[0]).expect("serialise");
        assert_eq!(later, prefix);
    }

    #[test]
    fn every_history_entry_becomes_exactly_one_message() {
        let mut context = context();
        context.push_assistant("a", vec![]);
        context.push_entry(Entry::tool("c1", "grep", "hit"));
        context.push_assistant("b", vec![]);
        assert_eq!(messages(&context).len(), 2 + context.steps());
    }
}
