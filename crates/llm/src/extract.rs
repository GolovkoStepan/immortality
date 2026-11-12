//! Getting tool calls out of a turn, natively first and by parsing text second.
//!
//! # Why there is a fallback at all
//!
//! S1 measured zero broken calls in thirty steps: the backend's native tool-calling is good.
//! But "zero in thirty" is not "zero forever", and the failure it degrades into is expensive:
//! a model that writes a call as prose produces a turn with no tool calls, and a loop that only
//! reads `tool_calls` sees an assistant message with no action and stops. The task ends
//! successfully having done nothing.
//!
//! # The rule that keeps the fallback from being worse than the problem
//!
//! **A repair normalises syntax; it never invents data.** A trailing comma, a single-quoted
//! string, a fenced block, a cut-off closing brace — all are recoverable because the information
//! is present and the shape is wrong. A missing `path` argument is *not* recoverable: filling it
//! with a guess means executing a command against a target nobody chose. When the fallback
//! cannot recover faithfully, it returns [`Extraction::None`] and the invalid call becomes an
//! observation for the model, which is a normal step rather than a failure.
//!
//! # What is deliberately not attempted
//!
//! Natural-language inference ("the model obviously means `shell`"). It cannot be tested against
//! what the model meant, only against what the author guessed, and a wrong guess here runs an
//! arbitrary command.

use crate::turn::{AssistantTurn, CalledFunction, ToolCall};

/// How the calls in a turn were obtained.
///
/// Recorded in the journal as the degradation metric: a rising `TextFallback` share says the
/// model's native tool-calling is getting worse, which is a reason to change profile rather
/// than to write more fallback code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extraction {
    /// The backend put calls in `tool_calls`. The normal path.
    Native,
    /// The calls were recovered from the text of the answer.
    TextFallback,
    /// The answer asked for an action in a form that could not be recovered faithfully.
    None,
}

/// The outcome of looking for calls in a turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Extracted {
    /// The calls to execute, in the order they should run.
    pub calls: Vec<ToolCall>,
    /// How they were obtained.
    pub source: Extraction,
    /// Why the fallback gave up, when it did. Empty when it did not run.
    ///
    /// Kept so the observation sent back to the model can say what was wrong with its answer
    /// instead of only that it was wrong.
    pub problem: Option<String>,
    /// The tool name the text appears to have used, when the name was readable but the arguments
    /// were not.
    ///
    /// This is the one piece of partial information worth carrying: it lets the observation say
    /// "your `edit_file` call is missing `old`" rather than "I could not understand you".
    pub partial_tool: Option<String>,
}

impl Extracted {
    const fn native(calls: Vec<ToolCall>) -> Self {
        Self {
            calls,
            source: Extraction::Native,
            problem: None,
            partial_tool: None,
        }
    }

    fn none(problem: impl Into<String>, partial_tool: Option<String>) -> Self {
        Self {
            calls: Vec::new(),
            source: Extraction::None,
            problem: Some(problem.into()),
            partial_tool,
        }
    }

    /// Whether anything is executable.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }
}

/// Find the tool calls in `turn`.
///
/// Native first and unconditionally: when the backend produced calls, the text is commentary
/// about them, and parsing it would duplicate or contradict them.
#[must_use]
pub fn extract(turn: &AssistantTurn) -> Extracted {
    if !turn.tool_calls.is_empty() {
        return Extracted::native(turn.tool_calls.clone());
    }
    extract_from_text(&turn.content)
}

/// Look for tool calls in prose.
///
/// Every shape the observed models produce, in the order they are tried. The order matters: a
/// fenced `json` block is unambiguous, an XML-ish tag is nearly so, and a bare `{` is a last
/// resort that needs a name next to it to be worth anything.
#[must_use]
pub fn extract_from_text(text: &str) -> Extracted {
    if text.trim().is_empty() {
        return Extracted::none("the answer was empty", None);
    }

    let mut first_problem: Option<(String, Option<String>)> = None;
    for candidate in candidates(text) {
        match parse_call(&candidate.name, &candidate.arguments) {
            Ok(call) => {
                return Extracted {
                    calls: vec![call],
                    source: Extraction::TextFallback,
                    problem: None,
                    partial_tool: None,
                };
            }
            Err(problem) => {
                // Keep the first failure: it comes from the most reliable shape, so its error
                // message is the most specific description of what went wrong.
                if first_problem.is_none() {
                    first_problem = Some((problem, Some(candidate.name)));
                }
            }
        }
    }

    match first_problem {
        Some((problem, tool)) => Extracted::none(problem, tool),
        None => Extracted::none("the answer contains no recognisable tool call", None),
    }
}

/// One candidate call found in the text, before its arguments are validated.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Candidate {
    name: String,
    arguments: String,
}

/// Yield candidate calls in decreasing order of reliability.
fn candidates(text: &str) -> Vec<Candidate> {
    let mut found = Vec::new();

    // 1. A fenced block, with or without a language tag.
    if let Some(block) = fenced_block(text) {
        found.extend(parse_block(&block));
    }

    // 2. XML-ish tags: `<tool_call>{"name":...,"arguments":{...}}</tool_call>`, and the variants
    //    with the name as an attribute.
    for tag in ["tool_call", "function_call", "tool", "function"] {
        for inner in tagged(text, tag) {
            found.extend(parse_block(&inner));
        }
    }

    // 3. A bare JSON object that has both `name` and `arguments`. Requires both keys, which is
    //    what keeps this from matching every stray brace in an explanation.
    if let Some(object) = bare_call_object(text) {
        found.extend(object);
    }

    found
}

/// The first fenced code block, preferring one tagged `json`.
fn fenced_block(text: &str) -> Option<String> {
    let mut rest = text;
    // Prefer a ```json block anywhere in the text over an untagged block.
    if let Some(start) = text.find("```json") {
        let body = &text[start + "```json".len()..];
        if let Some(end) = body.find("```") {
            return Some(body[..end].trim().to_string());
        }
    }
    // Otherwise the first fence of any kind.
    while let Some(start) = rest.find("```") {
        let after = &rest[start + 3..];
        // Skip the language tag on the opening line.
        let body = after.split_once('\n').map_or(after, |(_, body)| body);
        if let Some(end) = body.find("```") {
            return Some(body[..end].trim().to_string());
        }
        rest = after;
    }
    None
}

/// Every `<tag>…</tag>` body.
fn tagged(text: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(&open) {
        let body = &rest[start + open.len()..];
        let Some(end) = body.find(&close) else { break };
        out.push(body[..end].trim().to_string());
        rest = &body[end + close.len()..];
    }
    out
}

/// A `{"name": …, "arguments": …}` object embedded in the text.
///
/// The object may be followed by commentary, and it may be truncated. So the search runs from
/// the longest prefix that ends in `}` downwards, and only then falls back to repairing the
/// whole tail — which is what makes a cut-off object parseable.
fn bare_call_object(text: &str) -> Option<Vec<Candidate>> {
    let start = text.find('{')?;
    let slice = text.get(start..)?;

    // A complete object with trailing prose: try every prefix that ends at a closing brace,
    // longest first, so the commentary is dropped rather than parsed.
    for end in (0..slice.len()).rev() {
        let Some(candidate) = slice.get(..=end) else {
            continue;
        };
        if !candidate.ends_with('}') {
            continue;
        }
        if let Some(calls) = parse_block_value_of(candidate) {
            return Some(calls);
        }
    }

    // Nothing balanced: the object was cut off. Repair closes it so the *schema check* can
    // report the missing field by name; the repair still invents no data.
    parse_block_value_of(slice)
}

/// Parse a slice as a call object, repairing its syntax first.
fn parse_block_value_of(slice: &str) -> Option<Vec<Candidate>> {
    let repaired = repair_json(slice);
    let value: serde_json::Value = serde_json::from_str(&repaired).ok()?;
    parse_block_value(&value)
}

/// Parse the contents of a block as either a call object or a tool-call envelope.
fn parse_block(block: &str) -> Vec<Candidate> {
    // The block may itself be wrapped in an OpenAI-shaped envelope; accept that too, because a
    // model that "calls" a function in text often copies the API's own shape.
    let repaired = repair_json(block);
    serde_json::from_str::<serde_json::Value>(&repaired).map_or_else(
        |_| Vec::new(),
        |value| parse_block_value(&value).unwrap_or_default(),
    )
}

fn parse_block_value(value: &serde_json::Value) -> Option<Vec<Candidate>> {
    // `{"name": "...", "arguments": {...}}`
    if let (Some(name), Some(arguments)) = (
        value.get("name").and_then(|v| v.as_str()),
        value.get("arguments").or_else(|| value.get("parameters")),
    ) {
        return Some(vec![Candidate {
            name: name.to_string(),
            arguments: arguments_to_string(arguments),
        }]);
    }
    // `{"type": "function", "function": {"name": ..., "arguments": ...}}`
    if let Some(function) = value.get("function") {
        return parse_block_value(function);
    }
    // `{"tool": "...", "args": {...}}`, the shape some templates prefer.
    if let (Some(name), Some(arguments)) = (value.get("tool").and_then(|v| v.as_str()), value.get("args")) {
        return Some(vec![Candidate {
            name: name.to_string(),
            arguments: arguments_to_string(arguments),
        }]);
    }
    None
}

/// Arguments may be a JSON object or a JSON-encoded string; both appear in the wild.
fn arguments_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Normalise the syntax of a JSON blob without changing its data.
///
/// Every transformation here is reversible and loses nothing: a trailing comma, single quotes,
/// a smart quote pasted from a chat window, or a missing closing brace. Anything that would
/// require deciding what the model meant is not done.
#[must_use]
pub fn repair_json(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    // The quote character that opened the current string. Tracked rather than a boolean so
    // that a single-quoted string closes on `'` and a double-quoted one on `"`: a boolean
    // treats a `'` inside `"...'..."` as the terminator, which mangles the payload — the one
    // thing a repair must never do.
    let mut open: Option<char> = None;
    let mut escaped = false;
    // A plain `Chars` rather than a `Peekable`: the trailing-comma check needs to look *past*
    // whitespace, which `peek` cannot do, so it clones the iterator instead.
    let mut chars = text.chars();

    while let Some(ch) = chars.next() {
        if let Some(quote) = open {
            // Inside a string only the matching terminator matters; text is copied verbatim so
            // the payload survives.
            if escaped {
                out.push(ch);
                escaped = false;
            } else if ch == '\\' {
                out.push(ch);
                escaped = true;
            } else if ch == quote {
                out.push('"');
                open = None;
            } else if ch == '\n' {
                // A raw newline inside a JSON string is invalid. Escaping it is the only repair
                // that preserves the character.
                out.push_str("\\n");
            } else {
                out.push(ch);
            }
            continue;
        }

        match ch {
            // Smart quotes become straight ones. Outside a string the only thing a quote can be
            // is structural, so there is nothing to preserve.
            '\u{201c}' | '\u{201d}' => {
                out.push('"');
                open = Some('\u{201d}');
            }
            '\u{2018}' | '\u{2019}' => {
                out.push('"');
                open = Some('\u{2019}');
            }
            '\'' => {
                out.push('"');
                open = Some('\'');
            }
            '"' => {
                out.push('"');
                open = Some('"');
            }
            ',' => {
                // Drop a comma that precedes `}` or `]`.
                let mut lookahead = chars.clone();
                let next = lookahead.find(|c: &char| !c.is_whitespace());
                if matches!(next, Some('}' | ']')) {
                    continue;
                }
                out.push(',');
            }
            _ => out.push(ch),
        }
    }

    // An unterminated string keeps whatever was read and is closed, so the object can be
    // parsed and the schema check can name what is missing.
    if open.is_some() {
        out.push('"');
    }
    // Close an unterminated container. The count is of unclosed *structure*, computed from the
    // repaired text, and closing it adds no data: it makes a truncated object parseable so the
    // schema check can report the missing field by name.
    let closers = unclosed(&out);
    out.push_str(&closers);
    out
}

/// The closing brackets needed to balance `text`, outside of strings.
fn unclosed(text: &str) -> String {
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for ch in text.chars() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' if stack.last() == Some(&ch) => {
                stack.pop();
            }
            _ => {}
        }
    }
    stack.into_iter().rev().collect()
}

/// Turn a candidate into a `ToolCall`, or say why it cannot be one.
fn parse_call(name: &str, arguments: &str) -> Result<ToolCall, String> {
    if name.trim().is_empty() {
        return Err("the call has no tool name".to_string());
    }
    let repaired = repair_json(arguments);
    let parsed: serde_json::Value = serde_json::from_str(&repaired).map_err(|error| {
        // A bare string of arguments (`"ls -la"`) is the common shape here. Naming what was
        // expected is what lets the model correct itself on the next step.
        format!("the arguments of `{name}` are not an object: {error}")
    })?;
    if !parsed.is_object() && !parsed.is_null() {
        return Err(format!(
            "the arguments of `{name}` are a {}, not an object",
            json_kind(&parsed)
        ));
    }
    Ok(ToolCall::function(
        // The fallback invents an id because the model did not supply one, and the journal
        // requires one. The `text_` prefix makes it obvious in a log that this call came from
        // the fallback path rather than from the backend.
        format!("text_{}", stable_suffix(name, arguments)),
        name,
        repaired,
    ))
}

/// A short, stable suffix for a fallback call id.
///
/// Derived from the content rather than from a counter so that replaying the same turn produces
/// the same id — which matters because the journal's `intent` and `completed` records are matched
/// by id.
fn stable_suffix(name: &str, arguments: &str) -> String {
    use sha2::Digest as _;
    let mut hasher = sha2::Sha256::new();
    hasher.update(name.as_bytes());
    hasher.update(b"\0");
    hasher.update(arguments.as_bytes());
    let digest = hasher.finalize();
    digest.iter().take(6).fold(String::new(), |mut out, byte| {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
        out
    })
}

const fn json_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Whether a turn needs the fallback because the backend produced no calls.
///
/// Named separately so the call site reads as a decision rather than a negation.
#[must_use]
pub fn needs_fallback(turn: &AssistantTurn) -> bool {
    turn.tool_calls.is_empty() && !turn.content.trim().is_empty()
}

/// Whether a turn is an assistant *question* rather than an action.
///
/// A model that asks "should I proceed?" is not stuck and must not be treated as one: the loop
/// answers it and continues. The distinction is the absence of any call *and* the absence of a
/// call-shaped block, so the caller can choose to reply instead of retrying.
#[must_use]
pub fn is_question(turn: &AssistantTurn) -> bool {
    turn.tool_calls.is_empty() && extract_from_text(&turn.content).is_empty()
}

/// Rebuild a `CalledFunction` for a call the fallback produced.
///
/// Exists so that a caller holding only `(name, arguments)` does not have to remember the
/// `type: "function"` field, which is the one thing easy to forget.
#[must_use]
pub fn called_function(name: impl Into<String>, arguments: impl Into<String>) -> CalledFunction {
    CalledFunction {
        name: name.into(),
        arguments: arguments.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(content: &str) -> AssistantTurn {
        AssistantTurn {
            content: content.to_string(),
            finish_reason: "stop".to_string(),
            ..AssistantTurn::default()
        }
    }

    fn turn_with_call() -> AssistantTurn {
        AssistantTurn {
            content: "let me look".to_string(),
            tool_calls: vec![ToolCall::function("call_1", "read_file", r#"{"path":"a"}"#)],
            finish_reason: "tool_calls".to_string(),
            ..AssistantTurn::default()
        }
    }

    #[test]
    fn a_native_call_wins_and_the_text_is_ignored() {
        // The text mentions a *different* call in a fenced block. If the fallback ran anyway,
        // the loop would execute two commands where the model asked for one.
        let mut turn = turn_with_call();
        turn.content = "```json\n{\"name\":\"shell\",\"arguments\":{\"command\":\"rm -rf /\"}}\n```".into();
        let extracted = extract(&turn);
        assert_eq!(extracted.source, Extraction::Native);
        assert_eq!(extracted.calls.len(), 1);
        assert_eq!(extracted.calls[0].function.name, "read_file");
    }

    #[test]
    fn a_fenced_json_block_is_recovered() {
        let extracted = extract_from_text(
            "I'll read the file.\n\n```json\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/main.rs\"}}\n```",
        );
        assert_eq!(extracted.source, Extraction::TextFallback);
        assert_eq!(extracted.calls.len(), 1);
        assert_eq!(extracted.calls[0].function.name, "read_file");
        assert_eq!(
            extracted.calls[0].function.parse().unwrap()["path"],
            "src/main.rs"
        );
    }

    #[test]
    fn a_fenced_block_without_a_language_tag_is_recovered() {
        let extracted =
            extract_from_text("```\n{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}\n```");
        assert_eq!(extracted.source, Extraction::TextFallback);
        assert_eq!(extracted.calls[0].function.name, "shell");
    }

    #[test]
    fn the_openai_envelope_shape_is_recovered() {
        // Models frequently copy the API's own shape when they "call" a function in text.
        let extracted = extract_from_text(
            r#"{"type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a\"}"}}"#,
        );
        assert_eq!(extracted.source, Extraction::TextFallback);
        assert_eq!(extracted.calls[0].function.name, "read_file");
        assert_eq!(extracted.calls[0].function.parse().unwrap()["path"], "a");
    }

    #[test]
    fn an_xml_tagged_call_is_recovered() {
        let extracted = extract_from_text(
            "<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"cargo test\"}}</tool_call>",
        );
        assert_eq!(extracted.source, Extraction::TextFallback);
        assert_eq!(extracted.calls[0].function.name, "shell");
    }

    #[test]
    fn a_bare_call_object_in_prose_is_recovered() {
        let extracted = extract_from_text(
            "I need to check something {\"name\": \"read_file\", \"arguments\": {\"path\": \"x\"}} and then decide.",
        );
        assert_eq!(extracted.source, Extraction::TextFallback);
        assert_eq!(extracted.calls[0].function.name, "read_file");
    }

    #[test]
    fn a_trailing_comma_is_repaired() {
        let extracted = extract_from_text(r#"{"name":"shell","arguments":{"command":"ls","timeout":5,}}"#);
        assert_eq!(extracted.source, Extraction::TextFallback);
        let args = extracted.calls[0].function.parse().unwrap();
        assert_eq!(args["command"], "ls");
        assert_eq!(args["timeout"], 5);
    }

    #[test]
    fn single_quotes_are_repaired() {
        let extracted = extract_from_text("{'name': 'shell', 'arguments': {'command': 'echo hi'}}");
        assert_eq!(extracted.source, Extraction::TextFallback);
        assert_eq!(extracted.calls[0].function.parse().unwrap()["command"], "echo hi");
    }

    #[test]
    fn smart_quotes_pasted_from_a_chat_window_are_repaired() {
        let extracted = extract_from_text(
            "{\u{201c}name\u{201d}: \u{201c}read_file\u{201d}, \u{201c}arguments\u{201d}: {\u{201c}path\u{201d}: \u{201c}a\u{201d}}}",
        );
        assert_eq!(extracted.source, Extraction::TextFallback);
        assert_eq!(extracted.calls[0].function.parse().unwrap()["path"], "a");
    }

    #[test]
    fn a_truncated_object_is_closed_but_nothing_is_invented() {
        // The model stopped mid-JSON. Closing the brace makes the object parseable *so that the
        // schema check can name the missing field*. It must not add the field.
        let extracted = extract_from_text(r#"{"name":"edit_file","arguments":{"path":"a""#);
        // The candidate survives to the argument check, which is where the missing `old`/`new`
        // will be reported by name.
        assert_eq!(extracted.source, Extraction::TextFallback);
        let args = extracted.calls[0].function.parse().unwrap();
        assert_eq!(args["path"], "a");
        assert!(args.get("old").is_none(), "the repair invented a field: {args}");
    }

    #[test]
    fn a_quoted_arguments_string_is_accepted() {
        // The `function.arguments` field is a *string* in the API, so a model that copies it
        // exactly produces nested JSON. Both forms must work.
        let extracted = extract_from_text(r#"{"name":"read_file","arguments":"{\"path\":\"a\"}"}"#);
        assert_eq!(extracted.source, Extraction::TextFallback);
        assert_eq!(extracted.calls[0].function.parse().unwrap()["path"], "a");
    }

    #[test]
    fn arguments_that_are_not_an_object_are_rejected() {
        let extracted = extract_from_text(r#"{"name":"shell","arguments":"ls -la"}"#);
        assert_eq!(extracted.source, Extraction::None);
        assert!(
            extracted
                .problem
                .as_deref()
                .unwrap_or("")
                .contains("not an object"),
            "{extracted:?}"
        );
    }

    #[test]
    fn an_unknown_shape_is_not_a_call() {
        let extracted = extract_from_text("I think the answer is 42.");
        assert_eq!(extracted.source, Extraction::None);
        assert!(extracted.is_empty());
    }

    #[test]
    fn an_empty_answer_is_reported_as_empty_not_as_unparseable() {
        // The two need different reactions: an empty answer is a model failure to retry, an
        // unparseable call is an observation for the model.
        let extracted = extract_from_text("   \n");
        assert_eq!(extracted.source, Extraction::None);
        assert!(extracted.problem.unwrap().contains("empty"));
    }

    #[test]
    fn a_failed_repair_names_the_partial_tool() {
        // So the observation can say "your `edit_file` call is missing `old`" rather than
        // "I could not understand you".
        let extracted = extract_from_text(r#"{"name":"edit_file","arguments":"not json"}"#);
        assert_eq!(extracted.source, Extraction::None);
        assert_eq!(extracted.partial_tool.as_deref(), Some("edit_file"));
        assert!(extracted.problem.unwrap().contains("edit_file"));
    }

    #[test]
    fn the_repair_does_not_touch_string_contents() {
        // A command legitimately contains a comma before a brace, a single quote and a brace.
        let arguments = r#"{"command": "printf 'a,}' && echo }"}"#;
        let repaired = repair_json(arguments);
        assert_eq!(repaired, arguments, "the payload was altered");
    }

    #[test]
    fn the_repair_is_idempotent() {
        let once = repair_json(r#"{"name":'x',}"#);
        let twice = repair_json(&once);
        assert_eq!(once, twice);
    }

    #[test]
    fn a_fallback_id_is_stable_across_runs() {
        // The journal matches `intent` and `completed` by id, so a replay of the same turn must
        // produce the same id.
        let first = extract_from_text(r#"{"name":"shell","arguments":{"command":"ls"}}"#);
        let second = extract_from_text(r#"{"name":"shell","arguments":{"command":"ls"}}"#);
        assert_eq!(first.calls[0].id, second.calls[0].id);
        assert!(first.calls[0].id.starts_with("text_"));
    }

    #[test]
    fn different_arguments_produce_different_ids() {
        let first = extract_from_text(r#"{"name":"shell","arguments":{"command":"ls"}}"#);
        let second = extract_from_text(r#"{"name":"shell","arguments":{"command":"pwd"}}"#);
        assert_ne!(first.calls[0].id, second.calls[0].id);
    }

    #[test]
    fn an_escaped_quote_inside_a_string_does_not_end_the_string() {
        let repaired = repair_json(r#"{"command":"echo \"hi\",}"}"#);
        let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(parsed["command"], "echo \"hi\",}");
    }

    #[test]
    fn a_question_is_distinguishable_from_a_broken_call() {
        assert!(is_question(&turn("Should I proceed with the rename?")));
        assert!(!is_question(&turn(
            "```json\n{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}\n```"
        )));
    }

    #[test]
    fn a_turn_with_a_native_call_is_not_a_question() {
        assert!(!is_question(&turn_with_call()));
    }

    #[test]
    fn needs_fallback_is_false_when_calls_exist_and_false_when_there_is_no_text() {
        assert!(!needs_fallback(&turn_with_call()));
        assert!(!needs_fallback(&turn("")));
        assert!(needs_fallback(&turn("some prose")));
    }

    #[test]
    fn an_unbalanced_array_inside_arguments_is_closed() {
        let extracted = extract_from_text(r#"{"name":"shell","arguments":{"command":"ls","env":["A=1","B=2"#);
        assert_eq!(extracted.source, Extraction::TextFallback);
        let args = extracted.calls[0].function.parse().unwrap();
        assert_eq!(args["env"][0], "A=1");
    }
}
