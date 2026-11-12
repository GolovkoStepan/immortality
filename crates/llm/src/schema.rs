//! Checking a tool call's arguments against the schema the model was given.
//!
//! # Why this is not a `jsonschema` dependency
//!
//! The schemas are ours — [`crate::wire::ToolDef`] values this project authors — so the keyword
//! set is a contract we control and can test exhaustively, not an arbitrary document from the
//! internet. A full draft-07 implementation would bring `regex`, `url` and a resolver to answer
//! questions nobody asks, and its error messages would then have to be translated into the shape
//! below anyway: the one consumer of these errors is a language model, and it needs
//! `old: missing required field`, not a JSON-Pointer plus a keyword name.
//!
//! The trade-off is named rather than hidden: an assertion keyword this module does not implement
//! would silently do nothing. [`schema_defects`] exists for that reason — the tool registry calls
//! it once at startup and fails loudly on a typo like `maxLenght` instead of shipping a check that
//! never fires. On the hot path, unknown keywords are ignored, which is what JSON Schema itself
//! says to do.
//!
//! # Why the invalid call is an observation and not an error
//!
//! The design's rule: an invalid call never breaks the loop. The model gets a structured
//! description of what was wrong *and the expected schema*, so the next step can correct it. It
//! costs one step and teaches the model something real. Rejecting the turn instead would spend
//! the step and teach it nothing.
//!
//! # Why a type mismatch stops the walk
//!
//! If `path` is a number where an object was expected, reporting "missing required field: old"
//! for every property underneath it is noise that buries the one fact the model needs. So a
//! mismatch is reported once, at its own path, and the subtree is not descended.

use serde_json::{Map, Value};

use crate::turn::{CalledFunction, ToolCall};

/// How many problems are reported before the rest are summarised.
///
/// A model that gets sixteen specific complaints can fix them; one that gets a hundred fixes the
/// first and resubmits. The cap is deliberately small enough to stay readable in a terminal.
pub const MAX_PROBLEMS: usize = 16;

/// Deepest nesting inspected. Tool arguments are flat objects; anything past this is either a
/// model writing nonsense or an attempt to blow the stack, and either way the walk should stop.
const MAX_DEPTH: usize = 64;

/// One thing wrong with the arguments.
///
/// `path` is a dotted-and-indexed location (`""` for the root, `env[2]`, `edits[0].old`), which is
/// what a model can act on. It is not an RFC 6901 JSON Pointer because a leading `/` and escaped
/// `~1` are noise for this reader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Problem {
    pub path: String,
    pub message: String,
}

impl Problem {
    fn new(path: &str, message: impl Into<String>) -> Self {
        Self {
            path: path.to_string(),
            message: message.into(),
        }
    }

    /// `path: message`, with the root path rendered as `<arguments>`.
    #[must_use]
    pub fn render(&self) -> String {
        if self.path.is_empty() {
            format!("<arguments>: {}", self.message)
        } else {
            format!("{}: {}", self.path, self.message)
        }
    }
}

/// Something wrong with a *schema*, not with arguments.
///
/// Only [`schema_defects`] produces these, and only at startup: the schema is ours, so a defect is
/// a programming error, whereas the arguments come from a model and are expected to be wrong
/// sometimes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Defect {
    /// Path to the offending keyword inside the schema, e.g. `properties.path.type`.
    pub path: String,
    pub keyword: &'static str,
    pub detail: String,
}

impl Defect {
    fn new(path: &str, keyword: &'static str, detail: impl Into<String>) -> Self {
        Self {
            path: path.to_string(),
            keyword,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for Defect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: `{}` {}", self.path, self.keyword, self.detail)
    }
}

/// Why a call's arguments cannot be used.
///
/// Three cases rather than one string because the loop reacts differently at the edges: a
/// non-JSON body is a model failure worth counting separately from a well-formed object with a
/// missing field, and the counters `invalid_toolcall` vs `text_fallback` are how degradation is
/// spotted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgError {
    /// The `arguments` string is not JSON at all.
    NotJson { detail: String },
    /// It parsed, but it is not an object.
    NotObject { got: &'static str },
    /// It is an object, and it violates the schema.
    Invalid { problems: Vec<Problem> },
}

impl ArgError {
    /// Whether the model produced something that at least parsed.
    ///
    /// A parse failure is the more serious signal: the model is not even emitting well-formed
    /// JSON, which usually means the profile or the prompt needs attention rather than a retry.
    #[must_use]
    pub const fn is_malformed(&self) -> bool {
        matches!(self, Self::NotJson { .. })
    }
}

/// Parse `call.arguments` and validate them against `schema`.
///
/// # `null` becomes `{}`
///
/// A model calling a tool that takes no arguments frequently sends the string `null` rather than
/// `{}`. Both mean "no arguments", so `null` is read as an empty object: the walk then reports the
/// missing required field precisely, where refusing the value outright would only say "not an
/// object" and leave the model guessing. This is a reading of an unambiguous intent, not a repair
/// that invents data — the distinction the fallback parser's rule turns on.
///
/// # Errors
/// An [`ArgError`] describing the first thing that makes the call unusable.
pub fn parse_arguments(call: &CalledFunction, schema: &Value) -> Result<Value, ArgError> {
    let parsed = call.parse().map_err(|error| ArgError::NotJson {
        detail: error.to_string(),
    })?;

    if parsed.is_null() {
        let empty = Value::Object(Map::new());
        return validate(schema, &empty)
            .map(|()| empty)
            .map_err(|problems| ArgError::Invalid { problems });
    }
    if !parsed.is_object() {
        return Err(ArgError::NotObject {
            got: json_kind(&parsed),
        });
    }
    validate(schema, &parsed).map_err(|problems| ArgError::Invalid { problems })?;
    Ok(parsed)
}

/// Validate `arguments` against `schema`.
///
/// # Errors
/// The problems found, capped at [`MAX_PROBLEMS`] plus a summary line.
pub fn validate(schema: &Value, arguments: &Value) -> Result<(), Vec<Problem>> {
    let mut problems = Vec::new();
    walk(schema, arguments, "", 0, &mut problems);
    if problems.is_empty() {
        return Ok(());
    }
    if problems.len() > MAX_PROBLEMS {
        let hidden = problems.len() - (MAX_PROBLEMS - 1);
        problems.truncate(MAX_PROBLEMS - 1);
        problems.push(Problem::new(
            "",
            format!("and {hidden} more problems of the same kind"),
        ));
    }
    Err(problems)
}

/// Check the schema itself, returning every defect.
///
/// An empty result means the schema is well-formed *as far as this validator is concerned*: every
/// keyword it found is implemented, and each has an acceptable shape. Call it once when a tool is
/// registered, not per call.
#[must_use]
pub fn schema_defects(schema: &Value) -> Vec<Defect> {
    let mut defects = Vec::new();
    check(schema, "", 0, &mut defects);
    defects
}

/// Render the observation sent back to the model in place of a tool result.
///
/// JSON rather than prose because the model is being handed data to reason about, and because the
/// same string goes into the journal where a later reader (human or agent) wants to query it. The
/// expected schema is included verbatim: it is the most precise description of what was wanted,
/// and it is what the model was already shown when it chose the call, so repeating it costs
/// nothing in confusion.
///
/// Deterministic: `serde_json`'s default map is ordered, so two runs produce identical bytes and a
/// golden test is possible.
#[must_use]
pub fn render_observation(call: &ToolCall, error: &ArgError, schema: &Value) -> String {
    let mut body = Map::new();
    body.insert(
        "error".to_string(),
        Value::String("invalid_tool_arguments".to_string()),
    );
    body.insert("tool".to_string(), Value::String(call.function.name.clone()));
    body.insert("call_id".to_string(), Value::String(call.id.clone()));

    match error {
        ArgError::NotJson { detail } => {
            body.insert(
                "detail".to_string(),
                Value::String(format!("the arguments are not valid JSON: {detail}")),
            );
            body.insert(
                "received".to_string(),
                Value::String(clip(&call.function.arguments, 200)),
            );
        }
        ArgError::NotObject { got } => {
            body.insert(
                "detail".to_string(),
                Value::String(format!(
                    "the arguments must be a JSON object, but they are a {got}"
                )),
            );
            body.insert(
                "received".to_string(),
                Value::String(clip(&call.function.arguments, 200)),
            );
        }
        ArgError::Invalid { problems } => {
            body.insert(
                "problems".to_string(),
                Value::Array(
                    problems
                        .iter()
                        .map(|problem| Value::String(problem.render()))
                        .collect(),
                ),
            );
        }
    }

    body.insert("expected_schema".to_string(), schema.clone());
    body.insert(
        "hint".to_string(),
        Value::String(format!(
            "Nothing was executed. Call `{}` again with arguments that match `expected_schema`.",
            call.function.name
        )),
    );
    Value::Object(body).to_string()
}

/// A one-line summary for a journal record or a log line.
///
/// Kept separate from [`render_observation`] because the journal stores a compact reason
/// alongside the full observation, and reading a log should not require parsing JSON.
#[must_use]
pub fn summarise(error: &ArgError) -> String {
    match error {
        ArgError::NotJson { .. } => "arguments are not JSON".to_string(),
        ArgError::NotObject { got } => format!("arguments are a {got}, not an object"),
        ArgError::Invalid { problems } => problems
            .first()
            .map_or_else(|| "arguments are invalid".to_string(), Problem::render),
    }
}

// ── the walk ────────────────────────────────────────────────────────────────────

fn walk(schema: &Value, value: &Value, path: &str, depth: usize, out: &mut Vec<Problem>) {
    if depth > MAX_DEPTH {
        out.push(Problem::new(
            path,
            format!("nesting is deeper than {MAX_DEPTH} levels"),
        ));
        return;
    }

    // `true` accepts everything and `false` accepts nothing, per the spec. A non-object schema is
    // a defect, and `schema_defects` reports it; on the hot path the useful behaviour is to accept,
    // because refusing every call over a malformed schema would stop the agent on a programming
    // error it cannot fix.
    let Some(map) = schema.as_object() else {
        if schema == &Value::Bool(false) {
            out.push(Problem::new(path, "no value is allowed here"));
        }
        return;
    };

    if let Some(types) = map.get("type")
        && let Some(actual) = check_type(types, value)
    {
        out.push(Problem::new(
            path,
            format!("expected {actual}, got {}", describe(value)),
        ));
        // See the module docs: descend no further into a subtree of the wrong type.
        return;
    }

    if let Some(allowed) = map.get("enum").and_then(Value::as_array)
        && !allowed.iter().any(|candidate| candidate == value)
    {
        out.push(Problem::new(
            path,
            format!("expected one of {allowed:?}, got {}", describe(value)),
        ));
    }

    check_bounds(map, value, path, out);

    if let Some(object) = value.as_object() {
        check_object(map, object, path, depth, out);
    } else if let Some(array) = value.as_array() {
        check_array(map, array, path, depth, out);
    }
}

fn check_object(
    schema: &Map<String, Value>,
    object: &Map<String, Value>,
    path: &str,
    depth: usize,
    out: &mut Vec<Problem>,
) {
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for name in required.iter().filter_map(Value::as_str) {
            if !object.contains_key(name) {
                out.push(Problem::new(
                    &join(path, name),
                    "missing required field".to_string(),
                ));
            }
        }
    }

    let properties = schema.get("properties").and_then(Value::as_object);
    if let Some(properties) = properties {
        for (name, sub_schema) in properties {
            if let Some(value) = object.get(name) {
                walk(sub_schema, value, &join(path, name), depth + 1, out);
            }
        }
    }

    // Extra fields are reported because `additionalProperties: false` is how a schema says "I
    // listed everything you may send", and a model inventing `force: true` must be told rather
    // than silently ignored — the extra field is often the model's real intent.
    if let Some(extra) = schema.get("additionalProperties") {
        for (name, value) in object {
            let known = properties.is_some_and(|properties| properties.contains_key(name));
            if known {
                continue;
            }
            match extra {
                Value::Bool(false) => out.push(Problem::new(
                    &join(path, name),
                    "unknown field; the schema does not allow it",
                )),
                Value::Object(_) | Value::Bool(true) => {
                    walk(extra, value, &join(path, name), depth + 1, out);
                }
                _ => {}
            }
        }
    }
}

fn check_array(
    schema: &Map<String, Value>,
    array: &[Value],
    path: &str,
    depth: usize,
    out: &mut Vec<Problem>,
) {
    if let Some(items) = schema.get("items") {
        for (index, value) in array.iter().enumerate() {
            walk(items, value, &format!("{path}[{index}]"), depth + 1, out);
        }
    }
}

fn check_bounds(schema: &Map<String, Value>, value: &Value, path: &str, out: &mut Vec<Problem>) {
    if let Some(number) = value.as_f64() {
        let minimum = schema.get("minimum").and_then(Value::as_f64);
        let maximum = schema.get("maximum").and_then(Value::as_f64);
        if let Some(minimum) = minimum
            && number < minimum
        {
            out.push(Problem::new(
                path,
                format!("must be at least {minimum}, got {number}"),
            ));
        }
        if let Some(maximum) = maximum
            && number > maximum
        {
            out.push(Problem::new(
                path,
                format!("must be at most {maximum}, got {number}"),
            ));
        }
    }

    if let Some(text) = value.as_str() {
        let length = u64::try_from(text.chars().count()).unwrap_or(u64::MAX);
        check_length(schema, length, "minLength", "at least", path, out);
        check_length(schema, length, "maxLength", "at most", path, out);
    }

    if let Some(array) = value.as_array() {
        let length = u64::try_from(array.len()).unwrap_or(u64::MAX);
        check_length(schema, length, "minItems", "at least", path, out);
        check_length(schema, length, "maxItems", "at most", path, out);
    }
}

fn check_length(
    schema: &Map<String, Value>,
    length: u64,
    keyword: &str,
    wording: &str,
    path: &str,
    out: &mut Vec<Problem>,
) {
    let Some(limit) = schema.get(keyword).and_then(Value::as_u64) else {
        return;
    };
    let too_short = keyword.starts_with("min");
    let violated = if too_short { length < limit } else { length > limit };
    if violated {
        out.push(Problem::new(
            path,
            format!("must be {wording} {limit}, got {length}"),
        ));
    }
}

/// The expected types, if `value` is not one of them.
fn check_type(types: &Value, value: &Value) -> Option<String> {
    let names: Vec<&str> = match types {
        Value::String(name) => vec![name.as_str()],
        Value::Array(names) => names.iter().filter_map(Value::as_str).collect(),
        _ => return None,
    };
    if names.is_empty() {
        return None;
    }
    if names.iter().any(|name| type_matches(name, value)) {
        return None;
    }
    Some(join_with_articles(&names))
}

fn type_matches(name: &str, value: &Value) -> bool {
    match name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "number" => value.is_number(),
        // JSON Schema counts `1.0` as an integer, and a model that writes `"limit": 1.0` means
        // one. Accepting it is the spec's answer and the useful one.
        "integer" => value.as_f64().is_some_and(|number| number.fract() == 0.0),
        _ => false,
    }
}

/// `a string`, `an integer`, `a string or null`.
fn join_with_articles(names: &[&str]) -> String {
    let rendered: Vec<String> = names.iter().map(|name| with_article(name)).collect();
    rendered.join(" or ")
}

fn with_article(name: &str) -> String {
    let article = match name {
        "integer" | "object" | "array" => "an",
        _ => "a",
    };
    format!("{article} {name}")
}

/// A short rendering of a value for an error message, bounded so that a 40 KB string in the wrong
/// place does not become a 40 KB error.
fn describe(value: &Value) -> String {
    match value {
        Value::String(text) => format!("string {}", clip(text, 40)),
        Value::Array(items) => format!("array of {}", items.len()),
        Value::Object(map) => format!("object with {} field(s)", map.len()),
        Value::Number(number) => format!("number {number}"),
        Value::Bool(boolean) => format!("boolean {boolean}"),
        Value::Null => "null".to_string(),
    }
}

const fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// A path inside a *schema*, which is always rooted: `properties.path.type`, not `.path.type`.
fn schema_path(path: &str, name: &str) -> String {
    if path.is_empty() {
        name.to_string()
    } else {
        format!("{path}.{name}")
    }
}

fn join(path: &str, name: &str) -> String {
    if path.is_empty() {
        name.to_string()
    } else {
        format!("{path}.{name}")
    }
}

fn clip(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let head: String = text.chars().take(limit).collect();
    format!("{head}…")
}

// ── schema self-check ───────────────────────────────────────────────────────────

/// Keywords this validator implements.
///
/// Listed explicitly so that a new schema keyword is a visible edit here rather than a silent
/// no-op: `schema_defects` reports anything outside this set.
const KNOWN: [&str; 14] = [
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "description",
    "title",
];

/// Type names this validator understands.
const TYPE_NAMES: [&str; 7] = [
    "object", "array", "string", "boolean", "null", "number", "integer",
];

fn check(schema: &Value, path: &str, depth: usize, out: &mut Vec<Defect>) {
    if depth > MAX_DEPTH {
        out.push(Defect::new(path, "nesting", "is deeper than 64 levels"));
        return;
    }
    let Some(map) = schema.as_object() else {
        // A boolean schema is legal; anything else is not a schema.
        if !schema.is_boolean() {
            out.push(Defect::new(
                path,
                "schema",
                format!("must be an object or a boolean, is {}", json_kind(schema)),
            ));
        }
        return;
    };

    for (keyword, value) in map {
        if !KNOWN.contains(&keyword.as_str()) {
            out.push(Defect::new(
                path,
                "unknown",
                format!("keyword `{keyword}` is not implemented by this validator"),
            ));
            continue;
        }
        match keyword.as_str() {
            "type" => check_type_keyword(value, path, out),
            "properties" | "patternProperties" => {
                if let Some(properties) = value.as_object() {
                    for (name, sub_schema) in properties {
                        check(
                            sub_schema,
                            &schema_path(path, &format!("properties.{name}")),
                            depth + 1,
                            out,
                        );
                    }
                } else {
                    out.push(Defect::new(path, "properties", "must be an object"));
                }
            }
            "items" => check(value, &schema_path(path, "items"), depth + 1, out),
            "additionalProperties" => {
                if !value.is_boolean() {
                    check(value, &schema_path(path, "additionalProperties"), depth + 1, out);
                }
            }
            "required" => {
                let valid = value
                    .as_array()
                    .is_some_and(|names| names.iter().all(Value::is_string));
                if !valid {
                    out.push(Defect::new(path, "required", "must be an array of strings"));
                }
            }
            "enum" => {
                let valid = value.as_array().is_some_and(|items| !items.is_empty());
                if !valid {
                    out.push(Defect::new(path, "enum", "must be a non-empty array"));
                }
            }
            "minimum" | "maximum" => {
                if !value.is_number() {
                    out.push(Defect::new(
                        path,
                        "bound",
                        format!("`{keyword}` must be a number"),
                    ));
                }
            }
            "minLength" | "maxLength" | "minItems" | "maxItems"
                if value.as_u64().is_none() =>
            {
                out.push(Defect::new(
                    path,
                    "limit",
                    format!("`{keyword}` must be a non-negative integer"),
                ));
            }
            // `description` and `title` are documentation and are sent to the model as-is.
            _ => {}
        }
    }
}

fn check_type_keyword(value: &Value, path: &str, out: &mut Vec<Defect>) {
    let names: Vec<&str> = match value {
        Value::String(name) => vec![name.as_str()],
        Value::Array(items) => items.iter().filter_map(Value::as_str).collect(),
        _ => {
            out.push(Defect::new(
                path,
                "type",
                "must be a string or an array of strings",
            ));
            return;
        }
    };
    for name in names {
        if !TYPE_NAMES.contains(&name) {
            out.push(Defect::new(
                path,
                "type",
                format!("`{name}` is not a JSON Schema type"),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The shape of a real tool schema: an object with two required strings.
    fn edit_file_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "file to edit"},
                "old": {"type": "string"},
                "new": {"type": "string"},
                "replace_all": {"type": "boolean"}
            },
            "required": ["path", "old", "new"],
            "additionalProperties": false
        })
    }

    fn call(name: &str, arguments: &str) -> CalledFunction {
        CalledFunction {
            name: name.to_string(),
            arguments: arguments.to_string(),
        }
    }

    #[test]
    fn valid_arguments_pass() {
        let parsed = parse_arguments(
            &call(
                "edit_file",
                r#"{"path":"a.rs","old":"x","new":"y","replace_all":true}"#,
            ),
            &edit_file_schema(),
        )
        .expect("valid");
        assert_eq!(parsed["path"], "a.rs");
    }

    #[test]
    fn a_missing_required_field_is_named() {
        let error = parse_arguments(
            &call("edit_file", r#"{"path":"a.rs","new":"y"}"#),
            &edit_file_schema(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ArgError::Invalid {
                problems: vec![Problem::new("old", "missing required field")]
            }
        );
    }

    #[test]
    fn every_missing_field_is_reported_together() {
        // Reporting one per step would cost a step per field. The model is told everything at
        // once so it can fix the call in one go.
        let error = parse_arguments(&call("edit_file", "{}"), &edit_file_schema()).unwrap_err();
        let ArgError::Invalid { problems } = error else {
            panic!("expected problems");
        };
        assert_eq!(problems.len(), 3);
        let rendered: Vec<String> = problems.iter().map(Problem::render).collect();
        // In the order `required` lists them, which is the order the schema author chose to
        // emphasise — and the order the model is most likely to fix them in.
        assert_eq!(
            rendered,
            [
                "path: missing required field",
                "old: missing required field",
                "new: missing required field"
            ]
        );
    }

    #[test]
    fn a_wrong_type_says_what_was_expected_and_what_arrived() {
        let error = parse_arguments(
            &call("edit_file", r#"{"path":7,"old":"x","new":"y"}"#),
            &edit_file_schema(),
        )
        .unwrap_err();
        let ArgError::Invalid { problems } = error else {
            panic!("expected problems");
        };
        assert_eq!(problems.len(), 1, "one problem, not one per field");
        assert_eq!(problems[0].path, "path");
        assert!(
            problems[0].message.contains("expected a string, got number 7"),
            "{}",
            problems[0].message
        );
    }

    #[test]
    fn an_unknown_field_is_reported_when_the_schema_forbids_it() {
        // A model sending `force: true` has a real intent, and silently ignoring it makes the
        // next step inexplicable. Saying "unknown field" is what lets it notice.
        let error = parse_arguments(
            &call("edit_file", r#"{"path":"a","old":"x","new":"y","force":true}"#),
            &edit_file_schema(),
        )
        .unwrap_err();
        let ArgError::Invalid { problems } = error else {
            panic!("expected problems");
        };
        assert_eq!(problems[0].path, "force");
        assert!(problems[0].message.contains("unknown field"));
    }

    #[test]
    fn an_unknown_field_is_allowed_when_the_schema_says_nothing() {
        // Without `additionalProperties` the spec allows extras, and a schema that forgot the
        // keyword must not become stricter than it says.
        let schema = json!({"type": "object", "properties": {"path": {"type": "string"}}});
        assert!(parse_arguments(&call("read_file", r#"{"path":"a","extra":1}"#), &schema).is_ok());
    }

    #[test]
    fn a_nested_object_reports_the_full_path() {
        let schema = json!({
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"old": {"type": "string"}},
                        "required": ["old"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["edits"]
        });
        let error = parse_arguments(
            &call("multi_edit", r#"{"edits":[{"old":"a"},{"new":"b"}]}"#),
            &schema,
        )
        .unwrap_err();
        let ArgError::Invalid { problems } = error else {
            panic!("expected problems");
        };
        let paths: Vec<&str> = problems.iter().map(|problem| problem.path.as_str()).collect();
        // Properties are walked before `additionalProperties`, and `old` is required while
        // `new` is merely unknown — so the model is told about the missing field it must supply
        // first, then that `new` does not belong here.
        assert_eq!(paths, ["edits[1].old", "edits[1].new"]);
    }

    #[test]
    fn a_type_mismatch_on_a_container_does_not_report_every_field_below_it() {
        // The noise this avoids: "missing required field" for each property of an object that
        // was not an object at all.
        let schema = json!({
            "type": "object",
            "properties": {"opts": edit_file_schema()},
            "required": ["opts"]
        });
        let error = parse_arguments(&call("t", r#"{"opts":"not an object"}"#), &schema).unwrap_err();
        let ArgError::Invalid { problems } = error else {
            panic!("expected problems");
        };
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].path, "opts");
    }

    #[test]
    fn an_integer_field_rejects_a_fraction() {
        let schema = json!({"type": "object", "properties": {"limit": {"type": "integer"}}});
        let error = parse_arguments(&call("t", r#"{"limit":1.5}"#), &schema).unwrap_err();
        let ArgError::Invalid { problems } = error else {
            panic!("expected problems");
        };
        assert!(problems[0].message.contains("expected an integer"));
    }

    #[test]
    fn an_integer_field_accepts_one_written_as_a_float() {
        // JSON Schema says `1.0` is an integer, and a model writing it means one.
        let schema = json!({"type": "object", "properties": {"limit": {"type": "integer"}}});
        assert!(parse_arguments(&call("t", r#"{"limit":1.0}"#), &schema).is_ok());
    }

    #[test]
    fn a_number_field_accepts_an_integer() {
        let schema = json!({"type": "object", "properties": {"n": {"type": "number"}}});
        assert!(parse_arguments(&call("t", r#"{"n":3}"#), &schema).is_ok());
    }

    #[test]
    fn bounds_are_checked() {
        let schema = json!({
            "type": "object",
            "properties": {
                "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
                "limit": {"type": "integer", "minimum": 1}
            }
        });
        let error = parse_arguments(&call("t", r#"{"timeout":0,"limit":-5}"#), &schema).unwrap_err();
        let ArgError::Invalid { problems } = error else {
            panic!("expected problems");
        };
        assert_eq!(problems.len(), 2);
        assert!(problems.iter().any(|p| p.message.contains("at least 1, got 0")));
        assert!(problems.iter().any(|p| p.message.contains("at least 1, got -5")));

        let error = parse_arguments(&call("t", r#"{"timeout":601}"#), &schema).unwrap_err();
        let ArgError::Invalid { problems } = error else {
            panic!("expected problems");
        };
        assert!(problems[0].message.contains("at most 600"));
    }

    #[test]
    fn string_length_bounds_count_characters_not_bytes() {
        // Cyrillic is two bytes per char; counting bytes would reject a legal value.
        let schema = json!({
            "type": "object",
            "properties": {"key": {"type": "string", "maxLength": 3}}
        });
        assert!(parse_arguments(&call("t", r#"{"key":"абв"}"#), &schema).is_ok());
        assert!(parse_arguments(&call("t", r#"{"key":"абвг"}"#), &schema).is_err());
    }

    #[test]
    fn array_length_bounds_are_checked() {
        let schema = json!({
            "type": "object",
            "properties": {"options": {"type": "array", "minItems": 1, "maxItems": 2}}
        });
        assert!(parse_arguments(&call("t", r#"{"options":["a"]}"#), &schema).is_ok());
        assert!(parse_arguments(&call("t", r#"{"options":[]}"#), &schema).is_err());
        assert!(parse_arguments(&call("t", r#"{"options":["a","b","c"]}"#), &schema).is_err());
    }

    #[test]
    fn enum_rejects_a_value_outside_it() {
        let schema = json!({
            "type": "object",
            "properties": {"mode": {"type": "string", "enum": ["read", "write"]}}
        });
        assert!(parse_arguments(&call("t", r#"{"mode":"read"}"#), &schema).is_ok());
        let error = parse_arguments(&call("t", r#"{"mode":"append"}"#), &schema).unwrap_err();
        let ArgError::Invalid { problems } = error else {
            panic!("expected problems");
        };
        assert!(problems[0].message.contains("expected one of"));
    }

    #[test]
    fn a_union_type_accepts_each_member() {
        let schema = json!({
            "type": "object",
            "properties": {"offset": {"type": ["integer", "null"]}}
        });
        assert!(parse_arguments(&call("t", r#"{"offset":5}"#), &schema).is_ok());
        assert!(parse_arguments(&call("t", r#"{"offset":null}"#), &schema).is_ok());
        assert!(parse_arguments(&call("t", r#"{"offset":"5"}"#), &schema).is_err());
    }

    #[test]
    fn null_arguments_mean_no_arguments() {
        // A model calling a zero-argument tool sends `null` more often than `{}`. Both mean the
        // same thing, and reading `null` as `{}` lets the missing-field check say something
        // useful (`path: missing required field`) instead of `not an object`.
        let schema = json!({"type": "object", "properties": {}, "required": ["path"]});
        let error = parse_arguments(&call("t", "null"), &schema).unwrap_err();
        assert_eq!(
            error,
            ArgError::Invalid {
                problems: vec![Problem::new("path", "missing required field")]
            }
        );

        let empty = json!({"type": "object", "properties": {}});
        assert!(parse_arguments(&call("t", "null"), &empty).is_ok());
    }

    #[test]
    fn non_json_arguments_are_their_own_case() {
        let error = parse_arguments(&call("t", "{not json"), &edit_file_schema()).unwrap_err();
        assert!(error.is_malformed(), "{error:?}");
        assert!(matches!(error, ArgError::NotJson { .. }));
    }

    #[test]
    fn a_non_object_argument_is_reported_by_kind() {
        let error = parse_arguments(&call("t", r#"["a","b"]"#), &edit_file_schema()).unwrap_err();
        assert_eq!(error, ArgError::NotObject { got: "array" });
        assert!(!error.is_malformed());
    }

    #[test]
    fn the_observation_carries_the_schema_and_says_nothing_ran() {
        let schema = edit_file_schema();
        let call = ToolCall::function("call_1", "edit_file", r#"{"path":"a"}"#);
        let error = ArgError::Invalid {
            problems: vec![Problem::new("old", "missing required field")],
        };
        let rendered = render_observation(&call, &error, &schema);
        let body: Value = serde_json::from_str(&rendered).unwrap();

        assert_eq!(body["error"], "invalid_tool_arguments");
        assert_eq!(body["tool"], "edit_file");
        assert_eq!(body["problems"][0], "old: missing required field");
        assert_eq!(body["expected_schema"], schema);
        assert!(
            body["hint"].as_str().unwrap().contains("Nothing was executed"),
            "{rendered}"
        );
    }

    #[test]
    fn the_observation_for_unparseable_arguments_echoes_what_arrived() {
        // The model cannot fix a syntax error it cannot see.
        let call = ToolCall::function("call_1", "read_file", r#"{"path": "a.rs""#);
        let rendered = render_observation(
            &call,
            &ArgError::NotJson {
                detail: "EOF while parsing".to_string(),
            },
            &json!({"type": "object"}),
        );
        let body: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(body["received"], r#"{"path": "a.rs""#);
        assert!(body["detail"].as_str().unwrap().contains("not valid JSON"));
    }

    #[test]
    fn the_observation_of_a_huge_argument_is_bounded() {
        let call = ToolCall::function(
            "call_1",
            "write_file",
            format!("{{\"content\": \"{}\"", "x".repeat(5000)),
        );
        let rendered = render_observation(
            &call,
            &ArgError::NotJson {
                detail: "unexpected end of input".to_string(),
            },
            &json!({"type": "object"}),
        );
        assert!(rendered.len() < 1000, "{}", rendered.len());
        assert!(rendered.contains('…'));
    }

    #[test]
    fn rendering_is_deterministic() {
        // Same input, same bytes: goldens and journal diffs depend on it.
        let schema = edit_file_schema();
        let call = ToolCall::function("call_1", "edit_file", "{}");
        let first = ArgError::Invalid {
            problems: vec![Problem::new("path", "missing required field")],
        };
        let second = first.clone();
        assert_eq!(
            render_observation(&call, &first, &schema),
            render_observation(&call, &second, &schema)
        );
    }

    #[test]
    fn too_many_problems_are_summarised_rather_than_swamping_the_context() {
        let schema = json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        });
        let mut arguments = serde_json::Map::new();
        for index in 0..40 {
            arguments.insert(format!("field_{index:02}"), json!(index));
        }
        let error = parse_arguments(&call("t", &Value::Object(arguments).to_string()), &schema).unwrap_err();
        let ArgError::Invalid { problems } = error else {
            panic!("expected problems");
        };
        assert_eq!(problems.len(), MAX_PROBLEMS);
        assert!(problems.last().unwrap().message.contains("more problems"));
    }

    #[test]
    fn the_summary_is_one_line() {
        let error = ArgError::Invalid {
            problems: vec![
                Problem::new("path", "missing required field"),
                Problem::new("old", "missing required field"),
            ],
        };
        assert_eq!(summarise(&error), "path: missing required field");
        assert_eq!(
            summarise(&ArgError::NotObject { got: "array" }),
            "arguments are a array, not an object"
        );
    }

    #[test]
    fn a_root_level_error_renders_a_readable_path() {
        assert_eq!(
            Problem::new("", "unknown field").render(),
            "<arguments>: unknown field"
        );
    }

    #[test]
    fn a_well_formed_schema_has_no_defects() {
        assert_eq!(schema_defects(&edit_file_schema()), Vec::new());
        assert_eq!(schema_defects(&json!(true)), Vec::new());
        assert_eq!(
            schema_defects(&json!({
                "type": "array",
                "items": {"type": "string", "minLength": 1}
            })),
            Vec::new()
        );
    }

    #[test]
    fn a_misspelled_keyword_is_a_defect() {
        // This is the whole reason `schema_defects` exists: on the hot path an unknown keyword is
        // ignored, so without this check `maxLenght` would silently enforce nothing forever.
        let defects = schema_defects(&json!({
            "type": "object",
            "properties": {"key": {"type": "string", "maxLenght": 3}}
        }));
        assert_eq!(defects.len(), 1);
        assert_eq!(defects[0].keyword, "unknown");
        assert!(defects[0].detail.contains("maxLenght"), "{defects:?}");
        assert_eq!(defects[0].path, "properties.key");
    }

    #[test]
    fn an_unknown_type_name_is_a_defect() {
        let defects = schema_defects(&json!({"type": "strng"}));
        assert_eq!(defects.len(), 1);
        assert!(defects[0].detail.contains("strng"), "{defects:?}");
    }

    #[test]
    fn required_must_be_an_array_of_strings() {
        assert_eq!(
            schema_defects(&json!({"type": "object", "required": "path"})).len(),
            1
        );
        assert_eq!(
            schema_defects(&json!({"type": "object", "required": ["path", 7]})).len(),
            1
        );
    }

    #[test]
    fn an_empty_enum_is_a_defect() {
        // It can never match anything, so it is always a bug rather than a strict schema.
        assert_eq!(schema_defects(&json!({"type": "string", "enum": []})).len(), 1);
    }

    #[test]
    fn a_limit_that_is_not_a_count_is_a_defect() {
        assert_eq!(
            schema_defects(&json!({"type": "string", "maxLength": -1})).len(),
            1
        );
        assert_eq!(
            schema_defects(&json!({"type": "string", "maxLength": "3"})).len(),
            1
        );
    }

    #[test]
    fn the_defect_message_names_the_path_and_the_keyword() {
        let defects = schema_defects(&json!({
            "type": "object",
            "properties": {"a": {"type": "object", "properties": {"b": {"type": 7}}}}
        }));
        let text = defects[0].to_string();
        assert!(text.contains("properties.a.properties.b"), "{text}");
        assert!(text.contains("`type`"), "{text}");
    }

    #[test]
    fn description_and_title_are_sent_through_untouched() {
        // They are documentation for the model, not assertions; flagging them would make every
        // real schema a defect.
        assert_eq!(
            schema_defects(&json!({
                "type": "object",
                "title": "read a file",
                "description": "returns numbered lines",
                "properties": {"path": {"type": "string", "description": "the file"}}
            })),
            Vec::new()
        );
    }
}
