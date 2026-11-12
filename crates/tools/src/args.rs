//! Typed access to a tool call's arguments.
//!
//! # Why a wrapper and not `serde_json::Value`
//!
//! A tool body that reaches into a `Value` writes `args["path"].as_str().unwrap_or_default()`,
//! and the empty default is how a read tool ends up reading the worktree root and a write tool
//! truncating a file nobody named. Every accessor here returns `Result`, and the error is
//! already phrased for the model.
//!
//! # Why the schema check is not enough
//!
//! [`immortality_llm::schema`] validates the call before dispatch, so in the normal path these
//! getters never fire. They still exist because the validation lives in another crate and its
//! job is to explain a mistake, not to be the only thing between a model and the filesystem.
//! Defence that assumes its neighbour is correct is not defence.

use serde_json::Value;

use crate::ToolError;

/// The arguments of one call, as the model sent them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    value: Value,
}

impl Args {
    /// Wrap a parsed argument object.
    #[must_use]
    pub const fn new(value: Value) -> Self {
        Self { value }
    }

    /// An empty argument set, for a tool that takes none.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            value: Value::Object(serde_json::Map::new()),
        }
    }

    /// The underlying value, for a tool whose argument is a free-form object.
    #[must_use]
    pub const fn raw(&self) -> &Value {
        &self.value
    }

    /// A required string.
    ///
    /// # Errors
    /// Missing, or of another JSON type. The message names the field and what arrived, which is
    /// what lets the model fix the call instead of guessing.
    pub fn str(&self, name: &str) -> Result<&str, ToolError> {
        match self.value.get(name) {
            Some(Value::String(text)) => Ok(text),
            Some(other) => Err(ToolError::Arguments(format!(
                "`{name}` must be a string, got {}",
                kind(other)
            ))),
            None => Err(ToolError::Arguments(format!("`{name}` is required"))),
        }
    }

    /// An optional string, treating JSON `null` as absent.
    pub fn opt_str(&self, name: &str) -> Result<Option<&str>, ToolError> {
        match self.value.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(text)) => Ok(Some(text)),
            Some(other) => Err(ToolError::Arguments(format!(
                "`{name}` must be a string, got {}",
                kind(other)
            ))),
        }
    }

    /// An optional non-negative integer.
    ///
    /// `1.0` is accepted as `1`: JSON Schema says so and a model that writes it means one. A
    /// fractional value is rejected rather than rounded, because rounding `limit: 1.5` to `2`
    /// silently reads a line the caller did not ask for.
    pub fn opt_u64(&self, name: &str) -> Result<Option<u64>, ToolError> {
        match self.value.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(number)) => {
                if let Some(integer) = number.as_u64() {
                    return Ok(Some(integer));
                }
                match number.as_f64() {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    Some(value) if value >= 0.0 && value.fract() == 0.0 && value <= u64::MAX as f64 => {
                        Ok(Some(value as u64))
                    }
                    _ => Err(ToolError::Arguments(format!(
                        "`{name}` must be a non-negative integer, got {number}"
                    ))),
                }
            }
            Some(other) => Err(ToolError::Arguments(format!(
                "`{name}` must be a non-negative integer, got {}",
                kind(other)
            ))),
        }
    }

    /// An optional boolean.
    pub fn opt_bool(&self, name: &str) -> Result<Option<bool>, ToolError> {
        match self.value.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Bool(value)) => Ok(Some(*value)),
            Some(other) => Err(ToolError::Arguments(format!(
                "`{name}` must be a boolean, got {}",
                kind(other)
            ))),
        }
    }
}

impl Default for Args {
    fn default() -> Self {
        Self::empty()
    }
}

const fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(text: &str) -> Args {
        Args::new(serde_json::from_str(text).expect("valid JSON"))
    }

    #[test]
    fn a_present_string_is_returned() {
        assert_eq!(args(r#"{"path":"a.rs"}"#).str("path").unwrap(), "a.rs");
    }

    #[test]
    fn a_missing_string_says_it_is_required() {
        let error = args("{}").str("path").unwrap_err();
        assert!(error.to_string().contains("`path` is required"), "{error}");
    }

    #[test]
    fn a_wrong_type_names_what_arrived() {
        let error = args(r#"{"path":7}"#).str("path").unwrap_err();
        assert!(
            error.to_string().contains("must be a string, got a number"),
            "{error}"
        );
    }

    #[test]
    fn null_is_absent_for_an_optional_field() {
        assert_eq!(args(r#"{"limit":null}"#).opt_u64("limit").unwrap(), None);
        assert_eq!(args(r#"{"path":null}"#).opt_str("path").unwrap(), None);
        assert_eq!(args("{}").opt_str("path").unwrap(), None);
    }

    #[test]
    fn an_integer_written_as_a_float_is_accepted() {
        assert_eq!(args(r#"{"limit":3.0}"#).opt_u64("limit").unwrap(), Some(3));
    }

    #[test]
    fn a_fraction_is_rejected_rather_than_rounded() {
        // Rounding `1.5` to `2` would read a line the caller did not ask for.
        let error = args(r#"{"limit":1.5}"#).opt_u64("limit").unwrap_err();
        assert!(error.to_string().contains("non-negative integer"), "{error}");
    }

    #[test]
    fn a_negative_number_is_rejected() {
        assert!(args(r#"{"limit":-1}"#).opt_u64("limit").is_err());
    }

    #[test]
    fn a_boolean_is_read_and_null_is_absent() {
        assert_eq!(args(r#"{"all":true}"#).opt_bool("all").unwrap(), Some(true));
        assert_eq!(args(r#"{"all":null}"#).opt_bool("all").unwrap(), None);
        assert!(args(r#"{"all":"yes"}"#).opt_bool("all").is_err());
    }
}
