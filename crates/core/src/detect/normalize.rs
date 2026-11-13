//! Canonical form of a tool call, for the repeat detectors.
//!
//! # What is normalised, and what is deliberately not
//!
//! A model that re-sends `{"path": "./src/a.rs"}` after `{"path": "src/a.rs"}` has repeated
//! itself, and the detector has to see that. So paths inside the worktree become relative,
//! JSON keys are sorted, and runs of whitespace outside string values collapse.
//!
//! Two things are *not* normalised, on purpose:
//!
//! - **Whitespace inside a JSON string.** `write_file`'s `content` is a string value, and
//!   collapsing its whitespace would make two different files look like one call. Missing a
//!   cosmetic repeat is cheap; refusing to see that the model wrote a different file is not.
//! - **Fields that look volatile.** An earlier sketch dropped `ts` and `uuid` as noise. Rejected:
//!   those fields are what the tool will execute, so `{"ts": 1}` and `{"ts": 2}` are different
//!   commands that happen to look alike, not one command sent twice.
//!
//! When the arguments are not JSON at all, the fallback is a whitespace-collapsed trim. That is
//! what a malformed call looks like, and a malformed call repeated verbatim is still a repeat.

use std::path::Path;

use serde_json::Value;

/// The canonical form of one call's arguments.
#[must_use]
pub fn normalize(arguments: &str, worktree: Option<&Path>) -> String {
    let Ok(value) = serde_json::from_str::<Value>(arguments) else {
        return collapse_whitespace(arguments);
    };
    let value = match worktree.and_then(Path::to_str).filter(|root| !root.is_empty()) {
        Some(root) => relativize(value, root),
        None => value,
    };
    // `serde_json` without `preserve_order` stores objects in a `BTreeMap`, so this
    // serialisation is key-sorted. A test asserts that rather than trusting it.
    serde_json::to_string(&value).unwrap_or_else(|_| collapse_whitespace(arguments))
}

/// Replace the worktree prefix in every string value with nothing (or `.` for the root alone).
///
/// Only whole prefix components are rewritten: `/work/tree` inside `/work/treehouse` is a
/// different path, and rewriting it would make two distinct calls look identical.
fn relativize(value: Value, root: &str) -> Value {
    match value {
        Value::String(text) => Value::String(strip_root(&text, root)),
        Value::Array(items) => Value::Array(items.into_iter().map(|item| relativize(item, root)).collect()),
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, item)| (key, relativize(item, root)))
                .collect(),
        ),
        other => other,
    }
}

/// Rewrite every whole-component occurrence of `root` in `text` to a relative form.
///
/// `"/work/tree/src/a.rs"` becomes `"src/a.rs"`, and the bare root becomes `"."`, so the same
/// file named absolutely and relatively normalises to identical bytes.
fn strip_root(text: &str, root: &str) -> String {
    let root = root.trim_end_matches('/');
    if root.is_empty() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(index) = rest.find(root) {
        // The match must start at a component boundary: `..` in `/work/treehouse` must not be
        // treated as the root followed by a suffix.
        let before_ok = index == 0
            || !rest[..index].chars().next_back().is_some_and(|previous| {
                previous.is_alphanumeric() || matches!(previous, '_' | '-' | '.' | '/')
            });
        let after = &rest[index + root.len()..];
        let after_ok = after.is_empty() || after.starts_with('/');
        out.push_str(&rest[..index]);
        if before_ok && after_ok {
            if after.is_empty() {
                out.push('.');
                rest = after;
            } else {
                // Drop the separator too: `root + "/x"` is the relative path `x`.
                rest = &after[1..];
            }
        } else {
            out.push_str(root);
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// Trim, then collapse every run of whitespace to one space.
fn collapse_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = false;
    for character in text.trim().chars() {
        if character.is_whitespace() {
            space = true;
            continue;
        }
        if space && !out.is_empty() {
            out.push(' ');
        }
        space = false;
        out.push(character);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn root() -> PathBuf {
        PathBuf::from("/work/tree")
    }

    #[test]
    fn key_order_does_not_change_the_canonical_form() {
        // The detector compares what the model sent; a model that reorders keys is repeating
        // itself, and a canonical form that kept the order would miss every such repeat.
        let a = normalize(r#"{"b":2,"a":1}"#, None);
        let b = normalize(r#"{"a":1,"b":2}"#, None);
        assert_eq!(a, b);
    }

    #[test]
    fn paths_inside_the_worktree_become_relative() {
        let relative = normalize(r#"{"path":"src/a.rs"}"#, Some(&root()));
        let absolute = normalize(r#"{"path":"/work/tree/src/a.rs"}"#, Some(&root()));
        assert_eq!(relative, absolute);
    }

    #[test]
    fn a_sibling_directory_with_the_same_prefix_is_a_different_path() {
        // `/work/treehouse` must not be rewritten to `./house`: that would merge two calls that
        // touch two different directories.
        let inside = normalize(r#"{"path":"/work/tree/a"}"#, Some(&root()));
        let sibling = normalize(r#"{"path":"/work/treehouse/a"}"#, Some(&root()));
        assert_ne!(inside, sibling);
        assert!(sibling.contains("treehouse"), "{sibling}");
    }

    #[test]
    fn whitespace_inside_a_string_value_survives() {
        // `write_file` puts the file's body in a string; collapsing it would make two different
        // writes look like one, which is far worse than missing a cosmetic repeat.
        let a = normalize(r#"{"path":"a","content":"x  y"}"#, None);
        let b = normalize(r#"{"path":"a","content":"x y"}"#, None);
        assert_ne!(a, b);
    }

    #[test]
    fn json_whitespace_between_tokens_is_ignored() {
        let compact = normalize(r#"{"a":1,"b":[1,2]}"#, None);
        let pretty = normalize("{\n  \"a\": 1,\n  \"b\": [1, 2]\n}", None);
        assert_eq!(compact, pretty);
    }

    #[test]
    fn arguments_that_are_not_json_fall_back_to_collapsed_whitespace() {
        assert_eq!(normalize("  ls   -la\n", None), "ls -la");
        assert_eq!(normalize("ls -la", None), "ls -la");
        // Two different malformed payloads stay different.
        assert_ne!(normalize("{not json", None), normalize("{also json", None));
    }

    #[test]
    fn a_volatile_field_is_part_of_the_call() {
        // Rejected: dropping `ts` would call these two the same, but the tool executes both and
        // they do different things.
        assert_ne!(normalize(r#"{"ts":1}"#, None), normalize(r#"{"ts":2}"#, None));
    }

    #[test]
    fn an_empty_argument_object_is_stable() {
        assert_eq!(normalize("{}", None), "{}");
        assert_eq!(normalize(" { } ", None), "{}");
    }
}
