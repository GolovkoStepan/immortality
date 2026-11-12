//! Writing files and editing them in place.
//!
//! # Why `edit_file` refuses ambiguity instead of taking the first match
//!
//! `old` appearing twice means the model's mental model of the file is wrong — it thinks it knows
//! which `if` it is patching and it does not. Taking the first occurrence turns that into a silent
//! edit of the wrong branch, and the consequence surfaces ten steps later as a test failure whose
//! cause is nowhere near the change. Refusing costs one step and names the line numbers.
//!
//! # Why matching tolerates line-ending differences
//!
//! A model on a normalised context writes `\n`. A repository checked out on Windows — or with a
//! single file that came from there — has `\r\n`. A byte-exact `contains` then reports "old not
//! found" for text the model can plainly see, and the model retries with the same string, and the
//! loop detector fires. So `old` is matched with `\r?\n` at every line break, and `new` is written
//! back using the file's own convention so the diff does not become a whole-file rewrite.
//!
//! # Why writes are atomic
//!
//! A crash inside `write` leaves a half-written source file, and the next run's `read_file` shows
//! a file that does not compile for reasons the model cannot see. Writing to a sibling temp file,
//! syncing it, and renaming over the target means the file is either the old one or the new one,
//! never a torn mixture. Same-volume `rename` is atomic on APFS.
//!
//! # Why `write_file` is `ByContent`
//!
//! Recovery may replay a call whose effect is unknown. For `write_file` the check is exact and
//! cheap: if the file already holds the requested content, the effect happened. That is the
//! definition of `ByContent`, and it is why this pair is not `Unknown`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use immortality_core::protocol::Idempotency;
use regex::Regex;
use serde_json::{Value, json};

use crate::Effect;
use crate::args::Args;
use crate::policy::PathUse;
use crate::tool::{Tool, ToolCtx, ToolError, ToolResult};

/// Largest `write_file` payload accepted.
///
/// A model that sends more than this is not writing a source file; it is pasting a dataset into
/// the repository, and the failure should name the limit rather than fill the disk.
pub const MAX_WRITE_BYTES: usize = 8 * 1024 * 1024;

/// Largest file `edit_file` will rewrite.
///
/// The whole file is read to find the match, so this is about memory and step time, not about how
/// much may change.
pub const MAX_EDIT_BYTES: u64 = 8 * 1024 * 1024;

fn write_schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to write, relative to the worktree root. Parent directories are created."
                },
                "content": {
                    "type": "string",
                    "description": "The complete new contents of the file."
                }
            },
            "required": ["path", "content"],
            "additionalProperties": false
        })
    })
}

fn edit_schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to edit, relative to the worktree root."
                },
                "old": {
                    "type": "string",
                    "description": "Exact text to replace, including indentation. Must occur exactly once unless `replace_all` is true."
                },
                "new": {
                    "type": "string",
                    "description": "Replacement text. Must differ from `old`."
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence instead of refusing when `old` is ambiguous. Defaults to false."
                }
            },
            "required": ["path", "old", "new"],
            "additionalProperties": false
        })
    })
}

/// Write a whole file.
#[derive(Debug, Default, Clone, Copy)]
pub struct WriteFile;

impl Tool for WriteFile {
    fn name(&self) -> &'static str {
        "write_file"
    }

    fn schema(&self) -> &'static Value {
        write_schema()
    }

    fn description(&self) -> &'static str {
        "Write a complete file, creating parent directories. Use `edit_file` to change part of one."
    }

    fn effect(&self, _args: &Args) -> Effect {
        Effect::Mutate
    }

    fn idempotency(&self, _args: &Args) -> Idempotency {
        Idempotency::ByContent
    }

    fn paths(&self, args: &Args) -> Vec<PathUse> {
        args.opt_str("path")
            .ok()
            .flatten()
            .map(|path| vec![PathUse::write(path)])
            .unwrap_or_default()
    }

    fn execute(&self, ctx: &ToolCtx, args: Args) -> Result<ToolResult, ToolError> {
        let raw = args.str("path")?;
        let content = args.str("content")?;
        if content.len() > MAX_WRITE_BYTES {
            return Err(ToolError::Arguments(format!(
                "`content` is {} bytes; the limit is {MAX_WRITE_BYTES}. Write a smaller file or \
                 append in pieces with `edit_file`.",
                content.len()
            )));
        }

        let path = ctx.write_path(raw)?;

        // Creating the parent directories is part of "write this file", and doing it here means a
        // model cannot get a failure that says "no such directory" with no way forward.
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|error| ToolError::io(parent, &error))?;
        }

        let previous = std::fs::read(&path).ok();
        let existed = previous.is_some();
        let unchanged = previous.as_deref() == Some(content.as_bytes());

        if !unchanged {
            write_atomically(&path, content.as_bytes())?;
        }

        let lines = content.lines().count();
        let observation = if unchanged {
            format!("{raw} already had exactly that content; nothing changed.")
        } else if existed {
            format!(
                "Wrote {raw} ({lines} line(s), {} bytes), replacing the old contents.",
                content.len()
            )
        } else {
            format!("Created {raw} ({lines} line(s), {} bytes).", content.len())
        };

        Ok(ToolResult::new(
            observation,
            json!({
                "path": raw,
                "bytes": content.len(),
                "lines": lines,
                "created": !existed,
                "unchanged": unchanged,
            }),
        ))
    }
}

/// Replace an exact piece of text.
#[derive(Debug, Default, Clone, Copy)]
pub struct EditFile;

impl Tool for EditFile {
    fn name(&self) -> &'static str {
        "edit_file"
    }

    fn schema(&self) -> &'static Value {
        edit_schema()
    }

    fn description(&self) -> &'static str {
        "Replace an exact piece of text in a file. Refuses when `old` is missing or appears more than once."
    }

    fn effect(&self, _args: &Args) -> Effect {
        Effect::Mutate
    }

    fn idempotency(&self, _args: &Args) -> Idempotency {
        Idempotency::ByContent
    }

    fn paths(&self, args: &Args) -> Vec<PathUse> {
        args.opt_str("path")
            .ok()
            .flatten()
            .map(|path| vec![PathUse::write(path)])
            .unwrap_or_default()
    }

    fn execute(&self, ctx: &ToolCtx, args: Args) -> Result<ToolResult, ToolError> {
        let raw = args.str("path")?;
        let old = args.str("old")?;
        let new = args.str("new")?;
        let replace_all = args.opt_bool("replace_all")?.unwrap_or(false);
        check_replacement(old, new)?;

        let path = ctx.write_path(raw)?;

        let metadata = std::fs::metadata(&path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                ToolError::Refused(missing_file(raw))
            } else {
                ToolError::io(&path, &error)
            }
        })?;
        if metadata.len() > MAX_EDIT_BYTES {
            return Err(ToolError::Arguments(format!(
                "`{raw}` is {} bytes; `edit_file` is capped at {MAX_EDIT_BYTES}. Read the part you \
                 need and rewrite it with `write_file`.",
                metadata.len()
            )));
        }

        let bytes = std::fs::read(&path).map_err(|error| ToolError::io(&path, &error))?;
        if crate::read::looks_binary(&bytes) {
            return Err(ToolError::Refused(
                json!({
                    "error": "binary_file",
                    "path": raw,
                    "hint": "`edit_file` only edits text files; rewriting a binary would corrupt it."
                })
                .to_string(),
            ));
        }
        let text = String::from_utf8(bytes).map_err(|_| {
            ToolError::Refused(
                json!({
                    "error": "not_utf8",
                    "path": raw,
                    "hint": "The file is not valid UTF-8, so an exact text replacement cannot be \
                             trusted to round-trip. Rewrite it with `write_file` if that is intended."
                })
                .to_string(),
            )
        })?;

        let crlf = text.contains("\r\n");
        let pattern = match_any_line_ending(old);
        let regex = Regex::new(&pattern).map_err(|error| {
            // Unreachable in practice: the pattern is built from escaped text. Kept as a refusal
            // rather than a panic because a panic inside a tool kills the whole attempt.
            ToolError::Refused(format!("the replacement pattern is not valid: {error}"))
        })?;

        let positions: Vec<usize> = regex.find_iter(&text).map(|found| found.start()).collect();
        if positions.is_empty() {
            return Err(ToolError::Refused(not_found(raw, old, &text, crlf)));
        }
        if positions.len() > 1 && !replace_all {
            return Err(ToolError::Refused(ambiguous(raw, old, &positions, &text)));
        }

        // Normalise `new` to the file's own line endings, so editing a CRLF file does not turn the
        // touched hunk into a mixed-ending mess.
        let replacement = if crlf {
            new.replace("\r\n", "\n").replace('\n', "\r\n")
        } else {
            new.replace("\r\n", "\n")
        };

        let updated = if replace_all {
            regex
                .replace_all(&text, regex::NoExpand(&replacement))
                .into_owned()
        } else {
            regex.replace(&text, regex::NoExpand(&replacement)).into_owned()
        };

        if updated == text {
            return Err(ToolError::Refused(format!(
                "`{raw}` was already in the requested state; nothing changed."
            )));
        }

        write_atomically(&path, updated.as_bytes())?;

        let count = if replace_all { positions.len() } else { 1 };
        let line_numbers: Vec<usize> = positions.iter().map(|offset| line_of(&text, *offset)).collect();

        Ok(ToolResult::new(
            format!(
                "Edited {raw}: {count} replacement(s) at line(s) {}. The file is now {} line(s).",
                join_numbers(&line_numbers),
                updated.lines().count()
            ),
            json!({
                "path": raw,
                "replacements": count,
                "lines": line_numbers,
                "bytes": updated.len(),
                "line_endings": if crlf { "crlf" } else { "lf" },
            }),
        ))
    }
}

/// Reject a replacement that cannot mean anything.
///
/// Both cases are errors in the *call*, not in the file: an empty `old` matches everywhere, and an
/// identical `old`/`new` is a step that changes nothing while looking like progress — which is how
/// an agent spins without the loop detector ever firing.
fn check_replacement(old: &str, new: &str) -> Result<(), ToolError> {
    if old.is_empty() {
        return Err(ToolError::Arguments(
            "`old` is empty; an empty pattern would match at every position. Give the exact text \
             to replace."
                .to_string(),
        ));
    }
    if old == new {
        return Err(ToolError::Arguments(
            "`old` and `new` are identical, so nothing would change. Remove this call or fix `new`."
                .to_string(),
        ));
    }
    Ok(())
}

/// Escape `old` and make every line break match either ending.
///
/// `regex::escape` leaves a real newline character in the pattern rather than the two characters
/// `\` and `n`, so the substitution is on the whitespace itself. Both spellings of a break are
/// collapsed to `\r?\n`, which is what lets a model that writes `\n` match a CRLF file.
fn match_any_line_ending(old: &str) -> String {
    let escaped = regex::escape(old);
    escaped.replace("\r\n", "\\r?\\n").replace('\n', "\\r?\\n")
}

/// The 1-based line number of a byte offset.
fn line_of(text: &str, offset: usize) -> usize {
    text.get(..offset)
        .map_or(1, |prefix| prefix.matches('\n').count() + 1)
}

fn join_numbers(numbers: &[usize]) -> String {
    let mut out = String::new();
    for (index, number) in numbers.iter().enumerate() {
        if index > 0 {
            if out.len() > 40 {
                let _ = write!(out, ", … ({} total)", numbers.len());
                return out;
            }
            out.push_str(", ");
        }
        let _ = write!(out, "{number}");
    }
    out
}

/// `old` was not found. The observation shows the closest lines so the model can correct itself
/// without a second read.
fn not_found(raw: &str, old: &str, text: &str, crlf: bool) -> String {
    let wanted = old.lines().next().unwrap_or(old);
    let hint = text.lines().position(|line| similar(line, wanted)).map_or_else(
        || {
            "No line contains anything similar, so the file may have changed since you read \
                 it. Read it again before editing."
                .to_string()
        },
        |index| {
            let line = text.lines().nth(index).unwrap_or_default().trim_end_matches('\r');
            format!(
                "Line {} contains something similar: `{}`.",
                index + 1,
                line.trim()
            )
        },
    );

    json!({
        "error": "old_not_found",
        "path": raw,
        "hint": format!(
            "`old` must match the file byte-for-byte apart from line endings{}. {}",
            if crlf { " (this file uses CRLF, which is tolerated)" } else { "" },
            hint
        ),
        "old_preview": first_lines(old, 5),
    })
    .to_string()
}

/// Whether a line looks like the one the model meant, ignoring whitespace.
///
/// Whitespace-only differences are the overwhelmingly common miss — a model writes `fn f(){ }` for
/// `fn f() {}` — so comparing with all spacing removed finds the line the model was looking at
/// while still refusing to suggest an unrelated one.
fn similar(line: &str, wanted: &str) -> bool {
    fn squeeze(text: &str) -> String {
        text.chars().filter(|ch| !ch.is_whitespace()).collect()
    }
    let line = squeeze(line);
    let wanted = squeeze(wanted);
    if wanted.is_empty() || line.is_empty() {
        return false;
    }
    // Containment either way, but a one-character `old` would then "resemble" nearly every line
    // and the hint would point at the wrong place. A confident wrong answer is worse than none.
    let short = line.len().min(wanted.len());
    short >= 4 && (line.contains(&wanted) || wanted.contains(&line))
}

/// `old` matched more than once, so the model's picture of the file is wrong.
fn ambiguous(raw: &str, old: &str, positions: &[usize], text: &str) -> String {
    let lines: Vec<usize> = positions.iter().map(|offset| line_of(text, *offset)).collect();
    json!({
        "error": "old_ambiguous",
        "path": raw,
        "occurrences": positions.len(),
        "lines": lines,
        "hint": "`old` must identify one place. Include more surrounding lines so it is unique, or \
                 pass `replace_all: true` if every occurrence really should change.",
        "old_preview": first_lines(old, 5),
    })
    .to_string()
}

fn first_lines(text: &str, count: usize) -> String {
    let mut out = String::new();
    for line in text.lines().take(count) {
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Write bytes to a sibling temp file, sync it, rename it over the target.
///
/// The permission bits of an existing file are preserved: replacing an executable script with a
/// 0600 file would break `cargo` build scripts and git hooks in ways that look like the agent
/// "randomly" broke the build.
fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), ToolError> {
    let directory = path.parent().filter(|parent| !parent.as_os_str().is_empty());
    let temp = temp_path(path);
    if let Some(directory) = directory {
        std::fs::create_dir_all(directory).map_err(|error| ToolError::io(directory, &error))?;
    }

    let result = (|| -> std::io::Result<()> {
        let mut file = std::fs::File::create(&temp)?;
        std::io::Write::write_all(&mut file, bytes)?;
        file.sync_all()?;
        drop(file);
        if let Ok(metadata) = std::fs::metadata(path) {
            let _ = std::fs::set_permissions(&temp, metadata.permissions());
        }
        std::fs::rename(&temp, path)
    })();

    if let Err(error) = result {
        let _ = std::fs::remove_file(&temp);
        return Err(ToolError::io(path, &error));
    }
    Ok(())
}

/// A temp name in the same directory, so the rename stays on one filesystem.
fn temp_path(path: &Path) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let temp_name = format!(".{name}.immortality-tmp-{}-{sequence}", std::process::id());
    match path.parent() {
        Some(parent) => parent.join(temp_name),
        None => PathBuf::from(temp_name),
    }
}

fn missing_file(raw: &str) -> String {
    json!({
        "error": "file_missing",
        "path": raw,
        "hint": "`edit_file` only changes an existing file. Use `write_file` to create it."
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::PolicyGate;
    use std::os::unix::fs::PermissionsExt as _;
    use std::sync::Arc;

    fn ctx(dir: &Path) -> ToolCtx {
        let gate = PolicyGate::with_defaults(
            dir,
            immortality_core::config::Policy {
                workspace_root: dir.parent().unwrap().to_path_buf(),
                ..immortality_core::config::Policy::default()
            },
        )
        .unwrap();
        ToolCtx::new(Arc::new(gate), crate::OutputBudget::default())
    }

    fn fixture(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, content) in files {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, content).unwrap();
        }
        dir
    }

    fn write(dir: &Path, args: Value) -> Result<ToolResult, ToolError> {
        WriteFile.execute(&ctx(dir), Args::new(args))
    }

    fn edit(dir: &Path, args: Value) -> Result<ToolResult, ToolError> {
        EditFile.execute(&ctx(dir), Args::new(args))
    }

    fn read(dir: &Path, name: &str) -> String {
        std::fs::read_to_string(dir.join(name)).unwrap()
    }

    // ── write_file ──────────────────────────────────────────────────────────────

    #[test]
    fn a_new_file_is_created_with_parent_directories() {
        let dir = fixture(&[]);
        let result = write(
            dir.path(),
            json!({"path": "src/deep/a.rs", "content": "fn a() {}\n"}),
        )
        .unwrap();
        assert_eq!(read(dir.path(), "src/deep/a.rs"), "fn a() {}\n");
        assert_eq!(result.data["created"], true);
        assert_eq!(result.data["lines"], 1);
    }

    #[test]
    fn an_existing_file_is_replaced_whole() {
        let dir = fixture(&[("a.rs", "old\n")]);
        write(dir.path(), json!({"path": "a.rs", "content": "new\n"})).unwrap();
        assert_eq!(read(dir.path(), "a.rs"), "new\n");
    }

    #[test]
    fn writing_identical_content_reports_no_change() {
        // This is what makes `ByContent` honest: replaying the call must be recognisably a no-op.
        let dir = fixture(&[("a.rs", "same\n")]);
        let result = write(dir.path(), json!({"path": "a.rs", "content": "same\n"})).unwrap();
        assert_eq!(result.data["unchanged"], true);
        assert_eq!(result.data["created"], false);
        assert!(
            result.observation.contains("nothing changed"),
            "{}",
            result.observation
        );
    }

    #[test]
    fn writing_outside_the_worktree_is_refused_by_the_gate() {
        let dir = fixture(&[]);
        let error = write(dir.path(), json!({"path": "../escape.rs", "content": "x"})).unwrap_err();
        assert!(error.to_string().contains("denied_by_policy"), "{error}");
    }

    #[test]
    fn writing_a_protected_path_is_refused() {
        let dir = fixture(&[]);
        let error = write(
            dir.path(),
            json!({"path": ".git/hooks/pre-commit", "content": "x"}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("denied_by_policy"), "{error}");
    }

    #[test]
    fn an_oversized_content_is_refused_by_name() {
        let dir = fixture(&[]);
        let big = "x".repeat(MAX_WRITE_BYTES + 1);
        let error = write(dir.path(), json!({"path": "big.txt", "content": big})).unwrap_err();
        assert!(error.to_string().contains("limit"), "{error}");
    }

    #[test]
    fn empty_content_creates_an_empty_file() {
        let dir = fixture(&[]);
        write(dir.path(), json!({"path": "empty.txt", "content": ""})).unwrap();
        assert_eq!(read(dir.path(), "empty.txt"), "");
    }

    #[test]
    fn an_existing_files_permissions_are_preserved() {
        // Replacing a 0755 script with a 0600 file breaks things in a way that looks like the agent
        // randomly broke the build.
        let dir = fixture(&[("run.sh", "#!/bin/sh\n")]);
        let path = dir.path().join("run.sh");
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&path, permissions).unwrap();

        write(
            dir.path(),
            json!({"path": "run.sh", "content": "#!/bin/sh\necho hi\n"}),
        )
        .unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o755, "mode was {mode:o}");
    }

    #[test]
    fn no_temp_file_is_left_behind() {
        let dir = fixture(&[("a.rs", "old\n")]);
        write(dir.path(), json!({"path": "a.rs", "content": "new\n"})).unwrap();
        let leftovers: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains("immortality-tmp"))
            .collect();
        assert_eq!(leftovers, Vec::<String>::new());
    }

    #[test]
    fn the_tool_declares_its_path_as_a_write_and_mutate_effect() {
        assert_eq!(
            WriteFile.paths(&Args::new(json!({"path": "a.rs", "content": "x"}))),
            vec![PathUse::write("a.rs")]
        );
        assert_eq!(WriteFile.effect(&Args::empty()), Effect::Mutate);
        assert_eq!(WriteFile.idempotency(&Args::empty()), Idempotency::ByContent);
    }

    #[test]
    fn the_write_schema_is_well_formed() {
        assert_eq!(
            immortality_llm::schema::schema_defects(WriteFile.schema()),
            Vec::new()
        );
    }

    // ── edit_file ───────────────────────────────────────────────────────────────

    #[test]
    fn an_exact_replacement_changes_only_that_text() {
        let dir = fixture(&[("a.rs", "fn a() {}\nfn b() {}\n")]);
        let result = edit(
            dir.path(),
            json!({"path": "a.rs", "old": "fn b() {}", "new": "fn b() { todo!() }"}),
        )
        .unwrap();
        assert_eq!(read(dir.path(), "a.rs"), "fn a() {}\nfn b() { todo!() }\n");
        assert_eq!(result.data["replacements"], 1);
        assert_eq!(result.data["lines"], json!([2]));
    }

    #[test]
    fn a_missing_old_is_a_structured_refusal_not_a_silent_no_op() {
        let dir = fixture(&[("a.rs", "fn a() {}\n")]);
        let error = edit(
            dir.path(),
            json!({"path": "a.rs", "old": "fn missing()", "new": "x"}),
        )
        .unwrap_err();
        let text = error.to_string();
        assert!(text.contains("old_not_found"), "{text}");
        assert!(text.contains("Read it again"), "{text}");
        // Nothing was written.
        assert_eq!(read(dir.path(), "a.rs"), "fn a() {}\n");
    }

    #[test]
    fn a_similar_line_is_offered_when_old_is_not_found() {
        let dir = fixture(&[("a.rs", "fn alpha() {}\n")]);
        let error = edit(
            dir.path(),
            json!({"path": "a.rs", "old": "fn alpha(){ }", "new": "x"}),
        )
        .unwrap_err();
        let text = error.to_string();
        assert!(text.contains("Line 1"), "{text}");
        assert!(text.contains("fn alpha"), "{text}");
    }

    #[test]
    fn an_ambiguous_old_is_refused_with_the_line_numbers() {
        // The whole point of M3.3: "first occurrence" is how a model patches the wrong branch.
        let dir = fixture(&[("a.rs", "if x {\n}\nif x {\n}\n")]);
        let error = edit(
            dir.path(),
            json!({"path": "a.rs", "old": "if x {", "new": "if y {"}),
        )
        .unwrap_err();
        let text = error.to_string();
        assert!(text.contains("old_ambiguous"), "{text}");
        assert!(text.contains("occurrences"), "{text}");
        assert!(text.contains('2'), "{text}");
        assert_eq!(read(dir.path(), "a.rs"), "if x {\n}\nif x {\n}\n");
    }

    #[test]
    fn replace_all_changes_every_occurrence() {
        let dir = fixture(&[("a.rs", "if x {\n}\nif x {\n}\n")]);
        let result = edit(
            dir.path(),
            json!({"path": "a.rs", "old": "if x {", "new": "if y {", "replace_all": true}),
        )
        .unwrap();
        assert_eq!(read(dir.path(), "a.rs"), "if y {\n}\nif y {\n}\n");
        assert_eq!(result.data["replacements"], 2);
    }

    #[test]
    fn multiline_old_is_matched_exactly() {
        let dir = fixture(&[("a.rs", "fn a() {\n    one();\n    two();\n}\n")]);
        edit(
            dir.path(),
            json!({
                "path": "a.rs",
                "old": "    one();\n    two();",
                "new": "    three();"
            }),
        )
        .unwrap();
        assert_eq!(read(dir.path(), "a.rs"), "fn a() {\n    three();\n}\n");
    }

    #[test]
    fn old_is_a_substring_so_indentation_is_not_required() {
        // `old` is a substring, not a whole line — that is what makes a partial-line edit possible.
        // Safety comes from the ambiguity check, not from demanding the whole line: this `old`
        // occurs once, so it is unambiguous and exact in the only sense that matters.
        let dir = fixture(&[("a.rs", "    let x = 1;\n")]);
        let result = edit(
            dir.path(),
            json!({"path": "a.rs", "old": "let x = 1;", "new": "let x = 2;"}),
        )
        .unwrap();
        assert_eq!(read(dir.path(), "a.rs"), "    let x = 2;\n");
        assert_eq!(result.data["replacements"], 1);
    }

    #[test]
    fn whitespace_that_differs_is_still_not_found() {
        // The mirror of the test above: substring matching does not mean whitespace-insensitive
        // matching, and a model that writes the wrong internal spacing must be told.
        let dir = fixture(&[("a.rs", "let x=1;\n")]);
        let error = edit(
            dir.path(),
            json!({"path": "a.rs", "old": "let x  =  1;", "new": "let x = 2;"}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("old_not_found"), "{error}");
    }

    #[test]
    fn a_lf_old_matches_a_crlf_file() {
        // The bug this prevents: a model writes `\n`, the file has `\r\n`, `contains` says "not
        // found" for text the model can plainly see, and it retries until the loop detector fires.
        let dir = fixture(&[("win.txt", "one\r\ntwo\r\nthree\r\n")]);
        let result = edit(
            dir.path(),
            json!({"path": "win.txt", "old": "two\nthree", "new": "TWO\nTHREE"}),
        )
        .unwrap();
        assert_eq!(read(dir.path(), "win.txt"), "one\r\nTWO\r\nTHREE\r\n");
        assert_eq!(result.data["line_endings"], "crlf");
    }

    #[test]
    fn a_crlf_old_also_matches_a_crlf_file() {
        let dir = fixture(&[("win.txt", "a\r\nb\r\n")]);
        edit(
            dir.path(),
            json!({"path": "win.txt", "old": "a\r\nb", "new": "c"}),
        )
        .unwrap();
        assert_eq!(read(dir.path(), "win.txt"), "c\r\n");
    }

    #[test]
    fn replace_all_on_a_crlf_file_keeps_every_ending() {
        // The combination the two features have to survive together: a bare `\n` in `old` and in
        // `new`, `\r\n` on disk, every occurrence replaced, and no mixed endings afterwards.
        let dir = fixture(&[("win.bat", "set A=1\r\nset B=1\r\nset C=1\r\n")]);
        let result = edit(
            dir.path(),
            json!({"path": "win.bat", "old": "=1", "new": "=2", "replace_all": true}),
        )
        .unwrap();
        assert_eq!(read(dir.path(), "win.bat"), "set A=2\r\nset B=2\r\nset C=2\r\n");
        assert_eq!(result.data["replacements"], 3);
    }

    #[test]
    fn a_multiline_replacement_uses_the_files_own_endings() {
        let dir = fixture(&[("win.txt", "head\r\ntail\r\n")]);
        edit(
            dir.path(),
            json!({"path": "win.txt", "old": "head", "new": "one\ntwo"}),
        )
        .unwrap();
        assert_eq!(read(dir.path(), "win.txt"), "one\r\ntwo\r\ntail\r\n");
    }

    #[test]
    fn a_written_file_reads_back_identically_through_read_file() {
        // The two tools have to agree about the file, or the model reads something it did not
        // write and the next `edit_file` fails against text it believes is there.
        let dir = fixture(&[]);
        let content = "fn main() {\n    println!(\"hi\");\n}\n";
        write(dir.path(), json!({"path": "src/main.rs", "content": content})).unwrap();
        let result = crate::ReadFile
            .execute(&ctx(dir.path()), Args::new(json!({"path": "src/main.rs"})))
            .unwrap();
        assert_eq!(result.data["total_lines"], 3);
        assert!(result.observation.contains("3: }"), "{}", result.observation);
    }

    #[test]
    fn an_lf_file_is_not_turned_into_a_crlf_file() {
        let dir = fixture(&[("a.rs", "one\ntwo\n")]);
        edit(dir.path(), json!({"path": "a.rs", "old": "one", "new": "1"})).unwrap();
        assert_eq!(read(dir.path(), "a.rs"), "1\ntwo\n");
    }

    #[test]
    fn a_replacement_containing_regex_syntax_is_literal() {
        // `$1` and `\1` mean something in a regex replacement; here they are text.
        let dir = fixture(&[("a.rs", "value = 1\n")]);
        edit(
            dir.path(),
            json!({"path": "a.rs", "old": "value = 1", "new": "value = \"$1\\1\""}),
        )
        .unwrap();
        assert_eq!(read(dir.path(), "a.rs"), "value = \"$1\\1\"\n");
    }

    #[test]
    fn regex_metacharacters_in_old_are_literal() {
        let dir = fixture(&[("a.rs", "let v: Vec<i32> = vec![];\n")]);
        edit(
            dir.path(),
            json!({"path": "a.rs", "old": "Vec<i32> = vec![]", "new": "Vec<u8> = Vec::new()"}),
        )
        .unwrap();
        assert_eq!(read(dir.path(), "a.rs"), "let v: Vec<u8> = Vec::new();\n");
    }

    #[test]
    fn an_empty_old_is_refused() {
        let dir = fixture(&[("a.rs", "x\n")]);
        let error = edit(dir.path(), json!({"path": "a.rs", "old": "", "new": "y"})).unwrap_err();
        assert!(error.to_string().contains("empty"), "{error}");
    }

    #[test]
    fn identical_old_and_new_are_refused() {
        // A step that changes nothing while looking like progress is how an agent spins.
        let dir = fixture(&[("a.rs", "x\n")]);
        let error = edit(dir.path(), json!({"path": "a.rs", "old": "x", "new": "x"})).unwrap_err();
        assert!(error.to_string().contains("nothing would change"), "{error}");
    }

    #[test]
    fn editing_a_missing_file_says_to_use_write_file() {
        let dir = fixture(&[]);
        let error = edit(dir.path(), json!({"path": "nope.rs", "old": "a", "new": "b"})).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("file_missing"), "{text}");
        assert!(text.contains("write_file"), "{text}");
    }

    #[test]
    fn editing_a_binary_file_is_refused() {
        let dir = fixture(&[]);
        std::fs::write(dir.path().join("img.png"), b"\x89PNG\r\n\x1a\n\x00\x00").unwrap();
        let error = edit(dir.path(), json!({"path": "img.png", "old": "PNG", "new": "X"})).unwrap_err();
        assert!(error.to_string().contains("binary_file"), "{error}");
    }

    #[test]
    fn editing_invalid_utf8_is_refused_rather_than_lossily_requoted() {
        let dir = fixture(&[]);
        std::fs::write(dir.path().join("bad.txt"), b"ok\n\xff\xfe\n").unwrap();
        let error = edit(dir.path(), json!({"path": "bad.txt", "old": "ok", "new": "fine"})).unwrap_err();
        assert!(error.to_string().contains("not_utf8"), "{error}");
    }

    #[test]
    fn editing_outside_the_worktree_is_refused() {
        let dir = fixture(&[("a.rs", "x\n")]);
        let error = edit(dir.path(), json!({"path": "../a.rs", "old": "x", "new": "y"})).unwrap_err();
        assert!(error.to_string().contains("denied_by_policy"), "{error}");
    }

    #[test]
    fn the_edit_schema_is_well_formed() {
        assert_eq!(
            immortality_llm::schema::schema_defects(EditFile.schema()),
            Vec::new()
        );
    }

    #[test]
    fn the_tool_declares_its_path_as_a_write() {
        assert_eq!(
            EditFile.paths(&Args::new(json!({"path": "a.rs", "old": "x", "new": "y"}))),
            vec![PathUse::write("a.rs")]
        );
        assert_eq!(EditFile.idempotency(&Args::empty()), Idempotency::ByContent);
    }

    #[test]
    fn a_long_list_of_lines_is_summarised_rather_than_flooding() {
        // `replace_all` over a large file must not put ten thousand numbers in the observation.
        let numbers: Vec<usize> = (1..=2000).collect();
        let joined = join_numbers(&numbers);
        assert!(joined.contains("2000 total"), "{joined}");
        assert!(joined.len() < 120, "{}", joined.len());
    }

    #[test]
    fn line_numbers_are_one_based() {
        assert_eq!(line_of("a\nb\nc", 0), 1);
        assert_eq!(line_of("a\nb\nc", 2), 2);
        assert_eq!(line_of("a\nb\nc", 4), 3);
    }
}
