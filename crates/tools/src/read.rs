//! Reading files, listing them, and searching them.
//!
//! # What "read" means here
//!
//! These tools never write. `read_file` numbers lines so a model can cite `src/main.rs:42` and
//! an `edit_file` can quote exact text; `glob` sorts by modification time because the file the
//! agent just wrote is the one it usually wants; `grep` returns matches with context, because a
//! bare line without surroundings is how a model "fixes" the wrong branch of an `if`.
//!
//! # Why binary files are refused rather than dumped
//!
//! A PNG rendered into a prompt is a few thousand tokens of replacement characters that mean
//! nothing and crowd out the content that does. The refusal names the file and offers a
//! different tool (`grep` for a signature, `ls` for a size), which is more useful than a
//! truncated blob.
//!
//! # Why `read_file` truncates by lines, not bytes
//!
//! A model reading a file wants line numbers to line up with what it will later edit. Cutting
//! mid-line would show a half-written statement as if it were the source, and the next
//! `edit_file` would target text that does not exist. The response always says which line range
//! it returned and whether there is more, so the model can page rather than guess.

use std::fmt::Write as _;
use std::sync::OnceLock;

use immortality_core::protocol::Idempotency;
use serde_json::{Value, json};

use crate::Effect;
use crate::args::Args;
use crate::policy::PathUse;
use crate::tool::{Tool, ToolCtx, ToolError, ToolResult};

/// Lines `read_file` returns when the model does not say.
///
/// Large enough for a whole source file and small enough that ten of them fit a budget. The
/// design's context budget is 32K tokens; 2 000 lines of Rust is roughly 20K tokens, so this is
/// already near the limit for one call.
pub const DEFAULT_READ_LINES: u64 = 2_000;

/// Largest `read_file` range a model may ask for, regardless of its `limit`.
///
/// A model that asks for a million lines is not reading a file, it is trying to fill the window.
/// The cap is the design's hard budget expressed in lines: past it the answer is "narrow the
/// range", which costs one step and teaches the model to page.
pub const MAX_READ_LINES: u64 = 8_000;

/// Longest line kept intact.
///
/// A minified bundle is one line of megabytes. Truncating it keeps the tool usable and the
/// observation readable; the line says how much was cut.
const MAX_LINE_CHARS: usize = 4_000;

/// How many bytes are inspected to decide whether a file is binary.
///
/// The first block only: a UTF-8 file with one binary byte deep inside is rare, and reading a
/// gigabyte to be sure costs more than the mistake.
const SNIFF_BYTES: usize = 8_192;

/// The tool's schema, built once.
fn read_schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to read, relative to the worktree root."
                },
                "offset": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "1-based line to start at. Defaults to 1."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_READ_LINES,
                    "description": "How many lines to return. Defaults to 2000."
                }
            },
            "required": ["path"],
            "additionalProperties": false
        })
    })
}

/// Read a text file with line numbers.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReadFile;

impl Tool for ReadFile {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn schema(&self) -> &'static Value {
        read_schema()
    }

    fn description(&self) -> &'static str {
        "Read a text file with 1-based line numbers. Use `offset` and `limit` to page."
    }

    fn effect(&self, _args: &Args) -> Effect {
        Effect::Read
    }

    fn idempotency(&self, _args: &Args) -> Idempotency {
        Idempotency::Idempotent
    }

    fn paths(&self, args: &Args) -> Vec<PathUse> {
        args.opt_str("path")
            .ok()
            .flatten()
            .map(|path| vec![PathUse::read(path)])
            .unwrap_or_default()
    }

    fn execute(&self, ctx: &ToolCtx, args: Args) -> Result<ToolResult, ToolError> {
        let raw = args.str("path")?;
        let offset = args.opt_u64("offset")?.unwrap_or(1).max(1);
        let limit = args
            .opt_u64("limit")?
            .unwrap_or(DEFAULT_READ_LINES)
            .clamp(1, MAX_READ_LINES);

        let path = ctx.read_path(raw)?;
        let bytes = std::fs::read(&path).map_err(|error| ToolError::io(&path, &error))?;

        if let Some(kind) = binary_kind(&bytes) {
            return Err(ToolError::Refused(binary_observation(raw, kind, bytes.len())));
        }

        let text = String::from_utf8_lossy(&bytes);
        let all: Vec<&str> = text.lines().collect();
        let total = all.len();

        // An offset past the end is not an error: the model may have counted from a stale read,
        // and telling it the file has fewer lines is the correction it needs.
        let start_index = usize::try_from(offset.saturating_sub(1)).unwrap_or(usize::MAX);
        if start_index >= total {
            return Ok(ToolResult::new(
                format!("{raw} has {total} line(s); offset {offset} is past the end. Nothing to show."),
                json!({"path": raw, "total_lines": total, "returned": 0, "offset": offset}),
            ));
        }

        let take = usize::try_from(limit).unwrap_or(usize::MAX);
        let end_index = start_index.saturating_add(take).min(total);
        let slice = all.get(start_index..end_index).unwrap_or_default();

        let mut observation = String::new();
        let mut truncated_lines = Vec::new();
        for (index, line) in slice.iter().enumerate() {
            let number = start_index + index + 1;
            let (shown, cut) = clip_line(line);
            if cut {
                truncated_lines.push(number);
            }
            let _ = writeln!(observation, "{number:>6}: {shown}");
        }

        if end_index < total {
            let _ = writeln!(
                observation,
                "… {} more line(s); call again with offset {} to continue.",
                total - end_index,
                end_index + 1
            );
        }
        if !truncated_lines.is_empty() {
            let _ = writeln!(
                observation,
                "… {} line(s) were cut at {MAX_LINE_CHARS} characters: {truncated_lines:?}",
                truncated_lines.len()
            );
        }

        Ok(ToolResult::new(
            observation,
            json!({
                "path": raw,
                "total_lines": total,
                "offset": start_index + 1,
                "returned": end_index - start_index,
                "truncated_lines": truncated_lines,
            }),
        ))
    }
}

/// Clip one line, reporting whether anything was cut.
fn clip_line(line: &str) -> (String, bool) {
    if line.chars().count() <= MAX_LINE_CHARS {
        return (line.to_string(), false);
    }
    let head: String = line.chars().take(MAX_LINE_CHARS).collect();
    (format!("{head}…"), true)
}

/// Whether a byte block is not text.
///
/// Public within the crate because `grep` uses the same sniff: two definitions of "binary" in one
/// crate would drift, and the one that drifted would be the one that dumped a PNG into a prompt.
pub(crate) fn looks_binary(bytes: &[u8]) -> bool {
    binary_kind(bytes).is_some()
}

/// Why a byte block is not text, if it is not.
fn binary_kind(bytes: &[u8]) -> Option<&'static str> {
    let window = bytes.get(..SNIFF_BYTES.min(bytes.len())).unwrap_or(bytes);
    if window.is_empty() {
        return None;
    }
    // PNG, JPEG, GIF, PDF, ZIP, Mach-O and ELF signatures: the formats an agent is most likely to
    // meet in a repository, named so the refusal can say what the file actually is.
    for (signature, kind) in [
        (b"\x89PNG\r\n\x1a\n".as_slice(), "a PNG image"),
        (b"\xff\xd8\xff".as_slice(), "a JPEG image"),
        (b"GIF8".as_slice(), "a GIF image"),
        (b"%PDF".as_slice(), "a PDF document"),
        (b"PK\x03\x04".as_slice(), "a ZIP archive"),
        (b"\xcf\xfa\xed\xfe".as_slice(), "a Mach-O binary"),
        (b"\xca\xfe\xba\xbe".as_slice(), "a Mach-O binary"),
        (b"\x7fELF".as_slice(), "an ELF binary"),
        (b"SQLite format 3".as_slice(), "a SQLite database"),
    ] {
        if window.starts_with(signature) {
            return Some(kind);
        }
    }
    if window.contains(&0) {
        // A NUL byte in the first block is the classic binary signal, and it covers every format
        // the signature list does not name.
        return Some("binary data");
    }
    None
}

fn binary_observation(raw: &str, kind: &str, size: usize) -> String {
    json!({
        "error": "binary_file",
        "path": raw,
        "kind": kind,
        "bytes": size,
        "hint": "This is not a text file, so reading it would fill the context with noise. \
                 Use `glob` to list neighbours, or `grep` for a signature if you know one."
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::PolicyGate;
    use std::path::Path;
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

    fn fixture(files: &[(&str, &[u8])]) -> tempfile::TempDir {
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

    fn read(dir: &Path, args: Value) -> Result<ToolResult, ToolError> {
        ReadFile.execute(&ctx(dir), Args::new(args))
    }

    #[test]
    fn a_file_is_returned_with_line_numbers() {
        let dir = fixture(&[("a.txt", b"alpha\nbeta\ngamma\n")]);
        let result = read(dir.path(), json!({"path": "a.txt"})).unwrap();
        assert!(
            result.observation.contains("     1: alpha"),
            "{}",
            result.observation
        );
        assert!(
            result.observation.contains("     3: gamma"),
            "{}",
            result.observation
        );
        assert_eq!(result.data["total_lines"], 3);
        assert_eq!(result.data["returned"], 3);
    }

    #[test]
    fn an_offset_pages_through_the_file() {
        let dir = fixture(&[("a.txt", b"a\nb\nc\nd\ne\n")]);
        let result = read(dir.path(), json!({"path": "a.txt", "offset": 3, "limit": 2})).unwrap();
        assert!(result.observation.contains("     3: c"), "{}", result.observation);
        assert!(result.observation.contains("     4: d"), "{}", result.observation);
        assert!(!result.observation.contains("     5:"), "{}", result.observation);
    }

    #[test]
    fn truncation_says_how_many_lines_remain_and_where_to_resume() {
        let mut content = String::new();
        for n in 1..=10 {
            let _ = writeln!(content, "line {n}");
        }
        let content = content;
        let dir = fixture(&[("a.txt", content.as_bytes())]);
        let result = read(dir.path(), json!({"path": "a.txt", "limit": 3})).unwrap();
        assert!(
            result.observation.contains("7 more line(s)"),
            "{}",
            result.observation
        );
        assert!(
            result.observation.contains("offset 4"),
            "the model must know where to continue: {}",
            result.observation
        );
    }

    #[test]
    fn an_offset_past_the_end_is_not_an_error() {
        // The model may have counted from a stale read; telling it the length is the correction.
        let dir = fixture(&[("a.txt", b"a\nb\n")]);
        let result = read(dir.path(), json!({"path": "a.txt", "offset": 99})).unwrap();
        assert!(result.observation.contains("2 line(s)"), "{}", result.observation);
        assert_eq!(result.data["returned"], 0);
    }

    #[test]
    fn an_empty_file_reads_as_zero_lines() {
        let dir = fixture(&[("empty.txt", b"")]);
        let result = read(dir.path(), json!({"path": "empty.txt"})).unwrap();
        assert_eq!(result.data["total_lines"], 0);
        assert_eq!(result.data["returned"], 0);
    }

    #[test]
    fn a_missing_file_names_the_path() {
        let dir = fixture(&[]);
        let error = read(dir.path(), json!({"path": "nope.txt"})).unwrap_err();
        assert!(error.to_string().contains("nope.txt"), "{error}");
    }

    #[test]
    fn a_binary_file_is_refused_with_its_kind() {
        let dir = fixture(&[("img.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")]);
        let error = read(dir.path(), json!({"path": "img.png"})).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("binary_file"), "{text}");
        assert!(text.contains("PNG"), "{text}");
        assert!(text.contains("hint"), "{text}");
    }

    #[test]
    fn a_nul_byte_makes_a_file_binary_even_without_a_signature() {
        let dir = fixture(&[("blob.bin", b"text\x00more")]);
        let error = read(dir.path(), json!({"path": "blob.bin"})).unwrap_err();
        assert!(error.to_string().contains("binary data"), "{error}");
    }

    #[test]
    fn utf8_is_read_as_characters_not_bytes() {
        let dir = fixture(&[("ru.txt", "привет\nмир\n".as_bytes())]);
        let result = read(dir.path(), json!({"path": "ru.txt"})).unwrap();
        assert!(result.observation.contains("привет"), "{}", result.observation);
        assert_eq!(result.data["total_lines"], 2);
    }

    #[test]
    fn invalid_utf8_does_not_fail_the_read() {
        // A file with a stray byte is common in mixed repositories; refusing it would make the
        // tool useless for the one file the agent needs.
        let dir = fixture(&[("mixed.txt", b"ok\n\xff\xfe\nend\n")]);
        let result = read(dir.path(), json!({"path": "mixed.txt"})).unwrap();
        assert!(result.observation.contains("ok"));
        assert!(result.observation.contains("end"));
    }

    #[test]
    fn an_enormous_line_is_cut_and_reported() {
        let long = "x".repeat(MAX_LINE_CHARS + 100);
        let dir = fixture(&[("min.js", long.as_bytes())]);
        let result = read(dir.path(), json!({"path": "min.js"})).unwrap();
        assert!(result.observation.contains('…'), "the cut must be visible");
        assert_eq!(result.data["truncated_lines"], json!([1]));
    }

    #[test]
    fn a_limit_above_the_cap_is_clamped_rather_than_refused() {
        // The schema's `maximum` already says no; the tool still has to be safe if a caller
        // bypasses validation.
        let dir = fixture(&[("a.txt", b"a\n")]);
        let result = read(dir.path(), json!({"path": "a.txt", "limit": 1_000_000})).unwrap();
        assert_eq!(result.data["returned"], 1);
    }

    #[test]
    fn crlf_files_report_clean_lines() {
        let dir = fixture(&[("win.txt", b"a\r\nb\r\n")]);
        let result = read(dir.path(), json!({"path": "win.txt"})).unwrap();
        assert!(!result.observation.contains('\r'), "{:?}", result.observation);
    }

    #[test]
    fn a_trailing_newline_does_not_add_a_phantom_line() {
        let dir = fixture(&[("a.txt", b"a\nb\n")]);
        let result = read(dir.path(), json!({"path": "a.txt"})).unwrap();
        assert_eq!(result.data["total_lines"], 2);
    }

    #[test]
    fn reading_outside_the_worktree_is_refused_by_the_gate_not_the_tool() {
        let dir = fixture(&[("a.txt", b"a\n")]);
        let error = read(dir.path(), json!({"path": "../../etc/passwd"})).unwrap_err();
        assert!(error.to_string().contains("denied_by_policy"), "{error}");
    }

    #[test]
    fn a_symlink_followed_by_a_parent_segment_cannot_escape_the_worktree() {
        // The check-then-use bug this pins, and it is the only shape that actually diverges.
        //
        // `resolve` collapses `..` *lexically*, so `link/../secret.txt` is judged as
        // `worktree/secret.txt` — inside. The operating system does not read it that way: it
        // follows `link` first and then walks up from there, landing outside the worktree. If a
        // tool re-joined the raw argument onto the worktree, the gate would approve one path and
        // the read would open another.
        let dir = fixture(&[]);
        let outside = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(outside.path().join("target")).unwrap();
        std::fs::write(outside.path().join("secret.txt"), b"top secret\n").unwrap();
        std::os::unix::fs::symlink(outside.path().join("target"), dir.path().join("link")).unwrap();

        // The workspace root is the worktree itself, so a path outside it is refused outright.
        let gate = PolicyGate::with_defaults(
            dir.path(),
            immortality_core::config::Policy {
                workspace_root: dir.path().to_path_buf(),
                ..immortality_core::config::Policy::default()
            },
        )
        .unwrap();
        let ctx = ToolCtx::new(Arc::new(gate), crate::OutputBudget::default());

        let result = ReadFile.execute(&ctx, Args::new(json!({"path": "link/../secret.txt"})));
        match result {
            Ok(found) => panic!("escaped the worktree and read: {}", found.observation),
            Err(error) => assert!(
                !error.to_string().contains("top secret"),
                "the refusal must not carry the file: {error}"
            ),
        }
    }

    #[test]
    fn the_schema_is_well_formed() {
        assert_eq!(
            immortality_llm::schema::schema_defects(ReadFile.schema()),
            Vec::new()
        );
    }

    #[test]
    fn the_tool_declares_the_path_it_reads() {
        let paths = ReadFile.paths(&Args::new(json!({"path": "a.txt"})));
        assert_eq!(paths, vec![PathUse::read("a.txt")]);
        assert_eq!(ReadFile.effect(&Args::empty()), Effect::Read);
        assert_eq!(ReadFile.idempotency(&Args::empty()), Idempotency::Idempotent);
    }
}
