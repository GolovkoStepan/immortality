//! `grep` — search file contents.
//!
//! # Why not shell out to `grep`
//!
//! A subprocess would be simpler, and it would also need a second arg-quoting layer, a second
//! timeout, and a second way to be wrong about which files it searched. The regex engine is in
//! the process, the walk is the same one `glob` uses, and the output shape is ours: `path:line:
//! text` with the option of context, which is what a model quotes when it edits.
//!
//! # Why context is a separate count and not "the surrounding lines"
//!
//! A grep of a forty-line function with `-A 3` returns ten fragments that do not join up. Saying
//! `-A 3 -B 3` explicitly, and eliding between non-adjacent matches with `--`, is the only way
//! the model can tell a gap from a continuation.
//!
//! # Why a binary file is skipped silently
//!
//! A search over a repository meets images and build artefacts constantly. Reporting each one
//! would bury the real matches; the count of skipped files appears in the structured data, so
//! a human reading the journal still knows what happened.
//!
//! # Why the pattern is not `regex`-flavoured by default
//!
//! Models write `foo(bar)` meaning the literal text. Interpreting it as a regex tries to match
//! `foobar` and silently finds nothing. The tool therefore escapes the pattern unless the model
//! opts into `regex: true` — so the common case matches what the model meant, and the powerful
//! case is a deliberate choice with a clear failure if the pattern is malformed.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::OnceLock;

use immortality_core::protocol::Idempotency;
use regex::Regex;
use serde_json::{Value, json};

use crate::Effect;
use crate::args::Args;
use crate::glob::Glob;
use crate::policy::PathUse;
use crate::tool::{Tool, ToolCtx, ToolError, ToolResult};

/// Files searched when the model does not narrow the walk.
const MAX_FILES: usize = 20_000;

/// Matches returned by default.
pub const DEFAULT_MATCH_LIMIT: u64 = 100;

/// Hard cap on matches.
pub const MAX_MATCH_LIMIT: u64 = 1_000;

/// Largest file read during a search.
///
/// A 200 MB SQL dump is not source code. Skipping it is the difference between a search that
/// takes a second and one that swaps.
const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;

/// Directories never searched.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    ".venv",
    "__pycache__",
    ".mypy_cache",
];

fn grep_schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Text to find. Treated literally unless `regex` is true."
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search. Defaults to the worktree root."
                },
                "glob": {
                    "type": "string",
                    "description": "Only search files matching this glob, e.g. `*.rs`."
                },
                "regex": {
                    "type": "boolean",
                    "description": "Interpret `pattern` as a regular expression. Defaults to false."
                },
                "context_before": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 20,
                    "description": "Lines of context before each match. Defaults to 0."
                },
                "context_after": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 20,
                    "description": "Lines of context after each match. Defaults to 0."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_MATCH_LIMIT,
                    "description": "Maximum number of matches. Defaults to 100."
                }
            },
            "required": ["pattern"],
            "additionalProperties": false
        })
    })
}

/// Search file contents.
#[derive(Debug, Default, Clone, Copy)]
pub struct Grep;

impl Tool for Grep {
    fn name(&self) -> &'static str {
        "grep"
    }

    fn schema(&self) -> &'static Value {
        grep_schema()
    }

    fn description(&self) -> &'static str {
        "Search file contents for text, returning `path:line: text` with optional context."
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
        let pattern = args.str("pattern")?;
        if pattern.is_empty() {
            return Err(ToolError::Arguments(
                "`pattern` is empty; an empty search matches every line".to_string(),
            ));
        }
        let base = args.opt_str("path")?.unwrap_or(".");
        let use_regex = args.opt_bool("regex")?.unwrap_or(false);
        let file_glob = args.opt_str("glob")?;
        let budget = ctx.budget();
        let before = args
            .opt_u64("context_before")?
            .map_or(0, |value| usize::try_from(value).unwrap_or(usize::MAX));
        let after = args
            .opt_u64("context_after")?
            .map_or(0, |value| usize::try_from(value).unwrap_or(usize::MAX));
        let limit = args
            .opt_u64("limit")?
            .unwrap_or(DEFAULT_MATCH_LIMIT)
            .clamp(1, MAX_MATCH_LIMIT);

        if before > budget.max_context || after > budget.max_context {
            return Err(ToolError::Arguments(format!(
                "context is limited to {} line(s) each side",
                budget.max_context
            )));
        }

        let regex = compile(pattern, use_regex)?;
        let file_filter = match file_glob {
            Some(pattern) => Some(Glob::parse(pattern).map_err(|error| {
                ToolError::Arguments(format!(
                    "{error}. In `glob`, use `*.rs` for one directory or `**/*.rs` for all."
                ))
            })?),
            None => None,
        };

        let root = ctx.enter_path(base)?;
        let search = Search {
            regex: &regex,
            file_filter: file_filter.as_ref(),
            before,
            after,
            limit: usize::try_from(limit).unwrap_or(usize::MAX),
            base: normalise_relative(base),
        };
        let outcome = search.run(&root);

        let mut observation = String::new();
        for (index, hit) in outcome.hits.iter().enumerate() {
            if index > 0
                && outcome.hits.get(index - 1).is_some_and(|prev| {
                    prev.file != hit.file || hit.line_number > prev.line_number + hit.leading + 1
                })
            {
                observation.push_str("--\n");
            }
            if !hit.file.is_empty() {
                let _ = writeln!(observation, "{}:", hit.file);
            }
            for line in &hit.lines {
                let _ = writeln!(observation, "{line}");
            }
        }

        let summary = if outcome.hits.is_empty() {
            format!("No matches for `{pattern}` under `{base}`.")
        } else if outcome.truncated {
            format!(
                "… stopped after {} match(es); narrow the pattern or `path` to see the rest.",
                outcome.hits.len()
            )
        } else {
            format!(
                "{} match(es) in {} file(s).",
                outcome.hits.len(),
                outcome.files_with_matches
            )
        };
        if !observation.is_empty() {
            observation.push('\n');
        }
        observation.push_str(&summary);

        Ok(ToolResult::new(
            observation,
            json!({
                "pattern": pattern,
                "base": base,
                "matches": outcome.hits.len(),
                "files_searched": outcome.files_searched,
                "files_with_matches": outcome.files_with_matches,
                "files_skipped_binary": outcome.skipped_binary,
                "files_skipped_large": outcome.skipped_large,
                "truncated": outcome.truncated,
            }),
        ))
    }
}

/// `pattern` as a regex: literals escaped unless the caller asked for the real syntax.
fn compile(pattern: &str, use_regex: bool) -> Result<Regex, ToolError> {
    let source = if use_regex {
        pattern.to_string()
    } else {
        regex::escape(pattern)
    };
    Regex::new(&source).map_err(|error| {
        ToolError::Arguments(format!(
            "`pattern` is not a valid regular expression: {error}. \
             Pass `regex: false` to search for the text literally."
        ))
    })
}

#[derive(Debug)]
struct Hit {
    file: String,
    line_number: usize,
    /// How many context lines precede the match, for the gap calculation between hits.
    leading: usize,
    lines: Vec<String>,
}

#[derive(Debug, Default)]
struct Outcome {
    hits: Vec<Hit>,
    files_searched: usize,
    files_with_matches: usize,
    skipped_binary: usize,
    skipped_large: usize,
    truncated: bool,
}

struct Search<'a> {
    regex: &'a Regex,
    file_filter: Option<&'a Glob>,
    before: usize,
    after: usize,
    limit: usize,
    base: String,
}

impl Search<'_> {
    fn run(&self, root: &Path) -> Outcome {
        let mut outcome = Outcome::default();
        let mut files = Vec::new();
        collect_files(root, &mut files, MAX_FILES);

        for (path, relative) in files {
            if outcome.hits.len() >= self.limit {
                outcome.truncated = true;
                break;
            }
            // The filter sees the path relative to `base`, because the schema says `glob` is
            // relative to `path`; the observation shows the worktree-relative path, because the
            // model feeds that straight into `read_file`. One prefix for matching, one for output.
            if let Some(filter) = self.file_filter
                && !filter.matches(&relative)
            {
                continue;
            }
            let relative = if self.base.is_empty() {
                relative
            } else {
                format!("{}/{relative}", self.base)
            };
            if path.metadata().is_ok_and(|meta| meta.len() > MAX_FILE_BYTES) {
                outcome.skipped_large += 1;
                continue;
            }
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            if crate::read::looks_binary(&bytes) {
                outcome.skipped_binary += 1;
                continue;
            }
            outcome.files_searched += 1;

            let text = String::from_utf8_lossy(&bytes);
            let lines: Vec<&str> = text.lines().collect();
            let mut found_here = false;
            for (index, line) in lines.iter().enumerate() {
                if self.regex.is_match(line) {
                    found_here = true;
                    if outcome.hits.len() >= self.limit {
                        outcome.truncated = true;
                        break;
                    }
                    let start = index.saturating_sub(self.before);
                    let end = (index + self.after + 1).min(lines.len());
                    let mut rendered = Vec::with_capacity(end - start);
                    for (offset, context) in lines.get(start..end).unwrap_or_default().iter().enumerate() {
                        let number = start + offset + 1;
                        rendered.push(format!("{number}: {context}"));
                    }
                    outcome.hits.push(Hit {
                        file: relative.clone(),
                        line_number: index + 1,
                        leading: index - start,
                        lines: rendered,
                    });
                }
            }
            if found_here {
                outcome.files_with_matches += 1;
            }
        }

        outcome
    }
}

/// Every file under `root`, each with its path relative to `root`.
fn collect_files(root: &Path, out: &mut Vec<(std::path::PathBuf, String)>, budget: usize) {
    if root.is_file() {
        let name = root
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        out.push((root.to_path_buf(), name));
        return;
    }
    let mut stack = vec![(root.to_path_buf(), String::new())];
    while let Some((directory, prefix)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            if out.len() >= budget {
                return;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let relative = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}/{name}")
            };
            let path = entry.path();
            if path.is_dir() {
                if !SKIP_DIRS.contains(&name.as_str()) {
                    stack.push((path, relative));
                }
            } else {
                out.push((path, relative));
            }
        }
    }
}

fn normalise_relative(base: &str) -> String {
    let trimmed = base.trim_start_matches("./").trim_matches('/');
    if trimmed == "." {
        String::new()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::PolicyGate;
    use crate::tool::OutputBudget;
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
        ToolCtx::new(Arc::new(gate), OutputBudget::default())
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

    fn grep(dir: &Path, args: Value) -> Result<ToolResult, ToolError> {
        Grep.execute(&ctx(dir), Args::new(args))
    }

    #[test]
    fn a_match_is_reported_with_its_path_and_line() {
        let dir = fixture(&[("src/main.rs", "fn main() {\n    // TODO fix\n}\n")]);
        let result = grep(dir.path(), json!({"pattern": "TODO"})).unwrap();
        assert!(
            result.observation.contains("src/main.rs"),
            "{}",
            result.observation
        );
        assert!(result.observation.contains("2: "), "{}", result.observation);
        assert_eq!(result.data["matches"], 1);
    }

    #[test]
    fn a_pattern_is_literal_by_default() {
        // A model writes `foo(bar)` meaning the text. As a regex that means "foobar" and finds
        // nothing, which reads as "the code is not there".
        let dir = fixture(&[("a.rs", "call foo(bar);\n")]);
        let result = grep(dir.path(), json!({"pattern": "foo(bar)"})).unwrap();
        assert_eq!(result.data["matches"], 1, "{}", result.observation);
    }

    #[test]
    fn regex_mode_is_opted_into_and_works() {
        let dir = fixture(&[("a.rs", "fn alpha() {}\nfn beta() {}\n")]);
        let result = grep(dir.path(), json!({"pattern": "fn \\w+", "regex": true})).unwrap();
        assert_eq!(result.data["matches"], 2);
    }

    #[test]
    fn a_malformed_regex_says_how_to_search_literally() {
        let dir = fixture(&[("a.rs", "x")]);
        let error = grep(dir.path(), json!({"pattern": "([", "regex": true})).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("not a valid regular expression"), "{text}");
        assert!(text.contains("regex: false"), "{text}");
    }

    #[test]
    fn an_empty_pattern_is_refused() {
        let dir = fixture(&[("a.rs", "x")]);
        let error = grep(dir.path(), json!({"pattern": ""})).unwrap_err();
        assert!(error.to_string().contains("empty"), "{error}");
    }

    #[test]
    fn context_lines_are_shown_around_the_match() {
        let dir = fixture(&[("a.rs", "one\ntwo\nTHREE\nfour\nfive\n")]);
        let result = grep(
            dir.path(),
            json!({"pattern": "THREE", "context_before": 1, "context_after": 1}),
        )
        .unwrap();
        assert!(result.observation.contains("2: two"), "{}", result.observation);
        assert!(result.observation.contains("3: THREE"), "{}", result.observation);
        assert!(result.observation.contains("4: four"), "{}", result.observation);
        assert!(!result.observation.contains("one"), "{}", result.observation);
    }

    #[test]
    fn context_at_the_file_edges_does_not_panic() {
        // `OutputBudget::max_context` is the ceiling, so this asks for exactly that rather than
        // more: an out-of-budget request is a structured refusal by design, and mixing the two
        // would leave the edge-clamping itself untested.
        let dir = fixture(&[("a.rs", "FIRST\nmiddle\nLAST\n")]);
        let budget = OutputBudget::default().max_context;
        let first = grep(
            dir.path(),
            json!({"pattern": "FIRST", "context_before": budget, "context_after": budget}),
        )
        .unwrap();
        assert!(first.observation.contains("1: FIRST"), "{}", first.observation);
        assert!(first.observation.contains("3: LAST"), "{}", first.observation);

        let last = grep(
            dir.path(),
            json!({"pattern": "LAST", "context_before": budget, "context_after": budget}),
        )
        .unwrap();
        assert!(last.observation.contains("1: FIRST"), "{}", last.observation);
        assert!(last.observation.contains("3: LAST"), "{}", last.observation);
    }

    #[test]
    fn a_gap_between_distant_matches_is_marked() {
        let content: String = (1..=30)
            .map(|n| {
                if n == 3 || n == 25 {
                    format!("HIT {n}\n")
                } else {
                    format!("line {n}\n")
                }
            })
            .collect();
        let dir = fixture(&[("a.rs", &content)]);
        let result = grep(dir.path(), json!({"pattern": "HIT"})).unwrap();
        assert!(result.observation.contains("--"), "{}", result.observation);
    }

    #[test]
    fn adjacent_matches_are_not_separated_by_a_gap_marker() {
        let dir = fixture(&[("a.rs", "HIT\nHIT\n")]);
        let result = grep(dir.path(), json!({"pattern": "HIT"})).unwrap();
        assert!(!result.observation.contains("--"), "{}", result.observation);
    }

    #[test]
    fn the_file_glob_narrows_the_search() {
        let dir = fixture(&[("a.rs", "needle\n"), ("b.txt", "needle\n")]);
        let result = grep(dir.path(), json!({"pattern": "needle", "glob": "*.rs"})).unwrap();
        assert_eq!(result.data["matches"], 1);
        assert_eq!(result.data["files_searched"], 1);
    }

    #[test]
    fn the_limit_truncates_and_says_so() {
        let content: String = (1..=50).map(|_| "hit\n").collect();
        let dir = fixture(&[("a.rs", &content)]);
        let result = grep(dir.path(), json!({"pattern": "hit", "limit": 5})).unwrap();
        assert_eq!(result.data["matches"], 5);
        assert_eq!(result.data["truncated"], true);
        assert!(
            result.observation.contains("stopped after"),
            "{}",
            result.observation
        );
    }

    #[test]
    fn a_binary_file_is_skipped_and_counted() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("img.png"), b"\x89PNG\r\n\x1a\n\x00\x00").unwrap();
        std::fs::write(dir.path().join("a.rs"), b"needle\n").unwrap();
        let result = grep(dir.path(), json!({"pattern": "needle"})).unwrap();
        assert_eq!(result.data["matches"], 1);
        assert_eq!(result.data["files_skipped_binary"], 1);
    }

    #[test]
    fn no_match_says_so_plainly() {
        let dir = fixture(&[("a.rs", "nothing here\n")]);
        let result = grep(dir.path(), json!({"pattern": "zzz"})).unwrap();
        assert!(
            result.observation.contains("No matches"),
            "{}",
            result.observation
        );
        assert_eq!(result.data["matches"], 0);
    }

    #[test]
    fn a_heavy_directory_is_not_searched() {
        let dir = fixture(&[("target/debug/x.rs", "needle\n"), ("src/a.rs", "needle\n")]);
        let result = grep(dir.path(), json!({"pattern": "needle"})).unwrap();
        assert_eq!(result.data["matches"], 1);
    }

    #[test]
    fn searching_one_file_works() {
        let dir = fixture(&[("a.rs", "needle\n"), ("b.rs", "needle\n")]);
        let result = grep(dir.path(), json!({"pattern": "needle", "path": "a.rs"})).unwrap();
        assert_eq!(result.data["matches"], 1);
    }

    #[test]
    fn a_path_outside_the_worktree_is_refused_by_the_gate() {
        let dir = fixture(&[("a.rs", "x")]);
        let error = grep(dir.path(), json!({"pattern": "x", "path": "../../.."})).unwrap_err();
        assert!(error.to_string().contains("denied_by_policy"), "{error}");
    }

    #[test]
    fn context_is_bounded_by_the_budget() {
        let dir = fixture(&[("a.rs", "x")]);
        let error = grep(
            dir.path(),
            json!({"pattern": "x", "context_before": 20, "context_after": 20}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("context is limited"), "{error}");
    }

    #[test]
    fn cyrillic_patterns_match() {
        let dir = fixture(&[("ru.md", "это тест\n")]);
        let result = grep(dir.path(), json!({"pattern": "тест"})).unwrap();
        assert_eq!(result.data["matches"], 1, "{}", result.observation);
    }

    #[test]
    fn the_schema_is_well_formed() {
        assert_eq!(immortality_llm::schema::schema_defects(Grep.schema()), Vec::new());
    }

    #[test]
    fn the_tool_declares_its_base_as_a_read() {
        assert_eq!(
            Grep.paths(&Args::new(json!({"pattern": "x", "path": "src"}))),
            vec![PathUse::read("src")]
        );
        assert_eq!(Grep.effect(&Args::empty()), Effect::Read);
    }
}
