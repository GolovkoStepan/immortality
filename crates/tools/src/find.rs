//! `glob` — find files by name.
//!
//! # Why sorting by modification time
//!
//! The file an agent wants is almost always the one it just wrote or the one that changed under
//! it. Name order buries that under `a.rs`, `b.rs`, `c.rs`. Ties break on the path so that two
//! files written in the same second still come out in a deterministic order — a golden test
//! depends on it.
//!
//! # Why the walk has limits and the result says so
//!
//! A pattern of `**/*` over a repository with `node_modules` is millions of stat calls. The walk
//! stops at a fixed number of *visited* entries, not just at matched ones, because the expensive
//! case is the search that finds nothing; and the observation says whether it stopped, so the
//! model narrows the pattern instead of concluding the file does not exist.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::OnceLock;

use immortality_core::protocol::Idempotency;
use serde_json::{Value, json};

use crate::Effect;
use crate::args::Args;
use crate::glob::{Glob, GlobError};
use crate::policy::PathUse;
use crate::tool::{Tool, ToolCtx, ToolError, ToolResult};

/// Matches returned when the model does not ask.
pub const DEFAULT_GLOB_LIMIT: u64 = 200;

/// Hard cap on returned matches.
///
/// A model that asks for ten thousand paths is trying to read the repository index. Two thousand
/// names is already more context than the next step needs.
pub const MAX_GLOB_LIMIT: u64 = 2_000;

/// Entries the walk will visit before giving up.
///
/// Deliberately larger than the match cap: a narrow pattern in a big tree has to visit many
/// entries to find few matches, and stopping early there would be wrong. 50 000 `stat` calls is
/// roughly a second on APFS.
pub const MAX_VISITED: usize = 50_000;

/// Directories never worth walking.
///
/// Not a correctness rule but a cost one: `target/` and `node_modules/` are almost always the
/// reason a glob is slow, and the agent that wants them can name them explicitly.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    ".venv",
    "__pycache__",
    ".mypy_cache",
];

fn glob_schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, relative to `path`. Supports `*`, `**`, `?`, `[a-z]`."
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in. Defaults to the worktree root."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_GLOB_LIMIT,
                    "description": "Maximum number of paths to return. Defaults to 200."
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Whether to include dotfiles. Defaults to false."
                }
            },
            "required": ["pattern"],
            "additionalProperties": false
        })
    })
}

/// Find files by pattern.
#[derive(Debug, Default, Clone, Copy)]
pub struct GlobTool;

impl Tool for GlobTool {
    fn name(&self) -> &'static str {
        "glob"
    }

    fn schema(&self) -> &'static Value {
        glob_schema()
    }

    fn description(&self) -> &'static str {
        "List files matching a glob pattern, newest first. Dotfiles are excluded unless asked."
    }

    fn effect(&self, _args: &Args) -> Effect {
        Effect::Read
    }

    fn idempotency(&self, _args: &Args) -> Idempotency {
        Idempotency::Idempotent
    }

    fn paths(&self, args: &Args) -> Vec<PathUse> {
        // The base directory is entered, not written; the pattern is not a path.
        args.opt_str("path")
            .ok()
            .flatten()
            .map(|path| vec![PathUse::read(path)])
            .unwrap_or_default()
    }

    /// The pattern and the base, because `path` is optional and a bare `glob` would name neither.
    fn expected_effect(&self, args: &Args) -> String {
        let pattern = args.str("pattern").unwrap_or("?");
        let base = args.opt_str("path").ok().flatten().unwrap_or(".");
        crate::tool::one_line(
            &format!("list files matching `{pattern}` under {base}"),
            crate::tool::EFFECT_LIMIT,
        )
    }

    fn execute(&self, ctx: &ToolCtx, args: Args) -> Result<ToolResult, ToolError> {
        let pattern = args.str("pattern")?;
        let base = args.opt_str("path")?.unwrap_or(".");
        let limit = args
            .opt_u64("limit")?
            .unwrap_or(DEFAULT_GLOB_LIMIT)
            .clamp(1, MAX_GLOB_LIMIT);
        let include_hidden = args.opt_bool("include_hidden")?.unwrap_or(false);

        let root = ctx.enter_path(base)?;
        // A model that writes an absolute pattern is confused about the base, and silently
        // relativising it would search somewhere it did not ask for.
        if pattern.starts_with('/') {
            return Err(ToolError::Arguments(format!(
                "`pattern` must be relative to `{base}`, not absolute: {pattern}"
            )));
        }
        let matcher = Glob::parse(pattern).map_err(|error| glob_error(&error))?;

        let limit = usize::try_from(limit).unwrap_or(usize::MAX);
        let walk = Walk {
            matcher: &matcher,
            include_hidden,
            budget: MAX_VISITED,
        };
        let outcome = walk.run(&root, base);

        if outcome.matches.is_empty() {
            let mut observation = format!("No files match `{pattern}` under `{base}`.");
            if outcome.stopped_early {
                observation.push_str(
                    " The search stopped early to stay inside its budget, so the file may still \
                     exist: narrow the pattern or name a directory in `path`.",
                );
            }
            return Ok(ToolResult::new(
                observation,
                json!({
                    "pattern": pattern,
                    "base": base,
                    "matches": [],
                    "visited": outcome.visited,
                    "stopped_early": outcome.stopped_early,
                }),
            ));
        }

        let total = outcome.matches.len();
        let shown: Vec<&Found> = outcome.matches.iter().take(limit).collect();
        let mut observation = String::new();
        for found in &shown {
            let _ = writeln!(observation, "{}", found.relative);
        }
        if total > shown.len() {
            let _ = writeln!(
                observation,
                "… {} more match(es) not shown; raise `limit` or narrow `pattern`.",
                total - shown.len()
            );
        }

        Ok(ToolResult::new(
            observation,
            json!({
                "pattern": pattern,
                "base": base,
                "matches": shown.iter().map(|found| found.relative.clone()).collect::<Vec<_>>(),
                "total": total,
                "visited": outcome.visited,
                "stopped_early": outcome.stopped_early,
            }),
        ))
    }
}

#[derive(Debug)]
struct Found {
    /// Path relative to the worktree root, with forward slashes.
    relative: String,
    modified: Option<std::time::SystemTime>,
}

#[derive(Debug)]
struct Outcome {
    matches: Vec<Found>,
    visited: usize,
    stopped_early: bool,
}

struct Walk<'a> {
    matcher: &'a Glob,
    include_hidden: bool,
    budget: usize,
}

impl Walk<'_> {
    fn run(&self, root: &Path, base: &str) -> Outcome {
        let mut matches = Vec::new();
        let mut visited = 0usize;
        let base_prefix = normalise_relative(base);
        // Two prefixes per directory, and they are deliberately not the same thing:
        //
        // * the *match* prefix is relative to `base`, because the schema says the pattern is
        //   relative to `path` — `pattern: "*.rs", path: "src"` has to match `a.rs`, not
        //   `src/a.rs`, or `*` would have to cross a segment and the common call would match
        //   nothing;
        // * the *output* prefix is relative to the worktree root, because the model has to be
        //   able to feed the answer straight into `read_file`.
        let mut stack = vec![(root.to_path_buf(), String::new(), base_prefix)];

        while let Some((directory, match_prefix, out_prefix)) = stack.pop() {
            // A directory that cannot be read is not a failure of the whole search: a
            // permission wall in one subtree must not hide matches elsewhere.
            let Ok(entries) = std::fs::read_dir(&directory) else {
                continue;
            };

            for entry in entries.flatten() {
                visited += 1;
                if visited > self.budget {
                    matches.sort_by(compare);
                    return Outcome {
                        matches,
                        visited: visited - 1,
                        stopped_early: true,
                    };
                }

                let name = entry.file_name();
                let name = name.to_string_lossy();
                if !self.include_hidden && name.starts_with('.') {
                    continue;
                }

                let path = entry.path();
                let relative = join_relative(&match_prefix, &name);
                let shown = join_relative(&out_prefix, &name);

                if path.is_dir() {
                    if !SKIP_DIRS.contains(&name.as_ref()) {
                        stack.push((path, relative, shown));
                    }
                    continue;
                }

                if self.matcher.matches(&relative) {
                    let modified = entry.metadata().ok().and_then(|meta| meta.modified().ok());
                    matches.push(Found {
                        relative: shown,
                        modified,
                    });
                }
            }
        }

        matches.sort_by(compare);
        Outcome {
            matches,
            visited,
            stopped_early: false,
        }
    }
}

/// Newest first, then by path.
fn compare(left: &Found, right: &Found) -> std::cmp::Ordering {
    match (&right.modified, &left.modified) {
        (Some(a), Some(b)) => a.cmp(b).then_with(|| left.relative.cmp(&right.relative)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => left.relative.cmp(&right.relative),
    }
}

fn join_relative(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}/{name}")
    }
}

/// A base directory as a pattern prefix: no leading `./`, no trailing slash.
fn normalise_relative(base: &str) -> String {
    let trimmed = base.trim_start_matches("./").trim_matches('/');
    if trimmed == "." {
        String::new()
    } else {
        trimmed.to_string()
    }
}

impl From<GlobError> for ToolError {
    fn from(error: GlobError) -> Self {
        glob_error(&error)
    }
}

fn glob_error(error: &GlobError) -> ToolError {
    ToolError::Arguments(format!(
        "{error}. Supported: `*` within a segment, `**` across segments, `?`, `[a-z]`."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OutputBudget;
    use crate::policy::PolicyGate;
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

    fn fixture(files: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for name in files {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, b"x").unwrap();
        }
        dir
    }

    fn glob(dir: &Path, args: Value) -> Result<ToolResult, ToolError> {
        GlobTool.execute(&ctx(dir), Args::new(args))
    }

    #[test]
    fn a_pattern_finds_matching_files() {
        let dir = fixture(&["a.rs", "b.rs", "c.txt"]);
        let result = glob(dir.path(), json!({"pattern": "*.rs"})).unwrap();
        let matches: Vec<&str> = result.data["matches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        assert_eq!(matches.len(), 2, "{matches:?}");
        assert!(matches.contains(&"a.rs"), "{matches:?}");
    }

    #[test]
    fn double_star_searches_subdirectories() {
        let dir = fixture(&["src/a.rs", "src/deep/b.rs", "c.txt"]);
        let result = glob(dir.path(), json!({"pattern": "**/*.rs"})).unwrap();
        assert_eq!(result.data["matches"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn a_base_directory_prefixes_the_results() {
        // Two rules at once, and they pull in opposite directions on purpose: the pattern is
        // relative to `path` (so `*.rs` must match inside `src`), and the output is relative to
        // the worktree root (so the model can feed it to `read_file` unchanged).
        let dir = fixture(&["src/a.rs", "src/b.rs", "other/c.rs"]);
        let result = glob(dir.path(), json!({"pattern": "*.rs", "path": "src"})).unwrap();
        let matches: Vec<&str> = result.data["matches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        assert_eq!(matches.len(), 2, "{matches:?}");
        assert!(matches.contains(&"src/a.rs"), "{matches:?}");
        assert!(matches.contains(&"src/b.rs"), "{matches:?}");
        assert!(
            !matches.contains(&"other/c.rs"),
            "the base must confine the search: {matches:?}"
        );
    }

    #[test]
    fn a_nested_pattern_is_relative_to_the_base_too() {
        let dir = fixture(&["src/deep/a.rs", "src/deep/b.rs", "other/deep/c.rs"]);
        let result = glob(dir.path(), json!({"pattern": "deep/*.rs", "path": "src"})).unwrap();
        let matches: Vec<&str> = result.data["matches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        assert_eq!(matches.len(), 2, "{matches:?}");
        assert!(matches.contains(&"src/deep/a.rs"), "{matches:?}");
    }

    #[test]
    fn results_are_newest_first() {
        let dir = fixture(&["old.rs", "new.rs"]);
        // Make the order unambiguous: filesystem timestamps have coarse granularity.
        let old = dir.path().join("old.rs");
        let new = dir.path().join("new.rs");
        let past = std::time::SystemTime::now() - std::time::Duration::from_secs(100);
        std::fs::File::options()
            .write(true)
            .open(&old)
            .unwrap()
            .set_modified(past)
            .unwrap();
        let result = glob(dir.path(), json!({"pattern": "*.rs"})).unwrap();
        assert_eq!(result.data["matches"][0], "new.rs");
        assert_eq!(result.data["matches"][1], "old.rs");
        assert!(new.exists());
    }

    #[test]
    fn dotfiles_are_excluded_unless_asked() {
        let dir = fixture(&[".hidden.rs", "visible.rs"]);
        let result = glob(dir.path(), json!({"pattern": "*.rs"})).unwrap();
        assert_eq!(result.data["matches"].as_array().unwrap().len(), 1);
        let result = glob(dir.path(), json!({"pattern": "*.rs", "include_hidden": true})).unwrap();
        assert_eq!(result.data["matches"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn heavy_directories_are_skipped() {
        let dir = fixture(&["target/debug/a.rs", "src/b.rs"]);
        let result = glob(dir.path(), json!({"pattern": "**/*.rs"})).unwrap();
        let matches = result.data["matches"].as_array().unwrap();
        assert_eq!(matches.len(), 1, "{matches:?}");
        assert_eq!(matches[0], "src/b.rs");
    }

    #[test]
    fn the_limit_caps_results_and_says_how_many_were_hidden() {
        let dir = fixture(&["a.rs", "b.rs", "c.rs", "d.rs"]);
        let result = glob(dir.path(), json!({"pattern": "*.rs", "limit": 2})).unwrap();
        assert_eq!(result.data["matches"].as_array().unwrap().len(), 2);
        assert_eq!(result.data["total"], 4);
        assert!(
            result.observation.contains("2 more match(es)"),
            "{}",
            result.observation
        );
    }

    #[test]
    fn a_pattern_that_matches_nothing_says_so_plainly() {
        let dir = fixture(&["a.txt"]);
        let result = glob(dir.path(), json!({"pattern": "*.rs"})).unwrap();
        assert!(
            result.observation.contains("No files match"),
            "{}",
            result.observation
        );
        assert_eq!(result.data["matches"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn an_absolute_pattern_is_refused_rather_than_relativised() {
        let dir = fixture(&["a.rs"]);
        let error = glob(dir.path(), json!({"pattern": "/etc/passwd"})).unwrap_err();
        assert!(error.to_string().contains("not absolute"), "{error}");
    }

    #[test]
    fn an_unsupported_pattern_names_the_supported_syntax() {
        let dir = fixture(&["a.rs"]);
        let error = glob(dir.path(), json!({"pattern": "**/{a,b}.rs"})).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("brace"), "{text}");
        assert!(text.contains("Supported"), "{text}");
    }

    #[test]
    fn a_base_outside_the_worktree_is_refused_by_the_gate() {
        let dir = fixture(&["a.rs"]);
        let error = glob(dir.path(), json!({"pattern": "*", "path": "../../.."})).unwrap_err();
        assert!(error.to_string().contains("denied_by_policy"), "{error}");
    }

    #[test]
    fn an_empty_directory_returns_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        let result = glob(dir.path(), json!({"pattern": "*"})).unwrap();
        assert_eq!(result.data["matches"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn a_dot_base_searches_the_worktree_root() {
        let dir = fixture(&["a.rs"]);
        let result = glob(dir.path(), json!({"pattern": "*.rs", "path": "."})).unwrap();
        assert_eq!(result.data["matches"][0], "a.rs");
    }

    #[test]
    fn the_schema_is_well_formed() {
        assert_eq!(
            immortality_llm::schema::schema_defects(GlobTool.schema()),
            Vec::new()
        );
    }

    #[test]
    fn the_tool_declares_its_base_as_a_read() {
        assert_eq!(
            GlobTool.paths(&Args::new(json!({"pattern": "*", "path": "src"}))),
            vec![PathUse::read("src")]
        );
    }

    #[test]
    fn the_walk_budget_marks_a_truncated_search() {
        // A budget of one entry cannot finish, and `stopped_early` is the flag the observation
        // uses to stop the model from concluding "the file does not exist".
        let dir = fixture(&["a.rs", "b.rs", "c.rs"]);
        let matcher = Glob::parse("*.rs").unwrap();
        let walk = Walk {
            matcher: &matcher,
            include_hidden: false,
            budget: 1,
        };
        let outcome = walk.run(dir.path(), "");
        assert!(outcome.stopped_early, "{outcome:?}");
        assert!(outcome.visited <= 1);
    }
}
