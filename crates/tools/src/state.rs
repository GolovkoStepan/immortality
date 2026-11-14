//! The agent's own state: `read_state`, `update_state`, `grep_journal`.
//!
//! # Why these are tools and not part of the prompt
//!
//! The design's context layers put `PLAN.md`, `NOTES.md` and a task digest in every step, and
//! separately give the model tools to look at them on demand. The split is deliberate: the
//! layers that must be stable for the prefix cache are assembled by the daemon, and these tools
//! exist for the case where the model needs something *specific* — the acceptance criteria for
//! one subtask, what was decided about a function three days ago — without paying for all of it
//! every step. A model that had to have everything in context would spend its budget on files it
//! is not currently touching.
//!
//! # Why there is a fixed list of state files
//!
//! `.agent/` contains five files a human reviews (`PLAN.md`, `ACCEPTANCE.md`, `TASKS.json`,
//! `DECISIONS.md`, `NOTES.md`) and one it must not touch (`JOURNAL.jsonl`, handled by the
//! journal tool). A tool that took an arbitrary path would be `read_file` with a misleading name,
//! and `update_state` over an arbitrary path would be `write_file` with no policy check — that is
//! the whole reason the file argument is an enum rather than a string.
//!
//! # Why `update_state` is not `write_file`
//!
//! Three differences, each of which has bitten a coding agent somewhere:
//!
//! * It refuses a state file that does not exist instead of creating it. Creating `NOTES.md` by
//!   accident is how a repository acquires an empty file in a pull request; `immortality init`
//!   puts the files there, and a missing one means a wrong worktree, not a new document.
//! * It refuses to shrink a file by more than half without `replace: true`. A model that meant
//!   to append and sent only the new paragraph would otherwise delete the plan.
//! * It writes atomically and syncs, like `write_file`, because these files are read by the very
//!   next step and a torn `NOTES.md` would poison every later prompt.
//!
//! # Why `grep_journal` reads the journal directly
//!
//! It could ask the daemon, but the journal is the one thing that is guaranteed to be on disk and
//! complete — it is what recovery itself reads. A tool that needed a live daemon to answer "what
//! did I try before the crash" would be unavailable in exactly the situation where the answer
//! matters most.

use std::fmt::Write as _;
use std::sync::OnceLock;

use immortality_core::journal::{JournalReader, Kind};
use immortality_core::protocol::Idempotency;
use serde_json::{Value, json};

use crate::Effect;
use crate::args::Args;
use crate::policy::PathUse;
use crate::tool::{Tool, ToolCtx, ToolError, ToolResult};
use crate::write::write_atomically;

/// The in-repository state directory.
pub const AGENT_DIR: &str = ".agent";

/// One of the five files the agent may read and edit.
///
/// The journal is deliberately absent: it is append-only and written by the daemon's journal
/// writer, never by a tool. `grep_journal` reads it; nothing here writes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateFile {
    /// The plan: decomposition and ordering.
    Plan,
    /// Acceptance criteria and the commands that check them.
    Acceptance,
    /// The task tree and subtask statuses.
    Tasks,
    /// Decisions taken, with their reasons.
    Decisions,
    /// Working memory: facts, constraints, rejected hypotheses.
    Notes,
}

impl StateFile {
    /// Every state file, in the order a prompt or a listing shows them.
    pub const ALL: &'static [Self] = &[
        Self::Plan,
        Self::Acceptance,
        Self::Tasks,
        Self::Decisions,
        Self::Notes,
    ];

    /// The name a model writes in `file`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Plan => "PLAN.md",
            Self::Acceptance => "ACCEPTANCE.md",
            Self::Tasks => "TASKS.json",
            Self::Decisions => "DECISIONS.md",
            Self::Notes => "NOTES.md",
        }
    }

    /// The path relative to the worktree root.
    #[must_use]
    pub fn relative_path(self) -> String {
        format!("{AGENT_DIR}/{}", self.as_str())
    }

    /// Whether this document is working memory the agent creates for itself.
    ///
    /// `NOTES.md` and `DECISIONS.md` have no producer besides the agent: the planner writes the
    /// plan, `init` writes the config. The other three are executed by the scheduler, so their
    /// absence means a wrong directory rather than a document waiting to be written.
    #[must_use]
    pub const fn is_working_memory(self) -> bool {
        matches!(self, Self::Decisions | Self::Notes)
    }

    /// Whether `name` is exactly one of the five documents.
    ///
    /// The bare name only: `.agent/NOTES.md` is [`Self::parse`]'s job, and a caller that has a
    /// string from outside the model — a config, a journal record — has no business supplying the
    /// prefix. `is_state_file("")`, `is_state_file("JOURNAL.jsonl")` and
    /// `is_state_file("sub/NOTES.md")` are all false, which is what keeps the set closed.
    #[must_use]
    pub fn is_state_file(name: &str) -> bool {
        Self::ALL.iter().any(|file| file.as_str() == name)
    }

    /// Parse the model's `file` argument.
    ///
    /// Accepts the bare name and the `.agent/`-prefixed form, because a model that just used
    /// `read_file` on `.agent/PLAN.md` will paste that path. Rejecting the second form would
    /// produce a step spent on a spelling lesson.
    ///
    /// # Errors
    /// The name is not one of [`Self::ALL`], or it names the journal.
    pub fn parse(name: &str) -> Result<Self, ToolError> {
        let bare = name
            .strip_prefix(AGENT_DIR)
            .map_or(name, |rest| rest.strip_prefix('/').unwrap_or(rest));
        for file in Self::ALL {
            if file.as_str() == bare {
                return Ok(*file);
            }
        }
        if bare.starts_with("JOURNAL") {
            return Err(ToolError::Arguments(
                "`JOURNAL.jsonl` is the audit trail and is appended by the daemon; it is not a \
                 state file. Use `grep_journal` to read it."
                    .to_string(),
            ));
        }
        Err(ToolError::Arguments(format!(
            "`{name}` is not a state file. Available: {}.",
            Self::ALL
                .iter()
                .map(|file| file.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )))
    }
}

fn read_state_schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "enum": StateFile::ALL.iter().map(|file| file.as_str()).collect::<Vec<_>>(),
                    "description": "Which state document to read."
                },
                "offset": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "1-based line to start at. Defaults to 1."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_STATE_LINES,
                    "description": "How many lines to return. Defaults to 400."
                }
            },
            "required": ["file"],
            "additionalProperties": false
        })
    })
}

fn update_state_schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "enum": StateFile::ALL.iter().map(|file| file.as_str()).collect::<Vec<_>>(),
                    "description": "Which state document to replace."
                },
                "content": {
                    "type": "string",
                    "description": "The complete new contents of the file."
                },
                "replace": {
                    "type": "boolean",
                    "description": "Allow replacing the file with one less than half its current size. Defaults to false."
                }
            },
            "required": ["file", "content"],
            "additionalProperties": false
        })
    })
}

fn grep_journal_schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to find in the journal, matched literally against the event kind and the serialized payload."
                },
                "since_seq": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Start at this `seq` (inclusive). Pass the `next_seq` from a previous call to page forward without repeats."
                },
                "kind": {
                    "type": "string",
                    "description": "Only records of this event kind, e.g. `tool_call_completed`."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_JOURNAL_MATCHES,
                    "description": "Maximum number of records. Defaults to 20."
                }
            },
            "additionalProperties": false
        })
    })
}

/// Lines `read_state` returns when the model does not say.
///
/// Two hundred lines of markdown is a plan or a notes file; the cap exists because `NOTES.md`
/// grows for the life of a task and reading all of it every time is how the budget disappears.
pub const DEFAULT_STATE_LINES: u64 = 200;

/// Hard cap on `read_state` lines.
pub const MAX_STATE_LINES: u64 = 2_000;

/// Records `grep_journal` returns when the model does not say.
pub const DEFAULT_JOURNAL_MATCHES: u64 = 20;

/// Hard cap on `grep_journal` records.
pub const MAX_JOURNAL_MATCHES: u64 = 200;

/// Largest state file `update_state` will rewrite.
pub const MAX_STATE_BYTES: usize = 4 * 1024 * 1024;

/// Read one state document.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReadState;

impl Tool for ReadState {
    fn name(&self) -> &'static str {
        "read_state"
    }

    fn schema(&self) -> &'static Value {
        read_state_schema()
    }

    fn description(&self) -> &'static str {
        "Read a state document: PLAN.md, ACCEPTANCE.md, TASKS.json, DECISIONS.md or NOTES.md."
    }

    fn effect(&self, _args: &Args) -> Effect {
        Effect::Read
    }

    fn idempotency(&self, _args: &Args) -> Idempotency {
        Idempotency::Idempotent
    }

    fn paths(&self, args: &Args) -> Vec<PathUse> {
        args.opt_str("file")
            .ok()
            .flatten()
            .and_then(|name| StateFile::parse(name).ok())
            .map(|file| vec![PathUse::read(file.relative_path())])
            .unwrap_or_default()
    }

    fn execute(&self, ctx: &ToolCtx, args: Args) -> Result<ToolResult, ToolError> {
        let file = StateFile::parse(args.str("file")?)?;
        let offset = args.opt_u64("offset")?.unwrap_or(1).max(1);
        let limit = args
            .opt_u64("limit")?
            .unwrap_or(DEFAULT_STATE_LINES)
            .clamp(1, MAX_STATE_LINES);

        let relative = file.relative_path();
        // The five documents live in the *project's* `.agent/`, not the worktree's stale copy.
        // The gate is asked by name, because a name from a closed list is the whole reason this
        // tool takes an enum rather than a path.
        match ctx.gate().check_state_read(file.as_str()) {
            crate::Verdict::Allowed => {}
            crate::Verdict::Denied(denial) => return Err(ToolError::Refused(denial.observation())),
            crate::Verdict::Escalated(escalation) => {
                return Err(ToolError::Refused(escalation.observation()));
            }
        }
        let path = ctx.gate().state_path(file.as_str());
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                // A missing state file is a real answer, not a failure: the task may not have a
                // plan yet, and turning that into an error would make the model retry.
                return Ok(ToolResult::new(
                    format!("{relative} does not exist. The task may not have written it yet."),
                    json!({"file": file.as_str(), "exists": false}),
                ));
            }
            Err(error) => return Err(ToolError::io(&path, &error)),
        };
        if crate::read::looks_binary(&bytes) {
            return Err(ToolError::Refused(format!(
                "{relative} is not text; it cannot be a state document."
            )));
        }

        let text = String::from_utf8_lossy(&bytes);
        let all: Vec<&str> = text.lines().collect();
        let total = all.len();
        let start = usize::try_from(offset.saturating_sub(1)).unwrap_or(usize::MAX);
        if start >= total {
            return Ok(ToolResult::new(
                format!("{relative} has {total} line(s); offset {offset} is past the end."),
                json!({"file": file.as_str(), "exists": true, "total_lines": total}),
            ));
        }
        let end = total.min(start.saturating_add(usize::try_from(limit).unwrap_or(usize::MAX)));
        let shown = all.get(start..end).unwrap_or_default();

        let mut observation = String::new();
        for (index, line) in shown.iter().enumerate() {
            let _ = writeln!(observation, "{}: {line}", start + index + 1);
        }
        if end < total {
            let _ = writeln!(
                observation,
                "… {relative} has {total} line(s); showing {offset}–{end}. Raise `limit` or use \
                 `offset` for the rest."
            );
        }

        Ok(ToolResult::new(
            observation,
            json!({
                "file": file.as_str(),
                "exists": true,
                "total_lines": total,
                "from": offset,
                "to": end,
            }),
        ))
    }
}

/// Replace one state document.
#[derive(Debug, Default, Clone, Copy)]
pub struct UpdateState;

impl Tool for UpdateState {
    fn name(&self) -> &'static str {
        "update_state"
    }

    fn schema(&self) -> &'static Value {
        update_state_schema()
    }

    fn description(&self) -> &'static str {
        "Replace a state document wholesale. Refuses a missing plan document, and refuses to shrink one below half without `replace: true`."
    }

    fn effect(&self, _args: &Args) -> Effect {
        Effect::Mutate
    }

    fn idempotency(&self, _args: &Args) -> Idempotency {
        // Same rule as `write_file`: the content is the check, so a replay is safe *after*
        // reading the file, which is what `ByContent` means.
        Idempotency::ByContent
    }

    fn paths(&self, args: &Args) -> Vec<PathUse> {
        args.opt_str("file")
            .ok()
            .flatten()
            .and_then(|name| StateFile::parse(name).ok())
            .map(|file| vec![PathUse::write(file.relative_path())])
            .unwrap_or_default()
    }

    fn execute(&self, ctx: &ToolCtx, args: Args) -> Result<ToolResult, ToolError> {
        let file = StateFile::parse(args.str("file")?)?;
        let content = args.str("content")?;
        let replace = args.opt_bool("replace")?.unwrap_or(false);
        if content.len() > MAX_STATE_BYTES {
            return Err(ToolError::Arguments(format!(
                "`content` is {} bytes; the limit is {MAX_STATE_BYTES}. A state document this \
                 large belongs in a regular file.",
                content.len()
            )));
        }

        let relative = file.relative_path();
        // The five documents live in the *project's* `.agent/`, not the worktree's stale copy.
        match ctx.gate().check_state_write(file.as_str()) {
            crate::Verdict::Allowed => {}
            crate::Verdict::Denied(denial) => return Err(ToolError::Refused(denial.observation())),
            crate::Verdict::Escalated(escalation) => {
                return Err(ToolError::Refused(escalation.observation()));
            }
        }
        let path = ctx.gate().state_path(file.as_str());
        let previous = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && file.is_working_memory() => {
                // The two working-memory documents are the ones nobody else writes: the planner
                // produces the plan, `init` produces the config, and no component produces notes or
                // decisions. Refusing to create them meant the model could only ever *think* about
                // writing what it learned — measured on the live run, where `update_state` on
                // `NOTES.md` was refused with "State files are created by `immortality init`",
                // which was untrue: `init` writes `config.toml` and nothing else. The model then
                // wrote `NOTES.md` in the repository root to keep the facts somewhere, and that
                // file went into its commit.
                Vec::new()
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                // The plan documents are different: `PLAN.md`, `ACCEPTANCE.md` and `TASKS.json` are
                // what the scheduler executes. A missing one means a wrong directory, not a new
                // document, and creating it would give the run an empty plan to execute.
                return Err(ToolError::Refused(format!(
                    "{relative} does not exist. The plan documents come from `immortality plan`; \
                     a missing one means this is not the task's state directory."
                )));
            }
            Err(error) => return Err(ToolError::io(&path, &error)),
        };

        let unchanged = previous == content.as_bytes();
        if unchanged {
            return Ok(ToolResult::new(
                format!("{relative} already had exactly that content; nothing changed."),
                json!({
                    "file": file.as_str(),
                    "unchanged": true,
                    "bytes": content.len(),
                }),
            ));
        }

        // A file that lost more than half its bytes is usually a model that meant to append and
        // sent only the new part. Refusing costs a step; accepting deletes the plan.
        let shrunk_badly = content.len().saturating_mul(2) < previous.len();
        if shrunk_badly && !replace {
            return Err(ToolError::Refused(
                json!({
                    "error": "state_shrink_refused",
                    "file": file.as_str(),
                    "current_bytes": previous.len(),
                    "new_bytes": content.len(),
                    "hint": "The new content is less than half the current size. If this is \
                             intended, send the call again with `replace: true`.",
                })
                .to_string(),
            ));
        }

        write_atomically(&path, content.as_bytes())?;

        let lines = content.lines().count();
        Ok(ToolResult::new(
            format!(
                "Wrote {relative} ({lines} line(s), {} bytes), replacing {} bytes.",
                content.len(),
                previous.len()
            ),
            json!({
                "file": file.as_str(),
                "unchanged": false,
                "bytes": content.len(),
                "previous_bytes": previous.len(),
                "lines": lines,
            }),
        ))
    }
}

/// Search the task journal.
#[derive(Debug, Default, Clone, Copy)]
pub struct GrepJournal;

impl Tool for GrepJournal {
    fn name(&self) -> &'static str {
        "grep_journal"
    }

    fn schema(&self) -> &'static Value {
        grep_journal_schema()
    }

    fn description(&self) -> &'static str {
        "Search this task's journal for records, optionally after a `seq` and of one `kind`."
    }

    fn effect(&self, _args: &Args) -> Effect {
        Effect::Read
    }

    fn idempotency(&self, _args: &Args) -> Idempotency {
        Idempotency::Idempotent
    }

    fn paths(&self, _args: &Args) -> Vec<PathUse> {
        // Every segment, because the search spans rotations. Declaring the directory is enough
        // for the gate; naming a file would silently skip the rotated ones.
        vec![PathUse::read(format!("{AGENT_DIR}/"))]
    }

    /// The query and the kind, since the declared path is the whole journal directory and says
    /// nothing about what was being looked for.
    fn expected_effect(&self, args: &Args) -> String {
        let query = args.opt_str("query").ok().flatten().unwrap_or("");
        let kind = args.opt_str("kind").ok().flatten().unwrap_or("any kind");
        if query.is_empty() {
            crate::tool::one_line(
                &format!("search the journal for records of {kind}"),
                crate::tool::EFFECT_LIMIT,
            )
        } else {
            crate::tool::one_line(
                &format!("search the journal for `{query}` in records of {kind}"),
                crate::tool::EFFECT_LIMIT,
            )
        }
    }

    fn execute(&self, ctx: &ToolCtx, args: Args) -> Result<ToolResult, ToolError> {
        let query = args.opt_str("query")?.unwrap_or("").trim().to_string();
        // `None`, not zero. Sequence numbers start at 0, so a zero default would mean "skip the
        // first record of every journal" — the record that says the task was created. Paging
        // passes the previous call's `next_seq`, which is always ≥ 1.
        let since_seq = args.opt_u64("since_seq")?;
        let kind_filter = args.opt_str("kind")?.map(parse_kind).transpose()?;
        let limit = args
            .opt_u64("limit")?
            .unwrap_or(DEFAULT_JOURNAL_MATCHES)
            .clamp(1, MAX_JOURNAL_MATCHES);

        // The journal lives in the project's `.agent/` too, which is the whole reason it survives a
        // rollback: `reset --hard` runs in the worktree, and a journal inside it would be deleted
        // by the recovery step that reads it (D-083).
        let dir = ctx.gate().state_dir();
        if !dir.is_dir() {
            return Ok(ToolResult::new(
                format!("There is no `{AGENT_DIR}/` directory for this project, so no journal."),
                json!({"exists": false, "matches": []}),
            ));
        }

        let reader = JournalReader::open_unnamed(&dir)
            .map_err(|error| ToolError::Refused(format!("the journal could not be opened: {error}")))?;
        let outcome = reader
            .read()
            .map_err(|error| ToolError::Refused(format!("the journal could not be read: {error}")))?;

        // A journal that fails its chain check is reported rather than searched: answers from a
        // file that was edited underneath recovery are worse than no answers.
        if outcome.is_corrupt() {
            let problems = outcome
                .problems
                .iter()
                .filter(|problem| problem.is_fatal())
                .map(|problem| format!("{problem:?}"))
                .collect::<Vec<_>>();
            return Err(ToolError::Refused(
                json!({
                    "error": "journal_corrupt",
                    "problems": problems,
                    "hint": "The journal does not pass its own chain check. Report this; do not \
                             rewrite the file.",
                })
                .to_string(),
            ));
        }

        let reports = search_records(
            &outcome.records,
            &Search {
                query: &query,
                since_seq,
                kind: kind_filter,
            },
        );
        let total = reports.len();
        let shown = &reports[..reports.len().min(usize::try_from(limit).unwrap_or(usize::MAX))];

        let mut observation = String::new();
        for report in shown {
            let _ = writeln!(observation, "#{} {} {}", report.seq, report.at, report.kind);
            let _ = writeln!(
                observation,
                "   {}",
                clip(&report.payload, ctx.budget().max_bytes)
            );
            if let Some(artifact) = &report.artifact {
                let _ = writeln!(observation, "   artifact: {artifact}");
            }
        }
        if total > shown.len() {
            let _ = writeln!(
                observation,
                "… {} more record(s) not shown; raise `limit` or page with `since_seq`.",
                total - shown.len()
            );
        }
        if observation.is_empty() {
            observation = if query.is_empty() && kind_filter.is_none() {
                let after = since_seq.map_or_else(String::new, |since| format!(" after seq {since}"));
                format!(
                    "No records{after}. The journal has {} record(s).",
                    outcome.records.len()
                )
            } else {
                "No matching records.".to_string()
            };
        }

        // The cursor is the seq *after the page*, not after the whole match set: paging forward
        // from here must yield the next unseen record.
        let next_seq = shown.last().map_or_else(
            || since_seq.map_or(0, |since| since.saturating_add(1)),
            |report| report.seq.saturating_add(1),
        );

        Ok(ToolResult::new(
            observation,
            json!({
                "exists": true,
                "matches": total,
                "shown": shown.len(),
                "next_seq": next_seq,
                "query": query,
                "since_seq": since_seq,
            }),
        ))
    }
}

/// One matching journal record, rendered flat.
struct Report {
    seq: u64,
    at: String,
    kind: String,
    payload: String,
    artifact: Option<String>,
}

/// What a `grep_journal` call asked for.
struct Search<'a> {
    query: &'a str,
    since_seq: Option<u64>,
    kind: Option<Kind>,
}

/// Filter the journal, returning every match (the caller pages).
///
/// Separated from `execute` so the matching rules are testable without a `ToolCtx`, and so the
/// three conditions — sequence window, kind, text — read as three clauses instead of nesting
/// inside a loop inside a function that also formats output.
fn search_records(records: &[immortality_core::journal::JournalRecord], search: &Search<'_>) -> Vec<Report> {
    let needle = search.query.to_lowercase();
    let mut out = Vec::new();
    for record in records {
        // Inclusive lower bound: `since_seq` is the seq to start at, so the `next_seq` of one
        // page is exactly the first record the next page must show.
        if search.since_seq.is_some_and(|since| record.fields.seq < since) {
            continue;
        }
        if search.kind.is_some_and(|kind| kind != record.fields.kind) {
            continue;
        }
        if !needle.is_empty() {
            let haystack =
                format!("{} {}", kind_name(record.fields.kind), record.event.payload).to_lowercase();
            if !haystack.contains(&needle) {
                continue;
            }
        }
        out.push(Report {
            seq: record.fields.seq,
            at: record.fields.at.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string(),
            kind: kind_name(record.fields.kind),
            payload: record.event.payload.to_string(),
            artifact: record.event.artifact.clone(),
        });
    }
    out
}

/// Whether this event kind name is one the journal knows.
///
/// Parsed through `serde` rather than compared to a list here: the enum is the schema, and a
/// second copy of its names would drift.
fn parse_kind(name: &str) -> Result<Kind, ToolError> {
    serde_json::from_value::<Kind>(Value::String(name.to_string())).map_err(|_| {
        ToolError::Arguments(format!(
            "`{name}` is not a journal event kind. Try `tool_call_completed`, `attempt_started`, \
             `checkpoint_created`."
        ))
    })
}

/// The `snake_case` name of a kind, as the journal writes it.
fn kind_name(kind: Kind) -> String {
    serde_json::to_value(kind)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_else(|| format!("{kind:?}"))
}

/// Keep a line short enough to read, with the cut marked.
fn clip(line: &str, budget: usize) -> String {
    let max = budget.clamp(80, 2_000);
    if line.chars().count() <= max {
        return line.to_string();
    }
    let kept: String = line.chars().take(max).collect();
    format!("{kept}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::PolicyGate;
    use std::sync::Arc;

    use std::path::{Path, PathBuf};

    /// Where a state file lives, so the tests can read and clobber it directly.
    fn state_path(worktree: &Path, file: StateFile) -> PathBuf {
        worktree.join(AGENT_DIR).join(file.as_str())
    }

    struct Fixture {
        dir: tempfile::TempDir,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            std::fs::create_dir_all(dir.path().join(AGENT_DIR)).unwrap();
            for file in StateFile::ALL {
                std::fs::write(state_path(dir.path(), *file), format!("# {}\n", file.as_str())).unwrap();
            }
            Self { dir }
        }

        fn ctx(&self) -> ToolCtx {
            let gate = PolicyGate::with_defaults(
                self.dir.path(),
                immortality_core::config::Policy {
                    workspace_root: self.dir.path().parent().unwrap().to_path_buf(),
                    ..immortality_core::config::Policy::default()
                },
            )
            .unwrap();
            ToolCtx::new(Arc::new(gate), crate::OutputBudget::default())
        }

        fn write(&self, file: StateFile, body: &str) {
            std::fs::write(state_path(self.dir.path(), file), body).unwrap();
        }

        fn read(&self, file: StateFile) -> String {
            std::fs::read_to_string(state_path(self.dir.path(), file)).unwrap()
        }
    }

    fn read(fixture: &Fixture, args: Value) -> Result<ToolResult, ToolError> {
        ReadState.execute(&fixture.ctx(), Args::new(args))
    }

    fn update(fixture: &Fixture, args: Value) -> Result<ToolResult, ToolError> {
        UpdateState.execute(&fixture.ctx(), Args::new(args))
    }

    /// A fixture with a project checkout *and* a worktree, which is the daemon's real layout.
    ///
    /// The two `.agent/` directories deliberately disagree: the project's says a subtask is done,
    /// the worktree's still says it is pending. That is not a contrived state — it is what the
    /// `word-count` dogfood fixture was found in, because the worktree branch was cut before the
    /// plan was finished. A tool that reads the worktree copy reports a plan that has already
    /// moved on, which is the split-brain this fixture exists to make impossible.
    struct SplitFixture {
        dir: tempfile::TempDir,
        project: PathBuf,
        worktree: PathBuf,
    }

    impl SplitFixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let project = dir.path().join("project");
            let worktree = dir.path().join("project-agent").join("tsk_01");
            for root in [&project, &worktree] {
                std::fs::create_dir_all(root.join(AGENT_DIR)).unwrap();
            }
            for file in StateFile::ALL {
                std::fs::write(
                    state_path(&project, *file),
                    format!("# {} (project, current)\n", file.as_str()),
                )
                .unwrap();
                std::fs::write(
                    state_path(&worktree, *file),
                    format!("# {} (worktree, stale)\n", file.as_str()),
                )
                .unwrap();
            }
            Self {
                dir,
                project,
                worktree,
            }
        }

        fn ctx(&self) -> ToolCtx {
            let gate = PolicyGate::with_defaults(
                &self.worktree,
                immortality_core::config::Policy {
                    workspace_root: self.dir.path().to_path_buf(),
                    ..immortality_core::config::Policy::default()
                },
            )
            .unwrap()
            .with_project_agent_dir(self.project.join(AGENT_DIR));
            ToolCtx::new(Arc::new(gate), crate::OutputBudget::default())
        }

        fn project_body(&self, file: StateFile) -> String {
            std::fs::read_to_string(state_path(&self.project, file)).unwrap()
        }

        fn worktree_body(&self, file: StateFile) -> String {
            std::fs::read_to_string(state_path(&self.worktree, file)).unwrap()
        }
    }

    // ── read_state ────────────────────────────────────────────────────────────────────────

    #[test]
    fn read_state_numbers_the_lines() {
        let fixture = Fixture::new();
        fixture.write(StateFile::Notes, "one\ntwo\nthree\n");
        let result = read(&fixture, json!({"file": "NOTES.md"})).unwrap();
        assert!(result.observation.contains("1: one"), "{}", result.observation);
        assert!(result.observation.contains("3: three"), "{}", result.observation);
        assert_eq!(result.data["total_lines"], 3);
    }

    #[test]
    fn read_state_pages_with_offset_and_limit() {
        let fixture = Fixture::new();
        fixture.write(StateFile::Plan, "a\nb\nc\nd\ne\n");
        let result = read(&fixture, json!({"file": "PLAN.md", "offset": 2, "limit": 2})).unwrap();
        assert!(result.observation.contains("2: b"), "{}", result.observation);
        assert!(result.observation.contains("3: c"), "{}", result.observation);
        assert!(!result.observation.contains("4: d"), "{}", result.observation);
        assert!(
            result.observation.contains("more") || result.observation.contains("showing"),
            "the model must know there is more: {}",
            result.observation
        );
    }

    #[test]
    fn read_state_accepts_the_agent_prefixed_form() {
        // A model that just read `.agent/PLAN.md` will paste that path.
        let fixture = Fixture::new();
        fixture.write(StateFile::Plan, "the plan\n");
        let result = read(&fixture, json!({"file": ".agent/PLAN.md"})).unwrap();
        assert!(result.observation.contains("the plan"), "{}", result.observation);
        assert_eq!(result.data["file"], "PLAN.md");
    }

    #[test]
    fn read_state_of_a_missing_file_is_an_answer_not_an_error() {
        let fixture = Fixture::new();
        std::fs::remove_file(state_path(fixture.dir.path(), StateFile::Acceptance)).unwrap();
        let result = read(&fixture, json!({"file": "ACCEPTANCE.md"})).unwrap();
        assert_eq!(result.data["exists"], false);
        assert!(
            result.observation.contains("does not exist"),
            "{}",
            result.observation
        );
    }

    #[test]
    fn read_state_refuses_the_journal_with_a_pointer() {
        let fixture = Fixture::new();
        let error = read(&fixture, json!({"file": "JOURNAL.jsonl"})).unwrap_err();
        assert!(error.to_string().contains("grep_journal"), "{error}");
    }

    #[test]
    fn read_state_refuses_a_file_that_is_not_a_state_document() {
        let fixture = Fixture::new();
        let error = read(&fixture, json!({"file": "src/main.rs"})).unwrap_err();
        assert!(error.to_string().contains("not a state file"), "{error}");
    }

    #[test]
    fn read_state_past_the_end_says_so_without_failing() {
        let fixture = Fixture::new();
        fixture.write(StateFile::Notes, "only\n");
        let result = read(&fixture, json!({"file": "NOTES.md", "offset": 99})).unwrap();
        assert!(
            result.observation.contains("past the end"),
            "{}",
            result.observation
        );
    }

    // ── update_state ──────────────────────────────────────────────────────────────────────

    #[test]
    fn update_state_replaces_the_content() {
        let fixture = Fixture::new();
        let body = "a much longer body that is definitely more than twice the original header\n";
        update(&fixture, json!({"file": "NOTES.md", "content": body})).unwrap();
        assert_eq!(fixture.read(StateFile::Notes), body);
    }

    #[test]
    fn update_state_of_identical_content_changes_nothing() {
        let fixture = Fixture::new();
        let before = fixture.read(StateFile::Notes);
        let result = update(&fixture, json!({"file": "NOTES.md", "content": before})).unwrap();
        assert_eq!(result.data["unchanged"], true);
        assert_eq!(fixture.read(StateFile::Notes), before);
    }

    #[test]
    fn update_state_refuses_to_create_a_missing_plan_document() {
        // `PLAN.md` is what the scheduler executes, so its absence means a wrong directory rather
        // than a document waiting to be written. Creating an empty one would give the run an empty
        // plan.
        let fixture = Fixture::new();
        std::fs::remove_file(state_path(fixture.dir.path(), StateFile::Plan)).unwrap();
        let error = update(&fixture, json!({"file": "PLAN.md", "content": "x\n"})).unwrap_err();
        assert!(
            error.to_string().contains("does not exist"),
            "the refusal must name the cause: {error}"
        );
        assert!(!state_path(fixture.dir.path(), StateFile::Plan).exists());
    }

    #[test]
    fn update_state_creates_missing_working_memory() {
        // `NOTES.md` and `DECISIONS.md` have no producer besides the agent: the planner writes the
        // plan, `init` writes the config, and nothing writes notes. Refusing to create them meant
        // the agent could not record what it learned at all — measured on the live run, where the
        // refusal told the model to run `immortality init`, which does not create them, and the
        // model responded by writing `NOTES.md` in the repository root instead.
        let fixture = Fixture::new();
        for file in [StateFile::Notes, StateFile::Decisions] {
            let path = state_path(fixture.dir.path(), file);
            std::fs::remove_file(&path).unwrap();
            update(&fixture, json!({"file": file.as_str(), "content": "# learned\n"})).unwrap();
            assert_eq!(
                std::fs::read_to_string(&path).unwrap(),
                "# learned\n",
                "{} must be creatable",
                file.as_str()
            );
        }
    }

    #[test]
    fn state_tools_use_the_project_agent_dir_not_the_worktree_copy() {
        // The split-brain, as a test. A worktree's `.agent/` is a checkout from when the branch was
        // cut, so it is stale by construction; the plan the scheduler executes lives in the
        // project. Reading the worktree copy would hand the model a plan that has already moved on,
        // and writing to it would report success while changing nothing anyone reads.
        let split = SplitFixture::new();

        let result = ReadState
            .execute(&split.ctx(), Args::new(json!({"file": "TASKS.json"})))
            .unwrap();
        assert!(
            result.observation.contains("project, current"),
            "the project's copy must win: {}",
            result.observation
        );

        // `replace` because the fixture's notes are longer than the replacement: the halving
        // guard is a separate rule and firing it here would test that rule, not the destination.
        UpdateState
            .execute(
                &split.ctx(),
                Args::new(json!({"file": "NOTES.md", "content": "# new facts\n", "replace": true})),
            )
            .unwrap();
        assert_eq!(split.project_body(StateFile::Notes), "# new facts\n");
        assert!(
            split.worktree_body(StateFile::Notes).contains("stale"),
            "the worktree copy must be left alone: {}",
            split.worktree_body(StateFile::Notes)
        );
    }

    #[test]
    fn update_state_refuses_to_halve_a_file_without_replace() {
        // The failure this prevents: a model that meant to append sends only its new paragraph
        // and deletes the plan.
        let fixture = Fixture::new();
        let long = "x".repeat(1_000);
        fixture.write(StateFile::Notes, &long);
        let error = update(&fixture, json!({"file": "NOTES.md", "content": "short\n"})).unwrap_err();
        assert!(error.to_string().contains("state_shrink_refused"), "{error}");
        assert_eq!(fixture.read(StateFile::Notes), long, "the file must be untouched");
    }

    #[test]
    fn update_state_halves_a_file_when_replace_is_true() {
        let fixture = Fixture::new();
        fixture.write(StateFile::Notes, &"x".repeat(1_000));
        update(
            &fixture,
            json!({"file": "NOTES.md", "content": "short\n", "replace": true}),
        )
        .unwrap();
        assert_eq!(fixture.read(StateFile::Notes), "short\n");
    }

    #[test]
    fn update_state_allows_a_small_trim_that_is_not_a_halving() {
        // Trimming a paragraph out of a plan is normal editing and must not need a flag.
        let fixture = Fixture::new();
        fixture.write(StateFile::Plan, &"line\n".repeat(20));
        update(
            &fixture,
            json!({"file": "PLAN.md", "content": "line\n".repeat(15)}),
        )
        .unwrap();
        assert_eq!(fixture.read(StateFile::Plan).lines().count(), 15);
    }

    #[test]
    fn update_state_refuses_the_journal() {
        let fixture = Fixture::new();
        let error = update(&fixture, json!({"file": "JOURNAL.jsonl", "content": "forged\n"})).unwrap_err();
        assert!(error.to_string().contains("grep_journal"), "{error}");
    }

    #[test]
    fn update_state_cannot_reach_a_path_outside_the_state_dir() {
        // The `file` argument is an enum, so a path is not representable; this asserts the
        // parse refuses anything that is not one of the five names.
        let fixture = Fixture::new();
        let error = update(&fixture, json!({"file": "../README.md", "content": "x\n"})).unwrap_err();
        assert!(error.to_string().contains("not a state file"), "{error}");
    }

    // ── grep_journal ──────────────────────────────────────────────────────────────────────

    fn journal_fixture() -> (Fixture, immortality_core::ids::TaskId) {
        use immortality_core::journal::{Journal, JournalEvent, Kind, RecordFields};
        use immortality_core::time;

        let fixture = Fixture::new();
        let task = immortality_core::ids::TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let dir = fixture.dir.path().join(AGENT_DIR);
        std::fs::remove_file(dir.join("JOURNAL.jsonl")).ok();
        let mut journal = Journal::open(&dir, task.clone()).unwrap();
        for (kind, payload) in [
            (
                Kind::AttemptStarted,
                json!({"attempt": 1, "strategy": "read the file first"}),
            ),
            (
                Kind::ToolCallCompleted,
                json!({"tool": "read_file", "path": "src/main.rs"}),
            ),
            (
                Kind::CheckpointCreated,
                json!({"sha": "abc123", "reason": "before shell"}),
            ),
            (
                Kind::ToolCallCompleted,
                json!({"tool": "write_file", "path": "src/lib.rs"}),
            ),
        ] {
            journal
                .append(
                    RecordFields::new(journal.next_seq(), time::now(), task.clone(), kind),
                    JournalEvent::with_payload(payload),
                )
                .unwrap();
        }
        journal.close().unwrap();
        (fixture, task)
    }

    fn grep(fixture: &Fixture, args: Value) -> Result<ToolResult, ToolError> {
        GrepJournal.execute(&fixture.ctx(), Args::new(args))
    }

    #[test]
    fn grep_journal_matches_the_payload_text() {
        let (fixture, _) = journal_fixture();
        let result = grep(&fixture, json!({"query": "src/main.rs"})).unwrap();
        assert_eq!(result.data["matches"], 1, "{:?}", result.data);
        assert!(result.observation.contains("read_file"), "{}", result.observation);
    }

    #[test]
    fn grep_journal_filters_by_kind() {
        let (fixture, _) = journal_fixture();
        let result = grep(&fixture, json!({"kind": "tool_call_completed"})).unwrap();
        assert_eq!(result.data["matches"], 2, "{:?}", result.data);
    }

    #[test]
    fn grep_journal_pages_forward_with_since_seq() {
        // The contract that makes it a paging tool: `next_seq` from one call is `since_seq` for
        // the next, and no record appears twice.
        let (fixture, _) = journal_fixture();
        let first = grep(&fixture, json!({"limit": 2})).unwrap();
        assert_eq!(first.data["shown"], 2);
        let next = first.data["next_seq"].as_u64().unwrap();

        let second = grep(&fixture, json!({"limit": 2, "since_seq": next})).unwrap();
        assert_eq!(second.data["matches"], 2, "{:?}", second.data);
        assert!(
            second.observation.contains("write_file"),
            "{}",
            second.observation
        );
        assert!(
            !second.observation.contains("read_file"),
            "a paged call must not repeat earlier records: {}",
            second.observation
        );
    }

    #[test]
    fn grep_journal_with_no_filter_counts_the_whole_journal() {
        let (fixture, _) = journal_fixture();
        let result = grep(&fixture, json!({"limit": 100})).unwrap();
        assert_eq!(result.data["matches"], 4, "{:?}", result.data);
    }

    #[test]
    fn grep_journal_says_when_nothing_matches() {
        let (fixture, _) = journal_fixture();
        let result = grep(&fixture, json!({"query": "a string that is not there"})).unwrap();
        assert_eq!(result.data["matches"], 0);
        assert!(
            result.observation.contains("No matching records"),
            "{}",
            result.observation
        );
    }

    #[test]
    fn grep_journal_refuses_an_unknown_kind_by_name() {
        let (fixture, _) = journal_fixture();
        let error = grep(&fixture, json!({"kind": "not_a_kind"})).unwrap_err();
        assert!(error.to_string().contains("not a journal event kind"), "{error}");
    }

    #[test]
    fn grep_journal_without_a_journal_says_so() {
        let dir = tempfile::TempDir::new().unwrap();
        let gate = PolicyGate::with_defaults(
            dir.path(),
            immortality_core::config::Policy {
                workspace_root: dir.path().parent().unwrap().to_path_buf(),
                ..immortality_core::config::Policy::default()
            },
        )
        .unwrap();
        let ctx = ToolCtx::new(Arc::new(gate), crate::OutputBudget::default());
        let result = GrepJournal.execute(&ctx, Args::new(json!({}))).unwrap();
        assert_eq!(result.data["exists"], false);
    }

    #[test]
    fn grep_journal_refuses_a_corrupt_journal_rather_than_answering() {
        // Answers from a file that was edited underneath recovery are worse than no answers.
        let (fixture, _) = journal_fixture();
        let path = fixture.dir.path().join(AGENT_DIR).join("JOURNAL.jsonl");
        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines: Vec<&str> = text.lines().collect();
        // Drop a middle record: the chain no longer joins up.
        lines.remove(1);
        std::fs::write(&path, format!("{}\n", lines.join("\n"))).unwrap();

        let error = grep(&fixture, json!({})).unwrap_err();
        assert!(error.to_string().contains("journal_corrupt"), "{error}");
    }

    // ── shared ────────────────────────────────────────────────────────────────────────────

    #[test]
    fn the_schemas_are_well_formed() {
        let schemas = [
            ("read_state", ReadState.schema()),
            ("update_state", UpdateState.schema()),
            ("grep_journal", GrepJournal.schema()),
        ];
        for (name, schema) in schemas {
            assert_eq!(
                immortality_llm::schema::schema_defects(schema),
                Vec::new(),
                "{name}"
            );
        }
    }

    #[test]
    fn the_file_enum_offers_exactly_the_five_documents() {
        let offered = read_state_schema()["properties"]["file"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|value| value.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            offered,
            vec![
                "PLAN.md",
                "ACCEPTANCE.md",
                "TASKS.json",
                "DECISIONS.md",
                "NOTES.md"
            ]
        );
    }

    #[test]
    fn the_journal_is_not_in_the_enum() {
        for schema in [read_state_schema(), update_state_schema()] {
            let offered = schema["properties"]["file"]["enum"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|value| value.as_str())
                .collect::<Vec<_>>();
            assert!(
                !offered.iter().any(|name| name.contains("JOURNAL")),
                "{offered:?}"
            );
        }
    }

    #[test]
    fn effects_and_idempotency_match_the_design() {
        let args = Args::new(json!({"file": "NOTES.md", "content": "x"}));
        assert_eq!(ReadState.effect(&args), Effect::Read);
        assert_eq!(UpdateState.effect(&args), Effect::Mutate);
        assert_eq!(GrepJournal.effect(&args), Effect::Read);
        assert_eq!(ReadState.idempotency(&args), Idempotency::Idempotent);
        assert_eq!(UpdateState.idempotency(&args), Idempotency::ByContent);
        assert_eq!(GrepJournal.idempotency(&args), Idempotency::Idempotent);
    }

    #[test]
    fn paths_declare_a_read_for_reads_and_a_write_for_updates() {
        assert_eq!(
            ReadState.paths(&Args::new(json!({"file": "PLAN.md"}))),
            vec![PathUse::read(".agent/PLAN.md")]
        );
        assert_eq!(
            UpdateState.paths(&Args::new(json!({"file": "PLAN.md"}))),
            vec![PathUse::write(".agent/PLAN.md")]
        );
    }
}
