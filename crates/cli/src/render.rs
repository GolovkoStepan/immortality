//! Rendering RPC results for a terminal.
//!
//! Kept apart from the commands so that "what the human sees" can be tested without a daemon,
//! a socket or a process. Every function here takes the JSON the daemon answered with and
//! returns text; none of them read the filesystem or the clock.
//!
//! # Two shapes, one reason
//!
//! Short commands (`status`, `escalations`) print a compact report. `attach` prints a *stream*,
//! and a stream of hundreds of uniform lines is unreadable: the eye cannot tell a model's
//! sentence from a tool result. So the stream is rendered as blocks — a call opens with its
//! name and intended effect, the result follows indented — while everything that is not a tool
//! call stays one line with its sequence number.
//!
//! The sequence number stays on every block's first line because it is what `--since-seq` takes;
//! a human resuming a stream reads it off the screen.

// `pub`, not `pub(crate)`: this is a binary crate, so the module is only reachable from the
// crate root and `pub(crate)` on its items is what `clippy::redundant_pub_crate` objects to.
// `unreachable_pub` then objects to `pub`. The pair is unsatisfiable for a private module in a
// binary, so the reachability lint is allowed here and only here.
#![allow(unreachable_pub)]

use std::collections::BTreeMap;

use serde_json::Value;

use crate::style::Style;

/// A status report: the projects, then a line per task.
///
/// Deliberately terse. This is what a human glances at to answer "is it still working?", and a
/// screen of JSON does not answer that. The `--json` flag exists for everything else.
#[must_use]
pub fn status(result: &Value, style: Style) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let projects = result
        .get("projects")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();

    if projects.is_empty() {
        out.push_str("no projects registered.\n");
        out.push_str("run `immortality init` in a repository to register it.\n");
        return out;
    }

    let _ = writeln!(out, "{}", style.bold("projects"));
    for project in projects {
        let path = text(project, "path");
        let id = style.cyan(&text(project, "id"));
        match project.get("remote").and_then(Value::as_str) {
            Some(remote) => {
                let _ = writeln!(out, "  {id}  {path}  ({remote})");
            }
            None => {
                let _ = writeln!(out, "  {id}  {path}");
            }
        }
    }

    let tasks = result
        .get("tasks")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    out.push('\n');
    if tasks.is_empty() {
        out.push_str("no tasks yet.\n");
        return out;
    }

    let _ = writeln!(out, "{}", style.bold("tasks"));
    for task in tasks {
        render_task_line(&mut out, task, style);
    }

    // Why nothing is running, when nothing is. An idle daemon with a table a human must interpret
    // is the failure mode this project exists to avoid; the four reasons have four different
    // fixes, so the line names the one that applies rather than leaving it to be guessed.
    if let Some(scheduling) = result.get("scheduling") {
        if let Some(error) = scheduling.get("error").and_then(Value::as_str) {
            let _ = writeln!(out, "\n{}: {error}", style.yellow("scheduling unavailable"));
        } else if let Some(reason) = scheduling.get("reason").and_then(Value::as_str)
            && !reason.is_empty()
        {
            let _ = writeln!(out, "\n{}  {reason}", style.yellow("idle:"));
        }
    }
    out
}

/// One task in `status`, including the two things that may need a human: approvals and a corrupt
/// journal.
fn render_task_line(out: &mut String, task: &Value, style: Style) {
    use std::fmt::Write as _;

    let journal = task.get("journal").cloned().unwrap_or(Value::Null);
    if let Some(error) = journal.get("error").and_then(Value::as_str) {
        let _ = writeln!(
            out,
            "  {}  {}",
            style.cyan(&text(task, "task")),
            style.red(&format!("journal unreadable: {error}"))
        );
        return;
    }
    let last_kind = journal
        .get("last")
        .and_then(|last| last.get("kind"))
        .and_then(Value::as_str)
        .unwrap_or("no records");
    let records = journal.get("records").and_then(Value::as_u64).unwrap_or(0);
    let corrupt = journal.get("corrupt").and_then(Value::as_bool).unwrap_or(false);
    let escalations = task.get("open_escalations").and_then(Value::as_u64).unwrap_or(0);

    let _ = write!(
        out,
        "  {}  {records} record(s), last: {last_kind}",
        style.cyan(&text(task, "task"))
    );
    if escalations > 0 {
        // First, because it is the one thing that may need the human.
        let _ = write!(
            out,
            "  {}",
            style.yellow(&format!("[{escalations} awaiting approval]"))
        );
    }
    if corrupt {
        let _ = write!(out, "  {}", style.red("[JOURNAL CORRUPT]"));
    }
    out.push('\n');
}

/// The open questions, as a numbered list a human can act on.
#[must_use]
pub fn escalations(result: &Value, style: Style) -> String {
    use std::fmt::Write as _;

    let entries = result
        .get("escalations")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if entries.is_empty() {
        return "nothing is waiting for a human.\n".to_string();
    }

    let mut out = String::new();
    let _ = writeln!(
        out,
        "{}",
        style.bold(&format!("{} question(s) waiting:", entries.len()))
    );
    for entry in entries {
        let blocking = if entry.get("blocking").and_then(Value::as_bool) == Some(true) {
            style.red("  [BLOCKING]")
        } else {
            String::new()
        };
        let _ = writeln!(out, "\n  {}{blocking}", style.cyan(&text(entry, "id")));
        let _ = writeln!(out, "  task: {}", text(entry, "task"));
        let _ = writeln!(out, "  kind: {}", text(entry, "kind"));
        let _ = writeln!(out, "  {}", text(entry, "question"));
        let options: Vec<&str> = entry
            .get("options")
            .and_then(Value::as_array)
            .map(|options| options.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if !options.is_empty() {
            let _ = writeln!(out, "  answers: {}", options.join(", "));
        }
    }
    out.push_str("\nanswer one with: immortality approve <id> <answer>\n");
    out
}

/// The confirmation after an answer.
#[must_use]
pub fn approved(result: &Value, style: Style) -> String {
    format!(
        "answered {}: {}\n",
        style.cyan(&text(result, "id")),
        style.green(&text(result, "answer"))
    )
}

/// The plan as read from disk, or the reason it cannot be read.
///
/// The load error is printed as a report, not raised: a tree that fails validation is exactly
/// what a human ran `immortality plan` to see, and the daemon's own message already names the
/// subtask and the field.
#[must_use]
pub fn plan(result: &Value, style: Style) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let _ = writeln!(
        out,
        "{}  {}",
        style.bold(&text(result, "project")),
        style.dim(&text(result, "root"))
    );
    if let Some(error) = result.get("load_error").and_then(Value::as_str) {
        let _ = writeln!(out, "\n{}", style.red(&format!("cannot read the plan: {error}")));
        // The file's location is the actionable half: the message says *what* is wrong, this says
        // where to fix it.
        let _ = writeln!(
            out,
            "  expected at {}/.agent/TASKS.json and {}/.agent/ACCEPTANCE.md",
            text(result, "root"),
            text(result, "root")
        );
        return out;
    }
    let subtasks = result
        .get("subtasks")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if subtasks.is_empty() {
        let _ = writeln!(out, "\nthe plan is valid and has no subtasks.");
        return out;
    }
    let depth = result
        .get("max_subtask_depth")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let _ = writeln!(
        out,
        "\n{} ({}), depth limit {depth}",
        style.bold(&format!("{} subtask(s)", subtasks.len())),
        text(result, "project")
    );
    for subtask in subtasks {
        render_subtask_line(&mut out, subtask, 0, style);
    }
    out
}

/// A plan the planner has just written: the files it touched, then the tree.
///
/// Same tree rendering as [`plan`], with the files and the goal above it. A human who just spent
/// minutes of prefill on this needs to see *where* it landed — the roadmap's contract is that
/// they may open those files and rewrite them before the run starts.
#[must_use]
pub fn planned(result: &Value, style: Style) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let _ = writeln!(
        out,
        "{}",
        style.green(&format!("planned: {}", text(result, "goal")))
    );
    let root = text(result, "root");
    for name in result
        .get("written")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .filter_map(Value::as_str)
    {
        let _ = writeln!(out, "  wrote {root}/.agent/{name}");
    }
    // The tree itself through the same renderer as `immortality plan`, because it *is* the same
    // tree: a human who has seen one should not have to learn to read the other.
    out.push_str(&plan(result, style));
    out.push_str("\nedit these files before `immortality start` if the plan is not what you meant.\n");
    out
}

/// One subtask and its children, indented by depth.
fn render_subtask_line(out: &mut String, subtask: &Value, depth: usize, style: Style) {
    use std::fmt::Write as _;

    let state = text(subtask, "state");
    let painted = match state.as_str() {
        "done" => style.green(&state),
        "blocked" | "failed" => style.red(&state),
        "pending" => style.dim(&state),
        _ => style.yellow(&state),
    };
    let indent = "  ".repeat(depth + 1);
    let _ = writeln!(
        out,
        "{indent}{painted:<12} {}  {}",
        style.cyan(&text(subtask, "id")),
        text(subtask, "title")
    );
    for child in subtask
        .get("children")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        render_subtask_line(out, child, depth + 1, style);
    }
}

/// An admission: the task id, where it runs, and what to type to watch it.
///
/// The first thing a human needs after `immortality start` is the id, because every other command
/// takes it; the second is that the run is *theirs to watch* and not already over. That is why the
/// next command is printed rather than described.
#[must_use]
pub fn admitted(result: &Value, style: Style) -> String {
    use std::fmt::Write as _;

    let task = text(result, "task");
    let resumed = result.get("resumed").and_then(Value::as_bool) == Some(true);
    let planned = result.get("planned").and_then(Value::as_bool) == Some(true);
    let mut out = String::new();
    let verb = if resumed { "resumed" } else { "started" };
    let _ = writeln!(out, "{}", style.green(&format!("{verb} {task}")));
    let _ = writeln!(out, "  project:  {}", text(result, "path"));
    let _ = writeln!(out, "  state:    {}", text(result, "state"));
    if let Some(goal) = result.get("goal").and_then(Value::as_str)
        && !goal.is_empty()
    {
        let _ = writeln!(out, "  goal:     {goal}");
    }
    if let Some(count) = result.get("subtasks").and_then(Value::as_u64) {
        let _ = writeln!(out, "  subtasks: {count}");
    }
    if planned {
        let _ = writeln!(
            out,
            "  planned:  the tree was just written; edit .agent/TASKS.json if you want to \
             change it before the first subtask runs"
        );
    }
    let _ = writeln!(out, "watch it with `immortality attach {task}`");
    out
}

/// A transcript: every record, one block or line each, in journal order.
///
/// Rendered through the same [`Renderer`] as `attach`, because a transcript that looked different
/// from the live stream would make a human re-learn how to read it — and the records are the same
/// records.
#[must_use]
pub fn log(result: &Value, style: Style) -> String {
    use std::fmt::Write as _;

    let records = result
        .get("records")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let mut out = String::new();
    let corrupt = result.get("corrupt").and_then(Value::as_bool) == Some(true);
    if corrupt {
        let _ = writeln!(
            out,
            "{}",
            style.red("the journal fails its own chain check; records below are what survived")
        );
    }
    if records.is_empty() {
        // Two different facts, and the difference changes what a human does: an unreadable journal
        // is not a task that never ran, and calling it "not started" would send them looking for a
        // task that is already there.
        let _ = writeln!(
            out,
            "{}",
            if corrupt {
                format!("no record of {} survived the damage.", text(result, "task"))
            } else {
                format!(
                    "no records for {} yet: the task has not started.",
                    text(result, "task")
                )
            }
        );
        return out;
    }
    let mut renderer = Renderer::new(style);
    for record in records {
        out.push_str(&renderer.line(record));
    }
    let _ = writeln!(
        out,
        "{}",
        style.dim(&format!(
            "{} record(s), next seq {}",
            records.len(),
            result.get("next_seq").and_then(Value::as_u64).unwrap_or(0)
        ))
    );
    out
}

/// The v0.1 numbers, as a human reads them.
///
/// One block per task with the counts that decide the milestone: subtasks closed by green gates,
/// how many of those closed with an escalation open, detector firings by `detector/level`, and the
/// restarts and escalations that say how much of the run was the agent and how much was the
/// ladder. A task whose journal cannot be read is reported as unreadable rather than as zeros —
/// zero and "unknown" are different facts, and only one of them is evidence.
/// The line that reports the longest run of green closures against the v0.1 bar.
///
/// A reader has to be able to tell "three in a row" from "two in a row" without comparing the
/// number against the document, so a report that meets the bar is painted green and one that does
/// not is dim. The comparison is done here rather than left to the reader because the bar is the
/// whole point of the command.
fn streak_line(streak: u64, style: Style) -> String {
    let text = format!("longest streak {streak} in a row — the v0.1 bar is three");
    if streak >= 3 {
        style.green(&text)
    } else {
        style.dim(&text)
    }
}

#[must_use]
pub fn metrics(result: &Value, style: Style) -> String {
    use std::fmt::Write as _;

    let tasks = result
        .get("tasks")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if tasks.is_empty() {
        return format!("{}\n", style.dim("no tasks yet."));
    }
    let mut out = String::new();
    for task in tasks {
        let id = text(task, "task");
        if let Some(reason) = task.get("unreadable").and_then(Value::as_str) {
            let _ = writeln!(out, "{} {}", style.bold(&id), style.red("unreadable"));
            let _ = writeln!(out, "  {}", style.dim(reason));
            continue;
        }

        let closed = task.get("closed_green").and_then(Value::as_u64).unwrap_or(0);
        let with_escalation = task
            .get("closed_with_escalation")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let _ = writeln!(
            out,
            "{} {} closed by a green gate, {} of them with an escalation open",
            style.bold(&id),
            style.green(&closed.to_string()),
            with_escalation
        );
        let streak = task
            .get("longest_green_streak")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let _ = writeln!(out, "{}", streak_line(streak, style));

        let subtasks = task
            .get("subtasks")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        for subtask in subtasks {
            let state = text(subtask, "state");
            let painted = if state == "done" {
                style.green(&state)
            } else {
                style.yellow(&state)
            };
            let aid = subtask.get("closed_with_escalation").and_then(Value::as_bool);
            let _ = writeln!(
                out,
                "  {} {}{}",
                painted,
                style.dim(&text(subtask, "subtask")),
                if aid == Some(true) {
                    style.dim(" (escalation open)")
                } else {
                    String::new()
                }
            );
        }

        let restarts = task.get("restarts").and_then(Value::as_u64).unwrap_or(0);
        let total = task.get("escalations_total").and_then(Value::as_u64).unwrap_or(0);
        let pending = task
            .get("escalations_pending")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let _ = writeln!(
            out,
            "  {}",
            style.dim(&format!(
                "restarts {restarts}, escalations {total} ({pending} still open)"
            ))
        );

        let detectors = task
            .get("detectors")
            .and_then(Value::as_object)
            .map(|object| {
                let mut pairs: Vec<(&String, u64)> = object
                    .iter()
                    .map(|(name, count)| (name, count.as_u64().unwrap_or(0)))
                    .collect();
                pairs.sort();
                pairs
            })
            .unwrap_or_default();
        if detectors.is_empty() {
            let _ = writeln!(out, "  {}", style.dim("no detector fired."));
        } else {
            for (name, count) in detectors {
                let _ = writeln!(out, "  {name} {}", style.yellow(&count.to_string()));
            }
        }

        if task.get("corrupt").and_then(Value::as_bool) == Some(true) {
            let _ = writeln!(
                out,
                "  {}",
                style.red("the journal fails its own chain check; these counts are incomplete")
            );
        }
    }
    out
}

/// What `gc` did, or what it would do.
#[must_use]
pub fn gc_report(result: &Value, style: Style) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    if result.get("dry_run").and_then(Value::as_bool) == Some(true) {
        let _ = writeln!(out, "{}", style.yellow("dry run: nothing was deleted"));
    }
    let protected = result.get("protected").and_then(Value::as_u64).unwrap_or(0);
    let unfinished = result
        .get("unfinished")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let _ = writeln!(
        out,
        "{} artifact digest(s) kept because {} task(s) are unfinished",
        protected,
        unfinished.len()
    );
    for task in unfinished {
        let _ = writeln!(out, "  {}", style.dim(task.as_str().unwrap_or("?")));
    }
    if result.get("dry_run").and_then(Value::as_bool) == Some(true) {
        return out;
    }
    let removed = result
        .get("artifacts_removed")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let bytes = result.get("bytes_reclaimed").and_then(Value::as_u64).unwrap_or(0);
    let temp = result.get("temp_removed").and_then(Value::as_u64).unwrap_or(0);
    let _ = writeln!(
        out,
        "{} artifact(s) removed, {} reclaimed, {} temp file(s) removed",
        removed,
        human_bytes(bytes),
        temp
    );
    if result.get("noop").and_then(Value::as_bool) == Some(true) {
        let _ = writeln!(out, "{}", style.dim("nothing needed collecting."));
    }
    out
}

/// Bytes as a human reads them.
fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// The confirmation after a merge.
#[must_use]
pub fn merged(result: &Value, style: Style) -> String {
    format!(
        "merged {} into {}\n  state: {}\n",
        style.cyan(&text(result, "branch")),
        text(result, "path"),
        style.green(&text(result, "state"))
    )
}

/// A client-side notice, for the things that are not journal records.
///
/// Written in the same gutter as the stream so it reads as part of it, but with no sequence
/// number: it is about the connection, not about the task, and inventing a number for it would
/// make `--since-seq` ambiguous.
#[must_use]
pub fn notice(notice: crate::client::Notice, style: Style) -> String {
    use std::fmt::Write as _;

    match notice {
        crate::client::Notice::Reconnecting { from, attempt } => {
            let mut out = String::new();
            let _ = writeln!(
                out,
                "{} {} {}",
                style.dim("      "),
                style.yellow("⟳"),
                style.yellow(&format!(
                    "stream closed; reconnecting from seq {from} (attempt {attempt})"
                ))
            );
            out
        }
    }
}

/// Renders one `attach` stream.
///
/// Holds the only piece of state the block layout needs: which tool each open call belongs to.
/// The intent record carries the tool name and the completion record does not, so without this
/// map a result would appear under no heading at all — and a stream attached mid-run (`--since-seq`)
/// would have results with no intents, which is exactly when the name is missing.
///
/// It is stateful but still pure: no files, no clock, no globals. Feeding the same records in the
/// same order produces the same bytes on every machine.
#[derive(Debug, Clone)]
pub struct Renderer {
    style: Style,
    open: BTreeMap<String, String>,
}

impl Renderer {
    /// A renderer that emits `style`'s sequences. Use [`Style::plain`] for a pipe or a test.
    #[must_use]
    pub const fn new(style: Style) -> Self {
        Self {
            style,
            open: BTreeMap::new(),
        }
    }

    /// Render one journal record, returning the text to write.
    ///
    /// May return an empty string: `tool_call_started` is deliberately silent (see
    /// [`Renderer::started`]).
    #[must_use]
    pub fn line(&mut self, record: &Value) -> String {
        let kind = record.get("kind").and_then(Value::as_str).unwrap_or("unknown");
        let call = record
            .get("payload")
            .and_then(|payload| payload.get("call"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let seq = record.get("seq").and_then(Value::as_u64).unwrap_or(0);

        match kind {
            "tool_call_intent" => self.intent(seq, record, call.as_deref()),
            // Silent on purpose: a separate line doubles the output of every call and the
            // intent line above is already the visible signal that a call is running.
            "tool_call_started" => String::new(),
            "tool_call_completed" => self.completed(record, call.as_deref()),
            "tool_call_failed" | "tool_call_unknown" => self.failed(record, call.as_deref()),
            "escalation_opened" => self.escalation_opened(seq, record),
            "escalation_resolved" => self.escalation_resolved(seq, record),
            "model_response" => self.model_response(seq, record),
            _ => self.one_line(seq, record),
        }
    }

    /// A call's opening line: what will run and what it is expected to change.
    ///
    /// The expected effect is the sentence the model wrote for a human, and it is the single most
    /// useful thing to show before the tool does anything: it turns `read_file` into "check the
    /// config before editing it".
    fn intent(&mut self, seq: u64, record: &Value, call: Option<&str>) -> String {
        let payload = record.get("payload").cloned().unwrap_or(Value::Null);
        let tool = payload
            .get("tool")
            .and_then(Value::as_str)
            .unwrap_or("unknown tool")
            .to_string();
        let effect = payload
            .get("expected_effect")
            .and_then(Value::as_str)
            .unwrap_or("");

        if let Some(call) = call {
            self.open.insert(call.to_string(), tool.clone());
        }

        let mut out = String::from("\n");
        out.push_str(&self.gutter(seq));
        out.push_str(&self.style.green("⏺"));
        out.push(' ');
        out.push_str(&self.style.bold(&tool));
        if !effect.is_empty() {
            out.push_str("  ");
            out.push_str(&self.style.dim(&truncate(effect, 160)));
        }
        // `Unknown` is the idempotency class that must be probed after a crash; the human seeing
        // this line benefits more from that flag than from any other field.
        if payload.get("idempotency").and_then(Value::as_str) == Some("unknown") {
            out.push(' ');
            out.push_str(&self.style.yellow("[unsafe to retry]"));
        }
        out.push('\n');
        out
    }

    /// The result, indented under its call's opening line.
    ///
    /// The tool name is repeated only when the intent for this call was seen. That happens for a
    /// normal run and not when a human attaches mid-run with `--since-seq`, where there was no
    /// intent to learn it from — in that case the indent alone is honest, rather than a guess.
    fn completed(&mut self, record: &Value, call: Option<&str>) -> String {
        let payload = record.get("payload").cloned().unwrap_or(Value::Null);
        let tool = call.and_then(|call| self.open.remove(call));
        let detail = describe_result(&payload);
        let mut out = String::new();
        out.push_str("   ");
        out.push_str(&self.style.dim("⎿"));
        out.push(' ');
        if let Some(tool) = tool {
            out.push_str(&self.style.dim(&format!("{tool}: ")));
        }
        out.push_str(&detail);
        out.push('\n');
        out
    }

    /// A failure, in red, with whatever the tool said went wrong.
    fn failed(&mut self, record: &Value, call: Option<&str>) -> String {
        let payload = record.get("payload").cloned().unwrap_or(Value::Null);
        let tool = call.and_then(|call| self.open.remove(call));
        let detail = payload
            .get("error")
            .or_else(|| payload.get("detail"))
            .or_else(|| payload.get("reason"))
            .map_or_else(|| describe_result(&payload), render_value);
        let mut out = String::new();
        out.push_str("   ");
        out.push_str(&self.style.dim("⎿"));
        out.push(' ');
        if let Some(tool) = tool {
            out.push_str(&self.style.dim(&format!("{tool}: ")));
        }
        out.push_str(&self.style.red(&truncate(&detail, 200)));
        out.push('\n');
        out
    }

    /// A question for the human, loud enough to be seen in a long stream.
    fn escalation_opened(&self, seq: u64, record: &Value) -> String {
        let payload = record.get("payload").cloned().unwrap_or(Value::Null);
        let id = payload.get("id").and_then(Value::as_str).unwrap_or("?");
        let blocking = payload.get("blocking").and_then(Value::as_bool) == Some(true);
        let mut out = String::from("\n");
        out.push_str(&self.gutter(seq));
        out.push_str(&self.style.yellow("?"));
        out.push(' ');
        out.push_str(&self.style.bold("a human is needed"));
        if blocking {
            out.push(' ');
            out.push_str(&self.style.red("[BLOCKING]"));
        }
        out.push_str("  ");
        out.push_str(id);
        out.push('\n');

        let question = payload.get("question").and_then(Value::as_str);
        if let Some(question) = question {
            out.push_str("        ");
            out.push_str(&truncate(question, 200));
            out.push('\n');
        }
        let options = payload
            .get("options")
            .and_then(Value::as_array)
            .map(|options| {
                options
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        let hint = if options.is_empty() {
            format!("answer with: immortality approve {id} <answer>")
        } else {
            format!("answer with: immortality approve {id} <{options}>")
        };
        out.push_str("        ");
        out.push_str(&self.style.dim(&hint));
        out.push('\n');
        out
    }

    /// The answer to a question, so a watcher sees the decision without opening `escalations`.
    fn escalation_resolved(&self, seq: u64, record: &Value) -> String {
        let payload = record.get("payload").cloned().unwrap_or(Value::Null);
        let answer = payload
            .get("answer")
            .map_or_else(|| "?".to_string(), render_value);
        let mut out = self.gutter(seq);
        out.push_str(&self.style.dim("escalation_resolved"));
        out.push_str("  ");
        out.push_str(&self.style.green(&answer));
        out.push('\n');
        out
    }

    /// The model's own words, in full rather than as a `kind  message` line.
    ///
    /// Prose is what a human actually reads during a long run; formatting it like a log entry is
    /// how a transcript becomes unreadable.
    fn model_response(&self, seq: u64, record: &Value) -> String {
        let payload = record.get("payload").cloned().unwrap_or(Value::Null);
        let body = payload
            .get("message")
            .or_else(|| payload.get("text"))
            .or_else(|| payload.get("content"))
            .and_then(Value::as_str);
        let Some(body) = body else {
            return self.one_line(seq, record);
        };

        let mut out = String::from("\n");
        out.push_str(&self.gutter(seq));
        out.push_str(&self.style.cyan("⏺"));
        out.push(' ');
        out.push_str(&self.style.bold("model"));
        out.push('\n');
        for line in body.lines() {
            out.push_str("  ");
            out.push_str(line);
            out.push('\n');
        }
        out
    }

    /// Everything that is not a tool call or a model turn: one indented line.
    fn one_line(&self, seq: u64, record: &Value) -> String {
        let kind = record.get("kind").and_then(Value::as_str).unwrap_or("unknown");
        let colour = match kind {
            "daemon_recovered" | "journal_truncated_tail" => self.style.yellow(kind),
            "backend_unavailable" => self.style.red(kind),
            _ => self.style.dim(kind),
        };
        let mut out = self.gutter(seq);
        out.push_str(&colour);
        if let Some(summary) = summarise(record) {
            out.push_str("  ");
            out.push_str(&summary);
        }
        out.push('\n');
        out
    }

    /// The right-aligned sequence number every block starts with, plus the separating space.
    ///
    /// The space is part of the gutter rather than the caller's business: forgetting it once
    /// produced `     1⏺ model` on screen, which reads as a different number.
    fn gutter(&self, seq: u64) -> String {
        self.style.dim(&format!("{seq:>6} "))
    }
}

/// The most useful detail of a tool's structured result, as one line.
///
/// A tool result is `data`, not prose: `exit_code`, `path`, `matches`, and so on. Rendering a
/// fixed key set keeps the line stable for any tool, including one added later — a missing key
/// simply does not appear, and a tool with no known keys falls back to the first scalar it has.
fn describe_result(payload: &Value) -> String {
    // Ordered by what answers "what happened" fastest, in the words a human already uses.
    const FIELDS: [(&str, &str); 14] = [
        ("path", ""),
        ("command", "$ "),
        ("branch", "branch "),
        ("exit_code", "exit "),
        ("handle", "handle "),
        ("finish", "finish "),
        ("matches", ""),
        ("files_with_matches", "files "),
        ("total_lines", "lines "),
        ("replacements", "edits "),
        ("lines", "at line "),
        ("created", ""),
        ("unchanged", ""),
        ("steps", "steps "),
    ];

    if let Some(observation) = payload.get("observation").and_then(Value::as_str) {
        return truncate(&first_line(observation), 200);
    }

    let mut parts: Vec<String> = Vec::new();
    for (key, prefix) in FIELDS {
        let Some(value) = payload.get(key) else {
            continue;
        };
        match value {
            Value::Bool(true) => parts.push(if key == "created" {
                "created".to_string()
            } else if key == "unchanged" {
                "unchanged".to_string()
            } else {
                format!("{prefix}{key}")
            }),
            Value::Bool(false) | Value::Null => {}
            Value::Array(items) => {
                // `edit_file` reports the line numbers it touched as an array. Joining the
                // scalars reads as "at line 2, 3"; the full list is in `--json`.
                let joined = items
                    .iter()
                    .take(5)
                    .map(render_value)
                    .collect::<Vec<_>>()
                    .join(", ");
                if !joined.is_empty() {
                    parts.push(format!("{prefix}{joined}"));
                }
            }
            _ => parts.push(format!("{prefix}{}", render_value(value))),
        }
        if parts.len() == 3 {
            break;
        }
    }

    if parts.is_empty() {
        // No known field: show the compact form rather than an empty result line, which would
        // read as a tool that did nothing.
        for (key, value) in payload
            .as_object()
            .map(|object| object.iter().collect::<Vec<_>>())
            .unwrap_or_default()
        {
            if key == "call" || key == "artifact" {
                continue;
            }
            if value.is_null() || value.is_array() || value.is_object() {
                continue;
            }
            parts.push(format!("{key}={}", render_value(value)));
            break;
        }
    }
    if parts.is_empty() {
        return "ok".to_string();
    }
    truncate(&parts.join(" · "), 200)
}

/// The most useful single field of a non-tool record, for the one-line view.
///
/// Ordered by what a human watching a run needs first: what the model said, how a step ended,
/// why something was refused. Unknown kinds fall back to nothing rather than a dump —
/// `attach --json` is the complete view.
fn summarise(record: &Value) -> Option<String> {
    let payload = record.get("payload").filter(|value| value.is_object())?;
    for key in [
        "question", "message", "text", "answer", "state", "error", "reason", "summary",
    ] {
        if let Some(value) = payload.get(key) {
            let rendered = render_value(value);
            if !rendered.is_empty() {
                return Some(truncate(&first_line(&rendered), 160));
            }
        }
    }
    // No known key: show the first scalar rather than nothing. `task_state {"state": ...}` is
    // covered above, but a future kind with a single field should still say something.
    for (key, value) in payload.as_object().map(|o| o.iter().collect::<Vec<_>>())? {
        if key == "call" || key == "artifact" {
            continue;
        }
        if value.is_array() || value.is_object() || value.is_null() {
            continue;
        }
        return Some(truncate(&format!("{key}={}", render_value(value)), 160));
    }
    None
}

/// A JSON scalar as text, without the quotes a string would otherwise carry.
fn render_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// The first non-empty line of a possibly multi-line string.
fn first_line(text: &str) -> String {
    text.lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or_default()
        .to_string()
}

/// A string field, or a placeholder rather than a panic.
fn text(value: &Value, key: &str) -> String {
    value.get(key).and_then(Value::as_str).unwrap_or("?").to_string()
}

/// Cut a string to `limit` bytes on a character boundary.
///
/// A byte slice would panic or split a UTF-8 sequence; these strings can be Russian, so the
/// boundary matters.
fn truncate(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_string();
    }
    let mut end = limit;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The style every test uses unless it is about colour: a pipe, a log file, CI.
    const PLAIN: Style = Style::plain();

    fn rendered(record: &serde_json::Value) -> String {
        Renderer::new(PLAIN).line(record)
    }

    #[test]
    fn an_empty_registry_says_what_to_do_next() {
        let out = status(&json!({"projects": [], "tasks": []}), PLAIN);
        assert!(out.contains("no projects"), "{out}");
        assert!(out.contains("immortality init"), "{out}");
    }

    #[test]
    fn a_registered_project_with_no_tasks_is_not_an_error() {
        let out = status(
            &json!({
                "projects": [{"id": "immortality-9a1e0c", "path": "/repo"}],
                "tasks": []
            }),
            PLAIN,
        );
        assert!(out.contains("immortality-9a1e0c"), "{out}");
        assert!(out.contains("/repo"), "{out}");
        assert!(out.contains("no tasks yet"), "{out}");
    }

    #[test]
    fn a_task_line_carries_its_record_count_and_last_kind() {
        let out = status(
            &json!({
                "projects": [{"id": "p", "path": "/repo"}],
                "tasks": [{
                    "task": "tsk_01",
                    "journal": {"records": 12, "last": {"kind": "tool_call_completed"}, "corrupt": false},
                    "open_escalations": 0
                }]
            }),
            PLAIN,
        );
        assert!(out.contains("tsk_01"), "{out}");
        assert!(out.contains("12 record"), "{out}");
        assert!(out.contains("tool_call_completed"), "{out}");
        assert!(!out.contains("awaiting approval"), "{out}");
    }

    #[test]
    fn pending_approvals_and_corruption_are_called_out_first() {
        // Both are the reason a human opened this command at all.
        let out = status(
            &json!({
                "projects": [{"id": "p", "path": "/repo"}],
                "tasks": [{
                    "task": "tsk_01",
                    "journal": {"records": 3, "last": {"kind": "task_state"}, "corrupt": true},
                    "open_escalations": 2
                }]
            }),
            PLAIN,
        );
        assert!(out.contains("2 awaiting approval"), "{out}");
        assert!(out.contains("JOURNAL CORRUPT"), "{out}");
    }

    #[test]
    fn a_task_with_no_records_says_so_rather_than_printing_nothing() {
        let out = status(
            &json!({
                "projects": [{"id": "p", "path": "/repo"}],
                "tasks": [{"task": "tsk_01", "journal": {"records": 0}, "open_escalations": 0}]
            }),
            PLAIN,
        );
        assert!(out.contains("0 record"), "{out}");
        assert!(out.contains("no records"), "{out}");
    }

    #[test]
    fn an_unreadable_journal_is_reported_rather_than_hidden() {
        let out = status(
            &json!({
                "projects": [{"id": "p", "path": "/repo"}],
                "tasks": [{"task": "tsk_01", "journal": {"error": "chain broken"}, "open_escalations": 0}]
            }),
            PLAIN,
        );
        assert!(out.contains("chain broken"), "{out}");
    }

    #[test]
    fn no_escalations_says_so_plainly() {
        let out = escalations(&json!({"escalations": []}), PLAIN);
        assert!(out.contains("nothing is waiting"), "{out}");
    }

    #[test]
    fn an_escalation_shows_the_question_and_its_options() {
        let out = escalations(
            &json!({
                "escalations": [{
                    "id": "esc_01",
                    "task": "tsk_01",
                    "kind": "external_action",
                    "question": "may I push to origin?",
                    "options": ["allow_once", "allow_for_task", "deny"],
                    "blocking": false
                }]
            }),
            PLAIN,
        );
        assert!(out.contains("esc_01"), "{out}");
        assert!(out.contains("may I push to origin?"), "{out}");
        assert!(out.contains("allow_once, allow_for_task, deny"), "{out}");
        assert!(out.contains("immortality approve"), "{out}");
        assert!(!out.contains("[BLOCKING]"), "{out}");
    }

    #[test]
    fn a_blocking_escalation_is_marked_as_stopping_everything() {
        let out = escalations(
            &json!({
                "escalations": [{
                    "id": "esc_01", "task": "tsk_01", "kind": "journal_corrupt",
                    "question": "the journal is corrupt", "options": ["deny"], "blocking": true
                }]
            }),
            PLAIN,
        );
        assert!(out.contains("[BLOCKING]"), "{out}");
    }

    #[test]
    fn a_tool_intent_opens_a_block_with_the_tool_and_its_intended_effect() {
        let out = rendered(&json!({
            "seq": 7,
            "kind": "tool_call_intent",
            "payload": {
                "call": "call_1",
                "tool": "read_file",
                "idempotency": "idempotent",
                "expected_effect": "check the config before editing it"
            }
        }));
        assert!(out.contains("read_file"), "{out}");
        // The effect is the sentence the model wrote for a human, and the point of showing it.
        assert!(out.contains("check the config"), "{out}");
        assert!(out.contains('7'), "the seq is what --since-seq takes: {out}");
        assert!(!out.contains("unsafe to retry"), "{out}");
    }

    #[test]
    fn an_intent_for_an_unsafe_call_is_flagged_for_the_human() {
        // `Unknown` means "probe before retrying"; a human watching a crash recovery needs it.
        let out = rendered(&json!({
            "seq": 1, "kind": "tool_call_intent",
            "payload": {"call": "c", "tool": "shell", "idempotency": "unknown", "expected_effect": "rm"}
        }));
        assert!(out.contains("unsafe to retry"), "{out}");
    }

    #[test]
    fn the_gutter_always_separates_the_number_from_what_follows() {
        // A missing space made a real screenshot read `     1⏺ model` — the marker looked like
        // part of the number. Every block's first line must be `<6-wide seq><space><marker>`.
        let mut renderer = Renderer::new(PLAIN);
        let lines = [
            renderer.line(&json!({
                "seq": 1, "kind": "tool_call_intent",
                "payload": {"call": "c", "tool": "shell", "idempotency": "idempotent", "expected_effect": ""}
            })),
            renderer.line(&json!({"seq": 2, "kind": "model_response", "payload": {"message": "hi"}})),
            renderer.line(&json!({"seq": 3, "kind": "daemon_started", "payload": {}})),
        ];
        for line in lines {
            // Leading blank line for blocks; drop it before examining the gutter.
            let body = line.trim_start_matches('\n');
            let bytes = body.as_bytes();
            assert!(
                bytes[..6]
                    .iter()
                    .all(|byte| byte.is_ascii_digit() || *byte == b' '),
                "the seq column is digits and spaces: {body:?}"
            );
            assert_eq!(bytes[6], b' ', "seq must not touch the marker: {body:?}");
            assert_ne!(bytes[7], b' ', "the marker follows one space: {body:?}");
        }
    }

    #[test]
    fn a_reconnect_notice_has_no_sequence_number_but_sits_in_the_gutter() {
        // The notice is about the connection, not the task. A number would make `--since-seq`
        // ambiguous, so the gutter carries blank space instead.
        let out = notice(
            crate::client::Notice::Reconnecting { from: 42, attempt: 2 },
            PLAIN,
        );
        assert!(out.contains("seq 42"), "{out}");
        assert!(out.contains("attempt 2"), "{out}");
        // Same column as a block's marker: `      <seq> ⏺` puts the marker at index 7, so the
        // notice's must be there too or the stream looks ragged on every reconnect.
        let marker_column = out.find('⟳').expect("marker");
        assert_eq!(marker_column, 7, "the marker sits in the block column: {out:?}");
        assert!(
            out[..marker_column].trim().is_empty(),
            "no sequence number: {out:?}"
        );
        assert_eq!(out.matches('\n').count(), 1, "{out:?}");
    }

    #[test]
    fn a_reconnect_notice_is_yellow_and_says_the_stream_closed() {
        // The whole point: a frozen screen must not be mistaken for a thinking model.
        let out = notice(
            crate::client::Notice::Reconnecting { from: 0, attempt: 1 },
            Style::colour(),
        );
        assert!(out.contains("\u{1b}[33m"), "{out:?}");
        assert!(out.contains("stream closed"), "{out}");
    }

    #[test]
    fn a_started_record_prints_nothing_at_all() {
        // Two lines per call would double the stream for no information: the intent line already
        // says the call is running.
        let mut renderer = Renderer::new(PLAIN);
        let out = renderer.line(&json!({
            "seq": 2, "kind": "tool_call_started", "payload": {"call": "c", "pid": 1}
        }));
        assert_eq!(out, "");
    }

    #[test]
    fn a_completed_record_is_indented_and_names_its_tool_from_the_intent() {
        let mut renderer = Renderer::new(PLAIN);
        let _ = renderer.line(&json!({
            "seq": 1, "kind": "tool_call_intent",
            "payload": {"call": "c1", "tool": "read_file", "expected_effect": "look"}
        }));
        let out = renderer.line(&json!({
            "seq": 2, "kind": "tool_call_completed",
            "payload": {"call": "c1", "path": "src/main.rs", "returned": 40}
        }));
        assert!(out.starts_with("   "), "the result hangs under its call: {out:?}");
        assert!(out.contains("read_file"), "{out}");
        assert!(out.contains("src/main.rs"), "{out}");
    }

    #[test]
    fn a_result_attached_mid_run_is_rendered_without_a_name_rather_than_a_guess() {
        // `--since-seq` can start between the intent and the completion. Inventing a tool name
        // there would be a lie; the indent is enough to show it belongs to a call.
        let out = rendered(&json!({
            "seq": 5, "kind": "tool_call_completed",
            "payload": {"call": "unknown_to_us", "path": "a.rs"}
        }));
        assert!(out.contains("a.rs"), "{out}");
        assert!(out.starts_with("   "), "{out:?}");
    }

    #[test]
    fn a_failed_call_shows_the_error_verbatim() {
        let mut renderer = Renderer::new(PLAIN);
        let _ = renderer.line(&json!({
            "seq": 1, "kind": "tool_call_intent",
            "payload": {"call": "c1", "tool": "shell", "expected_effect": "run tests"}
        }));
        let out = renderer.line(&json!({
            "seq": 2, "kind": "tool_call_failed",
            "payload": {"call": "c1", "error": "command not found: cargo"}
        }));
        assert!(out.contains("command not found"), "{out}");
        assert!(out.contains("shell"), "{out}");
    }

    #[test]
    fn a_shell_result_leads_with_the_exit_code_and_the_command() {
        let out = rendered(&json!({
            "seq": 9, "kind": "tool_call_completed",
            "payload": {"call": "c", "command": "cargo test", "exit_code": 0, "duration_ms": 1200}
        }));
        assert!(out.contains("cargo test"), "{out}");
        assert!(out.contains("exit 0"), "{out}");
        assert!(!out.contains("duration_ms"), "raw keys are not for humans: {out}");
    }

    #[test]
    fn a_result_with_no_known_fields_says_ok_rather_than_printing_nothing() {
        // An empty result line would read as a tool that did nothing.
        let out = rendered(&json!({
            "seq": 1, "kind": "tool_call_completed", "payload": {"call": "c"}
        }));
        assert!(out.contains("ok"), "{out}");
    }

    #[test]
    fn a_model_response_is_shown_as_prose_not_as_a_log_line() {
        let out = rendered(&json!({
            "seq": 3, "kind": "model_response",
            "payload": {"message": "I will read the config first.\nThen edit it."}
        }));
        assert!(out.contains("I will read the config first."), "{out}");
        assert!(out.contains("Then edit it."), "{out}");
        // Both lines are indented under the heading, so a paragraph stays a paragraph.
        assert!(out.contains("\n  Then edit it."), "{out:?}");
    }

    #[test]
    fn a_model_response_without_text_falls_back_to_one_line() {
        // A future or older build may carry something else; it must not vanish silently.
        let out = rendered(&json!({
            "seq": 3, "kind": "model_response", "payload": {"tokens": 42}
        }));
        assert!(out.contains("model_response"), "{out}");
        assert!(out.contains("42"), "{out}");
    }

    #[test]
    fn an_escalation_in_the_stream_says_how_to_answer_it() {
        let out = rendered(&json!({
            "seq": 11, "kind": "escalation_opened",
            "payload": {
                "id": "esc_01", "kind": "external_action", "blocking": false,
                "options": ["allow_once", "deny"]
            }
        }));
        assert!(out.contains("a human is needed"), "{out}");
        assert!(out.contains("immortality approve esc_01"), "{out}");
        assert!(out.contains("allow_once, deny"), "{out}");
        assert!(!out.contains("[BLOCKING]"), "{out}");
    }

    #[test]
    fn a_blocking_escalation_in_the_stream_is_marked() {
        let out = rendered(&json!({
            "seq": 11, "kind": "escalation_opened",
            "payload": {"id": "e", "blocking": true, "options": []}
        }));
        assert!(out.contains("[BLOCKING]"), "{out}");
        // With no options the hint must still be usable.
        assert!(out.contains("immortality approve e <answer>"), "{out}");
    }

    #[test]
    fn a_resolved_escalation_shows_the_answer() {
        let out = rendered(&json!({
            "seq": 12, "kind": "escalation_resolved",
            "payload": {"id": "esc_01", "answer": "allow_once"}
        }));
        assert!(out.contains("allow_once"), "{out}");
    }

    #[test]
    fn an_unknown_kind_is_still_one_readable_line_and_is_not_a_panic() {
        let out = rendered(&json!({"seq": 1, "kind": "daemon_started", "payload": {}}));
        assert_eq!(out.matches('\n').count(), 1, "{out:?}");
        assert!(out.contains("daemon_started"), "{out}");
        assert!(!out.contains("{}"), "{out}");
    }

    #[test]
    fn a_recovery_event_is_coloured_as_a_warning_and_a_dead_backend_as_an_error() {
        let style = Style::colour();
        let mut renderer = Renderer::new(style);
        let recovered = renderer.line(&json!({"seq": 1, "kind": "daemon_recovered", "payload": {}}));
        let backend = renderer.line(&json!({"seq": 2, "kind": "backend_unavailable", "payload": {}}));
        assert!(
            recovered.contains("\u{1b}[33m"),
            "warning is yellow: {recovered:?}"
        );
        assert!(backend.contains("\u{1b}[31m"), "error is red: {backend:?}");
    }

    #[test]
    fn a_multi_line_observation_is_cut_to_its_first_line() {
        // A tool result is one line in the stream; the second line would break the block shape.
        let out = rendered(&json!({
            "seq": 1, "kind": "tool_call_completed",
            "payload": {"call": "c", "observation": "line one\nline two"}
        }));
        assert!(out.contains("line one"), "{out}");
        assert!(!out.contains("line two"), "{out}");
    }

    #[test]
    fn a_long_summary_is_cut_on_a_character_boundary() {
        // These strings are Russian as often as English; a byte slice would panic mid-sequence.
        let long = "я".repeat(200);
        let out = rendered(&json!({
            "seq": 1, "kind": "model_response", "payload": {"message": long}
        }));
        assert!(std::str::from_utf8(out.as_bytes()).is_ok());
        assert!(out.contains('я'), "{out}");
    }

    #[test]
    fn truncate_keeps_short_text_intact_and_cuts_long_text() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("exactly-10", 10), "exactly-10");
        assert_eq!(truncate("abcdefghijk", 5), "abcde…");
        // Never split a multi-byte character.
        assert_eq!(truncate("яяя", 2), "я…");
    }

    #[test]
    fn a_missing_field_renders_a_placeholder_rather_than_panicking() {
        // A daemon from a slightly older build may not send every field; the CLI must print
        // something a human can act on rather than crash on a `None`.
        let out = status(&json!({"projects": [{}], "tasks": [{"journal": {}}]}), PLAIN);
        assert!(out.contains('?'), "{out}");
    }

    #[test]
    fn an_idle_daemon_says_why_it_is_idle() {
        // A table with no explanation is what makes a human assume the daemon died. The four
        // reasons have four different fixes, so the line has to name the one that applies.
        let out = status(
            &json!({
                "projects": [{}],
                "tasks": [{"journal": {}}],
                "scheduling": {"next": null, "reason": "2 waiting for the model backend"},
            }),
            PLAIN,
        );
        assert!(out.contains("idle:"), "{out}");
        assert!(out.contains("2 waiting for the model backend"), "{out}");
    }

    #[test]
    fn a_busy_daemon_does_not_print_an_idle_line() {
        // The other half: a next task present means something is running, and an `idle:` line
        // beside it would be a lie the human has to resolve by reading the table.
        let out = status(
            &json!({
                "projects": [{}],
                "tasks": [{"journal": {}}],
                "scheduling": {"next": "tsk_01", "reason": ""},
            }),
            PLAIN,
        );
        assert!(!out.contains("idle:"), "{out}");
    }

    #[test]
    fn an_unavailable_scheduling_answer_is_said_out_loud() {
        // "Could not compute" is not "nothing to run": reporting the second when the first is
        // true would hide a broken registry behind a plausible idle daemon.
        let out = status(
            &json!({
                "projects": [{}],
                "tasks": [{"journal": {}}],
                "scheduling": {"error": "cannot read registry.json"},
            }),
            PLAIN,
        );
        assert!(out.contains("scheduling unavailable"), "{out}");
        assert!(out.contains("cannot read registry.json"), "{out}");
    }

    #[test]
    fn the_approved_confirmation_names_the_question_and_the_answer() {
        let out = approved(&json!({"id": "esc_01", "answer": "allow_once"}), PLAIN);
        assert!(out.contains("esc_01"), "{out}");
        assert!(out.contains("allow_once"), "{out}");
    }

    #[test]
    fn a_plain_renderer_emits_no_escape_bytes_anywhere() {
        // The guarantee that makes `attach > run.log` safe: the same records, byte for byte,
        // with no styling layer at all.
        let mut renderer = Renderer::new(PLAIN);
        let records = [
            json!({"seq": 1, "kind": "tool_call_intent", "payload": {"call": "c", "tool": "shell", "idempotency": "unknown", "expected_effect": "x"}}),
            json!({"seq": 2, "kind": "tool_call_completed", "payload": {"call": "c", "exit_code": 1}}),
            json!({"seq": 3, "kind": "tool_call_failed", "payload": {"call": "d", "error": "boom"}}),
            json!({"seq": 4, "kind": "model_response", "payload": {"message": "hi"}}),
            json!({"seq": 5, "kind": "escalation_opened", "payload": {"id": "e", "blocking": true, "options": []}}),
            json!({"seq": 6, "kind": "daemon_recovered", "payload": {}}),
            json!({"seq": 7, "kind": "backend_unavailable", "payload": {}}),
        ];
        for record in records {
            let out = renderer.line(&record);
            assert!(!out.contains('\u{1b}'), "{record} → {out:?}");
        }
    }

    #[test]
    fn a_coloured_renderer_wraps_every_block_it_draws() {
        let mut renderer = Renderer::new(Style::colour());
        let out = renderer.line(&json!({
            "seq": 1, "kind": "tool_call_intent",
            "payload": {"call": "c", "tool": "read_file", "expected_effect": "look"}
        }));
        assert!(out.contains("\u{1b}[32m"), "the call marker is green: {out:?}");
        assert!(out.contains("\u{1b}[1m"), "the tool name is bold: {out:?}");
        // Every opening sequence has its reset exactly once per painted span, so a terminal is
        // never left in a borrowed style after the stream ends.
        assert_eq!(
            out.matches("\u{1b}[0m").count(),
            out.matches('\u{1b}').count() / 2,
            "{out:?}"
        );
        assert!(out.ends_with("\u{1b}[0m\n"), "{out:?}");
    }

    #[test]
    fn one_renderer_keeps_track_of_several_open_calls() {
        // The daemon may have several calls whose intents were seen before the results arrive;
        // a single "last tool" field would mislabel them.
        let mut renderer = Renderer::new(PLAIN);
        for (call, tool) in [("a", "read_file"), ("b", "shell")] {
            let _ = renderer.line(&json!({
                "seq": 1, "kind": "tool_call_intent",
                "payload": {"call": call, "tool": tool, "expected_effect": ""}
            }));
        }
        let first = renderer.line(&json!({
            "seq": 2, "kind": "tool_call_completed", "payload": {"call": "a", "matches": 3}
        }));
        let second = renderer.line(&json!({
            "seq": 3, "kind": "tool_call_completed", "payload": {"call": "b", "exit_code": 0}
        }));
        assert!(first.contains("read_file"), "{first}");
        assert!(second.contains("shell"), "{second}");
    }

    #[test]
    fn a_readable_plan_lists_each_subtask_with_its_state() {
        let out = plan(
            &json!({
                "project": "repo-abc",
                "root": "/tmp/repo",
                "max_subtask_depth": 2,
                "count": 2,
                "subtasks": [
                    {"id": "sub_01", "state": "done", "title": "write the parser", "children": []},
                    {"id": "sub_02", "state": "pending", "title": "add tests", "children": []},
                ],
            }),
            PLAIN,
        );
        assert!(out.contains("2 subtask(s)"), "{out}");
        assert!(out.contains("depth limit 2"), "{out}");
        assert!(out.contains("write the parser"), "{out}");
        assert!(out.contains("add tests"), "{out}");
    }

    #[test]
    fn children_are_indented_under_their_parent() {
        // The whole point of a tree view: which subtask belongs to which. A flat list would make
        // `plan` silently the same command as `grep title TASKS.json`.
        let out = plan(
            &json!({
                "project": "repo-abc", "root": "/tmp/repo", "max_subtask_depth": 2,
                "subtasks": [{
                    "id": "sub_01", "state": "running", "title": "parent",
                    "children": [{"id": "sub_02", "state": "pending", "title": "child", "children": []}],
                }],
            }),
            PLAIN,
        );
        let parent = out.lines().find(|line| line.contains("parent")).unwrap();
        let child = out.lines().find(|line| line.contains("child")).unwrap();
        let indent = |line: &str| line.len() - line.trim_start().len();
        assert!(
            indent(child) > indent(parent),
            "child indented deeper than parent:\n{out}"
        );
    }

    #[test]
    fn an_unreadable_plan_names_the_files_to_fix() {
        // The message says what is wrong; the paths say where. A human without the paths has to
        // remember the layout, which is exactly the knowledge they ran the command to avoid.
        let out = plan(
            &json!({
                "project": "repo-abc", "root": "/tmp/repo",
                "load_error": "TASKS.json is not valid JSON",
            }),
            PLAIN,
        );
        assert!(out.contains("cannot read the plan"), "{out}");
        assert!(out.contains("TASKS.json is not valid JSON"), "{out}");
        assert!(out.contains("/tmp/repo/.agent/TASKS.json"), "{out}");
        assert!(out.contains("/tmp/repo/.agent/ACCEPTANCE.md"), "{out}");
    }

    #[test]
    fn an_empty_task_log_says_the_task_has_not_started() {
        let out = log(&json!({"task": "tsk_01", "records": [], "next_seq": 0}), PLAIN);
        assert!(out.contains("tsk_01"), "{out}");
        assert!(out.contains("has not started"), "{out}");
    }

    #[test]
    fn a_log_renders_records_the_same_way_attach_does() {
        // One renderer for both, or a human re-learns how to read their own transcript.
        let record = json!({
            "seq": 7, "kind": "tool_call_intent",
            "payload": {"call": "c", "tool": "read_file", "expected_effect": "look"}
        });
        let out = log(
            &json!({"task": "tsk_01", "records": [&record], "next_seq": 8}),
            PLAIN,
        );
        assert!(out.contains(&rendered(&record)), "{out}");
        assert!(out.contains("read_file"), "{out}");
        assert!(out.contains("next seq 8"), "{out}");
    }

    #[test]
    fn a_corrupt_journal_is_announced_before_its_records() {
        // "These records are what survived" is a different reading from "here is the history",
        // and only the announcement distinguishes them.
        let out = log(
            &json!({"task": "tsk_01", "corrupt": true, "records": [], "next_seq": 0}),
            PLAIN,
        );
        assert!(out.contains("chain check"), "{out}");
        // And it must not claim the task never ran: that is a different fact, and it sends a human
        // looking for work that is already on disk.
        assert!(!out.contains("has not started"), "{out}");
        assert!(out.contains("survived the damage"), "{out}");
    }

    #[test]
    fn a_dry_run_gc_says_nothing_was_deleted() {
        let out = gc_report(
            &json!({"dry_run": true, "protected": 3, "unfinished": ["tsk_01"]}),
            PLAIN,
        );
        assert!(out.contains("nothing was deleted"), "{out}");
        assert!(out.contains('3'), "{out}");
        assert!(out.contains("tsk_01"), "{out}");
        assert!(
            !out.contains("reclaimed"),
            "a dry run must not report collecting: {out}"
        );
    }

    #[test]
    fn a_real_gc_reports_what_it_reclaimed() {
        let out = gc_report(
            &json!({
                "dry_run": false, "protected": 0, "unfinished": [],
                "artifacts_removed": 5, "bytes_reclaimed": 2_097_152, "temp_removed": 1,
                "noop": false,
            }),
            PLAIN,
        );
        assert!(out.contains('5'), "{out}");
        assert!(out.contains("2.0 MiB"), "bytes are human-readable: {out}");
        assert!(!out.contains("nothing was deleted"), "{out}");
    }

    #[test]
    fn a_noop_gc_says_so_rather_than_printing_zeros() {
        let out = gc_report(
            &json!({
                "dry_run": false, "protected": 0, "unfinished": [],
                "artifacts_removed": 0, "bytes_reclaimed": 0, "temp_removed": 0, "noop": true,
            }),
            PLAIN,
        );
        assert!(out.contains("nothing needed collecting"), "{out}");
    }

    #[test]
    fn metrics_show_the_counts_that_decide_the_milestone() {
        let out = metrics(
            &json!({"count": 1, "tasks": [{
                "task": "tsk_01A",
                "closed_green": 3,
                "closed_with_escalation": 1,
                "longest_green_streak": 2,
                "subtasks": [
                    {"subtask": "sub_A", "state": "done", "closed_with_escalation": false},
                    {"subtask": "sub_B", "state": "done", "closed_with_escalation": true},
                    {"subtask": "sub_C", "state": "blocked", "closed_with_escalation": false},
                ],
                "detectors": {"d1/l1": 2, "d5/l3": 1},
                "escalations_total": 1,
                "escalations_pending": 0,
                "restarts": 2,
                "records": 40,
                "corrupt": false,
            }]}),
            PLAIN,
        );
        assert!(out.contains("tsk_01A"), "{out}");
        assert!(out.contains("3 closed by a green gate"), "{out}");
        assert!(out.contains("1 of them with an escalation open"), "{out}");
        assert!(out.contains("longest streak 2 in a row"), "{out}");
        assert!(out.contains("sub_A"), "{out}");
        assert!(out.contains("(escalation open)"), "{out}");
        assert!(out.contains("blocked"), "{out}");
        assert!(out.contains("restarts 2"), "{out}");
        assert!(out.contains("d1/l1"), "{out}");
        assert!(out.contains("d5/l3"), "{out}");
    }

    #[test]
    fn a_met_bar_is_painted_differently_from_a_missed_one() {
        // A reader has to be able to tell "three in a row" from "two in a row" without comparing
        // the number against the document. The bar is the whole point of the command; leaving it
        // as an unadorned count makes the reader do arithmetic the tool already knows how to do.
        let report = |streak: u64| {
            json!({"count": 1, "tasks": [{
                "task": "tsk_01A", "closed_green": streak, "closed_with_escalation": 0,
                "longest_green_streak": streak, "subtasks": [], "detectors": {},
                "escalations_total": 0, "escalations_pending": 0, "restarts": 0,
                "records": 5, "corrupt": false,
            }]})
        };
        let met = metrics(&report(3), Style::colour());
        let missed = metrics(&report(2), Style::colour());
        // `3` is the bar and gets the green SGR open; `2` is dim. The assertion is about the
        // streak's own line, because a report also paints the green-closure count green and a
        // whole-output `contains` would pass for a missed bar too.
        let streak_line = |text: &str| {
            text.lines()
                .find(|line| line.contains("longest streak"))
                .expect("every report has a streak line")
                .to_string()
        };
        assert!(
            streak_line(&met).contains("\x1b[32m"),
            "the met bar is not green: {:?}",
            streak_line(&met)
        );
        assert!(
            !streak_line(&missed).contains("\x1b[32m"),
            "a missed bar was painted as met: {:?}",
            streak_line(&missed)
        );
        // And the plain renderer still says both numbers, which is what a pipe sees.
        assert!(
            metrics(&report(3), PLAIN).contains("longest streak 3 in a row"),
            "the number must survive the plain rendering"
        );
        assert!(
            metrics(&report(2), PLAIN).contains("longest streak 2 in a row"),
            "the number must survive the plain rendering"
        );
    }

    #[test]
    fn a_report_from_an_older_daemon_prints_zero_instead_of_a_missing_line() {
        // The field is new; a daemon started before it existed answers without it. Silence would
        // read as "not measured", and this number is measured — as zero, because no settlement
        // was recorded.
        let out = metrics(
            &json!({"count": 1, "tasks": [{
                "task": "tsk_01A", "closed_green": 1, "closed_with_escalation": 0,
                "subtasks": [], "detectors": {}, "escalations_total": 0,
                "escalations_pending": 0, "restarts": 0, "records": 5, "corrupt": false,
            }]}),
            PLAIN,
        );
        assert!(out.contains("longest streak 0 in a row"), "{out}");
    }

    #[test]
    fn a_clean_run_says_no_detector_fired_instead_of_nothing() {
        // Silence and "zero" are different claims. A reader checking the detector claim needs to
        // see that the run had none, not an absent line they must interpret.
        let out = metrics(
            &json!({"count": 1, "tasks": [{
                "task": "tsk_01A", "closed_green": 1, "closed_with_escalation": 0,
                "subtasks": [], "detectors": {}, "escalations_total": 0,
                "escalations_pending": 0, "restarts": 0, "records": 5, "corrupt": false,
            }]}),
            PLAIN,
        );
        assert!(out.contains("no detector fired"), "{out}");
    }

    #[test]
    fn an_unreadable_journal_is_reported_as_unreadable_not_as_zeros() {
        // Zero closed subtasks is evidence of a run that did nothing; an unreadable journal is
        // evidence of nothing. Printing the first for the second is the one lie metrics must not
        // tell.
        let out = metrics(
            &json!({"count": 1, "tasks": [{
                "task": "tsk_01A",
                "unreadable": "the chain breaks at seq 41",
            }]}),
            PLAIN,
        );
        assert!(out.contains("unreadable"), "{out}");
        assert!(out.contains("chain breaks"), "{out}");
        assert!(!out.contains("closed by a green gate"), "{out}");
    }

    #[test]
    fn metrics_with_no_tasks_says_so() {
        let out = metrics(&json!({"count": 0, "tasks": []}), PLAIN);
        assert!(out.contains("no tasks yet"), "{out}");
    }

    #[test]
    fn a_corrupt_chain_is_flagged_next_to_the_numbers_it_weakens() {
        let out = metrics(
            &json!({"count": 1, "tasks": [{
                "task": "tsk_01A", "closed_green": 1, "closed_with_escalation": 0,
                "subtasks": [], "detectors": {}, "escalations_total": 0,
                "escalations_pending": 0, "restarts": 0, "records": 5, "corrupt": true,
            }]}),
            PLAIN,
        );
        assert!(out.contains("fails its own chain check"), "{out}");
        assert!(out.contains("incomplete"), "{out}");
    }

    #[test]
    fn human_bytes_never_prints_a_bare_float() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(999), "999 B");
        assert_eq!(human_bytes(1024), "1.0 KiB");
        assert_eq!(human_bytes(1024 * 1024), "1.0 MiB");
        assert!(human_bytes(u64::MAX).contains("GiB"));
    }

    #[test]
    fn a_merge_confirmation_names_the_branch_and_the_state() {
        let out = merged(
            &json!({"branch": "agent/tsk_01", "path": "/tmp/repo", "state": "merged"}),
            PLAIN,
        );
        assert!(out.contains("agent/tsk_01"), "{out}");
        assert!(out.contains("/tmp/repo"), "{out}");
        assert!(out.contains("merged"), "{out}");
    }

    #[test]
    fn the_new_reports_emit_no_escape_bytes_in_a_pipe() {
        // The same guarantee as the stream, extended to the commands added beside it: redirecting
        // any of them to a file must not write terminal control codes into that file.
        let outputs = [
            plan(
                &json!({
                    "project": "p", "root": "/r", "max_subtask_depth": 1,
                    "subtasks": [{"id": "s", "state": "done", "title": "t", "children": []}],
                }),
                PLAIN,
            ),
            plan(&json!({"project": "p", "root": "/r", "load_error": "bad"}), PLAIN),
            log(
                &json!({"task": "t", "records": [{"seq": 1, "kind": "task_state", "payload": {"state": "running"}}]}),
                PLAIN,
            ),
            gc_report(
                &json!({"dry_run": true, "protected": 1, "unfinished": ["t"]}),
                PLAIN,
            ),
            gc_report(
                &json!({"dry_run": false, "protected": 0, "unfinished": [], "artifacts_removed": 1,
                        "bytes_reclaimed": 2048, "temp_removed": 0, "noop": false}),
                PLAIN,
            ),
            merged(&json!({"branch": "b", "path": "/p", "state": "merged"}), PLAIN),
        ];
        for out in outputs {
            assert_ne!(out, String::new(), "every report prints something");
            assert!(!out.contains('\u{1b}'), "{out:?}");
        }
    }

    #[test]
    fn a_freshly_written_plan_names_the_goal_and_the_files_it_touched() {
        // The roadmap's contract: a human may edit these three files before the run. A
        // confirmation that did not say where they landed would leave them hunting for them.
        let out = planned(
            &json!({
                "project": "p", "root": "/repo", "max_subtask_depth": 2,
                "goal": "add a parser",
                "created": true,
                "written": ["PLAN.md", "ACCEPTANCE.md", "TASKS.json"],
                "subtasks": [{"id": "sub_01", "state": "pending", "title": "write it", "children": []}],
            }),
            PLAIN,
        );
        assert!(out.contains("add a parser"), "{out}");
        for name in ["PLAN.md", "ACCEPTANCE.md", "TASKS.json"] {
            assert!(
                out.contains(&format!("/repo/.agent/{name}")),
                "{name} missing: {out}"
            );
        }
        // The tree comes through the same renderer as `immortality plan`.
        assert!(out.contains("sub_01"), "{out}");
        assert!(out.contains("immortality start"), "{out}");
    }

    #[test]
    fn a_freshly_written_plan_emits_no_escape_bytes_in_a_pipe() {
        let out = planned(
            &json!({
                "project": "p", "root": "/repo", "max_subtask_depth": 2, "goal": "g",
                "created": true, "written": ["PLAN.md"],
                "subtasks": [{"id": "sub_01", "state": "pending", "title": "t", "children": []}],
            }),
            PLAIN,
        );
        assert!(!out.contains('\u{1b}'), "{out:?}");
    }

    #[test]
    fn describe_result_prefers_explicit_observation_text() {
        // A tool that hands over prose says more than any structured field can.
        let detail = describe_result(&json!({
            "observation": "Created src/new.rs (12 line(s)).",
            "path": "src/new.rs",
            "created": true
        }));
        assert_eq!(detail, "Created src/new.rs (12 line(s)).");
    }
}
