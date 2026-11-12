//! Rendering RPC results for a terminal.
//!
//! Kept apart from the commands so that "what the human sees" can be tested without a daemon,
//! a socket or a process. Every function here takes the JSON the daemon answered with and
//! returns text; none of them read the filesystem or the clock.

// `pub`, not `pub(crate)`: this is a binary crate, so the module is only reachable from the
// crate root and `pub(crate)` on its items is what `clippy::redundant_pub_crate` objects to.
// `unreachable_pub` then objects to `pub`. The pair is unsatisfiable for a private module in a
// binary, so the reachability lint is allowed here and only here.
#![allow(unreachable_pub)]

use serde_json::Value;

/// A status report: the projects, then a line per task.
///
/// Deliberately terse. This is what a human glances at to answer "is it still working?", and a
/// screen of JSON does not answer that. The `--json` flag exists for everything else.
#[must_use]
pub fn status(result: &Value) -> String {
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

    let _ = writeln!(out, "projects");
    for project in projects {
        let path = text(project, "path");
        match project.get("remote").and_then(Value::as_str) {
            Some(remote) => {
                let _ = writeln!(out, "  {}  {path}  ({remote})", text(project, "id"));
            }
            None => {
                let _ = writeln!(out, "  {}  {path}", text(project, "id"));
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

    let _ = writeln!(out, "tasks");
    for task in tasks {
        let journal = task.get("journal").cloned().unwrap_or(Value::Null);
        if let Some(error) = journal.get("error").and_then(Value::as_str) {
            let _ = writeln!(out, "  {}  journal unreadable: {error}", text(task, "task"));
            continue;
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
            text(task, "task")
        );
        if escalations > 0 {
            // First, because it is the one thing that may need the human.
            let _ = write!(out, "  [{escalations} awaiting approval]");
        }
        if corrupt {
            let _ = write!(out, "  [JOURNAL CORRUPT]");
        }
        out.push('\n');
    }
    out
}

/// The open questions, as a numbered list a human can act on.
#[must_use]
pub fn escalations(result: &Value) -> String {
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
    let _ = writeln!(out, "{} question(s) waiting:", entries.len());
    for entry in entries {
        let blocking = if entry.get("blocking").and_then(Value::as_bool) == Some(true) {
            "  [BLOCKING]"
        } else {
            ""
        };
        let _ = writeln!(out, "\n  {}{blocking}", text(entry, "id"));
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
pub fn approved(result: &Value) -> String {
    format!("answered {}: {}\n", text(result, "id"), text(result, "answer"))
}

/// One journal record, as one line.
///
/// `attach` renders every record this way. The shape is chosen to be scannable during a long
/// run: sequence, kind, then whatever the payload's most useful field is. It is not a
/// replacement for `--json`, which carries the record verbatim.
#[must_use]
pub fn record(record: &Value) -> String {
    use std::fmt::Write as _;

    let seq = record.get("seq").and_then(Value::as_u64).unwrap_or(0);
    let kind = record.get("kind").and_then(Value::as_str).unwrap_or("unknown");
    let mut out = String::new();
    let _ = write!(out, "{seq:>6}  {kind}");

    if let Some(summary) = summarise(record) {
        let _ = write!(out, "  {summary}");
    }
    out.push('\n');
    out
}

/// The most useful single field of a record's payload, for the one-line view.
///
/// Ordered by what a human watching a run needs first: what the model said, what tool ran, how
/// it ended. Unknown kinds fall back to nothing rather than a dump — `attach --json` is the
/// complete view.
fn summarise(record: &Value) -> Option<String> {
    let payload = record.get("payload")?;
    for key in ["tool", "call", "state", "answer", "question", "message", "error"] {
        if let Some(value) = payload.get(key) {
            let rendered = match value {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            if !rendered.is_empty() {
                return Some(truncate(&rendered, 160));
            }
        }
    }
    None
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

    #[test]
    fn an_empty_registry_says_what_to_do_next() {
        let rendered = status(&json!({"projects": [], "tasks": []}));
        assert!(rendered.contains("no projects"), "{rendered}");
        assert!(rendered.contains("immortality init"), "{rendered}");
    }

    #[test]
    fn a_registered_project_with_no_tasks_is_not_an_error() {
        let rendered = status(&json!({
            "projects": [{"id": "immortality-9a1e0c", "path": "/repo"}],
            "tasks": []
        }));
        assert!(rendered.contains("immortality-9a1e0c"), "{rendered}");
        assert!(rendered.contains("/repo"), "{rendered}");
        assert!(rendered.contains("no tasks yet"), "{rendered}");
    }

    #[test]
    fn a_task_line_carries_its_record_count_and_last_kind() {
        let rendered = status(&json!({
            "projects": [{"id": "p", "path": "/repo"}],
            "tasks": [{
                "task": "tsk_01",
                "journal": {"records": 12, "last": {"kind": "tool_call_completed"}, "corrupt": false},
                "open_escalations": 0
            }]
        }));
        assert!(rendered.contains("tsk_01"), "{rendered}");
        assert!(rendered.contains("12 record"), "{rendered}");
        assert!(rendered.contains("tool_call_completed"), "{rendered}");
        assert!(!rendered.contains("awaiting approval"), "{rendered}");
    }

    #[test]
    fn pending_approvals_and_corruption_are_called_out_first() {
        // Both are the reason a human opened this command at all.
        let rendered = status(&json!({
            "projects": [{"id": "p", "path": "/repo"}],
            "tasks": [{
                "task": "tsk_01",
                "journal": {"records": 3, "last": {"kind": "task_state"}, "corrupt": true},
                "open_escalations": 2
            }]
        }));
        assert!(rendered.contains("2 awaiting approval"), "{rendered}");
        assert!(rendered.contains("JOURNAL CORRUPT"), "{rendered}");
    }

    #[test]
    fn a_task_with_no_records_says_so_rather_than_printing_nothing() {
        let rendered = status(&json!({
            "projects": [{"id": "p", "path": "/repo"}],
            "tasks": [{"task": "tsk_01", "journal": {"records": 0}, "open_escalations": 0}]
        }));
        assert!(rendered.contains("0 record"), "{rendered}");
        assert!(rendered.contains("no records"), "{rendered}");
    }

    #[test]
    fn an_unreadable_journal_is_reported_rather_than_hidden() {
        let rendered = status(&json!({
            "projects": [{"id": "p", "path": "/repo"}],
            "tasks": [{"task": "tsk_01", "journal": {"error": "chain broken"}, "open_escalations": 0}]
        }));
        assert!(rendered.contains("chain broken"), "{rendered}");
    }

    #[test]
    fn no_escalations_says_so_plainly() {
        let rendered = escalations(&json!({"escalations": []}));
        assert!(rendered.contains("nothing is waiting"), "{rendered}");
    }

    #[test]
    fn an_escalation_shows_the_question_and_its_options() {
        let rendered = escalations(&json!({
            "escalations": [{
                "id": "esc_01",
                "task": "tsk_01",
                "kind": "external_action",
                "question": "may I push to origin?",
                "options": ["allow_once", "allow_for_task", "deny"],
                "blocking": false
            }]
        }));
        assert!(rendered.contains("esc_01"), "{rendered}");
        assert!(rendered.contains("may I push to origin?"), "{rendered}");
        assert!(
            rendered.contains("allow_once, allow_for_task, deny"),
            "{rendered}"
        );
        assert!(rendered.contains("immortality approve"), "{rendered}");
        assert!(!rendered.contains("[BLOCKING]"), "{rendered}");
    }

    #[test]
    fn a_blocking_escalation_is_marked_as_stopping_everything() {
        let rendered = escalations(&json!({
            "escalations": [{
                "id": "esc_01", "task": "tsk_01", "kind": "journal_corrupt",
                "question": "the journal is corrupt", "options": ["deny"], "blocking": true
            }]
        }));
        assert!(rendered.contains("[BLOCKING]"), "{rendered}");
    }

    #[test]
    fn a_record_line_is_scannable_and_carries_the_tool_name() {
        let rendered = record(&json!({
            "seq": 42,
            "kind": "tool_call_completed",
            "payload": {"tool": "read_file", "call": "call_1"}
        }));
        assert!(rendered.contains("42"), "{rendered}");
        assert!(rendered.contains("tool_call_completed"), "{rendered}");
        // The tool is more informative than the call id, so it is the one shown.
        assert!(rendered.contains("read_file"), "{rendered}");
        assert!(!rendered.contains("call_1"), "{rendered}");
        assert!(rendered.ends_with('\n'));
    }

    #[test]
    fn a_record_with_nothing_summarisable_is_still_one_clean_line() {
        let rendered = record(&json!({"seq": 1, "kind": "daemon_started", "payload": {}}));
        assert_eq!(rendered.matches('\n').count(), 1, "{rendered:?}");
        assert!(rendered.contains("daemon_started"), "{rendered}");
        assert!(!rendered.contains("{}"), "{rendered}");
    }

    #[test]
    fn a_long_summary_is_cut_on_a_character_boundary() {
        // These strings are Russian as often as English; a byte slice would panic mid-sequence.
        let long = "я".repeat(200);
        let rendered = record(&json!({
            "seq": 1, "kind": "model_response", "payload": {"message": long}
        }));
        assert!(rendered.contains('…'), "the line must be cut");
        assert!(std::str::from_utf8(rendered.as_bytes()).is_ok());
    }

    #[test]
    fn truncate_keeps_short_text_intact_and_cuts_long_text() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("exactly-10", 10), "exactly-10");
        let cut = truncate("abcdefghijk", 5);
        assert_eq!(cut, "abcde…");
    }

    #[test]
    fn a_missing_field_renders_a_placeholder_rather_than_panicking() {
        // A daemon from a slightly older build may not send every field; the CLI must print
        // something a human can act on rather than crash on a `None`.
        let rendered = status(&json!({"projects": [{}], "tasks": [{"journal": {}}]}));
        assert!(rendered.contains('?'), "{rendered}");
    }

    #[test]
    fn the_approved_confirmation_names_the_question_and_the_answer() {
        let rendered = approved(&json!({"id": "esc_01", "answer": "allow_once"}));
        assert!(rendered.contains("esc_01"), "{rendered}");
        assert!(rendered.contains("allow_once"), "{rendered}");
    }
}
