//! `escalate` — ask the human a question without stopping the run.
//!
//! # What the tool does and does not do
//!
//! It writes a question to the task's escalation queue and returns "waiting for an answer" to
//! the model. It does **not** wait. Every other design decision here follows from that: the model
//! is expected to continue on work that does not need the answer, and the answer arrives later as
//! an observation when the daemon next looks at the queue.
//!
//! # Why the tool is `Effect::External` but not itself escalated
//!
//! `Effect::External` is the classification for "this leaves the machine or is irreversible",
//! and by the gate's rule (`PolicyGate::check`) that is normally a refusal that becomes an
//! escalation. This tool is the exception by construction: it *is* the escalation mechanism, so
//! a policy check on it would be a queue entry asking permission to create a queue entry. It
//! performs no external effect at all — the file it writes is under the task's state directory.
//!
//! # Why the key and not the id decides deduplication
//!
//! The model may ask the same question on every step of a loop. The `key` is the caller's
//! assertion that it is the same question, so the store answers a repeat from memory: an already
//! answered question returns its answer immediately (the model can act now), and a pending one
//! returns "still waiting". Only a genuinely new key writes to disk and notifies the human.
//!
//! # Why the key is validated
//!
//! A key that is empty or unnormalised makes deduplication useless in exactly the case it exists
//! for: `external:push` and `external:push ` are different strings and the human is asked twice.
//! The key is trimmed and length-capped, and an empty one is refused with an explanation rather
//! than silently replaced by the question text (which changes every time the model paraphrases).

use std::sync::Arc;

use immortality_core::Result as CoreResult;
use immortality_core::domain::Timestamp;
use immortality_core::escalation::{Answer, Entry, EscalationKind, EscalationStore, Opened, group_open};
use immortality_core::ids::TaskId;
use immortality_core::time;
use serde_json::{Value, json};

use crate::Effect;
use crate::args::Args;
use crate::notify::Notifier;
use crate::policy::PathUse;
use crate::tool::{Tool, ToolCtx, ToolError, ToolResult};
use immortality_core::protocol::Idempotency;

/// Longest key accepted. Long enough for a path or a URL, short enough to keep a log line whole.
pub const MAX_KEY_LEN: usize = 200;

/// Longest question accepted, in bytes.
pub const MAX_QUESTION_LEN: usize = 4_000;

/// The store behind an `escalate` call, plus the hooks that record a change.
///
/// Two responsibilities in one trait because they always travel together: the file is the working
/// set, and the journal is the audit trail. A queue implementation that journals when the file
/// changes gives the daemon both without the tool knowing what a journal is.
pub trait EscalationQueue: Send + Sync + std::fmt::Debug {
    /// Load the queue for `task`.
    fn load(&self, task: &TaskId) -> CoreResult<EscalationStore>;

    /// A new question is durably on disk. Append `escalation_opened`.
    fn opened(&self, entry: &Entry) -> CoreResult<()>;

    /// A question has been answered and the answer is on disk. Append `escalation_resolved`.
    fn resolved(&self, entry: &Entry) -> CoreResult<()>;
}

/// A queue backed by a directory on disk: `<root>/<task>/escalations.json`.
///
/// The file is written by [`EscalationStore`] itself, so the hooks here are the journal's. This
/// implementation has no journal, which is what a test or a read-only tool run wants: the file
/// still records everything, and nothing else is appended anywhere.
#[derive(Debug, Clone)]
pub struct DirectoryQueue {
    root: std::path::PathBuf,
}

impl DirectoryQueue {
    /// A queue whose files live under `root`, one directory per task.
    #[must_use]
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The directory a task's queue lives in.
    #[must_use]
    pub fn task_dir(&self, task: &TaskId) -> std::path::PathBuf {
        self.root.join(task.as_str())
    }
}

impl EscalationQueue for DirectoryQueue {
    fn load(&self, task: &TaskId) -> CoreResult<EscalationStore> {
        EscalationStore::open(&self.task_dir(task), task.clone())
    }

    fn opened(&self, _entry: &Entry) -> CoreResult<()> {
        Ok(())
    }

    fn resolved(&self, _entry: &Entry) -> CoreResult<()> {
        Ok(())
    }
}

/// A queue that also hands the two escalation events to whoever holds the journal.
///
/// # Why this does not write the journal itself
///
/// It used to. The daemon holds the task journal's writer for the whole run, so a queue that
/// opened a second [`immortality_core::journal::Journal`] on the same directory to append one
/// record produced `seq=1` twice and a broken digest chain — the reader then calls the journal
/// corrupt. Measured, not assumed: the probe that showed the duplicate is recorded in
/// `docs/decisions.md` (D-117).
///
/// So the two hooks *collect* and the journal's owner drains them, which is the arrangement the
/// judge already uses ([`crate::judge::Recorded::take_fresh`]). The difference is that a verdict
/// may be dropped — the next restart pays for the judgement again — while an escalation may not:
/// the question is already on disk in the queue file, and a lost `escalation_opened` would make
/// `escalations_total` under-report exactly the events this milestone is about.
#[derive(Debug, Clone)]
pub struct JournalQueue {
    directory: DirectoryQueue,
    pending: Arc<std::sync::Mutex<Vec<Pending>>>,
}

/// One journal record an escalation produced, waiting for the journal's owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pending {
    /// Which of the two records this is.
    pub kind: immortality_core::journal::Kind,
    /// The task the question belongs to.
    pub task: TaskId,
    /// The payload the queue entry renders for this record.
    pub payload: Value,
}

impl JournalQueue {
    /// A queue in `root` whose journal records are collected for the caller to write.
    ///
    /// The queue files live under `root`, one directory per task, exactly as [`DirectoryQueue`].
    #[must_use]
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        Self {
            directory: DirectoryQueue::new(root),
            pending: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// The records waiting to be written to a journal, clearing them.
    ///
    /// Called by whoever holds `&mut Journal` — in the daemon, the runner after each tool call.
    /// A poisoned lock returns nothing rather than panicking: the caller is mid-run, and a
    /// question that is durably on disk but missing its audit line is a gap in the log, not a
    /// reason to abort work a human is waiting for.
    #[must_use]
    pub fn take_pending(&self) -> Vec<Pending> {
        self.pending
            .lock()
            .map(|mut pending| std::mem::take(&mut *pending))
            .unwrap_or_default()
    }

    fn push(&self, pending: Pending) {
        if let Ok(mut slot) = self.pending.lock() {
            slot.push(pending);
        }
    }
}

impl EscalationQueue for JournalQueue {
    fn load(&self, task: &TaskId) -> CoreResult<EscalationStore> {
        self.directory.load(task)
    }

    fn opened(&self, entry: &Entry) -> CoreResult<()> {
        self.push(Pending {
            kind: immortality_core::journal::Kind::EscalationOpened,
            task: entry.task.clone(),
            payload: entry.opened_payload(),
        });
        Ok(())
    }

    fn resolved(&self, entry: &Entry) -> CoreResult<()> {
        self.push(Pending {
            kind: immortality_core::journal::Kind::EscalationResolved,
            task: entry.task.clone(),
            payload: entry.resolved_payload(),
        });
        Ok(())
    }
}

/// An answer, from wherever the human gave it — the CLI, the TUI, or a future socket message.
///
/// Separate from [`Escalate`] because the two run in different processes: the agent asks, and a
/// human (or `immortality approve`) answers. Both go through the same store and the same journal
/// hook, so there is exactly one code path that can record an answer.
///
/// # Errors
/// No such escalation in this task's queue, a conflicting answer to one already answered, or the
/// queue cannot be read or written.
pub fn answer_question(
    queue: &dyn EscalationQueue,
    task: &TaskId,
    id: &immortality_core::ids::EscalationId,
    answer: Answer,
    at: Timestamp,
) -> CoreResult<Entry> {
    let mut store = queue.load(task)?;
    let entry = store.answer(id, answer, at)?;
    // The file is durable before the journal record: a crash in between leaves a resolved entry
    // that was never journaled, which recovery reads from the file. The other order would leave
    // a journal that claims an answer the human never sees in the queue.
    queue.resolved(&entry)?;
    Ok(entry)
}

fn escalate_schema() -> &'static Value {
    static SCHEMA: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_KEY_LEN,
                    "description": "Stable identity of the question, e.g. `network:example.com:docs`. The same key is never asked twice; use the same key to check for an answer you are waiting on."
                },
                "question": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_QUESTION_LEN,
                    "description": "What you need the human to decide, in their language, with the specific action and the reason."
                },
                "kind": {
                    "type": "string",
                    "enum": EscalationKind::ALL.iter().map(|kind| kind.as_str()).collect::<Vec<_>>(),
                    "description": "Why you are asking. `external_action` for push/network/sudo, `blocked` when no subtask can proceed."
                },
                "options": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": Answer::ALL.iter().map(|answer| answer.as_str()).collect::<Vec<_>>()
                    },
                    "description": "Which answers to offer. Defaults to allow_once, allow_for_task, deny."
                }
            },
            "required": ["key", "question", "kind"],
            "additionalProperties": false
        })
    })
}

/// Ask the human a question.
pub struct Escalate {
    queue: Arc<dyn EscalationQueue>,
    notifier: Arc<dyn Notifier>,
}

impl std::fmt::Debug for Escalate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Escalate")
            .field("queue", &self.queue)
            .field("notifier", &self.notifier)
            .finish()
    }
}

impl Escalate {
    /// A tool writing to `queue` and telling the human through `notifier`.
    #[must_use]
    pub fn new(queue: Arc<dyn EscalationQueue>, notifier: Arc<dyn Notifier>) -> Self {
        Self { queue, notifier }
    }

    /// A tool whose escalation records are collected for the journal's owner.
    ///
    /// The concrete type is kept rather than widened at the call site so that "the tool and the
    /// runner hold the *same* collector" is visible in the signature: two `JournalQueue`s built
    /// from one path would look identical and silently drop each other's records.
    #[must_use]
    pub fn collecting(queue: Arc<JournalQueue>, notifier: Arc<dyn Notifier>) -> Self {
        Self::new(queue, notifier)
    }

    /// A tool backed by a directory and a notifier.
    #[must_use]
    pub fn with_directory(root: impl Into<std::path::PathBuf>, notifier: Arc<dyn Notifier>) -> Self {
        Self::new(Arc::new(DirectoryQueue::new(root)), notifier)
    }
}

impl Tool for Escalate {
    fn name(&self) -> &'static str {
        "escalate"
    }

    fn schema(&self) -> &'static Value {
        escalate_schema()
    }

    fn description(&self) -> &'static str {
        "Ask the human to approve an irreversible action or unblock you. Returns immediately; continue with other work while it is pending. Calling again with the same `key` returns the current answer."
    }

    fn effect(&self, _args: &Args) -> Effect {
        // The escalation mechanism itself. See the module docs for why this does not recurse
        // back into `PolicyGate::check`.
        Effect::External
    }

    fn idempotency(&self, _args: &Args) -> Idempotency {
        // Safe to repeat: the store deduplicates by key, so the second call answers from the
        // queue instead of asking again. That is exactly `ByContent` — a replay is harmless but
        // only after consulting the record.
        Idempotency::ByContent
    }

    fn paths(&self, _args: &Args) -> Vec<PathUse> {
        // Nothing in the worktree. The queue is under the daemon's state directory, which the
        // model never names and the gate does not govern.
        Vec::new()
    }

    /// The question, not the tool: an escalation in the log is read by the human who has to answer
    /// it, and "escalate" on its own tells them nothing about what is being asked.
    fn expected_effect(&self, args: &Args) -> String {
        let key = args.str("key").unwrap_or("?");
        let question = args.str("question").unwrap_or("");
        crate::tool::one_line(
            &format!("ask the human about `{key}`: {question}"),
            crate::tool::EFFECT_LIMIT,
        )
    }

    fn execute(&self, ctx: &ToolCtx, args: Args) -> Result<ToolResult, ToolError> {
        let key = normalise_key(args.str("key")?)?;
        let question = args.str("question")?.trim().to_string();
        if question.is_empty() {
            return Err(ToolError::Arguments(
                "`question` is empty. Say what the human has to decide.".to_string(),
            ));
        }
        if question.len() > MAX_QUESTION_LEN {
            return Err(ToolError::Arguments(format!(
                "`question` is {} bytes; the limit is {MAX_QUESTION_LEN}. Ask what you need, \
                 not the whole plan.",
                question.len()
            )));
        }
        let kind = EscalationKind::parse(args.str("kind")?).map_err(|_| {
            ToolError::Arguments(format!(
                "`kind` must be one of {}. `journal_corrupt` is only for a journal that fails \
                 its chain check, and it is the only kind that stops the task.",
                EscalationKind::ALL
                    .iter()
                    .map(|kind| kind.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
        let options = read_options(&args)?;

        let task = ctx.task().ok_or_else(|| {
            ToolError::Refused(
                json!({
                    "error": "no_task",
                    "hint": "An escalation belongs to a task. This call has no task context, which \
                             is a daemon bug rather than something to retry.",
                })
                .to_string(),
            )
        })?;
        let mut store = self.queue.load(task).map_err(|error| queue_error(&error))?;

        let entry = Entry::new(task.clone(), key, kind, question.clone(), options, now());
        let opened = store.ask(entry).map_err(|error| queue_error(&error))?;
        let outcome = match opened {
            // Already answered: hand the answer back so the model can act in this step.
            Opened::Answered(entry) => answered_result(&entry, "already answered"),
            // Pending: nothing changed, so nothing is journaled and nobody is notified again.
            Opened::Existing(entry) => pending_result(&entry, "already pending"),
            Opened::New(entry) => {
                self.queue.opened(&entry).map_err(|error| queue_error(&error))?;
                self.notifier.notify(
                    "immortality — decision needed",
                    &format!("{}: {question}", entry.kind.as_str()),
                );
                pending_result(&entry, "asked")
            }
        };
        Ok(outcome)
    }
}

/// The clock, in one place, so a test can see the same value the entry carries.
fn now() -> Timestamp {
    time::now()
}

/// Trim and bound the key, refusing an empty one.
///
/// A key is a deduplication identity. `external:push` and `external:push ` being distinct is the
/// one failure that makes the queue useless, so the trim happens here and everywhere else that
/// compares keys uses the stored (trimmed) form.
fn normalise_key(raw: &str) -> Result<String, ToolError> {
    let key = raw.trim();
    if key.is_empty() {
        return Err(ToolError::Arguments(
            "`key` is empty. It is the identity used to avoid asking the same question twice; \
             use something stable like `network:example.com:docs`."
                .to_string(),
        ));
    }
    if key.len() > MAX_KEY_LEN {
        return Err(ToolError::Arguments(format!(
            "`key` is {} bytes; the limit is {MAX_KEY_LEN}. A key is a stable identity, not a \
             description.",
            key.len()
        )));
    }
    Ok(key.to_string())
}

/// Parse the offered answers.
fn read_options(args: &Args) -> Result<Vec<Answer>, ToolError> {
    let Some(Value::Array(items)) = args.raw().get("options") else {
        if args.raw().get("options").is_some() {
            return Err(ToolError::Arguments(
                "`options` must be an array of answer names".to_string(),
            ));
        }
        return Ok(Answer::ALL.to_vec());
    };
    if items.is_empty() {
        return Ok(Answer::ALL.to_vec());
    }
    let mut options = Vec::new();
    for item in items {
        let Some(name) = item.as_str() else {
            return Err(ToolError::Arguments(
                "`options` must be an array of answer names".to_string(),
            ));
        };
        options.push(Answer::parse(name).map_err(|_| {
            ToolError::Arguments(format!(
                "`{name}` is not an answer. Use `allow_once`, `allow_for_task` or `deny`."
            ))
        })?);
    }
    Ok(options)
}

/// The observation for a question that still has no answer.
fn pending_result(entry: &Entry, state: &str) -> ToolResult {
    let mut observation = format!(
        "Escalation {} is {state}. The human has been asked: {}\n\
         Do not wait and do not ask again with this key. Continue with work that does not need \
         this answer; the decision arrives as an observation.",
        entry.id, entry.question
    );
    if entry.blocking {
        observation.push_str(
            "\nThis question is blocking: no subtask can proceed until it is answered, so wait \
             for the human rather than starting new work.",
        );
    }
    ToolResult::new(
        observation,
        json!({
            "escalation": entry.key,
            "id": entry.id,
            "kind": entry.kind,
            "status": "pending",
            "blocking": entry.blocking,
            "options": entry.options,
            "created_at": entry.created_at,
        }),
    )
}

/// The observation for a question that already has an answer.
fn answered_result(entry: &Entry, state: &str) -> ToolResult {
    let answer = entry.answer;
    let observation = match answer {
        Some(answer) if answer.permits() => format!(
            "Escalation {} was {state}: permitted ({}). You may proceed with the action the \
             question described.",
            entry.id,
            answer.as_str()
        ),
        Some(answer) => format!(
            "Escalation {} was {state}: refused ({}). Do not perform the action; choose another \
             way or finish what does not need it.",
            entry.id,
            answer.as_str()
        ),
        None => format!("Escalation {} is still pending.", entry.id),
    };
    ToolResult::new(
        observation,
        json!({
            "escalation": entry.key,
            "id": entry.id,
            "kind": entry.kind,
            "status": if entry.answer.is_some() { "answered" } else { "pending" },
            "answer": entry.answer,
            "allowed": entry.is_allowed(),
            "resolved_at": entry.resolved_at,
        }),
    )
}

/// Turn a store error into the message the model sees.
fn queue_error(error: &immortality_core::Error) -> ToolError {
    ToolError::Refused(
        json!({
            "error": "escalation_queue_unavailable",
            "detail": error.to_string(),
            "hint": "The question was not recorded. This is a storage problem, not a policy \
                     decision; report it if it persists.",
        })
        .to_string(),
    )
}

/// A pending-and-blocking question, if any, for the scheduler.
///
/// Free function rather than a method so the daemon can call it with only a queue handle.
///
/// # Errors
/// The queue could not be read.
pub fn blocking_question(queue: &dyn EscalationQueue, task: &TaskId) -> CoreResult<Option<Entry>> {
    let store = queue.load(task)?;
    Ok(store.blocking().cloned())
}

/// The whole queue, grouped by kind, for a CLI listing.
///
/// # Errors
/// The queue could not be read.
#[must_use = "the grouped listing is the point of the call"]
pub fn listing(
    queue: &dyn EscalationQueue,
    task: &TaskId,
) -> CoreResult<std::collections::BTreeMap<EscalationKind, Vec<Entry>>> {
    let store = queue.load(task)?;
    Ok(group_open(store.open_entries())
        .into_iter()
        .map(|(kind, entries)| (kind, entries.into_iter().cloned().collect()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notify::Silent;
    use crate::policy::PolicyGate;
    use std::sync::Mutex;

    /// Records notifications, so a test can assert the human was told exactly once.
    #[derive(Debug, Default)]
    struct Recorder {
        calls: Mutex<Vec<(String, String)>>,
    }

    impl Recorder {
        fn count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }
        fn body(&self) -> String {
            self.calls
                .lock()
                .unwrap()
                .first()
                .map(|(_, body)| body.clone())
                .unwrap_or_default()
        }
    }

    impl Notifier for Recorder {
        fn notify(&self, title: &str, body: &str) {
            self.calls
                .lock()
                .unwrap()
                .push((title.to_string(), body.to_string()));
        }
    }

    struct Fixture {
        dir: tempfile::TempDir,
        task: TaskId,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            Self {
                dir,
                task: TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV"),
            }
        }

        fn queue(&self) -> DirectoryQueue {
            DirectoryQueue::new(self.dir.path())
        }

        fn store(&self) -> EscalationStore {
            self.queue().load(&self.task).unwrap()
        }

        /// A context carrying a task and a gate over the tempdir.
        fn ctx(&self) -> ToolCtx {
            let gate = PolicyGate::with_defaults(
                self.dir.path(),
                immortality_core::config::Policy {
                    workspace_root: self.dir.path().parent().unwrap().to_path_buf(),
                    ..immortality_core::config::Policy::default()
                },
            )
            .unwrap();
            ToolCtx::new(Arc::new(gate), crate::OutputBudget::default()).with_task(self.task.clone())
        }

        fn tool(&self, notifier: Arc<dyn Notifier>) -> Escalate {
            Escalate::new(Arc::new(self.queue()), notifier)
        }
    }

    fn ask(fixture: &Fixture, notifier: Arc<dyn Notifier>, args: Value) -> Result<ToolResult, ToolError> {
        fixture.tool(notifier).execute(&fixture.ctx(), Args::new(args))
    }

    #[test]
    fn a_new_question_is_recorded_and_the_human_is_notified_once() {
        let fixture = Fixture::new();
        let notifier = Arc::new(Recorder::default());
        let result = ask(
            &fixture,
            notifier.clone(),
            json!({
                "key": "network:example.com:docs",
                "question": "May I fetch the API docs from example.com?",
                "kind": "external_action"
            }),
        )
        .unwrap();

        assert_eq!(result.data["status"], "pending");
        assert_eq!(result.data["blocking"], false);
        assert_eq!(notifier.count(), 1, "the human must be told");
        assert!(notifier.body().contains("example.com"), "{}", notifier.body());

        let store = fixture.store();
        assert_eq!(store.open_entries().count(), 1);
        assert_eq!(
            store.by_key("network:example.com:docs").unwrap().kind,
            EscalationKind::ExternalAction
        );
    }

    #[test]
    fn asking_the_same_key_twice_notifies_once() {
        // The dedup rule, seen from the agent's side: a loop that asks every step must not
        // produce a notification every step.
        let fixture = Fixture::new();
        let notifier = Arc::new(Recorder::default());
        let args = json!({"key": "k", "question": "May I?", "kind": "external_action"});
        ask(&fixture, notifier.clone(), args.clone()).unwrap();
        let second = ask(&fixture, notifier.clone(), args).unwrap();

        assert_eq!(notifier.count(), 1);
        assert_eq!(second.data["status"], "pending");
        assert_eq!(fixture.store().entries().len(), 1);
    }

    #[test]
    fn a_repeat_after_an_answer_returns_the_answer_in_the_same_step() {
        // The model must be able to act on an answer without waiting for a later step: the whole
        // point of the key is that asking again is how you check.
        let fixture = Fixture::new();
        let notifier = Arc::new(Recorder::default());
        let args = json!({"key": "push", "question": "May I push?", "kind": "external_action"});
        let first = ask(&fixture, notifier.clone(), args.clone()).unwrap();
        let id = first.data["id"].as_str().unwrap().to_string();

        let mut store = fixture.store();
        store
            .answer(
                &immortality_core::ids::EscalationId::parse(&id).unwrap(),
                Answer::AllowForTask,
                time::now(),
            )
            .unwrap();

        let again = ask(&fixture, notifier.clone(), args).unwrap();
        assert_eq!(again.data["status"], "answered");
        assert_eq!(again.data["answer"], "allow_for_task");
        assert_eq!(again.data["allowed"], true);
        assert!(again.observation.contains("permitted"), "{}", again.observation);
        assert_eq!(notifier.count(), 1, "an answered key must not notify again");
    }

    #[test]
    fn a_denied_answer_tells_the_model_to_stop() {
        let fixture = Fixture::new();
        let notifier = Arc::new(Recorder::default());
        let args = json!({"key": "sudo", "question": "May I sudo?", "kind": "external_action"});
        let id = ask(&fixture, notifier.clone(), args.clone()).unwrap().data["id"]
            .as_str()
            .unwrap()
            .to_string();
        let mut store = fixture.store();
        store
            .answer(
                &immortality_core::ids::EscalationId::parse(&id).unwrap(),
                Answer::Deny,
                time::now(),
            )
            .unwrap();

        let again = ask(&fixture, notifier, args).unwrap();
        assert_eq!(again.data["allowed"], false);
        assert!(again.observation.contains("refused"), "{}", again.observation);
    }

    #[test]
    fn the_blocking_flag_follows_the_kind_and_not_the_model() {
        // `blocking` is not a tool argument: a model cannot stop the whole task with a network
        // question, and cannot fail to stop it with a corrupt journal.
        let fixture = Fixture::new();
        let schema = fixture.tool(Arc::new(Silent)).schema().clone();
        assert!(schema["properties"].get("blocking").is_none(), "{schema}");

        let ordinary = ask(
            &fixture,
            Arc::new(Silent),
            json!({"key": "a", "question": "net?", "kind": "external_action"}),
        )
        .unwrap();
        assert_eq!(ordinary.data["blocking"], false);

        let corrupt = ask(
            &fixture,
            Arc::new(Silent),
            json!({"key": "b", "question": "journal is broken", "kind": "journal_corrupt"}),
        )
        .unwrap();
        assert_eq!(corrupt.data["blocking"], true);
        assert!(
            corrupt.observation.contains("blocking"),
            "the model must know it may not start new work: {}",
            corrupt.observation
        );
    }

    #[test]
    fn an_empty_key_is_refused_with_a_usable_message() {
        let fixture = Fixture::new();
        let error = ask(
            &fixture,
            Arc::new(Silent),
            json!({"key": "   ", "question": "?", "kind": "external_action"}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("`key` is empty"), "{error}");
    }

    #[test]
    fn a_padded_key_deduplicates_against_the_trimmed_one() {
        // `external:push` and `external:push ` being distinct is the failure that makes the
        // queue useless in exactly the loop it exists for.
        let fixture = Fixture::new();
        let notifier = Arc::new(Recorder::default());
        ask(
            &fixture,
            notifier.clone(),
            json!({"key": "push", "question": "?", "kind": "external_action"}),
        )
        .unwrap();
        ask(
            &fixture,
            notifier.clone(),
            json!({"key": "  push\t", "question": "?", "kind": "external_action"}),
        )
        .unwrap();
        assert_eq!(notifier.count(), 1);
        assert_eq!(fixture.store().entries().len(), 1);
    }

    #[test]
    fn an_unknown_kind_is_refused_by_name() {
        let fixture = Fixture::new();
        let error = ask(
            &fixture,
            Arc::new(Silent),
            json!({"key": "k", "question": "?", "kind": "make_me_a_sandwich"}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("`kind` must be one of"), "{error}");
    }

    #[test]
    fn an_empty_question_is_refused() {
        let fixture = Fixture::new();
        let error = ask(
            &fixture,
            Arc::new(Silent),
            json!({"key": "k", "question": "   ", "kind": "blocked"}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("`question` is empty"), "{error}");
    }

    #[test]
    fn a_question_over_the_limit_is_refused_without_writing() {
        let fixture = Fixture::new();
        let long = "x".repeat(MAX_QUESTION_LEN + 1);
        let error = ask(
            &fixture,
            Arc::new(Silent),
            json!({"key": "k", "question": long, "kind": "blocked"}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("limit"), "{error}");
        assert!(!fixture.store().path().exists());
    }

    #[test]
    fn options_default_to_the_three_answers_and_can_be_narrowed() {
        let fixture = Fixture::new();
        ask(
            &fixture,
            Arc::new(Silent),
            json!({"key": "a", "question": "?", "kind": "external_action"}),
        )
        .unwrap();
        assert_eq!(fixture.store().by_key("a").unwrap().options, Answer::ALL.to_vec());

        ask(
            &fixture,
            Arc::new(Silent),
            json!({"key": "b", "question": "?", "kind": "blocked", "options": ["allow_once", "deny"]}),
        )
        .unwrap();
        assert_eq!(
            fixture.store().by_key("b").unwrap().options,
            vec![Answer::AllowOnce, Answer::Deny]
        );
    }

    #[test]
    fn an_unknown_option_is_refused() {
        let fixture = Fixture::new();
        let error = ask(
            &fixture,
            Arc::new(Silent),
            json!({"key": "k", "question": "?", "kind": "blocked", "options": ["maybe"]}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("not an answer"), "{error}");
    }

    #[test]
    fn a_call_without_a_task_is_refused_rather_than_writing_somewhere_arbitrary() {
        let fixture = Fixture::new();
        let gate = PolicyGate::with_defaults(
            fixture.dir.path(),
            immortality_core::config::Policy {
                workspace_root: fixture.dir.path().parent().unwrap().to_path_buf(),
                ..immortality_core::config::Policy::default()
            },
        )
        .unwrap();
        let ctx = ToolCtx::new(Arc::new(gate), crate::OutputBudget::default());
        let error = fixture
            .tool(Arc::new(Silent))
            .execute(
                &ctx,
                Args::new(json!({"key": "k", "question": "?", "kind": "blocked"})),
            )
            .unwrap_err();
        assert!(error.to_string().contains("no_task"), "{error}");
    }

    #[test]
    fn a_corrupt_queue_is_a_storage_problem_with_a_pointer() {
        let fixture = Fixture::new();
        let dir = fixture.queue().task_dir(&fixture.task);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(EscalationStore::path_for(&dir), "{not json").unwrap();
        let error = ask(
            &fixture,
            Arc::new(Silent),
            json!({"key": "k", "question": "?", "kind": "blocked"}),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("escalation_queue_unavailable"),
            "{error}"
        );
    }

    #[test]
    fn the_tool_declares_no_worktree_paths() {
        // The queue is under the daemon's state directory. Declaring a worktree path would send
        // the gate looking for a file that is not there.
        assert_eq!(
            Escalate::with_directory("/tmp", Arc::new(Silent)).paths(&Args::empty()),
            Vec::new()
        );
    }

    #[test]
    fn effects_read_the_queue_and_journal_the_change() {
        let tool = Escalate::with_directory("/tmp", Arc::new(Silent));
        let args = Args::new(json!({"key": "k", "question": "?", "kind": "blocked"}));
        assert_eq!(tool.effect(&args), Effect::External);
        assert_eq!(tool.idempotency(&args), Idempotency::ByContent);
    }

    #[test]
    fn the_schema_is_well_formed_and_offers_the_documented_kinds() {
        let schema = escalate_schema();
        assert_eq!(immortality_llm::schema::schema_defects(schema), Vec::new());
        let kinds = schema["properties"]["kind"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|value| value.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            EscalationKind::ALL.iter().map(|k| k.as_str()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn listing_groups_the_open_questions() {
        let fixture = Fixture::new();
        let queue = DirectoryQueue::new(fixture.dir.path());
        ask(
            &fixture,
            Arc::new(Silent),
            json!({"key": "a", "question": "net?", "kind": "external_action"}),
        )
        .unwrap();
        ask(
            &fixture,
            Arc::new(Silent),
            json!({"key": "b", "question": "disk", "kind": "disk"}),
        )
        .unwrap();

        let grouped = listing(&queue, &fixture.task).unwrap();
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[&EscalationKind::ExternalAction][0].key, "a");
    }

    /// A task with a real journal directory, so both halves of the change are observable.
    struct JournalFixture {
        dir: tempfile::TempDir,
        task: TaskId,
    }

    impl JournalFixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            Self {
                dir,
                task: TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV"),
            }
        }

        fn agent_dir(&self) -> std::path::PathBuf {
            self.dir.path().join("repo/.agent")
        }

        fn queue(&self) -> JournalQueue {
            JournalQueue::new(self.dir.path().join("state"))
        }

        /// What the journal's owner does after a tool call: take what the queue collected and
        /// write it. The queue cannot do this itself — it would be a second writer on a segment
        /// the runner holds open (D-117).
        fn drain(&self, queue: &JournalQueue) {
            let pending = queue.take_pending();
            if pending.is_empty() {
                return;
            }
            let mut journal =
                immortality_core::journal::Journal::open(self.agent_dir(), self.task.clone()).unwrap();
            for one in pending {
                journal
                    .append_event(journal.now(), one.kind, one.payload)
                    .unwrap();
            }
            journal.close().unwrap();
        }

        fn ctx(&self) -> ToolCtx {
            let repo = self.dir.path().join("repo");
            std::fs::create_dir_all(&repo).unwrap();
            let gate = PolicyGate::with_defaults(
                &repo,
                immortality_core::config::Policy {
                    workspace_root: self.dir.path().to_path_buf(),
                    ..immortality_core::config::Policy::default()
                },
            )
            .unwrap();
            ToolCtx::new(Arc::new(gate), crate::OutputBudget::default()).with_task(self.task.clone())
        }

        fn records(&self) -> Vec<immortality_core::journal::JournalRecord> {
            immortality_core::journal::JournalReader::open(self.agent_dir(), &self.task)
                .unwrap()
                .read()
                .unwrap()
                .records
        }
    }

    #[test]
    fn the_queue_collects_and_the_journal_owner_writes() {
        // The file is the working set; the journal is the audit trail. Both must record the
        // change, in the order a human reading them expects: opened, then resolved.
        //
        // The split is not cosmetic. A queue that opened the journal itself would be a second
        // writer on a segment the runner holds, which duplicates `seq` and breaks the digest
        // chain — `a_second_writer_on_a_held_journal_breaks_its_chain` in `core` pins that.
        use immortality_core::journal::Kind;

        let fixture = JournalFixture::new();
        let queue = fixture.queue();
        let tool = Escalate::new(Arc::new(queue.clone()), Arc::new(Silent));
        let asked = tool
            .execute(
                &fixture.ctx(),
                Args::new(json!({
                    "key": "network:example.com",
                    "question": "May I fetch the docs?",
                    "kind": "external_action"
                })),
            )
            .unwrap();
        let id = immortality_core::ids::EscalationId::parse(asked.data["id"].as_str().unwrap()).unwrap();

        // Collected, and *not* written: there is no journal file until the owner takes it.
        assert_eq!(
            fixture.records(),
            Vec::<immortality_core::journal::JournalRecord>::new(),
            "the queue wrote the journal itself; in the daemon that is a second writer"
        );

        fixture.drain(&queue);
        let kinds: Vec<Kind> = fixture.records().iter().map(|r| r.fields.kind).collect();
        assert_eq!(kinds, vec![Kind::EscalationOpened]);

        // The same instance, not a fresh one: the records live in the collector, so a caller that
        // rebuilt the queue would silently drop them. In the daemon there is one collector per
        // task, held by whoever holds the journal.
        let entry = answer_question(&queue, &fixture.task, &id, Answer::AllowForTask, time::now()).unwrap();
        assert!(entry.is_allowed());
        fixture.drain(&queue);

        let records = fixture.records();
        let kinds: Vec<Kind> = records.iter().map(|r| r.fields.kind).collect();
        assert_eq!(kinds, vec![Kind::EscalationOpened, Kind::EscalationResolved]);
        assert_eq!(
            records[1].event.payload["answer"], "allow_for_task",
            "{:?}",
            records[1].event.payload
        );
        assert_eq!(records[0].event.payload["key"], "network:example.com");
    }

    #[test]
    fn a_question_that_was_already_asked_is_not_collected_twice() {
        // The journal must gain one record per real change, not one per call: the model asks the
        // same key on every step, and a collector that pushed on every call would make the audit
        // trail say the human was asked sixty times.
        let fixture = JournalFixture::new();
        let queue = fixture.queue();
        let tool = Escalate::new(Arc::new(queue.clone()), Arc::new(Silent));
        let args = Args::new(json!({"key": "k", "question": "?", "kind": "external_action"}));
        tool.execute(&fixture.ctx(), args.clone()).unwrap();
        tool.execute(&fixture.ctx(), args).unwrap();

        assert_eq!(
            queue.take_pending().len(),
            1,
            "the second ask changed nothing, so it must not be journalled again"
        );
        assert!(queue.take_pending().is_empty(), "the take cleared it");
    }

    #[test]
    fn answering_through_the_shared_path_persists_before_it_journals() {
        // The order matters: a crash between the two leaves a resolved file the human sees and
        // a journal that recovery can catch up from. The reverse would journal an answer the
        // queue does not contain.
        let fixture = JournalFixture::new();
        let queue = fixture.queue();
        let tool = Escalate::new(Arc::new(queue.clone()), Arc::new(Silent));
        let asked = tool
            .execute(
                &fixture.ctx(),
                Args::new(json!({"key": "k", "question": "?", "kind": "blocked"})),
            )
            .unwrap();
        let id = immortality_core::ids::EscalationId::parse(asked.data["id"].as_str().unwrap()).unwrap();
        answer_question(&queue, &fixture.task, &id, Answer::Deny, time::now()).unwrap();

        // The file is already durable, before any journal write.
        let store = queue.load(&fixture.task).unwrap();
        assert_eq!(store.get(&id).unwrap().answer, Some(Answer::Deny));
        // And the record is waiting, not lost.
        let pending = queue.take_pending();
        assert_eq!(
            pending.iter().map(|one| one.kind).collect::<Vec<_>>(),
            vec![
                immortality_core::journal::Kind::EscalationOpened,
                immortality_core::journal::Kind::EscalationResolved,
            ]
        );
    }

    #[test]
    fn an_answer_to_a_question_that_was_never_asked_is_refused() {
        let fixture = JournalFixture::new();
        let error = answer_question(
            &fixture.queue(),
            &fixture.task,
            &immortality_core::ids::EscalationId::generate(),
            Answer::Deny,
            time::now(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("no escalation"), "{error}");
    }

    #[test]
    fn a_blocking_question_is_visible_without_the_tool() {
        // What the scheduler asks before a step. It must not need a `ToolCtx` or an argument.
        let fixture = Fixture::new();
        ask(
            &fixture,
            Arc::new(Silent),
            json!({"key": "j", "question": "journal broken", "kind": "journal_corrupt"}),
        )
        .unwrap();
        let queue = DirectoryQueue::new(fixture.dir.path());
        assert_eq!(
            blocking_question(&queue, &fixture.task).unwrap().unwrap().key,
            "j"
        );
    }
}
