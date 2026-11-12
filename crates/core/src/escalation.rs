//! The escalation queue: questions for the human that do not stop the work.
//!
//! # Why a queue and not a prompt
//!
//! The design's rule (D-010) is that an irreversible action is *asked about*, not *confirmed
//! inline*. A blocking prompt would mean every `git push` and every untrusted host stalls a run
//! that is supposed to last days, and the human is asleep for most of it. So a question becomes
//! a record on disk, the agent is told "waiting for an answer" and continues with other
//! subtasks, and the answer can arrive hours later from a different process.
//!
//! # Why the file is the source of truth
//!
//! The daemon is a single process that restarts; the human's answer may arrive while it is
//! down. Storing the queue in `escalations.json` under the task's state directory means an
//! answer is durable the moment the CLI writes it, and the daemon reads it when it comes back.
//! The journal records `escalation_opened`/`escalation_resolved` as the audit trail; this file is
//! the working set. They can disagree for a moment after a crash, and the *file* wins, because
//! it is the only one the human touched.
//!
//! # Deduplication
//!
//! The `key` is the identity of a question, not the id. Asking "may I fetch example.com/docs"
//! twice is noise: the human answers once and the agent either has permission or does not. A
//! second [`EscalationStore::open`] with the same key returns the existing entry — if it is
//! already answered, the answer is returned immediately and no new question is created.
//!
//! # `allow_for_task`
//!
//! The one answer with an expiry. It grants the *key* until the task ends, and the task's end is
//! not something this file can observe. So the grant is stored as a deadline-bearing decision
//! and [`EscalationStore::allowance`] answers "is this key allowed right now", which the policy
//! layer asks. Nothing here deletes the grant: a task has one state directory, and `gc` removes
//! the directory when the task is finished.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::atomic::write_atomically;
use crate::domain::Timestamp;
use crate::ids::{EscalationId, TaskId};

/// The file name inside a task's state directory.
pub const FILE_NAME: &str = "escalations.json";

/// Version of the on-disk schema.
pub const SCHEMA_VERSION: u16 = 1;

/// Why a question was asked.
///
/// A closed set rather than a free string: the CLI groups by it, and `blocking` is derived from
/// it — only [`EscalationKind::JournalCorrupt`] may stop the whole task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EscalationKind {
    /// Push, network outside `trusted_hosts`, sudo, an install outside the worktree.
    ExternalAction,
    /// No subtask can proceed until a human decides (D-025's `Hold`, or a human-only merge).
    Blocked,
    /// The backend has been unreachable longer than `unavailable_escalate_after_secs`.
    BackendUnavailable,
    /// The journal fails its own chain check. The only kind that sets `blocking`.
    JournalCorrupt,
    /// The disk guard fired and the task is paused on `Hold::PausedDisk`.
    Disk,
}

impl EscalationKind {
    /// Every kind, in the order a CLI listing shows them.
    pub const ALL: &'static [Self] = &[
        Self::ExternalAction,
        Self::Blocked,
        Self::BackendUnavailable,
        Self::JournalCorrupt,
        Self::Disk,
    ];

    /// The name the model writes and the file stores.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExternalAction => "external_action",
            Self::Blocked => "blocked",
            Self::BackendUnavailable => "backend_unavailable",
            Self::JournalCorrupt => "journal_corrupt",
            Self::Disk => "disk",
        }
    }

    /// Whether a question of this kind stops everything.
    ///
    /// `journal_corrupt` is the only one, and the reason is literal: every other question can be
    /// left pending while the agent works on a different subtask, whereas a corrupt journal means
    /// the system cannot tell what already happened. Continuing would risk duplicating an
    /// irreversible effect, so the task waits.
    #[must_use]
    pub const fn blocks_everything(self) -> bool {
        matches!(self, Self::JournalCorrupt)
    }

    /// # Errors
    /// The name is not one of [`Self::ALL`].
    pub fn parse(name: &str) -> crate::Result<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|kind| kind.as_str() == name)
            .ok_or_else(|| {
                crate::Error::Protocol(format!(
                    "`{name}` is not an escalation kind; expected one of {}",
                    Self::ALL
                        .iter()
                        .map(|kind| kind.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })
    }
}

/// What the human may answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Answer {
    /// This one time.
    AllowOnce,
    /// Until the task ends. The key, not the call: a second `git push` to the same remote is the
    /// same question.
    AllowForTask,
    /// No, and do not ask again — the key stays denied until the task ends.
    Deny,
}

impl Answer {
    /// The three answers, in the order the CLI offers them.
    pub const ALL: &'static [Self] = &[Self::AllowOnce, Self::AllowForTask, Self::Deny];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AllowOnce => "allow_once",
            Self::AllowForTask => "allow_for_task",
            Self::Deny => "deny",
        }
    }

    /// Whether the action may run.
    #[must_use]
    pub const fn permits(self) -> bool {
        matches!(self, Self::AllowOnce | Self::AllowForTask)
    }

    /// # Errors
    /// The name is not one of [`Self::ALL`].
    pub fn parse(name: &str) -> crate::Result<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|answer| answer.as_str() == name)
            .ok_or_else(|| {
                crate::Error::Protocol(format!(
                    "`{name}` is not an answer; expected one of allow_once, allow_for_task, deny"
                ))
            })
    }
}

/// One question, and its answer once there is one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub id: EscalationId,
    /// Deduplication key. Stable across restarts; `external:command:git` for a policy refusal.
    pub key: String,
    pub task: TaskId,
    pub kind: EscalationKind,
    /// The question, in the human's language.
    pub question: String,
    /// The answers offered. Never empty; [`EscalationKind`] and the caller decide which.
    pub options: Vec<Answer>,
    /// Whether the whole task waits. Only `journal_corrupt` sets it; the constructor enforces it.
    pub blocking: bool,
    #[serde(with = "crate::time::millis")]
    pub created_at: Timestamp,
    /// Absent until answered. Not `null`: a human reads this file.
    #[serde(
        default,
        with = "crate::time::millis_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub resolved_at: Option<Timestamp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<Answer>,
}

impl Entry {
    /// A new, unanswered question.
    ///
    /// `blocking` is not a parameter: it follows from the kind, so a caller cannot accidentally
    /// stop the whole task with an ordinary network question.
    #[must_use]
    pub fn new(
        task: TaskId,
        key: impl Into<String>,
        kind: EscalationKind,
        question: impl Into<String>,
        options: Vec<Answer>,
        at: Timestamp,
    ) -> Self {
        Self {
            id: EscalationId::generate(),
            key: key.into(),
            task,
            kind,
            question: question.into(),
            options: if options.is_empty() {
                Answer::ALL.to_vec()
            } else {
                options
            },
            blocking: kind.blocks_everything(),
            created_at: at,
            resolved_at: None,
            answer: None,
        }
    }

    /// Whether a human has answered yet.
    #[must_use]
    pub const fn is_open(&self) -> bool {
        self.answer.is_none()
    }

    /// Whether the answer permits the action.
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        self.answer.is_some_and(Answer::permits)
    }

    /// Record an answer, checking it against the options the human was offered.
    ///
    /// # Errors
    /// Already answered, or the answer was not on offer. Both are programming errors: the CLI
    /// validates against `options` before writing, so reaching here means a caller skipped it.
    pub fn resolve(&mut self, answer: Answer, at: Timestamp) -> crate::Result<()> {
        if self.answer.is_some() {
            return Err(crate::Error::Protocol(format!(
                "escalation {} was already answered",
                self.id
            )));
        }
        if !self.options.contains(&answer) {
            return Err(crate::Error::Protocol(format!(
                "`{}` was not offered for {}",
                answer.as_str(),
                self.id
            )));
        }
        self.answer = Some(answer);
        self.resolved_at = Some(at);
        Ok(())
    }

    /// The journal payload for `escalation_opened`.
    #[must_use]
    pub fn opened_payload(&self) -> Value {
        json!({
            "id": self.id,
            "key": self.key,
            "kind": self.kind,
            "blocking": self.blocking,
            "options": self.options,
        })
    }

    /// The journal payload for `escalation_resolved`.
    #[must_use]
    pub fn resolved_payload(&self) -> Value {
        json!({
            "id": self.id,
            "key": self.key,
            "kind": self.kind,
            "answer": self.answer,
        })
    }
}

/// What `open` did, because the caller must journal only what actually happened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Opened {
    /// A new question was written. The caller appends `escalation_opened`.
    New(Entry),
    /// The key was already asked and is still pending; nothing changed on disk.
    Existing(Entry),
    /// The key was already answered. The answer is returned so the caller can act on it now.
    Answered(Entry),
}

impl Opened {
    /// The entry, whichever case.
    #[must_use]
    pub const fn entry(&self) -> &Entry {
        match self {
            Self::New(entry) | Self::Existing(entry) | Self::Answered(entry) => entry,
        }
    }
}

/// The on-disk file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileBody {
    v: u16,
    entries: Vec<Entry>,
}

/// The queue for one task.
///
/// Holds the path rather than an open handle: every mutation rewrites the file atomically, and a
/// long-lived handle would be a second source of truth. The daemon and the CLI are different
/// processes touching the same path, which is exactly the arrangement this shape survives.
#[derive(Debug, Clone)]
pub struct EscalationStore {
    path: PathBuf,
    task: TaskId,
    entries: Vec<Entry>,
}

impl EscalationStore {
    /// The file a task's queue lives in.
    #[must_use]
    pub fn path_for(task_dir: &Path) -> PathBuf {
        task_dir.join(FILE_NAME)
    }

    /// Open a task's queue, creating the file if it does not exist.
    ///
    /// A missing file is an empty queue, not an error: a task that never asked anything must not
    /// need a file on disk to prove it.
    ///
    /// # Errors
    /// The file exists but cannot be read or parsed, or the directory cannot be created. A parse
    /// failure is deliberately fatal rather than "start fresh": starting fresh would silently
    /// lose an answer the human already gave.
    pub fn open(task_dir: &Path, task: TaskId) -> crate::Result<Self> {
        let path = Self::path_for(task_dir);
        let entries = match std::fs::read_to_string(&path) {
            Ok(text) => {
                let body: FileBody = serde_json::from_str(&text).map_err(|error| {
                    crate::Error::Protocol(format!(
                        "{} is not a valid escalation file: {error}",
                        path.display()
                    ))
                })?;
                if body.v != SCHEMA_VERSION {
                    return Err(crate::Error::JournalVersion {
                        found: body.v,
                        expected: SCHEMA_VERSION,
                    });
                }
                body.entries
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(crate::Error::io(&path, error)),
        };
        Ok(Self { path, task, entries })
    }

    /// The task this queue belongs to.
    #[must_use]
    pub const fn task(&self) -> &TaskId {
        &self.task
    }

    /// Where the file is.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Every entry, oldest first.
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Entries a human has not answered yet.
    pub fn open_entries(&self) -> impl Iterator<Item = &Entry> {
        self.entries.iter().filter(|entry| entry.is_open())
    }

    /// Whether anything pending blocks the whole task.
    ///
    /// The scheduler asks this before starting a step: a blocking question stops the task even
    /// though every other kind leaves it running (D-010).
    #[must_use]
    pub fn blocking(&self) -> Option<&Entry> {
        self.open_entries().find(|entry| entry.blocking)
    }

    /// The entry with this id, if any.
    #[must_use]
    pub fn get(&self, id: &EscalationId) -> Option<&Entry> {
        self.entries.iter().find(|entry| &entry.id == id)
    }

    /// The entry with this key, if any. Answers deduplication.
    #[must_use]
    pub fn by_key(&self, key: &str) -> Option<&Entry> {
        self.entries.iter().find(|entry| entry.key == key)
    }

    /// Is `key` currently granted by an `allow_for_task` answer?
    ///
    /// The policy layer asks this before escalating. A denial answers `false` as well: the key
    /// was refused and must not be asked again during this task.
    #[must_use]
    pub fn allowance(&self, key: &str) -> bool {
        self.by_key(key)
            .is_some_and(|entry| entry.answer == Some(Answer::AllowForTask))
    }

    /// Ask a question, deduplicating by `key`.
    ///
    /// Writes the file only when a new entry is created, so a retry that asks the same question
    /// costs no disk write and no `escalation_opened` record.
    ///
    /// # Errors
    /// The file could not be written.
    pub fn ask(&mut self, entry: Entry) -> crate::Result<Opened> {
        if let Some(existing) = self.by_key(&entry.key) {
            return Ok(if existing.is_open() {
                Opened::Existing(existing.clone())
            } else {
                Opened::Answered(existing.clone())
            });
        }
        if entry.task != self.task {
            return Err(crate::Error::Protocol(format!(
                "escalation for {} was added to the queue of {}",
                entry.task, self.task
            )));
        }
        self.entries.push(entry.clone());
        self.save()?;
        Ok(Opened::New(entry))
    }

    /// Answer an open question.
    ///
    /// Returns the updated entry so the caller can journal `escalation_resolved` and tell the
    /// agent. Answering an already-answered entry with the *same* answer is a no-op that returns
    /// the entry: the CLI retries after a lost socket more often than a human double-clicks.
    ///
    /// # Errors
    /// No such id, or a conflicting answer to an already-answered entry, or the file cannot be
    /// written.
    pub fn answer(&mut self, id: &EscalationId, answer: Answer, at: Timestamp) -> crate::Result<Entry> {
        let Some(index) = self.entries.iter().position(|entry| &entry.id == id) else {
            return Err(crate::Error::Protocol(format!(
                "no escalation `{id}` in this task's queue"
            )));
        };
        if self.entries[index].answer.is_some() {
            if self.entries[index].answer == Some(answer) {
                return Ok(self.entries[index].clone());
            }
            return Err(crate::Error::Protocol(format!(
                "escalation {id} already has a different answer"
            )));
        }
        self.entries[index].resolve(answer, at)?;
        self.save()?;
        Ok(self.entries[index].clone())
    }

    /// Rewrite the file, atomically.
    fn save(&self) -> crate::Result<()> {
        let body = FileBody {
            v: SCHEMA_VERSION,
            entries: self.entries.clone(),
        };
        let mut text = serde_json::to_string_pretty(&body)
            .map_err(|error| crate::Error::Protocol(format!("cannot serialize escalations: {error}")))?;
        text.push('\n');
        write_atomically(&self.path, text.as_bytes())
    }
}

/// Group open entries by kind, for a CLI listing.
#[must_use]
pub fn group_open<'a>(
    entries: impl IntoIterator<Item = &'a Entry>,
) -> BTreeMap<EscalationKind, Vec<&'a Entry>> {
    let mut out: BTreeMap<EscalationKind, Vec<&Entry>> = BTreeMap::new();
    for entry in entries {
        out.entry(entry.kind).or_default().push(entry);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task() -> TaskId {
        TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV")
    }

    fn at(offset_ms: i64) -> Timestamp {
        crate::time::now() + chrono::Duration::milliseconds(offset_ms)
    }

    fn store() -> (tempfile::TempDir, EscalationStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = EscalationStore::open(dir.path(), task()).unwrap();
        (dir, store)
    }

    fn entry(key: &str, kind: EscalationKind) -> Entry {
        Entry::new(
            task(),
            key,
            kind,
            "May I do the thing?",
            Answer::ALL.to_vec(),
            at(0),
        )
    }

    #[test]
    fn a_new_question_is_written_and_survives_a_reopen() {
        let (dir, mut store) = store();
        let Opened::New(created) = store
            .ask(entry("network:example.com", EscalationKind::ExternalAction))
            .unwrap()
        else {
            panic!("expected a new entry");
        };
        assert!(created.id.as_str().starts_with("esc_"));
        assert!(created.is_open());

        let reopened = EscalationStore::open(dir.path(), task()).unwrap();
        assert_eq!(reopened.entries().len(), 1);
        assert_eq!(reopened.by_key("network:example.com").unwrap().id, created.id);
    }

    #[test]
    fn the_same_key_is_never_asked_twice() {
        // The rule from the design: an identical question is put to the human once.
        let (_dir, mut store) = store();
        let first = store.ask(entry("k", EscalationKind::ExternalAction)).unwrap();
        let second = store.ask(entry("k", EscalationKind::ExternalAction)).unwrap();
        assert!(matches!(first, Opened::New(_)));
        assert!(matches!(second, Opened::Existing(_)));
        assert_eq!(store.entries().len(), 1);
        assert_eq!(first.entry().id, second.entry().id);
    }

    #[test]
    fn a_repeat_of_an_answered_key_returns_the_answer() {
        // Not `Existing`: the caller must be able to act on it without waiting for anybody.
        let (_dir, mut store) = store();
        let opened = store.ask(entry("k", EscalationKind::ExternalAction)).unwrap();
        let id = opened.entry().id.clone();
        store.answer(&id, Answer::AllowForTask, at(1)).unwrap();

        let again = store.ask(entry("k", EscalationKind::ExternalAction)).unwrap();
        let Opened::Answered(answered) = again else {
            panic!("expected the stored answer, got {again:?}");
        };
        assert_eq!(answered.answer, Some(Answer::AllowForTask));
        assert!(answered.is_allowed());
        assert_eq!(store.entries().len(), 1);
    }

    #[test]
    fn a_dedup_hit_does_not_rewrite_the_file() {
        // Nothing changed, so nothing is journaled and nothing hits the disk. The mtime is the
        // cheapest observable proof that a `save` did not happen.
        let (_dir, mut store) = store();
        store.ask(entry("k", EscalationKind::ExternalAction)).unwrap();
        let before = std::fs::metadata(store.path()).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        store.ask(entry("k", EscalationKind::ExternalAction)).unwrap();
        let after = std::fs::metadata(store.path()).unwrap().modified().unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn an_answer_is_durable_and_idempotent() {
        let (dir, mut store) = store();
        let id = store
            .ask(entry("k", EscalationKind::ExternalAction))
            .unwrap()
            .entry()
            .id
            .clone();
        let answered = store.answer(&id, Answer::Deny, at(5)).unwrap();
        assert_eq!(answered.answer, Some(Answer::Deny));
        assert!(answered.resolved_at.is_some());
        assert!(!answered.is_allowed());

        // The CLI retrying after a lost response must not be an error.
        assert_eq!(store.answer(&id, Answer::Deny, at(9)).unwrap(), answered);

        let reopened = EscalationStore::open(dir.path(), task()).unwrap();
        assert_eq!(reopened.get(&id).unwrap().answer, Some(Answer::Deny));
    }

    #[test]
    fn a_conflicting_second_answer_is_refused() {
        let (_dir, mut store) = store();
        let id = store
            .ask(entry("k", EscalationKind::ExternalAction))
            .unwrap()
            .entry()
            .id
            .clone();
        store.answer(&id, Answer::AllowOnce, at(1)).unwrap();
        assert!(store.answer(&id, Answer::Deny, at(2)).is_err());
        assert_eq!(store.get(&id).unwrap().answer, Some(Answer::AllowOnce));
    }

    #[test]
    fn an_answer_that_was_not_offered_is_refused() {
        let (_dir, mut store) = store();
        let mut question = entry("k", EscalationKind::ExternalAction);
        question.options = vec![Answer::Deny];
        let id = store.ask(question).unwrap().entry().id.clone();
        assert!(store.answer(&id, Answer::AllowOnce, at(1)).is_err());
    }

    #[test]
    fn answering_an_unknown_id_names_the_problem() {
        let (_dir, mut store) = store();
        let error = store
            .answer(&EscalationId::generate(), Answer::Deny, at(1))
            .unwrap_err();
        assert!(error.to_string().contains("no escalation"), "{error}");
    }

    #[test]
    fn allow_for_task_grants_the_key_and_allow_once_does_not() {
        let (_dir, mut store) = store();
        let first = store
            .ask(entry("push", EscalationKind::ExternalAction))
            .unwrap()
            .entry()
            .id
            .clone();
        store.answer(&first, Answer::AllowForTask, at(1)).unwrap();
        assert!(store.allowance("push"));
        assert!(!store.allowance("fetch"), "grant is per key, not global");

        let second = store
            .ask(entry("fetch", EscalationKind::ExternalAction))
            .unwrap()
            .entry()
            .id
            .clone();
        store.answer(&second, Answer::AllowOnce, at(2)).unwrap();
        assert!(
            !store.allowance("fetch"),
            "allow_once covers the call that asked, not the task"
        );
    }

    #[test]
    fn a_denied_key_is_not_an_allowance() {
        let (_dir, mut store) = store();
        let id = store
            .ask(entry("k", EscalationKind::ExternalAction))
            .unwrap()
            .entry()
            .id
            .clone();
        store.answer(&id, Answer::Deny, at(1)).unwrap();
        assert!(!store.allowance("k"));
    }

    #[test]
    fn only_journal_corrupt_blocks_the_whole_task() {
        for kind in EscalationKind::ALL {
            let entry = entry("k", *kind);
            assert_eq!(
                entry.blocking,
                *kind == EscalationKind::JournalCorrupt,
                "{kind:?}"
            );
        }
    }

    #[test]
    fn a_blocking_question_is_reported_and_a_pending_ordinary_one_is_not() {
        let (_dir, mut store) = store();
        store
            .ask(entry("network", EscalationKind::ExternalAction))
            .unwrap();
        assert!(
            store.blocking().is_none(),
            "an external action must not stop the task"
        );

        store
            .ask(entry("journal", EscalationKind::JournalCorrupt))
            .unwrap();
        assert_eq!(store.blocking().unwrap().key, "journal");

        // Answering it clears the block: a corrupt journal that a human has acknowledged is no
        // longer something the scheduler must wait on.
        let id = store.blocking().unwrap().id.clone();
        store.answer(&id, Answer::AllowOnce, at(1)).unwrap();
        assert!(store.blocking().is_none());
    }

    #[test]
    fn an_open_blocking_question_survives_a_reopen() {
        // The block is the one thing that must not be lost across a daemon restart.
        let (dir, mut store) = store();
        store
            .ask(entry("journal", EscalationKind::JournalCorrupt))
            .unwrap();
        drop(store);
        let reopened = EscalationStore::open(dir.path(), task()).unwrap();
        assert_eq!(reopened.blocking().unwrap().key, "journal");
    }

    #[test]
    fn a_missing_file_is_an_empty_queue() {
        let dir = tempfile::tempdir().unwrap();
        let store = EscalationStore::open(dir.path(), task()).unwrap();
        assert_eq!(store.entries(), []);
        assert!(store.blocking().is_none());
        assert!(!store.path().exists(), "reading must not create the file");
    }

    #[test]
    fn a_corrupt_file_is_fatal_rather_than_an_empty_queue() {
        // Starting fresh would silently discard an answer the human already gave.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(EscalationStore::path_for(dir.path()), "{not json").unwrap();
        let error = EscalationStore::open(dir.path(), task()).unwrap_err();
        assert!(
            error.to_string().contains("not a valid escalation file"),
            "{error}"
        );
    }

    #[test]
    fn a_file_from_a_newer_schema_is_refused_by_version() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            EscalationStore::path_for(dir.path()),
            r#"{"v": 99, "entries": []}"#,
        )
        .unwrap();
        let error = EscalationStore::open(dir.path(), task()).unwrap_err();
        assert!(error.to_string().contains("version"), "{error}");
    }

    #[test]
    fn the_stored_file_is_readable_and_carries_no_nulls() {
        // A human opens this file. An unresolved timestamp must be absent, not `null`, and the
        // file must be indented enough to read.
        let (_dir, mut store) = store();
        store.ask(entry("k", EscalationKind::ExternalAction)).unwrap();
        let text = std::fs::read_to_string(store.path()).unwrap();
        assert!(text.contains("\"v\": 1"), "{text}");
        assert!(!text.contains("null"), "{text}");
        assert!(text.contains('\n'), "{text}");
        assert!(
            text.ends_with('\n'),
            "a file a shell appends to ends with a newline"
        );
    }

    #[test]
    fn the_journal_payloads_name_the_entry_and_its_answer() {
        let (_dir, mut store) = store();
        let id = store
            .ask(entry("k", EscalationKind::Disk))
            .unwrap()
            .entry()
            .id
            .clone();
        let opened = store.get(&id).unwrap();
        assert_eq!(opened.opened_payload()["kind"], "disk");
        assert_eq!(opened.opened_payload()["blocking"], false);

        let resolved = store.answer(&id, Answer::AllowOnce, at(1)).unwrap();
        assert_eq!(
            resolved.resolved_payload()["answer"],
            "allow_once",
            "{:?}",
            resolved.resolved_payload()
        );
    }

    #[test]
    fn an_entry_from_another_task_is_refused() {
        // Two tasks sharing a directory would otherwise answer each other's questions.
        let (_dir, mut store) = store();
        let mut foreign = entry("k", EscalationKind::ExternalAction);
        foreign.task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAW");
        assert!(store.ask(foreign).is_err());
        assert_eq!(store.entries(), []);
    }

    #[test]
    fn kinds_and_answers_roundtrip_through_their_names() {
        for kind in EscalationKind::ALL {
            assert_eq!(&EscalationKind::parse(kind.as_str()).unwrap(), kind);
        }
        for answer in Answer::ALL {
            assert_eq!(&Answer::parse(answer.as_str()).unwrap(), answer);
        }
        assert!(EscalationKind::parse("not_a_kind").is_err());
        assert!(Answer::parse("maybe").is_err());
    }

    #[test]
    fn grouping_keeps_the_documented_kind_order() {
        let (_dir, mut store) = store();
        store.ask(entry("a", EscalationKind::JournalCorrupt)).unwrap();
        store.ask(entry("b", EscalationKind::ExternalAction)).unwrap();
        let grouped = group_open(store.open_entries());
        let kinds: Vec<EscalationKind> = grouped.keys().copied().collect();
        assert_eq!(
            kinds,
            vec![EscalationKind::ExternalAction, EscalationKind::JournalCorrupt]
        );
    }
}
