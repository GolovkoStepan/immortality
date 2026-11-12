//! Delegation reports: what a sub-agent achieved, addressed by the call that asked for it.
//!
//! # Why a report file and not just a tool result
//!
//! A `delegate` call is the one tool call that spends a whole sub-agent's budget, and its
//! idempotency class is `Unknown`: a crash after the sub-agent ran leaves no way to know
//! whether the work happened. Recovery therefore probes, and the probe needs a durable
//! artefact to look for. That artefact is this file.
//!
//! The file also outlives the conversation. The parent's next step is assembled from state,
//! not from the sub-agent's context — that is the whole point of a fresh context — so the
//! report is the *only* channel through which the sub-agent's findings reach the parent.
//!
//! # Why the id is a hash of the call id
//!
//! The probe runs after a crash, when all that survives is the journal's `intent` record with
//! its `call_id`. The report must be findable from exactly that, with no arguments and no
//! memory. So the id is derived: `del_` plus the first eight bytes of `sha256(call_id)`.
//!
//! The call id itself cannot be the file name even though it would be the obvious choice: it
//! comes from the model, so it may contain `/`, `..`, or a few thousand bytes, and a
//! model-chosen path component is a directory-traversal bug waiting for a model that has one.
//!
//! # Why a failure still writes a report
//!
//! If the sub-agent exhausts its step budget, that is an outcome the parent must know about —
//! it means the brief needs splitting differently. Writing the report in that case also makes
//! the probe's `Applied` answer correct: the delegated run *did* happen, so retrying it would
//! spend the budget twice and repeat whatever the sub-agent already did.
//!
//! The one case that writes nothing is a runner error *before* the sub-agent starts: nothing
//! happened, the probe says `NotApplied`, and recovery retries.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::atomic::write_atomically;
use crate::domain::{SubtaskReport, Timestamp};

/// The file name inside a task's `delegations` directory carries this suffix.
pub const FILE_SUFFIX: &str = ".json";

/// Version of the on-disk schema.
pub const SCHEMA_VERSION: u16 = 1;

/// Prefix of a delegation id.
pub const ID_PREFIX: &str = "del_";

/// How many digest bytes the textual id keeps.
///
/// Eight is 64 bits: a task would need billions of delegations for a collision to be plausible,
/// and the id stays short enough to read in a journal line.
const ID_BYTES: usize = 8;

/// The identity of one delegated run.
///
/// Derivable from the call id alone, which is what makes the crash-time probe possible. Not a
/// ULID: it must be the *same* value before and after a restart, and a ULID is by definition
/// fresh.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DelegationId(String);

impl DelegationId {
    /// The id for a tool call.
    #[must_use]
    pub fn for_call(call_id: &str) -> Self {
        let digest = crate::artifacts::digest_of(call_id.as_bytes());
        let mut out = String::with_capacity(ID_PREFIX.len() + ID_BYTES * 2);
        out.push_str(ID_PREFIX);
        for byte in digest.iter().take(ID_BYTES) {
            use std::fmt::Write as _;
            // Writing into a `String` cannot fail.
            let _ = write!(out, "{byte:02x}");
        }
        Self(out)
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// # Errors
    /// The text is not `del_` followed by sixteen lowercase hex digits.
    pub fn parse(text: &str) -> crate::Result<Self> {
        let suffix = text
            .strip_prefix(ID_PREFIX)
            .ok_or_else(|| crate::Error::InvalidId(format!("`{text}` does not start with `{ID_PREFIX}`")))?;
        if suffix.len() != ID_BYTES * 2 || !suffix.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(crate::Error::InvalidId(format!(
                "`{text}` is not `{ID_PREFIX}` followed by {} hex digits",
                ID_BYTES * 2
            )));
        }
        Ok(Self(text.to_string()))
    }
}

impl std::fmt::Display for DelegationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// How a delegated run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Finish {
    /// The sub-agent declared the brief done.
    Done,
    /// The step budget ran out. The work may be partially done; the parent must decide.
    Exhausted,
    /// The sub-agent failed for a reason it could report: a red gate, an unrecoverable tool
    /// error. Distinct from a runner error, which writes no report at all.
    Failed,
}

closed_state!(Finish {
    Done => "done",
    Exhausted => "exhausted",
    Failed => "failed",
});

impl Finish {
    /// Whether the parent should treat the brief as complete.
    #[must_use]
    pub const fn is_done(self) -> bool {
        matches!(self, Self::Done)
    }
}

/// One delegated run's durable result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelegationReport {
    /// Schema version, first in the file so a human sees it before the prose.
    pub v: u16,
    pub id: DelegationId,
    /// The brief the sub-agent was given, verbatim.
    ///
    /// Kept because it is the only record of what was asked: the journal's `intent` payload
    /// may be spilled to an artifact, and a report whose question is unreadable is not
    /// reviewable.
    pub brief: String,
    /// What the sub-agent was allowed to touch, as normalised path prefixes.
    pub scope: Vec<String>,
    /// The limit the sub-agent was given.
    pub max_steps: u32,
    /// How many steps it actually used.
    pub steps: u32,
    pub finish: Finish,
    /// The structured result, the same shape a subtask reports.
    pub report: SubtaskReport,
    /// Why it failed, when `finish` is not `done`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(with = "crate::time::millis")]
    pub created_at: Timestamp,
}

impl DelegationReport {
    /// A report with no outcome yet; the caller fills one in with [`Self::with_outcome`].
    #[must_use]
    pub fn new(
        id: DelegationId,
        brief: impl Into<String>,
        scope: Vec<String>,
        max_steps: u32,
        at: Timestamp,
    ) -> Self {
        Self {
            v: SCHEMA_VERSION,
            id,
            brief: brief.into(),
            scope,
            max_steps,
            steps: 0,
            finish: Finish::Done,
            report: SubtaskReport::default(),
            error: None,
            created_at: at,
        }
    }

    /// Attach the outcome the runner returned.
    #[must_use]
    pub fn with_outcome(
        mut self,
        steps: u32,
        finish: Finish,
        report: SubtaskReport,
        error: Option<String>,
    ) -> Self {
        self.steps = steps;
        self.finish = finish;
        self.report = report;
        self.error = error;
        self
    }

    /// Whether `path` was inside the scope the sub-agent was given.
    #[must_use]
    pub fn respects_scope(&self, path: &Path) -> bool {
        within_scope(path, &self.scope)
    }
}

/// The reports of one task.
///
/// Rooted at the task's state directory, not at a global store: a report is part of a task's
/// history, deleted with it, and never shared between tasks.
#[derive(Debug, Clone)]
pub struct DelegationStore {
    dir: PathBuf,
}

impl DelegationStore {
    /// The store for `task` under a state root.
    ///
    /// The `delegations` subdirectory exists so that `gc` and a human can tell the reports
    /// apart from `escalations.json`, `subtasks.json` and the rest without knowing our names.
    #[must_use]
    pub fn for_task(state_root: &Path, task: &crate::ids::TaskId) -> Self {
        Self {
            dir: state_root.join(task.as_str()).join("delegations"),
        }
    }

    /// A store over an explicit directory, for tests and for a caller that computes the path.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Where the reports live.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The file for one report.
    #[must_use]
    pub fn path_for(&self, id: &DelegationId) -> PathBuf {
        self.dir.join(format!("{}{FILE_SUFFIX}", id.as_str()))
    }

    /// Write a report, atomically and durably.
    ///
    /// # Errors
    /// The directory cannot be created, or the write fails.
    pub fn write(&self, report: &DelegationReport) -> crate::Result<()> {
        let mut text = serde_json::to_string_pretty(report).map_err(|error| {
            crate::Error::Protocol(format!("cannot serialize a delegation report: {error}"))
        })?;
        text.push('\n');
        write_atomically(&self.path_for(&report.id), text.as_bytes())
    }

    /// Read one report.
    ///
    /// `None` means "the sub-agent never completed". A file that exists but does not parse is
    /// an error, never `None`: silently treating corruption as absence is how a crash-time
    /// probe answers `NotApplied` and the whole delegated run is repeated.
    ///
    /// # Errors
    /// The file cannot be read, does not parse, is from a newer schema, or belongs to another
    /// delegation.
    pub fn read(&self, id: &DelegationId) -> crate::Result<Option<DelegationReport>> {
        let path = self.path_for(id);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(crate::Error::io(&path, error)),
        };
        let report: DelegationReport = serde_json::from_str(&text).map_err(|error| {
            crate::Error::Protocol(format!(
                "{} is not a valid delegation report: {error}",
                path.display()
            ))
        })?;
        if report.v != SCHEMA_VERSION {
            return Err(crate::Error::JournalVersion {
                found: report.v,
                expected: SCHEMA_VERSION,
            });
        }
        if &report.id != id {
            return Err(crate::Error::Protocol(format!(
                "{} holds a report for {}, not {id}",
                path.display(),
                report.id
            )));
        }
        Ok(Some(report))
    }

    /// Whether a completed report exists. This is the probe, in one line.
    #[must_use]
    pub fn exists(&self, id: &DelegationId) -> bool {
        self.path_for(id).is_file()
    }

    /// Every readable report, oldest id first.
    ///
    /// Ids are hashes, so the order is arbitrary but stable — good enough for a CLI listing,
    /// which shows `created_at` anyway.
    ///
    /// # Errors
    /// The directory cannot be listed, or a report in it is corrupt.
    pub fn list(&self) -> crate::Result<Vec<DelegationReport>> {
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(crate::Error::io(&self.dir, error)),
        };
        let mut ids = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| crate::Error::io(&self.dir, error))?;
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            let Some(stem) = name.strip_suffix(FILE_SUFFIX) else {
                continue;
            };
            if let Ok(id) = DelegationId::parse(stem) {
                ids.push(id);
            }
        }
        ids.sort();
        let mut reports = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(report) = self.read(&id)? {
                reports.push(report);
            }
        }
        Ok(reports)
    }
}

/// Normalise one scope entry, or `None` if it is not a usable path prefix.
///
/// A scope entry is how the parent narrows what a sub-agent may touch, so it must be a relative
/// path inside the worktree: an absolute path, a `~`, or anything climbing with `..` would
/// widen the sub-agent's reach instead of narrowing it.
#[must_use]
pub fn normalise_scope(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.starts_with('~') {
        return None;
    }
    let path = Path::new(trimmed);
    if path.is_absolute() {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            // `ParentDir`, a root, or a prefix: all of them leave the worktree or are not a
            // relative path at all.
            _ => return None,
        }
    }
    if parts.is_empty() {
        return None;
    }
    Some(parts.join("/"))
}

/// Whether `path`, relative to the worktree, is inside `scope`.
///
/// Component-wise, so a scope of `src` covers `src/main.rs` but not `srcfoo/main.rs`. A path
/// that cannot be normalised is outside every scope: a sub-agent reporting `../etc/passwd` has
/// not respected a scope of `src`, whatever the string comparison would say.
#[must_use]
pub fn within_scope(path: &Path, scope: &[String]) -> bool {
    let Some(normalised) = normalise_scope(&path.to_string_lossy()) else {
        return false;
    };
    let candidate = Path::new(&normalised);
    scope.iter().any(|entry| {
        normalise_scope(entry).is_some_and(|entry| {
            let entry = Path::new(&entry);
            candidate == entry || candidate.starts_with(entry)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task() -> crate::ids::TaskId {
        crate::ids::TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV")
    }

    fn store() -> (tempfile::TempDir, DelegationStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = DelegationStore::for_task(dir.path(), &task());
        (dir, store)
    }

    fn report(id: DelegationId) -> DelegationReport {
        DelegationReport::new(
            id,
            "fix the parser",
            vec!["src".to_string()],
            20,
            crate::time::now(),
        )
        .with_outcome(
            7,
            Finish::Done,
            SubtaskReport {
                outcome: "the parser handles trailing commas".to_string(),
                facts: vec!["the grammar table is in src/grammar.rs".to_string()],
                files_touched: vec![PathBuf::from("src/parser.rs")],
                rejected_ideas: vec!["rewriting the lexer".to_string()],
            },
            None,
        )
    }

    #[test]
    fn an_id_is_derived_from_the_call_and_stays_the_same() {
        // The probe runs after a crash with nothing but the journal's call id. If this were
        // random, the probe could never find what it wrote.
        let a = DelegationId::for_call("call_01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let b = DelegationId::for_call("call_01ARZ3NDEKTSV4RRFFQ69G5FAV");
        assert_eq!(a, b);
        assert!(a.as_str().starts_with(ID_PREFIX));
        assert_eq!(a.as_str().len(), ID_PREFIX.len() + ID_BYTES * 2);
        assert_ne!(a, DelegationId::for_call("call_other"));
    }

    #[test]
    fn an_id_roundtrips_through_text_and_rejects_nonsense() {
        let id = DelegationId::for_call("call_x");
        assert_eq!(DelegationId::parse(id.as_str()).unwrap(), id);
        assert!(DelegationId::parse("call_x").is_err());
        assert!(DelegationId::parse("del_").is_err());
        assert!(DelegationId::parse("del_zz").is_err());
        // A path-ish id must never parse: it would become a file name.
        assert!(DelegationId::parse("del_../../etc/passwd").is_err());
    }

    #[test]
    fn a_report_roundtrips_and_creates_its_directory() {
        let (_dir, store) = store();
        let report = report(DelegationId::for_call("call_a"));
        store.write(&report).unwrap();
        assert_eq!(store.read(&report.id).unwrap(), Some(report.clone()));
        assert!(store.exists(&report.id));
        assert!(store.dir().is_dir());
    }

    #[test]
    fn a_missing_report_is_none_and_does_not_create_anything() {
        let (_dir, store) = store();
        assert_eq!(store.read(&DelegationId::for_call("call_a")).unwrap(), None);
        assert!(!store.exists(&DelegationId::for_call("call_a")));
        assert_eq!(store.list().unwrap(), []);
    }

    #[test]
    fn a_corrupt_report_is_an_error_not_an_absence() {
        // Treating corruption as absence is how the probe answers `NotApplied` and the whole
        // delegated run — a whole sub-agent's budget — is spent twice.
        let (_dir, store) = store();
        let id = DelegationId::for_call("call_a");
        std::fs::create_dir_all(store.dir()).unwrap();
        std::fs::write(store.path_for(&id), "{not json").unwrap();
        let error = store.read(&id).unwrap_err();
        assert!(
            error.to_string().contains("not a valid delegation report"),
            "{error}"
        );
        assert!(store.list().is_err());
    }

    #[test]
    fn a_report_in_the_wrong_file_is_refused() {
        let (_dir, store) = store();
        let first = report(DelegationId::for_call("call_a"));
        store.write(&first).unwrap();
        let second = DelegationId::for_call("call_b");
        std::fs::copy(store.path_for(&first.id), store.path_for(&second)).unwrap();
        let error = store.read(&second).unwrap_err();
        assert!(error.to_string().contains("holds a report for"), "{error}");
    }

    #[test]
    fn a_newer_schema_is_refused_by_version() {
        let (_dir, store) = store();
        let id = DelegationId::for_call("call_a");
        std::fs::create_dir_all(store.dir()).unwrap();
        let mut value = serde_json::to_value(report(id.clone())).unwrap();
        value["v"] = serde_json::json!(99);
        std::fs::write(store.path_for(&id), value.to_string()).unwrap();
        let error = store.read(&id).unwrap_err();
        assert!(error.to_string().contains("version"), "{error}");
    }

    #[test]
    fn listing_sorts_reports_and_ignores_foreign_files() {
        let (_dir, store) = store();
        let a = report(DelegationId::for_call("call_a"));
        let b = report(DelegationId::for_call("call_b"));
        store.write(&a).unwrap();
        store.write(&b).unwrap();
        std::fs::write(store.dir().join("notes.txt"), "mine").unwrap();

        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 2);
        // Sorted, and both present: the listing must not drop one because the id order is by
        // hash rather than by creation time.
        let ids: Vec<&str> = listed.iter().map(|r| r.id.as_str()).collect();
        let mut expected = vec![a.id.as_str(), b.id.as_str()];
        expected.sort_unstable();
        assert_eq!(ids, expected);
    }

    #[test]
    fn the_stored_file_is_readable_and_ends_with_a_newline() {
        let (_dir, store) = store();
        let report = report(DelegationId::for_call("call_a"));
        store.write(&report).unwrap();
        let text = std::fs::read_to_string(store.path_for(&report.id)).unwrap();
        assert!(text.contains("\"v\": 1"), "{text}");
        assert!(!text.contains("null"), "{text}");
        assert!(text.ends_with('\n'), "{text}");
        assert!(text.contains("trailing commas"), "{text}");
    }

    #[test]
    fn a_failed_run_still_writes_a_report_with_its_reason() {
        // The parent must learn that the brief was too big; and the probe must answer
        // `Applied`, because the sub-agent really did run.
        let (_dir, store) = store();
        let mut report = report(DelegationId::for_call("call_a"));
        report.finish = Finish::Exhausted;
        report.error = Some("step budget exhausted".to_string());
        store.write(&report).unwrap();

        let back = store.read(&report.id).unwrap().unwrap();
        assert_eq!(back.finish, Finish::Exhausted);
        assert_eq!(back.error.as_deref(), Some("step budget exhausted"));
    }

    #[test]
    fn a_scope_entry_is_a_relative_prefix_or_nothing() {
        assert_eq!(normalise_scope("src").as_deref(), Some("src"));
        assert_eq!(normalise_scope(" src/a.rs ").as_deref(), Some("src/a.rs"));
        assert_eq!(normalise_scope("./src//a.rs").as_deref(), Some("src/a.rs"));
        for bad in ["", "   ", "/etc", "~/x", "src/../etc", "..", "."] {
            assert_eq!(normalise_scope(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn scope_matching_is_component_wise_and_refuses_escapes() {
        let scope = vec!["src".to_string(), "docs/readme.md".to_string()];
        assert!(within_scope(Path::new("src/main.rs"), &scope));
        assert!(within_scope(Path::new("src/deep/nested.rs"), &scope));
        assert!(within_scope(Path::new("docs/readme.md"), &scope));
        // The prefix trap: `srcfoo` must not match a scope of `src`.
        assert!(!within_scope(Path::new("srcfoo/main.rs"), &scope));
        assert!(!within_scope(Path::new("tests/main.rs"), &scope));
        // An escape is outside every scope, whatever the leading components say.
        assert!(!within_scope(Path::new("src/../etc/passwd"), &scope));
        assert!(!within_scope(Path::new("/etc/passwd"), &scope));
    }

    #[test]
    fn finish_roundtrips_through_its_names() {
        for finish in Finish::ALL {
            assert_eq!(&Finish::parse(finish.as_str()).unwrap(), finish);
        }
        assert!(Finish::parse("maybe").is_err());
        assert!(Finish::Done.is_done());
        assert!(!Finish::Exhausted.is_done());
    }

    #[test]
    fn a_report_checks_its_own_scope() {
        let report = report(DelegationId::for_call("call_a"));
        assert!(report.respects_scope(Path::new("src/parser.rs")));
        assert!(!report.respects_scope(Path::new("Cargo.toml")));
    }
}
