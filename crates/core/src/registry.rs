//! Which repositories this daemon knows about.
//!
//! # Why this is needed now
//!
//! Everything a task owns is split across two places by design: its escalation queue and
//! transcripts live under `~/Library/Application Support/immortality/state/<project>/<task>/`,
//! while its journal — the audit trail a human reviews and git tracks — lives in the
//! repository's `.agent/`. Given only a task id, something has to answer "which repository?".
//! That something is this file.
//!
//! # Why the id is not the path
//!
//! Directories move and get renamed; a state directory keyed by a path would be orphaned by a
//! `mv`. The id is derived once, at `init`, from the directory name plus a digest of the
//! canonical path, and never changes afterwards. The current path is a field, so moving a
//! repository means editing one line here rather than losing its history.
//!
//! # Why a plain JSON file and not a database
//!
//! It is read at daemon start and after `init`, it is small enough to read in a `cat`, and a
//! human who needs to fix a stale path should be able to do it in an editor. [`crate::atomic`]
//! makes the write crash-safe, so the file is never half-written.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::atomic::write_atomically;
use crate::ids::TaskId;
use crate::journal::JournalReader;

/// File name inside the state root.
pub const FILE_NAME: &str = "registry.json";

/// Schema version.
pub const SCHEMA_VERSION: u16 = 1;

/// One repository the daemon may run tasks in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Project {
    /// Stable, path-independent id. Also the name of its state subdirectory.
    pub id: String,
    /// Canonical path of the repository root.
    pub path: PathBuf,
    /// `origin` URL when the repository has one. Informational: nothing here pushes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote: Option<String>,
}

impl Project {
    /// The directory holding this project's per-task state.
    #[must_use]
    pub fn state_dir(&self, root: &Path) -> PathBuf {
        root.join("state").join(&self.id)
    }

    /// The directory holding durable, git-tracked state for a task worktree.
    #[must_use]
    pub fn agent_dir(&self) -> PathBuf {
        self.path.join(crate::paths::AGENT_DIR)
    }
}

/// The on-disk file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileBody {
    v: u16,
    projects: Vec<Project>,
}

/// The set of known projects.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Registry {
    path: PathBuf,
    projects: Vec<Project>,
}

impl Registry {
    /// Open the registry at `path`. A missing file is an empty registry.
    ///
    /// # Errors
    /// The file exists but cannot be read or parsed, or is from a newer schema. Starting empty
    /// on a parse failure would silently detach every task from its repository.
    pub fn open(path: impl Into<PathBuf>) -> crate::Result<Self> {
        let path = path.into();
        let projects = match std::fs::read_to_string(&path) {
            Ok(text) => {
                let body: FileBody = serde_json::from_str(&text).map_err(|error| {
                    crate::Error::Protocol(format!("{} is not a valid registry: {error}", path.display()))
                })?;
                if body.v != SCHEMA_VERSION {
                    return Err(crate::Error::JournalVersion {
                        found: body.v,
                        expected: SCHEMA_VERSION,
                    });
                }
                body.projects
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(crate::Error::io(&path, error)),
        };
        Ok(Self { path, projects })
    }

    /// Where this registry is stored.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Every project.
    #[must_use]
    pub fn projects(&self) -> &[Project] {
        &self.projects
    }

    /// The project whose path is `path`, comparing canonical forms.
    ///
    /// Both sides are canonicalised because `/var` and `/private/var` are the same directory on
    /// macOS and a state directory looked up by the wrong spelling is a task with no history.
    #[must_use]
    pub fn by_path(&self, path: &Path) -> Option<&Project> {
        let wanted = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        self.projects.iter().find(|project| {
            project
                .path
                .canonicalize()
                .unwrap_or_else(|_| project.path.clone())
                == wanted
        })
    }

    /// The project with this id.
    #[must_use]
    pub fn by_id(&self, id: &str) -> Option<&Project> {
        self.projects.iter().find(|project| project.id == id)
    }

    /// The project that owns `task`, however the task is currently recorded.
    /// A task's existence has two independent traces: its private state directory under the
    /// state root, and `kind`-bearing records for its id inside its repository's journal. The
    /// state directory is created lazily — a task that has only journaled so far, or one whose
    /// machine-local state was pruned, is still a real task — so a lookup that consulted only
    /// the directory would report "no such task" for a run that is visibly in progress.
    ///
    /// The directory is checked first because it is one `stat`; the journal scan is the
    /// fallback and is why this is not on a hot path.
    ///
    /// # Errors
    /// The registry or a journal cannot be read.
    pub fn owner_of(&self, state_root: &Path, task: &TaskId) -> crate::Result<Option<&Project>> {
        for project in &self.projects {
            if project.state_dir(state_root).join(task.as_str()).is_dir() {
                return Ok(Some(project));
            }
        }
        for project in &self.projects {
            let outcome = crate::journal::JournalReader::open_unnamed(project.agent_dir())?.read()?;
            if outcome.records.iter().any(|record| &record.fields.task == task) {
                return Ok(Some(project));
            }
        }
        Ok(None)
    }

    /// Every task a project currently has, from both of its traces.
    ///
    /// A task directory under the state root, and records bearing the task id in the
    /// repository's journal, are independent traces of the same thing: the directory is
    /// machine-local and created lazily, the journal is durable and git-tracked. The union is
    /// what `status` shows, because a task whose state directory was pruned and whose journal
    /// remains is a task whose history a human can still read — and a run that has journaled
    /// its first record has begun even if no directory exists yet.
    ///
    /// # Errors
    /// A journal cannot be read.
    pub fn tasks(&self, state_root: &Path) -> crate::Result<Vec<(Project, TaskId, PathBuf)>> {
        let mut out = Vec::new();
        for project in &self.projects {
            let mut seen: BTreeSet<String> = BTreeSet::new();
            let dir = project.state_dir(state_root);
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    let Ok(task) = TaskId::parse(&name) else {
                        continue;
                    };
                    if entry.path().is_dir() && seen.insert(task.as_str().to_string()) {
                        out.push((project.clone(), task, entry.path()));
                    }
                }
            }

            // A journal directory that does not exist yet is an empty journal, not an error:
            // a project that has never run a task is a normal project. `open_unnamed` is used
            // because the task id is not known yet — discovering it is the point.
            let Ok(outcome) =
                JournalReader::open_unnamed(project.agent_dir()).and_then(|reader| reader.read())
            else {
                continue;
            };
            for record in &outcome.records {
                if seen.insert(record.fields.task.as_str().to_string()) {
                    let task = record.fields.task.clone();
                    let path = dir.join(task.as_str());
                    out.push((project.clone(), task, path));
                }
            }
        }
        out.sort_by(|left, right| left.1.cmp(&right.1));
        Ok(out)
    }

    /// Register `path`, or return the id it already has.
    ///
    /// Idempotent by canonical path, so running `immortality init` twice does not create a
    /// second identity for the same repository and orphan the first one's state.
    ///
    /// # Errors
    /// The registry cannot be written.
    pub fn register(&mut self, path: impl AsRef<Path>, remote: Option<String>) -> crate::Result<String> {
        let path = path.as_ref();
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        if let Some(existing) = self.by_path(&canonical) {
            return Ok(existing.id.clone());
        }
        let id = project_id(&canonical);
        self.projects.push(Project {
            id: id.clone(),
            path: canonical,
            remote,
        });
        self.projects.sort_by(|left, right| left.id.cmp(&right.id));
        self.save()?;
        Ok(id)
    }

    /// Write the file, atomically.
    fn save(&self) -> crate::Result<()> {
        let body = FileBody {
            v: SCHEMA_VERSION,
            projects: self.projects.clone(),
        };
        let mut text = serde_json::to_string_pretty(&body)
            .map_err(|error| crate::Error::Protocol(format!("cannot serialize the registry: {error}")))?;
        text.push('\n');
        write_atomically(&self.path, text.as_bytes())
    }
}

/// A stable id for a repository path: a readable slug and a digest of the full path.
///
/// The slug makes `state/immortality-9a1e0c/` recognisable in a `ls`; the digest makes two
/// repositories called `app` in different directories distinct. Only the digest decides
/// uniqueness, which is why the slug may be truncated without changing the id.
#[must_use]
pub fn project_id(path: &Path) -> String {
    let name = path.file_name().map_or_else(
        || "project".to_string(),
        |name| name.to_string_lossy().into_owned(),
    );
    let mut slug = String::new();
    for character in name.chars() {
        if character.is_ascii_alphanumeric() {
            slug.push(character.to_ascii_lowercase());
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
        if slug.len() >= 24 {
            break;
        }
    }
    let slug = slug.trim_matches('-');
    let slug = if slug.is_empty() { "project" } else { slug };

    let digest = crate::artifacts::digest_of(path.to_string_lossy().as_bytes());
    let suffix: String = digest.iter().take(3).fold(String::new(), |mut out, byte| {
        use std::fmt::Write as _;
        // Writing into a `String` cannot fail.
        let _ = write!(out, "{byte:02x}");
        out
    });
    format!("{slug}-{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> (tempfile::TempDir, Registry) {
        let dir = tempfile::tempdir().unwrap();
        let registry = Registry::open(dir.path().join(FILE_NAME)).unwrap();
        (dir, registry)
    }

    #[test]
    fn a_missing_registry_is_empty_and_does_not_create_a_file() {
        let (_dir, registry) = registry();
        assert_eq!(registry.projects(), []);
        assert!(!registry.path().exists());
    }

    #[test]
    fn registering_a_project_survives_a_reopen() {
        let (dir, mut registry) = registry();
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();

        let id = registry
            .register(&repo, Some("git@example.com:x/y".to_string()))
            .unwrap();
        let reopened = Registry::open(registry.path()).unwrap();
        let project = reopened.by_id(&id).unwrap();
        assert_eq!(project.path, repo.canonicalize().unwrap());
        assert_eq!(project.remote.as_deref(), Some("git@example.com:x/y"));
    }

    #[test]
    fn registering_twice_returns_the_same_id_and_keeps_one_entry() {
        // `init` is meant to be safe to run again; a second identity for one repository would
        // orphan the first one's task state.
        let (dir, mut registry) = registry();
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        let first = registry.register(&repo, None).unwrap();
        let second = registry.register(&repo, None).unwrap();
        assert_eq!(first, second);
        assert_eq!(registry.projects().len(), 1);
    }

    #[test]
    fn a_symlinked_path_is_the_same_project() {
        let (dir, mut registry) = registry();
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        let id = registry.register(&repo, None).unwrap();

        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&repo, &link).unwrap();
        assert_eq!(registry.register(&link, None).unwrap(), id);
        assert_eq!(registry.projects().len(), 1);
    }

    #[test]
    fn two_repositories_with_the_same_name_get_different_ids() {
        let (dir, mut registry) = registry();
        let first = dir.path().join("a/app");
        let second = dir.path().join("b/app");
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        let one = registry.register(&first, None).unwrap();
        let two = registry.register(&second, None).unwrap();
        assert_ne!(one, two);
        assert!(one.starts_with("app-"), "{one}");
        assert!(two.starts_with("app-"), "{two}");
    }

    #[test]
    fn an_id_is_readable_and_has_no_path_characters() {
        let id = project_id(Path::new("/Users/someone/rust_projects/immortality"));
        assert!(id.starts_with("immortality-"), "{id}");
        assert_eq!(id.len(), "immortality-".len() + 6, "{id}");
        for bad in ['/', '\\', '.', ' '] {
            assert!(!id.contains(bad), "{id}");
        }
    }

    #[test]
    fn a_project_with_no_usable_name_still_gets_an_id() {
        let id = project_id(Path::new("/"));
        assert!(id.starts_with("project-"), "{id}");
    }

    #[test]
    fn tasks_are_listed_per_project_and_foreign_directories_are_ignored() {
        let (dir, mut registry) = registry();
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        let id = registry.register(&repo, None).unwrap();

        let project = registry.by_id(&id).unwrap();
        let tasks = project.state_dir(dir.path());
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        std::fs::create_dir_all(tasks.join(task.as_str())).unwrap();
        std::fs::create_dir_all(tasks.join("not-a-task")).unwrap();

        let listed = registry.tasks(dir.path()).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].1, task);
        assert_eq!(listed[0].0.id, id);
    }

    #[test]
    fn listing_tasks_without_a_state_directory_is_empty() {
        let (dir, registry) = registry();
        assert_eq!(registry.tasks(dir.path()).unwrap(), []);
    }

    /// Write one record for `task` into `repo`'s journal, creating `.agent/` on the way.
    fn journal_record(repo: &Path, task: &TaskId) {
        let agent = repo.join(crate::paths::AGENT_DIR);
        let mut journal = crate::journal::Journal::open(&agent, task.clone()).unwrap();
        journal
            .append_event(
                crate::time::now(),
                crate::journal::Kind::TaskCreated,
                serde_json::json!({"title": "t"}),
            )
            .unwrap();
    }

    #[test]
    fn owner_of_finds_a_task_by_its_state_directory() {
        let (dir, mut registry) = registry();
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        let id = registry.register(&repo, None).unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let project = registry.by_id(&id).unwrap();
        std::fs::create_dir_all(project.state_dir(dir.path()).join(task.as_str())).unwrap();

        let owner = registry.owner_of(dir.path(), &task).unwrap().unwrap();
        assert_eq!(owner.id, id);
    }

    #[test]
    fn owner_of_falls_back_to_the_journal_when_no_state_directory_exists() {
        // The state directory is machine-local and created lazily, so a task that has journaled
        // its first record is real before its directory exists. Reporting "no such task" for a
        // run that is visibly in progress is the failure this covers.
        let (dir, mut registry) = registry();
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        let id = registry.register(&repo, None).unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        journal_record(&repo, &task);

        let owner = registry.owner_of(dir.path(), &task).unwrap().unwrap();
        assert_eq!(owner.id, id);
        assert!(
            !registry
                .by_id(&id)
                .unwrap()
                .state_dir(dir.path())
                .join(task.as_str())
                .exists(),
            "the test must not have created the state directory"
        );
    }

    #[test]
    fn owner_of_returns_none_for_a_task_nobody_claims() {
        let (dir, mut registry) = registry();
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        registry.register(&repo, None).unwrap();
        // Another repository's journal mentions the task; it is still not this registry's.
        let elsewhere = dir.path().join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        journal_record(&elsewhere, &task);

        assert!(registry.owner_of(dir.path(), &task).unwrap().is_none());
    }

    #[test]
    fn tasks_includes_a_task_that_only_exists_in_the_journal() {
        let (dir, mut registry) = registry();
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        let id = registry.register(&repo, None).unwrap();
        let journal_only = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        journal_record(&repo, &journal_only);

        let listed = registry.tasks(dir.path()).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].1, journal_only);
        assert_eq!(listed[0].0.id, id);
        // The path is named for the caller even though the directory is absent: creating it is
        // scheduling's job, not discovery's.
        assert!(listed[0].2.ends_with(journal_only.as_str()));
        assert!(!listed[0].2.exists());
    }

    #[test]
    fn tasks_lists_a_task_with_a_directory_and_a_journal_only_once() {
        // Both traces describe the same task; a union that did not deduplicate would make
        // `status` print it twice and a human would think two runs were going.
        let (dir, mut registry) = registry();
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        let id = registry.register(&repo, None).unwrap();
        let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let project = registry.by_id(&id).unwrap();
        std::fs::create_dir_all(project.state_dir(dir.path()).join(task.as_str())).unwrap();
        journal_record(&repo, &task);

        let listed = registry.tasks(dir.path()).unwrap();
        assert_eq!(listed.len(), 1, "{listed:?}");
        assert_eq!(listed[0].1, task);
    }

    #[test]
    fn a_corrupt_registry_is_fatal_rather_than_empty() {
        let (dir, registry) = registry();
        std::fs::write(registry.path(), "{not json").unwrap();
        let error = Registry::open(registry.path()).unwrap_err();
        assert!(error.to_string().contains("not a valid registry"), "{error}");
        let _ = dir;
    }

    #[test]
    fn a_newer_schema_is_refused_by_version() {
        let (dir, registry) = registry();
        std::fs::write(registry.path(), r#"{"v": 99, "projects": []}"#).unwrap();
        let error = Registry::open(registry.path()).unwrap_err();
        assert!(error.to_string().contains("version"), "{error}");
        let _ = dir;
    }

    #[test]
    fn the_file_is_indented_and_ends_with_a_newline() {
        let (dir, mut registry) = registry();
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        registry.register(&repo, None).unwrap();
        let text = std::fs::read_to_string(registry.path()).unwrap();
        assert!(text.contains("\"v\": 1"), "{text}");
        assert!(text.contains('\n'));
        assert!(text.ends_with('\n'), "{text}");
        // No `remote` key at all when there is none: absent, not null.
        assert!(!text.contains("null"), "{text}");
    }
}
