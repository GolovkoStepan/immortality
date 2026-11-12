//! Canonical filesystem locations.
//!
//! Two kinds of state with different lifetimes:
//!
//! * **Durable, human-readable, in the repository** (`.agent/`): plan, acceptance criteria,
//!   task tree, decisions, working memory, journal. Versioned, reviewable, survives moving
//!   the project. This is the source of truth.
//! * **Machine-local, outside the repository** (`~/Library/Application Support/immortality`):
//!   transcripts, content-addressed artefacts, indexes, logs, daemon socket. Lossy by design,
//!   rebuildable from the journal.
//!
//! Nothing under `.agent/` may be regenerated silently: the journal is the audit trail.

use std::path::{Path, PathBuf};

/// Directory inside the repository holding durable task state.
pub const AGENT_DIR: &str = ".agent";

/// In-repository paths. Pure functions so tests do not need a filesystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoPaths {
    root: PathBuf,
}

impl RepoPaths {
    pub fn new(repo_root: impl Into<PathBuf>) -> Self {
        Self {
            root: repo_root.into(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn agent_dir(&self) -> PathBuf {
        self.root.join(AGENT_DIR)
    }

    pub fn plan(&self) -> PathBuf {
        self.agent_dir().join("PLAN.md")
    }

    pub fn acceptance(&self) -> PathBuf {
        self.agent_dir().join("ACCEPTANCE.md")
    }

    pub fn tasks(&self) -> PathBuf {
        self.agent_dir().join("TASKS.json")
    }

    pub fn decisions(&self) -> PathBuf {
        self.agent_dir().join("DECISIONS.md")
    }

    pub fn notes(&self) -> PathBuf {
        self.agent_dir().join("NOTES.md")
    }

    pub fn journal(&self) -> PathBuf {
        self.agent_dir().join("JOURNAL.jsonl")
    }

    pub fn project_config(&self) -> PathBuf {
        self.agent_dir().join("config.toml")
    }

    /// Files that must never be committed with secrets or overwritten wholesale.
    pub fn protected(&self) -> Vec<PathBuf> {
        vec![
            self.root.join(".git").join("config"),
            self.agent_dir().join("config.toml"),
        ]
    }
}

/// Machine-local paths under the application data directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatePaths {
    root: PathBuf,
}

impl StatePaths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Default location. `IMMORTALITY_STATE_DIR` wins, which keeps tests and CI hermetic.
    ///
    /// The path is computed here rather than through the `directories` crate: that crate
    /// returns `dev.immortality.immortality` on macOS (not the name in the docs) and hands
    /// back the *same* directory for config and data, while pulling in an MPL-2.0
    /// dependency for what is three `join` calls on a macOS-only tool.
    pub fn discover() -> Self {
        if let Some(dir) = std::env::var_os("IMMORTALITY_STATE_DIR") {
            return Self::new(dir);
        }
        Self::new(data_dir())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn global_config(&self) -> PathBuf {
        config_dir().join("config.toml")
    }

    pub fn daemon_lock(&self) -> PathBuf {
        self.root.join("daemon.lock")
    }

    pub fn daemon_socket(&self) -> PathBuf {
        self.root.join("daemon.sock")
    }

    pub fn daemon_token(&self) -> PathBuf {
        self.root.join("daemon.token")
    }

    pub fn index_db(&self) -> PathBuf {
        self.root.join("index.db")
    }

    pub fn registry(&self) -> PathBuf {
        self.root.join("registry.json")
    }

    pub fn task_dir(&self, project: &str, task: &str) -> PathBuf {
        self.root.join("state").join(project).join(task)
    }

    pub fn artifacts_dir(&self) -> PathBuf {
        self.root.join("artifacts")
    }

    pub fn logs_dir(&self) -> PathBuf {
        self.root.join("logs")
    }

    pub fn transcripts_dir(&self, project: &str, task: &str) -> PathBuf {
        self.task_dir(project, task).join("transcripts")
    }
}

/// macOS application support directory for this user.
///
/// `$HOME` is used rather than the `directories` crate so the path matches what the
/// documentation promises and needs no extra dependency.
fn data_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("/nonexistent"), PathBuf::from);
    home.join("Library")
        .join("Application Support")
        .join("immortality")
}

fn config_dir() -> PathBuf {
    data_dir()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_paths_live_under_agent_dir() {
        let paths = RepoPaths::new("/tmp/project");
        assert_eq!(paths.agent_dir(), PathBuf::from("/tmp/project/.agent"));
        assert_eq!(
            paths.journal(),
            PathBuf::from("/tmp/project/.agent/JOURNAL.jsonl")
        );
        assert_eq!(paths.tasks(), PathBuf::from("/tmp/project/.agent/TASKS.json"));
        for file in [paths.plan(), paths.acceptance(), paths.notes(), paths.decisions()] {
            assert!(file.starts_with(paths.agent_dir()), "{file:?}");
        }
    }

    #[test]
    fn protected_files_include_git_config() {
        let paths = RepoPaths::new("/tmp/project");
        let protected = paths.protected();
        assert!(protected.iter().any(|p| p.ends_with(".git/config")));
        assert!(protected.iter().any(|p| p.ends_with(".agent/config.toml")));
    }

    #[test]
    fn state_dir_is_overridable_by_env() {
        // `discover` reads the environment, so check the override path directly to keep
        // the test free of global state.
        let paths = StatePaths::new("/tmp/state");
        assert_eq!(paths.root(), Path::new("/tmp/state"));
        assert_eq!(paths.daemon_socket(), PathBuf::from("/tmp/state/daemon.sock"));
        assert_eq!(paths.artifacts_dir(), PathBuf::from("/tmp/state/artifacts"));
    }

    #[test]
    fn default_state_dir_is_under_application_support() {
        // Not the `directories` crate's `dev.immortality.immortality`: the documented path
        // is what users are told to edit, so it is what the code must produce.
        let dir = data_dir();
        assert!(dir.ends_with("Application Support/immortality"), "{dir:?}");
        assert_eq!(config_dir(), dir);
    }

    #[test]
    fn per_task_dirs_are_nested() {
        let paths = StatePaths::new("/tmp/state");
        let dir = paths.task_dir("proj", "tsk_01");
        assert_eq!(dir, PathBuf::from("/tmp/state/state/proj/tsk_01"));
        assert_eq!(
            paths.transcripts_dir("proj", "tsk_01"),
            PathBuf::from("/tmp/state/state/proj/tsk_01/transcripts")
        );
    }
}
