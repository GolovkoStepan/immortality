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

/// Permissions on a state directory: owner only.
///
/// The state root holds the daemon token, the journals' artefacts and the registry, none of which
/// another local account has any business reading. `0o700` rather than `0o600` because this is a
/// directory: the execute bit is what lets the owner traverse it at all.
pub const STATE_DIR_MODE: u32 = 0o700;

/// Create `dir` and every missing parent, owner-only.
///
/// # Why this is not `create_dir_all`
///
/// `create_dir_all` creates with the process umask, and on a typical macOS or Linux box that is
/// `0o755`: every local account can read the daemon token and every journal artefact. The mode
/// therefore has to be set explicitly, and only for the directories this call creates — a mode
/// applied to an *existing* directory would silently tighten something the user may have shared
/// on purpose, and changing permissions on a directory the daemon did not create is a surprise,
/// not a protection.
///
/// # Errors
/// The directory cannot be created, or its permissions cannot be set on a platform where they
/// matter.
pub fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    if dir.is_dir() {
        return Ok(());
    }
    std::fs::create_dir_all(dir)?;
    set_private_dir_mode(dir)
}

/// Set `0o700` on a directory this process owns.
///
/// Split from [`create_private_dir`] so a caller that created the directory by other means — the
/// socket's parent in `bind`, which calls `create_dir_all` itself — can still apply the mode
/// deliberately rather than by inheriting a umask.
///
/// A no-op off Unix, where the mode does not exist. The state root is documented as
/// machine-local, and every supported platform today is Unix.
///
/// # Errors
/// The permissions cannot be set, which leaves a directory readable that should not be.
pub fn set_private_dir_mode(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(STATE_DIR_MODE))?;
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
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

    #[cfg(unix)]
    fn mode(path: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::metadata(path).expect("metadata").permissions().mode() & 0o777
    }

    #[cfg(unix)]
    #[test]
    fn a_state_directory_this_call_creates_is_owner_only() {
        // The measured hole: `create_dir_all` uses the umask, which is `0o755` on a default
        // macOS account. The token, the registry and every artefact then sit in a directory any
        // local user can traverse. The mode is asserted, not the umask, because the umask is the
        // input and this is the answer.
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("state");
        create_private_dir(&root).expect("create");
        assert_eq!(mode(&root), STATE_DIR_MODE);
        assert_eq!(STATE_DIR_MODE, 0o700);
    }

    #[cfg(unix)]
    #[test]
    fn an_existing_state_directory_keeps_the_mode_the_user_chose() {
        // The other half of the decision: the daemon changes only what it creates. Silently
        // tightening a directory the user may have shared on purpose is a surprise, and on a
        // migration it would rewrite permissions on data the daemon did not make.
        use std::os::unix::fs::PermissionsExt as _;
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("state");
        std::fs::create_dir_all(&root).expect("create");
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o750)).expect("chmod");
        create_private_dir(&root).expect("no-op");
        assert_eq!(mode(&root), 0o750, "an existing directory was rewritten");
    }

    #[cfg(unix)]
    #[test]
    fn creating_a_private_directory_also_creates_the_parents() {
        // The state root is `~/Library/Application Support/immortality`, so on a machine where
        // the application support directory does not exist yet this call is the one that makes
        // it. Refusing because a parent is missing would break a first run.
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("a").join("b").join("c");
        create_private_dir(&root).expect("create");
        assert!(root.is_dir());
        assert_eq!(mode(&root), STATE_DIR_MODE);
    }
}
