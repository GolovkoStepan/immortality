//! `immortality serve`: running the daemon in the foreground.
//!
//! Everything this needs already exists — [`server::bind`], [`ServerState::discover`],
//! [`server::serve`]. What this module adds is the three things a *process* needs that a test
//! does not:
//!
//! 1. **A token.** The daemon refuses to start without `daemon.token`, on purpose: a server with
//!    no token would accept every client. But a human running `serve` for the first time has no
//!    token, and telling them to invent one by hand is a trap — the file must be mode `0600`,
//!    non-empty, and in the right directory. So `serve` mints one on first start and never
//!    touches an existing one, because replacing it would lock out every already-running client.
//!
//! 2. **A single instance.** Two daemons on one state root would both accept clients and both
//!    append to the journal, which is the one thing this project cannot survive. `daemon.lock`
//!    is held with `std::fs::TryLock`, so the second start fails with a message naming the
//!    holder rather than silently racing. No new dependency: `try_lock` is in std since 1.89.
//!
//! 3. **A clean exit.** `kill` and Ctrl-C remove the socket, so the next start is not told
//!    "nothing is accepting connections on it". A socket left by `kill -9` is already handled by
//!    [`server::bind`], which removes a stale file before binding.
//!
//! # Why the lock is a file rather than the socket
//!
//! Binding the socket is *already* exclusive, so the second daemon would fail on `bind`. But it
//! would fail while `server::bind` has already deleted the first daemon's socket — taking the
//! running daemon's clients down with it. Checking the lock first means the second start never
//! touches the socket of a live daemon.

// Same reason as `render`: in a binary crate a private module's `pub` items trip
// `unreachable_pub` while `pub(crate)` trips `redundant_pub_crate`, and the pair is unsatisfiable.
#![allow(unreachable_pub)]

use std::io::Write as _;
use std::path::Path;

use anyhow::{Context as _, Result, bail};

use immortality_core::ids::Ulid;
use immortality_core::paths::StatePaths;
use immortality_daemon::server::{self, Events, ServerState};

mod lock {
    //! The single-instance lock, kept apart so the guard type is impossible to misplace.
    //!
    //! The file is held open for the lifetime of the value; dropping it releases the lock and
    //! removes the file. Both matter: a lock file left behind with no process holding it is
    //! indistinguishable from a crashed one, and the next start would have to guess.

    use std::fs::File;
    use std::path::{Path, PathBuf};

    use anyhow::{Context as _, Result, bail};

    /// A held lock on `daemon.lock`.
    #[derive(Debug)]
    pub struct InstanceLock {
        file: File,
        path: PathBuf,
    }

    impl InstanceLock {
        /// Take the lock, or report who holds it.
        ///
        /// # Errors
        /// Another daemon holds it, or the file cannot be opened.
        pub fn acquire(path: &Path) -> Result<Self> {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("cannot create {}", parent.display()))?;
            }
            let file = std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .open(path)
                .with_context(|| format!("cannot open {}", path.display()))?;
            // `try_lock` rather than `lock`: a second daemon should fail immediately with a
            // useful message, not block forever waiting for the first one to exit. The lock is
            // advisory but the only writers are our own processes.
            match file.try_lock() {
                Ok(()) => {}
                Err(std::fs::TryLockError::WouldBlock) => {
                    let holder = std::fs::read_to_string(path).unwrap_or_default();
                    bail!(
                        "another daemon is already running on this state directory.\n\
                         It holds {} (pid {}).\n\
                         Attach to it with `immortality attach`, or stop it first.",
                        path.display(),
                        if holder.trim().is_empty() {
                            "unknown"
                        } else {
                            holder.trim()
                        }
                    );
                }
                Err(std::fs::TryLockError::Error(error)) => {
                    return Err(error).with_context(|| format!("cannot lock {}", path.display()));
                }
            }
            Ok(Self {
                file,
                path: path.to_path_buf(),
            })
        }
    }

    impl Drop for InstanceLock {
        fn drop(&mut self) {
            // Best effort: the OS releases the lock when the file closes either way, and the
            // file is removed so a human looking at the directory does not see a phantom.
            let _ = std::fs::remove_file(&self.path);
            // Keeping the handle alive until here is what holds the lock; this is explicit so
            // the intent survives a refactor that might otherwise drop it early.
            let _ = &self.file;
        }
    }
}

pub use lock::InstanceLock;

/// Make sure a token exists, minting one on first start.
///
/// A token that already exists is never rewritten: a client that read it keeps working, and a
/// daemon restart must not lock out every terminal attached to the previous one.
///
/// # Errors
/// The token file exists but cannot be read, or a new one cannot be written.
pub fn ensure_token(path: &Path) -> Result<()> {
    match std::fs::read_to_string(path) {
        Ok(text) if !text.trim().is_empty() => return Ok(()),
        // An existing-but-empty token is a broken state, not a fresh one: overwriting silently
        // would hide that something truncated it.
        Ok(_) => bail!(
            "{} exists but is empty. A daemon with no token accepts every client.\n\
             Delete it to have a new one generated, or write one yourself.",
            path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("cannot read {}", path.display()));
        }
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
    }
    // 256 bits from two identifiers: a ULID carries 80 random bits, and this is a secret that
    // guards a local socket, not a password. Mode 0600 from the moment it exists.
    let token = format!("{}{}", Ulid::now(), Ulid::now());
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("cannot create {}", path.display()))?;
    file.write_all(token.as_bytes())
        .with_context(|| format!("cannot write {}", path.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("cannot write {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("cannot sync {}", path.display()))?;
    Ok(())
}

/// Run the daemon until the process is killed.
///
/// # Errors
/// Another daemon holds the lock, the token cannot be created or read, or the socket cannot be
/// bound.
pub async fn run() -> Result<()> {
    let state = StatePaths::discover();
    let root = state.root().to_path_buf();

    let lock = InstanceLock::acquire(&state.daemon_lock())?;
    // The lock file records the pid for the "another daemon is running" message. Written after
    // the lock, so a losing daemon never overwrites the winner's pid.
    if let Err(error) = std::fs::write(state.daemon_lock(), format!("{}\n", std::process::id())) {
        tracing::warn!(
            "cannot record the pid in {}: {error}",
            state.daemon_lock().display()
        );
    }

    ensure_token(&state.daemon_token())?;
    let server_state = ServerState::discover(&root)
        .map_err(|error| anyhow::anyhow!("{error}"))
        .context("the daemon cannot start")?;

    let socket = state.daemon_socket();
    let listener = server::bind(&socket).map_err(|error| anyhow::anyhow!("{error}"))?;
    // Printed to stderr and not stdout: this line is diagnostics, and stdout is reserved for
    // machine-readable output like every other command.
    eprintln!(
        "immortality daemon listening on {}\n\
         state root: {}\n\
         stop with Ctrl-C or `kill {}`",
        socket.display(),
        root.display(),
        std::process::id()
    );

    let outcome = server::serve(listener, server_state, Events::new()).await;
    // `serve` returns only on an accept error; the socket is removed so the next start does not
    // find a corpse. `bind` also removes a stale file, so this is for the common case only.
    let _ = std::fs::remove_file(&socket);
    drop(lock);
    outcome.context("the daemon stopped")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_token_is_minted_once_and_kept_on_the_next_start() {
        // The restart case is the one that matters: replacing the token would lock out every
        // client that read the old one.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("daemon.token");

        ensure_token(&path).expect("first start");
        let first = std::fs::read_to_string(&path).expect("token");
        assert_ne!(first.trim(), "", "a minted token must not be empty");
        assert!(first.ends_with('\n'), "a trailing newline matches the fixture");

        ensure_token(&path).expect("second start");
        assert_eq!(std::fs::read_to_string(&path).expect("token"), first);
    }

    #[cfg(unix)]
    #[test]
    fn a_minted_token_is_readable_only_by_its_owner() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("daemon.token");
        ensure_token(&path).expect("mint");

        let mode = std::fs::metadata(&path).expect("metadata").permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "token mode {mode:o}");
    }

    #[test]
    fn an_empty_token_is_an_error_rather_than_silently_regenerated() {
        // A truncated token means something went wrong; overwriting it would hide that and
        // quietly invalidate clients.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("daemon.token");
        std::fs::write(&path, "").expect("write");

        let error = ensure_token(&path).expect_err("empty token must be refused");
        assert!(error.to_string().contains("empty"), "{error}");
        assert_eq!(std::fs::read_to_string(&path).expect("token"), "");
    }

    #[test]
    fn the_lock_refuses_a_second_instance_and_names_the_pid() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("daemon.lock");

        let first = InstanceLock::acquire(&path).expect("first");
        std::fs::write(&path, "4242\n").expect("pid");
        let error = InstanceLock::acquire(&path).expect_err("second must be refused");
        let text = error.to_string();
        assert!(text.contains("already running"), "{text}");
        assert!(text.contains("4242"), "the message names the holder: {text}");

        drop(first);
        // The lock is advisory and released on drop, so a fresh start succeeds.
        InstanceLock::acquire(&path).expect("after the first exits");
    }

    #[test]
    fn the_lock_file_is_removed_when_the_daemon_exits() {
        // A lock file left behind looks like a crashed daemon to a human reading the directory.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("daemon.lock");
        {
            let _lock = InstanceLock::acquire(&path).expect("lock");
            assert!(path.exists());
        }
        assert!(!path.exists(), "the file is gone after the guard drops");
    }

    #[test]
    fn the_lock_works_from_a_directory_that_does_not_exist_yet() {
        // The state root is created lazily by every other command; `serve` must not be the one
        // command that requires it to exist first.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("deep").join("nested").join("daemon.lock");
        // The guard is bound, not discarded: an unbound temporary is dropped at the end of the
        // statement, which releases the lock and removes the file — correct behaviour, but it
        // would make this test assert about a file that no longer exists.
        let _lock = InstanceLock::acquire(&path).expect("lock in a fresh tree");
        assert!(path.exists());
    }
}
