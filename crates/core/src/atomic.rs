//! Whole-file writes that survive a crash.
//!
//! # Why this exists
//!
//! Two files in this system must be either entirely old or entirely new, never a mix: the
//! escalation queue, which holds the only state the documentation says cannot be lost, and the
//! per-task state snapshots under `~/Library/Application Support`. A `File::create` followed by
//! a `write_all` has a window in which the file is empty or half-written, and a crash in that
//! window destroys the previous contents too.
//!
//! The sequence here is the standard one: write a sibling temp file, `fsync` it, `rename` it
//! over the target, then `fsync` the directory so the rename itself is durable. A crash anywhere
//! before the rename leaves the old file untouched; a crash after it leaves the new file
//! complete. The only cost is one leftover temp file, which [`sweep_temps`] removes.
//!
//! # Why the temp name is a ULID
//!
//! A pid-based name can collide with a temp file left behind by a *previous* process with the
//! same pid, and the collision is silent: the second writer would open the first one's partial
//! bytes. A ULID is unique across processes and across restarts of the same pid.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::ids::Ulid;

/// Writing to a file while it is being read is how a concurrent reader sees a partial document.
/// Best effort: a filesystem that refuses to sync a directory still has a correct file, because
/// the data was synced before the rename.
fn sync_dir(dir: &Path) {
    if let Ok(handle) = std::fs::File::open(dir) {
        let _ = handle.sync_all();
    }
}

/// Write `bytes` to `path` atomically and durably.
///
/// # Errors
/// The parent directory could not be created, or the write, sync or rename failed. The original
/// file is intact in every one of those cases.
pub fn write_atomically(path: &Path, bytes: &[u8]) -> crate::Result<()> {
    if let Some(directory) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
        std::fs::create_dir_all(directory).map_err(|e| crate::Error::io(directory, e))?;
    }

    let temp = temp_path(path);
    let result = (|| -> std::io::Result<()> {
        let mut file = std::fs::File::create(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        // Keep the target's permissions (a state file may be 0600) rather than inheriting the
        // temp file's default. Failure is not fatal: the content is what matters.
        if let Ok(metadata) = std::fs::metadata(path) {
            let _ = std::fs::set_permissions(&temp, metadata.permissions());
        }
        std::fs::rename(&temp, path)
    })();

    if let Err(error) = result {
        let _ = std::fs::remove_file(&temp);
        return Err(crate::Error::io(path, error));
    }
    if let Some(directory) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
        sync_dir(directory);
    }
    Ok(())
}

/// Remove temp files this module left behind in `dir`.
///
/// Called by `gc`. Returns how many were removed. A temp file is only ours if its name carries
/// the marker, so a user's `.foo.tmp` is never touched.
///
/// # Errors
/// The directory could not be listed.
pub fn sweep_temps(dir: &Path) -> crate::Result<usize> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(crate::Error::io(dir, error)),
    };
    let mut removed = 0;
    for entry in entries {
        let entry = entry.map_err(|e| crate::Error::io(dir, e))?;
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if name.contains(TEMP_MARKER) && std::fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        }
    }
    Ok(removed)
}

/// Marks a temp file as this module's, so [`sweep_temps`] can recognise it.
const TEMP_MARKER: &str = ".immortality-tmp-";

/// A sibling temp path: same directory, so the rename never crosses a filesystem.
fn temp_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let temp_name = format!(".{name}{TEMP_MARKER}{}", Ulid::now());
    match path.parent() {
        Some(parent) => parent.join(temp_name),
        None => PathBuf::from(temp_name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_write_replaces_the_whole_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.json");
        std::fs::write(&path, "old and considerably longer than the new body").unwrap();
        write_atomically(&path, b"new").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
    }

    #[test]
    fn a_write_creates_missing_directories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/deeper/file.json");
        write_atomically(&path, b"{}").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"{}");
    }

    #[test]
    fn no_temp_file_is_left_behind() {
        // A leftover temp file is not just clutter: `gc` uses its presence to decide a write
        // crashed, and a successful write must not look like one.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.json");
        write_atomically(&path, b"{}").unwrap();
        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["file.json".to_string()], "{names:?}");
    }

    #[test]
    fn the_targets_permissions_are_preserved() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secret.json");
        std::fs::write(&path, b"old").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        write_atomically(&path, b"new").unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "{mode:o}");
    }

    #[test]
    fn sweeping_removes_only_our_temps() {
        let dir = tempfile::tempdir().unwrap();
        let ours = dir
            .path()
            .join(format!(".state.json{TEMP_MARKER}01ARZ3NDEKTSV4RRFFQ69G5FAV"));
        let theirs = dir.path().join(".state.json.tmp");
        let real = dir.path().join("state.json");
        for path in [&ours, &theirs, &real] {
            std::fs::write(path, b"x").unwrap();
        }
        assert_eq!(sweep_temps(dir.path()).unwrap(), 1);
        assert!(!ours.exists());
        assert!(theirs.exists(), "a user's own temp file must survive");
        assert!(real.exists());
    }

    #[test]
    fn sweeping_a_missing_directory_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(sweep_temps(&dir.path().join("not-there")).unwrap(), 0);
    }

    #[test]
    fn a_failed_write_leaves_the_old_file_intact() {
        // A directory where the file should be: the rename fails, and the previous contents
        // must still be readable. This is the property the whole module exists for.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("inner"), b"keep me").unwrap();
        assert!(write_atomically(&path, b"replacement").is_err());
        assert_eq!(std::fs::read(path.join("inner")).unwrap(), b"keep me".to_vec());
    }
}
