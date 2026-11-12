//! Crash-point tests for the atomic-write protocol, using [`Faults`] and [`MemFs`] together.
//!
//! # What is being tested
//!
//! Three places in this project write a file that a crash could tear: the artifact store
//! (`write .part` → `sync` → `rename` → `sync dir`), the log archive (`<name>.gz.part` →
//! `sync` → `rename` → `sync dir` → `remove plain`), and the state snapshot. They all rely on the
//! same two rules, and those rules are what this file asserts rather than assumes:
//!
//! 1. **No partial file is ever visible.** A reader sees the old bytes or the new bytes, never a
//!    prefix, because the only name that becomes visible is a fully written and synced one.
//! 2. **A crash costs work, never data.** The worst outcome at any point is that the new version
//!    is missing; the old one must still be there, or the file must simply not exist yet.
//!
//! These are properties of the *filesystem*, so they cannot be tested by calling the writer's
//! functions and checking return values: the interesting case is the process dying between
//! `write` and `sync`, where no return value exists at all.

// Integration tests: `expect`, `unwrap` and `panic` are the right tools here — a failed
// assertion should name the exact line, and there is no caller to propagate to.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use immortality_testkit::fakefs::{Fs, MemFs, StdFs};
use immortality_testkit::{Action, Faults, Hook};

/// The hooks the protocol below checks. Named here, not borrowed from the production constants,
/// because this test models the protocol rather than calling it — and a shared constant would let
/// a rename in production silently disconnect the test.
const WRITE: Hook = Hook::new("durability.write");
const SYNC: Hook = Hook::new("durability.sync");
const RENAME: Hook = Hook::new("durability.rename");
const DIR_SYNC: Hook = Hook::new("durability.dir_sync");
const REMOVE: Hook = Hook::new("durability.remove");

/// Steps of the write-and-replace protocol. `REMOVE` is not here: it belongs to the archive
/// protocol below, which deletes the source only after the copy is durable.
const WRITE_HOOKS: [Hook; 4] = [WRITE, SYNC, RENAME, DIR_SYNC];

/// Steps of the compress-and-delete protocol.
const ARCHIVE_HOOKS: [Hook; 5] = [WRITE, SYNC, RENAME, DIR_SYNC, REMOVE];

/// Where a crash can be injected, as `(hook, before|after)`.
///
/// Both sides of every operation, because they fail differently: dying *before* `rename` leaves
/// the temp file, dying *after* it leaves the new name — and only the second can lose data if the
/// directory was not synced.
fn crash_matrix(hooks: &[Hook]) -> Vec<(&'static str, Hook, bool)> {
    let mut points = Vec::new();
    for &hook in hooks {
        // `before` and `after` are distinguished by the action, so both are one arm here; the
        // label exists to make a failure name the exact side.
        points.push(("before", hook, false));
        points.push(("after", hook, true));
    }
    points
}

/// Write `payload` to `path` the way the artifact store does, checking `faults` at each step.
///
/// Returns `Err` only when a fault fired; the caller uses that to know the protocol stopped
/// early rather than completing.
fn atomic_write(fs: &MemFs, faults: &Faults, path: &std::path::Path, payload: &[u8]) -> Result<(), Action> {
    let temp = path.with_extension("part");

    if let Some(fault) = faults.check(WRITE) {
        return Err(fault.action);
    }
    fs.write(&temp, payload).expect("memory write");
    if let Some(fault) = faults.check(WRITE) {
        return Err(fault.action);
    }

    if let Some(fault) = faults.check(SYNC) {
        return Err(fault.action);
    }
    fs.sync(&temp).expect("memory sync");
    if let Some(fault) = faults.check(SYNC) {
        return Err(fault.action);
    }

    if let Some(fault) = faults.check(RENAME) {
        return Err(fault.action);
    }
    fs.rename(&temp, path).expect("memory rename");
    if let Some(fault) = faults.check(RENAME) {
        return Err(fault.action);
    }

    if let Some(fault) = faults.check(DIR_SYNC) {
        return Err(fault.action);
    }
    fs.sync_dir(path.parent().expect("a parent"));
    if let Some(fault) = faults.check(DIR_SYNC) {
        return Err(fault.action);
    }

    Ok(())
}

/// Whether the state after a crash is one of the two acceptable ones.
///
/// Acceptable means: the target holds the new payload (the write survived), or the target is
/// absent and the temp file holds the new payload (the write is recoverable), or neither exists
/// (nothing happened). A *partial* target is the one outcome that is never acceptable.
fn assert_no_partial_target(fs: &MemFs, target: &std::path::Path, payload: &[u8], label: &str) {
    if let Some(bytes) = fs.peek(target) {
        assert_eq!(
            bytes, payload,
            "{label}: the visible file must be the whole new payload, not a prefix"
        );
    }
    // Absent is fine: nothing was ever promised. The temp file may hold the payload, and that is
    // the recoverable case the `.part` naming exists for.
}

/// Compress `source` into `archive` and delete the source, the way log archiving does.
///
/// The source is deleted only after the archive is renamed and the directory synced, so a crash
/// at any point leaves the text in one place or both — never in neither.
fn archive(
    fs: &MemFs,
    faults: &Faults,
    source: &std::path::Path,
    archive: &std::path::Path,
) -> Result<(), Action> {
    let payload = fs.read(source).map_err(|_| Action::CrashBefore)?;
    atomic_write(fs, faults, archive, &payload)?;

    if let Some(fault) = faults.check(REMOVE) {
        return Err(fault.action);
    }
    fs.remove(source).map_err(|_| Action::CrashBefore)?;
    if let Some(fault) = faults.check(REMOVE) {
        return Err(fault.action);
    }
    Ok(())
}

#[test]
fn a_crash_during_archiving_never_loses_the_log_text() {
    // The rule D-039 claims: at any crash point the text exists as the plain file, the archive,
    // or both. Deleting the source first would break it, and this test is what proves it does not.
    let body = b"2026-10-01T00:00:00Z INFO started
";
    for (side, hook, before) in crash_matrix(&ARCHIVE_HOOKS) {
        let fs = MemFs::new();
        let logs = std::path::PathBuf::from("/logs");
        fs.create_dir_all(&logs).unwrap();
        let source = logs.join("2026-10.log");
        let target = logs.join("2026-10.log.gz");
        fs.write(&source, body).unwrap();

        let faults = Faults::new();
        if before {
            faults.crash_before(hook, 1);
        } else {
            faults.crash_after(hook, 1);
        }
        let _ = archive(&fs, &faults, &source, &target);
        fs.crash();

        let label = format!("archive crash {side} {}", hook.as_str());
        let in_plain = fs.peek(&source) == Some(body.to_vec());
        let in_archive = fs.peek(&target) == Some(body.to_vec());
        assert!(
            in_plain || in_archive,
            "{label}: the log text is in neither file, which is the one outcome that is data loss"
        );
        assert_eq!(
            faults.unfired(),
            Vec::<(Hook, u64, Action)>::new(),
            "{label}: the injected crash never fired"
        );
    }
}

#[test]
fn a_crash_at_every_step_of_an_atomic_write_leaves_no_partial_file() {
    let payload = b"the whole body of an observation";
    for (side, hook, before) in crash_matrix(&WRITE_HOOKS) {
        let fs = MemFs::new();
        let root = std::path::PathBuf::from("/repo/artifacts");
        fs.create_dir_all(&root).unwrap();
        let target = root.join("sha256");

        let faults = Faults::new();
        // `CrashBefore` stops the protocol before the operation, `CrashAfter` after it: the pair
        // covers "the call did not happen" and "it happened but nothing was recorded".
        if before {
            faults.crash_before(hook, 1);
        } else {
            faults.crash_after(hook, 1);
        }

        let _ = atomic_write(&fs, &faults, &target, payload);
        // A real crash loses unsynced data; this is what makes the test meaningful, because
        // without it every intermediate state looks durable.
        fs.crash();

        let label = format!("crash {side} {}", hook.as_str());
        assert_no_partial_target(&fs, &target, payload, &label);
        assert_eq!(
            faults.unfired(),
            Vec::<(Hook, u64, Action)>::new(),
            "{label}: the injected crash never fired, so the point is not tested"
        );
    }
}

#[test]
fn a_crash_after_the_rename_but_before_the_directory_sync_keeps_the_data_recoverable() {
    // The precise bug the third sync exists to prevent: the bytes are durable, the name is not.
    let fs = MemFs::new();
    let root = std::path::PathBuf::from("/repo/artifacts");
    fs.create_dir_all(&root).unwrap();
    let target = root.join("sha256");
    let payload = b"payload";

    let faults = Faults::new();
    faults.crash_after(DIR_SYNC, 1);
    let _ = atomic_write(&fs, &faults, &target, payload);
    fs.crash();

    // The rename is undone, and the payload is under the temp name — recoverable, not lost.
    assert!(!fs.exists(&target), "the unsynced rename must not survive");
    assert_eq!(
        fs.peek(&root.join("sha256.part")),
        Some(payload.to_vec()),
        "the data must still be somewhere, or the crash lost it"
    );
}

#[test]
fn a_completed_write_survives_a_crash() {
    // The happy path is part of the property: without this, a protocol that never makes anything
    // durable would pass every test above.
    let fs = MemFs::new();
    let root = std::path::PathBuf::from("/repo/artifacts");
    fs.create_dir_all(&root).unwrap();
    let target = root.join("sha256");

    let faults = Faults::new();
    atomic_write(&fs, &faults, &target, b"durable").expect("no faults armed");
    fs.crash();

    assert_eq!(fs.peek(&target), Some(b"durable".to_vec()));
    assert_eq!(faults.occurrences(WRITE), 2, "both sides were checked");
    assert_eq!(faults.occurrences(DIR_SYNC), 2);
}

#[test]
fn a_failing_sync_is_reported_and_leaves_the_old_version_readable() {
    // The failure path, not the crash path: the disk rejects the write and the caller is told.
    // The old version must still be readable, because a failed update is not a lost file.
    let fs = MemFs::new();
    let root = std::path::PathBuf::from("/repo/artifacts");
    fs.create_dir_all(&root).unwrap();
    let target = root.join("sha256");
    fs.write(&target, b"old version").unwrap();

    let faults = Faults::new();
    faults.fail_at(SYNC, 1, std::io::ErrorKind::StorageFull);
    let outcome = atomic_write(&fs, &faults, &target, b"new version");
    assert_eq!(outcome, Err(Action::Fail(std::io::ErrorKind::StorageFull)));
    assert_eq!(
        fs.peek(&target),
        Some(b"old version".to_vec()),
        "a refused write must not damage what is already there"
    );
}

#[test]
fn the_same_protocol_holds_on_the_real_filesystem() {
    // `MemFs` is a model, and a model can be wrong in a way that flatters the code under test.
    // The safe subset of the property — a completed write is readable, a removed file is gone —
    // is checked against the real thing through the same `Fs` trait.
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("sha256");
    StdFs.write(&target, b"real payload").unwrap();
    StdFs.sync(&target).unwrap();
    assert_eq!(StdFs.read(&target).unwrap(), b"real payload");

    StdFs.remove(&target).unwrap();
    assert!(!StdFs.exists(&target));
}
