//! A filesystem abstraction that can fail on purpose.
//!
//! # Why not just `std::fs`
//!
//! Every interesting failure in this project is an I/O failure at a particular moment: the disk
//! fills exactly between the write and the `fsync`, the rename succeeds and the process dies
//! before it can record that it did. Testing those against the real filesystem means either
//! filling a real disk or racing a real process, and neither is something to put in CI.
//!
//! # Two implementations, one trait
//!
//! [`StdFs`] is the real thing and exists so that production call sites can be written against
//! the trait without a `#[cfg(test)]` fork. [`MemFs`] is an in-memory tree with the same
//! semantics for the operations this project uses: `create`, append, `sync`, `rename`, remove.
//! It is deliberately *not* a general filesystem — no permissions, no symlinks, no partial
//! writes — because every behaviour it adds is behaviour the tests would then be testing.
//!
//! # Crash semantics
//!
//! `MemFs` models what survives a crash, not just what a call returns. [`MemFs::crash`] discards
//! unsynced data and keeps what was synced; files that were renamed but whose directory was not
//! synced lose the rename. That is what makes "the journal is durable before the model sees the
//! result" testable rather than assumed.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The filesystem operations this project depends on.
///
/// Kept small on purpose: `read_dir` is here because `gc` needs it, but `metadata` returns only
/// the two fields the code actually uses, since a full `Metadata` would have to be faked.
pub trait Fs: std::fmt::Debug + Send + Sync {
    /// Read a whole file.
    fn read(&self, path: &Path) -> std::io::Result<Vec<u8>>;

    /// Write a file, replacing it, and sync it.
    ///
    /// Replace-and-sync rather than create-and-append because that is the atomic-write pattern
    /// the artifact store relies on; a `MemFs` that only appended would not catch a real
    /// implementation that forgot to truncate.
    fn write(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()>;

    /// Append to a file, creating it if needed, without syncing.
    fn append(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()>;

    /// Flush a file's data to durable storage.
    fn sync(&self, path: &Path) -> std::io::Result<()>;

    /// Rename, replacing the destination.
    fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()>;

    /// Remove a file. Removing a missing file is an error, as it is in the standard library.
    fn remove(&self, path: &Path) -> std::io::Result<()>;

    /// Create a directory and its parents.
    fn create_dir_all(&self, path: &Path) -> std::io::Result<()>;

    /// Entries directly inside `path`, as full paths.
    fn read_dir(&self, path: &Path) -> std::io::Result<Vec<PathBuf>>;

    /// File size in bytes.
    fn len(&self, path: &Path) -> std::io::Result<u64>;

    /// Whether the path exists at all.
    fn exists(&self, path: &Path) -> bool;
}

/// The real filesystem.
#[derive(Debug, Clone, Copy, Default)]
pub struct StdFs;

impl Fs for StdFs {
    fn read(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        std::fs::read(path)
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        file.write_all(bytes)?;
        file.sync_all()
    }

    fn append(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        file.write_all(bytes)
    }

    fn sync(&self, path: &Path) -> std::io::Result<()> {
        std::fs::OpenOptions::new().write(true).open(path)?.sync_data()
    }

    fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
        std::fs::rename(from, to)
    }

    fn remove(&self, path: &Path) -> std::io::Result<()> {
        std::fs::remove_file(path)
    }

    fn create_dir_all(&self, path: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(path)
    }

    fn read_dir(&self, path: &Path) -> std::io::Result<Vec<PathBuf>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(path)? {
            out.push(entry?.path());
        }
        Ok(out)
    }

    fn len(&self, path: &Path) -> std::io::Result<u64> {
        Ok(std::fs::metadata(path)?.len())
    }

    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }
}

/// One file's durable and volatile content.
#[derive(Debug, Clone, Default)]
struct FileState {
    /// Bytes that survive a crash.
    synced: Vec<u8>,
    /// Bytes written since the last sync. Lost by [`MemFs::crash`].
    pending: Vec<u8>,
    /// Whether the file's existence itself is durable.
    ///
    /// Separate from `synced` because an empty synced file and a file that was never synced both
    /// have zero durable bytes, and only the first should survive a crash. `write` makes the
    /// file durable; an `append` to a file that did not exist creates it volatile.
    durable: bool,
}

impl FileState {
    fn visible(&self) -> Vec<u8> {
        let mut out = self.synced.clone();
        out.extend_from_slice(&self.pending);
        out
    }
}

#[derive(Debug, Default)]
struct Tree {
    files: BTreeMap<PathBuf, FileState>,
    dirs: std::collections::BTreeSet<PathBuf>,
}

/// An in-memory filesystem with crash semantics.
#[derive(Debug, Default)]
pub struct MemFs {
    tree: Mutex<Tree>,
    /// Renames that have happened but whose directory was not synced.
    ///
    /// A rename is a directory operation: the data is already durable (the artifact was synced
    /// before it), but the *name* is not until the directory is. Losing it on a crash is exactly
    /// the bug that leaves a perfectly good artifact invisible.
    unsynced_renames: Mutex<Vec<(PathBuf, PathBuf)>>,
}

impl MemFs {
    /// An empty filesystem.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop everything that was not synced, as a power cut would.
    ///
    /// Files revert to their last synced content; files created but never synced disappear;
    /// renames whose directory was never synced are undone.
    pub fn crash(&self) {
        {
            let mut tree = self.lock();
            for file in tree.files.values_mut() {
                file.pending.clear();
            }
            // A file survives only if its existence was made durable: `write` does that, a bare
            // `append` to a new path does not.
            tree.files.retain(|_, file| file.durable);
        }
        let renames = std::mem::take(&mut *self.renames());
        for (from, to) in renames.into_iter().rev() {
            let mut tree = self.lock();
            if let Some(file) = tree.files.remove(&to) {
                tree.files.insert(from, file);
            }
        }
    }

    /// Flush pending renames, as syncing a directory would.
    pub fn sync_dir(&self, _dir: &Path) {
        self.renames().clear();
    }

    /// Every path currently present, sorted.
    #[must_use]
    pub fn paths(&self) -> Vec<PathBuf> {
        self.lock().files.keys().cloned().collect()
    }

    /// The visible content of a file, including unsynced writes.
    #[must_use]
    pub fn peek(&self, path: &Path) -> Option<Vec<u8>> {
        self.lock().files.get(path).map(FileState::visible)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Tree> {
        self.tree
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn renames(&self) -> std::sync::MutexGuard<'_, Vec<(PathBuf, PathBuf)>> {
        self.unsynced_renames
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

// The lock is deliberately held for the whole operation: a `MemFs` call is one atomic unit, and
// releasing the guard early would let another thread observe a half-applied write. That is the
// opposite of what `significant_drop_tightening` wants here, so it is allowed for this impl.
#[allow(clippy::significant_drop_tightening)]
impl Fs for MemFs {
    fn read(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        self.lock()
            .files
            .get(path)
            .map(FileState::visible)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, path.display().to_string()))
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        let mut tree = self.lock();
        // `write` is the atomic-replace pattern: durable immediately.
        tree.files.insert(
            path.to_path_buf(),
            FileState {
                synced: bytes.to_vec(),
                pending: Vec::new(),
                durable: true,
            },
        );
        Ok(())
    }

    fn append(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        let mut tree = self.lock();
        let file = tree.files.entry(path.to_path_buf()).or_default();
        file.pending.extend_from_slice(bytes);
        Ok(())
    }

    fn sync(&self, path: &Path) -> std::io::Result<()> {
        let mut tree = self.lock();
        let Some(file) = tree.files.get_mut(path) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                path.display().to_string(),
            ));
        };
        file.synced.extend_from_slice(&file.pending);
        file.pending.clear();
        // An `fsync` on a file makes the file itself durable, not just its bytes: a journal line
        // written to a brand-new segment must survive a crash.
        file.durable = true;
        Ok(())
    }

    fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
        let mut tree = self.lock();
        let Some(file) = tree.files.remove(from) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                from.display().to_string(),
            ));
        };
        tree.files.insert(to.to_path_buf(), file);
        drop(tree);
        self.renames().push((from.to_path_buf(), to.to_path_buf()));
        Ok(())
    }

    fn remove(&self, path: &Path) -> std::io::Result<()> {
        let mut tree = self.lock();
        tree.files
            .remove(path)
            .map(|_| ())
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, path.display().to_string()))
    }

    fn create_dir_all(&self, path: &Path) -> std::io::Result<()> {
        let mut tree = self.lock();
        let mut current = Some(path);
        while let Some(dir) = current {
            tree.dirs.insert(dir.to_path_buf());
            current = dir.parent();
        }
        Ok(())
    }

    fn read_dir(&self, path: &Path) -> std::io::Result<Vec<PathBuf>> {
        let tree = self.lock();
        if !tree.dirs.contains(path) && !tree.files.keys().any(|file| file.parent() == Some(path)) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                path.display().to_string(),
            ));
        }
        let mut out: Vec<PathBuf> = tree
            .files
            .keys()
            .filter(|file| file.parent() == Some(path))
            .cloned()
            .collect();
        out.sort();
        Ok(out)
    }

    fn len(&self, path: &Path) -> std::io::Result<u64> {
        Ok(u64::try_from(self.read(path)?.len()).unwrap_or(u64::MAX))
    }

    fn exists(&self, path: &Path) -> bool {
        let tree = self.lock();
        tree.files.contains_key(path) || tree.dirs.contains(path)
    }
}

/// A filesystem that fails a named operation after N successful calls.
///
/// Wraps any [`Fs`] so a test can inherit the real semantics and interrupt one call. Kept
/// separate from [`MemFs`] so that a fault can be injected into `StdFs` when the test needs a
/// real `fsync` — for example to prove that a temp directory really is on a durable filesystem.
#[derive(Debug)]
pub struct FlakyFs<F: Fs> {
    inner: F,
    plan: Mutex<BTreeMap<Op, (u64, u64, std::io::ErrorKind)>>,
    counts: Mutex<BTreeMap<Op, u64>>,
}

/// Which operation a [`FlakyFs`] count refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Op {
    Read,
    Write,
    Append,
    Sync,
    Rename,
    Remove,
    ReadDir,
}

impl<F: Fs> FlakyFs<F> {
    /// Wrap `inner` with no failures armed.
    pub const fn new(inner: F) -> Self {
        Self {
            inner,
            plan: Mutex::new(BTreeMap::new()),
            counts: Mutex::new(BTreeMap::new()),
        }
    }

    /// Fail `op` on its `occurrence`-th call (1-based) with `kind`.
    pub fn fail_on(&self, op: Op, occurrence: u64, kind: std::io::ErrorKind) {
        self.plan
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(op, (occurrence, occurrence, kind));
    }

    /// Fail `op` for every call from `first` to `last` inclusive.
    ///
    /// A range rather than a single call because "the disk is full until something is freed"
    /// is the shape of the real incident, and a one-shot failure lets a retry loop succeed and
    /// hide the bug.
    pub fn fail_range(&self, op: Op, first: u64, last: u64, kind: std::io::ErrorKind) {
        self.plan
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(op, (first, last, kind));
    }

    /// How many times `op` was called.
    #[must_use]
    pub fn count(&self, op: Op) -> u64 {
        self.counts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&op)
            .copied()
            .unwrap_or(0)
    }

    /// The underlying filesystem, for assertions.
    #[must_use]
    pub const fn inner(&self) -> &F {
        &self.inner
    }

    // The guard scopes are deliberate: `counts` must be released before `plan` is taken so the
    // two locks are never held together. Clippy's suggestion to drop earlier is what the block
    // already does, and collapsing it would make the ordering invisible.
    #[allow(clippy::significant_drop_tightening)]
    fn tick(&self, op: Op) -> std::io::Result<()> {
        // The count and the plan are read under separate short locks so that the check never
        // holds two guards at once and a caller's own lock ordering cannot deadlock it.
        let count = {
            let mut counts = self
                .counts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let count = counts.entry(op).or_insert(0);
            *count += 1;
            *count
        };
        let failure = {
            let plan = self
                .plan
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            plan.get(&op)
                .filter(|(first, last, _)| (*first..=*last).contains(&count))
                .map(|(_, _, kind)| *kind)
        };
        failure.map_or(Ok(()), |kind| {
            Err(std::io::Error::new(kind, format!("injected {op:?} failure")))
        })
    }
}

#[allow(clippy::significant_drop_tightening)]
impl<F: Fs> Fs for FlakyFs<F> {
    fn read(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        self.tick(Op::Read)?;
        self.inner.read(path)
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        self.tick(Op::Write)?;
        self.inner.write(path, bytes)
    }

    fn append(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        self.tick(Op::Append)?;
        self.inner.append(path, bytes)
    }

    fn sync(&self, path: &Path) -> std::io::Result<()> {
        self.tick(Op::Sync)?;
        self.inner.sync(path)
    }

    fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
        self.tick(Op::Rename)?;
        self.inner.rename(from, to)
    }

    fn remove(&self, path: &Path) -> std::io::Result<()> {
        self.tick(Op::Remove)?;
        self.inner.remove(path)
    }

    fn create_dir_all(&self, path: &Path) -> std::io::Result<()> {
        self.inner.create_dir_all(path)
    }

    fn read_dir(&self, path: &Path) -> std::io::Result<Vec<PathBuf>> {
        self.tick(Op::ReadDir)?;
        self.inner.read_dir(path)
    }

    fn len(&self, path: &Path) -> std::io::Result<u64> {
        self.inner.len(path)
    }

    fn exists(&self, path: &Path) -> bool {
        self.inner.exists(path)
    }
}

/// Read a whole file from any `Fs` into a `String`.
pub fn read_text(fs: &impl Fs, path: &Path) -> std::io::Result<String> {
    let bytes = fs.read(path)?;
    // A journal is UTF-8 by construction; a non-UTF-8 file is corruption, and reporting it as an
    // I/O error keeps this helper from needing a second error type.
    String::from_utf8(bytes).map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
}

/// Copy a file from one `Fs` to another, used to move a `MemFs` scenario onto a real disk.
pub fn copy_between(from: &impl Fs, from_path: &Path, to: &impl Fs, to_path: &Path) -> std::io::Result<()> {
    let bytes = from.read(from_path)?;
    to.write(to_path, &bytes)
}

/// A `Read` adapter over an `Fs` file, for APIs that insist on a reader.
#[derive(Debug)]
pub struct FsReader<'a, F: Fs> {
    fs: &'a F,
    path: PathBuf,
    buffer: std::io::Cursor<Vec<u8>>,
}

impl<'a, F: Fs> FsReader<'a, F> {
    /// Read `path` eagerly and expose it as a reader.
    pub fn new(fs: &'a F, path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let bytes = fs.read(&path)?;
        Ok(Self {
            fs,
            path,
            buffer: std::io::Cursor::new(bytes),
        })
    }

    /// The filesystem this reader came from.
    #[must_use]
    pub const fn fs(&self) -> &'a F {
        self.fs
    }

    /// The path it is reading.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl<F: Fs> std::io::Read for FsReader<'_, F> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.buffer.read(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem() -> (MemFs, PathBuf) {
        let fs = MemFs::new();
        let root = PathBuf::from("/repo");
        fs.create_dir_all(&root).unwrap();
        (fs, root)
    }

    // ── MemFs semantics ───────────────────────────────────────────────────────────────────

    #[test]
    fn a_written_file_reads_back() {
        let (fs, root) = mem();
        let path = root.join("a.txt");
        fs.write(&path, b"hello").unwrap();
        assert_eq!(fs.read(&path).unwrap(), b"hello");
        assert_eq!(fs.len(&path).unwrap(), 5);
        assert!(fs.exists(&path));
    }

    #[test]
    fn write_replaces_rather_than_appends() {
        // The artifact store relies on replace; an append-only fake would let a real bug through.
        let (fs, root) = mem();
        let path = root.join("a.txt");
        fs.write(&path, b"first").unwrap();
        fs.write(&path, b"second").unwrap();
        assert_eq!(fs.read(&path).unwrap(), b"second");
    }

    #[test]
    fn appending_without_syncing_is_visible_but_volatile() {
        let (fs, root) = mem();
        let path = root.join("j.jsonl");
        fs.append(&path, b"line one\n").unwrap();
        assert_eq!(fs.read(&path).unwrap(), b"line one\n");
        fs.crash();
        assert!(!fs.exists(&path), "an unsynced file must not survive a crash");
    }

    #[test]
    fn syncing_makes_the_append_durable() {
        let (fs, root) = mem();
        let path = root.join("j.jsonl");
        fs.append(&path, b"line one\n").unwrap();
        fs.sync(&path).unwrap();
        fs.append(&path, b"line two\n").unwrap();
        fs.crash();
        assert_eq!(
            fs.read(&path).unwrap(),
            b"line one\n",
            "only the synced prefix survives"
        );
    }

    #[test]
    fn an_empty_synced_file_survives_a_crash_but_an_empty_unsynced_one_does_not() {
        let (fs, root) = mem();
        let synced = root.join("synced");
        let unsynced = root.join("unsynced");
        fs.write(&synced, b"").unwrap();
        fs.append(&unsynced, b"").unwrap();
        fs.crash();
        assert!(fs.exists(&synced), "a synced empty file is a real file");
        assert!(
            !fs.exists(&unsynced),
            "a file created by an unsynced append was never durable"
        );
    }

    #[test]
    fn a_new_journal_segment_survives_after_its_first_sync() {
        // Rotation creates a segment and appends to it; the segment must be durable once the
        // line is synced, or every rotation would lose its first record.
        let (fs, root) = mem();
        let segment = root.join("JOURNAL.2026-11-12T10-00-00.jsonl");
        fs.append(&segment, b"{\"seq\":0}\n").unwrap();
        fs.sync(&segment).unwrap();
        fs.crash();
        assert_eq!(fs.read(&segment).unwrap(), b"{\"seq\":0}\n");
    }

    #[test]
    fn a_rename_lost_before_the_directory_sync_restores_the_old_name() {
        // The artifact store syncs the file, then renames it, then syncs the directory. This
        // test is the reason that third step exists: without it a crash leaves the artifact
        // under its temp name and `gc` sees nothing.
        let (fs, root) = mem();
        let temp = root.join("x.part");
        let final_path = root.join("x");
        fs.write(&temp, b"payload").unwrap();
        fs.rename(&temp, &final_path).unwrap();
        assert!(fs.exists(&final_path));
        fs.crash();
        assert!(fs.exists(&temp), "the rename did not survive");
        assert!(!fs.exists(&final_path));
    }

    #[test]
    fn a_rename_survives_when_the_directory_was_synced() {
        let (fs, root) = mem();
        let temp = root.join("x.part");
        let final_path = root.join("x");
        fs.write(&temp, b"payload").unwrap();
        fs.rename(&temp, &final_path).unwrap();
        fs.sync_dir(&root);
        fs.crash();
        assert!(fs.exists(&final_path));
        assert!(!fs.exists(&temp));
    }

    #[test]
    fn renaming_a_missing_file_is_an_error() {
        let (fs, root) = mem();
        let error = fs.rename(&root.join("nope"), &root.join("x")).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn removing_a_missing_file_is_an_error() {
        let (fs, root) = mem();
        let error = fs.remove(&root.join("nope")).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn read_dir_lists_only_direct_children() {
        let (fs, root) = mem();
        fs.create_dir_all(&root.join("nested")).unwrap();
        fs.write(&root.join("a"), b"a").unwrap();
        fs.write(&root.join("nested/b"), b"b").unwrap();
        let entries = fs.read_dir(&root).unwrap();
        assert_eq!(entries, [root.join("a")]);
    }

    #[test]
    fn read_dir_on_a_missing_directory_is_an_error() {
        let (fs, _root) = mem();
        let error = fs.read_dir(Path::new("/nowhere")).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn the_standard_filesystem_satisfies_the_same_trait() {
        // Generic code must compile against both, or the trait is a test-only fiction.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        StdFs.write(&path, b"real").unwrap();
        StdFs.append(&path, b" bytes").unwrap();
        assert_eq!(StdFs.read(&path).unwrap(), b"real bytes");
        assert_eq!(read_text(&StdFs, &path).unwrap(), "real bytes");
    }

    // ── FlakyFs ───────────────────────────────────────────────────────────────────────────

    #[test]
    fn a_named_occurrence_fails() {
        let (mem_fs, root) = mem();
        let fs = FlakyFs::new(mem_fs);
        fs.fail_on(Op::Write, 2, std::io::ErrorKind::StorageFull);
        let path = root.join("a");
        fs.write(&path, b"1").unwrap();
        let error = fs.write(&path, b"2").unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::StorageFull);
        fs.write(&path, b"3").unwrap();
        assert_eq!(fs.count(Op::Write), 3);
    }

    #[test]
    fn a_range_fails_every_call_inside_it() {
        // Models "the disk stays full until something is freed", which a one-shot failure would
        // not: a retry loop would succeed on the second attempt and hide the bug.
        let (mem_fs, root) = mem();
        let fs = FlakyFs::new(mem_fs);
        fs.fail_range(Op::Append, 2, 4, std::io::ErrorKind::StorageFull);
        let path = root.join("j");
        assert_eq!(fs.append(&path, b"1").unwrap(), ());
        for _ in 0..3 {
            assert!(fs.append(&path, b"x").is_err());
        }
        assert!(fs.append(&path, b"5").is_ok());
        assert_eq!(fs.count(Op::Append), 5);
    }

    #[test]
    fn a_failing_write_changes_nothing() {
        // The failure happens before the underlying write, so the filesystem sees no partial
        // state. If it happened after, the test would be asserting on a different bug.
        let (mem_fs, root) = mem();
        let fs = FlakyFs::new(mem_fs);
        fs.fail_on(Op::Write, 1, std::io::ErrorKind::StorageFull);
        let path = root.join("a");
        assert!(fs.write(&path, b"nope").is_err());
        assert!(!fs.exists(&path));
    }

    #[test]
    fn injecting_into_the_real_filesystem_leaves_it_consistent() {
        let dir = tempfile::tempdir().unwrap();
        let fs = FlakyFs::new(StdFs);
        fs.fail_on(Op::Sync, 1, std::io::ErrorKind::StorageFull);
        let path = dir.path().join("a");
        fs.write(&path, b"durable").unwrap();
        assert!(fs.sync(&path).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"durable");
        assert_eq!(fs.count(Op::Sync), 1);
    }

    #[test]
    fn operations_are_counted_independently() {
        let (mem_fs, root) = mem();
        let fs = FlakyFs::new(mem_fs);
        let path = root.join("a");
        fs.write(&path, b"x").unwrap();
        fs.read(&path).unwrap();
        fs.read(&path).unwrap();
        assert_eq!(fs.count(Op::Write), 1);
        assert_eq!(fs.count(Op::Read), 2);
        assert_eq!(fs.count(Op::Remove), 0);
    }

    #[test]
    fn a_reader_over_a_filesystem_returns_the_bytes() {
        use std::io::Read as _;
        let (fs, root) = mem();
        let path = root.join("a");
        fs.write(&path, b"payload").unwrap();
        let mut reader = FsReader::new(&fs, &path).unwrap();
        let mut out = String::new();
        reader.read_to_string(&mut out).unwrap();
        assert_eq!(out, "payload");
        assert_eq!(reader.path(), path);
    }

    #[test]
    fn copying_between_filesystems_preserves_content() {
        let dir = tempfile::tempdir().unwrap();
        let (mem_fs, root) = mem();
        let source = root.join("a");
        mem_fs.write(&source, b"shared").unwrap();
        let target = dir.path().join("a");
        copy_between(&mem_fs, &source, &StdFs, &target).unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"shared");
        assert_eq!(mem_fs.read(&source).unwrap(), b"shared");
    }
}
