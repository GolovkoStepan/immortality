//! Content-addressed storage for heavy bodies: tool output, model responses, diffs.
//!
//! The journal stays compact and greppable because anything large is spilled here and
//! referenced by `sha256:<hex>`. The name is the hash, so:
//!
//! - writing the same payload twice costs one file (deduplication is not a feature, it is a
//!   consequence of the naming);
//! - a file's content can be verified against its name, which is the whole point — a journal
//!   that says "the tool printed this" is only trustworthy if the bytes it points at are the
//!   bytes the tool printed.
//!
//! # Corruption is loud
//!
//! [`ArtifactStore::read`] re-hashes what it reads and fails if the digest does not match. It
//! would be cheaper to trust the filename; it would also mean a truncated write, a full disk
//! or a bit flip silently hands the model fabricated output. The cost is one hash per read,
//! and hashing an 8 KiB tool result is nothing next to the model call that follows it.
//!
//! # Atomicity
//!
//! Write to `.tmp/<unique>`, sync it, rename into place, sync the directory. A crash at any
//! point leaves either nothing, a stale temp file that `gc` sweeps, or a complete artifact.
//! A partial file can never appear under a digest name, so a reader never has to guess whether
//! a short file is corrupt or still being written.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write as _};
use std::path::{Path, PathBuf};

use sha2::{Digest as _, Sha256};

use crate::ids::Ulid;
use crate::journal::{Digest, digest_text};

/// Subdirectory for in-flight writes. Not a sibling of the artifacts: keeping it inside the
/// store means `gc` can clear it without knowing where the store lives.
const TEMP_DIR: &str = ".tmp";

/// One file in the store, as `gc` needs to see it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactEntry {
    pub digest: Digest,
    pub bytes: u64,
    /// Modification time in seconds since the epoch, or 0 if the platform would not say.
    ///
    /// TTL is based on the file's own mtime rather than on the journal: an artifact written
    /// ten minutes ago and one written last month are indistinguishable from their content,
    /// and only the filesystem knows the difference.
    pub modified_secs: u64,
}

/// A directory of content-addressed blobs.
#[derive(Debug, Clone)]
pub struct ArtifactStore {
    dir: PathBuf,
}

impl ArtifactStore {
    /// Open a store, creating the directory (and the temp directory) if needed.
    pub fn open(dir: impl AsRef<Path>) -> crate::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| crate::Error::io(&dir, e))?;
        let temp = dir.join(TEMP_DIR);
        std::fs::create_dir_all(&temp).map_err(|e| crate::Error::io(&temp, e))?;
        Ok(Self { dir })
    }

    /// The directory backing this store.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Path where a digest's contents live.
    #[must_use]
    pub fn path_for(&self, digest: &Digest) -> PathBuf {
        self.dir.join(hex(digest))
    }

    /// Store `bytes`, returning their digest. Idempotent.
    ///
    /// If the artifact already exists it is not rewritten. That is not only an optimisation:
    /// rewriting would mean reading and rewriting a file other readers may be using, and the
    /// content is by definition identical.
    pub fn put(&self, bytes: &[u8]) -> crate::Result<Digest> {
        let digest = digest_of(bytes);
        let final_path = self.path_for(&digest);
        if final_path.exists() {
            return Ok(digest);
        }
        let mut temp = TempFile::create(&self.dir)?;
        temp.write_all(bytes)?;
        temp.finish(&final_path)?;
        Ok(digest)
    }

    /// Store everything a reader produces, without holding it in memory.
    ///
    /// Used for streamed tool output, which can exceed what is reasonable to buffer just to
    /// compute a hash before writing.
    pub fn put_from<R: Read>(&self, mut reader: R) -> crate::Result<Digest> {
        let mut temp = TempFile::create(&self.dir)?;
        temp.copy_from(&mut reader)?;
        let digest = temp.digest();
        let final_path = self.path_for(&digest);
        if final_path.exists() {
            // The dedup path: the bytes were identical to an artifact already present, so the
            // temp file is dropped by `Drop` when `temp` goes out of scope here.
            return Ok(digest);
        }
        temp.finish(&final_path)?;
        Ok(digest)
    }

    /// Read an artifact, verifying it against its name.
    pub fn read(&self, digest: &Digest) -> crate::Result<Vec<u8>> {
        let path = self.path_for(digest);
        let bytes = std::fs::read(&path).map_err(|e| crate::Error::io(&path, e))?;
        Self::verify_bytes(digest, &bytes)?;
        Ok(bytes)
    }

    /// Read an artifact as text, for journals and diffs that are always UTF-8.
    pub fn read_text(&self, digest: &Digest) -> crate::Result<String> {
        let bytes = self.read(digest)?;
        String::from_utf8(bytes).map_err(|error| {
            crate::Error::ArtifactCorrupt(format!(
                "artifact {} is not valid UTF-8: {error}",
                digest_text(digest)
            ))
        })
    }

    /// Check that an artifact exists and has the expected contents.
    pub fn verify(&self, digest: &Digest) -> crate::Result<()> {
        let path = self.path_for(digest);
        let mut file = File::open(&path).map_err(|e| crate::Error::io(&path, e))?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer).map_err(|e| crate::Error::io(&path, e))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        let found: Digest = hasher.finalize().into();
        if &found != digest {
            return Err(crate::Error::ArtifactCorrupt(format!(
                "artifact {} has content {}",
                digest_text(digest),
                digest_text(&found)
            )));
        }
        Ok(())
    }

    /// Whether a file with this name exists. Says nothing about its contents: use
    /// [`Self::verify`] before trusting it.
    #[must_use]
    pub fn contains(&self, digest: &Digest) -> bool {
        self.path_for(digest).is_file()
    }

    /// Every artifact, sorted by digest. Used by `gc` and by the disk-usage line in `status`.
    pub fn entries(&self) -> crate::Result<Vec<ArtifactEntry>> {
        let mut entries = Vec::new();
        let read_dir = match std::fs::read_dir(&self.dir) {
            Ok(read_dir) => read_dir,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(entries),
            Err(error) => return Err(crate::Error::io(&self.dir, error)),
        };
        for entry in read_dir {
            let entry = entry.map_err(|e| crate::Error::io(&self.dir, e))?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(digest) = parse_hex(name) else {
                // `.tmp` and anything else that is not a digest name is not an artifact.
                continue;
            };
            let metadata = entry.metadata().map_err(|e| crate::Error::io(entry.path(), e))?;
            if !metadata.is_file() {
                continue;
            }
            let modified_secs = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());
            entries.push(ArtifactEntry {
                digest,
                bytes: metadata.len(),
                modified_secs,
            });
        }
        entries.sort_by_key(|entry| entry.digest);
        Ok(entries)
    }

    /// Remove an artifact. Returns whether it existed.
    ///
    /// No verification: `gc` must be able to delete a corrupt artifact too, otherwise a bad
    /// file becomes immortal.
    pub fn remove(&self, digest: &Digest) -> crate::Result<bool> {
        let path = self.path_for(digest);
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(crate::Error::io(path, error)),
        }
    }

    /// Total bytes used, including stale temp files, which a crashed writer may have left.
    pub fn total_bytes(&self) -> crate::Result<u64> {
        let mut total = 0;
        for entry in self.entries()? {
            total += entry.bytes;
        }
        let temp = self.dir.join(TEMP_DIR);
        if let Ok(read_dir) = std::fs::read_dir(&temp) {
            for entry in read_dir.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    total += metadata.len();
                }
            }
        }
        Ok(total)
    }

    /// Delete stale temp files left by a writer that died mid-`put`.
    ///
    /// Called by `gc` and on daemon start. Bounded by age so that a file being written *right
    /// now* by another thread is not removed from under it — the daemon is the only writer, but
    /// tests and `doctor` run concurrently, and an in-flight temp file is exactly one second
    /// old.
    pub fn sweep_temp(&self, older_than_secs: u64, now_secs: u64) -> crate::Result<usize> {
        let temp = self.dir.join(TEMP_DIR);
        let read_dir = match std::fs::read_dir(&temp) {
            Ok(read_dir) => read_dir,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(crate::Error::io(&temp, error)),
        };
        let mut removed = 0;
        for entry in read_dir {
            let entry = entry.map_err(|e| crate::Error::io(&temp, e))?;
            let Some(modified) = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
            else {
                // No usable mtime means no way to prove the file is stale. Leaving it costs a
                // few bytes; deleting a file another thread is writing costs correctness.
                continue;
            };
            // `now` before `modified` means the clock went backwards (NTP step, a test using a
            // fake epoch). Such a file is not old, so it is kept.
            if modified <= now_secs
                && now_secs - modified >= older_than_secs
                && std::fs::remove_file(entry.path()).is_ok()
            {
                removed += 1;
            }
        }
        Ok(removed)
    }

    fn verify_bytes(digest: &Digest, bytes: &[u8]) -> crate::Result<()> {
        let found = digest_of(bytes);
        if &found != digest {
            return Err(crate::Error::ArtifactCorrupt(format!(
                "artifact {} has content {}",
                digest_text(digest),
                digest_text(&found)
            )));
        }
        Ok(())
    }
}

/// SHA-256 of a buffer, as stored.
#[must_use]
pub fn digest_of(bytes: &[u8]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// Lowercase hex, no prefix: the on-disk name.
fn hex(digest: &Digest) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(64);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Inverse of [`hex`]. Returns `None` for anything that is not exactly a digest name.
fn parse_hex(name: &str) -> Option<Digest> {
    if name.len() != 64 || !name.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut digest = [0u8; 32];
    let (pairs, _) = name.as_bytes().as_chunks::<2>();
    for (index, pair) in pairs.iter().enumerate() {
        let pair = std::str::from_utf8(pair).ok()?;
        digest[index] = u8::from_str_radix(pair, 16).ok()?;
    }
    Some(digest)
}

/// A write-in-progress file that removes itself unless it is finished.
///
/// The guard matters for the error path: `put` can fail between creating the file and renaming
/// it (a full disk, a closed handle), and leaving a temp file behind on every failure would let
/// a broken disk fill the store with garbage.
struct TempFile {
    path: PathBuf,
    file: File,
    hasher: Sha256,
    finished: bool,
}

impl TempFile {
    fn create(dir: &Path) -> crate::Result<Self> {
        let temp_dir = dir.join(TEMP_DIR);
        std::fs::create_dir_all(&temp_dir).map_err(|e| crate::Error::io(&temp_dir, e))?;
        // ULID is unique per process and across processes, so concurrent writers never collide
        // the way a pid-based name would between threads.
        let path = temp_dir.join(format!("{}.part", Ulid::now()));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| crate::Error::io(&path, e))?;
        Ok(Self {
            path,
            file,
            hasher: Sha256::new(),
            finished: false,
        })
    }

    fn write_all(&mut self, bytes: &[u8]) -> crate::Result<()> {
        self.file
            .write_all(bytes)
            .map_err(|e| crate::Error::io(&self.path, e))?;
        self.hasher.update(bytes);
        Ok(())
    }

    fn copy_from(&mut self, reader: &mut impl Read) -> crate::Result<()> {
        let mut buffer = vec![0u8; 64 * 1024];
        loop {
            let read = reader
                .read(&mut buffer)
                .map_err(|e| crate::Error::io(&self.path, e))?;
            if read == 0 {
                return Ok(());
            }
            self.write_all(&buffer[..read])?;
        }
    }

    fn digest(&self) -> Digest {
        // Safe to call mid-write: `Sha256` clones its state.
        self.hasher.clone().finalize().into()
    }

    fn finish(mut self, final_path: &Path) -> crate::Result<()> {
        self.file
            .sync_all()
            .map_err(|e| crate::Error::io(&self.path, e))?;
        std::fs::rename(&self.path, final_path).map_err(|e| crate::Error::io(&self.path, e))?;
        self.finished = true;
        // The rename is only durable once the directory entry is on disk.
        if let Some(parent) = final_path.parent() {
            let handle = File::open(parent).map_err(|e| crate::Error::io(parent, e))?;
            handle.sync_all().map_err(|e| crate::Error::io(parent, e))?;
        }
        Ok(())
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        if !self.finished {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, ArtifactStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = ArtifactStore::open(dir.path().join("artifacts")).unwrap();
        (dir, store)
    }

    #[test]
    fn put_then_read_roundtrips() {
        let (_dir, store) = store();
        let digest = store.put(b"hello, world").unwrap();
        assert_eq!(store.read(&digest).unwrap(), b"hello, world");
        assert_eq!(digest, digest_of(b"hello, world"));
    }

    #[test]
    fn identical_payloads_share_one_file() {
        let (_dir, store) = store();
        let first = store.put(b"same").unwrap();
        let second = store.put(b"same").unwrap();
        assert_eq!(first, second);
        assert_eq!(store.entries().unwrap().len(), 1);
    }

    #[test]
    fn different_payloads_get_different_files() {
        let (_dir, store) = store();
        store.put(b"a").unwrap();
        store.put(b"b").unwrap();
        assert_eq!(store.entries().unwrap().len(), 2);
    }

    #[test]
    fn empty_payload_is_a_valid_artifact() {
        let (_dir, store) = store();
        let digest = store.put(b"").unwrap();
        assert_eq!(store.read(&digest).unwrap(), Vec::<u8>::new());
        assert_eq!(store.entries().unwrap()[0].bytes, 0);
    }

    #[test]
    fn a_tampered_artifact_is_reported_not_returned() {
        let (_dir, store) = store();
        let digest = store.put(b"original").unwrap();
        // Same length, different bytes: the failure case a size check would miss.
        std::fs::write(store.path_for(&digest), b"tampered").unwrap();
        let error = store.read(&digest).unwrap_err();
        assert!(matches!(error, crate::Error::ArtifactCorrupt(_)), "{error}");
        assert!(store.verify(&digest).is_err());
    }

    #[test]
    fn a_truncated_artifact_is_reported() {
        let (_dir, store) = store();
        let digest = store.put(b"a longer payload").unwrap();
        std::fs::write(store.path_for(&digest), b"a long").unwrap();
        assert!(store.read(&digest).is_err());
    }

    #[test]
    fn reading_a_missing_artifact_is_an_io_error() {
        let (_dir, store) = store();
        let missing = digest_of(b"never written");
        assert!(!store.contains(&missing));
        assert!(matches!(
            store.read(&missing).unwrap_err(),
            crate::Error::Io { .. }
        ));
    }

    #[test]
    fn put_from_matches_put() {
        let (_dir, store) = store();
        let payload = b"streamed in two chunks";
        let digest = store.put_from(&payload[..]).unwrap();
        assert_eq!(digest, digest_of(payload));
        assert_eq!(store.read(&digest).unwrap(), payload);
    }

    #[test]
    fn put_from_deduplicates_without_rewriting() {
        let (_dir, store) = store();
        let first = store.put(b"dup").unwrap();
        let stamp = store.entries().unwrap()[0].modified_secs;
        let second = store.put_from(&b"dup"[..]).unwrap();
        assert_eq!(first, second);
        assert_eq!(store.entries().unwrap().len(), 1);
        assert_eq!(store.entries().unwrap()[0].modified_secs, stamp);
    }

    #[test]
    fn no_temp_files_survive_a_successful_put() {
        let (_dir, store) = store();
        store.put(b"x").unwrap();
        let temp = store.dir().join(TEMP_DIR);
        assert_eq!(std::fs::read_dir(temp).unwrap().count(), 0);
    }

    #[test]
    fn entries_ignore_non_digest_names() {
        let (_dir, store) = store();
        store.put(b"x").unwrap();
        std::fs::write(store.dir().join("README"), b"notes").unwrap();
        std::fs::write(store.dir().join("abc"), b"short").unwrap();
        let entries = store.entries().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].digest, digest_of(b"x"));
    }

    #[test]
    fn remove_is_idempotent() {
        let (_dir, store) = store();
        let digest = store.put(b"bye").unwrap();
        assert!(store.remove(&digest).unwrap());
        assert!(!store.remove(&digest).unwrap());
        assert!(!store.contains(&digest));
    }

    #[test]
    fn total_bytes_counts_artifacts_and_temp_files() {
        let (_dir, store) = store();
        store.put(&[0u8; 100]).unwrap();
        store.put(&[1u8; 50]).unwrap();
        assert_eq!(store.total_bytes().unwrap(), 150);
        std::fs::write(store.dir().join(TEMP_DIR).join("x.part"), vec![2u8; 10]).unwrap();
        assert_eq!(store.total_bytes().unwrap(), 160);
    }

    #[test]
    fn sweep_temp_respects_age() {
        let (_dir, store) = store();
        let temp = store.dir().join(TEMP_DIR);
        let file = temp.join("stale.part");
        std::fs::write(&file, b"stale").unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // A file written a moment ago is never swept, however small the threshold.
        assert_eq!(store.sweep_temp(0, now).unwrap(), 1, "zero threshold sweeps");

        std::fs::write(&file, b"stale").unwrap();
        assert_eq!(store.sweep_temp(3_600, now).unwrap(), 0, "a fresh file is kept");
        assert!(file.exists());
        assert_eq!(store.sweep_temp(3_600, now + 7_200).unwrap(), 1);
        assert!(!file.exists());
    }

    #[test]
    fn sweep_temp_keeps_files_with_a_future_mtime() {
        // A clock that stepped backwards must not turn a live temp file into a "stale" one.
        let (_dir, store) = store();
        let file = store.dir().join(TEMP_DIR).join("future.part");
        std::fs::write(&file, b"in flight").unwrap();
        assert_eq!(store.sweep_temp(1, 0).unwrap(), 0);
        assert!(file.exists());
    }

    #[test]
    fn concurrent_puts_of_the_same_payload_do_not_corrupt_it() {
        let dir = tempfile::tempdir().unwrap();
        let store = ArtifactStore::open(dir.path().join("artifacts")).unwrap();
        let payload = b"contended".to_vec();
        let mut handles = Vec::new();
        for _ in 0..8 {
            let store = store.clone();
            let payload = payload.clone();
            handles.push(std::thread::spawn(move || store.put(&payload).unwrap()));
        }
        let digests: Vec<Digest> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert!(digests.iter().all(|d| d == &digests[0]));
        assert_eq!(store.read(&digests[0]).unwrap(), payload);
        assert_eq!(store.entries().unwrap().len(), 1);
        assert_eq!(
            std::fs::read_dir(store.dir().join(TEMP_DIR)).unwrap().count(),
            0,
            "temp files leaked"
        );
    }

    #[test]
    fn path_is_plain_lowercase_hex() {
        let (_dir, store) = store();
        let digest = store.put(b"naming").unwrap();
        let name = store
            .path_for(&digest)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        assert_eq!(name.len(), 64);
        assert!(
            name.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        );
        assert_eq!(parse_hex(&name), Some(digest));
    }

    #[test]
    fn read_text_rejects_binary() {
        let (_dir, store) = store();
        let digest = store.put(&[0xff, 0xfe, 0xfd]).unwrap();
        assert!(store.read_text(&digest).is_err());
    }

    #[test]
    fn a_crash_mid_put_leaves_no_readable_artifact() {
        // Emulate a writer killed between creating its temp file and renaming it: a partial
        // file exists, but not under a digest name. The store must neither expose it as an
        // artifact nor let it shadow a later successful write.
        let (_dir, store) = store();
        let payload = b"the whole body";
        let digest = digest_of(payload);
        std::fs::write(store.dir().join(TEMP_DIR).join("interrupted.part"), &payload[..6]).unwrap();

        assert!(!store.contains(&digest));
        assert_eq!(store.entries().unwrap().len(), 0);
        assert!(store.read(&digest).is_err());

        // Recovery: the later write succeeds and the temp file is irrelevant to it.
        assert_eq!(store.put(payload).unwrap(), digest);
        assert_eq!(store.read(&digest).unwrap(), payload);
        assert_eq!(store.entries().unwrap().len(), 1);
    }

    #[test]
    fn large_payload_survives_the_streaming_path() {
        let (_dir, store) = store();
        let payload: Vec<u8> = (0..1_000_000u32)
            .map(|i| u8::try_from(i % 251).unwrap())
            .collect();
        let digest = store.put_from(&payload[..]).unwrap();
        assert_eq!(store.read(&digest).unwrap(), payload);
        assert_eq!(store.entries().unwrap()[0].bytes, 1_000_000);
    }
}
