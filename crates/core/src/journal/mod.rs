//! Append-only event log: the authoritative record of everything that happened.
//!
//! # Why the journal is the source of truth
//!
//! Every other file in `.agent/` and in the state directory is derived: `TASKS.json` is a
//! materialized view for humans, `attempts.json` a cache for fast status, `PLAN.md` prose.
//! When they disagree with the journal after a crash, the journal wins, because it is the
//! only file whose contents are *proven* to have been on disk before an effect happened.
//!
//! # Format
//!
//! JSON Lines: one JSON object per line, `\n`-terminated. Locked down by the rules below.
//!
//! - **Atomic append.** One `write_all` of the complete line plus its newline, then a
//!   `sync_data`. A line is therefore either fully present or absent, and a partial line can
//!   only be caused by a torn tail, which is repaired by truncation.
//! - **Monotonic `seq`** per task. Gaps mean lost records and are reported, never smoothed
//!   over.
//! - **`prev` chain.** Each record carries the hash of the previous record's canonical form.
//!   This detects silent edits and localizes a break to a single pair of records. A broken
//!   chain is *not* repaired: the task is blocked and a human is asked, because a chain
//!   break means the audit trail is no longer trustworthy.
//! - **Segments of 8 MiB.** Rotation keeps a single open file, so appends never need to seek,
//!   and the active file stays small enough to read in one go during recovery.
//!
//! # Why not `fsync` the directory after every append
//!
//! The directory entry is created once per segment by rotating; after that, appends only
//! touch file content. Paying a directory `fsync` per event would roughly double the cost of
//! a step that already includes a multi-second model call, for a guarantee that only matters
//! for the segment's very first record — which rotation syncs explicitly.
//!
//! # Hashing
//!
//! `prev` is the SHA-256 of the previous record's **line as written**, without its newline.
//! Hashing the exact bytes rather than a re-serialization means detection does not depend on
//! the serializer being deterministic, and it lets a reader verify a file with no parsing
//! beyond splitting lines.

mod record;
mod writer;

pub use record::{JournalEvent, JournalRecord, Kind, RecordFields, SCHEMA_VERSION};
pub use writer::{Journal, JournalReader, ReadOutcome, ReadProblem, SEGMENT_LIMIT_BYTES, hash_line};

/// A 32-byte SHA-256 digest, written as `sha256:<64 hex>` wherever it appears in a file.
///
/// Distinguishing this from a bare hex string matters in the journal schema: a `prev` field
/// and an `artifact` field are both hashes, and a future signature field must not be
/// confusable with either.
pub type Digest = [u8; 32];

/// Render a digest the way it appears in files.
#[must_use]
pub fn digest_text(digest: &Digest) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(7 + 64);
    out.push_str("sha256:");
    for byte in digest {
        // Writing into a `String` cannot fail.
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Parse `sha256:<64 hex>`.
pub fn parse_digest(text: &str) -> crate::Result<Digest> {
    let Some(hex) = text.strip_prefix("sha256:") else {
        return Err(crate::Error::JournalCorrupt(format!(
            "digest `{text}` is not prefixed with `sha256:`"
        )));
    };
    if hex.len() != 64 {
        return Err(crate::Error::JournalCorrupt(format!(
            "digest `{text}` has {} hex digits, expected 64",
            hex.len()
        )));
    }
    let mut digest = [0u8; 32];
    let (pairs, _) = hex.as_bytes().as_chunks::<2>();
    for (index, chunk) in pairs.iter().enumerate() {
        let pair = std::str::from_utf8(chunk)
            .map_err(|_| crate::Error::JournalCorrupt(format!("digest `{text}` is not valid UTF-8")))?;
        digest[index] = u8::from_str_radix(pair, 16)
            .map_err(|_| crate::Error::JournalCorrupt(format!("digest `{text}` has a non-hex character")))?;
    }
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digests_roundtrip_through_text() {
        let digest: Digest = std::array::from_fn(|i| u8::try_from(i).unwrap() * 8);
        let text = digest_text(&digest);
        assert_eq!(text.len(), 7 + 64);
        assert!(text.starts_with("sha256:"));
        assert_eq!(parse_digest(&text).unwrap(), digest);
    }

    #[test]
    fn digest_parsing_rejects_malformed_input() {
        assert!(parse_digest("deadbeef").is_err());
        assert!(parse_digest("sha256:zz").is_err());
        assert!(parse_digest(&format!("sha256:{}", "0".repeat(63))).is_err());
        assert!(parse_digest(&format!("sha256:{}", "g".repeat(64))).is_err());
    }

    #[test]
    fn all_zero_digest_is_valid() {
        let text = digest_text(&[0u8; 32]);
        assert_eq!(text, format!("sha256:{}", "0".repeat(64)));
    }
}
