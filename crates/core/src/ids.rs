//! Task, subtask and attempt identifiers, plus attempt nonces.
//!
//! IDs are prefixed (`tsk_`, `sub_`, `att_`) so that a bare string in a log or a git commit
//! body is unambiguous. They are sortable by creation time, which makes journal scans and
//! the `status` command cheap.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// ULID-like sortable identifier: 48-bit millisecond timestamp + 80 bits of randomness.
///
/// Crockford base32, 26 characters. Implemented here rather than pulled in as a dependency
/// because the format is frozen by the journal schema and must not change under us.
///
/// Encoding detail that is easy to get wrong: 26 base32 characters hold 130 bits, but the
/// value is 128 bits, so the **top two bits of the first character are always zero**. Shifting
/// the 128-bit value left to "make room" overflows a `u128` and silently destroys the
/// timestamp's high bits; the correct encoding writes the value into bits 2..130.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Ulid([u8; 16]);

const CROCKFORD: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// Monotonic tie-breaker so that identifiers minted within one millisecond stay ordered.
static COUNTER: AtomicU64 = AtomicU64::new(0);

impl Ulid {
    /// Build from an explicit timestamp and 10 random bytes. Useful in tests.
    #[must_use]
    pub fn from_parts(timestamp_ms: u64, randomness: [u8; 10]) -> Self {
        let mut bytes = [0u8; 16];
        // Only the low 48 bits of the timestamp fit; take the last six big-endian bytes.
        bytes[..6].copy_from_slice(&timestamp_ms.to_be_bytes()[2..]);
        bytes[6..].copy_from_slice(&randomness);
        Self(bytes)
    }

    /// Generate a new identifier from the system clock and the OS entropy pool.
    ///
    /// The counter is xor-ed in so that two identifiers requested in the same millisecond still
    /// differ; a duplicated task id would corrupt the journal's keying.
    #[must_use]
    pub fn now() -> Self {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
        let counter = COUNTER.fetch_add(1, Ordering::Relaxed);

        let mut randomness = [0u8; 10];
        // If the OS cannot supply entropy the process is in no state to be trusted anyway;
        // fall back to the counter rather than panicking inside an identifier constructor.
        if getrandom::fill(&mut randomness).is_err() {
            randomness[..8].copy_from_slice(&counter.to_be_bytes());
        }
        let counter_bytes = counter.to_be_bytes();
        randomness[8] ^= counter_bytes[6];
        randomness[9] ^= counter_bytes[7];

        Self::from_parts(timestamp_ms, randomness)
    }

    /// Parse a 26-character base32 identifier.
    pub fn parse(text: &str) -> Result<Self> {
        if text.len() != 26 {
            return Err(Error::InvalidId(format!(
                "expected 26 characters, got {}",
                text.len()
            )));
        }
        // 26 characters carry 130 bits, of which the top two are padding and must be zero:
        // the first character may therefore only be `0`..=`7`.
        if text.as_bytes()[0] > b'7' {
            return Err(Error::InvalidId(
                "first character is out of range for a 128-bit identifier".into(),
            ));
        }

        let mut value: u128 = 0;
        for (position, ch) in text.bytes().enumerate() {
            // Crockford base32 skips `I`, `L`, `O` and `U`, so the digit is the character's
            // *position* in the alphabet, not arithmetic on its ASCII code: `Z` is 31, not 35.
            let Some(digit) = CROCKFORD.iter().position(|&c| c == ch) else {
                return Err(Error::InvalidId(format!(
                    "invalid character `{}` at position {position}",
                    ch as char
                )));
            };
            let digit = u128::try_from(digit).unwrap_or_default();
            // The first character is already three bits wide, so shifting it would add five
            // and overflow the 128-bit accumulator before the remaining 125 bits arrive.
            value = if position == 0 {
                digit
            } else {
                (value << 5) | digit
            };
        }
        Ok(Self(value.to_be_bytes()))
    }

    /// Milliseconds since the Unix epoch encoded in the identifier.
    #[must_use]
    pub fn timestamp_ms(&self) -> u64 {
        let b = self.0;
        (u64::from(b[0]) << 40)
            | (u64::from(b[1]) << 32)
            | (u64::from(b[2]) << 24)
            | (u64::from(b[3]) << 16)
            | (u64::from(b[4]) << 8)
            | u64::from(b[5])
    }
}

impl fmt::Display for Ulid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut out = [0u8; 26];
        let mut value = u128::from_be_bytes(self.0);
        for slot in out.iter_mut().rev() {
            // `value & 0x1F` is < 32, so the conversion cannot fail.
            let index = usize::try_from(value & 0x1F).unwrap_or_default();
            *slot = CROCKFORD[index];
            value >>= 5;
        }
        f.write_str(std::str::from_utf8(&out).unwrap_or("?"))
    }
}

macro_rules! prefixed_id {
    ($name:ident, $prefix:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            #[doc = concat!("Prefix used in textual form: `", $prefix, "`.")]
            pub const PREFIX: &'static str = $prefix;

            /// Wrap a fresh ULID.
            #[must_use]
            pub fn generate() -> Self {
                Self(format!("{}{}", Self::PREFIX, Ulid::now()))
            }

            /// Build from a raw suffix, for tests and fixtures.
            #[must_use]
            pub fn from_suffix(suffix: &str) -> Self {
                Self(format!("{}{}", Self::PREFIX, suffix))
            }

            pub fn parse(text: &str) -> Result<Self> {
                let suffix = text.strip_prefix(Self::PREFIX).ok_or_else(|| {
                    Error::InvalidId(format!("`{text}` does not start with `{}`", Self::PREFIX))
                })?;
                Ulid::parse(suffix)?;
                Ok(Self(text.to_string()))
            }

            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// Sortable creation time.
            #[must_use]
            pub fn timestamp_ms(&self) -> Option<u64> {
                self.0
                    .strip_prefix(Self::PREFIX)
                    .and_then(|s| Ulid::parse(s).ok())
                    .map(|u| u.timestamp_ms())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

prefixed_id!(TaskId, "tsk_");
prefixed_id!(SubtaskId, "sub_");
prefixed_id!(AttemptId, "att_");
prefixed_id!(EscalationId, "esc_");

/// Random-ish token embedded in commits and process environments so that a crash-time
/// `probe` can tell "did this action already happen?".
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Nonce(String);

impl Nonce {
    #[must_use]
    pub fn generate() -> Self {
        Self(Ulid::now().to_string())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Nonce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulid_is_26_chars_and_roundtrips() {
        let id = Ulid::now();
        let text = id.to_string();
        assert_eq!(text.len(), 26, "{text}");
        assert_eq!(Ulid::parse(&text).unwrap(), id);
    }

    #[test]
    fn ulid_roundtrips_high_bit_patterns() {
        // The old encoder shifted left by two bits and lost these. Guard the boundary.
        for randomness in [[0xFF; 10], [0x00; 10], [0xAA; 10], [0x55; 10]] {
            for timestamp in [0u64, 1, 0x0000_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF] {
                let id = Ulid::from_parts(timestamp & 0x0000_FFFF_FFFF_FFFF, randomness);
                let text = id.to_string();
                assert_eq!(Ulid::parse(&text).unwrap(), id, "timestamp {timestamp}");
            }
        }
    }

    #[test]
    fn ulid_sorts_by_timestamp() {
        let earlier = Ulid::from_parts(1_000, [0; 10]);
        let later = Ulid::from_parts(2_000, [0; 10]);
        assert!(earlier < later);
        assert!(earlier.to_string() < later.to_string());
    }

    #[test]
    fn ulid_keeps_timestamp() {
        let id = Ulid::from_parts(1_700_000_000_123, [7; 10]);
        assert_eq!(id.timestamp_ms(), 1_700_000_000_123);
    }

    #[test]
    fn many_ulids_in_one_millisecond_are_unique() {
        let ids: std::collections::HashSet<String> = (0..10_000).map(|_| Ulid::now().to_string()).collect();
        assert_eq!(ids.len(), 10_000);
    }

    #[test]
    fn ulid_rejects_wrong_length_and_bad_chars() {
        assert!(Ulid::parse("short").is_err());
        assert!(Ulid::parse("01ARZ3NDEKTSV4RRFFQ69G5FA").is_err()); // 25 chars
        assert!(Ulid::parse("01ARZ3NDEKTSV4RRFFQ69G5FAI").is_err()); // I is excluded
        assert!(Ulid::parse("01arz3ndektsv4rrffq69g5fav").is_err()); // lowercase
    }

    #[test]
    fn ulid_rejects_overflowing_first_character() {
        // `Z` in the first position sets the padded bits, which a 128-bit id cannot hold.
        assert!(Ulid::parse("Z1ARZ3NDEKTSV4RRFFQ69G5FAV").is_err());
    }

    #[test]
    fn prefixed_ids_roundtrip() {
        let task = TaskId::generate();
        assert!(task.as_str().starts_with("tsk_"));
        assert_eq!(TaskId::parse(task.as_str()).unwrap(), task);

        let sub = SubtaskId::generate();
        assert!(sub.as_str().starts_with("sub_"));
        let attempt = AttemptId::generate();
        assert!(attempt.as_str().starts_with("att_"));
        let escalation = EscalationId::generate();
        assert!(escalation.as_str().starts_with("esc_"));
        assert_eq!(EscalationId::parse(escalation.as_str()).unwrap(), escalation);
    }

    #[test]
    fn an_escalation_id_keeps_its_own_prefix() {
        // The prefix is what makes an id self-describing in a log line and in a CLI argument;
        // an `esc_` that parsed as a task id would let `immortality approve tsk_…` resolve an
        // escalation that does not exist.
        let escalation = EscalationId::generate();
        assert!(TaskId::parse(escalation.as_str()).is_err());
        assert!(EscalationId::parse(&TaskId::generate().to_string()).is_err());
    }

    #[test]
    fn prefixed_id_rejects_wrong_prefix() {
        let task = TaskId::generate();
        assert!(SubtaskId::parse(task.as_str()).is_err());
        assert!(TaskId::parse("tsk_not-a-ulid").is_err());
    }

    #[test]
    fn prefixed_id_exposes_timestamp() {
        let task = TaskId::generate();
        assert!(task.timestamp_ms().unwrap() > 1_700_000_000_000);
    }

    #[test]
    fn nonces_are_unique() {
        let a = Nonce::generate();
        let b = Nonce::generate();
        assert_ne!(a, b);
        assert_eq!(a.as_str().len(), 26);
    }
}
