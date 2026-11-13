//! One timestamp format for everything that lands on disk.
//!
//! [`chrono`]'s default serde representation prints nanoseconds when they are non-zero and
//! omits them when they are zero, so the same field is `...:03.221Z` in one record and
//! `...:03Z` in the next. That is legal RFC 3339 and useless in an append-only journal:
//! records are compared, sorted and grepped as text, and a column that changes width breaks
//! naive readers.
//!
//! Here everything is truncated to milliseconds before it is stored, so a timestamp has
//! exactly one spelling and lexicographic order matches chronological order. Truncation also
//! means the value written is the value read: an event cannot load as `03.2210001Z` after
//! being written as `03.221Z`.

use chrono::{SubsecRound as _, Utc};

use crate::domain::Timestamp;

/// Current time, truncated to the journal's resolution.
#[must_use]
pub fn now() -> Timestamp {
    Utc::now().trunc_subsecs(3)
}

/// Where the current time comes from.
///
/// # Why this is a trait and not the free [`now`]
///
/// Three things in this system are measured in wall time: an attempt's 45-minute budget, the
/// 30-day artifact TTL and the 15-minute backend-outage escalation. A test that waits for any of
/// them is not a test, and a golden journal whose `ts` fields come from the machine clock is not
/// a golden. Runs that must be reproducible — the simulator (M4.8), most of all — take a clock;
/// production takes [`SystemClock`].
///
/// It is deliberately *not* a global: two tests in one process would share it, and the order they
/// happen to run in would decide their results.
pub trait Clock: std::fmt::Debug + Send + Sync {
    /// The current instant, at journal resolution.
    fn now(&self) -> Timestamp;
}

/// The system clock, the only implementation production uses.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Timestamp {
        now()
    }
}

/// Seconds from the Unix epoch to 2026-11-12T10:00:00Z, the documentation's fixture date.
///
/// Written as a constant offset rather than a calendar construction so that
/// [`FixedClock::fixture`] cannot fail: it is used by tests and by the simulator, and a fixture
/// date that panics is a fixture every call site has to handle. A test pins the constant to the
/// date it is supposed to be.
const FIXTURE_SECONDS: i64 = 1_794_477_600;

/// A clock that does not move on its own.
///
/// `advance` is explicit because a simulator has to be able to say "the 45 minutes passed"
/// without spending them. It is also the seeded value a golden journal is measured from: the
/// first record sits at `start`, so a normalised journal has offsets rather than dates.
#[derive(Debug)]
pub struct FixedClock {
    start: Timestamp,
    elapsed: std::sync::Mutex<chrono::Duration>,
}

impl FixedClock {
    /// A clock frozen at `start`.
    #[must_use]
    pub const fn at(start: Timestamp) -> Self {
        Self {
            start,
            elapsed: std::sync::Mutex::new(chrono::Duration::zero()),
        }
    }

    /// The fixture instant the documentation uses: `2026-11-12T10:00:00Z`.
    #[must_use]
    pub fn fixture() -> Self {
        Self::at(Timestamp::UNIX_EPOCH + chrono::Duration::seconds(FIXTURE_SECONDS))
    }

    /// Move forward. Never panics on a poisoned lock: a poisoned clock is a test that already
    /// failed, and returning the time hides one failure instead of two.
    pub fn advance(&self, by: chrono::Duration) {
        let mut elapsed = self
            .elapsed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *elapsed += by;
    }

    /// Move forward by whole seconds.
    pub fn advance_secs(&self, secs: i64) {
        self.advance(chrono::Duration::seconds(secs));
    }
}

impl Clock for FixedClock {
    fn now(&self) -> Timestamp {
        let elapsed = *self
            .elapsed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        truncate(self.start + elapsed)
    }
}

/// Truncate an arbitrary instant to the resolution the journal stores.
#[must_use]
pub fn truncate(at: Timestamp) -> Timestamp {
    at.trunc_subsecs(3)
}

/// Serde helpers that keep the format fixed at milliseconds.
///
/// Used as `#[serde(with = "crate::time::millis")]`, which keeps the JSON a plain string
/// rather than an object — a record line stays readable when a human greps it.
pub mod millis {
    use chrono::{DateTime, SecondsFormat, SubsecRound as _, Utc};
    use serde::{Deserialize, Deserializer, Serializer};

    use super::Timestamp;

    /// `2026-11-12T10:14:03.221Z`.
    pub fn serialize<S: Serializer>(at: &Timestamp, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&format_rfc3339(at))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Timestamp, D::Error> {
        // Owned `String`, not `&str`: the journal reader hands over a borrowed line, but
        // `serde_json::Value`-based callers (tests, the CLI) hand over a fresh allocation.
        let text = String::deserialize(deserializer)?;
        DateTime::parse_from_rfc3339(&text)
            .map(|at| at.with_timezone(&Utc))
            .map_err(serde::de::Error::custom)
    }

    /// Format without going through `chrono`'s variable-precision formatter.
    fn format_rfc3339(at: &Timestamp) -> String {
        at.trunc_subsecs(3).to_rfc3339_opts(SecondsFormat::Millis, true)
    }
}

/// The same fixed-width format, for a field that may be absent.
///
/// Used as `#[serde(with = "crate::time::millis_opt")]` with `skip_serializing_if` for the
/// `Option` itself, so an unresolved timestamp is absent rather than `null`. A `null` would be
/// a second spelling of "absent" in a file a human reads.
pub mod millis_opt {
    use chrono::{DateTime, SecondsFormat, SubsecRound as _, Utc};
    use serde::{Deserialize, Deserializer, Serializer};

    use super::Timestamp;

    /// `2026-11-12T10:14:03.221Z`, or nothing.
    pub fn serialize<S: Serializer>(at: &Option<Timestamp>, serializer: S) -> Result<S::Ok, S::Error> {
        match at {
            Some(at) => {
                serializer.serialize_some(&at.trunc_subsecs(3).to_rfc3339_opts(SecondsFormat::Millis, true))
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Timestamp>, D::Error> {
        Option::<String>::deserialize(deserializer)?
            .map(|text| {
                DateTime::parse_from_rfc3339(&text)
                    .map(|at| at.with_timezone(&Utc))
                    .map_err(serde::de::Error::custom)
            })
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone as _;

    #[test]
    fn truncation_keeps_milliseconds() {
        let at = Utc.with_ymd_and_hms(2026, 11, 12, 10, 14, 3).unwrap()
            + chrono::Duration::nanoseconds(221_999_999);
        assert_eq!(truncate(at).timestamp_subsec_millis(), 221);
    }

    #[test]
    fn serialized_form_is_fixed_width() {
        let a = Utc.with_ymd_and_hms(2026, 11, 12, 10, 14, 3).unwrap();
        let b = a + chrono::Duration::milliseconds(5);
        let (ja, jb) = (to_json(a), to_json(b));
        assert_eq!(ja.len(), jb.len(), "{ja} vs {jb}");
        assert!(ja.contains(".000Z"), "{ja}");
        assert!(jb.contains(".005Z"), "{jb}");
    }

    #[test]
    fn format_roundtrips() {
        let at = Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap() + chrono::Duration::milliseconds(678);
        assert_eq!(from_json(&to_json(at)), at);
    }

    #[test]
    fn an_optional_timestamp_roundtrips_both_ways() {
        let at = Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap() + chrono::Duration::milliseconds(9);
        for value in [Some(at), None] {
            let text = serde_json::to_value(OptWrapper { at: value }).unwrap();
            if value.is_none() {
                // Absent, not `null`. The escalation file is read by humans.
                assert_eq!(text, serde_json::json!({}), "{text}");
            } else {
                let stamp = text["at"].as_str().unwrap();
                assert!(stamp.contains(".009Z"), "{text}");
            }
            let back: OptWrapper = serde_json::from_value(text).unwrap();
            assert_eq!(back.at, value);
        }
    }

    #[test]
    fn text_order_matches_chronological_order() {
        let base = Utc.with_ymd_and_hms(2026, 11, 12, 10, 14, 3).unwrap();
        let mut previous = to_json(base);
        for offset in [1, 10, 100, 1_000, 60_000, 86_400_000, 31 * 86_400_000] {
            let next = to_json(base + chrono::Duration::milliseconds(offset));
            assert!(previous < next, "{previous} !< {next}");
            previous = next;
        }
    }

    fn to_json(at: Timestamp) -> String {
        serde_json::to_value(Wrapper { at })
            .unwrap()
            .as_object()
            .unwrap()
            .get("at")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    }

    fn from_json(text: &str) -> Timestamp {
        serde_json::from_value::<Wrapper>(serde_json::json!({ "at": text }))
            .unwrap()
            .at
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    struct Wrapper {
        #[serde(with = "millis")]
        at: Timestamp,
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    struct OptWrapper {
        #[serde(with = "millis_opt", skip_serializing_if = "Option::is_none", default)]
        at: Option<Timestamp>,
    }
    #[test]
    fn the_system_clock_matches_the_free_function() {
        // The trait is a seam, not a second definition of "now": a caller that took the clock
        // must get the same answer as one that did not, or a mixed run would order records by
        // two different clocks.
        let clock = SystemClock;
        let before = now();
        let seen = clock.now();
        let after = now();
        assert!(before <= seen && seen <= after, "{before} {seen} {after}");
    }

    #[test]
    fn a_fixed_clock_does_not_move_until_told_to() {
        let clock = FixedClock::fixture();
        assert_eq!(clock.now(), clock.now());
        assert_eq!(clock.now().to_rfc3339(), "2026-11-12T10:00:00+00:00");
    }

    #[test]
    fn advancing_a_fixed_clock_moves_it_by_exactly_that_much() {
        // This is what makes "the 45-minute budget ran out" a microsecond test instead of a
        // 45-minute one.
        let clock = FixedClock::fixture();
        clock.advance_secs(45 * 60);
        assert_eq!(clock.now().to_rfc3339(), "2026-11-12T10:45:00+00:00");
    }

    #[test]
    fn a_fixed_clock_keeps_journal_resolution() {
        let clock = FixedClock::fixture();
        clock.advance(chrono::Duration::nanoseconds(221_999_999));
        assert_eq!(clock.now().timestamp_subsec_millis(), 221);
        // Fixed width: the same truncation the writer applies, so a seeded journal has one
        // spelling per timestamp.
        assert!(to_json(clock.now()).contains(".221Z"));
    }
}
