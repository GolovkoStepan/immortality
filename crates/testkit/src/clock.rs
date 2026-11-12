//! A clock the test controls.
//!
//! # Why not `chrono::Utc::now()`
//!
//! Every deadline in this system is long: 30-day artifact TTL, 12-month log retention, a 45-minute
//! attempt budget, 15 minutes of backend unavailability before an escalation. A test that waits
//! for any of them is not a test. Worse, a TTL test written against the real clock depends on
//! when it runs, so it passes in November and fails in December.
//!
//! # Monotone, and honest about it
//!
//! [`Clock`] reports a wall time, and [`Clock::advance`] moves it forward. It can also be moved
//! *backwards*, deliberately: an NTP step and a user changing the timezone both happen in
//! production, and `gc` has explicit guards for a file dated in the future. A clock that refused
//! to go backwards could not test them.

use std::sync::Mutex;

/// A wall clock that a test can move.
#[derive(Debug)]
pub struct Clock {
    now: Mutex<chrono::DateTime<chrono::Utc>>,
}

impl Clock {
    /// A clock frozen at `now`.
    #[must_use]
    pub const fn at(now: chrono::DateTime<chrono::Utc>) -> Self {
        Self { now: Mutex::new(now) }
    }

    /// A clock frozen at 2026-11-12T10:00:00Z, the date the documentation uses.
    #[must_use]
    pub fn fixture() -> Self {
        use chrono::TimeZone as _;
        Self::at(
            chrono::Utc
                .with_ymd_and_hms(2026, 11, 12, 10, 0, 0)
                .single()
                .expect("the fixture date is valid"),
        )
    }

    /// The current wall time.
    #[must_use]
    pub fn now(&self) -> chrono::DateTime<chrono::Utc> {
        *self.lock()
    }

    /// Move forward by `duration`.
    pub fn advance(&self, duration: chrono::Duration) {
        let mut now = self.lock();
        *now += duration;
    }

    /// Move forward by whole days.
    pub fn advance_days(&self, days: i64) {
        self.advance(chrono::Duration::days(days));
    }

    /// Move forward by whole seconds.
    pub fn advance_secs(&self, secs: i64) {
        self.advance(chrono::Duration::seconds(secs));
    }

    /// Move backwards, for testing clock-step guards.
    pub fn rewind(&self, duration: chrono::Duration) {
        let mut now = self.lock();
        *now -= duration;
    }

    /// Set the wall time outright.
    pub fn set(&self, now: chrono::DateTime<chrono::Utc>) {
        *self.lock() = now;
    }

    /// The current time as the local zone, which is what `gc` and logging take.
    #[must_use]
    pub fn now_local(&self) -> chrono::DateTime<chrono::Local> {
        self.now().with_timezone(&chrono::Local)
    }

    /// Seconds since the Unix epoch, the form artifact TTLs are compared in.
    #[must_use]
    pub fn unix_secs(&self) -> u64 {
        u64::try_from(self.now().timestamp()).unwrap_or(0)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, chrono::DateTime<chrono::Utc>> {
        // A poisoned clock is a test that already failed; returning the time is more useful than
        // a second panic that hides the first.
        self.now.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::fixture()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fixture_clock_agrees_with_the_documentation_date() {
        let clock = Clock::fixture();
        assert_eq!(clock.now().to_rfc3339(), "2026-11-12T10:00:00+00:00");
    }

    #[test]
    fn advancing_moves_the_wall_clock() {
        let clock = Clock::fixture();
        clock.advance_days(30);
        assert_eq!(clock.now().to_rfc3339(), "2026-12-12T10:00:00+00:00");
    }

    #[test]
    fn advancing_by_days_crosses_a_month_boundary() {
        // Log retention is measured in months; a test that cannot cross a boundary cannot test
        // the retention rule that matters.
        let clock = Clock::fixture();
        clock.advance_days(31);
        assert_eq!(clock.now_local().format("%Y-%m").to_string(), "2026-12");
    }

    #[test]
    fn rewinding_is_allowed_because_ntp_steps_are_real() {
        let clock = Clock::fixture();
        clock.rewind(chrono::Duration::hours(1));
        assert_eq!(clock.now().to_rfc3339(), "2026-11-12T09:00:00+00:00");
    }

    #[test]
    fn setting_the_time_outright_works() {
        let clock = Clock::fixture();
        clock.set(chrono::DateTime::from_timestamp(0, 0).unwrap());
        assert_eq!(clock.unix_secs(), 0);
    }

    #[test]
    fn seconds_and_local_time_agree_with_now() {
        let clock = Clock::fixture();
        assert_eq!(clock.unix_secs(), u64::try_from(clock.now().timestamp()).unwrap());
        assert_eq!(clock.now_local().timestamp(), clock.now().timestamp());
    }

    #[test]
    fn the_clock_is_usable_from_several_threads() {
        let clock = std::sync::Arc::new(Clock::fixture());
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let clock = std::sync::Arc::clone(&clock);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        clock.advance_secs(1);
                        let _ = clock.now();
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(clock.now().to_rfc3339(), "2026-11-12T10:06:40+00:00");
    }
}
