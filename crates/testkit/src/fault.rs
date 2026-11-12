//! Deterministic fault injection.
//!
//! # Why a plan, not a random failure rate
//!
//! A test that says "fail 5% of writes" either never fires or fires somewhere unrelated, and a
//! failure that moves between runs cannot be bisected. So a test names a hook and an occurrence:
//! *the second append to this journal fails*. That is reproducible, it fails for the reason the
//! test is about, and when it stops firing the test fails instead of passing vacuously.
//!
//! # Why hooks are strings
//!
//! A closed enum in this crate would have to name every operation in the system, including ones
//! that do not exist yet, and M4's runner would be unable to add a hook without editing the
//! testkit. [`Hook`] is a newtype over `&'static str` with constants for the operations that
//! exist today, so a typo is a compile error at the call site and a new hook costs nothing.
//!
//! # What a fault is not
//!
//! [`Faults`] does not touch the filesystem. It answers "should this operation fail, and how" —
//! the caller performs the failure. That keeps the injection point next to the operation it
//! interrupts, which is the only place the failure is meaningful: failing "before `write_all`"
//! and "after `write_all`" are different bugs and only the caller knows which one it is testing.

use std::collections::BTreeMap;
use std::sync::Mutex;

/// A point in the system where a test can inject a fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Hook(&'static str);

impl Hook {
    /// The journal is about to append a line.
    pub const JOURNAL_APPEND: Self = Self("journal.append");
    /// The journal has written a line and is about to `fsync` it.
    pub const JOURNAL_SYNC: Self = Self("journal.sync");
    /// A journal segment is about to rotate.
    pub const JOURNAL_ROTATE: Self = Self("journal.rotate");
    /// An artifact is about to be renamed into place.
    pub const ARTIFACT_RENAME: Self = Self("artifact.rename");
    /// An artifact is about to be read back.
    pub const ARTIFACT_READ: Self = Self("artifact.read");
    /// A git command is about to run.
    pub const GIT_COMMAND: Self = Self("git.command");
    /// A request is about to be sent to the backend.
    pub const MODEL_REQUEST: Self = Self("model.request");
    /// A model response has arrived and its body is about to be recorded.
    pub const MODEL_RESPONSE: Self = Self("model.response");
    /// A tool is about to execute.
    pub const TOOL_EXECUTE: Self = Self("tool.execute");
    /// Recovery has computed a plan and is about to execute it.
    pub const RECOVERY_EXECUTE: Self = Self("recovery.execute");

    /// A hook for an operation this crate does not know about.
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    /// The name, for failure messages.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl std::fmt::Display for Hook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// What the injected fault does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// The operation fails with this `io::ErrorKind` before it happens.
    Fail(std::io::ErrorKind),
    /// The operation succeeds, but the process dies immediately afterwards: a real `kill -9`
    /// in production terms, a `abort()` in a test. The "afterwards" is the caller's job, which
    /// is what makes a mid-write crash testable.
    CrashAfter,
    /// The process dies before the operation happens, so the operation never ran. Distinct from
    /// [`CrashAfter`](Self::CrashAfter) because recovery must treat them differently: an
    /// interrupted `write_all` may have left a partial line, an unstarted one cannot have.
    CrashBefore,
}

impl Action {
    /// Whether the injected fault kills the process.
    ///
    /// Public because the caller that receives a fault has to branch: an error is returned, a
    /// crash is performed.
    #[must_use]
    pub const fn is_crash(self) -> bool {
        matches!(self, Self::CrashAfter | Self::CrashBefore)
    }

    /// The `io::ErrorKind` this action produces, if it produces one.
    #[must_use]
    pub const fn error_kind(self) -> Option<std::io::ErrorKind> {
        match self {
            Self::Fail(kind) => Some(kind),
            Self::CrashAfter | Self::CrashBefore => None,
        }
    }
}

/// A fault that fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fault {
    pub hook: Hook,
    pub action: Action,
    /// Which occurrence of the hook this was, 1-based.
    pub occurrence: u64,
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "injected fault at `{}` (occurrence {}): {:?}",
            self.hook, self.occurrence, self.action
        )
    }
}

/// An armed set of faults.
///
/// Shared by reference across threads: the daemon is single-threaded per task, but tests run in
/// parallel and a fault plan is cheap to lock.
#[derive(Debug)]
pub struct Faults {
    inner: Mutex<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    /// `(hook, occurrence) -> action`.
    armed: BTreeMap<(Hook, u64), Action>,
    /// How many times each hook has been checked.
    seen: BTreeMap<Hook, u64>,
}

impl Faults {
    /// No faults armed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
        }
    }

    /// Fail the `occurrence`-th call of `hook` with `kind`.
    ///
    /// Occurrences are 1-based so that "the first write" reads as `1` and not as an off-by-one
    /// puzzle. Arming the same `(hook, occurrence)` twice replaces the action rather than
    /// stacking, because two different failures at one instant is not a state the system can be
    /// in and would make the test assert something impossible.
    pub fn fail_at(&self, hook: Hook, occurrence: u64, kind: std::io::ErrorKind) {
        self.arm(hook, occurrence, Action::Fail(kind));
    }

    /// Crash after the `occurrence`-th call of `hook` completes.
    pub fn crash_after(&self, hook: Hook, occurrence: u64) {
        self.arm(hook, occurrence, Action::CrashAfter);
    }

    /// Crash before the `occurrence`-th call of `hook` starts.
    pub fn crash_before(&self, hook: Hook, occurrence: u64) {
        self.arm(hook, occurrence, Action::CrashBefore);
    }

    fn arm(&self, hook: Hook, occurrence: u64, action: Action) {
        debug_assert!(occurrence > 0, "occurrences are 1-based");
        self.lock().armed.insert((hook, occurrence), action);
    }

    /// Count a call of `hook` and report the fault, if any.
    ///
    /// `CrashBefore` returns the fault *and* does not increment past the crash: the operation is
    /// treated as never having happened, which is what the caller then does.
    pub fn check(&self, hook: Hook) -> Option<Fault> {
        let mut inner = self.lock();
        let occurrence = inner.seen.entry(hook).or_insert(0);
        *occurrence += 1;
        let occurrence = *occurrence;
        let action = *inner.armed.get(&(hook, occurrence))?;
        drop(inner);
        Some(Fault {
            hook,
            action,
            occurrence,
        })
    }

    /// How many times `hook` has been checked, for assertions after the fact.
    #[must_use]
    pub fn occurrences(&self, hook: Hook) -> u64 {
        self.lock().seen.get(&hook).copied().unwrap_or(0)
    }

    /// Whether an armed fault never fired.
    ///
    /// A test asserts this so that an injection point that was removed, renamed or moved out of
    /// the executed path fails the test instead of silently passing.
    #[must_use]
    pub fn unfired(&self) -> Vec<(Hook, u64, Action)> {
        let inner = self.lock();
        inner
            .armed
            .iter()
            .filter(|((hook, occurrence), _)| inner.seen.get(hook).copied().unwrap_or(0) < *occurrence)
            .map(|((hook, occurrence), action)| (*hook, *occurrence, *action))
            .collect()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        // A poisoned lock means a test panicked while holding it. The plan is still readable and
        // the caller is about to fail anyway; refusing to run would turn one failure into many.
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl Default for Faults {
    fn default() -> Self {
        Self::new()
    }
}

/// Turn a fault into the `io::Error` the caller should return.
///
/// A crash action produces `ErrorKind::Other` with a distinctive message rather than a panic:
/// the caller decides how to stop, and a test that wants a real process death uses
/// [`crate::crash`].
#[must_use]
pub fn error_for(fault: Fault) -> std::io::Error {
    fault.action.error_kind().map_or_else(
        || std::io::Error::other(fault.to_string()),
        |kind| std::io::Error::new(kind, fault.to_string()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_fires_despite_no_plan() {
        let faults = Faults::new();
        for _ in 0..5 {
            assert_eq!(faults.check(Hook::JOURNAL_APPEND), None);
        }
        assert_eq!(faults.occurrences(Hook::JOURNAL_APPEND), 5);
    }

    #[test]
    fn a_fault_fires_on_the_named_occurrence_and_nowhere_else() {
        // The central property: a test that says "the second write" fails on the second write,
        // not on the first and not on the third.
        let faults = Faults::new();
        faults.fail_at(Hook::JOURNAL_APPEND, 2, std::io::ErrorKind::StorageFull);

        assert_eq!(faults.check(Hook::JOURNAL_APPEND), None, "first call");
        let fault = faults.check(Hook::JOURNAL_APPEND).expect("second call");
        assert_eq!(fault.occurrence, 2);
        assert_eq!(fault.action, Action::Fail(std::io::ErrorKind::StorageFull));
        assert_eq!(faults.check(Hook::JOURNAL_APPEND), None, "third call");
    }

    #[test]
    fn hooks_are_counted_independently() {
        let faults = Faults::new();
        faults.fail_at(Hook::ARTIFACT_READ, 1, std::io::ErrorKind::NotFound);
        assert_eq!(faults.check(Hook::JOURNAL_APPEND), None);
        assert!(faults.check(Hook::ARTIFACT_READ).is_some());
    }

    #[test]
    fn rearming_replaces_the_action() {
        let faults = Faults::new();
        faults.fail_at(Hook::GIT_COMMAND, 1, std::io::ErrorKind::PermissionDenied);
        faults.crash_after(Hook::GIT_COMMAND, 1);
        let fault = faults.check(Hook::GIT_COMMAND).expect("armed");
        assert_eq!(fault.action, Action::CrashAfter);
    }

    #[test]
    fn an_unfired_fault_is_reported() {
        // This is what stops a test from passing because its injection point was renamed away.
        let faults = Faults::new();
        faults.crash_after(Hook::JOURNAL_SYNC, 3);
        faults.check(Hook::JOURNAL_SYNC);
        assert_eq!(faults.unfired(), [(Hook::JOURNAL_SYNC, 3, Action::CrashAfter)]);
        faults.check(Hook::JOURNAL_SYNC);
        faults.check(Hook::JOURNAL_SYNC);
        assert_eq!(faults.unfired(), []);
    }

    #[test]
    fn crash_actions_carry_no_error_kind() {
        assert_eq!(Action::CrashAfter.error_kind(), None);
        assert_eq!(Action::CrashBefore.error_kind(), None);
        assert_eq!(
            Action::Fail(std::io::ErrorKind::TimedOut).error_kind(),
            Some(std::io::ErrorKind::TimedOut)
        );
    }

    #[test]
    fn an_error_carries_the_kind_the_test_asked_for() {
        let faults = Faults::new();
        faults.fail_at(Hook::MODEL_REQUEST, 1, std::io::ErrorKind::TimedOut);
        let fault = faults.check(Hook::MODEL_REQUEST).unwrap();
        let error = error_for(fault);
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        assert!(error.to_string().contains("model.request"), "{error}");
    }

    #[test]
    fn one_plan_is_usable_from_several_threads() {
        // The daemon is single-threaded per task, but the testkit is shared across a parallel
        // test run, and a `Mutex` that deadlocks would hang the suite instead of failing it.
        let faults = std::sync::Arc::new(Faults::new());
        faults.crash_after(Hook::TOOL_EXECUTE, 50);
        let crashes: usize = (0..4)
            .map(|_| {
                let faults = std::sync::Arc::clone(&faults);
                std::thread::spawn(move || {
                    (0..50)
                        .filter(|_| {
                            faults
                                .check(Hook::TOOL_EXECUTE)
                                .is_some_and(|fault| fault.action.is_crash())
                        })
                        .count()
                })
            })
            .map(|handle| handle.join().unwrap())
            .sum();
        assert_eq!(crashes, 1, "exactly one of 200 checks must crash");
        assert_eq!(faults.occurrences(Hook::TOOL_EXECUTE), 200);
    }

    #[test]
    fn a_custom_hook_can_be_named_without_editing_the_testkit() {
        let hook = Hook::new("runner.step");
        let faults = Faults::new();
        faults.crash_before(hook, 1);
        assert_eq!(faults.check(hook).unwrap().action, Action::CrashBefore);
        assert_eq!(hook.to_string(), "runner.step");
    }
}
