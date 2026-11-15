//! Telling the human something happened, without a library or a hang.
//!
//! v0.1 has exactly one outbound channel: a macOS notification via `osascript`. The design
//! deliberately stops there — no email, no webhook, no push service — because every additional
//! channel is a network dependency and a credential, and this is a tool whose selling point is
//! that it runs for days with nobody watching.
//!
//! # Why this cannot fail the caller
//!
//! An escalation is recorded on disk before the notification is attempted, so the question is
//! never lost if `osascript` is missing, refused by notification permissions, or running in a
//! headless CI job. The notification is a courtesy on top of the durable record, and
//! [`Notifier::notify`] returns nothing: there is no failure for a caller to handle, and
//! therefore none to mishandle. It is logged at `debug` and forgotten.
//!
//! # Why the text is argv, not part of the script
//!
//! `osascript -e "display notification \"...\""` interpolates the question into an
//! `AppleScript`
//! literal. A question containing a quote — and every question the model writes may — is then
//! either a syntax error or, worse, an injection that runs whatever followed. The
//! script here is a fixed `on run argv` handler and the text arrives as arguments, which is the
//! only shape where the payload cannot change the program.
//!
//! # Why the wait is bounded
//!
//! `osascript` normally answers in tens of milliseconds. It can also block on a permissions
//! prompt that never appears in a headless session, and a daemon that waits forever on a
//! notification is a daemon that stopped working because of a toast. The child is polled for a
//! short deadline and killed if it overruns; the poll is a sleep loop rather than a thread,
//! because a thread waiting on a child that never exits is a leak with extra steps.

use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long a notification may take before it is abandoned.
///
/// Generous for a program that usually takes ~50 ms, short enough that a stuck one costs a
/// fraction of a step.
pub const NOTIFY_TIMEOUT: Duration = Duration::from_millis(1_500);

/// The fixed script. `argv` is `[body, title]`.
const SCRIPT: &str = "on run argv\n\
     display notification (item 1 of argv) with title (item 2 of argv)\n\
     end run";

/// Something that can raise a notification.
///
/// A trait rather than a direct `Command` call so a test can assert *that* the human was told,
/// and how many times, without spawning anything — and so a future channel (a log line, a
/// socket push to the TUI) can be added without touching the escalation tool.
pub trait Notifier: Send + Sync + std::fmt::Debug {
    /// Raise a notification. Never reports failure: the durable record already exists.
    fn notify(&self, title: &str, body: &str);
}

/// Does nothing. The default in tests and in a build where notifications are off.
#[derive(Debug, Clone, Copy, Default)]
pub struct Silent;

impl Notifier for Silent {
    fn notify(&self, _title: &str, _body: &str) {}
}

/// A macOS notification through `osascript`.
///
/// Which notifier is in force is the *daemon's* decision, not this module's. Reading the
/// environment here would make every test depend on global state and would leave the caller
/// unable to substitute one; instead the daemon picks `MacNotifier` or [`Silent`] once, at
/// startup, and everything downstream sees only [`Notifier`].
#[derive(Debug, Clone)]
pub struct MacNotifier {
    /// The program to run. Defaults to `/usr/bin/osascript`; a test points it elsewhere.
    program: std::path::PathBuf,
}

impl Default for MacNotifier {
    fn default() -> Self {
        Self {
            program: std::path::PathBuf::from(DEFAULT_PROGRAM),
        }
    }
}

impl MacNotifier {
    /// The real thing.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A notifier that runs `program` instead of `/usr/bin/osascript`.
    #[must_use]
    pub fn with_program(program: impl Into<std::path::PathBuf>) -> Self {
        Self {
            program: program.into(),
        }
    }

    /// The program this notifier runs.
    #[must_use]
    pub fn program(&self) -> &std::path::Path {
        &self.program
    }
}

/// Where `osascript` lives on macOS.
pub const DEFAULT_PROGRAM: &str = "/usr/bin/osascript";

/// The notifier a daemon should use: the real one on macOS, silence elsewhere.
///
/// # Why the platform decides and not the environment
///
/// `osascript` exists only on macOS, so on Linux the honest answer is [`Silent`]: an escalation
/// is still recorded durably, and `escalations`/`status` still show it. Picking by an environment
/// variable would make the same binary notify on one machine and not another for a reason nothing
/// records, and a test would then be exercising a different channel than production.
///
/// This is the *daemon's* choice, made once and handed to every task's toolset. A caller that
/// needs a different channel — a test's recorder — passes its own to
/// [`crate::Escalate`] or [`immortality_core`]-free `notifying`, because substituting here would
/// be a global.
#[must_use]
pub fn system() -> Arc<dyn Notifier> {
    #[cfg(target_os = "macos")]
    {
        Arc::new(MacNotifier::new())
    }
    #[cfg(not(target_os = "macos"))]
    {
        Arc::new(Silent)
    }
}

impl Notifier for MacNotifier {
    fn notify(&self, title: &str, body: &str) {
        // `--` ends option parsing so a body starting with `-e` cannot become a flag.
        let child = Command::new(&self.program)
            .arg("-e")
            .arg(SCRIPT)
            .arg("--")
            .arg(body)
            .arg(title)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        let Ok(child) = child else {
            // No `osascript` (a headless box, a Linux CI runner): the escalation file already
            // has the question. Nothing to do and nothing to report upward.
            tracing::debug!("notification skipped: osascript could not be started");
            return;
        };
        if reap(child, NOTIFY_TIMEOUT).is_none() {
            tracing::debug!("notification skipped: osascript did not finish in time");
        }
    }
}

/// Wait for `child`, killing it after `limit`.
///
/// Returns the exit status, or `None` if the deadline passed. The child is killed and reaped in
/// that case, so a stuck `osascript` leaves no zombie behind.
fn reap(mut child: Child, limit: Duration) -> Option<std::process::ExitStatus> {
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) => {}
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
        if started.elapsed() >= limit {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shell script standing in for `osascript`, which writes its arguments to a file.
    fn fake_osascript(dir: &std::path::Path, script_body: &str) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt as _;
        let path = dir.join("fake-osascript");
        std::fs::write(&path, format!("#!/bin/sh\n{script_body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[test]
    fn the_silent_notifier_records_nothing_and_does_not_panic() {
        Silent.notify("title", "body");
    }

    #[test]
    fn the_system_notifier_is_the_real_one_only_where_osascript_exists() {
        // The production toolset's default used to be `Silent`, so an escalation was written to
        // disk and nobody was told. `system()` is the one place the choice is made, and the
        // choice is the platform's: `osascript` does not exist off macOS, and a daemon that
        // claimed to notify there would be lying in the one function that decides.
        let notifier = system();
        let debug = format!("{notifier:?}");
        if cfg!(target_os = "macos") {
            assert!(
                debug.contains("MacNotifier"),
                "on macOS the daemon must try to notify: {debug}"
            );
        } else {
            assert!(
                debug.contains("Silent"),
                "without osascript the honest notifier is silence: {debug}"
            );
        }
    }

    #[test]
    fn the_body_and_title_arrive_as_argv_at_fixed_positions() {
        // The point of argv: this body contains a quote and a `$(...)`, and the script is fixed
        // so neither can become code. Asserting *positions* rather than mere presence is what
        // makes that testable — an implementation that interpolated the body into the `-e`
        // script would still contain the text, just in the wrong argument.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("argv");
        // NUL-separated, because the script itself contains newlines and a line split would
        // make one argument look like three.
        let program = fake_osascript(
            dir.path(),
            &format!("printf '%s\\0' \"$@\" > '{}'", log.display()),
        );
        let body = "may I \"run\" $(rm -rf /) this?";
        MacNotifier::with_program(&program).notify("immortality", body);

        let raw = std::fs::read(&log).unwrap();
        let args: Vec<&str> = raw
            .split(|byte| *byte == 0)
            .filter(|part| !part.is_empty())
            .map(|part| std::str::from_utf8(part).unwrap())
            .collect();
        assert_eq!(args, vec!["-e", SCRIPT, "--", body, "immortality"]);
    }

    #[test]
    fn a_missing_program_is_not_an_error() {
        // The headless case the roadmap names. Nothing to return, nothing to unwind.
        let dir = tempfile::tempdir().unwrap();
        MacNotifier::with_program(dir.path().join("does-not-exist")).notify("title", "body");
    }

    #[test]
    fn a_program_that_fails_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let program = fake_osascript(dir.path(), "exit 7");
        MacNotifier::with_program(&program).notify("title", "body");
    }

    #[test]
    fn a_stuck_program_is_killed_rather_than_waited_on() {
        // The failure this prevents: `osascript` blocking on a permissions prompt in a session
        // where nobody can answer, and the daemon stopping because of a toast.
        let dir = tempfile::tempdir().unwrap();
        let program = fake_osascript(dir.path(), "sleep 30");
        let started = Instant::now();
        MacNotifier::with_program(&program).notify("title", "body");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the call must return once the deadline passes, took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn the_default_program_is_the_one_macos_ships() {
        assert_eq!(
            MacNotifier::new().program(),
            std::path::Path::new(DEFAULT_PROGRAM)
        );
    }

    #[test]
    fn the_fake_program_is_genuinely_invoked() {
        // Without this, the "missing program" and "stuck program" tests would also pass against
        // a notifier that never spawns anything, and would prove nothing.
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("spawned");
        let program = fake_osascript(dir.path(), &format!("touch '{}'", marker.display()));
        MacNotifier::with_program(&program).notify("title", "body");
        assert!(marker.exists());
    }
}
