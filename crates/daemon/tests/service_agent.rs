//! The install/reload/uninstall sequence, driven end to end against a scripted `launchctl`.
//!
//! # Why a script and not a mock
//!
//! `Agent` shells out. A mock in Rust would prove that `Agent` calls something, not that the
//! sequence reaches a process, sees its exit code, and reads its stdout — which is the whole
//! mechanism. The script below is a tiny launchd: `print` reports a job only after `bootstrap`
//! created it, and every invocation is appended to a log, so the *order* is assertable.
//!
//! # Why this must not use the real `launchctl`
//!
//! `bootstrap` on the developer's own `gui/<uid>` domain loads a real agent that then starts a real
//! daemon and holds the real socket. A test that did that would fail for reasons unrelated to the
//! code (a daemon already running), and would leave the machine changed. The seam exists so that
//! the only processes involved are this test's.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};

// Imported from `launchd` directly rather than through the platform-selected re-export: this file
// drives a scripted `launchctl`, so it is testable on any host, and the re-export points at
// `systemd` on Linux. The same reasoning as `service_systemd.rs`, which runs everywhere too.
use immortality_daemon::service::launchd::{Agent, Launcher};
use immortality_daemon::service::{Installed, ServiceSpec};

/// A launchd stand-in whose "loaded" state is a file.
///
/// Kept as a real script for the reason in the module doc: exit codes and stdout are part of the
/// behaviour under test, and a Rust fake would bypass them.
const FAKE: &str = r#"#!/bin/sh
# $1 is the subcommand; the loaded marker is what `bootstrap`/`bootout` change. The state lives
# beside the script rather than in an environment variable: the test process passes its own
# environment through, and mutating it would mean `set_var`, which this workspace forbids.
state="$(dirname "$0")/launchd"
printf '%s\n' "$*" >> "$state/calls.log"
case "$1" in
  print)
    if [ -f "$state/loaded" ]; then
      printf 'gui/501/dev.immortality.daemon = {\n\tstate = running\n\tpid = 4242\n}\n'
      exit 0
    fi
    printf 'Could not find service "dev.immortality.daemon" in domain for login\n' >&2
    exit 113
    ;;
  bootstrap)
    : > "$state/loaded"
    exit 0
    ;;
  bootout)
    rm -f "$state/loaded"
    exit 0
    ;;
esac
printf 'unsupported: %s\n' "$*" >&2
exit 64
"#;

struct Fixture {
    dir: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = dir.path().join("launchd");
        std::fs::create_dir_all(&state).expect("state");
        let script = dir.path().join("launchctl");
        std::fs::write(&script, FAKE).expect("write script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        }
        Self { dir }
    }

    fn agent(&self) -> Agent {
        Agent::at(
            self.dir.path().join("LaunchAgents").join("agent.plist"),
            Launcher::new(self.dir.path().join("launchctl")),
        )
    }

    fn calls(&self) -> Vec<String> {
        let path = self.dir.path().join("launchd").join("calls.log");
        std::fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    fn spec(&self) -> ServiceSpec {
        ServiceSpec {
            program: self.dir.path().join("bin").join("immortality"),
            state_root: self.dir.path().join("state"),
            log_path: self.dir.path().join("state").join("logs").join("daemon.log"),
            path: None,
            lang: None,
        }
    }
}

fn index_of(calls: &[String], prefix: &str) -> usize {
    calls
        .iter()
        .position(|call| call.starts_with(prefix))
        .unwrap_or_else(|| panic!("no `{prefix}` call in {calls:?}"))
}

#[test]
fn installing_twice_does_not_disturb_a_healthy_agent() {
    // The roadmap's idempotency requirement, and the one with teeth: reloading a loaded job kills
    // the daemon and every attempt it was in the middle of. A second `service install` is what a
    // setup script does, and it must be a no-op.
    let fixture = Fixture::new();
    let agent = fixture.agent();
    let spec = fixture.spec();

    assert_eq!(agent.install(&spec).expect("first install"), Installed::Fresh);
    let first = fixture.calls();
    assert_eq!(
        first.iter().filter(|call| call.starts_with("bootstrap")).count(),
        1,
        "{first:?}"
    );

    assert_eq!(
        agent.install(&spec).expect("second install"),
        Installed::Unchanged
    );
    let second = fixture.calls();
    assert_eq!(
        second.len(),
        first.len() + 1,
        "the second install must only ask `print`: {second:?}"
    );
    assert!(second.last().expect("a call").starts_with("print"), "{second:?}");
    assert_eq!(
        second.iter().filter(|call| call.starts_with("bootstrap")).count(),
        1,
        "a healthy agent was reloaded: {second:?}"
    );
    assert!(
        !second.iter().any(|call| call.starts_with("bootout")),
        "a healthy agent was unloaded: {second:?}"
    );
}

#[test]
fn a_changed_plist_is_reloaded_in_unload_then_load_order() {
    // A human who reinstalls after moving the binary or the state directory expects the running
    // daemon to match the file. `bootstrap` over a loaded job fails, so the old one must go first —
    // and the order is the thing that breaks if this is refactored.
    let fixture = Fixture::new();
    let agent = fixture.agent();
    assert_eq!(agent.install(&fixture.spec()).expect("install"), Installed::Fresh);

    let moved = ServiceSpec {
        program: fixture.dir.path().join("elsewhere").join("immortality"),
        ..fixture.spec()
    };
    let before = fixture.calls().len();
    assert_eq!(agent.install(&moved).expect("reinstall"), Installed::Reloaded);

    let tail = &fixture.calls()[before..];
    let unload = index_of(tail, "bootout");
    let load = index_of(tail, "bootstrap");
    assert!(
        unload < load,
        "the old job must be unloaded before the new one is loaded: {tail:?}"
    );
    assert!(
        std::fs::read_to_string(agent.unit())
            .expect("plist")
            .contains(&fixture.dir.path().join("elsewhere").display().to_string()),
        "the written plist must describe the spec that was installed"
    );
}

#[test]
fn uninstalling_twice_is_not_an_error() {
    // `uninstall` in a teardown script runs whether or not the install succeeded. Failing the
    // second time makes it unusable where it matters most.
    let fixture = Fixture::new();
    let agent = fixture.agent();
    agent.install(&fixture.spec()).expect("install");
    assert!(agent.unit().exists());

    agent.uninstall().expect("first uninstall");
    assert!(!agent.unit().exists());
    assert!(
        !fixture.dir.path().join("launchd").join("loaded").exists(),
        "the job must be unloaded, not just the file deleted"
    );

    agent.uninstall().expect("second uninstall is a no-op");
}

#[test]
fn status_separates_installed_from_running() {
    // The distinction a human needs: a crashed job inside its throttle window is installed and not
    // running, and a report that collapsed the two would say the opposite of the truth.
    let fixture = Fixture::new();
    let agent = fixture.agent();

    let before = agent.status();
    assert!(!before.installed);
    assert!(!before.running);
    assert_eq!(before.pid, None);

    agent.install(&fixture.spec()).expect("install");
    let after = agent.status();
    assert!(after.installed);
    assert!(after.running);
    assert_eq!(after.pid, Some(4242));

    // Installed but not running: the file is there, launchd has dropped the job.
    std::fs::remove_file(fixture.dir.path().join("launchd").join("loaded")).expect("unload");
    let crashed = agent.status();
    assert!(crashed.installed, "the plist is still on disk");
    assert!(!crashed.running, "launchd no longer knows the job");
}

#[test]
fn a_failed_load_names_the_path_and_launchds_own_message() {
    // "install failed" is not actionable. The plist is already written when `bootstrap` runs, so
    // the message has to say that the file is in place and what launchd objected to.
    let fixture = Fixture::new();
    let broken = fixture.dir.path().join("broken-launchctl");
    std::fs::write(
        &broken,
        "#!/bin/sh\nprintf 'Bootstrap failed: 5: Input/output error\\n' >&2\nexit 5\n",
    )
    .expect("write");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&broken, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    }
    let agent = Agent::at(
        fixture.dir.path().join("LaunchAgents").join("agent.plist"),
        Launcher::new(&broken),
    );

    let error = agent.install(&fixture.spec()).expect_err("must fail");
    // The script says `Bootstrap failed`; launchd's own wording is passed through unaltered,
    // which is the point — a paraphrase would drop the error number a human searches for.
    assert!(error.contains("Bootstrap failed: 5"), "{error}");
    assert!(error.contains("Input/output error"), "{error}");
    assert!(
        error.contains(&agent.unit().display().to_string()),
        "the message must name the plist: {error}"
    );
    assert!(
        agent.unit().exists(),
        "the plist is written before loading; the message says so and must be true"
    );
}

#[test]
fn the_written_plist_is_the_one_launchd_is_told_to_load() {
    // A very cheap-looking assertion that catches a real class of bug: `install` writing one path
    // and `bootstrap` naming another (say, the discovered plist instead of this agent's).
    let fixture = Fixture::new();
    let agent = fixture.agent();
    agent.install(&fixture.spec()).expect("install");

    let calls = fixture.calls();
    let load = calls
        .iter()
        .find(|call| call.starts_with("bootstrap"))
        .expect("a load");
    assert!(
        load.contains(&agent.unit().display().to_string()),
        "bootstrap was not given this agent's plist: {load}"
    );
    assert_eq!(
        std::fs::read_to_string(agent.unit()).expect("plist"),
        immortality_daemon::service::launchd::plist(&fixture.spec())
    );
}

/// The plist path helper is what `Agent::discover` uses; keep it reachable for the CLI.
#[test]
fn the_discovered_agent_points_into_the_users_launch_agents_directory() {
    let path: PathBuf = immortality_daemon::service::launchd::plist_path().expect("HOME");
    assert!(Path::new(&path).ends_with(format!(
        "Library/LaunchAgents/{}.plist",
        immortality_daemon::service::LABEL
    )));
}
