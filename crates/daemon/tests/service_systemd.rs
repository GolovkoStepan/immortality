//! The install/reload/uninstall sequence, driven end to end against a scripted `systemctl`.
//!
//! # Why this runs on macOS too
//!
//! The four operations are: write an INI file, create a symlink, and call `systemctl --user`
//! twice. Only the last is platform-specific, and it is reached through [`Launcher`], so a script
//! named `systemctl` is enough. Gating this file on `cfg(target_os = "linux")` would mean the
//! sequence is only ever exercised in CI — which is exactly the state D-110 was written to end.
//!
//! # Why a script and not a mock
//!
//! A mock in Rust would prove that `Agent` calls something, not that the sequence reaches a
//! process, sees its exit code, and reads its stdout. The script below is a tiny user manager:
//! `show` reports a unit only after `start` created it, and every invocation is appended to a log
//! so the *order* is assertable.
//!
//! # Why this must not use the real `systemctl`
//!
//! On a developer's Linux machine `--user restart` would stop and start a real daemon that holds
//! the real socket. The seam exists so the only processes involved are this test's.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use immortality_daemon::service::systemd::{Agent, Launcher, unit};
use immortality_daemon::service::{Installed, LABEL, ServiceSpec};

/// A `systemctl --user` stand-in whose "running" state is a file.
///
/// `--user` is asserted by the script rather than ignored: a launcher that dropped it would
/// address the system manager, and on the developer's machine that is a different unit entirely.
const FAKE: &str = r#"#!/bin/sh
state="$(dirname "$0")/systemd"
printf '%s\n' "$*" >> "$state/calls.log"
if [ "$1" != "--user" ]; then
  printf 'missing --user: %s\n' "$*" >&2
  exit 64
fi
shift
case "$1" in
  show)
    if [ -f "$state/running" ]; then
      printf 'ActiveState=active\nMainPID=4242\n'
      exit 0
    fi
    printf 'ActiveState=inactive\nMainPID=0\n'
    exit 0
    ;;
  daemon-reload)
    exit 0
    ;;
  restart|start)
    : > "$state/running"
    exit 0
    ;;
  disable)
    rm -f "$state/running"
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
        let state = dir.path().join("systemd");
        std::fs::create_dir_all(&state).expect("state");
        let script = dir.path().join("systemctl");
        std::fs::write(&script, FAKE).expect("write script");
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        }
        Self { dir }
    }

    fn agent(&self) -> Agent {
        Agent::at(
            self.dir.path().join("systemd").join("user").join("agent.service"),
            Launcher::new(self.dir.path().join("systemctl")),
        )
    }

    fn calls(&self) -> Vec<String> {
        let path = self.dir.path().join("systemd").join("calls.log");
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
        }
    }
}

/// The logged line without the `--user` the launcher always adds.
///
/// The log keeps `--user` — asserting it is half the point of the script, since a launcher that
/// dropped it would address the system manager. The assertions that care about *which* subcommand
/// ran strip it so they do not read `"--user restart"` as something other than a restart.
fn invocation(call: &str) -> &str {
    call.strip_prefix("--user ").unwrap_or(call)
}

#[test]
fn installing_twice_does_not_disturb_a_healthy_unit() {
    // The idempotency rule, and the one that matters: a second `install` must not restart a
    // daemon that is in the middle of an attempt.
    let fixture = Fixture::new();
    let agent = fixture.agent();

    assert_eq!(agent.install(&fixture.spec()).expect("first"), Installed::Fresh);
    let after_first = fixture.calls().len();
    assert_eq!(
        agent.install(&fixture.spec()).expect("second"),
        Installed::Unchanged
    );
    assert_eq!(
        fixture.calls().len(),
        after_first,
        "an unchanged unit made systemd do more work: {:?}",
        fixture.calls()
    );
    let restarts = fixture
        .calls()
        .iter()
        .filter(|call| invocation(call).starts_with("restart"))
        .count();
    assert_eq!(restarts, 1, "the daemon was restarted more than once");
}

#[test]
fn installing_enables_the_unit_with_a_link_in_default_target_wants() {
    // `systemctl enable` is a symlink and nothing more, so writing it directly is the same
    // operation without a shell-out. A unit that is written but never enabled is a file systemd
    // reads and never starts at login, which is the promise in the docs.
    let fixture = Fixture::new();
    let agent = fixture.agent();
    agent.install(&fixture.spec()).expect("install");

    let link = agent.wants_link();
    assert!(link.exists(), "not enabled: {}", link.display());
    let target = std::fs::read_link(&link).expect("a symlink");
    assert_eq!(
        target,
        agent.unit(),
        "the link must be absolute and name this unit; a bare file name is resolved inside \
         default.target.wants/ and dangles"
    );
}

#[test]
fn a_stale_enablement_link_is_replaced_rather_than_kept() {
    // A link can survive the unit it named: `systemctl enable` from an older install, a unit moved
    // by hand, a link left behind after someone removed the file. `systemctl is-enabled` still
    // calls a dangling link enabled, so leaving it is a unit that never starts at login while
    // `service status` reports it installed.
    let fixture = Fixture::new();
    let agent = fixture.agent();
    let link = agent.wants_link();
    std::fs::create_dir_all(link.parent().expect("parent")).expect("mkdir");
    std::os::unix::fs::symlink("/nonexistent/old.service", &link).expect("stale link");

    agent.install(&fixture.spec()).expect("install");

    let target = std::fs::read_link(&link).expect("still a symlink");
    assert_eq!(
        target,
        agent.unit(),
        "the link still points at the old unit, so login would start nothing"
    );
}

#[test]
fn a_unit_that_is_already_written_but_disabled_is_enabled_again() {
    // `same` and `enabled` are independent facts. An unchanged file whose link was removed — by
    // `systemctl disable`, by a cleanup script, by a human — is a unit systemd reads and never
    // starts at login. Skipping `enable` because the file matched would leave the promise in the
    // docs broken while `install` reported success.
    let fixture = Fixture::new();
    let agent = fixture.agent();
    agent.install(&fixture.spec()).expect("install");
    let link = agent.wants_link();
    std::fs::remove_file(&link).expect("disable");

    agent.install(&fixture.spec()).expect("reinstall");

    assert!(
        link.exists(),
        "the unchanged unit was not re-enabled, so login would start nothing"
    );
}

#[test]
fn a_changed_unit_is_reloaded_after_daemon_reload_and_then_restarted() {
    // systemd caches unit files, so a changed file is invisible until `daemon-reload`. The order
    // is the mechanism: reload the manager, then restart the job.
    let fixture = Fixture::new();
    let agent = fixture.agent();
    agent.install(&fixture.spec()).expect("first");

    let mut spec = fixture.spec();
    spec.program = fixture.dir.path().join("bin").join("immortality-next");
    assert_eq!(agent.install(&spec).expect("second"), Installed::Reloaded);

    let calls = fixture.calls();
    let restarts: Vec<usize> = calls
        .iter()
        .enumerate()
        .filter(|(_, call)| invocation(call).starts_with("restart "))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(restarts.len(), 2, "{calls:?}");
    // Each install reloads before it restarts, so the *second* reload is the one belonging to the
    // changed unit. Taking the first would accept an implementation that reloaded once and then
    // restarted twice, which is precisely the bug this asserts against: systemd caches the unit
    // file, so a restart without a fresh reload runs the old program.
    let reloads: Vec<usize> = calls
        .iter()
        .enumerate()
        .filter(|(_, call)| invocation(call).starts_with("daemon-reload"))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(reloads.len(), 2, "{calls:?}");
    assert!(
        reloads[1] > restarts[0] && reloads[1] < restarts[1],
        "daemon-reload must sit between the two restarts: {calls:?}"
    );

    // And the file on disk is what the launcher was told to run.
    let written = std::fs::read_to_string(agent.unit()).expect("unit");
    assert_eq!(written, unit(&spec));
    assert!(written.contains("immortality-next"), "{written}");
}

#[test]
fn uninstalling_twice_is_not_an_error() {
    // A script runs `uninstall` unconditionally.
    let fixture = Fixture::new();
    let agent = fixture.agent();
    agent.install(&fixture.spec()).expect("install");
    assert!(agent.unit().exists());

    agent.uninstall().expect("first");
    assert!(!agent.unit().exists());
    assert!(!agent.wants_link().exists(), "the enablement link survived");
    agent.uninstall().expect("second");
}

#[test]
fn status_separates_installed_from_running() {
    // Two different facts. A unit that failed and is inside `RestartSec` is installed and not
    // running, and answering only "installed" tells a human the opposite of what they need.
    let fixture = Fixture::new();
    let agent = fixture.agent();
    agent.install(&fixture.spec()).expect("install");

    let running = agent.status();
    assert!(running.installed);
    assert!(running.running);
    assert_eq!(running.pid, Some(4242));

    // Something removed the marker: installed, not running, and no pid to report.
    std::fs::remove_file(fixture.dir.path().join("systemd").join("running")).expect("unmark");
    let stopped = agent.status();
    assert!(stopped.installed);
    assert!(!stopped.running);
    assert_eq!(stopped.pid, None, "`MainPID=0` is not a process");
}

#[test]
fn a_failed_restart_names_the_unit_and_systemds_own_message() {
    // "install failed" is not actionable. The unit is already written when `restart` runs, so the
    // message has to say that, or a human will look for a file that is there.
    let fixture = Fixture::new();
    // A systemctl that refuses: `install` writes the unit and then fails to start it.
    let broken = fixture.dir.path().join("broken-systemctl");
    std::fs::write(
        &broken,
        "#!/bin/sh\ncase \"$2\" in restart|start) printf 'Job failed: unit not found\\n' >&2; exit 5;; esac\nexit 0\n",
    )
    .expect("write");
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&broken, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    }
    let agent = Agent::at(
        fixture
            .dir
            .path()
            .join("systemd")
            .join("user")
            .join("agent.service"),
        Launcher::new(&broken),
    );
    let error = agent.install(&fixture.spec()).expect_err("must fail");
    assert!(
        error.contains(&agent.unit().display().to_string()),
        "the message must name the unit: {error}"
    );
    assert!(
        error.contains("unit not found"),
        "systemd's own words are more useful than a summary: {error}"
    );
    assert!(
        agent.unit().exists(),
        "the unit is written before starting; the message says so and must be true"
    );
}

#[test]
fn the_written_unit_is_the_one_the_manager_reads() {
    // A unit written from a different spec than the one installed is the silent failure this
    // whole module exists to prevent: `service status` would describe a file nobody reads.
    let fixture = Fixture::new();
    let agent = fixture.agent();
    agent.install(&fixture.spec()).expect("install");
    let written = std::fs::read_to_string(agent.unit()).expect("unit");
    assert_eq!(written, unit(&fixture.spec()));
    assert!(written.contains("WantedBy=default.target"), "{written}");
    assert!(
        written.contains("RestartSec=10"),
        "the restart delay must be written down: {written}"
    );
}

#[test]
fn the_unit_directory_is_created_if_it_does_not_exist() {
    // A first install on a machine that has never used a user unit: `~/.config/systemd/user` does
    // not exist yet, and refusing because a parent is missing would make `service install` fail
    // exactly once, in the case it is for.
    let fixture = Fixture::new();
    let agent = fixture.agent();
    assert!(!agent.unit().parent().expect("parent").exists());
    agent.install(&fixture.spec()).expect("install");
    assert!(agent.unit().parent().expect("parent").is_dir());
}

#[test]
fn the_discovered_agent_points_at_the_user_unit_directory() {
    // The discovered path is what `service install` uses on a real machine. A `system` unit would
    // run as root, which is the thing D-012 and D-110 both refuse.
    let path = immortality_daemon::service::systemd::unit_path().expect("a config directory");
    assert!(
        path.ends_with(format!("systemd/user/{LABEL}.service")),
        "{path:?}"
    );
    assert!(
        !path.starts_with("/etc"),
        "a user unit must not be installed into /etc: {path:?}"
    );
}

/// The log directory used by the unit is created, because `StandardOutput=append:` fails a unit
/// that names a file in a directory that does not exist.
#[test]
fn the_log_directory_is_created_before_the_unit_starts() {
    let fixture = Fixture::new();
    let agent = fixture.agent();
    let spec = fixture.spec();
    assert!(!spec.log_path.parent().expect("parent").exists());
    agent.install(&spec).expect("install");
    assert!(spec.log_path.parent().expect("parent").is_dir());
}
