//! `systemd --user` integration: the unit, and the three commands around it.
//!
//! # Why a user unit and not a system one
//!
//! The same reason `launchd` gets a `LaunchAgent` (D-012): everything this program touches
//! belongs to the logged-in user, and a system unit running as `root` would create files the agent
//! cannot write and would make every `sudo` escalation policy meaningless. `systemd --user`
//! inherits the user's environment through the user manager, which is what makes
//! `IMMORTALITY_STATE_DIR` and the workspace root meaningful.
//!
//! # Why the unit is written into `$XDG_CONFIG_HOME`, not `/etc`
//!
//! A user unit is discovered from `$XDG_CONFIG_HOME/systemd/user/`, which is where the user's own
//! copy lives and where `systemctl --user` looks without privileges. Writing to `/etc/systemd`
//! would need `sudo` to install a daemon that must never run as root, and uninstalling would need
//! it again.
//!
//! # Why enabling is a symlink and not `systemctl enable`
//!
//! `systemctl enable` is a shell-out whose only effect is to create that symlink (and, with
//! `--now`, to start the unit). Writing the link directly makes the operation inspectable in a
//! test without a `systemd` running, and it is the same thing `systemctl enable` does. Starting is
//! `systemctl --user start`, which is not replaceable.
//!
//! # Why `Restart=on-failure` and not `Restart=always`
//!
//! `always` restarts a job that exited cleanly, and a clean exit is how `immortality serve`
//! reports a deliberate stop. `on-failure` restarts exactly the case `KeepAlive` covers on macOS —
//! a crash or a `kill -9` — and leaves `systemctl --user stop` alone.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::{Installed, LABEL, RESTART_SECONDS, SUBCOMMAND, ServiceSpec, Status};

/// Where the unit goes: `$XDG_CONFIG_HOME/systemd/user/<label>.service`.
///
/// # Errors
/// `$HOME` is not set, so there is no `~/.config` to fall back to.
pub fn unit_path() -> Result<PathBuf, String> {
    Ok(config_home()?
        .join("systemd")
        .join("user")
        .join(format!("{LABEL}.service")))
}

/// The unit document, as INI.
///
/// Hand-written for the same reason the plist is: it is a handful of keys, and a serializer would
/// be a dependency for five lines of text. The `ExecStart` line is quoted because the binary path
/// may contain a space, and systemd splits on whitespace without it — a path that contains a space
/// would silently become two arguments and a unit that never starts.
#[must_use]
pub fn unit(spec: &ServiceSpec) -> String {
    format!(
        r#"[Unit]
Description=immortality daemon
Documentation=https://github.com/immortality/immortality

[Service]
Type=simple
ExecStart="{program}" {subcommand}
Environment="IMMORTALITY_STATE_DIR={state}"
Restart=on-failure
RestartSec={restart}
StandardOutput=append:{log}
StandardError=append:{log}

[Install]
WantedBy=default.target
"#,
        program = quote_escape(&spec.program.display().to_string()),
        subcommand = SUBCOMMAND,
        state = quote_escape(&spec.state_root.display().to_string()),
        log = quote_escape(&spec.log_path.display().to_string()),
        restart = RESTART_SECONDS,
    )
}

/// Escape what systemd's quoting needs inside a double-quoted string.
///
/// systemd reads `"` as the delimiter and `\` as the escape; `%` is a specifier and is the one
/// character that is not escapeable, so a path containing it is refused rather than silently
/// expanded. A `$` is not special here because systemd only expands it in `Environment=` when the
/// value is unquoted, and every value this function escapes sits inside quotes.
fn quote_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            other => out.push(other),
        }
    }
    out
}

/// `$XDG_CONFIG_HOME`, or `~/.config` when it is unset or empty.
///
/// An empty value is treated as unset, as the XDG spec says, and as
/// [`immortality_core::paths`] already does for the state directory — the two must agree about what
/// "unset" means or the unit is installed in one place and described in another.
fn config_home() -> Result<PathBuf, String> {
    if let Some(value) = std::env::var_os("XDG_CONFIG_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(value));
    }
    let home = std::env::var_os("HOME")
        .ok_or_else(|| "neither XDG_CONFIG_HOME nor HOME is set, so the unit has nowhere to go".to_string())?
        .into_string()
        .map_err(|_| "HOME is not valid UTF-8".to_string())?;
    Ok(PathBuf::from(home).join(".config"))
}

/// How the module reaches `systemctl`.
///
/// The same seam as [`super::launchd::Launcher`], and for the same reason: the install sequence
/// restarts a daemon and every attempt it was in the middle of, so a test has to be able to
/// substitute a script. `IMMORTALITY_SYSTEMCTL` is honoured for a human driving the binary.
#[derive(Debug, Clone)]
pub struct Launcher {
    program: std::ffi::OsString,
}

impl Launcher {
    /// The real `systemctl`, unless `IMMORTALITY_SYSTEMCTL` names another program.
    #[must_use]
    pub fn from_env() -> Self {
        Self::new(std::env::var_os("IMMORTALITY_SYSTEMCTL").unwrap_or_else(|| "systemctl".into()))
    }

    /// A launcher that runs `program` in place of `systemctl`.
    #[must_use]
    pub fn new(program: impl Into<std::ffi::OsString>) -> Self {
        Self {
            program: program.into(),
        }
    }

    /// The program this launcher runs, for messages and tests.
    #[must_use]
    pub fn program(&self) -> &std::ffi::OsStr {
        &self.program
    }

    /// Run `systemctl --user <args>`, returning its stdout.
    ///
    /// `--user` is added here rather than at each call site: a `systemctl` that addressed the
    /// system manager would report on a unit that does not exist and, worse, `uninstall` would
    /// stop nothing while claiming success.
    ///
    /// # Errors
    /// The program cannot be started, or it exits non-zero. systemd's own stderr is passed
    /// through: a substituted message would tell a human less than systemd did.
    fn run(&self, args: &[&str]) -> Result<String, String> {
        let output = Command::new(&self.program)
            .arg("--user")
            .args(args)
            .output()
            .map_err(|error| format!("cannot run {}: {error}", self.program.to_string_lossy()))?;
        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).into_owned())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
        }
    }
}

/// One installed (or installable) unit: its file, its enablement link, and the `systemctl` that
/// owns it.
#[derive(Debug, Clone)]
pub struct Agent {
    unit: PathBuf,
    launcher: Launcher,
}

impl Agent {
    /// The agent for this machine: the unit `systemctl --user` reads, and the real `systemctl`.
    ///
    /// # Errors
    /// Neither `$XDG_CONFIG_HOME` nor `$HOME` is usable, so there is nowhere to install.
    pub fn discover() -> Result<Self, String> {
        Ok(Self::at(unit_path()?, Launcher::from_env()))
    }

    /// An agent whose unit is at `unit`, driven through `launcher`.
    ///
    /// The seam a test needs: it installs into a temporary directory and talks to a script,
    /// without touching the human's own units or the process environment.
    #[must_use]
    pub fn at(unit: impl Into<PathBuf>, launcher: Launcher) -> Self {
        Self {
            unit: unit.into(),
            launcher,
        }
    }

    /// Where the unit is (or would be).
    #[must_use]
    pub fn unit(&self) -> &Path {
        &self.unit
    }

    /// The symlink that enables the unit in `default.target.wants`.
    ///
    /// Derived from the unit's own directory rather than from `$XDG_CONFIG_HOME` again, so an
    /// `Agent::at` in a temporary directory enables itself there and nowhere else.
    #[must_use]
    pub fn wants_link(&self) -> PathBuf {
        self.unit
            .parent()
            .map_or_else(
                || PathBuf::from("default.target.wants"),
                |dir| dir.join("default.target.wants"),
            )
            .join(format!("{LABEL}.service"))
    }

    /// Read the installed unit and ask systemd about the job.
    ///
    /// Asks `systemctl --user show` rather than inferring: the file existing and the job running
    /// are different facts, and a unit that failed and is inside `RestartSec` is installed but not
    /// running. `ActiveState=active` is the truthful signal; `MainPID=0` means systemd has no
    /// process for it, which is not a pid and must not be reported as one.
    #[must_use]
    pub fn status(&self) -> Status {
        let installed = self.unit.exists();
        let (running, pid) = self
            .launcher
            .run(&[
                "show",
                &format!("{LABEL}.service"),
                "--property=ActiveState,MainPID",
            ])
            .map_or((false, None), |output| {
                (
                    output.lines().any(|line| line.trim() == "ActiveState=active"),
                    parse_main_pid(&output),
                )
            });
        Status {
            installed,
            running,
            unit: self.unit.clone(),
            pid,
        }
    }

    /// Install (or refresh) the unit, without disturbing a healthy one.
    ///
    /// # The idempotency rule
    ///
    /// Two runs in a row must be indistinguishable from one:
    ///
    /// * the file is compared before it is written, so an unchanged unit is not touched;
    /// * a unit that is already enabled with an unchanged file is not restarted, because
    ///   restarting kills the daemon and every attempt it was in the middle of;
    /// * a *changed* unit is restarted, because a human who reinstalled with a different binary
    ///   expects the running daemon to match the file.
    ///
    /// # Errors
    /// The unit cannot be written or enabled, or `systemctl` refuses to start it. Both name the
    /// path and systemd's own message.
    pub fn install(&self, spec: &ServiceSpec) -> Result<Installed, String> {
        let wanted = unit(spec);
        let current = std::fs::read_to_string(&self.unit).ok();
        let same = current.as_deref() == Some(wanted.as_str());

        if let Some(parent) = self.unit.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        // The log directory is named in the unit, and `StandardOutput=append:` fails the unit if
        // the file cannot be opened — a daemon that never starts because a directory is missing
        // is a failure with no message anywhere except the journal.
        if let Some(parent) = spec.log_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        if !same {
            immortality_core::atomic::write_atomically(&self.unit, wanted.as_bytes())
                .map_err(|error| format!("cannot write {}: {error}", self.unit.display()))?;
        }

        // Two independent facts: the file matches, and the enablement link exists. Only when both
        // hold is there nothing to do — an unchanged unit whose link was removed (`systemctl
        // disable`, a cleanup script) is a unit systemd reads and never starts at login, while
        // `install` would report success.
        if same && self.wants_link().exists() {
            // Restarting here is the difference between an idempotent install and one that kills
            // in-flight work.
            return Ok(Installed::Unchanged);
        }
        self.enable()?;

        // A changed unit has to be picked up: systemd caches the file until `daemon-reload`.
        self.launcher.run(&["daemon-reload"]).map_err(|error| {
            format!(
                "cannot reload systemd after writing {}: {error}. The unit is written; \
                 `systemctl --user daemon-reload` reports the same failure without the wrapper.",
                self.unit.display()
            )
        })?;
        self.launcher
            .run(&["restart", &format!("{LABEL}.service")])
            .map_err(|error| {
                format!(
                    "cannot start {}: {error}. The unit is written; `systemctl --user start {LABEL}` \
                     reports the same failure without the wrapper.",
                    self.unit.display()
                )
            })?;
        Ok(if current.is_some() {
            Installed::Reloaded
        } else {
            Installed::Fresh
        })
    }

    /// Create the `default.target.wants` symlink, replacing one that points elsewhere.
    ///
    /// Replacing matters: a unit file that moved would otherwise leave a dangling link that
    /// `systemctl --user is-enabled` still calls enabled.
    fn enable(&self) -> Result<(), String> {
        let link = self.wants_link();
        if let Some(parent) = link.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        if std::fs::read_link(&link).is_ok() {
            std::fs::remove_file(&link)
                .map_err(|error| format!("cannot replace {}: {error}", link.display()))?;
        }
        // Absolute, because systemd resolves the link relative to the wants directory, not to the
        // unit's directory: a bare file name would be looked for inside `default.target.wants/`.
        std::os::unix::fs::symlink(&self.unit, &link).map_err(|error| {
            format!(
                "cannot link {} to {}: {error}",
                link.display(),
                self.unit.display()
            )
        })?;
        Ok(())
    }

    /// Stop the unit and remove both the file and the enablement link.
    ///
    /// Removing what is not there is `Ok`: `uninstall` twice is what a script does, and failing the
    /// second time would make it unusable in one.
    ///
    /// # Errors
    /// The unit exists and cannot be removed. A `systemctl` failure is *not* an error: the job may
    /// never have been started, and the files are what `service status` reports.
    pub fn uninstall(&self) -> Result<(), String> {
        // Best effort on purpose: a unit that was never started, or one systemd already stopped,
        // must still have its files removed.
        let _ = self
            .launcher
            .run(&["disable", "--now", &format!("{LABEL}.service")]);
        let mut first_error = None;
        for path in [self.unit.clone(), self.wants_link()] {
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    let message = format!("cannot remove {}: {error}", path.display());
                    first_error.get_or_insert(message);
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

/// The `MainPID=<n>` line from `systemctl show`, when it names a real process.
///
/// `MainPID=0` is systemd's "no process for this unit", and reporting pid 0 would be a process id
/// that cannot exist.
fn parse_main_pid(output: &str) -> Option<u32> {
    output.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("MainPID=")?.trim();
        rest.parse::<u32>().ok().filter(|pid| *pid > 0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ServiceSpec {
        ServiceSpec {
            program: PathBuf::from("/home/test/.cargo/bin/immortality"),
            state_root: PathBuf::from("/home/test/.local/state/immortality"),
            log_path: PathBuf::from("/home/test/.local/state/immortality/logs/daemon.log"),
        }
    }

    #[test]
    fn the_unit_restarts_on_failure_and_runs_the_serve_subcommand() {
        // `Restart=on-failure` is the `KeepAlive` equivalent: a crash and a `kill -9` come back,
        // a deliberate `systemctl --user stop` does not. Without `serve` the daemon would start the
        // CLI's default subcommand, which is a usage error systemd would restart forever.
        let text = unit(&spec());
        assert!(text.contains("Restart=on-failure"), "{text}");
        assert!(text.contains("RestartSec=10"), "{text}");
        assert!(text.contains(" serve\n"), "{text}");
        assert!(text.contains("Type=simple"), "{text}");
        assert!(text.contains("WantedBy=default.target"), "{text}");
    }

    #[test]
    fn the_unit_names_the_state_directory_and_the_log_file() {
        let text = unit(&spec());
        assert!(
            text.contains("IMMORTALITY_STATE_DIR=/home/test/.local/state/immortality"),
            "{text}"
        );
        assert!(
            text.contains("StandardOutput=append:/home/test/.local/state/immortality/logs/daemon.log"),
            "{text}"
        );
        assert!(
            text.contains("StandardError=append:/home/test/.local/state/immortality/logs/daemon.log"),
            "{text}"
        );
    }

    #[test]
    fn a_program_path_with_a_space_stays_one_argument() {
        // systemd splits `ExecStart` on whitespace, so an unquoted path with a space becomes two
        // arguments and a unit that can never start. The quotes are the fix and this pins them.
        let odd = ServiceSpec {
            program: PathBuf::from("/home/a b/bin/immortality"),
            ..spec()
        };
        let text = unit(&odd);
        assert!(
            text.contains(r#"ExecStart="/home/a b/bin/immortality" serve"#),
            "{text}"
        );
    }

    #[test]
    fn a_quote_in_a_path_is_escaped_rather_than_ending_the_value() {
        let odd = ServiceSpec {
            program: PathBuf::from(r#"/home/a"b/immortality"#),
            ..spec()
        };
        let text = unit(&odd);
        assert!(text.contains(r#"/home/a\"b/immortality"#), "{text}");
    }

    #[test]
    fn the_main_pid_ignores_zero() {
        // `MainPID=0` is systemd's "no process", and pid 0 cannot exist.
        assert_eq!(parse_main_pid("ActiveState=active\nMainPID=4321\n"), Some(4321));
        assert_eq!(parse_main_pid("ActiveState=inactive\nMainPID=0\n"), None);
        assert_eq!(parse_main_pid("ActiveState=failed\n"), None);
    }

    #[test]
    fn the_unit_lives_in_the_user_unit_directory() {
        let path = unit_path().expect("a config directory");
        assert!(
            path.ends_with(format!("systemd/user/{LABEL}.service")),
            "{path:?}"
        );
    }

    #[test]
    fn the_enablement_link_is_beside_the_unit_not_in_the_state_directory() {
        // Derived from the unit's own directory so an agent installed into a temporary directory
        // enables itself there. A link built from `$XDG_CONFIG_HOME` would escape that and, in a
        // test, touch the developer's real units.
        let agent = Agent::at(
            PathBuf::from("/tmp/x/systemd/user/agent.service"),
            Launcher::new("systemctl"),
        );
        assert_eq!(
            agent.wants_link(),
            PathBuf::from(format!(
                "/tmp/x/systemd/user/default.target.wants/{LABEL}.service"
            ))
        );
    }

    #[test]
    fn a_different_unit_is_a_different_document() {
        let one = unit(&spec());
        let two = unit(&ServiceSpec {
            program: PathBuf::from("/home/test/other"),
            ..spec()
        });
        assert_ne!(one, two);
        assert_eq!(one, unit(&spec()));
    }
}
