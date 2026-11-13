//! `launchd` integration: the plist, and the three commands around it.
//!
//! # Why the daemon is a `LaunchAgent` and not a `LaunchDaemon`
//!
//! A `LaunchAgent` runs as the logged-in user and inherits that user's environment; a
//! `LaunchDaemon` runs as `root` before anyone logs in. Everything this program touches — the
//! state directory under `~/Library/Application Support`, the task worktrees under the user's
//! workspace root, the model backend on the local network — belongs to the user, and a daemon
//! running as `root` would create files the agent then cannot write. `root` would also make every
//! `sudo` escalation policy meaningless, because the process would already have the privilege it
//! is supposed to ask for (D-012).
//!
//! # Why `KeepAlive` and not `StartInterval`
//!
//! The roadmap's promise is that a `kill -9` leaves a daemon that comes back. `KeepAlive` is the
//! only key that means "restart whenever it is not running"; `StartInterval` would wait for the
//! timer, and a task would sit idle for however long that is. `KeepAlive` also covers the reboot
//! case: launchd starts an agent at login when the key is present.
//!
//! `ThrottleInterval` is set because `KeepAlive` on a program that exits immediately is a tight
//! restart loop; launchd's default is 10 s, and naming it makes the intent visible rather than
//! inherited from a man page.
//!
//! # Why the plist is generated rather than shipped
//!
//! It contains absolute paths: the binary, the state directory, the log files. A checked-in plist
//! would be right for exactly one machine, and a human editing it by hand is a human whose
//! `service status` no longer describes what is installed. Generating it means the installed file
//! and `service status` are derived from the same values.
//!
//! # Why installing is idempotent
//!
//! `service install` twice is what a script does. The second run must not reload a running
//! daemon (that would kill in-flight work) and must not leave two plists with one label — launchd
//! keys an agent by label, so two files would be a race whose winner is not defined.

use std::path::PathBuf;
use std::process::Command;

/// The `launchd` label: both the plist's name and the job's identity.
///
/// Reverse-DNS because launchd uses it as a key in a shared namespace; `immortality` alone could
/// collide with anything a human installed.
pub const LABEL: &str = "dev.immortality.daemon";

/// Seconds launchd waits before restarting a job that exited.
///
/// launchd's own default. Named here so the plist says what it means: without it a daemon that
/// fails during startup is restarted in a tight loop that fills the log with identical lines.
const THROTTLE_SECONDS: u32 = 10;

/// Where the plist goes: `~/Library/LaunchAgents/<label>.plist`.
///
/// A `LaunchAgent` is only loaded from this directory. Putting the file anywhere else — beside the
/// binary, in the state directory — produces a plist launchd never reads and a `service status`
/// that says "installed" about a file nobody loads.
///
/// # Errors
/// `$HOME` is not set or is not valid UTF-8, which on macOS means the environment is broken in a
/// way no fallback path can fix.
pub fn plist_path() -> Result<PathBuf, String> {
    let home = std::env::var_os("HOME")
        .ok_or_else(|| "HOME is not set, so the LaunchAgents directory cannot be found".to_string())?
        .into_string()
        .map_err(|_| "HOME is not valid UTF-8".to_string())?;
    Ok(PathBuf::from(home)
        .join("Library")
        .join("LaunchAgents")
        .join(format!("{LABEL}.plist")))
}

/// Everything the plist needs that the daemon cannot guess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceSpec {
    /// The absolute path of the `immortality` binary.
    pub program: PathBuf,
    /// The absolute state directory, passed as `IMMORTALITY_STATE_DIR`.
    pub state_root: PathBuf,
    /// Where stdout and stderr go. launchd appends; it never rotates.
    pub log_path: PathBuf,
}

impl ServiceSpec {
    /// The spec for this machine: the running binary, and the discovered state root.
    ///
    /// # Errors
    /// `current_exe` cannot be read, or `$HOME` is unusable.
    pub fn discover() -> Result<Self, String> {
        let program =
            std::env::current_exe().map_err(|error| format!("cannot locate this binary: {error}"))?;
        // `current_exe` on macOS can be a path through a symlink that launchd resolves
        // differently; `canonicalize` makes the plist name the file the kernel would run.
        let program = std::fs::canonicalize(&program).unwrap_or(program);
        let state_root = state_root();
        Ok(Self {
            log_path: state_root.join("logs").join("daemon.log"),
            program,
            state_root,
        })
    }
}

/// The state root as an absolute path, from the same rule the daemon uses.
///
/// The same function `serve` calls, not a second copy of the `~/Library/Application Support` path:
/// the plist and the daemon must agree about where state lives, and two literals is how they would
/// stop agreeing.
fn state_root() -> PathBuf {
    immortality_core::paths::StatePaths::discover()
        .root()
        .to_path_buf()
}

/// The plist document, as XML.
///
/// Hand-written rather than serialized from a crate: a plist is a small, well-specified XML shape,
/// and pulling in a serializer for five keys would be the dependency this project has been
/// avoiding for everything else. Every string is escaped, because a workspace root containing `&`
/// or `<` is legal on macOS and would otherwise produce a plist launchd refuses to parse.
#[must_use]
pub fn plist(spec: &ServiceSpec) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{program}</string>
    <string>serve</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>IMMORTALITY_STATE_DIR</key>
    <string>{state}</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>{throttle}</integer>
  <key>StandardOutPath</key>
  <string>{log}</string>
  <key>StandardErrorPath</key>
  <string>{log}</string>
  <key>ProcessType</key>
  <string>Background</string>
</dict>
</plist>
"#,
        label = LABEL,
        program = escape(&spec.program.display().to_string()),
        state = escape(&spec.state_root.display().to_string()),
        log = escape(&spec.log_path.display().to_string()),
        throttle = THROTTLE_SECONDS,
    )
}

/// Escape the five characters the XML spec requires in text content.
///
/// Not a general XML escaper: the only way a character reaches this function is from a path, and a
/// path cannot contain a tag. Escaping `<`, `>`, `&`, `'` and `"` is what a path can need.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            other => out.push(other),
        }
    }
    out
}

/// What `service status` reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Status {
    /// The plist is on disk.
    pub installed: bool,
    /// launchd reports the job as running.
    pub running: bool,
    /// Where the plist is (or would be).
    pub plist: PathBuf,
    /// launchd's `pid` line, when it reports one.
    pub pid: Option<u32>,
}

/// How the module reaches launchd.
///
/// A value rather than a free function because the install sequence has consequences — reloading
/// restarts a daemon and every attempt it was in the middle of — so a test has to be able to
/// substitute a script. The seam is a field, not only an environment variable, because
/// `std::env::set_var` is `unsafe` in edition 2024 and racy under parallel tests; a value passed in
/// is neither. The variable is still honoured, for the one case a constructor cannot serve: a human
/// or a test driving the *binary*, where there is nowhere to pass an argument.
#[derive(Debug, Clone)]
pub struct Launcher {
    program: std::ffi::OsString,
}

impl Launcher {
    /// The real `launchctl`, unless `IMMORTALITY_LAUNCHCTL` names another program.
    #[must_use]
    pub fn from_env() -> Self {
        Self::new(std::env::var_os("IMMORTALITY_LAUNCHCTL").unwrap_or_else(|| "launchctl".into()))
    }

    /// A launcher that runs `program` in place of `launchctl`.
    #[must_use]
    pub fn new(program: impl Into<std::ffi::OsString>) -> Self {
        Self {
            program: program.into(),
        }
    }

    /// Run `launchctl`, returning its stdout.
    ///
    /// # Errors
    /// The program cannot be started, or it exits non-zero. launchd's own stderr is passed through:
    /// a substituted "launchctl failed" would tell a human less than launchd did.
    fn run(&self, args: &[&str]) -> Result<String, String> {
        let output = Command::new(&self.program)
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

/// One installed (or installable) agent: its plist and the launchd that owns it.
///
/// The two travel together because every operation needs both, and passing them separately is how
/// one of them ends up describing a different agent than the other — the plist at one path, the
/// job looked up in another domain.
#[derive(Debug, Clone)]
pub struct Agent {
    plist: PathBuf,
    launcher: Launcher,
}

impl Agent {
    /// The agent for this machine: the plist launchd reads, and the real `launchctl`.
    ///
    /// # Errors
    /// `$HOME` is unusable, so there is no `LaunchAgents` directory to install into.
    pub fn discover() -> Result<Self, String> {
        Ok(Self::at(plist_path()?, Launcher::from_env()))
    }

    /// An agent whose plist is at `plist`, driven through `launcher`.
    ///
    /// This is the seam a test needs: it installs into a temporary directory and talks to a
    /// script, without touching the human's own agent or the process environment.
    #[must_use]
    pub fn at(plist: impl Into<PathBuf>, launcher: Launcher) -> Self {
        Self {
            plist: plist.into(),
            launcher,
        }
    }

    /// Where the plist is (or would be).
    #[must_use]
    pub fn plist(&self) -> &std::path::Path {
        &self.plist
    }

    /// Read the installed plist and ask launchd about the job.
    ///
    /// Asks `launchctl` instead of inferring: the plist existing and the job running are different
    /// facts, and a daemon that crashed and is inside its throttle window is installed but not
    /// running. Saying only "installed" would answer the opposite of the question a human has.
    #[must_use]
    pub fn status(&self) -> Status {
        let installed = self.plist.exists();
        let (running, pid) = self
            .launcher
            .run(&["print", &format!("gui/{}/{LABEL}", uid())])
            .map_or((false, None), |output| {
                // `launchctl print` exits 0 for a loaded job and 113 for one it has never heard
                // of; its output also carries `state = running`, which is the truthful signal
                // during a throttle window where the job is loaded but not running.
                (output.contains("state = running"), parse_pid(&output))
            });
        Status {
            installed,
            running,
            plist: self.plist.clone(),
            pid,
        }
    }

    /// Install (or refresh) the agent, without disturbing a healthy one.
    ///
    /// # The idempotency rule
    ///
    /// Two runs in a row must be indistinguishable from one. That means:
    ///
    /// * the file is compared before it is written, so an unchanged plist is not touched;
    /// * a job that is already loaded with an unchanged plist is **not** reloaded, because
    ///   reloading kills the daemon and every attempt it was in the middle of;
    /// * a *changed* plist is reloaded, because a human who restarted with a different binary or
    ///   state directory expects the running daemon to match the file.
    ///
    /// # Errors
    /// The plist cannot be written, or `launchctl` refuses to load the job. Both name the path and
    /// launchd's own message, because "install failed" is not something a human can act on.
    pub fn install(&self, spec: &ServiceSpec) -> Result<Installed, String> {
        let wanted = plist(spec);
        let current = std::fs::read_to_string(&self.plist).ok();
        let same = current.as_deref() == Some(wanted.as_str());

        if !same {
            immortality_core::atomic::write_atomically(&self.plist, wanted.as_bytes())
                .map_err(|error| format!("cannot write {}: {error}", self.plist.display()))?;
        }

        let domain = format!("gui/{}", uid());
        let loaded = self
            .launcher
            .run(&["print", &format!("{domain}/{LABEL}")])
            .is_ok();
        if loaded && same {
            // Nothing changed and the job is already loaded: leave it alone. Reloading here is
            // the difference between an idempotent install and one that kills in-flight work.
            return Ok(Installed::Unchanged);
        }
        if loaded {
            // Unload before loading: `bootstrap` over a loaded job fails with "service already
            // loaded" and leaves the *old* program running, which is the confusing half of this.
            self.launcher
                .run(&["bootout", &format!("{domain}/{LABEL}")])
                .map_err(|error| {
                    format!(
                        "cannot unload the running agent before reloading it: {error}. \
                         The plist is written; `launchctl bootout {domain}/{LABEL}` reports the \
                         same failure without the wrapper."
                    )
                })?;
        }
        self.launcher
            .run(&["bootstrap", &domain, &self.plist.display().to_string()])
            .map_err(|error| {
                format!(
                    "cannot load {}: {error}. The plist is written; `launchctl bootstrap {domain} \
                     {}` reports the same failure without the wrapper.",
                    self.plist.display(),
                    self.plist.display()
                )
            })?;
        Ok(if current.is_some() {
            Installed::Reloaded
        } else {
            Installed::Fresh
        })
    }

    /// Unload the job and remove the plist.
    ///
    /// Removing a file that is not there is `Ok`: `uninstall` twice is what a script does, and
    /// failing the second time would make it unusable in one.
    ///
    /// # Errors
    /// The plist exists and cannot be removed. A `launchctl` failure is *not* an error: the job may
    /// never have been loaded, and the file is what `service status` reports.
    pub fn uninstall(&self) -> Result<(), String> {
        let domain = format!("gui/{}", uid());
        // Best effort on purpose: the interesting outcomes are a job that was never loaded and a
        // job launchd already dropped, and both must still remove the plist.
        let _ = self.launcher.run(&["bootout", &format!("{domain}/{LABEL}")]);
        match std::fs::remove_file(&self.plist) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(format!("cannot remove {}: {error}", self.plist.display())),
        }
    }
}

/// The `pid = <n>` line from `launchctl print`, if launchd printed one.
fn parse_pid(output: &str) -> Option<u32> {
    output.lines().find_map(|line| {
        line.trim()
            .strip_prefix("pid = ")
            .and_then(|rest| rest.trim().parse().ok())
    })
}

/// The current user's uid, needed to address a `gui/<uid>/` domain.
///
/// Read from `id -u` rather than the `libc` crate: this project has no `libc` dependency, and the
/// one thing needed from it is a number that `id` prints without any configuration.
fn uid() -> String {
    Command::new("id")
        .arg("-u")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map_or_else(|| "501".to_string(), |text| text.trim().to_string())
}

/// What `service install` did, so the command can say it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Installed {
    /// The plist was written and the job newly loaded.
    Fresh,
    /// The plist already said exactly this, and the job was already loaded.
    Unchanged,
    /// The plist changed, so the job was reloaded to pick it up.
    Reloaded,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ServiceSpec {
        ServiceSpec {
            program: PathBuf::from("/Users/test/bin/immortality"),
            state_root: PathBuf::from("/Users/test/Library/Application Support/immortality"),
            log_path: PathBuf::from("/Users/test/Library/Application Support/immortality/logs/daemon.log"),
        }
    }

    #[test]
    fn the_plist_keeps_the_daemon_alive_and_starts_at_login() {
        // The whole point of M5.3: `kill -9` and a reboot both leave a running daemon. `KeepAlive`
        // is the key that means both, and dropping it would make the promise in the docs false.
        let text = plist(&spec());
        assert!(text.contains("<key>KeepAlive</key>\n  <true/>"), "{text}");
        assert!(text.contains("<key>RunAtLoad</key>\n  <true/>"), "{text}");
        // A short throttle is a restart loop that fills the log; the default must be named.
        assert!(text.contains("<key>ThrottleInterval</key>"), "{text}");
        assert!(text.contains("<integer>10</integer>"), "{text}");
    }

    #[test]
    fn the_plist_names_the_state_directory_and_the_serve_subcommand() {
        // `ProgramArguments` without `serve` would start the CLI's default subcommand — which is
        // a usage error, so launchd would restart a command that can never succeed.
        let text = plist(&spec());
        assert!(text.contains("<string>serve</string>"), "{text}");
        assert!(text.contains("IMMORTALITY_STATE_DIR"), "{text}");
        assert!(
            text.contains("/Users/test/Library/Application Support/immortality"),
            "{text}"
        );
        assert!(text.contains(LABEL), "{text}");
    }

    #[test]
    fn a_path_with_xml_characters_is_escaped() {
        // Legal on macOS, and a plist launchd cannot parse is worse than an install that refuses:
        // the failure would appear as "the daemon never starts" with no message anywhere.
        let odd = ServiceSpec {
            program: PathBuf::from("/Users/a&b/<c>/immortality"),
            ..spec()
        };
        let text = plist(&odd);
        assert!(text.contains("a&amp;b"), "{text}");
        assert!(text.contains("&lt;c&gt;"), "{text}");
        assert!(!text.contains("a&b"), "an unescaped `&` reached the plist");
    }

    #[test]
    fn the_plist_lives_where_launchd_reads_agents() {
        // A plist anywhere else is a file launchd never reads and a `status` that lies.
        let path = plist_path().expect("HOME");
        assert!(
            path.ends_with(format!("Library/LaunchAgents/{LABEL}.plist")),
            "{path:?}"
        );
    }

    #[test]
    fn the_pid_is_read_from_launchctl_output() {
        // `launchctl print` is the only source: the plist existing says nothing about whether the
        // job is running, and a crashed job inside its throttle window is installed but stopped.
        let output = "gui/501/dev.immortality.daemon = {\n\tstate = running\n\tpid = 4321\n}";
        assert_eq!(parse_pid(output), Some(4321));
        assert_eq!(parse_pid("\tstate = not running\n"), None);
        assert_eq!(parse_pid("pid = nonsense\n"), None);
    }

    #[test]
    fn a_different_plist_is_a_different_document() {
        // The comparison `install` makes before writing. Two specs that differ in one path must
        // not compare equal, or a human's config edit would never reach the running daemon.
        let one = plist(&spec());
        let two = plist(&ServiceSpec {
            program: PathBuf::from("/Users/test/other"),
            ..spec()
        });
        assert_ne!(one, two);
        assert_eq!(one, plist(&spec()));
    }
}
