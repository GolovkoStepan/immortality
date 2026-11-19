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
//! # Why the plist is generated rather than shipped
//!
//! It contains absolute paths: the binary, the state directory, the log files. A checked-in plist
//! would be right for exactly one machine, and a human editing it by hand is a human whose
//! `service status` no longer describes what is installed. Generating it means the installed file
//! and `service status` are derived from the same values.

use std::path::PathBuf;
use std::process::Command;

use super::{Installed, LABEL, RESTART_SECONDS, SUBCOMMAND, ServiceSpec, Status};

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
    <string>{subcommand}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>IMMORTALITY_STATE_DIR</key>
    <string>{state}</string>{path}{lang}
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
        subcommand = SUBCOMMAND,
        program = escape(&spec.program.display().to_string()),
        state = escape(&spec.state_root.display().to_string()),
        log = escape(&spec.log_path.display().to_string()),
        path = path_entry(spec.path.as_deref()),
        lang = lang_entry(spec.lang.as_deref()),
        throttle = RESTART_SECONDS,
    )
}

/// The `PATH` entry for the plist's `EnvironmentVariables`, or nothing.
///
/// Indented to match its sibling key because the plist is a document a human may read; an entry
/// appended without its leading newline would produce `...</string><key>PATH</key>`, which parses
/// but looks like a mistake.
fn path_entry(path: Option<&str>) -> String {
    path.map_or_else(String::new, |path| {
        format!("\n    <key>PATH</key>\n    <string>{}</string>", escape(path))
    })
}

/// The `LANG` entry, or nothing.
///
/// Same shape as [`path_entry`] and for the same reason: a second key in the same dictionary, not
/// a value appended to the state directory's string.
fn lang_entry(lang: Option<&str>) -> String {
    lang.map_or_else(String::new, |lang| {
        format!("\n    <key>LANG</key>\n    <string>{}</string>", escape(lang))
    })
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

    /// The program this launcher runs, for messages and tests.
    #[must_use]
    pub fn program(&self) -> &std::ffi::OsStr {
        &self.program
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
    pub fn unit(&self) -> &std::path::Path {
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
            unit: self.plist.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ServiceSpec {
        ServiceSpec {
            program: PathBuf::from("/Users/test/bin/immortality"),
            state_root: PathBuf::from("/Users/test/Library/Application Support/immortality"),
            log_path: PathBuf::from("/Users/test/Library/Application Support/immortality/logs/daemon.log"),
            path: None,
            lang: None,
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

    #[test]
    fn the_plist_carries_the_path_it_was_installed_with() {
        // launchd gives an agent `/usr/bin:/bin:/usr/sbin:/sbin` and reads no shell profile, so a
        // toolchain from asdf or mise is invisible to the daemon. The PATH captured in the shell
        // that ran `service install` is the only correct value that ever exists, and losing it
        // here means the daemon silently runs an older interpreter.
        let with_path = ServiceSpec {
            path: Some("/Users/test/.asdf/shims:/usr/bin:/bin".to_string()),
            ..spec()
        };
        let text = plist(&with_path);
        assert!(text.contains("<key>PATH</key>"), "{text}");
        assert!(text.contains("/Users/test/.asdf/shims:/usr/bin:/bin"), "{text}");
        // And it sits inside `EnvironmentVariables`, not loose in the dictionary.
        let environment = text
            .split("<key>EnvironmentVariables</key>")
            .nth(1)
            .expect("the plist has an EnvironmentVariables dict");
        assert!(environment.contains("<key>PATH</key>"), "{environment}");
    }

    #[test]
    fn a_spec_without_a_path_names_no_path_key() {
        // `None` must mean "let the manager decide", not "set PATH to the empty string": an empty
        // PATH is a daemon that cannot find `git`, and `git` is how the whole run works.
        let text = plist(&spec());
        assert!(!text.contains("<key>PATH</key>"), "{text}");
    }

    #[test]
    fn a_path_with_xml_characters_is_escaped_in_the_environment_too() {
        // The same escaping rule as `ProgramArguments`; a PATH is a string from a shell and can
        // contain the same characters.
        let odd = ServiceSpec {
            path: Some("/Users/a&b/bin:/usr/bin".to_string()),
            ..spec()
        };
        let text = plist(&odd);
        assert!(text.contains("/Users/a&amp;b/bin:/usr/bin"), "{text}");
        assert!(!text.contains("a&b/bin"), "an unescaped `&` reached the plist");
    }

    #[test]
    fn the_plist_carries_the_lang_it_was_installed_with() {
        // Measured on the third `black_jack` run: launchd gives an agent no locale at all, so
        // Ruby's `Encoding.default_external` was `US-ASCII` and RSpec's failure messages came back
        // as `????` in 48 journal records. Nothing crashed; the one failure the model had to read
        // was replaced by question marks. The locale is the same kind of fact as PATH: it exists in
        // the installing shell and nowhere else.
        let with_lang = ServiceSpec {
            lang: Some("en_US.UTF-8".to_string()),
            ..spec()
        };
        let text = plist(&with_lang);
        assert!(text.contains("<key>LANG</key>"), "{text}");
        assert!(text.contains("<string>en_US.UTF-8</string>"), "{text}");
        // Inside `EnvironmentVariables`, not loose in the outer dictionary: a `LANG` beside
        // `Label` is valid XML and does nothing.
        let environment = text
            .split("<key>EnvironmentVariables</key>")
            .nth(1)
            .expect("the plist has an EnvironmentVariables dict");
        assert!(environment.contains("<key>LANG</key>"), "{environment}");
        // And it is a sibling of PATH, not a suffix on the state directory's value: two keys in
        // one dictionary, each with its own `<key>`/`<string>` pair.
        assert!(
            environment.contains("<key>IMMORTALITY_STATE_DIR</key>"),
            "{environment}"
        );
    }

    #[test]
    fn a_changed_lang_is_a_changed_document_so_the_daemon_restarts_with_it() {
        // `install` compares the generated text, so a locale that differs makes the file differ and
        // the job reload. That is the point — a daemon installed before the fix keeps `US-ASCII`
        // until it is restarted — and it is also the cost: a `service install` run from a shell
        // with another locale kills an in-flight attempt, exactly as a changed PATH already does.
        // The alternative, ignoring LANG in the comparison, would write a file that says one thing
        // and leave a process doing another.
        let one = plist(&ServiceSpec {
            lang: Some("en_US.UTF-8".to_string()),
            ..spec()
        });
        let two = plist(&ServiceSpec {
            lang: Some("ru_RU.UTF-8".to_string()),
            ..spec()
        });
        assert_ne!(one, two);
        assert_eq!(
            one,
            plist(&ServiceSpec {
                lang: Some("en_US.UTF-8".to_string()),
                ..spec()
            })
        );
    }

    #[test]
    fn a_spec_without_a_lang_names_no_lang_key() {
        // Same rule as PATH: `None` means "let the manager decide". Writing `LANG=` with an empty
        // value would be worse than silence — it is a locale whose name does not exist, and libc
        // would fall back to `C` while the plist claimed otherwise.
        let text = plist(&spec());
        assert!(!text.contains("<key>LANG</key>"), "{text}");
    }

    #[test]
    fn the_lang_and_the_path_do_not_run_into_each_other() {
        // Both entries are appended with a leading newline; a missing one would produce
        // `</string><key>LANG</key>`, which parses and reads as a typo a human would then "fix".
        let both = ServiceSpec {
            path: Some("/usr/bin:/bin".to_string()),
            lang: Some("en_US.UTF-8".to_string()),
            ..spec()
        };
        let text = plist(&both);
        assert!(text.contains("<key>PATH</key>"), "{text}");
        assert!(text.contains("<key>LANG</key>"), "{text}");
        assert!(!text.contains("</string><key>"), "{text}");
    }
}
