//! Service integration: the installed background daemon, per platform.
//!
//! # Why this is two modules and not one with branches
//!
//! `launchd` and `systemd` agree on nothing below the command names. The plist is an XML
//! dictionary keyed by `KeepAlive`; the unit is an INI file keyed by `Restart=`. Enabling is
//! `launchctl bootstrap` against a `gui/<uid>/` domain; on systemd it is a symlink in
//! `default.target.wants`. Status is a `pid = <n>` line in one and `MainPID` in the other, and
//! `RestartSec` has no `ThrottleInterval` spelling. A single module with `cfg` branches inside
//! would be four functions each of whose body is entirely the other platform's business. D-110.
//!
//! Both modules are compiled on every platform, and only the re-export below is selected. That is
//! deliberate: generation of the unit file and the argument sequences sent to `systemctl` are
//! testable with a scripted program on any host, and gating the *module* would mean the Linux
//! tests only ever run in CI. What genuinely needs the platform — `plutil` validating the plist,
//! a real `launchctl` — stays gated where it is.
//!
//! # Why installing is idempotent
//!
//! `service install` twice is what a script does. The second run must not restart a running
//! daemon (that would kill in-flight work) and must not leave two definitions with one name.

use std::path::PathBuf;

pub mod launchd;
pub mod systemd;
#[cfg(target_os = "macos")]
pub use launchd::{Agent, Launcher, plist, plist_path};
#[cfg(not(target_os = "macos"))]
pub use systemd::{Agent, Launcher, unit, unit_path};

/// The job's identity, and the name of the file that defines it.
///
/// Reverse-DNS because both managers key a job in a shared namespace; `immortality` alone could
/// collide with anything a human installed.
pub const LABEL: &str = "dev.immortality.daemon";

/// Seconds the manager waits before restarting a job that exited.
///
/// Named rather than inherited so the intent is visible: `KeepAlive` and `Restart=on-failure` on
/// a program that fails during startup are a tight restart loop that fills the log with identical
/// lines. `launchd`'s own default is 10 s; `RestartSec` defaults to 100 ms, so on systemd the
/// value has to be written down or the difference is a bug.
pub const RESTART_SECONDS: u32 = 10;

/// What the daemon is started as: the same subcommand on both platforms.
pub const SUBCOMMAND: &str = "serve";

/// The manager's name, for messages a human reads.
///
/// A constant rather than a sentence assembled at each call site: "launchd is not running it" on
/// Linux is a message about a program that is not installed there.
pub const MANAGER: &str = if cfg!(target_os = "macos") {
    "launchd"
} else {
    "systemd --user"
};

/// Everything the installed definition needs that the daemon cannot guess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceSpec {
    /// The absolute path of the `immortality` binary.
    pub program: PathBuf,
    /// The absolute state directory, passed as `IMMORTALITY_STATE_DIR`.
    pub state_root: PathBuf,
    /// Where stdout and stderr go. Both managers append; neither rotates.
    pub log_path: PathBuf,
    /// `PATH` to install the job with, captured from the shell that ran `service install`.
    ///
    /// `None` means "do not set one", and the manager supplies its own. Both managers give a
    /// service a PATH that is not the human's: `launchd` hands an agent
    /// `/usr/bin:/bin:/usr/sbin:/sbin` and `systemd --user` does not read `.zprofile` either. For
    /// a project whose toolchain comes from a version manager (asdf, mise, nvm, pyenv) that means
    /// the daemon runs a *different* interpreter than the human's shell — the tool is not found,
    /// or an older one is found first. Capturing the PATH at install time is the only moment the
    /// good value exists: the human is in the shell that has it, and the daemon is not.
    pub path: Option<String>,
    /// The character locale to install the job with, captured from the same shell as [`Self::path`].
    ///
    /// `None` means "do not set one", and the manager supplies its default — which on macOS is no
    /// locale at all. A process with no `LANG` does not guess UTF-8: libc falls back to the `C`
    /// locale, and a language runtime built on it picks `US-ASCII`. Measured on the third
    /// `black_jack` run: Ruby's `Encoding.default_external` was `US-ASCII`, so `RSpec`'s failure
    /// messages came back as `????` in 48 journal records, and the one failure the model had to
    /// read — the poisoned `allow(sleep)` — was unreadable. Nothing crashed; the diagnosis was
    /// just silently destroyed.
    ///
    /// A single value rather than three, collapsed by the precedence the C library itself uses
    /// (`LC_ALL`, then `LC_CTYPE`, then `LANG`). That is what makes copying faithful instead of
    /// approximate: a shell with `LC_ALL=C` gets a daemon that also sees `C`, and a shell with only
    /// `LC_CTYPE=en_US.UTF-8` gets a daemon whose character locale is `en_US.UTF-8` — the same
    /// encoding the shell's children had. It is about encoding, not message translation.
    pub lang: Option<String>,
}

impl ServiceSpec {
    /// The spec for this machine: the running binary, the discovered state root, and this process's
    /// `PATH`.
    ///
    /// # Errors
    /// `current_exe` cannot be read, or `$HOME` is unusable.
    pub fn discover() -> Result<Self, String> {
        let program =
            std::env::current_exe().map_err(|error| format!("cannot locate this binary: {error}"))?;
        // `current_exe` can be a path through a symlink the manager resolves differently;
        // `canonicalize` makes the definition name the file the kernel would run.
        let program = std::fs::canonicalize(&program).unwrap_or(program);
        let state_root = state_root();
        Ok(Self {
            log_path: state_root.join("logs").join("daemon.log"),
            program,
            state_root,
            path: std::env::var("PATH").ok().filter(|path| !path.trim().is_empty()),
            lang: character_locale(),
        })
    }
}

/// The locale that decides how this process's children encode text, or `None` if it is unset.
fn character_locale() -> Option<String> {
    character_locale_of(|name| std::env::var(name).ok())
}

/// [`character_locale`] over an arbitrary lookup, so the precedence is testable.
///
/// The order is the one glibc and Apple's libc use: `LC_ALL` overrides everything, `LC_CTYPE` names
/// the character locale specifically, `LANG` is the fallback for both. Empty is unset — a `LANG=`
/// in a shell profile is a variable a human cleared, not a locale whose name is the empty string.
///
/// The lookup is a parameter rather than `std::env::var` read inline because `set_var` is `unsafe`
/// in edition 2024 and racy under parallel tests: a test that mutated the environment to exercise
/// this would be changing every other test's `LANG` at the same time.
fn character_locale_of(lookup: impl Fn(&str) -> Option<String>) -> Option<String> {
    ["LC_ALL", "LC_CTYPE", "LANG"]
        .iter()
        .filter_map(|name| lookup(name))
        .find(|value| !value.trim().is_empty())
}

/// The state root as an absolute path, from the same rule the daemon uses.
///
/// The same function `serve` calls, not a second copy of the path: the installed definition and
/// the daemon must agree about where state lives, and two literals is how they would stop
/// agreeing.
fn state_root() -> PathBuf {
    immortality_core::paths::StatePaths::discover()
        .root()
        .to_path_buf()
}

/// What `service status` reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Status {
    /// The definition is on disk.
    pub installed: bool,
    /// The manager reports the job as running.
    pub running: bool,
    /// Where the definition is (or would be): a plist or a unit file.
    pub unit: PathBuf,
    /// The manager's running pid, when it reports one.
    pub pid: Option<u32>,
}

/// What `service install` did, so the command can say it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Installed {
    /// The definition was written and the job newly loaded.
    Fresh,
    /// The definition already said exactly this, and the job was already loaded.
    Unchanged,
    /// The definition changed, so the job was reloaded to pick it up.
    Reloaded,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_platform_names_the_manager_it_actually_uses() {
        // The message a human reads when a job is installed but stopped. Naming `launchd` on
        // Linux would send them looking for a program that is not installed.
        let expected = if cfg!(target_os = "macos") {
            "launchd"
        } else {
            "systemd --user"
        };
        assert_eq!(MANAGER, expected);
    }

    #[test]
    fn both_managers_restart_after_the_same_delay() {
        // One number, two spellings: `ThrottleInterval` and `RestartSec`. A second literal in the
        // systemd unit would let the two platforms disagree about how hard a crash loop is.
        assert_eq!(RESTART_SECONDS, 10);
    }

    #[test]
    fn the_character_locale_prefers_lc_all_then_lc_ctype_then_lang() {
        // The precedence is not a preference: it is what libc does, so copying it is what makes the
        // installed value the value the shell's own children had. A daemon with `LC_ALL=C` in the
        // installing shell must not be handed `LANG=en_US.UTF-8` — that would be a locale the human
        // explicitly overrode.
        let with = |pairs: &'static [(&'static str, &'static str)]| {
            character_locale_of(move |name| {
                pairs
                    .iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| (*value).to_string())
            })
        };
        assert_eq!(
            with(&[("LC_ALL", "C"), ("LANG", "en_US.UTF-8")]),
            Some("C".to_string())
        );
        assert_eq!(
            with(&[("LC_CTYPE", "en_US.UTF-8"), ("LANG", "C")]),
            Some("en_US.UTF-8".to_string())
        );
        assert_eq!(with(&[("LANG", "ru_RU.UTF-8")]), Some("ru_RU.UTF-8".to_string()));
        assert_eq!(with(&[]), None);
    }

    #[test]
    fn an_empty_locale_variable_is_unset_not_a_locale_named_empty() {
        // `LANG=` in a profile is a variable a human cleared. Copying the empty string into the
        // plist would install a locale whose name does not exist, and libc would fall back to `C`
        // while the definition claimed otherwise — the silent `US-ASCII` bug with a paper trail.
        let with = |pairs: &'static [(&'static str, &'static str)]| {
            character_locale_of(move |name| {
                pairs
                    .iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| (*value).to_string())
            })
        };
        assert_eq!(with(&[("LANG", "")]), None);
        assert_eq!(
            with(&[("LC_ALL", ""), ("LANG", "en_US.UTF-8")]),
            Some("en_US.UTF-8".to_string())
        );
        assert_eq!(with(&[("LANG", "   ")]), None);
    }
}
