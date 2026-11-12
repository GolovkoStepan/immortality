//! Terminal styling: the escape sequences, and the decision to emit them at all.
//!
//! Two jobs, deliberately in one place:
//!
//! 1. Turning text into coloured text. Every method returns a `String`, so a caller never has to
//!    remember a reset code; without colour the text is returned unchanged and no sequence is
//!    emitted anywhere.
//! 2. Deciding whether colour belongs in this stream. That decision is [`should_colour`], a pure
//!    function of three facts the process can observe (a TTY, `NO_COLOR`, `TERM`), so it is
//!    testable without a real terminal and without touching the environment.
//!
//! The sequences are written by hand. `crossterm` would be a dependency for six SGR codes, and the
//! project's rule is that dependencies are added when they buy something a few lines cannot.

// Same reason as `render`: in a binary crate a private module's `pub` items trip
// `unreachable_pub` while `pub(crate)` trips `redundant_pub_crate`, and the pair is unsatisfiable.
#![allow(unreachable_pub)]

use std::ffi::OsStr;

/// Whether the text this process writes should carry colour.
///
/// The three inputs are passed in rather than read here, so the rule is covered by a table of
/// cases instead of by whatever the test harness's stdout happens to be. `is_tty` is
/// `std::io::IsTerminal` on stdout, which is what makes a redirected run, a pipe and CI plain
/// automatically.
#[must_use]
pub fn should_colour(is_tty: bool, no_color: Option<&OsStr>, term: Option<&OsStr>) -> bool {
    if !is_tty {
        // A pipe is a file, a log, or another program. Escape sequences there are noise at best
        // and a broken diff at worst.
        return false;
    }
    // The `NO_COLOR` convention: set and non-empty means "no colour", *empty* means the variable
    // was set without a value and therefore expresses no preference.
    if no_color.is_some_and(|value| !value.is_empty()) {
        return false;
    }
    // `TERM=dumb` is how a terminal says it cannot do this. Some CI wrappers set it while still
    // providing a TTY.
    if term.is_some_and(|value| value.eq_ignore_ascii_case("dumb")) {
        return false;
    }
    true
}

/// A styling decision: coloured or plain.
///
/// Copyable and tiny, so it travels by value into renderers without plumbing lifetimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Style {
    colour: bool,
}

impl Style {
    /// Plain text. What a pipe, a log file and every test get by default.
    #[must_use]
    pub const fn plain() -> Self {
        Self { colour: false }
    }

    /// Coloured text. Only ever constructed after [`should_colour`] said yes.
    #[must_use]
    pub const fn colour() -> Self {
        Self { colour: true }
    }

    /// Wrap `text` in the SGR codes in `codes` (e.g. `"1;32"`), or return it untouched.
    ///
    /// One call emits at most one opening and one closing sequence. Nesting is what makes escape
    /// output fragile — an inner reset ends the outer colour — so callers compose a single code
    /// list instead of wrapping styled text again.
    fn paint(self, codes: &str, text: &str) -> String {
        if !self.colour {
            return text.to_string();
        }
        format!("\u{1b}[{codes}m{text}\u{1b}[0m")
    }

    /// Emphasised: names, kinds, identifiers.
    #[must_use]
    pub fn bold(self, text: &str) -> String {
        self.paint("1", text)
    }

    /// De-emphasised: the sequence gutter, separators, bookkeeping.
    #[must_use]
    pub fn dim(self, text: &str) -> String {
        self.paint("2", text)
    }

    /// Failure.
    #[must_use]
    pub fn red(self, text: &str) -> String {
        self.paint("31", text)
    }

    /// Success and completion.
    #[must_use]
    pub fn green(self, text: &str) -> String {
        self.paint("32", text)
    }

    /// Something the human should look at, but that is not a failure.
    #[must_use]
    pub fn yellow(self, text: &str) -> String {
        self.paint("33", text)
    }

    /// Structure: task identity, model activity.
    #[must_use]
    pub fn cyan(self, text: &str) -> String {
        self.paint("36", text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    fn os(text: &str) -> &OsStr {
        OsStr::new(text)
    }

    #[test]
    fn a_tty_with_no_preference_gets_colour() {
        assert!(should_colour(true, None, Some(os("xterm-256color"))));
        assert!(should_colour(true, None, None));
    }

    #[test]
    fn a_pipe_never_gets_colour() {
        // The rule that matters most: `immortality attach > run.log` must produce a clean file.
        assert!(!should_colour(false, None, Some(os("xterm-256color"))));
        assert!(!should_colour(false, Some(os("0")), None));
    }

    #[test]
    fn no_color_switches_colour_off_but_an_empty_value_does_not() {
        assert!(!should_colour(true, Some(os("1")), Some(os("xterm"))));
        // The convention is "present and non-empty"; `NO_COLOR=` alone is not a request.
        assert!(should_colour(true, Some(os("")), Some(os("xterm"))));
    }

    #[test]
    fn a_dumb_terminal_gets_no_colour_whatever_the_case() {
        assert!(!should_colour(true, None, Some(os("dumb"))));
        assert!(!should_colour(true, None, Some(os("DUMB"))));
        // Not a prefix match: `dumb-ish` is a different terminal.
        assert!(should_colour(true, None, Some(os("dumb-ish"))));
    }

    #[test]
    fn plain_style_returns_text_byte_for_byte() {
        let style = Style::plain();
        assert_eq!(style.bold("bold"), "bold");
        assert_eq!(style.dim("dim"), "dim");
        assert_eq!(style.red("red"), "red");
        assert_eq!(style.green("green"), "green");
        assert_eq!(style.yellow("yellow"), "yellow");
        assert_eq!(style.cyan("task_state"), "task_state");
    }

    #[test]
    fn coloured_style_wraps_and_resets_exactly_once() {
        let style = Style::colour();
        assert_eq!(style.green("done"), "\u{1b}[32mdone\u{1b}[0m");
        assert_eq!(style.bold("x"), "\u{1b}[1mx\u{1b}[0m");
        let rendered = style.dim("seq");
        assert_eq!(rendered.matches('\u{1b}').count(), 2, "{rendered:?}");
        assert!(rendered.ends_with("\u{1b}[0m"), "{rendered:?}");
    }

    #[test]
    fn every_colour_uses_a_distinct_code() {
        // A copy-paste that gives red and green the same code would be invisible in review and
        // obvious only to a user. Pinning the exact sequences is what catches it.
        let style = Style::colour();
        assert_eq!(style.bold("t"), "\u{1b}[1mt\u{1b}[0m");
        assert_eq!(style.dim("t"), "\u{1b}[2mt\u{1b}[0m");
        assert_eq!(style.red("t"), "\u{1b}[31mt\u{1b}[0m");
        assert_eq!(style.green("t"), "\u{1b}[32mt\u{1b}[0m");
        assert_eq!(style.yellow("t"), "\u{1b}[33mt\u{1b}[0m");
        assert_eq!(style.cyan("t"), "\u{1b}[36mt\u{1b}[0m");
    }
}
