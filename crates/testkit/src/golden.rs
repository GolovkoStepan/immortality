//! Golden files for the artifacts whose exact bytes matter.
//!
//! # What deserves a golden
//!
//! A golden test says "this output must not change" and is right when the output is a contract:
//! the journal line format (its `prev` chain depends on byte identity), the `status` rendering a
//! human reads, the prompt assembly (a change at the top invalidates the backend's prefix cache
//! and costs a 95-second prefill). It is wrong for anything that is expected to evolve, where it
//! turns every improvement into a snapshot update and trains the author to bless diffs.
//!
//! # Where the expected value lives
//!
//! Under `tests/golden/<name>`, so a reviewer sees the change in the diff. `UPDATE_GOLDEN=1`
//! rewrites the file; that is a deliberate two-step because a golden updated in the same run that
//! failed is a golden that never tested anything.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Outcome of a comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Check {
    /// The output matched the file on disk.
    Match,
    /// No file existed. Nothing was written: creating a golden silently would let a broken
    /// output become the expected one in the same run that was supposed to catch it.
    Missing(PathBuf),
    /// The file exists and differs.
    Mismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
}

impl Check {
    /// Panic unless the output matched, with a diff the reader can act on.
    ///
    /// The diff is line-based rather than a single blob: a journal segment is one long line per
    /// record, and a character-level diff of a 4 KB line is unreadable.
    pub fn assert_match(&self) {
        match self {
            Self::Match => {}
            Self::Missing(path) => panic!(
                "golden file {} does not exist; run with `UPDATE_GOLDEN=1` to create it, then \
                 inspect and commit it",
                path.display()
            ),
            Self::Mismatch {
                path,
                expected,
                actual,
            } => panic!(
                "golden mismatch in {}\n{}",
                path.display(),
                diff(expected, actual)
            ),
        }
    }
}

/// Compare `actual` with `tests/golden/<name>`, relative to the calling crate.
///
/// `CARGO_MANIFEST_DIR` is used rather than the current directory because tests run from the
/// workspace root and a golden path relative to it would break when a crate is tested alone.
#[must_use]
pub fn check(name: &str, actual: &str) -> Check {
    check_in(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("golden"),
        name,
        actual,
    )
}

/// Compare with an explicit directory, for tests that keep goldens elsewhere.
///
/// Rewriting happens only when `UPDATE_GOLDEN=1` is set *and* the content differs, and the
/// result is reported as [`Check::Match`] so the updating run is green — the file is then
/// reviewed in the diff, which is the point of a golden rather than an auto-blessed snapshot.
#[must_use]
pub fn check_in(dir: &Path, name: &str, actual: &str) -> Check {
    check_with(dir, name, actual, update_requested())
}

/// Whether this process was asked to rewrite goldens.
///
/// Read from the environment on every call rather than cached in a `static`: a cache would make
/// the answer depend on which test ran first, and `set_var` would then be needed (and is `unsafe`
/// since Rust 2024) to test the update path at all.
#[must_use]
fn update_requested() -> bool {
    std::env::var_os("UPDATE_GOLDEN").is_some_and(|value| value == "1")
}

/// The comparison with update mode as an argument instead of a global.
///
/// Every decision that depends on `update` lives here, so tests exercise the real code path with
/// `update = true` and no process-wide state is touched.
fn check_with(dir: &Path, name: &str, actual: &str, update: bool) -> Check {
    let path = dir.join(name);

    match std::fs::read_to_string(&path) {
        Ok(expected) if expected == actual => Check::Match,
        Ok(_) if update => {
            write_golden(&path, actual);
            Check::Match
        }
        Ok(expected) => Check::Mismatch {
            path,
            expected,
            actual: actual.to_string(),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if update {
                write_golden(&path, actual);
                return Check::Match;
            }
            Check::Missing(path)
        }
        Err(error) => panic!("cannot read golden {}: {error}", path.display()),
    }
}

/// Write a golden, creating its directory. Panics on failure: a golden that cannot be written in
/// update mode must not be silently skipped.
fn write_golden(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|error| {
            panic!("cannot create {}: {error}", parent.display());
        });
    }
    std::fs::write(path, content).unwrap_or_else(|error| panic!("cannot write {}: {error}", path.display()));
}

/// A line-oriented diff, showing only the differing lines plus context.
///
/// `+` is the actual output, `-` is the expectation. Enough context to see where it went wrong,
/// not so much that a one-line change buries itself.
#[must_use]
pub fn diff(expected: &str, actual: &str) -> String {
    const CONTEXT: usize = 2;
    let expected: Vec<&str> = expected.lines().collect();
    let actual: Vec<&str> = actual.lines().collect();
    let mut differing = Vec::new();
    for index in 0..expected.len().max(actual.len()) {
        if expected.get(index) != actual.get(index) {
            differing.push(index);
        }
    }
    if differing.is_empty() {
        // Equal lines but unequal text: a trailing newline. Worth naming, because "the diff is
        // empty" is the least useful failure message possible.
        return format!(
            "lines are equal but the bytes differ (expected {} bytes, actual {} bytes)",
            expected.len(),
            actual.len()
        );
    }

    let first = differing.first().copied().unwrap_or(0);
    let last = differing.last().copied().unwrap_or(0);
    let start = first.saturating_sub(CONTEXT);
    let end = (last + CONTEXT + 1).min(expected.len().max(actual.len()));
    let mut out = format!("first difference at line {}\n", first + 1);
    for index in start..end {
        match (expected.get(index), actual.get(index)) {
            (Some(want), Some(got)) if want == got => {
                // Writing into a `String` cannot fail; ignoring the result would hide a bug that
                // is impossible, so it is discarded explicitly.
                let _ = writeln!(out, " {:>4} | {want}", index + 1);
            }
            (want, got) => {
                if let Some(want) = want {
                    let _ = writeln!(out, "-{:>4} | {want}", index + 1);
                }
                if let Some(got) = got {
                    let _ = writeln!(out, "+{:>4} | {got}", index + 1);
                }
            }
        }
    }
    out
}

/// Normalize a value to pretty JSON, the form goldens for protocol payloads are stored in.
///
/// Sorted keys because a `serde_json::Value` with `preserve_order` would otherwise make the
/// golden depend on insertion order, which is not part of the contract.
#[must_use]
pub fn canonical_json(value: &serde_json::Value) -> String {
    let sorted = sort(value);
    let mut out = serde_json::to_string_pretty(&sorted).unwrap_or_else(|_| sorted.to_string());
    out.push('\n');
    out
}

fn sort(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by_key(|(key, _)| *key);
            let mut out = serde_json::Map::new();
            for (key, value) in entries {
                out.insert(key.clone(), sort(value));
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(items) => serde_json::Value::Array(items.iter().map(sort).collect()),
        other => other.clone(),
    }
}

/// A temporary golden directory that removes itself, for tests of the golden machinery.
#[derive(Debug)]
pub struct GoldenDir {
    dir: PathBuf,
}

impl GoldenDir {
    /// An empty temporary directory.
    ///
    /// Built from the system temp directory rather than through `tempfile` so that the helper is
    /// usable from any crate's tests without that crate depending on `tempfile`.
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = format!(
            "immortality-golden-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).expect("a temp dir");
        Self { dir }
    }

    /// The directory path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.dir
    }

    /// Compare against a golden in this directory.
    #[must_use]
    pub fn check(&self, name: &str, actual: &str) -> Check {
        check_in(&self.dir, name, actual)
    }
}

impl Drop for GoldenDir {
    fn drop(&mut self) {
        // Best effort: a leftover directory in the system temp folder is not a test failure, and
        // panicking in `drop` during an unwinding test would abort the process.
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl Default for GoldenDir {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_golden_is_reported_and_left_absent() {
        // Writing it here would mean a first run blesses whatever the code happened to produce.
        // The file is created only by an explicit `UPDATE_GOLDEN=1` run, so it gets reviewed.
        let dir = GoldenDir::new();
        let check = dir.check("first.txt", "hello\n");
        assert!(matches!(check, Check::Missing(_)));
        assert!(
            !dir.path().join("first.txt").exists(),
            "a normal run must not write goldens"
        );
        assert!(std::panic::catch_unwind(|| check.assert_match()).is_err());
    }

    #[test]
    fn update_mode_writes_the_file_and_passes() {
        let dir = GoldenDir::new();
        // The flag is an argument, not an environment variable: `set_var` is `unsafe` in edition
        // 2024 and the workspace forbids `unsafe`, and passing it makes the update branch testable
        // without mutating process-wide state that other tests read.
        let check = check_with(dir.path(), "first.txt", "hello\n", true);
        assert_eq!(check, Check::Match);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("first.txt")).unwrap(),
            "hello\n"
        );
        assert_eq!(dir.check("first.txt", "hello\n"), Check::Match);
    }

    #[test]
    fn update_mode_also_replaces_a_differing_file() {
        let dir = GoldenDir::new();
        std::fs::write(dir.path().join("a.txt"), "stale\n").unwrap();
        assert_eq!(check_with(dir.path(), "a.txt", "fresh\n", true), Check::Match);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "fresh\n"
        );
    }

    #[test]
    fn a_difference_is_a_mismatch_with_both_sides() {
        let dir = GoldenDir::new();
        // Seed the expectation directly: the helpers deliberately have no "write" API outside
        // update mode, so a test of the comparison writes the file itself.
        std::fs::write(dir.path().join("a.txt"), "one\ntwo\n").unwrap();
        let check = dir.check("a.txt", "one\nTWO\n");
        let Check::Mismatch { expected, actual, .. } = &check else {
            panic!("expected a mismatch, got {check:?}");
        };
        assert_eq!(expected, "one\ntwo\n");
        assert_eq!(actual, "one\nTWO\n");
        assert!(std::panic::catch_unwind(|| check.assert_match()).is_err());
    }

    #[test]
    fn a_match_does_not_panic() {
        let dir = GoldenDir::new();
        std::fs::write(dir.path().join("a.txt"), "same\n").unwrap();
        dir.check("a.txt", "same\n").assert_match();
    }

    #[test]
    fn the_diff_names_the_first_differing_line() {
        let text = diff("a\nb\nc\n", "a\nB\nc\n");
        assert!(text.contains("first difference at line 2"), "{text}");
        assert!(text.contains("-   2 | b"), "{text}");
        assert!(text.contains("+   2 | B"), "{text}");
        assert!(text.contains("    1 | a"), "context line: {text}");
    }

    #[test]
    fn the_diff_trims_context_to_a_window() {
        let mut expected = String::new();
        for number in 0..100 {
            let _ = writeln!(expected, "line {number}");
        }
        let mut actual = expected.clone();
        actual = actual.replace("line 50\n", "CHANGED\n");
        let text = diff(&expected, &actual);
        assert!(text.contains("CHANGED"));
        assert!(
            !text.contains("line 10\n"),
            "a 100-line diff must not print every line: {text}"
        );
    }

    #[test]
    fn equal_lines_with_a_trailing_newline_difference_are_reported_precisely() {
        // Otherwise the failure says "the diff is empty", which helps nobody.
        let text = diff("a\n", "a");
        assert!(text.contains("bytes differ"), "{text}");
    }

    #[test]
    fn a_missing_line_is_shown_on_the_side_that_has_it() {
        let text = diff("a\nb\n", "a\n");
        assert!(text.contains("-   2 | b"), "{text}");
        assert!(!text.contains("+   2 |"), "nothing to add: {text}");
    }

    #[test]
    fn canonical_json_sorts_keys_and_ends_with_a_newline() {
        let value = serde_json::json!({"b": 1, "a": {"d": 2, "c": 3}});
        let text = canonical_json(&value);
        assert!(text.ends_with('\n'));
        assert!(text.find("\"a\"").unwrap() < text.find("\"b\"").unwrap());
        assert!(text.find("\"c\"").unwrap() < text.find("\"d\"").unwrap());
    }

    #[test]
    fn canonical_json_is_stable_regardless_of_insertion_order() {
        let first: serde_json::Value = serde_json::from_str(r#"{"x":1,"y":2}"#).unwrap();
        let second: serde_json::Value = serde_json::from_str(r#"{"y":2,"x":1}"#).unwrap();
        assert_eq!(canonical_json(&first), canonical_json(&second));
    }

    #[test]
    fn canonical_json_preserves_array_order() {
        // Keys are a set, arrays are a sequence. Sorting arrays would hide a real reordering bug
        // in something like the plan's subtask list.
        let value = serde_json::json!({"items": [3, 1, 2]});
        let text = canonical_json(&value);
        assert!(text.find('3').unwrap() < text.find('1').unwrap());
    }
}
