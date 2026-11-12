//! Test harness: a mock `OpenAI` backend and replayable fixtures.
//!
//! The whole project's test strategy rests on not needing a live model. `fixtures/` contains
//! real traces captured from the backend during S1, so the mock reproduces the awkward parts
//! of the protocol — character-by-character tool arguments, `: keep-alive` comments — rather
//! than an idealised version of it.

// Assertion helpers in this crate exist in order to fail: `Check::assert_match`, `crash_at` and
// the simulators all abort a test loudly on purpose. Allow the lints crate-wide rather than per
// item, and keep the rule that the *rest* of the workspace is denied.
#![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

pub mod clock;
pub mod fakefs;
pub mod fault;
pub mod golden;
pub mod mock;
pub mod sim;

pub use clock::Clock;
pub use fakefs::{FlakyFs, Fs, FsReader, MemFs, Op, StdFs};
pub use fault::{Action, Fault, Faults, Hook};
pub use golden::{Check, GoldenDir};
pub use mock::{MockServer, ReceivedRequest, Reply, script_from_trace};

/// A recorded exchange with the backend, used to drive the mock.
///
/// Line format matches `spike/traces/*.jsonl`: one JSON object per step with `req` and `resp`.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct TraceStep {
    pub step: u32,
    #[serde(default)]
    pub req: Option<serde_json::Value>,
    #[serde(default)]
    pub resp: Option<serde_json::Value>,
    #[serde(default)]
    pub fatal: Option<String>,
}

/// Parse a trace file. Blank lines are skipped; a malformed line is an error rather than
/// silently ignored, because a corrupted fixture must not turn into a passing test.
pub fn parse_trace(text: &str) -> Result<Vec<TraceStep>, serde_json::Error> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lines_and_skips_blanks() {
        let text = "{\"step\":1}\n\n{\"step\":2,\"fatal\":\"boom\"}\n";
        let steps = parse_trace(text).unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[1].fatal.as_deref(), Some("boom"));
    }

    #[test]
    fn malformed_line_is_an_error() {
        assert!(parse_trace("{\"step\":1}\nnot json\n").is_err());
    }

    #[test]
    fn real_spike_traces_parse() {
        // The captured traces are part of the repository; if they stop parsing, the fixture
        // format has drifted from what the engine expects. These are the exact seven S1
        // scenarios, so a missing file means a fixture was deleted by accident.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .expect("fixtures directory must exist")
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
            .collect();
        paths.sort();
        assert_eq!(paths.len(), 7, "expected the seven S1 traces, got {paths:?}");

        for path in paths {
            let text = std::fs::read_to_string(&path).unwrap();
            let steps = parse_trace(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert!(!steps.is_empty(), "{} is empty", path.display());
            // Spike traces are 1-based; a gap would desynchronise a replay exactly the way
            // a real crash does, but with no useful error.
            let expected: Vec<u32> = (1..=u32::try_from(steps.len()).unwrap()).collect();
            let actual: Vec<u32> = steps.iter().map(|step| step.step).collect();
            assert_eq!(actual, expected, "{} has non-contiguous steps", path.display());
        }
    }
}
