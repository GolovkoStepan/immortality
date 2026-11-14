//! The binary output the `word-count` task's second subtask must deliver.
//!
//! Red until `main.rs` prints the once-only count. It is a separate test target for the same
//! reason as `tests/once.rs`: a subtask's gate must be red when that subtask starts, and the
//! previous subtask's green must not make this one green for free.
//!
//! # Why the binary and not the library
//!
//! The library's count is already covered by `tests/once.rs`, so asserting it again would let this
//! subtask close without touching `main.rs` — which is exactly the work it exists to do. Running
//! the real binary against a file is what makes "the output changed" observable.

use std::process::Command;

/// A two-line file in a directory of its own, so the assertion does not depend on the cwd.
///
/// Written by hand rather than with a crate: the fixture has no dependencies on purpose, because
/// its gate has to run offline in seconds.
fn sample() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tally-once-line-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("sample.txt");
    // `a` twice, `b` once: exactly one once-word.
    std::fs::write(&path, "a b a\n").expect("write the sample");
    path
}

#[test]
fn the_binary_prints_the_once_only_count() {
    let path = sample();
    let output = Command::new(env!("CARGO_BIN_EXE_tally"))
        .arg(&path)
        .output()
        .expect("run the binary");
    assert!(
        output.status.success(),
        "the binary failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("once: 1"),
        "the binary does not print the once-only count; stdout was:\n{stdout}"
    );
    // The counts that were already printed must survive: the subtask adds a line, it does not
    // replace the output. `lines` is deliberately not pinned — that defect belongs to the
    // `line-count` task, and pinning it here would make this task require a fix it never mentions.
    for line in ["words: 3", "unique: 2"] {
        assert!(stdout.contains(line), "`{line}` disappeared from:\n{stdout}");
    }
}
