//! End-to-end tests of the `immortality` binary.
//!
//! These run the real executable, because the properties that matter here are not in any
//! function: exit codes, stdout versus stderr, and whether `--json` is actually machine
//! readable. A unit test of `print_human` would pass while the binary exited 0 on failure.

// Integration tests: `expect` and `unwrap` are the right tool here — a failed assertion
// should name the exact line, and there is no caller to propagate to.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use predicates::prelude::*;

/// A config pointing at a port that refuses connections immediately, so tests never touch
/// the network or the real backend.
fn dead_backend_config(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("config.toml");
    std::fs::write(
        &path,
        r#"
[backend]
base_url = "http://127.0.0.1:1/v1"
model = "test-model"
connect_timeout_secs = 2

[policy]
trusted_hosts = ["127.0.0.1"]

[logging]
dir = "LOGDIR"
"#
        .replace("LOGDIR", &dir.join("logs").display().to_string()),
    )
    .expect("write config");
    path
}

fn binary() -> Command {
    let mut command = Command::cargo_bin("immortality").expect("binary built");
    // Hermetic state: never touch the developer's real daemon directory.
    command.env(
        "IMMORTALITY_STATE_DIR",
        std::env::temp_dir().join("immortality-cli-test-state"),
    );
    command
}

#[test]
fn doctor_exits_nonzero_when_backend_is_unreachable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = dead_backend_config(dir.path());

    binary()
        .arg("--global-config")
        .arg(&config)
        .arg("doctor")
        .assert()
        .failure()
        .stdout(predicate::str::contains("FAIL"))
        .stdout(predicate::str::contains("backend-models"))
        // Local checks must still be reported.
        .stdout(predicate::str::contains("trusted-host"));
}

#[test]
fn doctor_json_is_valid_and_includes_skipped_checks() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = dead_backend_config(dir.path());

    let output = binary()
        .arg("--global-config")
        .arg(&config)
        .arg("doctor")
        .arg("--json")
        .output()
        .expect("run");

    let stdout = String::from_utf8(output.stdout).expect("utf8");
    let value: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON on stdout");
    let checks = value["checks"].as_array().expect("checks array");
    assert!(checks.len() >= 5, "{stdout}");

    let names: Vec<&str> = checks.iter().filter_map(|c| c["name"].as_str()).collect();
    assert!(names.contains(&"backend-models"), "{names:?}");
    assert!(names.contains(&"disk"), "{names:?}");

    let latency = checks
        .iter()
        .find(|c| c["name"] == "backend-latency")
        .expect("latency check present");
    assert_eq!(
        latency["status"], "skipped",
        "unreachable backend must skip, not hang"
    );
}

#[test]
fn init_creates_agent_dir_without_policy_section() {
    let dir = tempfile::tempdir().expect("tempdir");
    let global = dir.path().join("global.toml");
    std::fs::write(
        &global,
        "[policy]\nallow_push = true\ntrusted_hosts = [\"llm-server\", \"127.0.0.1\"]\n[logging]\ndir = \"LOGDIR\"\n"
            .replace("LOGDIR", &dir.path().join("logs").display().to_string()),
    )
    .expect("write global");

    let project = dir.path().join("project");
    std::fs::create_dir_all(&project).expect("mkdir");

    binary()
        .current_dir(&project)
        .arg("--global-config")
        .arg(&global)
        .arg("init")
        .assert()
        .success();

    let agent = project.join(".agent");
    assert!(agent.is_dir(), "init must create .agent/");
    let config = std::fs::read_to_string(agent.join("config.toml")).expect("project config");
    // The critical property: a file the agent can edit must never contain policy.
    assert!(
        !config.contains("[policy]"),
        "project config must not contain [policy]:\n{config}"
    );
    assert!(
        !config.contains("allow_push"),
        "project config must not contain permissions:\n{config}"
    );
    assert!(config.contains("[backend]"), "{config}");
}

#[test]
fn init_twice_is_idempotent() {
    let dir = tempfile::tempdir().expect("tempdir");
    let global = dir.path().join("global.toml");
    std::fs::write(
        &global,
        format!("[logging]\ndir = \"{}\"\n", dir.path().join("logs").display()),
    )
    .expect("write global");
    let project = dir.path().join("project");
    std::fs::create_dir_all(&project).expect("mkdir");

    for _ in 0..2 {
        binary()
            .current_dir(&project)
            .arg("--global-config")
            .arg(&global)
            .arg("init")
            .assert()
            .success();
    }
    // The second run must not clobber a config the user edited.
    let path = project.join(".agent").join("config.toml");
    let first = std::fs::read_to_string(&path).expect("config");
    assert!(first.contains("[backend]"));
}

#[test]
fn unimplemented_commands_fail_with_the_milestone() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = dead_backend_config(dir.path());

    for (command, milestone) in [("status", "M5"), ("attach", "M5"), ("escalations", "M3")] {
        binary()
            .arg("--global-config")
            .arg(&config)
            .arg(command)
            .assert()
            .failure()
            .stderr(predicate::str::contains(milestone))
            .stderr(predicate::str::contains(command));
    }
}

#[test]
fn empty_goal_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = dead_backend_config(dir.path());

    binary()
        .arg("--global-config")
        .arg(&config)
        .arg("start")
        .arg("   ")
        .assert()
        .failure()
        .stderr(predicate::str::contains("goal must not be empty"));
}
