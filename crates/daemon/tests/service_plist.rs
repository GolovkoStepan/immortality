//! The generated plist, validated by the tool that will consume it.
//!
//! # Why `plutil` and not an XML parser
//!
//! `launchd` reads plists through CoreFoundation, and `plutil -lint` is the same parser. A
//! hand-rolled check ("contains `<dict>`") proves nothing about whether the file loads; a plist
//! that is well-formed XML with a duplicated key is still rejected by launchd, and the only
//! symptom would be a daemon that never starts with no message anywhere.
//!
//! # Why this file is macOS-only
//!
//! `plutil` ships with macOS, and so does the consumer of what it validates. On Linux the tests
//! would fail on a missing binary, which says nothing about the plist; skipping them keeps the
//! Linux job honest about what it can actually prove. The macOS job runs on every push, so the
//! coverage is not lost — it simply lives where the toolchain does.

#![cfg(target_os = "macos")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;
use std::process::Command;

use immortality_daemon::service::{self, ServiceSpec};

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
fn the_generated_plist_is_valid_to_launchds_own_parser() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("agent.plist");
    std::fs::write(&path, service::plist(&spec())).expect("write");

    let output = Command::new("plutil")
        .arg("-lint")
        .arg(&path)
        .output()
        .expect("plutil");
    assert!(
        output.status.success(),
        "launchd would refuse this plist: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn the_plist_round_trips_through_plutil_with_the_expected_keys() {
    // Parsed back, not just linted: a plist whose keys are spelled `Keepalive` is valid XML and
    // silently does nothing, which is the failure this project cannot see from inside.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("agent.plist");
    std::fs::write(&path, service::plist(&spec())).expect("write");

    let output = Command::new("plutil")
        .args(["-convert", "json", "-o", "-"])
        .arg(&path)
        .output()
        .expect("plutil");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).expect("plutil emits JSON");

    assert_eq!(value["Label"], service::LABEL);
    assert_eq!(value["KeepAlive"], true);
    assert_eq!(value["RunAtLoad"], true);
    assert_eq!(value["ProcessType"], "Background");
    assert_eq!(
        value["EnvironmentVariables"]["IMMORTALITY_STATE_DIR"],
        "/Users/test/Library/Application Support/immortality"
    );
    let arguments = value["ProgramArguments"]
        .as_array()
        .expect("ProgramArguments is an array");
    assert_eq!(arguments[0], "/Users/test/bin/immortality");
    assert_eq!(arguments[1], "serve");
}

#[test]
fn the_path_from_install_time_reaches_launchd_as_one_string() {
    // The daemon inherits the *installed* environment, not the shell's. launchd's own default is
    // `/usr/bin:/bin:/usr/sbin:/sbin`, which on a machine whose Ruby or Node comes from asdf finds
    // an older system interpreter first — the run then either cannot find its toolchain or, worse,
    // uses the wrong one. This asserts the PATH survives CoreFoundation's parser intact.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("agent.plist");
    let with_path = ServiceSpec {
        path: Some("/Users/test/.asdf/shims:/opt/homebrew/bin:/usr/bin:/bin".to_string()),
        ..spec()
    };
    std::fs::write(&path, service::plist(&with_path)).expect("write");

    let output = Command::new("plutil")
        .args(["-convert", "json", "-o", "-"])
        .arg(&path)
        .output()
        .expect("plutil");
    assert!(
        output.status.success(),
        "launchd would refuse this plist: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).expect("plutil emits JSON");
    assert_eq!(
        value["EnvironmentVariables"]["PATH"],
        "/Users/test/.asdf/shims:/opt/homebrew/bin:/usr/bin:/bin"
    );
}

#[test]
fn the_lang_from_install_time_reaches_launchd_as_one_string() {
    // The measured defect behind this (D-153): launchd gives an agent no locale, Ruby then reports
    // `Encoding.default_external == US-ASCII`, and every non-ASCII byte in an RSpec failure becomes
    // `?`. 48 journal records of the third `black_jack` run are that mojibake, including the one
    // failure the model needed to read. Asserted through `plutil`, not as text in the file: what
    // matters is the value launchd hands the process.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("agent.plist");
    let with_lang = ServiceSpec {
        path: Some("/opt/homebrew/bin:/usr/bin:/bin".to_string()),
        lang: Some("ru_RU.UTF-8".to_string()),
        ..spec()
    };
    std::fs::write(&path, service::plist(&with_lang)).expect("write");

    let output = Command::new("plutil")
        .args(["-convert", "json", "-o", "-"])
        .arg(&path)
        .output()
        .expect("plutil");
    assert!(
        output.status.success(),
        "launchd would refuse this plist: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).expect("plutil emits JSON");
    assert_eq!(value["EnvironmentVariables"]["LANG"], "ru_RU.UTF-8");
    // Both variables coexist: a plist that lost PATH while gaining LANG would fix the mojibake
    // and break the toolchain, which is a worse trade than either bug alone.
    assert_eq!(
        value["EnvironmentVariables"]["PATH"],
        "/opt/homebrew/bin:/usr/bin:/bin"
    );
}

#[test]
fn an_escaped_path_survives_launchds_parser_as_the_original_text() {
    // The escape function's real contract: what launchd hands the program is the path, not the
    // escaped spelling of it. A path with `&` is legal on macOS.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("agent.plist");
    let odd = ServiceSpec {
        program: PathBuf::from("/Users/a&b/c/immortality"),
        ..spec()
    };
    std::fs::write(&path, service::plist(&odd)).expect("write");

    let output = Command::new("plutil")
        .args(["-convert", "json", "-o", "-"])
        .arg(&path)
        .output()
        .expect("plutil");
    assert!(output.status.success());
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).expect("plutil emits JSON");
    assert_eq!(value["ProgramArguments"][0], "/Users/a&b/c/immortality");
}
