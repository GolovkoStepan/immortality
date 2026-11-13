//! The dogfood fixtures are tasks the daemon would accept, and each one's gate actually starts red.
//!
//! # Why this is a test and not a note in `dogfood/README.md`
//!
//! A fixture task is a claim that a run can execute it and that success means something.
//! `Plan::load` is the authority on the first half: a dangling dependency, a `DoD` with no gate or
//! a `depth` that lies about nesting all make the fixture a task the daemon would refuse before
//! spending a single model call.
//!
//! The second half is subtler and is the reason this file runs the fixture repository's own gate:
//! a gate that passes on the untouched repository closes a subtask that changed nothing, and M6's
//! criterion would be met by a run that did no work. So each task installs a failing test through
//! `admit.sh`, and the red run is asserted here rather than remembered during a dogfood session.
//!
//! # Why `cargo` is invoked for real
//!
//! The fixture repository is a dependency-free crate, so its test gate is a local compile and run
//! in under a second. A fake `cargo` would prove that this test calls something, not that the
//! fixture is red — and "is it red" is exactly the claim.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use immortality_core::plan::{Plan, SCHEMA_VERSION};

/// The deepest a task may nest, as `Config::default` sets it.
const MAX_DEPTH: u8 = 3;

fn dogfood() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("dogfood")
}

fn repo() -> PathBuf {
    dogfood().join("repo")
}

fn admit() -> PathBuf {
    dogfood().join("scripts").join("admit.sh")
}

/// Every task directory under `dogfood/tasks`, sorted so a failure names them in a stable order.
fn tasks() -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dogfood().join("tasks"))
        .expect("dogfood/tasks must exist")
        .map(|entry| entry.expect("entry").path())
        .filter(|path| path.is_dir())
        .collect();
    found.sort();
    found
}

/// A fresh copy of the fixture repository, so admitting into it cannot disturb the checked-in one.
fn copy_of_repo(into: &Path) {
    for entry in std::fs::read_dir(repo()).expect("repo") {
        let entry = entry.expect("entry");
        let name = entry.file_name();
        // `target/` is build output; copying it would make each test pay for a stale tree and
        // would compare a binary built from other sources.
        if name == "target" || name == ".agent" || name == "tests" {
            continue;
        }
        let from = entry.path();
        let to = into.join(&name);
        if from.is_dir() {
            copy_dir(&from, &to);
        } else {
            std::fs::copy(&from, &to).expect("copy");
        }
    }
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("mkdir");
    for entry in std::fs::read_dir(from).expect("read dir") {
        let entry = entry.expect("entry");
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).expect("copy");
        }
    }
}

#[test]
fn every_dogfood_task_is_a_plan_the_daemon_would_execute() {
    let tasks = tasks();
    assert_eq!(
        tasks.len(),
        3,
        "the roadmap asks for three tasks; found {tasks:?}"
    );
    for task in &tasks {
        let plan = Plan::load(task, MAX_DEPTH)
            .unwrap_or_else(|error| panic!("{} does not load: {error}", task.display()));
        // `Plan::load` validates the whole tree — nesting, dangling dependencies, `DoD` gates —
        // so a plan that reaches here is one the daemon would execute. The assertion left to
        // make is that there is something in it to run.
        assert!(!plan.subtasks().is_empty(), "{} has no subtasks", task.display());
    }
}

#[test]
fn every_dogfood_task_has_the_three_files_the_daemon_reads() {
    // `admit.sh` copies three files; a task directory missing one would fail at copy time with a
    // message about a missing path rather than about a broken fixture.
    for task in tasks() {
        for name in ["PLAN.md", "ACCEPTANCE.md", "TASKS.json"] {
            let path = task.join(immortality_core::paths::AGENT_DIR).join(name);
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{} is unreadable: {error}", path.display()));
            assert!(
                !text.trim().is_empty(),
                "{} is empty; a plan file must say something",
                path.display()
            );
        }
        let body: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(task.join(immortality_core::paths::AGENT_DIR).join("TASKS.json"))
                .expect("tasks"),
        )
        .expect("TASKS.json must be JSON");
        assert_eq!(
            body["v"],
            SCHEMA_VERSION,
            "{} declares an unknown schema version",
            task.display()
        );
    }
}

#[test]
fn every_dogfood_task_starts_with_a_failing_gate() {
    // The milestone's criterion is "three subtasks closed with green gates, unaided". A gate that
    // is green before the run makes that criterion satisfiable by doing nothing, so each task's
    // gate is run against a fresh copy of the repository and must fail. This is the assertion
    // that stops a fixture from quietly becoming a no-op.
    for task in tasks() {
        let name = task.file_name().expect("name").to_string_lossy().into_owned();
        let scratch = tempfile::tempdir().expect("tempdir");
        let target = scratch.path().join("repo");
        std::fs::create_dir_all(&target).expect("repo dir");
        copy_of_repo(&target);

        let admitted = Command::new("bash")
            .arg(admit())
            .arg(&name)
            .arg(&target)
            .output()
            .expect("run admit.sh");
        assert!(
            admitted.status.success(),
            "admit.sh failed for {name}: {}",
            String::from_utf8_lossy(&admitted.stderr)
        );

        let gate = Command::new("cargo")
            .args(["test", "--offline", "--quiet"])
            .current_dir(&target)
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .expect("run the fixture's gate");
        assert!(
            !gate.status.success(),
            "the gate of {name} passes before the work, so a subtask could close without \
             changing anything; stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&gate.stdout),
            String::from_utf8_lossy(&gate.stderr)
        );
    }
}
