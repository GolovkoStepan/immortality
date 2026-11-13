//! The dogfood fixtures load as plans.
//!
//! # Why this is a test and not a note in `dogfood/README.md`
//!
//! A fixture task is a claim that a run can execute it. `Plan::load` is the only authority on
//! that claim: a dangling dependency, a `DoD` with no gate, a `depth` that lies about nesting or
//! a timestamp that does not parse all make the fixture a task the daemon would refuse before
//! spending a single model call. Hand-written JSON drifts the moment a field is renamed, and a
//! fixture that cannot load would show up as a mysterious failure hours into M6.2.
//!
//! # Why the workspace root and not a copy
//!
//! The fixtures are the tasks a human will actually run, so the test must read the same bytes.
//! A copy under `tests/` would be a second authority and would pass while the real file rotted.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};

use immortality_core::plan::{Plan, SCHEMA_VERSION};

/// The deepest a task may nest, as `Config::default` sets it.
const MAX_DEPTH: u8 = 3;

fn dogfood() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("dogfood")
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
fn every_dogfood_task_names_itself_in_the_three_files_the_daemon_reads() {
    // The admit script copies three files; a task directory missing one would fail at copy time
    // with a message about a missing path rather than about a broken fixture.
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
fn the_line_count_task_reproduces_a_real_bug_before_the_fix() {
    // The bugfix task is only a fair test if the bug exists. This pins the defect against the
    // fixture repository's own code, so a fixture edit that accidentally fixes it makes this
    // test fail rather than producing a task with nothing to do.
    let source =
        std::fs::read_to_string(dogfood().join("repo").join("src").join("count.rs")).expect("count.rs");
    assert!(
        source.contains("fn lines"),
        "the fixture's counting module moved; the bugfix task points at a function that is gone"
    );
    // The defect is `split('\n').count()` without trimming a trailing empty field. It is present
    // in the initial fixture and the task exists to remove it; the assertion is on the *shape*
    // that produces the bug, not on the bug being permanent.
    let body = source.split("pub fn lines").nth(1).expect("the function body");
    let body = body.split("pub fn words").next().expect("the body ends");
    assert!(
        body.contains("split('\\n')"),
        "the fixture's `lines` no longer splits on newlines, so the bugfix task's premise is gone: {body}"
    );
}
