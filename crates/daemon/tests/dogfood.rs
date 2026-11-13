//! The dogfood fixtures are tasks the daemon would accept, and each one's gate actually starts red.
//!
//! # Why this is a test and not a note in `dogfood/README.md`
//!
//! A fixture task is a claim that a run can execute it and that success means something.
//! `Plan::load` is the authority on the first half: a dangling dependency, a `DoD` with no gate or
//! a `depth` that lies about nesting all make the fixture a task the daemon would refuse before
//! spending a single model call.
//!
//! The second half is subtler and is the reason this file drives `admit.sh` for real: a gate that
//! passes on the untouched repository closes a subtask that changed nothing, and M6's criterion
//! would be met by a run that did no work. Two ways that happens, both silent, and `admit.sh` is
//! written against both:
//!
//!   * the specification is written but not **committed**, so the worktree git hands the agent does
//!     not contain it and `cargo test --offline` is green before anything was done;
//!   * the target is not its own git repository, so `git worktree add` produces a worktree of
//!     `immortality` itself and the agent is handed the tool's workspace instead of the fixture.
//!
//! So the test admits into a temp directory, asserts the result is a repository whose HEAD already
//! contains the failing specification, and runs the gate in a **worktree** of it — which is where
//! the run actually works.
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

/// Run git in `dir`, failing the test with git's own words.
fn git(dir: &Path, args: &[&str]) {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("LC_ALL", "C")
        .output()
        .expect("run git");
    assert!(
        output.status.success(),
        "git {args:?} in {} failed: {}",
        dir.display(),
        String::from_utf8_lossy(&output.stderr)
    );
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

/// Whether `cargo test --offline` in `dir` fails, with its output for the message.
fn gate_fails(dir: &Path) -> Result<(), String> {
    let gate = Command::new("cargo")
        .args(["test", "--offline", "--quiet"])
        .current_dir(dir)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .map_err(|error| format!("cannot run cargo in {}: {error}", dir.display()))?;
    if gate.status.success() {
        return Err(format!(
            "the gate passed; stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&gate.stdout),
            String::from_utf8_lossy(&gate.stderr)
        ));
    }
    Ok(())
}

/// Admit `name` into `target`, failing the test with the script's own words.
fn admit_into(name: &str, target: &Path) {
    let admitted = Command::new("bash")
        .arg(admit())
        .arg(name)
        .arg(target)
        .output()
        .expect("run admit.sh");
    assert!(
        admitted.status.success(),
        "admit.sh failed for {name}: {}",
        String::from_utf8_lossy(&admitted.stderr)
    );
}

#[test]
fn every_dogfood_task_starts_with_a_failing_gate() {
    // The milestone's criterion is "three subtasks closed with green gates, unaided". A gate that
    // is green before the run makes that criterion satisfiable by doing nothing, so each task's
    // gate is run in the directory the run actually works in and must fail. This is the assertion
    // that stops a fixture from quietly becoming a no-op.
    for task in tasks() {
        let name = task.file_name().expect("name").to_string_lossy().into_owned();
        let scratch = tempfile::tempdir().expect("tempdir");
        let target = scratch.path().join("repo");
        admit_into(&name, &target);

        // The worktree, not the main checkout: `start` hands the agent a worktree, and a
        // specification that was installed but never committed exists only in the checkout.
        let worktree = scratch.path().join("worktree");
        git(
            &target,
            &[
                "worktree",
                "add",
                "-q",
                &worktree.to_string_lossy(),
                "-b",
                "agent/probe",
            ],
        );
        assert!(
            worktree.join("tests").join("behaviour.rs").exists(),
            "{name}: the specification is not in the worktree, so the gate there is green"
        );
        if let Err(message) = gate_fails(&worktree) {
            panic!(
                "the gate of {name} passes before the work, so a subtask could close without changing anything: {message}"
            );
        }
    }
}

#[test]
fn an_admitted_task_is_a_git_repository_of_its_own() {
    // `git worktree add` run inside `dogfood/repo` would produce a worktree of *immortality*, not
    // of the fixture, because `dogfood/repo` is a plain directory in this repository. The run
    // would then hand the agent the tool's workspace and ask it to implement `count::once_words`
    // in it. `admit.sh` copies to a target and runs `git init` there, and this is the assertion
    // that the result is what `WorktreeManager` needs.
    let scratch = tempfile::tempdir().expect("tempdir");
    let target = scratch.path().join("repo");
    admit_into("line-count", &target);

    let top = Command::new("git")
        .arg("-C")
        .arg(&target)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .expect("run git");
    assert!(top.status.success(), "{}", String::from_utf8_lossy(&top.stderr));
    let reported = PathBuf::from(String::from_utf8_lossy(&top.stdout).trim());
    // macOS reports `/private/var/...` for a path under `/var/...`; compare the final component
    // and the file identity rather than a string the tempdir spelling can break.
    assert_eq!(
        reported.file_name(),
        target.file_name(),
        "the admitted target is not its own repository: git says {}",
        reported.display()
    );
    // The failing specification must already be committed: an untracked file is absent from every
    // worktree, and a worker that cannot see the test cannot fail it.
    let shown = Command::new("git")
        .arg("-C")
        .arg(&target)
        .args(["show", "HEAD:tests/behaviour.rs"])
        .output()
        .expect("run git");
    assert!(
        shown.status.success(),
        "tests/behaviour.rs is not in the admitted commit: {}",
        String::from_utf8_lossy(&shown.stderr)
    );
    // And the base checkout's own gate is red too, so a run that somehow worked in place rather
    // than in a worktree is still measured honestly.
    if let Err(message) = gate_fails(&target) {
        panic!("the admitted checkout's gate passes before the work: {message}");
    }
}
