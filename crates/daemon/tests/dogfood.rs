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
use std::sync::Arc;
use std::time::{Duration, Instant};

use immortality_core::config::{Config, Policy};
use immortality_core::domain::TaskState;
use immortality_core::journal::JournalReader;
use immortality_core::metrics;
use immortality_core::plan::{Plan, SCHEMA_VERSION};
use immortality_core::registry::Registry;
use immortality_daemon::runner::{FINISH_SUBTASK, Model, ModelRequest, Reply};
use immortality_daemon::server::{Events, ServerState};
use immortality_daemon::start;

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

/// Apply `patches` (path, `old`, `new`) to the admitted repository, the way a model would.
///
/// Exact text replacement rather than a whole-file write: a patch that no longer matches is a
/// fixture that drifted, and silently writing a file would hide that.
fn apply(repo: &Path, patches: &[(&str, &str, &str)]) {
    for (path, old, new) in patches {
        let full = repo.join(path);
        let text = std::fs::read_to_string(&full).expect("read the file to patch");
        assert!(
            text.contains(old),
            "the patch for {path} no longer matches; the fixture drifted:\n{text}"
        );
        std::fs::write(&full, text.replacen(old, new, 1)).expect("write the patch");
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

// ── the rehearsal ──────────────────────────────────────────────────────────────────────
//
// The tests above prove the fixture is red and that admitting it produces a repository the run
// can use. This one proves the whole production path works on it: `start` is called the way
// `serve` calls it, the cycle builds a real worktree, `RealGates` runs the plan's real
// `cargo test --offline` inside it, and the task reaches `ready_for_review` because the work made
// the gate green rather than because nothing checked it.
//
// # Why this is not a live run
//
// The model is the one seam; everything else — git, the worktree, the policy gate, the check
// runner, the journal — is production code. The backend is unreachable during development, and a
// milestone that can only be tested when a server is up is a milestone that is never tested. What
// this cannot cover is whether a real model writes the right code; what it does cover is every
// reason the run would fail *around* the model, which is where the two admission bugs were.

/// The edit that makes `line-count`'s gate green.
///
/// `split` yields a trailing empty field after a final separator, so the fix is to subtract the
/// one line that separator did not start. Filtering empty fields instead would also pass the
/// admitted spec while being wrong about `lines("a\n\nb")`, which is a good reminder that the
/// spec is the bar and not the truth — but a rehearsal that ships a knowingly wrong fix proves
/// less than one that ships a right one. A real model would call `read_file` first; this answers
/// with the edit directly, because the point here is the gate and not the exploration.
const LINE_COUNT_FIX: &str = r#"{"path":"src/count.rs","old":"    text.split('\\n').count()","new":"    text.split('\\n').count() - usize::from(text.ends_with('\\n'))"}"#;

/// A model that answers every role, and for the executor performs one real edit.
///
/// The planner is never asked here — the plan is admitted, and `start` respects it — so an answer
/// for it would only hide a bug where the plan was silently regenerated.
#[derive(Debug)]
struct Rehearsal {
    /// The `edit_file` arguments, sent on the first executor turn.
    edit: String,
    executor_calls: std::sync::atomic::AtomicUsize,
}

impl Rehearsal {
    fn new(edit: &str) -> Arc<Self> {
        Arc::new(Self {
            edit: edit.to_string(),
            executor_calls: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    fn executor_calls(&self) -> usize {
        self.executor_calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl Model for Rehearsal {
    fn complete(&self, request: &ModelRequest) -> Result<Reply, String> {
        let system = request
            .messages
            .first()
            .and_then(|message| message.content.clone())
            .unwrap_or_default();
        if system == immortality_core::planner::SYSTEM {
            return Err("the admitted plan must be run, not replaced".to_string());
        }
        if system == immortality_core::gate::REVIEW_SYSTEM {
            return Ok(Reply::text(r#"{"verdict":"approve"}"#));
        }
        let turn = self
            .executor_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if turn == 0 {
            Ok(Reply::call("call_edit", "edit_file", self.edit.clone()))
        } else {
            Ok(Reply::call(
                "call_finish",
                FINISH_SUBTASK,
                r#"{"summary":"lines no longer counts the trailing empty field"}"#,
            ))
        }
    }
}

/// How long the rehearsal waits for the run's own thread.
const REHEARSAL_BUDGET: Duration = Duration::from_secs(180);

/// A registered project around `repo`, with a state root and the daemon's config moved over it.
fn rehearsal_state(repo: &Path, model: Arc<dyn Model>) -> (tempfile::TempDir, ServerState) {
    let state = tempfile::tempdir().expect("state dir");
    std::fs::write(state.path().join("daemon.token"), "fixture-token").expect("token");
    // The policy confines reads to `workspace_root`, and the default is `~/rust_projects`; the
    // fixture lives in a temp directory, so the root has to name it.
    let config = Config {
        policy: Policy {
            workspace_root: repo.parent().expect("parent").to_path_buf(),
            ..Policy::default()
        },
        ..Config::default()
    };
    let mut registry =
        Registry::open(state.path().join(immortality_core::registry::FILE_NAME)).expect("registry");
    registry.register(repo, None).expect("register");
    let server = ServerState::discover(state.path())
        .expect("state")
        .with_config(&config)
        .with_model(model)
        .with_events(Events::new());
    (state, server)
}

/// Wait until the journal of `task` last announced `wanted`, or fail with what it did say.
fn wait_for_state(repo: &Path, task: &str, wanted: TaskState) -> Vec<String> {
    let agent = repo.join(immortality_core::paths::AGENT_DIR);
    let id = immortality_core::ids::TaskId::parse(task).expect("task id");
    let deadline = Instant::now() + REHEARSAL_BUDGET;
    loop {
        let states: Vec<String> = JournalReader::open(&agent, &id)
            .expect("reader")
            .read()
            .expect("records")
            .records
            .iter()
            .filter(|record| record.fields.kind == immortality_core::journal::Kind::TaskState)
            .filter_map(|record| {
                record
                    .event
                    .payload
                    .get("state")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            })
            .collect();
        if states.last().map(String::as_str) == Some(wanted.as_str()) {
            return states;
        }
        // A settled run is not going to announce anything else, so waiting the full budget for a
        // state it has already passed turns a wrong answer into a three-minute test. `failed_gates`
        // is the one that matters here: it is what a green-before-work fixture produces.
        if let Some(last) = states.last()
            && matches!(
                last.as_str(),
                "ready_for_review" | "failed_gates" | "blocked" | "merged" | "cancelled"
            )
        {
            panic!("the run settled on `{last}` instead of `{wanted}`; it announced {states:?}");
        }
        assert!(
            Instant::now() <= deadline,
            "the run never reached `{wanted}`; it announced {states:?}"
        );
        std::thread::sleep(Duration::from_millis(25));
    }
}

#[test]
fn a_dogfood_task_reaches_ready_for_review_through_the_real_gate() {
    // The M6.2 rehearsal: the production handler, a real worktree, the plan's real `cargo test`,
    // and a model that makes one real edit. The task may only report `ready_for_review` if the
    // gate went from red to green, so this is also the assertion that the fixture is a task and
    // not a no-op.
    let scratch = tempfile::tempdir().expect("tempdir");
    let repo = scratch.path().join("repo");
    admit_into("line-count", &repo);

    // Red before the work, in the main checkout, so the run starts from the state the milestone
    // describes. The worktree's own red is asserted by `every_dogfood_task_starts_with_a_failing_gate`.
    if let Err(message) = gate_fails(&repo) {
        panic!("the admitted gate is green before the work: {message}");
    }

    let model = Rehearsal::new(LINE_COUNT_FIX);
    let (_state_dir, state) = rehearsal_state(&repo, model.clone());

    let answer = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(start::start(
            &state,
            &serde_json::json!({ "project": repo, "goal": "fix the line count" }),
        ))
        .map_err(|error| error.message)
        .expect("the task must be admitted");
    let task = answer["task"].as_str().expect("a task id").to_string();
    assert_eq!(answer["planned"], false, "the admitted plan must be respected");

    let states = wait_for_state(&repo, &task, TaskState::ReadyForReview);
    assert_eq!(
        states,
        vec![
            "queued",
            "planning",
            "awaiting_plan_approval",
            "running",
            "verifying",
            "ready_for_review",
        ],
        "the run took an unexpected path: {states:?}"
    );
    assert!(
        model.executor_calls() >= 2,
        "the edit and the claim are two turns"
    );

    // The work exists on the task's branch, in the worktree the run created — not in the admitted
    // checkout, which is the difference between a run and a simulation.
    let worktree = immortality_core::worktree::WorktreeManager::new(&repo).path_for(&task);
    assert!(worktree.is_dir(), "the run must have a worktree");
    let count = std::fs::read_to_string(worktree.join("src").join("count.rs")).expect("count.rs");
    let admitted = std::fs::read_to_string(repo.join("src").join("count.rs")).expect("count.rs");
    assert_ne!(
        count, admitted,
        "the run claimed the subtask without changing the rule it was asked to fix"
    );

    // The gate really is green there now, and that is why the subtask closed.
    let gate = Command::new("cargo")
        .args(["test", "--offline", "--quiet"])
        .current_dir(&worktree)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .expect("run the gate");
    assert!(
        gate.status.success(),
        "the run closed a subtask whose gate is red; stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&gate.stdout),
        String::from_utf8_lossy(&gate.stderr)
    );

    // And the numbers a reader of `docs/metrics-v0.1.md` will check are in the journal, not in
    // this test: one subtask, closed green, no escalation.
    let id = immortality_core::ids::TaskId::parse(&task).expect("task id");
    let records = JournalReader::open(repo.join(immortality_core::paths::AGENT_DIR), &id)
        .expect("reader")
        .read()
        .expect("records");
    let report = metrics::compute(&task, &records.records);
    assert_eq!(report.closed_green, 1, "{report:?}");
    assert_eq!(report.closed_with_escalation, 0, "{report:?}");
    assert_eq!(report.escalations_pending, 0, "{report:?}");
}

/// The smallest honest fix for each task, as a model reading the failing spec would write it.
///
/// # Why this is a test
///
/// A fixture whose gate cannot go green is worse than no fixture: it reports "the model failed"
/// for a task no model could pass, and the milestone's bar would be unreachable for a reason
/// outside the agent. Red-before-work is only half the property; green-after-a-real-fix is the
/// other half, and it is the half nobody notices is missing until three live runs fail the same
/// way.
fn fixes(name: &str) -> Vec<(&'static str, &'static str, &'static str)> {
    match name {
        "line-count" => vec![(
            "src/count.rs",
            "    text.split('\\n').count()",
            "    text.split('\\n').count() - usize::from(text.ends_with('\\n'))",
        )],
        "word-count" => vec![
            (
                "src/count.rs",
                "/// Number of distinct words, compared case-insensitively.",
                "/// Number of words that appear exactly once, compared case-insensitively.\n\
                 pub fn once_words(text: &str) -> usize {\n\
                 \x20   let mut counts: std::collections::BTreeMap<String, usize> =\n\
                 \x20       std::collections::BTreeMap::new();\n\
                 \x20   for word in text.split_whitespace() {\n\
                 \x20       *counts.entry(normalise(word)).or_default() += 1;\n\
                 \x20   }\n\
                 \x20   counts.values().filter(|count| **count == 1).count()\n\
                 }\n\n\
                 /// Number of distinct words, compared case-insensitively.",
            ),
            (
                "src/main.rs",
                "    println!(\"unique: {}\", count::unique_words(&text));",
                "    println!(\"unique: {}\", count::unique_words(&text));\n\
                 \x20   println!(\"once: {}\", count::once_words(&text));",
            ),
        ],
        "dedupe" => vec![(
            "src/count.rs",
            "pub fn unique_words(text: &str) -> usize {\n\
             \x20   let mut seen: Vec<String> = Vec::new();\n\
             \x20   for word in text.split_whitespace() {\n\
             \x20       let normalised = normalise(word);\n\
             \x20       if !seen.contains(&normalised) {\n\
             \x20           seen.push(normalised);\n\
             \x20       }\n\
             \x20   }\n\
             \x20   seen.len()\n\
             }",
            "pub fn unique_words(text: &str) -> usize {\n\
             \x20   unique_words_sorted(text).len()\n\
             }\n\n\
             /// The same normalised words as [`unique_words`], sorted.\n\
             pub fn unique_words_sorted(text: &str) -> Vec<String> {\n\
             \x20   let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();\n\
             \x20   for word in text.split_whitespace() {\n\
             \x20       seen.insert(normalise(word));\n\
             \x20   }\n\
             \x20   seen.into_iter().collect()\n\
             }",
        )],
        other => panic!("no fix recorded for `{other}`; every dogfood task needs one"),
    }
}

#[test]
fn every_dogfood_task_can_be_made_green_by_a_plausible_fix() {
    // The other half of "the gate is red before the work". A gate that nothing can satisfy makes
    // every run fail for a reason that has nothing to do with the agent, and the failure reads
    // exactly like a model that could not do the job. So each task's spec must be green after the
    // kind of edit the goal asks for, and the existing tests must stay green with it.
    for task in tasks() {
        let name = task.file_name().expect("name").to_string_lossy().into_owned();
        let scratch = tempfile::tempdir().expect("tempdir");
        let target = scratch.path().join("repo");
        admit_into(&name, &target);
        if let Err(message) = gate_fails(&target) {
            panic!("{name} is green before the fix: {message}");
        }

        apply(&target, &fixes(&name));

        let gate = Command::new("cargo")
            .args(["test", "--offline", "--quiet"])
            .current_dir(&target)
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .expect("run the gate");
        assert!(
            gate.status.success(),
            "{name} is still red after a plausible fix, so no run could pass it; stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&gate.stdout),
            String::from_utf8_lossy(&gate.stderr)
        );
    }
}
