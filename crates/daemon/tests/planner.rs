//! `plan --goal`: the daemon asking a model for a tree, and what it does with the answer.
//!
//! # Why the model is injected
//!
//! Every one of these tests is about what the handler *asks* and what it *writes*, not about a
//! backend's JSON. A test that needed a server could not run in CI, and this project's rules say
//! tests never touch the network. [`Script`] is the same seam the runner's own tests use.
//!
//! # Why the refusals carry most of the weight
//!
//! A planner that turns a good goal into a good plan would pass almost anything. What makes it
//! safe for an unattended run is that a bad goal, a bad answer, and an existing plan are each
//! refused *without writing anything*, because a half-written plan is a task that looks planned
//! and cannot start.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use immortality_core::domain::TaskId;
use immortality_core::plan::Plan;
use immortality_core::registry::Registry;
use immortality_daemon::planner;
use immortality_daemon::rpc::ErrorCode;
use immortality_daemon::runner::{Model, Reply};
use immortality_daemon::server::ServerState;
use immortality_llm::wire::ChatRequest;

/// A model that returns whatever it was given and remembers the requests it saw.
#[derive(Debug, Default)]
struct Script {
    answer: Mutex<String>,
    seen: Mutex<Vec<ChatRequest>>,
    calls: AtomicUsize,
}

impl Script {
    fn answering(answer: &str) -> Self {
        Self {
            answer: Mutex::new(answer.to_string()),
            ..Self::default()
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn last_request(&self) -> ChatRequest {
        self.seen
            .lock()
            .expect("the test holds no other lock")
            .last()
            .cloned()
            .expect("the model must have been asked")
    }
}

impl Model for Script {
    fn complete(&self, request: &ChatRequest) -> Result<Reply, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.seen
            .lock()
            .map_err(|_| "the test poisoned the lock".to_string())?
            .push(request.clone());
        let answer = self
            .answer
            .lock()
            .map_err(|_| "the test poisoned the lock".to_string())?
            .clone();
        Ok(Reply::text(answer))
    }
}

/// A registered repository with one commit, plus a state root with a token.
struct Fixture {
    dir: tempfile::TempDir,
    repo: std::path::PathBuf,
    state_root: std::path::PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let repo = dir.path().join("repo");
        let state_root = dir.path().join("state-root");
        std::fs::create_dir_all(&repo).expect("repo");
        std::fs::create_dir_all(&state_root).expect("state root");
        std::fs::write(state_root.join("daemon.token"), "fixture-token").expect("token");
        std::fs::write(repo.join("Cargo.toml"), "[package]\nname = \"demo\"\n").expect("cargo");
        std::fs::write(repo.join("README.md"), "# demo\n").expect("readme");

        let mut registry =
            Registry::open(state_root.join(immortality_core::registry::FILE_NAME)).expect("registry");
        registry.register(&repo, None).expect("register");
        Self {
            dir,
            repo,
            state_root,
        }
    }

    fn state(&self, model: &std::sync::Arc<Script>) -> ServerState {
        ServerState::discover(&self.state_root)
            .expect("state")
            .with_model(model.clone())
    }

    fn params(&self, extra: &serde_json::Value) -> serde_json::Value {
        let mut params = serde_json::json!({ "project": self.repo });
        if let (Some(target), Some(source)) = (params.as_object_mut(), extra.as_object()) {
            for (key, value) in source {
                target.insert(key.clone(), value.clone());
            }
        }
        params
    }
}

/// An answer a compiler accepts: one subtask, a real check, no ids.
fn good_answer() -> String {
    serde_json::json!({
        "plan": "# Plan\n\nAdd the parser.",
        "acceptance": "`cargo test` passes and `parser::parse` exists.",
        "subtasks": [{
            "name": "parse",
            "title": "write the parser",
            "goal": "add `parser::parse` with the documented behaviour",
            "dod": {
                "statement": "the parser handles the documented cases",
                "checks": [{ "id": "test", "command": "cargo test" }],
                "review": false,
            },
        }],
    })
    .to_string()
}

fn run_plan(
    state: &ServerState,
    params: &serde_json::Value,
) -> Result<serde_json::Value, immortality_daemon::rpc::RpcError> {
    // The handler is async because it awaits a blocking task; a current-thread runtime is enough
    // and keeps the test from starting a thread pool per case.
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(planner::write_plan(state, params))
}

// ── the happy path ───────────────────────────────────────────────────────────

#[test]
fn a_goal_becomes_three_files_the_loader_accepts() {
    let fixture = Fixture::new();
    let model = std::sync::Arc::new(Script::answering(&good_answer()));
    let result = run_plan(
        &fixture.state(&model),
        &fixture.params(&serde_json::json!({
            "goal": "add a parser",
        })),
    )
    .expect("the plan must be written");

    assert_eq!(result["count"], 1);
    assert_eq!(result["goal"], "add a parser");
    assert_eq!(result["created"], true);
    // The written tree is loadable by the same loader the scheduler uses: the planner does not
    // have a looser idea of a valid plan than the thing that will execute it.
    let loaded = Plan::load(&fixture.repo, 2).expect("the written plan must load");
    assert_eq!(loaded.subtasks().len(), 1);
    assert!(fixture.repo.join(".agent").join("PLAN.md").exists());
    assert!(fixture.repo.join(".agent").join("ACCEPTANCE.md").exists());
}

#[test]
fn a_plan_cannot_be_forced_over_a_running_project() {
    // `plan --force` used to check only whether a plan was on disk, never whether a task was
    // executing it. The planner would then rewrite `PLAN.md` and `TASKS.json` under a live run,
    // and the run would continue against subtasks that no longer existed. `start --force` claims
    // the project's slot first; this is the same rule on the other door.
    let fixture = Fixture::new();
    let model = std::sync::Arc::new(Script::answering(&good_answer()));
    let state = fixture.state(&model);
    let holder = TaskId::generate();
    let project = Registry::open(fixture.state_root.join(immortality_core::registry::FILE_NAME))
        .expect("registry")
        .projects()
        .first()
        .expect("the fixture registers one project")
        .id
        .clone();
    state
        .running
        .lock()
        .expect("run set")
        .insert(project, holder.clone());

    let error = run_plan(
        &state,
        &fixture.params(&serde_json::json!({
            "goal": "a brand new goal",
            "force": true,
        })),
    )
    .expect_err("a forced plan over a running project must be refused");

    assert!(error.message.contains("running"), "{}", error.message);
    assert!(
        error.message.contains(&holder.to_string()),
        "the refusal must name the holder: {}",
        error.message
    );
    assert_eq!(model.calls(), 0, "the refusal must precede the model call");
    assert!(
        !fixture.repo.join(".agent").join("PLAN.md").exists(),
        "nothing may be written when the refusal is about a live run"
    );
}

#[test]
fn the_planner_is_asked_with_the_goal_and_a_listing_of_the_repository() {
    // Two things the request must carry, and one it must not: the goal, enough of the repository
    // to know what kind of project it is, and no tools. The absence of tools is what makes the
    // planner write a plan instead of starting the work.
    let fixture = Fixture::new();
    let model = std::sync::Arc::new(Script::answering(&good_answer()));
    run_plan(
        &fixture.state(&model),
        &fixture.params(&serde_json::json!({"goal": "add a parser"})),
    )
    .expect("plan");

    let request = model.last_request();
    assert_eq!(model.calls(), 1);
    assert_eq!(request.model, "swift-iq2_xs");
    assert!(request.tools.is_none(), "the planner must not be given tools");
    assert_eq!(request.temperature, Some(0.0), "a plan must be reproducible");
    assert_eq!(
        request.max_tokens,
        Some(planner::MAX_PLAN_TOKENS),
        "the step default truncates a plan into unparseable JSON"
    );

    let user = request
        .messages
        .iter()
        .find(|message| message.role == "user")
        .expect("a user message");
    let text = user.content.as_deref().unwrap_or_default();
    assert!(text.contains("add a parser"), "{text}");
    assert!(
        text.contains("Cargo.toml"),
        "the listing names the project's shape: {text}"
    );
    // Noise is dropped rather than crowding out the entries that matter. A `target/` in a Rust
    // repo names nothing a planner can use.
    assert!(!text.contains("target/"), "{text}");
}

#[test]
fn the_planner_sees_a_listing_and_not_file_contents() {
    // A planner that read file bodies would be doing the executor's reconnaissance without the
    // executor's tools, and would spend the window before writing a single subtask.
    let fixture = Fixture::new();
    let model = std::sync::Arc::new(Script::answering(&good_answer()));
    run_plan(
        &fixture.state(&model),
        &fixture.params(&serde_json::json!({"goal": "add a parser"})),
    )
    .expect("plan");

    let user = model.last_request().messages[1]
        .content
        .clone()
        .unwrap_or_default();
    // The fixture's README says "# demo"; only the name should reach the planner.
    assert!(!user.contains("# demo"), "{user}");
    assert!(user.contains("README.md"), "{user}");
}

// ── refusals that must not write ─────────────────────────────────────────────

#[test]
fn an_empty_goal_is_refused_without_asking_the_model() {
    let fixture = Fixture::new();
    let model = std::sync::Arc::new(Script::answering(&good_answer()));
    let error = run_plan(
        &fixture.state(&model),
        &fixture.params(&serde_json::json!({"goal": "   "})),
    )
    .expect_err("must refuse");
    assert_eq!(error.code, ErrorCode::BadParams);
    assert_eq!(model.calls(), 0, "a refused goal must cost no tokens");
    assert!(!fixture.repo.join(".agent").exists());
}

#[test]
fn an_existing_plan_is_refused_without_asking_the_model() {
    // The roadmap's rule: a human may rewrite the files before the run. Regenerating them would
    // discard exactly that, and — on a local backend — would cost minutes of prefill first.
    let fixture = Fixture::new();
    std::fs::create_dir_all(fixture.repo.join(".agent")).expect("agent dir");
    std::fs::write(fixture.repo.join(".agent").join("PLAN.md"), "my own plan\n").expect("plan");

    let model = std::sync::Arc::new(Script::answering(&good_answer()));
    let error = run_plan(
        &fixture.state(&model),
        &fixture.params(&serde_json::json!({"goal": "add a parser"})),
    )
    .expect_err("must refuse");
    assert_eq!(error.code, ErrorCode::BadParams);
    assert!(error.message.contains("--force"), "{}", error.message);
    assert!(error.message.contains("PLAN.md"), "{}", error.message);
    assert_eq!(model.calls(), 0, "the refusal must come before the model call");
    // The human's file is still theirs.
    assert_eq!(
        std::fs::read_to_string(fixture.repo.join(".agent").join("PLAN.md")).unwrap(),
        "my own plan\n"
    );
}

#[test]
fn force_replaces_an_existing_plan() {
    let fixture = Fixture::new();
    std::fs::create_dir_all(fixture.repo.join(".agent")).expect("agent dir");
    std::fs::write(fixture.repo.join(".agent").join("PLAN.md"), "my own plan\n").expect("plan");

    let model = std::sync::Arc::new(Script::answering(&good_answer()));
    run_plan(
        &fixture.state(&model),
        &fixture.params(&serde_json::json!({"goal": "add a parser", "force": true})),
    )
    .expect("force must replace");

    let text = std::fs::read_to_string(fixture.repo.join(".agent").join("PLAN.md")).unwrap();
    assert!(!text.contains("my own plan"), "{text}");
    Plan::load(&fixture.repo, 2).expect("the replacement must load");
}

#[test]
fn an_answer_that_is_not_json_is_refused_and_writes_nothing() {
    let fixture = Fixture::new();
    let model = std::sync::Arc::new(Script::answering("I would rather not."));
    let error = run_plan(
        &fixture.state(&model),
        &fixture.params(&serde_json::json!({"goal": "add a parser"})),
    )
    .expect_err("must refuse");
    assert_eq!(error.code, ErrorCode::Internal);
    assert!(error.message.contains("unusable"), "{}", error.message);
    assert!(
        !fixture.repo.join(".agent").exists(),
        "a refusal must write nothing"
    );
}

#[test]
fn an_answer_without_a_gate_is_refused_by_the_loader_s_own_rule() {
    // The planner does not have its own definition of "verifiable". `compile` runs
    // `plan::validate`, so a plan the planner accepts is one the scheduler will accept, and a
    // subtask with neither checks nor review is refused here for the same reason it would be
    // refused on load.
    let fixture = Fixture::new();
    let answer = serde_json::json!({
        "plan": "# Plan",
        "acceptance": "it works",
        "subtasks": [{
            "name": "nothing",
            "title": "do nothing verifiable",
            "goal": "unspecified",
            "dod": { "statement": "it works", "checks": [], "review": false },
        }],
    })
    .to_string();
    let model = std::sync::Arc::new(Script::answering(&answer));
    let error = run_plan(
        &fixture.state(&model),
        &fixture.params(&serde_json::json!({"goal": "add a parser"})),
    )
    .expect_err("must refuse");
    assert!(
        error.message.contains("no checks and no review"),
        "{}",
        error.message
    );
    assert!(!fixture.repo.join(".agent").exists());
}

#[test]
fn an_answer_with_no_acceptance_is_refused() {
    let fixture = Fixture::new();
    let answer = serde_json::json!({
        "plan": "# Plan",
        "acceptance": "   ",
        "subtasks": [{
            "name": "parse",
            "title": "t",
            "goal": "g",
            "dod": {
                "statement": "s",
                "checks": [{ "id": "test", "command": "cargo test" }],
                "review": false,
            },
        }],
    })
    .to_string();
    let model = std::sync::Arc::new(Script::answering(&answer));
    let error = run_plan(
        &fixture.state(&model),
        &fixture.params(&serde_json::json!({"goal": "add a parser"})),
    )
    .expect_err("must refuse");
    assert!(error.message.contains("acceptance"), "{}", error.message);
    assert!(!fixture.repo.join(".agent").exists());
}

#[test]
fn a_model_failure_is_reported_as_the_backend_s_problem() {
    // The goal and the repository were fine. Telling the human to check their arguments would
    // send them to the wrong place; the message has to name the backend failure.
    #[derive(Debug)]
    struct Failing;
    impl Model for Failing {
        fn complete(&self, _request: &ChatRequest) -> Result<Reply, String> {
            Err("connection refused".to_string())
        }
    }
    let fixture = Fixture::new();
    let state = ServerState::discover(&fixture.state_root)
        .expect("state")
        .with_model(std::sync::Arc::new(Failing));
    let error = run_plan(
        &state,
        &fixture.params(&serde_json::json!({"goal": "add a parser"})),
    )
    .expect_err("must refuse");
    assert_eq!(error.code, ErrorCode::Internal);
    assert!(error.message.contains("connection refused"), "{}", error.message);
    assert!(!fixture.repo.join(".agent").exists());
}

#[test]
fn planning_for_an_unregistered_directory_names_it() {
    let fixture = Fixture::new();
    let model = std::sync::Arc::new(Script::answering(&good_answer()));
    let error = run_plan(
        &fixture.state(&model),
        &serde_json::json!({
            "project": fixture.dir.path().join("elsewhere"),
            "goal": "add a parser",
        }),
    )
    .expect_err("must refuse");
    assert_eq!(error.code, ErrorCode::NotFound);
    assert!(error.message.contains("elsewhere"), "{}", error.message);
    assert_eq!(model.calls(), 0);
}

#[test]
fn planning_with_no_project_is_refused_rather_than_read_from_the_daemon_s_cwd() {
    let fixture = Fixture::new();
    let model = std::sync::Arc::new(Script::answering(&good_answer()));
    let error = run_plan(
        &fixture.state(&model),
        &serde_json::json!({"goal": "add a parser"}),
    )
    .expect_err("must refuse");
    assert_eq!(error.code, ErrorCode::BadParams);
    assert!(error.message.contains("working directory"), "{}", error.message);
    assert_eq!(model.calls(), 0);
}

#[test]
fn a_goal_for_a_task_already_registered_resolves_to_its_project() {
    // The same `project_for` as the read commands (D-088): one definition of "which project does
    // this request mean", so `plan --goal` and `plan` can never disagree about the answer.
    let fixture = Fixture::new();
    let task = TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV");
    let registry =
        Registry::open(fixture.state_root.join(immortality_core::registry::FILE_NAME)).expect("registry");
    let project = registry.projects()[0].clone();
    std::fs::create_dir_all(project.state_dir(&fixture.state_root).join(task.as_str())).expect("task dir");
    let model = std::sync::Arc::new(Script::answering(&good_answer()));
    let result = run_plan(
        &fixture.state(&model),
        &serde_json::json!({"task": task.as_str(), "goal": "add a parser"}),
    )
    .expect("the task names its project");
    assert_eq!(result["count"], 1);
}

// ── the tree that lands ──────────────────────────────────────────────────────

#[test]
fn the_written_tree_is_executable_not_only_loadable() {
    // "Loadable" and "executable" are different promises. This one takes the plan the handler
    // wrote and advances it with the scheduler's own `next_ready`, which is what the run will do.
    let fixture = Fixture::new();
    let answer = serde_json::json!({
        "plan": "# Plan",
        "acceptance": "`cargo test` passes.",
        "subtasks": [
            {
                "name": "parse",
                "title": "write the parser",
                "goal": "g",
                "dod": {
                    "statement": "s",
                    "checks": [{ "id": "test", "command": "cargo test" }],
                    "review": false,
                },
            },
            {
                "name": "tests",
                "title": "test the parser",
                "goal": "g",
                "depends_on": ["parse"],
                "dod": {
                    "statement": "s",
                    "checks": [{ "id": "test", "command": "cargo test" }],
                    "review": false,
                },
            },
        ],
    })
    .to_string();
    let model = std::sync::Arc::new(Script::answering(&answer));
    run_plan(
        &fixture.state(&model),
        &fixture.params(&serde_json::json!({"goal": "add a parser"})),
    )
    .expect("plan");

    let mut plan = Plan::load(&fixture.repo, 2).expect("load");
    let first = plan.next_ready().expect("a ready subtask").id.clone();
    plan.set_state(
        &first,
        immortality_core::domain::SubtaskState::Done,
        immortality_core::time::now(),
    )
    .expect("done");
    let second = plan.next_ready().expect("the dependent is now ready").id.clone();
    assert_ne!(second, first);
}

#[test]
fn the_listing_skips_noise_and_dotfiles_but_keeps_gitignore() {
    // The one dotfile worth showing: it says what the project deliberately does not track, which
    // is a fact about the repository a planner can use. The rest (`.git`, `.agent`) are noise or
    // the plan itself.
    let fixture = Fixture::new();
    std::fs::create_dir_all(fixture.repo.join("target")).expect("target");
    std::fs::write(fixture.repo.join(".gitignore"), "target/\n").expect("gitignore");
    std::fs::create_dir_all(fixture.repo.join(".git")).expect("git");

    let context = planner::context_of(&fixture.repo);
    assert!(context.contains("Cargo.toml"), "{context}");
    assert!(context.contains(".gitignore"), "{context}");
    assert!(!context.contains("target/"), "{context}");
    // `.git` as a whole line, not as a substring: `.gitignore` is the one dotfile kept, and a
    // `contains` would call it noise.
    assert!(
        !context.lines().any(|line| line.trim() == ".git"),
        "the git database is noise: {context}"
    );
}

#[test]
fn the_plan_request_carries_no_tool_definitions() {
    // Asserted on the request builder itself, because that is where the property lives: the
    // system message says "you have no tools", and only the absent field makes it true.
    let config = immortality_core::config::Config::default();
    let request = planner::request(&config, "a goal", "a listing");
    assert!(request.tools.is_none());
    assert_eq!(request.model, config.models.planner);
}
