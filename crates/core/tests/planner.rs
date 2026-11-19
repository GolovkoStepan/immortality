//! The planner's contract: what compiles, what is refused, and what a refusal leaves on disk.
//!
//! # Why the refusal cases carry most of this file
//!
//! A planner that produced a valid plan from a valid proposal would pass almost any test. What
//! makes it safe is that a *bad* proposal is refused with a reason a human can act on, and that a
//! refusal writes nothing: a half-written plan is a task that looks planned and cannot start.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use immortality_core::domain::{Dod, DodCheck, Subtask, SubtaskState, SubtaskTrigger};
use immortality_core::plan::{self, Plan, PlanError, SCHEMA_VERSION};
use immortality_core::planner::Files;
use immortality_core::planner::{self, Proposal, Proposed};

fn dod() -> Dod {
    Dod {
        statement: "the parser handles the documented cases".to_string(),
        checks: vec![DodCheck {
            id: "test".to_string(),
            command: "cargo test".to_string(),
            required: true,
            timeout_secs: None,
        }],
        review: false,
    }
}

fn proposed(name: &str, depends_on: &[&str]) -> Proposed {
    Proposed {
        name: name.to_string(),
        title: format!("do {name}"),
        goal: format!("achieve {name}"),
        dod: dod(),
        depends_on: depends_on.iter().map(|dep| (*dep).to_string()).collect(),
        children: Vec::new(),
    }
}

fn proposal(subtasks: Vec<Proposed>) -> Proposal {
    Proposal {
        plan: "# Plan\n\nDo the work.".to_string(),
        acceptance: "`cargo test` passes and `fn parse` exists.".to_string(),
        subtasks,
    }
}

fn now() -> immortality_core::domain::Timestamp {
    immortality_core::time::now()
}

// ── parsing the model's answer ───────────────────────────────────────────────

#[test]
fn a_json_object_inside_prose_is_read() {
    // Small local models wrap JSON in a sentence or a code fence. Refusing that would throw away
    // a usable plan; accepting prose as if it were JSON would be worse.
    let answer = "Sure, here is the plan:\n```json\n{\"subtasks\":[{\"name\":\"a\",\"title\":\"t\",\
                  \"goal\":\"g\",\"dod\":{\"statement\":\"s\",\"checks\":[{\"id\":\"t\",\"command\":\
                  \"cargo test\"}]}}]}\n```\n";
    let parsed = planner::parse(answer).expect("the object must be found");
    assert_eq!(parsed.subtasks.len(), 1);
    assert_eq!(parsed.subtasks[0].name, "a");
}

#[test]
fn an_answer_with_no_object_is_a_named_refusal() {
    let error = planner::parse("I cannot help with that.").expect_err("must refuse");
    assert!(matches!(error, PlanError::BadProposal { .. }), "{error}");
    assert!(error.to_string().contains("no JSON object"), "{error}");
}

#[test]
fn braces_inside_strings_do_not_end_the_object_early() {
    // A goal that contains `}` is ordinary (a JSON snippet in prose, a Rust match arm). A brace
    // count that ignored strings would cut the object there and report a parse failure in a
    // perfectly good answer.
    let answer = r#"{"subtasks":[{"name":"a","title":"fix the `}` case","goal":"handle } and {","dod":{"statement":"s","checks":[{"id":"t","command":"cargo test"}]}}]}"#;
    let parsed = planner::parse(answer).expect("must parse");
    assert_eq!(parsed.subtasks[0].title, "fix the `}` case");
}

// ── compiling ────────────────────────────────────────────────────────────────

#[test]
fn names_become_ids_and_dependencies_are_rewritten() {
    // The whole reason the model does not write ids: it writes `parse`, the compiler writes
    // `sub_01ARZ…`, and the dependency points at the right one.
    let compiled = planner::compile(
        "add a parser",
        &proposal(vec![proposed("parse", &[]), proposed("tests", &["parse"])]),
        2,
        now(),
    )
    .expect("must compile");
    assert_eq!(compiled.len(), 2);
    for subtask in &compiled {
        assert!(subtask.id.as_str().starts_with("sub_"));
        assert_eq!(subtask.state, SubtaskState::Pending);
        assert_eq!(subtask.depth, 1);
    }
    let tests = compiled.iter().find(|s| s.title == "do tests").unwrap();
    let parse = compiled.iter().find(|s| s.title == "do parse").unwrap();
    assert_eq!(tests.depends_on, vec![parse.id.clone()]);
}

#[test]
fn a_dependency_may_point_forward() {
    // `first` waits on `second` which is declared after it. That is a normal shape: the model
    // lists subtasks in reading order, not in execution order.
    let compiled = planner::compile(
        "g",
        &proposal(vec![proposed("first", &["second"]), proposed("second", &[])]),
        2,
        now(),
    )
    .expect("must compile");
    assert_eq!(compiled[0].depends_on.len(), 1);
    assert_eq!(compiled[0].depends_on[0], compiled[1].id);
}

#[test]
fn children_get_the_next_depth() {
    let mut parent = proposed("parent", &[]);
    parent.children = vec![proposed("child", &[])];
    let compiled = planner::compile("g", &proposal(vec![parent]), 2, now()).expect("compile");
    assert_eq!(compiled[0].depth, 1);
    assert_eq!(compiled[0].children[0].depth, 2);
}

#[test]
fn a_duplicate_name_is_refused_by_name() {
    // Names are the dependency key, so a duplicate makes every reference ambiguous. The message
    // names the duplicate rather than an id the model never wrote.
    let error = planner::compile(
        "g",
        &proposal(vec![proposed("same", &[]), proposed("same", &[])]),
        2,
        now(),
    )
    .expect_err("must refuse");
    assert!(matches!(error, PlanError::DuplicateName { .. }), "{error}");
    assert!(error.to_string().contains("same"), "{error}");
}

#[test]
fn a_dependency_on_a_name_that_does_not_exist_is_refused() {
    let error = planner::compile("g", &proposal(vec![proposed("a", &["nowhere"])]), 2, now())
        .expect_err("must refuse");
    assert!(matches!(error, PlanError::DanglingDependency { .. }), "{error}");
    assert!(error.to_string().contains("nowhere"), "{error}");
}

#[test]
fn a_subtask_with_no_gate_is_refused_by_the_same_rule_the_loader_uses() {
    // The planner does not have its own definition of "verifiable": it runs `plan::validate`, so
    // a plan the planner accepts and the scheduler refuses cannot exist.
    let mut gateless = proposed("a", &[]);
    gateless.dod = Dod {
        statement: "it works".to_string(),
        checks: Vec::new(),
        review: false,
    };
    let error = planner::compile("g", &proposal(vec![gateless]), 2, now()).expect_err("must refuse");
    assert!(matches!(error, PlanError::BadDod { .. }), "{error}");
    assert!(error.to_string().contains("no checks and no review"), "{error}");
}

#[test]
fn a_tree_deeper_than_the_limit_is_refused() {
    let mut parent = proposed("p", &[]);
    let mut child = proposed("c", &[]);
    child.children = vec![proposed("grandchild", &[])];
    parent.children = vec![child];
    let error = planner::compile("g", &proposal(vec![parent]), 2, now()).expect_err("must refuse");
    assert!(matches!(error, PlanError::TooDeep { .. }), "{error}");
}

#[test]
fn an_empty_goal_is_refused() {
    let error =
        planner::compile("   ", &proposal(vec![proposed("a", &[])]), 2, now()).expect_err("must refuse");
    assert!(matches!(error, PlanError::EmptyGoal), "{error}");
}

#[test]
fn a_plan_with_no_acceptance_is_refused() {
    // `Plan::load` requires `ACCEPTANCE.md`, so a plan without one is unloadable the moment it
    // lands. Refusing here means the failure is reported before anything is written.
    let mut body = proposal(vec![proposed("a", &[])]);
    body.acceptance = "  \n".to_string();
    let error = planner::compile("g", &body, 2, now()).expect_err("must refuse");
    assert!(matches!(error, PlanError::EmptyAcceptance), "{error}");
}

#[test]
fn an_empty_proposal_is_a_planner_failure() {
    let error = planner::compile("g", &proposal(Vec::new()), 2, now()).expect_err("must refuse");
    assert!(matches!(error, PlanError::NoSubtasks), "{error}");
}

#[test]
fn the_compiled_tree_passes_the_loader() {
    // The end of the contract: what `compile` returns is what `Plan::load` accepts.
    let compiled = planner::compile(
        "g",
        &proposal(vec![proposed("a", &[]), proposed("b", &["a"])]),
        2,
        now(),
    )
    .expect("compile");
    plan::validate(&compiled, 2).expect("the loader's own rule must accept it");
}

// ── writing ──────────────────────────────────────────────────────────────────

fn fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("repo");
    std::fs::create_dir_all(&root).expect("repo");
    (dir, root)
}

fn compiled() -> Vec<Subtask> {
    planner::compile(
        "add a parser",
        &proposal(vec![proposed("parse", &[]), proposed("tests", &["parse"])]),
        2,
        now(),
    )
    .expect("compile")
}

#[test]
fn writing_the_three_files_produces_a_loadable_plan() {
    let (_dir, root) = fixture();
    let body = proposal(vec![proposed("parse", &[])]);
    let subtasks = compiled();
    let files = planner::render(&body, &subtasks).expect("render");
    let plan = planner::write(&root, &files, 2, false).expect("write");

    assert_eq!(plan.subtasks().len(), 2);
    assert!(root.join(".agent").join("PLAN.md").exists());
    assert!(root.join(".agent").join("ACCEPTANCE.md").exists());
    let loaded = Plan::load(&root, 2).expect("the written plan must load");
    assert_eq!(loaded.subtasks().len(), 2);
}

#[test]
fn a_second_plan_refuses_to_overwrite_the_human_s_edit() {
    // The roadmap's contract: a human may rewrite these files before the run. Regenerating them
    // silently would discard exactly that work, and the human would find out when the task did
    // something they had removed.
    let (_dir, root) = fixture();
    let body = proposal(vec![proposed("parse", &[])]);
    let subtasks = compiled();
    let files = planner::render(&body, &subtasks).expect("render");
    planner::write(&root, &files, 2, false).expect("first write");

    let error = planner::write(&root, &files, 2, false).expect_err("must refuse");
    assert!(matches!(error, PlanError::AlreadyPlanned { .. }), "{error}");
    assert!(error.to_string().contains("--force"), "{error}");
    assert!(error.to_string().contains("TASKS.json"), "{error}");
}

#[test]
fn force_replaces_the_plan() {
    let (_dir, root) = fixture();
    let body = proposal(vec![proposed("parse", &[])]);
    let subtasks = compiled();
    let files = planner::render(&body, &subtasks).expect("render");
    planner::write(&root, &files, 2, false).expect("first write");

    // A human's edit, then a forced regeneration.
    let tasks = root.join(".agent").join("TASKS.json");
    std::fs::write(&tasks, "{ \"v\": 1, \"subtasks\": [] }").expect("human edit");
    planner::write(&root, &files, 2, true).expect("force must replace");
    let loaded = Plan::load(&root, 2).expect("loadable again");
    assert_eq!(loaded.subtasks().len(), 2);
}

#[test]
fn a_refused_write_leaves_the_directory_untouched() {
    // Nothing half-written. The first version of `write` created `.agent/` and then validated, so
    // a refusal left an empty directory a later `plan` would treat as progress.
    let (_dir, root) = fixture();
    let mut gateless = compiled();
    gateless[0].dod = Dod {
        statement: "nothing".to_string(),
        checks: Vec::new(),
        review: false,
    };
    // The rendered body carries the invalid tree, because `write` judges the bytes it would land
    // rather than a tree handed in beside them.
    let files = Files {
        plan: "plan\n".to_string(),
        acceptance: "acceptance\n".to_string(),
        tasks: serde_json::to_string_pretty(&serde_json::json!({
            "v": SCHEMA_VERSION,
            "subtasks": gateless,
        }))
        .expect("render")
            + "\n",
    };
    planner::write(&root, &files, 2, false).expect_err("must refuse");

    assert!(!root.join(".agent").exists(), "a refusal must not create .agent/");
}

#[test]
fn the_bytes_are_what_is_validated_not_a_tree_passed_beside_them() {
    // The reason `write` no longer takes the tree: two inputs that are supposed to agree can
    // disagree, and the bytes on disk are what the scheduler executes. A caller that rendered a
    // good tree and then produced invalid JSON is refused before anything lands.
    let (_dir, root) = fixture();
    let mut broken = compiled();
    let good = planner::render(&proposal(vec![proposed("a", &[])]), &broken).expect("render");
    broken[0].dod = Dod {
        statement: "nothing".to_string(),
        checks: Vec::new(),
        review: false,
    };
    let bad = Files {
        plan: good.plan.clone(),
        acceptance: good.acceptance,
        tasks: serde_json::to_string_pretty(&serde_json::json!({
            "v": SCHEMA_VERSION,
            "subtasks": broken,
        }))
        .expect("render")
            + "\n",
    };
    let error = planner::write(&root, &bad, 2, false).expect_err("the body is invalid");
    assert!(matches!(error, PlanError::BadDod { .. }), "{error}");
    assert!(!root.join(".agent").exists());
}

#[test]
fn the_rendered_files_end_with_a_newline() {
    // A file without a trailing newline shows up as `\ No newline at end of file` in every diff
    // the agent produces against it, and a human editing it loses that line every time.
    let body = proposal(vec![proposed("a", &[])]);
    let files = planner::render(&body, &compiled()).expect("render");
    for text in [&files.plan, &files.acceptance, &files.tasks] {
        assert!(text.ends_with('\n'), "{text:?}");
        assert!(!text.ends_with("\n\n"), "exactly one newline: {text:?}");
    }
}

#[test]
fn existing_reports_which_files_are_present() {
    let (_dir, root) = fixture();
    assert!(!planner::existing(&root).any());
    std::fs::create_dir_all(root.join(".agent")).expect("agent");
    std::fs::write(root.join(".agent").join("ACCEPTANCE.md"), "x").expect("write");
    let present = planner::existing(&root);
    assert!(present.acceptance);
    assert!(!present.plan);
    assert_eq!(present.names(), vec!["ACCEPTANCE.md"]);
}

// ── the prompt ───────────────────────────────────────────────────────────────

#[test]
fn the_prompt_carries_the_goal_and_the_repository_context() {
    let text = planner::prompt("add a parser", "Rust crate, `cargo test` is the suite");
    assert!(text.contains("add a parser"), "{text}");
    assert!(text.contains("cargo test"), "{text}");
}

#[test]
fn the_system_message_forbids_ids_and_demands_a_gate() {
    // Two properties the compiler depends on, asserted on the prompt so a reworded system message
    // cannot quietly drop them.
    assert!(planner::SYSTEM.contains("names"), "{}", planner::SYSTEM);
    assert!(planner::SYSTEM.contains("gate"), "{}", planner::SYSTEM);
    assert!(planner::SYSTEM.contains("have no tools") || planner::SYSTEM.contains("do not have tools"));
}

#[test]
fn the_system_message_teaches_the_per_check_deadline() {
    // The `black_jack` post-mortem: the planner was never told a check could declare more time, so
    // every plan it wrote inherited the 120 s default and a 6 m 39 s suite could only ever time out.
    // Telling the model about the field is the half of the fix that lives in the prompt; the other
    // half is `Dod::validate` refusing a deadline the runner would clamp.
    assert!(planner::SYSTEM.contains("timeout_secs"), "{}", planner::SYSTEM);
    assert!(planner::SYSTEM.contains("600"), "{}", planner::SYSTEM);
    // And it must still forbid the pipeline shortcut the timeout invites ("just pipe it to tail").
    assert!(planner::SYSTEM.contains("pipeline"), "{}", planner::SYSTEM);
}

#[test]
fn an_empty_context_does_not_leave_an_empty_section() {
    let text = planner::prompt("goal", "   ");
    assert!(!text.contains("# The repository"), "{text}");
}

// ── the tree the agent will actually execute ─────────────────────────────────

#[test]
fn a_compiled_plan_can_be_advanced_through_its_lifecycle() {
    // The plan is not only loadable, it is executable: the scheduler's own `next_ready` finds the
    // first subtask and a transition is accepted.
    let (_dir, root) = fixture();
    let body = proposal(vec![proposed("parse", &[]), proposed("tests", &["parse"])]);
    let subtasks = compiled();
    let files = planner::render(&body, &subtasks).expect("render");
    let mut plan = planner::write(&root, &files, 2, false).expect("write");

    let first = plan.next_ready().expect("a ready subtask").id.clone();
    plan.set_state(&first, SubtaskState::Done, now()).expect("done");
    let second = plan.next_ready().expect("the dependent is now ready").id.clone();
    assert_ne!(second, first);
    plan.set_state(&second, SubtaskState::Done, now()).expect("done");
    assert!(plan.all_settled(), "both settled");
}

#[test]
fn a_state_transition_on_a_planned_subtask_is_legal() {
    // `SubtaskTrigger` is the lifecycle's own vocabulary; this asserts the compiled state is the
    // one it expects to start from.
    let mut subtask = compiled().remove(0);
    subtask
        .transition(SubtaskTrigger::Start, now())
        .expect("pending must start");
    assert_eq!(subtask.state, SubtaskState::Running);
}
