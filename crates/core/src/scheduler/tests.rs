//! Tests for the split rules and for who runs next.

use super::*;
use crate::domain::{Dod, DodCheck, SubtaskId};

fn now() -> Timestamp {
    chrono::TimeZone::timestamp_opt(&chrono::Utc, 1_700_000_000, 0)
        .single()
        .expect("a fixed instant")
}

fn sub(id: &str, depth: u8) -> Subtask {
    Subtask {
        id: SubtaskId::from_suffix(id),
        depth,
        title: format!("title {id}"),
        goal: format!("goal {id}"),
        dod: Dod::reviewed("reviewed"),
        state: SubtaskState::Pending,
        depends_on: Vec::new(),
        current_attempt: None,
        attempts: Vec::new(),
        report: None,
        children: Vec::new(),
        created_at: now(),
        updated_at: now(),
    }
}

fn with_checks(checks: &[(&str, &str)], review: bool) -> Dod {
    Dod {
        statement: "the gate holds".to_string(),
        checks: checks
            .iter()
            .map(|(id, command)| DodCheck {
                id: (*id).to_string(),
                command: (*command).to_string(),
                required: true,
            })
            .collect(),
        review,
    }
}

#[test]
fn a_child_inherits_the_parents_definition_of_done() {
    let mut parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    parent.dod = with_checks(&[("test", "cargo test")], true);
    let child = inherit(&parent, "one half", "do the first half", now());
    assert_eq!(child.depth, 2);
    assert_eq!(child.dod, parent.dod);
    assert_eq!(child.state, SubtaskState::Pending);
    // The inherited children pass the split rules for the obvious case.
    let other = inherit(&parent, "the other half", "do the other half", now());
    validate_split(&parent, &[child, other], 2).expect("a plain split");
}

#[test]
fn a_plain_reviewed_split_is_legal() {
    let parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let children = vec![
        sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 2),
        sub("01ARZ3NDEKTSV4RRFFQ69G5FAC", 2),
    ];
    validate_split(&parent, &children, 2).expect("two reviewed children");
}

#[test]
fn splitting_at_the_depth_limit_is_refused() {
    // `max_subtask_depth = 2`: a depth-2 parent has nowhere to put children.
    let parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 2);
    let child = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 3);
    let other = sub("01ARZ3NDEKTSV4RRFFQ69G5FAC", 3);
    let error = validate_split(&parent, &[child, other], 2).expect_err("too deep");
    assert!(
        matches!(
            error,
            SplitError::TooDeep {
                needed: 3,
                max: 2,
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn a_split_with_no_children_is_refused() {
    let parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let error = validate_split(&parent, &[], 2).expect_err("empty");
    assert!(matches!(error, SplitError::Empty), "{error}");
}

#[test]
fn a_split_of_a_finished_subtask_is_refused() {
    // A done subtask's report already told the next attempt what happened; splitting it after
    // the fact would retire a subtitle whose work is on disk.
    let mut parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    parent.state = SubtaskState::Done;
    let child = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 2);
    let other = sub("01ARZ3NDEKTSV4RRFFQ69G5FAC", 2);
    let error = validate_split(&parent, &[child, other], 2).expect_err("terminal");
    assert!(
        matches!(
            error,
            SplitError::Terminal {
                state: SubtaskState::Done,
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn a_child_at_the_wrong_depth_is_refused() {
    // The depth is declared and therefore can lie; children of a depth-1 parent are depth 2.
    let parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let child = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 1);
    let other = sub("01ARZ3NDEKTSV4RRFFQ69G5FAC", 2);
    let error = validate_split(&parent, &[child, other], 2).expect_err("wrong depth");
    assert!(
        matches!(error, SplitError::WrongDepth { expected: 2, .. }),
        "{error}"
    );
}

#[test]
fn a_child_that_is_already_running_is_refused() {
    let parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let mut child = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 2);
    child.state = SubtaskState::Running;
    let other = sub("01ARZ3NDEKTSV4RRFFQ69G5FAC", 2);
    let error = validate_split(&parent, &[child, other], 2).expect_err("running child");
    assert!(matches!(error, SplitError::NotPending { .. }), "{error}");
}

#[test]
fn a_child_with_an_ungated_dod_is_refused() {
    let parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let mut child = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 2);
    child.dod = Dod {
        statement: "nice".to_string(),
        checks: Vec::new(),
        review: false,
    };
    let other = sub("01ARZ3NDEKTSV4RRFFQ69G5FAC", 2);
    let error = validate_split(&parent, &[child, other], 2).expect_err("no gate");
    assert!(matches!(error, SplitError::BadDod { .. }), "{error}");
}

#[test]
fn a_child_depending_on_its_parent_is_refused() {
    // Children run before the parent, so this waits forever: the parent will not start until
    // its children settle, and the child will not start until the parent settles.
    let parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let mut child = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 2);
    child.depends_on = vec![parent.id.clone()];
    let other = sub("01ARZ3NDEKTSV4RRFFQ69G5FAC", 2);
    let error = validate_split(&parent, &[child, other], 2).expect_err("deadlock");
    assert!(
        matches!(error, SplitError::ChildDependsOnParent { .. }),
        "{error}"
    );
}

#[test]
fn a_split_that_drops_a_required_check_is_refused() {
    // The failure this milestone exists to prevent: the parent needed `cargo test`, the piece
    // that was supposed to cover it does not run it, and the task goes green anyway.
    let mut parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    parent.dod = with_checks(&[("build", "cargo build"), ("test", "cargo test")], true);
    let mut child = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 2);
    child.dod = with_checks(&[("build", "cargo build")], true);
    let other = sub("01ARZ3NDEKTSV4RRFFQ69G5FAC", 2);
    let error = validate_split(&parent, &[child, other], 2).expect_err("dropped check");
    assert!(
        matches!(error, SplitError::MissingCheck { ref check, .. } if check == "test"),
        "{error}"
    );
}

#[test]
fn the_checks_may_be_distributed_across_children() {
    // A split is a decomposition, not a copy: `build` may live in one child and `test` in
    // another, as long as both run somewhere.
    let mut parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    parent.dod = with_checks(&[("build", "cargo build"), ("test", "cargo test")], false);
    let mut first = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 2);
    first.dod = with_checks(&[("build", "cargo build")], false);
    let mut second = sub("01ARZ3NDEKTSV4RRFFQ69G5FAC", 2);
    second.dod = with_checks(&[("test", "cargo test")], false);
    validate_split(&parent, &[first, second], 2).expect("distributed checks");
}

#[test]
fn a_check_with_the_same_id_but_a_different_command_does_not_count_as_covered() {
    // `id` is a label; the command is the guarantee. A child that renamed `test` to something
    // cheaper has not kept the parent's gate.
    let mut parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    parent.dod = with_checks(&[("test", "cargo test")], false);
    let mut child = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 2);
    child.dod = with_checks(&[("test", "cargo test --lib")], false);
    let other = sub("01ARZ3NDEKTSV4RRFFQ69G5FAC", 2);
    let error = validate_split(&parent, &[child, other], 2).expect_err("weaker command");
    assert!(matches!(error, SplitError::MissingCheck { .. }), "{error}");
}

#[test]
fn dropping_the_reviewer_is_refused_when_the_parent_needed_one() {
    // The parent's guarantee was "a human-quality read of the diff". No child reviewer means
    // half of that diff was never read.
    let parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let mut child = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 2);
    child.dod = Dod::reviewed("same statement");
    child.dod.review = false;
    child.dod.checks = vec![DodCheck {
        id: "build".to_string(),
        command: "cargo build".to_string(),
        required: true,
    }];
    let other = sub("01ARZ3NDEKTSV4RRFFQ69G5FAC", 2);
    let error = validate_split(&parent, &[child, other], 2).expect_err("review dropped");
    assert!(matches!(error, SplitError::ReviewDropped { .. }), "{error}");
}

#[test]
fn a_parent_that_needed_no_reviewer_may_split_into_unreviewed_children() {
    // Mechanical work: the checks already prove the outcome, so a reviewer would burn tokens.
    let mut parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    parent.dod = with_checks(&[("build", "cargo build")], false);
    let mut child = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 2);
    child.dod = with_checks(&[("build", "cargo build")], false);
    let other = sub("01ARZ3NDEKTSV4RRFFQ69G5FAC", 2);
    validate_split(&parent, &[child, other], 2).expect("mechanical split");
}

#[test]
fn splitting_a_blocked_subtask_is_legal_because_that_is_what_l4_does() {
    let mut parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    parent.state = SubtaskState::Blocked;
    let child = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 2);
    let other = sub("01ARZ3NDEKTSV4RRFFQ69G5FAC", 2);
    validate_split(&parent, &[child, other], 2).expect("an exhausted subtask may be split");
}

// ── the tree, through `Plan` ───────────────────────────────────────────────

fn fixture(subtasks: &[Subtask]) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("repo");
    let agent = root.join(".agent");
    std::fs::create_dir_all(&agent).expect("mkdir");
    std::fs::write(agent.join("ACCEPTANCE.md"), "# acceptance\n").expect("write");
    let body = serde_json::json!({"v": crate::plan::SCHEMA_VERSION, "subtasks": subtasks});
    std::fs::write(
        agent.join("TASKS.json"),
        serde_json::to_string_pretty(&body).expect("json"),
    )
    .expect("write");
    (dir, root)
}

#[test]
fn a_plan_with_pending_work_is_not_settled() {
    let (dir, root) = fixture(&[sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1)]);
    let plan = Plan::load(&root, 2).expect("load");
    assert!(!plan.all_settled());
    assert_eq!(plan.open_subtasks().len(), 1);
    drop(dir);
}

#[test]
fn a_plan_whose_only_subtask_is_blocked_is_settled_and_the_task_moves_on() {
    // The design's rule: one stuck subtask must not idle the task for a day. `Blocked` counts
    // as settled, so the scheduler is not left waiting for work that will never come.
    let mut blocked = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    blocked.state = SubtaskState::Blocked;
    let (dir, root) = fixture(&[blocked]);
    let plan = Plan::load(&root, 2).expect("load");
    assert!(plan.all_settled());
    assert!(matches!(next(&plan), Next::Finished));
    drop(dir);
}

#[test]
fn next_runs_the_first_dependency_free_subtask() {
    let first = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let mut second = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 1);
    second.depends_on = vec![first.id.clone()];
    let (dir, root) = fixture(&[first, second]);
    let plan = Plan::load(&root, 2).expect("load");
    match next(&plan) {
        Next::Run(subtask) => assert_eq!(subtask.id, SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAA")),
        other => panic!("expected a runnable subtask, got {other:?}"),
    }
    drop(dir);
}

#[test]
fn next_reports_stuck_rather_than_idling_when_a_dependency_is_unmet() {
    // A plan that passed validation cannot have a dangling dependency, but a subtask still
    // `Running` after a crash can hold its dependent: nothing is ready, nothing is settled, and
    // the daemon must say so instead of sleeping on it.
    let mut running = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    running.state = SubtaskState::Running;
    let mut waiting = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 1);
    waiting.depends_on = vec![running.id.clone()];
    let (dir, root) = fixture(&[running, waiting]);
    let plan = Plan::load(&root, 2).expect("load");
    match next(&plan) {
        Next::Stuck { open } => assert_eq!(open.len(), 2),
        other => panic!("expected stuck, got {other:?}"),
    }
    drop(dir);
}

#[test]
fn a_blocked_dependency_unblocks_its_dependent() {
    let mut first = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    first.state = SubtaskState::Blocked;
    let mut second = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 1);
    second.depends_on = vec![first.id.clone()];
    let (dir, root) = fixture(&[first, second]);
    let plan = Plan::load(&root, 2).expect("load");
    match next(&plan) {
        Next::Run(subtask) => assert_eq!(subtask.id, SubtaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAB")),
        other => panic!("expected the dependent to run, got {other:?}"),
    }
    drop(dir);
}

#[test]
fn a_split_retires_the_parent_and_makes_its_children_ready() {
    let parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let mut blocked = parent.clone();
    blocked.state = SubtaskState::Blocked;
    let (dir, root) = fixture(&[blocked]);
    let mut plan = Plan::load(&root, 2).expect("load");

    let children = vec![
        inherit(&parent, "first half", "do the first half", now()),
        inherit(&parent, "second half", "do the second half", now()),
    ];
    plan.split(&parent.id, children, 2, now()).expect("split");

    let reloaded = Plan::load(&root, 2).expect("reload");
    let stored = reloaded.subtask(&parent.id).expect("parent still recorded");
    assert_eq!(stored.state, SubtaskState::Skipped, "the parent is retired");
    assert_eq!(stored.children.len(), 2, "the tree records the decomposition");
    match next(&reloaded) {
        Next::Run(child) => assert_eq!(child.depth, 2),
        other => panic!("expected a child to run, got {other:?}"),
    }
    drop(dir);
}

#[test]
fn a_refused_split_leaves_the_file_untouched() {
    // The validation runs before anything is mutated, so a model that proposes to drop a gate
    // cannot leave the plan half-split on disk.
    let mut parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    parent.dod = with_checks(&[("build", "cargo build")], false);
    let (dir, root) = fixture(std::slice::from_ref(&parent));
    let path = root.join(".agent").join("TASKS.json");
    let before = std::fs::read_to_string(&path).expect("read");
    let mut plan = Plan::load(&root, 2).expect("load");

    let mut weak = inherit(&parent, "half", "do half", now());
    weak.dod = with_checks(&[("test", "cargo test")], false);
    let mut other = inherit(&parent, "other half", "do the other half", now());
    other.dod = with_checks(&[("lint", "cargo clippy")], false);
    let error = plan
        .split(&parent.id, vec![weak, other], 2, now())
        .expect_err("dropped gate");
    assert!(matches!(error, SplitError::MissingCheck { .. }), "{error}");
    assert_eq!(std::fs::read_to_string(&path).expect("read"), before);
    assert_eq!(
        plan.open_subtasks().len(),
        1,
        "the tree is unchanged in memory too"
    );
    drop(dir);
}

#[test]
fn a_split_of_an_unknown_subtask_is_refused() {
    let (dir, root) = fixture(&[sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1)]);
    let mut plan = Plan::load(&root, 2).expect("load");
    let parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 1);
    let children = vec![
        inherit(&parent, "half", "do half", now()),
        inherit(&parent, "other half", "do the other half", now()),
    ];
    let error = plan
        .split(&parent.id, children, 2, now())
        .expect_err("unknown parent");
    assert!(
        matches!(error, SplitError::Plan(PlanError::UnknownSubtask { .. })),
        "{error}"
    );
    drop(dir);
}

#[test]
fn a_decomposed_tree_still_passes_load_validation() {
    // The children land in `TASKS.json` and are read back by the next scheduler tick; a split
    // that produced a tree the loader refuses would be a plan that cannot be reloaded.
    let parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let (dir, root) = fixture(std::slice::from_ref(&parent));
    let mut plan = Plan::load(&root, 2).expect("load");
    let children = vec![
        inherit(&parent, "one", "do one", now()),
        inherit(&parent, "two", "do two", now()),
    ];
    plan.split(&parent.id, children, 2, now()).expect("split");
    let reloaded = Plan::load(&root, 2).expect("the decomposed tree loads");
    assert_eq!(reloaded.open_subtasks().len(), 2);
    drop(dir);
}

// ── L4/L5: what an exhausted subtask turns into ─────────────────────────────

#[test]
fn an_exhausted_shallow_subtask_is_split() {
    // L4 keeps the work in the task. Splitting is tried before blocking, always.
    let subtask = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    assert_eq!(on_exhausted(&subtask, 2, true), Exhausted::Split);
    assert_eq!(on_exhausted(&subtask, 2, false), Exhausted::Split);
}

#[test]
fn an_exhausted_subtask_at_the_depth_limit_blocks_when_another_can_run() {
    // `validate_split` would refuse a depth-3 child, so proposing the split would be a decision
    // the mechanism has already vetoed.
    let subtask = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 2);
    assert_eq!(on_exhausted(&subtask, 2, true), Exhausted::Block);
}

#[test]
fn an_exhausted_subtask_with_nothing_else_to_run_escalates() {
    // Blocking here would leave the task with no independent work: the design's `blocked`
    // "с причиной", which is where a human is asked rather than the daemon idling.
    let subtask = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 2);
    assert_eq!(on_exhausted(&subtask, 2, false), Exhausted::Escalate);
}

#[test]
fn a_sibling_counts_as_independent_work() {
    let exhausted = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let sibling = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 1);
    let (dir, root) = fixture(&[exhausted.clone(), sibling]);
    let plan = Plan::load(&root, 2).expect("load");
    assert!(has_independent(&plan, &exhausted.id));
    drop(dir);
}

#[test]
fn the_exhausted_subtask_does_not_count_as_its_own_independent_work() {
    // Counting it would make `has_independent` always true and L5 would never escalate.
    let only = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let (dir, root) = fixture(std::slice::from_ref(&only));
    let plan = Plan::load(&root, 2).expect("load");
    assert!(!has_independent(&plan, &only.id));
    drop(dir);
}

#[test]
fn a_settled_sibling_is_not_independent_work() {
    let exhausted = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let mut done = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 1);
    done.state = SubtaskState::Done;
    let (dir, root) = fixture(&[exhausted.clone(), done]);
    let plan = Plan::load(&root, 2).expect("load");
    assert!(!has_independent(&plan, &exhausted.id));
    drop(dir);
}

#[test]
fn a_split_into_one_piece_is_refused_as_not_a_decomposition() {
    // The loophole this closes: a single child is the same work under a new id but with a fresh
    // attempts budget, so a weak model could "split" forever to dodge `max_attempts_per_subtask`.
    let parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let only = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 2);
    let error = validate_split(&parent, &[only], 2).expect_err("one piece");
    assert!(matches!(error, SplitError::NotADecomposition { .. }), "{error}");
}

#[test]
fn a_sibling_blocked_by_its_own_dependency_is_not_independent_work() {
    // L5 must escalate here: nothing can run now even though a `Pending` subtask exists. Asking
    // "is there any pending subtask" would answer yes and the task would idle instead of asking
    // for a human.
    let exhausted = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let mut waiting = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 1);
    let mut unfinished = sub("01ARZ3NDEKTSV4RRFFQ69G5FAC", 1);
    unfinished.state = SubtaskState::Running;
    waiting.depends_on = vec![unfinished.id.clone()];
    let (dir, root) = fixture(&[exhausted.clone(), waiting, unfinished]);
    let plan = Plan::load(&root, 2).expect("load");
    assert!(!has_independent(&plan, &exhausted.id));
    drop(dir);
}

#[test]
fn a_sibling_freed_by_retiring_the_exhausted_one_counts_as_independent() {
    // The other half of the same rule: a sibling that *depended on* the exhausted subtask is
    // available once it is blocked, and blocking is then better than escalating.
    let exhausted = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let mut after = sub("01ARZ3NDEKTSV4RRFFQ69G5FAB", 1);
    after.depends_on = vec![exhausted.id.clone()];
    let (dir, root) = fixture(&[exhausted.clone(), after]);
    let plan = Plan::load(&root, 2).expect("load");
    assert!(has_independent(&plan, &exhausted.id));
    drop(dir);
}

#[test]
fn apply_split_retires_the_parent_and_journals_it() {
    let parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    let (dir, root) = fixture(std::slice::from_ref(&parent));
    let mut plan = Plan::load(&root, 2).expect("load");
    let task = crate::ids::TaskId::generate();
    let mut journal =
        crate::journal::Journal::open(dir.path().join("journal"), task.clone()).expect("journal");
    let reader = crate::journal::JournalReader::open(dir.path().join("journal"), &task).expect("reader");

    let pieces = vec![
        Piece::new("parse", "parse the header"),
        Piece::new("emit", "emit the body"),
    ];
    let ids = apply_split(&mut plan, &mut journal, &parent.id, &pieces, 2, now()).expect("split");
    assert_eq!(ids.len(), 2);

    let records = reader.read().expect("read").records;
    let split = records
        .iter()
        .find(|record| record.fields.kind == Kind::SubtaskSplit)
        .expect("a split record");
    assert_eq!(
        split.fields.subtask.as_ref(),
        Some(&parent.id),
        "the record is about the parent being retired"
    );
    assert_eq!(split.event.payload["parent"], parent.id.to_string());
    assert_eq!(split.event.payload["children"].as_array().map(Vec::len), Some(2));
    drop(dir);
}

#[test]
fn a_refused_apply_split_writes_no_journal_record() {
    // The tree is validated before anything happens, so a refused split must leave the journal
    // without a `subtask_split` line: a record of a decomposition that never occurred would send
    // a reader looking for children that do not exist.
    let mut parent = sub("01ARZ3NDEKTSV4RRFFQ69G5FAA", 1);
    parent.dod = with_checks(&[("build", "cargo build")], false);
    let (dir, root) = fixture(std::slice::from_ref(&parent));
    let mut plan = Plan::load(&root, 2).expect("load");
    let task = crate::ids::TaskId::generate();
    let mut journal =
        crate::journal::Journal::open(dir.path().join("journal"), task.clone()).expect("journal");
    let reader = crate::journal::JournalReader::open(dir.path().join("journal"), &task).expect("reader");

    let pieces = vec![Piece::new("one", "do one"), Piece::new("two", "do two")];
    // Deep-parent variant: the same tree, but the limit refuses.
    let error = apply_split(&mut plan, &mut journal, &parent.id, &pieces, 1, now())
        .expect_err("depth 2 child under a limit of 1");
    assert!(matches!(error, SplitError::TooDeep { .. }), "{error}");
    let records = reader.read().expect("read").records;
    assert!(
        records
            .iter()
            .all(|record| record.fields.kind != Kind::SubtaskSplit)
    );
    assert_eq!(plan.open_subtasks().len(), 1, "the tree is untouched");
    drop(dir);
}
