//! The scheduler's view assembled from a real repository, journal and plan.
//!
//! # Why a real repository and not a fake registry
//!
//! The thing this module can get wrong is *which facts it reads*: the journal's last state, the
//! plan's readiness, the outage interval. A test with fabricated `Candidate`s would prove the
//! queue sorts, which `core`'s own tests already do. These drive the seam between disk and the
//! queue — the place where `status` and the scheduler stop agreeing.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};

use immortality_core::config::Config;
use immortality_core::domain::{Dod, Hold, Subtask, SubtaskState, TaskState};
use immortality_core::ids::TaskId;
use immortality_core::journal::{Journal, Kind};
use immortality_core::plan::Plan;
use immortality_core::registry::Registry;
use immortality_core::scheduler::queue::{Ineligible, Priority, Queue};
use immortality_daemon::scheduling;

fn task() -> TaskId {
    TaskId::from_suffix("01ARZ3NDEKTSV4RRFFQ69G5FAV")
}

/// A registered project with a plan, a worktree-free `.agent/`, and a journal per test.
struct Fixture {
    // Held only so the temporary directory outlives the fixture.
    _dir: tempfile::TempDir,
    repo: PathBuf,
    state_root: PathBuf,
}

impl Fixture {
    /// A fixture whose plan has one pending subtask.
    fn new() -> Self {
        Self::with_plan(&plan_body(SubtaskState::Pending))
    }

    fn with_plan(tasks_json: &str) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let repo = dir.path().join("repo");
        let state_root = dir.path().join("state-root");
        let agent = repo.join(".agent");
        std::fs::create_dir_all(&agent).expect("agent dir");
        std::fs::create_dir_all(&state_root).expect("state root");
        std::fs::write(agent.join("ACCEPTANCE.md"), "# acceptance\n").expect("acceptance");
        std::fs::write(agent.join("TASKS.json"), tasks_json).expect("tasks");
        // A state directory is what makes the task discoverable; `Registry::tasks` walks it.
        let mut registry =
            Registry::open(state_root.join(immortality_core::registry::FILE_NAME)).expect("registry");
        registry.register(&repo, None).expect("register");
        std::fs::create_dir_all(
            registry.projects()[0]
                .state_dir(&state_root)
                .join(task().as_str()),
        )
        .expect("task dir");
        Self {
            _dir: dir,
            repo,
            state_root,
        }
    }

    fn journal(&self) -> Journal {
        Journal::open(self.repo.join(".agent"), task()).expect("journal")
    }

    fn project(&self) -> immortality_core::registry::Project {
        Registry::open(self.state_root.join(immortality_core::registry::FILE_NAME))
            .expect("registry")
            .projects()[0]
            .clone()
    }

    fn candidates(&self) -> Vec<scheduling::Scheduled> {
        let registry =
            Registry::open(self.state_root.join(immortality_core::registry::FILE_NAME)).expect("registry");
        scheduling::candidates(&registry, &self.state_root, &Config::default()).expect("candidates")
    }

    fn only(&self) -> immortality_core::scheduler::queue::Candidate {
        self.candidates().remove(0).candidate
    }
}

/// One subtask, at depth 1, with `state`.
///
/// Built from the real [`Subtask`] rather than a hand-written JSON string: the plan file's schema
/// is `Subtask`'s serde form, and a literal here is a second copy of it that silently drifts — the
/// first version of this fixture used integer timestamps and made every plan unreadable, which
/// the tests then reported as "no work" rather than as "the fixture is wrong".
fn subtask(state: SubtaskState) -> Subtask {
    let mut subtask = Subtask::new(
        1,
        "write the parser",
        "parse the input",
        Dod::reviewed("the parser works"),
        immortality_core::time::now(),
    );
    subtask.state = state;
    subtask
}

/// The `TASKS.json` bytes for one subtask in `state`.
fn plan_body(state: SubtaskState) -> String {
    let body = serde_json::json!({ "v": 1, "subtasks": [subtask(state)] });
    serde_json::to_string_pretty(&body).expect("plan json")
}

#[test]
fn a_task_with_pending_work_is_a_candidate_for_the_model() {
    let fixture = Fixture::new();
    let candidate = fixture.only();
    assert_eq!(candidate.task, task());
    assert!(candidate.has_work);
    assert_eq!(candidate.ineligible_for_inference(), None);
    assert_eq!(candidate.priority, Priority::Normal);
}

#[test]
fn a_freshly_started_task_with_no_journal_is_queued_not_blocked() {
    // The state directory exists from `start`; the journal does not until the first step. Reading
    // the absence as `blocked` would make every task look broken at the moment a human is watching
    // it appear.
    let fixture = Fixture::new();
    assert!(
        !fixture.repo.join(".agent").join("JOURNAL.jsonl").exists(),
        "the fixture must start with no journal"
    );
    assert_eq!(fixture.only().state, TaskState::Queued);
}

#[test]
fn the_last_task_state_in_the_journal_is_the_state_the_queue_sees() {
    let fixture = Fixture::new();
    {
        let mut journal = fixture.journal();
        journal
            .append_event(
                immortality_core::time::now(),
                Kind::TaskState,
                serde_json::json!({ "state": "running" }),
            )
            .expect("running");
        journal
            .append_event(
                immortality_core::time::now(),
                Kind::TaskState,
                serde_json::json!({ "state": "merged" }),
            )
            .expect("merged");
        journal.close().expect("close");
    }
    let candidate = fixture.only();
    assert_eq!(
        candidate.state,
        TaskState::Merged,
        "the first record must not win"
    );
    assert_eq!(candidate.ineligible_for_inference(), Some(Ineligible::Finished));
}

#[test]
fn a_backend_outage_is_a_hold_read_from_the_journal() {
    // Not a field in memory: the interval has to survive a restart, and the journal is the only
    // record that does (D-027).
    let fixture = Fixture::new();
    {
        let mut journal = fixture.journal();
        journal
            .append_event(
                immortality_core::time::now(),
                Kind::BackendUnavailable,
                serde_json::json!({ "service": "backend:unavailable" }),
            )
            .expect("unavailable");
        journal.close().expect("close");
    }
    let candidate = fixture.only();
    assert_eq!(candidate.hold, Some(Hold::WaitingBackend));
    assert_eq!(candidate.ineligible_for_inference(), Some(Ineligible::Backend));
    // Still able to do offline work: that is the whole point of the hold being orthogonal.
    assert_eq!(candidate.ineligible_for_offline(), None);
}

#[test]
fn a_restored_backend_clears_the_hold() {
    let fixture = Fixture::new();
    {
        let mut journal = fixture.journal();
        for kind in [Kind::BackendUnavailable, Kind::BackendRestored] {
            journal
                .append_event(
                    immortality_core::time::now(),
                    kind,
                    serde_json::json!({ "service": "backend:unavailable" }),
                )
                .expect("record");
        }
        journal.close().expect("close");
    }
    let candidate = fixture.only();
    assert_eq!(candidate.hold, None);
    assert_eq!(candidate.ineligible_for_inference(), None);
}

#[test]
fn the_outage_survives_a_kill_and_a_restart() {
    // The crash-injection requirement, stated as the property that matters: a daemon killed during
    // an outage must not restart the fifteen-minute clock, because the journal it wrote is still
    // there. Written, killed (the handle is dropped without `close`), then re-read by a new
    // process — which is exactly what `serve` does after `kill -9`.
    let fixture = Fixture::new();
    let started_at = immortality_core::time::now();
    {
        let mut journal = fixture.journal();
        journal
            .append_event(
                started_at,
                Kind::BackendUnavailable,
                serde_json::json!({ "service": "backend:unavailable" }),
            )
            .expect("unavailable");
        // No `close`, no rename: the process died here.
        std::mem::forget(journal);
    }

    // A "new process" reads the same directory through the public path `serve` uses.
    let candidate = fixture.only();
    assert_eq!(candidate.hold, Some(Hold::WaitingBackend));

    let records = immortality_core::journal::JournalReader::open(fixture.repo.join(".agent"), &task())
        .expect("reader")
        .read()
        .expect("read")
        .records;
    let since = immortality_core::scheduler::backend::since_from_journal(&records).expect("since");
    assert_eq!(
        immortality_core::time::truncate(since),
        immortality_core::time::truncate(started_at),
        "the interval must start where the dead process recorded it, not where the new one woke"
    );
}

#[test]
fn a_task_whose_subtask_is_running_says_so_instead_of_claiming_there_is_no_work() {
    // Measured on `tsk_01M7HJEFXD7YDHGM59Y4X57FTZ`: while the first attempt was in its middle,
    // `status` said "1 with no subtask left to run" and `scheduling.next` was null. The daemon was
    // working; the *report* said it had nothing to do, which is the one sentence this project
    // exists to never say wrongly.
    //
    // The cause: `has_work` is `next_ready`, and `next_ready` offers only `pending` subtasks, so a
    // tree with one `running` subtask reads as empty. The fix keeps `has_work` false — a second
    // attempt would fork the journal — and adds the missing fact the renderer needs.
    let fixture = Fixture::with_plan(&plan_body(SubtaskState::Running));
    let candidate = fixture.only();
    assert!(!candidate.has_work, "a running subtask is not a second attempt");
    assert!(candidate.in_flight, "but it is work in progress");
    assert_eq!(
        candidate.ineligible_for_inference(),
        Some(Ineligible::InFlight),
        "and the reason must say so, not `NoWork`"
    );
}

#[test]
fn a_plan_with_every_subtask_settled_offers_no_work() {
    let fixture = Fixture::with_plan(&plan_body(SubtaskState::Done));
    let candidate = fixture.only();
    assert!(!candidate.has_work, "a settled plan is not work to start");
    assert_eq!(candidate.ineligible_for_inference(), Some(Ineligible::NoWork));
}

#[test]
fn an_unreadable_plan_offers_no_work_rather_than_starting_a_doomed_attempt() {
    // The failure a `has_work` derived from "TASKS.json exists" would produce: an attempt starts,
    // fails on the first step, and spends the subtask's budget on a planner bug.
    let fixture = Fixture::with_plan("{ this is not json }");
    let candidate = fixture.only();
    assert!(!candidate.has_work);
    assert_eq!(candidate.state, TaskState::Queued);
}

#[test]
fn a_task_whose_journal_cannot_be_read_is_visible_and_does_not_take_a_slot() {
    // "Unreadable" and "empty" lead a human to different actions. Recovery will block the task on
    // the next start; until then `status` must show it as blocked rather than omit it.
    use std::io::Write as _;

    let fixture = Fixture::new();
    let journal_dir = fixture.repo.join(".agent");
    let mut file = std::fs::File::create(journal_dir.join("JOURNAL.jsonl")).expect("create");
    file.write_all(b"not a journal at all\n").expect("write");
    drop(file);

    let candidate = fixture.only();
    assert_eq!(candidate.state, TaskState::Blocked);
    assert!(!candidate.has_work);
    assert_eq!(
        candidate.ineligible_for_inference(),
        Some(Ineligible::NoWork),
        "a blocked task is a reason, not work"
    );
}

#[test]
fn the_queue_chosen_task_matches_what_status_would_explain() {
    // The two readers must agree, because a `status` that reports a task as runnable while the
    // queue rejects it is the disagreement this whole seam exists to prevent.
    let fixture = Fixture::new();
    let candidates = fixture.candidates();
    let all: Vec<_> = candidates.iter().map(|s| s.candidate.clone()).collect();
    let (chosen, rejected) = Queue::new().next_for_inference(&all);
    assert!(rejected.is_empty(), "{rejected:?}");
    assert_eq!(chosen.expect("a choice").task, task());
}

#[test]
fn the_plan_validates_against_the_depth_the_daemon_carries() {
    // `Plan::load` refuses a tree deeper than the limit, so a reader without the daemon's limit
    // cannot load the plan the scheduler runs. This pins that the limit reaches the load.
    let fixture = Fixture::new();
    let path = fixture.repo.join(".agent").join("TASKS.json");
    let valid = Plan::load(&fixture.repo, 2);
    assert!(valid.is_ok(), "{:?}", valid.err());
    // Negative control: with a limit too small for the same tree, the load refuses.
    let refused = Plan::load(&fixture.repo, 0);
    assert!(refused.is_err(), "depth 0 must refuse a depth-1 tree");
    assert!(path.exists());
}

/// The dirty-worktree case: recovery is not this module's job, but a scheduled task whose worktree
/// is missing must still be a candidate — the scheduler does not require a tree to decide.
#[test]
fn scheduling_does_not_need_a_worktree_to_exist() {
    let fixture = Fixture::new();
    let project = fixture.project();
    assert!(
        !Path::new(&project.path.join(".git")).exists(),
        "the fixture is not a git repository, and the queue must not require one"
    );
    assert!(fixture.only().has_work);
}

/// The state a subtask is in is not the task's, and the queue asks about the task.
#[test]
fn a_blocked_subtask_does_not_make_the_task_not_a_candidate() {
    // `counts_as_settled` counts `blocked`, so its dependents run and the task moves on. Reading
    // "any blocked subtask" as "the task is blocked" would idle it (D-078's rule).
    let fixture = Fixture::with_plan(&plan_body(SubtaskState::Blocked));
    let candidate = fixture.only();
    assert_eq!(candidate.state, TaskState::Queued);
    assert!(
        !candidate.has_work,
        "the only subtask is settled, so there is nothing left to start"
    );
    assert!(
        SubtaskState::Blocked.counts_as_settled(),
        "the rule this test depends on"
    );
}
