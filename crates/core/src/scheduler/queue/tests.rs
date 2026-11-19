//! Who runs next, and what "next" means when the model is unreachable.

use super::*;
use crate::domain::{Hold, TaskId, TaskState};
use chrono::TimeZone as _;

fn at(secs: i64) -> Timestamp {
    chrono::Utc
        .timestamp_opt(1_700_000_000 + secs, 0)
        .single()
        .expect("a fixed instant")
}

fn id(suffix: &str) -> TaskId {
    TaskId::from_suffix(suffix)
}

fn candidate(suffix: &str, created: i64) -> Candidate {
    Candidate {
        task: id(suffix),
        priority: Priority::Normal,
        state: TaskState::Running,
        hold: None,
        created_at: at(created),
        has_work: true,
        in_flight: false,
    }
}

#[test]
fn the_older_task_goes_first() {
    let queue = Queue::new();
    let tasks = [
        candidate("01ARZ3NDEKTSV4RRFFQ69G5FAB", 10),
        candidate("01ARZ3NDEKTSV4RRFFQ69G5FAA", 5),
    ];
    let (chosen, rejected) = queue.next_for_inference(&tasks);
    assert_eq!(rejected, Vec::new());
    assert_eq!(chosen.expect("a choice").task, id("01ARZ3NDEKTSV4RRFFQ69G5FAA"));
}

#[test]
fn high_priority_wins_over_age() {
    // A task the human just asked to start must not queue behind a day-old one.
    let queue = Queue::new();
    let mut urgent = candidate("01ARZ3NDEKTSV4RRFFQ69G5FAB", 100);
    urgent.priority = Priority::High;
    let tasks = [candidate("01ARZ3NDEKTSV4RRFFQ69G5FAA", 0), urgent];
    let (chosen, _) = queue.next_for_inference(&tasks);
    assert_eq!(chosen.expect("a choice").task, id("01ARZ3NDEKTSV4RRFFQ69G5FAB"));
}

#[test]
fn equal_tasks_alternate_so_one_cannot_starve_the_other() {
    // The property a rotation buys: two ready tasks are served in turn. Without it a long task
    // holds the slot for hours and the second one waits for the first to finish.
    let queue = Queue::new();
    let tasks = [
        candidate("01ARZ3NDEKTSV4RRFFQ69G5FAA", 0),
        candidate("01ARZ3NDEKTSV4RRFFQ69G5FAB", 5),
    ];

    let (first, _) = queue.next_for_inference(&tasks);
    let first = first.expect("a choice").task.clone();
    let mut queue = queue;
    queue.note_served(first.clone());

    let (second, _) = queue.next_for_inference(&tasks);
    let second = second.expect("a choice").task.clone();
    assert_ne!(second, first, "the same task was chosen twice in a row");

    // And back again: the rotation is not "everyone but the last one forever".
    queue.note_served(second);
    let (third, _) = queue.next_for_inference(&tasks);
    assert_eq!(third.expect("a choice").task, first);
}

#[test]
fn rotation_never_overrides_priority() {
    // The rotation is applied after sorting, so it can only break a tie. A human-started task must
    // stay first even if it is the one that was just served.
    let queue = Queue {
        last_served: Some(id("01ARZ3NDEKTSV4RRFFQ69G5FAB")),
    };
    let mut urgent = candidate("01ARZ3NDEKTSV4RRFFQ69G5FAB", 0);
    urgent.priority = Priority::High;
    let tasks = [urgent, candidate("01ARZ3NDEKTSV4RRFFQ69G5FAA", 0)];
    let (chosen, _) = queue.next_for_inference(&tasks);
    assert_eq!(chosen.expect("a choice").task, id("01ARZ3NDEKTSV4RRFFQ69G5FAB"));
}

#[test]
fn the_choice_does_not_depend_on_the_input_order() {
    // An untotal order makes the scheduler's answer and `status`'s differ depending on iteration
    // order, which is the kind of bug that shows up as "it ran the wrong task once".
    let queue = Queue::new();
    let one = candidate("01ARZ3NDEKTSV4RRFFQ69G5FAA", 0);
    let two = candidate("01ARZ3NDEKTSV4RRFFQ69G5FAB", 0);
    let forward = [one.clone(), two.clone()];
    let backward = [two, one];
    let (a, _) = queue.next_for_inference(&forward);
    let (b, _) = queue.next_for_inference(&backward);
    assert_eq!(
        a.expect("a choice").task,
        b.expect("a choice").task,
        "identical candidates with identical ages must resolve to one task regardless of order"
    );
}

#[test]
fn a_backend_hold_blocks_inference_but_not_offline_work() {
    // The reason `Hold` is orthogonal to the lifecycle: an outage must not stop tests or commits,
    // and a task that can only commit must not hold the slot.
    let queue = Queue::new();
    let mut waiting = candidate("01ARZ3NDEKTSV4RRFFQ69G5FAA", 0);
    waiting.hold = Some(Hold::WaitingBackend);

    let (chosen, rejected) = queue.next_for_inference(std::slice::from_ref(&waiting));
    assert!(chosen.is_none());
    let waiting_id = waiting.task.clone();
    assert_eq!(rejected, vec![(&waiting_id, Ineligible::Backend)]);

    let (offline, rejected) = queue.next_for_offline(std::slice::from_ref(&waiting));
    assert!(offline.is_some(), "offline work must continue during an outage");
    assert_eq!(rejected, Vec::new());
}

#[test]
fn a_disk_hold_stops_everything() {
    // The one hold that is not orthogonal: writing to a full disk corrupts, so neither kind of
    // work may start.
    let queue = Queue::new();
    let mut paused = candidate("01ARZ3NDEKTSV4RRFFQ69G5FAA", 0);
    paused.hold = Some(Hold::PausedDisk);
    assert_eq!(queue.next_for_inference(std::slice::from_ref(&paused)).0, None);
    assert_eq!(queue.next_for_offline(std::slice::from_ref(&paused)).0, None);
    let paused_id = paused.task.clone();
    assert_eq!(
        queue.next_for_offline(std::slice::from_ref(&paused)).1,
        vec![(&paused_id, Ineligible::Disk)]
    );
}

#[test]
fn a_settled_task_does_not_take_the_slot() {
    // "Ready to think" and "waiting for a gate run" are different, and treating the second as the
    // first is how a daemon spins: it starts an attempt, the scheduler says nothing to do, repeat.
    let queue = Queue::new();
    let mut settled = candidate("01ARZ3NDEKTSV4RRFFQ69G5FAA", 0);
    settled.has_work = false;
    let (chosen, rejected) = queue.next_for_inference(std::slice::from_ref(&settled));
    assert!(chosen.is_none());
    assert_eq!(
        rejected,
        vec![(&id("01ARZ3NDEKTSV4RRFFQ69G5FAA"), Ineligible::NoWork)]
    );
}

#[test]
fn a_finished_task_is_never_scheduled() {
    for state in [TaskState::Merged, TaskState::Cancelled] {
        let queue = Queue::new();
        let mut done = candidate("01ARZ3NDEKTSV4RRFFQ69G5FAA", 0);
        done.state = state;
        assert!(queue.next_for_inference(std::slice::from_ref(&done)).0.is_none());
        assert!(queue.next_for_offline(std::slice::from_ref(&done)).0.is_none());
    }
}

#[test]
fn the_idle_reason_names_what_a_human_can_fix() {
    // An idle daemon and an empty `status` are the failure this daemon exists to avoid; the answer
    // has to say which of the four reasons applies, because the fixes differ.
    assert_eq!(summarise(&[]), "no tasks are registered");
    let a = id("01ARZ3NDEKTSV4RRFFQ69G5FAA");
    let b = id("01ARZ3NDEKTSV4RRFFQ69G5FAB");
    let c = id("01ARZ3NDEKTSV4RRFFQ69G5FAC");
    let rejected = vec![
        (&a, Ineligible::Backend),
        (&b, Ineligible::Backend),
        (&c, Ineligible::NoWork),
    ];
    let text = summarise(&rejected);
    assert!(text.contains("2 waiting for the model backend"), "{text}");
    assert!(text.contains("1 with no subtask left to run"), "{text}");
    assert!(!text.contains("finished"), "{text}");
}

#[test]
fn a_task_that_failed_its_gates_is_waiting_for_a_human_not_for_work() {
    // Measured on the `black_jack` task that ended `failed_gates` after its `rspec_pass` gate timed
    // out: `status` said "1 with no subtask left to run". True, and it sends a person to the plan
    // to look for a pending subtask that does not exist. The task was waiting for one action —
    // somebody reading the timeout and reopening a subtask.
    let queue = Queue::new();
    let mut failed = candidate("01ARZ3NDEKTSV4RRFFQ69G5FAA", 0);
    failed.state = TaskState::FailedGates;
    failed.has_work = false;
    let (chosen, rejected) = queue.next_for_inference(std::slice::from_ref(&failed));
    assert!(chosen.is_none());
    assert_eq!(
        rejected,
        vec![(&id("01ARZ3NDEKTSV4RRFFQ69G5FAA"), Ineligible::FailedGates)]
    );
    assert_eq!(summarise(&rejected), "1 waiting for a human after failed gates");
}

#[test]
fn a_red_verdict_is_not_overturned_by_a_tree_that_still_has_work() {
    // The state, not `has_work`, is the authority once a verdict exists. A `failed_gates` tree can
    // very well contain a `pending` subtask — the gate that failed belongs to the task, not to the
    // subtask that was open — and scheduling it would start an attempt on top of a verdict nobody
    // reviewed. Resuming is `start --task`, which takes the `retry` edge explicitly.
    let mut failed = candidate("01ARZ3NDEKTSV4RRFFQ69G5FAA", 0);
    failed.state = TaskState::FailedGates;
    assert!(failed.has_work, "the fixture must have work to be a real test");
    assert_eq!(failed.ineligible_for_inference(), Some(Ineligible::FailedGates));
    assert_eq!(failed.ineligible_for_offline(), Some(Ineligible::FailedGates));
}

#[test]
fn a_task_waiting_for_a_merge_and_a_blocked_one_are_named_too() {
    // The same "no subtask left to run" sentence covered three different human actions. A task at
    // `ready_for_review` is waiting for a merge decision, not for a gate to finish; a task L5
    // blocked is waiting for a reopened subtask.
    let review = Candidate {
        state: TaskState::ReadyForReview,
        ..candidate("01ARZ3NDEKTSV4RRFFQ69G5FAA", 0)
    };
    assert_eq!(
        review.ineligible_for_inference(),
        Some(Ineligible::ReadyForReview)
    );

    let blocked = Candidate {
        state: TaskState::Blocked,
        has_work: false,
        ..candidate("01ARZ3NDEKTSV4RRFFQ69G5FAB", 0)
    };
    assert_eq!(blocked.ineligible_for_inference(), Some(Ineligible::Blocked));

    let rejected = vec![
        (&review.task, Ineligible::ReadyForReview),
        (&blocked.task, Ineligible::Blocked),
    ];
    let text = summarise(&rejected);
    assert!(text.contains("1 ready for review"), "{text}");
    assert!(text.contains("1 blocked, waiting for a human"), "{text}");
    assert!(!text.contains("no subtask"), "{text}");
}

#[test]
fn a_blocked_task_that_still_has_an_independent_subtask_is_runnable_again() {
    // L5 blocks a task only when nothing independent is left. A human who adds a subtask to a
    // blocked task's plan makes it runnable without touching the lifecycle, and the queue must
    // notice: keying the parked reason on the state alone would make that edit do nothing.
    let mut blocked = candidate("01ARZ3NDEKTSV4RRFFQ69G5FAA", 0);
    blocked.state = TaskState::Blocked;
    assert!(blocked.has_work);
    assert_eq!(blocked.ineligible_for_inference(), None);
}
