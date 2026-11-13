# Dogfood tasks

Three tasks for the v0.1 exit criterion: on this repository the agent must close at least three
consecutive subtasks with green gates, unaided, and survive a `kill -9` in the middle of a step.

Each task is a directory under `tasks/` holding `.agent/` with the three plan files the daemon
reads — `PLAN.md`, `ACCEPTANCE.md` and `TASKS.json`. They are hand-written rather than generated
by `immortality plan --goal` on purpose: the planner is tested in M5.7, and a dogfood task that
depended on it would fail for planner reasons while pretending to be a cycle failure.

`crates/daemon/tests/dogfood.rs` loads every one through `Plan::load`, so a fixture that the
daemon would refuse fails in `cargo test` rather than hours into a run.

## The three tasks

| Task | Kind | What it exercises |
| --- | --- | --- |
| `word-count` | feature | the happy path: a new subcommand and its tests |
| `line-count` | bugfix | a real bug in the fixture repository, reproducible by one command |
| `dedupe` | refactor | changing a data structure without changing behaviour, guarded by existing tests |

`line-count` is a genuine defect, not a planted one: `count::lines` counts the trailing empty
string after a final newline, so `tally` reports one line too many for any file that ends with
`\n`. The repository's own tests do not pin the correct answer for that case, which is what makes
it a bug and not a disagreement.

## How a task is run

```sh
immortality start --task <id>          # after admitting it with a goal
```

The plan is copied into the fixture repository's `.agent/` directory by
`../scripts/admit.sh`, which takes one task name and the repository directory, so every run
starts from the same three files and a failed run can be repeated without hand-editing JSON. The
script does not commit anything: the run itself owns the branch.

## What makes a run pass

The roadmap's bar, unchanged:

- every subtask closes with green gates — commands from its own `DoD`, plus the reviewer where
  the `DoD` asks for it;
- at least three consecutive subtasks close without a human answering anything;
- a `kill -9` between two steps leaves a journal recovery can continue from, with no lost work
  and no double-applied effect.

A run that needed an escalation is not a failure of the cycle, but it is not evidence for the
bar either: escalations are counted separately in `docs/metrics-v0.1.md`.
