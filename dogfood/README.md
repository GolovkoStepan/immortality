# Dogfood tasks

Three tasks for the v0.1 exit criterion: on this repository the agent must close at least three
consecutive subtasks with green gates, unaided, and survive a `kill -9` in the middle of a step.

| Task | Kind | Subtasks | What it exercises |
| --- | --- | --- | --- |
| `word-count` | feature | 3 | a counting rule, the binary that prints it, then a sorted listing |
| `line-count` | bugfix | 1 | a real defect in the fixture repository, reproducible by one command |
| `dedupe` | refactor | 1 | changing a data structure without changing behaviour, guarded from outside |

`word-count` has three subtasks because the v0.1 bar is three **consecutive** closures, and a task
is admitted into its own repository and run once. Two subtasks could not demonstrate it.

`line-count` is a genuine defect, not a planted one: `count::lines` counts the trailing empty field
after a final newline, so `tally` reports one line too many for any file that ends with `\n`. The
repository's own tests do not pin the correct answer for that case, which is what makes it a bug
and not a disagreement.

## Why each task ships failing tests

A gate has to distinguish "the work was done" from "the work was skipped". `cargo test --offline`
on the untouched repository is green, so a subtask that changed nothing would still close, and the
milestone's criterion would be met by a run that did no work.

Each task therefore keeps its specifications in `tests/`, and `admit.sh` installs and **commits**
them into the admitted repository. The run starts from a repository whose gate is
**failing**, and success means making it pass without editing the test. `tests/dogfood.rs` asserts
that red run for every task, in a worktree — where the run actually works — so a fixture that
quietly became a no-op fails in `cargo test` rather than during a dogfood session.

Committing it matters for the same reason. A run does not edit the tree it was given: it asks the
`WorktreeManager` for a worktree and works on a branch there. `git worktree add` reproduces the
commit, not the checkout, so an untracked specification simply would not exist on the branch and
the gate would be green before any work. The two failures look identical from the outside — a
subtask closes having changed nothing — and only one of them is caught by running the gate in the
main checkout.

## One gate per subtask, not one per task

A task with several subtasks needs a gate that is red **when that subtask starts**, not one that
became green when the first one landed. So each subtask names its own test target, and its spec
lives in its own file:

```json
{ "id": "test", "command": "cargo test --offline --test once_line", "required": true }
```

`cargo test --test <name>` compiles only the named target, so `tests/once.rs` passing leaves
`tests/once_line.rs` and `tests/once_sorted.rs` failing. A single shared `cargo test` would have
been green for the second and third subtasks the moment the first one landed, and three closures
would have proved that one file changed.

This is not hypothetical: it is what `word-count` did before this was fixed. After the counting
rule was added, `cargo test` and `cargo build` both passed while `src/main.rs` still printed
nothing, so the second subtask could have closed having changed no file. `tests/dogfood.rs` now
checks every subtask at its own start — admit fresh, apply the earlier subtasks' fixes, require
this subtask's checks to still fail — and a fixture that regresses to one shared gate fails in
`cargo test` rather than during a dogfood run.

The specifications live in `tests/` rather than inside `src/count.rs` for a second reason: a test
that does not exist yet is not a gate at all. `cargo test nothing_matches` exits zero, so "the test
the agent has not written" would be green for free.

The other half of that property is that the gate **can** go green. A task no honest fix can
satisfy reports "the model failed" for a reason outside the agent, and it reads exactly like a
model that could not do the job. `tests/dogfood.rs` therefore applies the smallest plausible fix
each goal asks for (a real one: the `BTreeSet` in `dedupe` is the refactor, not a stub) and requires
its gate to pass. Red before, green after, both run in CI.

The rehearsal runs twice, because the seam between the run and the wire is its own kind of code:
once with a model injected (`ServerState::with_model`) and once with none, so the run builds the
real `LlmModel` and speaks HTTP to a scripted server. The second pass covers the streaming decoder,
the tool-call reassembly and the retry policy — the code a live backend exercises and nothing else
does — and it costs no network. It also had to teach the mock one thing: the reviewer calls
`turn_blocking`, which asks for `stream: false`, so its answer is JSON and not SSE.

There is a third way the gate can never go green, and it is the one that looks like success until
you watch it: the gate is `cargo test`, whose diff is taken against the worktree and includes every
untracked file. Without a `.gitignore` the gate's own `target/` enters that diff, the sanity scan
reports `Finding::Binary` for the first compiled artefact, and the subtask can never close no
matter what the model writes — every attempt ends in `failed_gates`. `repo/.gitignore` exists for
that reason, not for tidiness, and `tests/dogfood.rs` runs the whole rehearsal (admission, a real
worktree, a real `cargo test` gate, a model that makes one real edit) so this cannot regress.

## Layout

Each task is a directory under `tasks/` holding:

- `.agent/` with the three plan files the daemon reads — `PLAN.md`, `ACCEPTANCE.md`, `TASKS.json`;
- `tests/`, the failing specifications `admit.sh` installs — one file per subtask, named after the
  `--test` target its `DoD` names.

`repo/` is the fixture's source material, not a repository to run in: it is a plain directory
inside this repository, so `git worktree add` run there would produce a worktree of `immortality`
and hand the agent the tool's own workspace. `admit.sh` copies it to a target, and that target is
the repository the run sees.

The plans are hand-written rather than generated by `immortality plan --goal` on purpose: the
planner is tested in M5.7, and a dogfood task that depended on it would fail for planner reasons
while pretending to be a cycle failure.

`crates/daemon/tests/dogfood.rs` loads every plan through `Plan::load`, so a fixture the daemon
would refuse fails in `cargo test` rather than hours into a run.

## How a task is run

```sh
scripts/admit.sh word-count                              # copy, own repo, commit the specs
cd work/word-count && cargo test --offline --test once   # the first subtask's gate is red
immortality start --project work/word-count "<the task's goal>"
```

`admit.sh` writes to `work/<task>` unless a second argument names another target. That directory is
outside version control (`.gitignore`) because it is the output of a run, and it is a copy rather
than the fixture for two reasons: `dogfood/repo` must stay free of a nested `.git`, and the run
needs a repository of its own to make a worktree from.

It refuses to reuse an existing target without `--force`: the target may hold a run's branch,
journal and uncommitted work, and discarding those silently is the one thing a dogfood script must
not do. `--force` replaces the whole directory, plan and tests included.

Nothing inside a target is committed to this repository: the target, its `.agent/`, `target/` and
its sibling `work-agent/` worktree are run artefacts and are ignored by git.

## What makes a run pass

- every subtask closes with green gates — commands from its own `DoD`, plus the reviewer where the
  `DoD` asks for it;
- at least three consecutive subtasks close without a human answering anything, which
  `immortality metrics` reports as `longest_green_streak`;
- a `kill -9` between two steps leaves a journal recovery can continue from, with no lost work and
  no double-applied effect.

A run that needed an escalation is not a failure of the cycle, but it is not evidence for the bar
either: escalations are counted separately in `docs/metrics-v0.1.md`.
