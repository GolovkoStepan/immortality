#!/bin/bash
# Admit a dogfood task: build a fresh working repository for it and install its plan.
#
# Usage: admit.sh <task> [target] [--force]
#
# The target defaults to `dogfood/work/<task>`, which git ignores: it is the output of a run, not
# a fixture. The script copies `dogfood/repo` there, makes it its own git repository, installs the
# task's plan and its **failing** specification, and commits them.
#
# # Why the target is a copy and its own repository
#
# The run does not edit the tree it was given. It asks `WorktreeManager` for a worktree of that
# repository and does the work on a branch there, so `dogfood/repo` itself would have to be a git
# repository for anything to make sense. It is not: it is a directory inside `immortality`'s own
# repository, and `git worktree add` from it would produce a worktree of *immortality* — the agent
# would be handed the tool's entire workspace and told to implement `count::once_words` in it.
#
# Copying to a target and running `git init` there is what makes the fixture the repository the run
# believes it is. It also keeps the checked-in fixture free of a nested `.git` that the parent
# repository would record as a gitlink.
#
# # Why the specification is committed, not merely written
#
# A gate has to be red before the work and green after. Two ways to get this wrong, both silent:
#
#   * the specification is untracked, so the worktree git hands the agent does not contain it and
#     `cargo test --offline` is green before anything was done — a subtask closes having changed
#     nothing;
#   * the specification passes on the untouched repository, which is the same failure with extra
#     steps.
#
# Committing it here makes the first impossible and `tests/dogfood.rs` asserts the second.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <task> [target] [--force]" >&2
  exit 2
fi

task=$1
here="$(cd "$(dirname "$0")/.." && pwd)"
target=${2:-"$here/work/$task"}
force=0
for arg in "$@"; do
  if [ "$arg" = "--force" ]; then
    force=1
  fi
done

source_dir="$here/tasks/$task"
if [ ! -d "$source_dir/.agent" ]; then
  echo "admit.sh: no such task: $task (looked in $source_dir/.agent)" >&2
  exit 1
fi
if [ ! -f "$source_dir/tests/behaviour.rs" ]; then
  echo "admit.sh: $task has no tests/behaviour.rs; its gate would pass before the work" >&2
  exit 1
fi

# Refusing to reuse a target is deliberate: it may hold a run's branch, journal and uncommitted
# work, and discarding those silently is the one thing a dogfood script must not do.
if [ -e "$target" ] && [ "$force" -ne 1 ]; then
  echo "admit.sh: $target already exists; pass --force to replace it" >&2
  exit 1
fi
rm -rf "$target"

mkdir -p "$target"
# `target/` and `Cargo.lock` are build output of the fixture, not its source.
cp -R "$here/repo/." "$target/"
rm -rf "$target/target" "$target/.agent" "$target/tests" "$target/Cargo.lock"

mkdir -p "$target/.agent"
cp "$source_dir/.agent/PLAN.md" "$source_dir/.agent/ACCEPTANCE.md" \
   "$source_dir/.agent/TASKS.json" "$target/.agent/"
mkdir -p "$target/tests"
cp "$source_dir/tests/behaviour.rs" "$target/tests/"

git -C "$target" init -q -b main
# The identity is passed per-command rather than written into `.git/config`: the fixture is
# discarded after a run, and a script that edits repository config is the habit the policy layer
# exists to prevent.
GIT_AUTHOR_NAME=tally GIT_AUTHOR_EMAIL=tally@example.invalid \
GIT_COMMITTER_NAME=tally GIT_COMMITTER_EMAIL=tally@example.invalid \
  git -C "$target" add -A
GIT_AUTHOR_NAME=tally GIT_AUTHOR_EMAIL=tally@example.invalid \
GIT_COMMITTER_NAME=tally GIT_COMMITTER_EMAIL=tally@example.invalid \
  git -C "$target" commit -q -m "fixture: $task admitted with its failing specification"

echo "admitted $task into $target"
