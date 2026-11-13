#!/bin/bash
# Admit a dogfood task: copy its plan into the fixture repository's `.agent/`.
#
# Usage: admit.sh <task> [repo]
#
# The repository defaults to `dogfood/repo`. Refusing to overwrite a plan that is already there
# is deliberate: a second admission would silently discard whatever the previous run's planner
# or a human left behind, and the point of a dogfood task is that the run starts from the same
# bytes every time. `admit.sh <task> <repo> --force` replaces them.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <task> [repo] [--force]" >&2
  exit 2
fi

task=$1
repo=${2:-$(dirname "$0")/../repo}
force=0
for arg in "$@"; do
  if [ "$arg" = "--force" ]; then
    force=1
  fi
done

source_dir="$(cd "$(dirname "$0")/../tasks/$task" && pwd)"
target_dir="$(cd "$repo" && pwd)/.agent"

if [ ! -d "$source_dir/.agent" ]; then
  echo "admit.sh: no such task: $task (looked in $source_dir/.agent)" >&2
  exit 1
fi

if [ "$force" -ne 1 ] && [ "$(ls -A "$target_dir" 2>/dev/null || true)" ]; then
  echo "admit.sh: $target_dir already has a plan; pass --force to replace it" >&2
  exit 1
fi

mkdir -p "$target_dir"
cp "$source_dir/.agent/PLAN.md" "$source_dir/.agent/ACCEPTANCE.md" "$source_dir/.agent/TASKS.json" "$target_dir/"
echo "admitted $task into $target_dir"
