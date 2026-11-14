# Acceptance — `once`

`word-count` is the task that demonstrates the v0.1 bar, so its acceptance is written per subtask:
each one has a gate that is **red when that subtask starts**. `cargo test --offline --test <name>`
compiles only its own target, which is what makes that true — a shared `cargo test` would be green
for the second and third subtasks as soon as the first one landed.

1. **Subtask 1.** `cargo test --offline --test once` exits zero. It is red before the work:
   `tests/once.rs` calls `count::once_words`, which does not exist yet.
2. **Subtask 2.** `cargo test --offline --test once_line` exits zero. It is red before the work:
   it runs the real binary and expects a `once: N` line, which `src/main.rs` does not print yet.
3. **Subtask 3.** `cargo test --offline --test once_sorted` exits zero, and so does the whole
   suite (`cargo test --offline`). It is red before the work: `tests/once_sorted.rs` calls
   `count::once_words_sorted`, which does not exist yet.
4. `printf 'a b a\n' > sample.txt` then `cargo run -q --offline -- sample.txt` prints a
   `once: 1` line, because `a` appears twice and `b` once.
5. No file under `tests/` was edited: the failing tests are the specification, not a draft.
