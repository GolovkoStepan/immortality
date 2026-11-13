# Acceptance — line count

1. `cargo test --offline` exits zero. It is **red before the work**: `tests/behaviour.rs` pins the
   trailing-newline rule and the shipped `count::lines` violates it.
2. A regression test pins the trailing newline case: `lines("a\nb\n")` is 2 and `lines("a\nb")`
   is 2, so the two spellings of the same two-line file agree.
3. `printf 'a\nb\n' > sample.txt` then `cargo run -q --offline -- sample.txt` prints `lines: 2`.
4. No file under `tests/` was edited: the failing test is the specification, not a draft.
