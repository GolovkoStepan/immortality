# Acceptance — `once`

1. `cargo test --offline` exits zero. It is **red before the work**: `tests/behaviour.rs` calls
   `count::once_words`, which does not exist yet.
2. `cargo build --offline` exits zero.
3. `printf 'a b a\n' > sample.txt` then `cargo run -q --offline -- sample.txt` prints a
   `once: 1` line, because `a` appears twice and `b` once.
4. No file under `tests/` was edited: the failing test is the specification, not a draft.
