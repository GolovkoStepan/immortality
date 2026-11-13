# Acceptance — dedupe

1. `cargo test --offline` exits zero.
2. Every pre-existing test in `src/count.rs` still passes **unchanged**: the refactor is not
   allowed to edit an assertion to match new behaviour.
3. `printf 'b a b\n' > sample.txt` then `cargo run -q --offline -- sample.txt` prints
   `unique: 2`.
