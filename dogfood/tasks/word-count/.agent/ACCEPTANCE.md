# Acceptance — `unique`

1. `cargo test --offline` exits zero.
2. `cargo build --offline` exits zero.
3. `printf 'a b a\n' > sample.txt` then `cargo run -q --offline -- sample.txt` prints a
   `unique: 1` line, because `a` appears twice and `b` once.
