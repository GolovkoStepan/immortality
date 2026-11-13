# Acceptance — dedupe

1. `cargo test --offline` exits zero. It is **red before the work**: `tests/behaviour.rs` calls
   `count::unique_words_sorted`, which does not exist yet.
2. Every pre-existing test in `src/count.rs` still passes **unchanged**: the refactor is not
   allowed to edit an assertion to match new behaviour.
3. `tests/behaviour.rs` pins the pre-existing normalisation from outside the module, so a
   refactor that changed it would fail even if the module's own tests had been edited to agree.
4. No file under `tests/` was edited: the failing test is the specification, not a draft.
