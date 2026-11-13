//! The library half of `tally`.
//!
//! The counting rules live here rather than in `main.rs` so that integration tests under `tests/`
//! can reach them. That is not tidiness: a dogfood task's gate has to be a command that is **red
//! before the work and green after**, and a unit test inside `count.rs` cannot provide that. A
//! filter that matches no test exits zero (`cargo test nothing_matches` is a green run), so a gate
//! pointing at a test the agent has not written yet would be green for free — and a subtask could
//! close without anything changing.

pub mod count;
