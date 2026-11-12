//! Domain, durable state and pure logic for `immortality`.
//!
//! This crate deliberately depends on no async runtime, no HTTP client and no PTY library.
//! Everything here is synchronous and testable in microseconds, which is what makes the
//! recovery protocol (the riskiest part of the project) cheap to test exhaustively.

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod config;
pub mod error;
pub mod ids;
pub mod logging;
pub mod paths;

pub use error::{Error, Result};
