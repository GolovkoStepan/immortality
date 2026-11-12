//! OpenAI-compatible backend client.
//!
//! # What is here in M0
//!
//! Only the SSE parser and the streamed tool-call accumulator, both lifted from the S1 spike
//! where they were exercised against the live backend. These two are the fragile parts of
//! the protocol — the live model emits tool-call arguments **character by character**, and
//! the stream carries `: keep-alive` comments — so they landed first.
//!
//! The HTTP client, retry/backoff and health checking arrive in M2.

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod sse;
pub mod stream;
