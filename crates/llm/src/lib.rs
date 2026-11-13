//! OpenAI-compatible backend client.
//!
//! # What lives here
//!
//! - [`turn`] — the normalised assistant turn the engine works with.
//! - [`wire`] — the JSON on the wire, exactly as the protocol defines it.
//! - [`stream`] — the SSE decoder that turns a byte stream into a turn.
//! - [`sse`] — the byte-level SSE parser underneath the decoder.
//!
//! - [`client`] — the HTTP client: one attempt, cancellation, health.
//! - [`failure`] — why an attempt failed, and whether sending it again could help.
//! - [`extract`] — native tool calls first, a conservative text fallback second.
//! - [`schema`] — argument validation against the tool's JSON Schema, with a model-facing error.
//! - [`tokens`] — a calibrated estimate for when the backend reports no usage.
//!
//! # Why the split
//!
//! Three different things are being modelled: the *protocol* (which has warts), the *answer*
//! (which is what the engine reasons about), and *the stream* (which is a state machine over
//! bytes). Collapsing them into one module means a serde annotation on the wire type can change
//! engine behaviour, and a byte-level fix has to touch the same file as the retry policy.
//!
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod client;
pub mod extract;
pub mod failure;
pub mod prompt;
pub mod schema;
pub mod sse;
pub mod stream;
pub mod tokens;
pub mod turn;
pub mod wire;

pub use client::{Backend, DEFAULT_CONNECT_TIMEOUT, DEFAULT_REQUEST_TIMEOUT};
pub use extract::{Extracted, Extraction, extract, extract_from_text, repair_json};
pub use failure::{
    Failure, MAX_ATTEMPTS, MAX_BACKOFF, ModelFault, RetryKind, backoff, classify_status, classify_turn,
    should_retry,
};
pub use prompt::{messages, stable_prefix_len};
pub use schema::{
    ArgError, Defect, MAX_PROBLEMS, Problem, parse_arguments, render_observation, schema_defects, summarise,
    validate,
};
pub use stream::{StreamDecoder, StreamEnd, StreamError, TurnAccumulator};
pub use tokens::{DEFAULT_CHARS_PER_TOKEN, Estimator, char_count};
pub use turn::{AssistantTurn, CalledFunction, FINISH_LENGTH, ToolCall, Usage};
pub use wire::{CacheReport, ChatRequest, ChatResponse, Message, StreamOptions, ToolDef};
