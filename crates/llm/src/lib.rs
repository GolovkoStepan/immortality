//! OpenAI-compatible backend client.
//!
//! # What lives here
//!
//! - [`turn`] — the normalised assistant turn the engine works with.
//! - [`wire`] — the JSON on the wire, exactly as the protocol defines it.
//! - [`stream`] — the SSE decoder that turns a byte stream into a turn.
//! - [`sse`] — the byte-level SSE parser underneath the decoder.
//!
//! The HTTP client, error classification and backoff arrive next; these are what they operate
//! on, which is why they land first.
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

pub mod sse;
pub mod stream;
pub mod turn;
pub mod wire;

pub use stream::{StreamDecoder, StreamEnd, StreamError, TurnAccumulator};
pub use turn::{AssistantTurn, CalledFunction, FINISH_LENGTH, ToolCall, Usage};
pub use wire::{CacheReport, ChatRequest, ChatResponse, Message, StreamOptions, ToolDef};
