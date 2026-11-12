//! The long-lived engine.
//!
//! M0 contains the diagnostic surface: `doctor` is the artefact of this milestone, because on
//! this project almost every confusing failure is environmental — the backend lives on
//! another machine, the prefix cache decides whether a step costs one second or ninety, and
//! a missing trusted host turns every model call into a human approval.
//!
//! The scheduler, runner and RPC server arrive at M4 and M5.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod doctor;

/// Version of the daemon<->client protocol. Bumped on any incompatible change; the client
/// refuses to talk to a daemon with a different major value.
pub const PROTOCOL_VERSION: u32 = 1;
