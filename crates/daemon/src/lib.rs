//! The long-lived engine.
//!
//! M0 contains the diagnostic surface: `doctor` is the artefact of this milestone, because on
//! this project almost every confusing failure is environmental — the backend lives on
//! another machine, the prefix cache decides whether a step costs one second or ninety, and
//! a missing trusted host turns every model call into a human approval.
//!
//! The scheduler, runner and RPC server arrive at M4 and M5.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod checkpoints;
pub mod doctor;
pub mod gates;
pub mod recovery;
pub mod rpc;
pub mod runner;
pub mod scheduling;
pub mod server;
pub mod service;
pub mod startup;
pub mod toolset;

/// Version of the daemon<->client protocol.
///
/// Re-exported rather than declared twice: [`rpc::VERSION`] is what is actually put on the
/// wire, and two constants with the same value is one constant and one accident waiting to
/// diverge.
pub const PROTOCOL_VERSION: u32 = rpc::VERSION;
