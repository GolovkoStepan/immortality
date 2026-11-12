//! Tool layer: execution, policy and workspace isolation.
//!
//! Filled in at M3. M0 establishes the crate so that the dependency direction
//! `daemon → tools → llm → core` is real from the first commit and so that `core` can be
//! proved free of framework dependencies by the build itself.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod classify;
pub mod glob;
pub mod policy;

pub use classify::{Classification, classify};
pub use glob::{Glob, GlobError, PatternSet};
pub use policy::{Denial, Escalation, PathUse, PolicyGate, Verdict};

/// Classification of a tool's side effects.
///
/// Every tool declares this, and one `PolicyGate` consults it for every call. Keeping the
/// classification in one enum is what makes "how much damage can this do?" auditable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Effect {
    /// Nothing outside the process changes.
    ///
    /// Ordering is severity: a command line made of several simple commands takes the maximum, so
    /// `Read < Mutate < External` is what makes that fold correct.
    Read,
    /// Files, git or processes change, but within the task worktree.
    Mutate,
    /// Leaves the machine or is irreversible: push, network, sudo, delete outside worktree.
    External,
}

/// Declared idempotency, which decides crash recovery behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Idempotency {
    /// Safe to repeat blindly.
    Idempotent,
    /// Repeating is safe if the content matches; check before repeating.
    ByContent,
    /// Repeating may duplicate the effect; probe before repeating.
    Unknown,
}
