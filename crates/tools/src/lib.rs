//! Tool layer: execution, policy and workspace isolation.
//!
//! Filled in at M3. M0 establishes the crate so that the dependency direction
//! `daemon → tools → llm → core` is real from the first commit and so that `core` can be
//! proved free of framework dependencies by the build itself.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod args;
pub mod classify;
pub mod delegate;
pub mod escalate;
pub mod find;
pub mod git;
pub mod glob;
pub mod judge;
pub mod notify;
pub mod policy;
pub mod processes;
pub mod read;
pub mod search;
pub mod shell;
pub mod state;
pub mod tool;
pub mod write;

pub use args::Args;
pub use classify::{Classification, External, classify};
pub use delegate::{
    Delegate, Delegation, NoSubAgent, SubAgent, SubAgentOutcome, SubAgentSlot, probe_delegated,
};
pub use escalate::{DirectoryQueue, Escalate, EscalationQueue, JournalQueue, Pending, answer_question};
pub use find::GlobTool;
pub use git::Git;
pub use glob::{Glob, GlobError, PatternSet};
pub use judge::{Judge, Judgement};
pub use notify::{MacNotifier, Notifier, Silent, system};
pub use policy::{Denial, Escalation, PathUse, PolicyGate, Verdict};
pub use processes::{
    KillProcess, Liveness, ProcessRecord, ProcessRegistry, ReadProcess, StartProcess, is_running,
    probe_started, start_digest as process_start_digest,
};
pub use read::{ReadFile, looks_binary};
pub use search::Grep;
pub use shell::{CommandOutput, run_command};
pub use shell::{Shell, ShellState};
pub use state::{GrepJournal, ReadState, StateFile, UpdateState};
pub use tool::{OutputBudget, Registry, RegistryError, Tool, ToolCtx, ToolError, ToolResult};
pub use write::{EditFile, WriteFile};

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
