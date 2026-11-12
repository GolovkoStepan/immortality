//! Error type for the core crate.
//!
//! Core code returns `Result<T, Error>` rather than `anyhow::Error`: the engine distinguishes
//! recoverable conditions (a blocked subtask, an unavailable backend) from programming errors,
//! and callers need to match on them.

use std::path::PathBuf;

/// Errors produced by the core crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("configuration error in {path}: {source}")]
    ConfigToml {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("cannot read {path}: {source}")]
    ConfigIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid configuration: {0}")]
    ConfigInvalid(String),

    #[error("invalid identifier: {0}")]
    InvalidId(String),

    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("logging setup failed: {0}")]
    Logging(String),

    /// The journal contains something the schema does not recognise.
    ///
    /// Deliberately not recovered from: a record that cannot be parsed might be the one that
    /// said a mutating command already ran, and guessing would either duplicate an effect or
    /// skip one.
    #[error("journal is corrupt: {0}")]
    JournalCorrupt(String),

    #[error("journal schema version {found} is not supported; this build reads version {expected}")]
    JournalVersion { found: u16, expected: u16 },

    /// A caller broke the tool-call protocol: wrong order, duplicate id, verdict for an
    /// unknown call. Always a programming error, never a runtime condition.
    #[error("tool-call protocol violation: {0}")]
    Protocol(String),

    /// An artifact's bytes do not match the digest that names it.
    ///
    /// Never papered over: the digest is the only reason to trust the content, and a model
    /// that is served a fabricated tool result will act on it.
    #[error("artifact is corrupt: {0}")]
    ArtifactCorrupt(String),
}

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

/// Convenient alias.
pub type Result<T> = std::result::Result<T, Error>;
