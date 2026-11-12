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
