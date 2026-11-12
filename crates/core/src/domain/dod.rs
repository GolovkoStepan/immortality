//! Definition of done: what a subtask must prove before it may be closed.
//!
//! The model never decides its own success. It *claims* done by calling `finish_subtask`,
//! and the runner then evaluates the [`Dod`] attached to the subtask: commands from
//! `ACCEPTANCE.md` are executed, and the reviewer's verdict is recorded. Only the system
//! writes `done` into the state machine.
//!
//! This type is the contract between the planner (which writes it into `TASKS.json`) and the
//! gate runner (which executes it). It is validated at plan time so that an unrunnable gate
//! is a planning error, not a surprise fifteen minutes into verification: an empty command,
//! or the same id twice, would produce a gate that silently always passes.

use serde::{Deserialize, Serialize};

/// A single machine-checkable claim about the working tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DodCheck {
    /// Short stable name, unique inside one `DoD`: `build`, `test`, `lint`.
    pub id: String,
    /// Shell command, executed by the same sandboxed runner as the `shell` tool.
    pub command: String,
    /// Whether a non-zero exit fails the gate. False is allowed for informational checks
    /// (`cargo outdated`), which is why the flag is explicit rather than implied.
    #[serde(default = "default_true")]
    pub required: bool,
}

const fn default_true() -> bool {
    true
}

/// What "done" means for one subtask.
///
/// `human_readable` is not decoration: the reviewer subagent receives it as the only
/// specification of intent, and it is what a human reads in `TASKS.json`. A `DoD` with checks
/// but no sentence is a gate nobody can interpret, so the field is mandatory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dod {
    /// One sentence stating the observable outcome.
    pub statement: String,
    #[serde(default)]
    pub checks: Vec<DodCheck>,
    /// Whether a reviewer subagent must approve the diff. Off for mechanical subtasks
    /// (formatting, file moves) where the checks already prove the outcome and a review
    /// would only burn tokens.
    #[serde(default = "default_true")]
    pub review: bool,
}

/// Why a `DoD` cannot be used as a gate.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DodInvalid {
    #[error("definition of done has no statement")]
    EmptyStatement,

    #[error("check `{id}` has an empty command")]
    EmptyCommand { id: String },

    #[error("duplicate check id `{id}`")]
    DuplicateId { id: String },

    #[error("check id `{id}` is not a usable file-line key")]
    BadId { id: String },

    #[error("no checks and no review: nothing would gate this subtask")]
    NoGate,
}

impl Dod {
    /// A `DoD` that only a reviewer can judge. Useful for prose and design subtasks.
    #[must_use]
    pub fn reviewed(statement: impl Into<String>) -> Self {
        Self {
            statement: statement.into(),
            checks: Vec::new(),
            review: true,
        }
    }

    /// Validate before the subtask starts, not after the model claims success.
    ///
    /// The `NoGate` rule is the important one: `checks: []` with `review: false` is accepted
    /// by every field-level check and makes `finish_subtask` unconditionally succeed.
    pub fn validate(&self) -> Result<(), DodInvalid> {
        if self.statement.trim().is_empty() {
            return Err(DodInvalid::EmptyStatement);
        }
        if self.checks.is_empty() && !self.review {
            return Err(DodInvalid::NoGate);
        }
        let mut seen = std::collections::BTreeSet::new();
        for check in &self.checks {
            if check.command.trim().is_empty() {
                return Err(DodInvalid::EmptyCommand { id: check.id.clone() });
            }
            if !is_line_key(&check.id) {
                return Err(DodInvalid::BadId { id: check.id.clone() });
            }
            if !seen.insert(check.id.as_str()) {
                return Err(DodInvalid::DuplicateId { id: check.id.clone() });
            }
        }
        Ok(())
    }

    /// Checks the runner must execute, in declaration order.
    ///
    /// Order is preserved rather than sorted so that `build` before `test` costs one compile.
    pub fn required_checks(&self) -> impl Iterator<Item = &DodCheck> {
        self.checks.iter().filter(|c| c.required)
    }
}

/// Ids end up in `--message`-like contexts and in log lines, so restrict them to a shape
/// that needs no quoting: lowercase letters, digits, `-` and `_`.
fn is_line_key(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 32
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reviewed_dod_needs_nothing_else() {
        assert!(Dod::reviewed("the endpoint returns 200").validate().is_ok());
    }

    #[test]
    fn an_ungated_dod_is_rejected() {
        let dod = Dod {
            statement: "something".into(),
            checks: Vec::new(),
            review: false,
        };
        assert_eq!(dod.validate().unwrap_err(), DodInvalid::NoGate);
    }

    #[test]
    fn an_empty_statement_is_rejected() {
        let dod = Dod {
            statement: "   ".into(),
            checks: Vec::new(),
            review: true,
        };
        assert_eq!(dod.validate().unwrap_err(), DodInvalid::EmptyStatement);
    }

    #[test]
    fn duplicate_check_ids_are_rejected() {
        let dod = Dod {
            statement: "s".into(),
            checks: vec![
                DodCheck {
                    id: "test".into(),
                    command: "cargo test".into(),
                    required: true,
                },
                DodCheck {
                    id: "test".into(),
                    command: "cargo nextest".into(),
                    required: true,
                },
            ],
            review: false,
        };
        assert_eq!(
            dod.validate().unwrap_err(),
            DodInvalid::DuplicateId { id: "test".into() }
        );
    }

    #[test]
    fn empty_commands_are_rejected() {
        let dod = Dod {
            statement: "s".into(),
            checks: vec![DodCheck {
                id: "build".into(),
                command: "  ".into(),
                required: true,
            }],
            review: false,
        };
        assert_eq!(
            dod.validate().unwrap_err(),
            DodInvalid::EmptyCommand { id: "build".into() }
        );
    }

    #[test]
    fn check_ids_must_be_line_keys() {
        for id in ["build all", "build:all", "Build", ""] {
            let dod = Dod {
                statement: "s".into(),
                checks: vec![DodCheck {
                    id: id.into(),
                    command: "true".into(),
                    required: true,
                }],
                review: false,
            };
            assert!(dod.validate().is_err(), "id `{id}` was accepted");
        }
    }

    #[test]
    fn required_checks_preserve_declaration_order() {
        let dod = Dod {
            statement: "s".into(),
            checks: vec![
                DodCheck {
                    id: "build".into(),
                    command: "cargo build".into(),
                    required: true,
                },
                DodCheck {
                    id: "outdated".into(),
                    command: "cargo outdated".into(),
                    required: false,
                },
                DodCheck {
                    id: "test".into(),
                    command: "cargo test".into(),
                    required: true,
                },
            ],
            review: false,
        };
        let ids: Vec<_> = dod.required_checks().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["build", "test"]);
    }

    #[test]
    fn optional_checks_default_to_required_when_omitted() {
        let dod: Dod =
            serde_json::from_str(r#"{"statement":"s","checks":[{"id":"t","command":"c"}]}"#).unwrap();
        assert!(dod.checks[0].required);
        assert!(dod.review, "review defaults to on");
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let err = serde_json::from_str::<Dod>(r#"{"statement":"s","cheks":[]}"#).unwrap_err();
        assert!(err.to_string().contains("cheks"), "{err}");
    }
}
