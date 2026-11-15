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

    #[error(
        "check `{id}` runs a pipeline, whose exit code is the last command's and not the work's: \
         `{command}` reports `tail`'s success even when the build failed"
    )]
    Piped { id: String, command: String },

    #[error("no checks and no review: nothing would gate this subtask")]
    NoGate,
}

/// Whether `command` contains an unquoted single `|`, which makes its exit code the wrong number.
///
/// # The defect this exists for
///
/// `/bin/sh` gives a pipeline the exit code of its **last** command. `cargo test 2>&1 | tail -30`
/// therefore exits `0` whenever `tail` succeeded, which is every time, and the gate reads that as
/// a green check on a build that did not compile. Measured on the `word-count` journals: eleven
/// such observations, each one `error[E…]` or `could not compile` in the output and `[exited 0 …]`
/// in the summary.
///
/// # Why a check is refused rather than run differently
///
/// The other fix is `set -o pipefail`, which is not in `/bin/sh`: `dash` rejects `set -o pipefail`
/// outright, and `bash`-as-`sh` accepts the option only by accident of the binary. A gate whose
/// verdict depends on which shell the machine happens to provide is worse than one that says it
/// cannot judge the command — the whole point of a gate is that its answer means the same thing
/// everywhere. Refusing is also the direction that fails closed: a plan with a masked check is
/// rejected at plan time, not trusted later.
///
/// `||` is left alone. It is a deliberate short-circuit, `a || b` is how a command says "try this,
/// otherwise that", and its status is the status of whichever branch ran.
///
/// Only unquoted pipes count: `grep -qE 'a|b' README.md` is an ordinary check and its pipe is the
/// pattern's. This is the one piece of shell the scan has to understand, and it is why quotes are
/// tracked byte by byte instead of calling `command.contains('|')`.
#[must_use]
pub fn masks_exit_code(command: &str) -> bool {
    let bytes = command.as_bytes();
    let mut index = 0;
    // Which quote opened the current region, if any. A backslash escapes the next byte outside
    // single quotes, exactly as the shell does.
    let mut quote: Option<u8> = None;
    while let Some(&byte) = bytes.get(index) {
        match byte {
            b'\\' if quote != Some(b'\'') => index += 1,
            b'\'' | b'"' => match quote {
                None => quote = Some(byte),
                Some(open) if open == byte => quote = None,
                // The other quote character inside this region is an ordinary byte.
                Some(_) => {}
            },
            // `||` is a logical or, and `|&` is a pipe that also passes stderr. The second one
            // masks the same way, so only the doubled bar is excused.
            b'|' if quote.is_none() => {
                if bytes.get(index + 1) != Some(&b'|') {
                    return true;
                }
                index += 1;
            }
            _ => {}
        }
        index += 1;
    }
    false
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
            if masks_exit_code(&check.command) {
                return Err(DodInvalid::Piped {
                    id: check.id.clone(),
                    command: check.command.trim().to_string(),
                });
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

    fn one_check(command: &str) -> Dod {
        Dod {
            statement: "s".into(),
            checks: vec![DodCheck {
                id: "test".into(),
                command: command.into(),
                required: true,
            }],
            review: false,
        }
    }

    #[test]
    fn a_piped_check_is_rejected_because_its_exit_code_is_the_last_commands() {
        // The measured defect: `/bin/sh -c 'cargo test 2>&1 | tail -30'` exits 0 on a build that
        // did not compile, so this check turns a real failure into a green gate.
        assert_eq!(
            one_check("cargo test 2>&1 | tail -30").validate().unwrap_err(),
            DodInvalid::Piped {
                id: "test".into(),
                command: "cargo test 2>&1 | tail -30".into(),
            }
        );
    }

    #[test]
    fn a_pipe_inside_quotes_is_part_of_the_pattern_and_is_accepted() {
        // `grep -qE 'a|b' README.md` is an ordinary definition-of-done check. Refusing it would
        // turn a guard into an obstacle and push plan authors away from `DoD` checks entirely.
        assert!(one_check("grep -qE 'a|b' README.md").validate().is_ok());
        assert!(one_check(r#"grep -qE "a|b" README.md"#).validate().is_ok());
        assert!(one_check(r"grep -qE a\|b README.md").validate().is_ok());
    }

    #[test]
    fn a_logical_or_is_not_a_masked_exit_code() {
        // `a || b` means "try this, otherwise that" and its status is the status of whichever
        // branch ran. That is a verdict, not a mask, and a plan may ask for it.
        assert!(one_check("cargo test || cargo test --offline").validate().is_ok());
    }

    #[test]
    fn a_pipe_that_also_reroutes_stderr_is_still_a_pipeline() {
        // `|&` passes stderr through the pipe and takes the last command's status, exactly like
        // `|`. Excusing everything that begins with a bar would leave the defect reachable.
        assert!(one_check("cargo test |& tail -5").validate().is_err());
    }

    #[test]
    fn quotes_do_not_hide_a_pipe_that_is_really_there() {
        // The scan must not be talked out of a real pipeline by an unbalanced or closed quote.
        assert!(one_check("echo 'done' | tee log").validate().is_err());
        assert!(one_check("echo \"x\" | tee log").validate().is_err());
    }
}
