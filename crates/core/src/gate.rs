//! Deciding whether a claim of "done" is true.
//!
//! # Why the decision is pure
//!
//! Running `cargo test`, calling a reviewer model and reading a diff are all I/O, and a function
//! that does them and also decides is a function that can only be tested by doing them again.
//! This module does none of that: it takes what the three gates observed and answers one
//! question — may this subtask be closed? The executor in the daemon fills the observations in,
//! and every rule below is then testable in microseconds, which is what makes "the model cannot
//! declare success in bypass" a claim with a test rather than a hope.
//!
//! # The three gates, and why they are ordered
//!
//! `docs/tools-and-policy.md` §5 names them: the commands from `ACCEPTANCE.md`, a reviewer with a
//! clean context, and a diff sanity check. The executor runs them cheap-first — sanity, then
//! commands, then the reviewer — because a diff that adds a private key is already a failure, and
//! paying for a two-minute test suite and a model call before saying so wastes the one resource
//! this project is built to conserve.
//!
//! The order is the caller's, but the *rule* is here: any red required check, any sanity finding,
//! and any `request_changes` all mean the same thing for the subtask — it does not close, and the
//! attempt ends [`FailureReason::GateFailed`] or [`FailureReason::ReviewRejected`]. A subtask that
//! failed its own definition of done is not a subtask that succeeded.

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::domain::FailureReason;

mod diff;
mod evaluate;
mod review;

pub use diff::parse as parse_diff;
pub use evaluate::{CommandRun, Gates, Reviewer, evaluate, findings_text};
pub use review::{
    ReviewRequest, SYSTEM as REVIEW_SYSTEM, describe as describe_findings, parse_review,
    prompt as review_prompt,
};

/// How a file changed between the base commit and the working tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileStatus {
    Added,
    Modified,
    Deleted,
    Renamed,
}

impl FileStatus {
    /// The one-letter form `git diff --name-status` uses.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Added => "a",
            Self::Modified => "m",
            Self::Deleted => "d",
            Self::Renamed => "r",
        }
    }

    /// Whether the file exists in the working tree after the change.
    #[must_use]
    pub const fn is_present(self) -> bool {
        !matches!(self, Self::Deleted)
    }
}

/// One changed file, as both the reviewer and the sanity check need it.
///
/// `added_lines` is the raw `+` side of the diff, already stripped of the leading `+`. It is kept
/// apart from the rendered [`Diff::text`] so the sanity scan never has to parse a diff: parsing
/// once, in the code that already had to parse it to build this struct, is the only version that
/// cannot disagree with itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangedFile {
    /// Worktree-relative, with forward slashes, exactly as git reports it.
    pub path: PathBuf,
    pub status: FileStatus,
    /// Whether git (or the binary sniff, for an untracked file) says the content is not text.
    pub binary: bool,
    #[serde(default)]
    pub added_lines: Vec<String>,
}

impl ChangedFile {
    /// A changed file with no diff body, for tests and for a source that only has names.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>, status: FileStatus) -> Self {
        Self {
            path: path.into(),
            status,
            binary: false,
            added_lines: Vec::new(),
        }
    }
}

/// Everything the change under review consists of.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Diff {
    pub files: Vec<ChangedFile>,
    /// The rendered diff the reviewer reads. May be truncated when the caller could not keep it
    /// all; [`Diff::truncated`] then makes that a failed gate rather than a silent approval.
    pub text: String,
    /// Whether any part of the diff was dropped.
    pub truncated: bool,
}

impl Diff {
    /// An empty change: nothing was touched.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Paths, for the journal and for `files_touched`.
    #[must_use]
    pub fn paths(&self) -> Vec<PathBuf> {
        self.files.iter().map(|file| file.path.clone()).collect()
    }
}

/// One rule the diff broke.
///
/// A struct rather than a bare string because the escalation list and `status` want to group by
/// what kind of mistake it was, and because the observation the model reads should name the path
/// and the rule separately rather than as one unparseable sentence.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "rule", rename_all = "snake_case")]
pub enum Finding {
    /// A binary file was added or modified. Agents write source, not objects.
    Binary { path: PathBuf },
    /// A likely credential in an added line.
    Secret { path: PathBuf, kind: String },
    /// Something under `.git/` changed. The policy also refuses this, so a finding here means
    /// either a bug in the policy or a change made outside the tools.
    GitMetadata { path: PathBuf },
    /// A path outside the worktree, or one that escapes it.
    OutsideWorktree { path: PathBuf },
    /// A deferral the agent left for later without saying why.
    TodoWithoutReason { path: PathBuf },
    /// The diff was too large to review in full, so approving it would mean approving
    /// something nobody read.
    DiffTooLarge { bytes: usize },
}

impl Finding {
    /// The path this finding is about, when it has one.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::Binary { path }
            | Self::Secret { path, .. }
            | Self::GitMetadata { path }
            | Self::OutsideWorktree { path }
            | Self::TodoWithoutReason { path } => Some(path),
            Self::DiffTooLarge { .. } => None,
        }
    }

    /// What the model is told. Deliberately concrete: "a secret" without a path cannot be fixed.
    #[must_use]
    pub fn observation(&self) -> String {
        match self {
            Self::Binary { path } => {
                format!("`{}` is a binary file", path.display())
            }
            Self::Secret { path, kind } => {
                format!("`{}` appears to contain {kind}", path.display())
            }
            Self::GitMetadata { path } => {
                format!(
                    "`{}` is git metadata, which the agent must not edit",
                    path.display()
                )
            }
            Self::OutsideWorktree { path } => {
                format!("`{}` is outside the task worktree", path.display())
            }
            Self::TodoWithoutReason { path } => format!(
                "`{}` has a `TODO(agent)` with no reason; write `TODO(agent: why)`",
                path.display()
            ),
            Self::DiffTooLarge { bytes } => {
                format!("the diff is {bytes} bytes, too large to review in full; split the change")
            }
        }
    }
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.observation())
    }
}

/// The largest diff a reviewer is asked to read.
///
/// A cap, not a truncation point: a diff past this is a failed gate, because "[… 4 000 lines
/// omitted …]" in a prompt is a diff the reviewer approved without seeing. Splitting the change is
/// the correct answer, and it is one the model can act on.
pub const MAX_REVIEW_BYTES: usize = 512 * 1024;

/// Names that mark a value as a credential when they are assigned one.
///
/// Matched case-insensitively on the left of `=` or `:`. The list is short on purpose: a heuristic
/// that flags `token_count = 3` trains everyone to ignore the gate, and an ignored gate is worse
/// than no gate.
const SECRET_NAMES: &[(&str, &str)] = &[
    ("api_key", "an API key"),
    ("apikey", "an API key"),
    ("secret_key", "a secret key"),
    ("secretkey", "a secret key"),
    ("client_secret", "a client secret"),
    ("access_token", "an access token"),
    ("auth_token", "an auth token"),
    ("private_key", "a private key"),
    ("password", "a password"),
];

/// Values that look like configuration rather than a credential.
const PLACEHOLDERS: &[&str] = &[
    "",
    "...",
    "…",
    "changeme",
    "change_me",
    "example",
    "placeholder",
    "redacted",
    "todo",
    "xxx",
    "your-key-here",
];

/// Scan the added side of a diff for the mistakes a subtask must not ship.
///
/// Pure over [`Diff`], so every rule has a test that constructs the shape it is about rather than
/// making a repository produce it.
#[must_use]
pub fn sanity(diff: &Diff) -> Vec<Finding> {
    let mut findings = Vec::new();
    if diff.truncated {
        findings.push(Finding::DiffTooLarge {
            bytes: diff.text.len(),
        });
    }
    for file in &diff.files {
        if file.binary && file.status.is_present() {
            findings.push(Finding::Binary {
                path: file.path.clone(),
            });
        }
        if let Some(finding) = path_finding(file) {
            findings.push(finding);
            // A path that cannot be inside the worktree is not worth scanning line by line: the
            // finding is already decisive, and the lines would be about a file we do not own.
            continue;
        }
        for line in &file.added_lines {
            if let Some(kind) = secret_in(line) {
                findings.push(Finding::Secret {
                    path: file.path.clone(),
                    kind: kind.to_string(),
                });
            }
            if todo_without_reason(line) {
                findings.push(Finding::TodoWithoutReason {
                    path: file.path.clone(),
                });
            }
        }
    }
    findings
}

/// The finding that comes from the path alone, if any.
fn path_finding(file: &ChangedFile) -> Option<Finding> {
    let path = &file.path;
    let text = path.to_string_lossy();
    // Git never reports an absolute path in a diff, so one here means the source bypassed git and
    // read the filesystem — and an absolute path is where a global config lives.
    if path.is_absolute() || text.starts_with('~') || text.split('/').any(|part| part == "..") {
        return Some(Finding::OutsideWorktree { path: path.clone() });
    }
    if text.split('/').any(|part| part == ".git") {
        return Some(Finding::GitMetadata { path: path.clone() });
    }
    None
}

/// Why a line looks like a credential, if it does.
#[must_use]
pub fn secret_in(line: &str) -> Option<&'static str> {
    if !line.contains("PRIVATE KEY-----") && !line.contains("PGP PRIVATE KEY") {
        return credential_assignment(line);
    }
    Some("a private key")
}

/// The `NAME=value` shape, which is where a hand-written credential usually appears.
fn credential_assignment(line: &str) -> Option<&'static str> {
    let lower = line.to_ascii_lowercase();
    for (needle, label) in SECRET_NAMES {
        let mut from = 0;
        while let Some(offset) = lower[from..].find(needle) {
            let at = from + offset;
            from = at + needle.len();
            let rest = &lower[from..];
            // `::` is Ruby's scope operator, not an assignment: `OneTimePassword::Client` and
            // `Errors::TooManyPerHour` are class names, and reading the second colon as the start
            // of a value made `::Client.new` a credential. Measured on `qna_rails`: five findings
            // in `docs/services.md`, every one of them a namespaced class. Checked before the
            // assignment prefix so the `:` is never consumed twice.
            if rest.starts_with("::") {
                continue;
            }
            let Some(rest) = rest.trim_start().strip_prefix(['=', ':']) else {
                continue;
            };
            if looks_like_value(rest) {
                return Some(label);
            }
        }
    }
    None
}

/// The right-hand side of an assignment, cut where the value ends.
///
/// Not the rest of the line: in code, `password` is usually not alone. `nil, sign: nil)` after a
/// signature's `password:` is the next parameter, not the value of this one, and judging the whole
/// remaining line makes 22 characters with commas and a bracket — which the signature check reads
/// as a credential. Measured on `qna_rails`: `initialize(username: nil, password: nil)` in a
/// generated `docs/inventory.md` was refused as "a password", and three attempts of the same
/// subtask failed because the document was telling the truth.
///
/// A quoted literal is taken whole, because `password = "correct horse"` has a space inside the
/// value and cutting at the first space would hide it. Unquoted, the value ends at whitespace or
/// at the syntax that closes it: `,` `)` `]` `}` `;` `#`.
fn value_of(rest: &str) -> &str {
    let rest = rest.trim_start_matches([' ', '\t']);
    let quoted = rest
        .chars()
        .next()
        .filter(|ch| matches!(ch, '"' | '\'' | '`'))
        .and_then(|opening| {
            let body = &rest[opening.len_utf8()..];
            body.find(opening).map(|end| &body[..end])
        });
    if let Some(value) = quoted {
        return value;
    }
    let end = rest
        .find(|ch: char| ch.is_whitespace() || matches!(ch, ',' | ')' | ']' | '}' | ';' | '#'))
        .unwrap_or(rest.len());
    &rest[..end]
}

/// Whether the right-hand side of an assignment is a literal secret.
///
/// Errs towards "not a secret": a false positive costs a blocked subtask and a human's attention,
/// while the patterns that matter — a base64 key, a hex token, a quoted string — are all long
/// enough to clear the length bar. Interpolations and environment lookups are the common way to
/// write a credential *reference*, and they are exactly what must not be flagged.
fn looks_like_value(rest: &str) -> bool {
    let value = value_of(rest).trim();
    if value.is_empty() {
        return false;
    }
    let lower = value.to_ascii_lowercase();
    if PLACEHOLDERS.contains(&lower.as_str()) {
        return false;
    }
    if value.starts_with(['$', '<', '{', '%', '@'])
        || value.contains("{{")
        || lower.contains("env::var")
        || lower.contains("std::env")
        || lower.contains("os.environ")
        || lower.contains("process.env")
    {
        return false;
    }
    // A bare word with no digits or symbols is a setting ("password: required"), not a value.
    value.len() >= 12
        && value
            .chars()
            .any(|ch| ch.is_ascii_digit() || !ch.is_ascii_alphanumeric())
}

/// Whether a line defers work with no explanation.
///
/// The canonical escape hatch is `TODO(agent: why)`. The bare `TODO(agent)` and the empty
/// `TODO(agent: )` both read as "the model gave up here", which is the single thing a definition
/// of done exists to prevent.
#[must_use]
pub fn todo_without_reason(line: &str) -> bool {
    let Some(start) = line.find("TODO(agent") else {
        return false;
    };
    let rest = &line[start + "TODO(agent".len()..];
    match rest.chars().next() {
        Some(')') => true,
        Some(':') => {
            let reason = rest[1..].split(')').next().unwrap_or("").trim();
            reason.is_empty() || reason.ends_with("TODO")
        }
        // `TODO(agents)` and the like: a different word, not our marker.
        _ => false,
    }
}

/// What one command from the definition of done did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckOutcome {
    pub id: String,
    pub command: String,
    /// Whether a non-zero exit fails the gate.
    pub required: bool,
    /// `None` when the command was killed and never reported a code.
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub duration_ms: u64,
    /// Whether this check counts as passing.
    pub passed: bool,
    /// The tail of the output, for the journal and for the next attempt's brief.
    #[serde(default)]
    pub output_tail: String,
}

impl CheckOutcome {
    /// A check that did not run at all, with the reason in `output_tail`.
    ///
    /// Used for a refused command: the policy said no, so nothing was executed and the check is
    /// not passing. Inventing a passing result here would be the exact bypass the gates exist to
    /// close.
    #[must_use]
    pub fn not_run(
        id: impl Into<String>,
        command: impl Into<String>,
        required: bool,
        why: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            command: command.into(),
            required,
            exit_code: None,
            timed_out: false,
            duration_ms: 0,
            passed: false,
            output_tail: why.into(),
        }
    }

    /// The first line of why the check did not run, or an empty string when it did.
    ///
    /// The tail is the whole explanation for a refusal (`the policy refused this command: …`), but
    /// a journal record wants one line, and the first is the sentence the tool wrote.
    #[must_use]
    pub fn reason(&self) -> String {
        self.output_tail
            .lines()
            .next()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .unwrap_or("the command produced no exit code and no reason")
            .to_string()
    }

    /// One line for the journal and the brief.
    ///
    /// A check that never ran has no exit code, and "(no exit code)" alone is a lie of omission:
    /// the first live run's `check_readme` was refused by the policy before it executed, the brief
    /// said only "`check_readme` (no exit code)", and the next attempt had no way to tell a policy
    /// refusal from a failing test. It rewrote a README that was already correct.
    #[must_use]
    pub fn summary(&self) -> String {
        if self.timed_out {
            return format!("`{}` timed out after {} ms", self.id, self.duration_ms);
        }
        self.exit_code.map_or_else(
            || format!("`{}` did not run: {}", self.id, self.reason()),
            |code| format!("`{}` (exit {code})", self.id),
        )
    }
}

/// One `gate_result` journal record.
///
/// A tagged enum rather than a `json!` at the call site: `gate_result` is what a human reads to
/// answer "why did this fail", so these fields are the schema. The `gate` tag is also what makes
/// the journal queryable — `payload.gate == "check"` is the question a post-mortem asks.
///
/// Four gates, four shapes, because they observe four different things. Flattening them into one
/// record with most fields empty would make a query for "which check failed" return the reviewer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "gate", rename_all = "snake_case")]
pub enum GateRecord {
    /// The diff sanity check. Written even when it found nothing: "the diff was scanned and was
    /// clean" and "the diff was never scanned" must not look the same afterwards.
    Sanity { findings: Vec<Finding> },
    /// One command from the definition of done.
    Check {
        /// The check's id from the `DoD`.
        check: String,
        command: String,
        /// `None` when the command was killed and never reported a code.
        exit_code: Option<i32>,
        duration_ms: u64,
        passed: bool,
        required: bool,
        /// Why there is no exit code: the policy refused the command, or it could not be started.
        ///
        /// Only populated when `exit_code` is `None`, and only the first line of the reason, so a
        /// check with output does not copy its tail into every journal. Without this the record
        /// said `"exit_code": null, "passed": false` and nothing else: the first live run's
        /// `check_readme` was refused by the policy before it ran, the journal could not say so,
        /// and the brief inherited the silence.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        why: Option<String>,
    },
    /// The reviewer's verdict.
    Review { approved: bool, issues: Vec<String> },
    /// The decision the whole report adds up to.
    ///
    /// Written last, always: a journal that ends with a check and no verdict is the record of a
    /// process that died before deciding, which is exactly what recovery needs to see.
    Verdict {
        passed: bool,
        /// `None` only when the verdict passed.
        reason: Option<FailureReason>,
        /// The sentence the next attempt reads, when the verdict failed.
        why: Option<String>,
    },
}

impl GateRecord {
    /// Every record one report produces, in the order they happened.
    ///
    /// The order is the gate order — sanity, checks, review, verdict — and it is produced here
    /// rather than at the call site so the journal cannot disagree with the evaluation: one
    /// function, one ordering, one place to test it.
    #[must_use]
    pub fn of(report: &GateReport) -> Vec<Self> {
        let mut records = vec![Self::Sanity {
            findings: report.findings.clone(),
        }];
        records.extend(report.checks.iter().map(|outcome| Self::Check {
            check: outcome.id.clone(),
            command: outcome.command.clone(),
            exit_code: outcome.exit_code,
            duration_ms: outcome.duration_ms,
            passed: outcome.passed,
            required: outcome.required,
            // Only when no code came back: a check that ran has its tail in the report the
            // reviewer saw, and copying that into the journal would make every record as long as
            // the test output it came from.
            why: outcome.exit_code.is_none().then(|| outcome.reason()),
        }));
        if let Some(review) = &report.review {
            records.push(Self::Review {
                approved: review.approved(),
                issues: match review {
                    ReviewVerdict::Approve => Vec::new(),
                    ReviewVerdict::RequestChanges { issues } => issues.clone(),
                },
            });
        }
        let verdict = report.verdict();
        records.push(Self::Verdict {
            passed: verdict.is_passed(),
            reason: verdict.failure(),
            why: verdict.why().map(str::to_string),
        });
        records
    }
}

/// What the reviewer said.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum ReviewVerdict {
    /// The diff does what the definition of done says.
    Approve,
    /// The diff does not, for the listed reasons. Never empty: a rejection without a reason is a
    /// rejection the next attempt cannot act on.
    RequestChanges { issues: Vec<String> },
}

impl ReviewVerdict {
    #[must_use]
    pub const fn approved(&self) -> bool {
        matches!(self, Self::Approve)
    }
}

/// Everything the three gates observed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GateReport {
    /// Findings from the diff sanity check.
    pub findings: Vec<Finding>,
    /// Command outcomes, in the order they were run.
    pub checks: Vec<CheckOutcome>,
    /// The reviewer's verdict. `None` when the subtask's `DoD` did not ask for a review, or when
    /// an earlier gate was already red and paying for a review would have been waste.
    pub review: Option<ReviewVerdict>,
}

impl GateReport {
    /// The one answer the caller acts on.
    #[must_use]
    pub fn verdict(&self) -> GateVerdict {
        if let Some(finding) = self.findings.first() {
            return GateVerdict::Failed {
                reason: FailureReason::GateFailed,
                why: finding.observation(),
            };
        }
        if let Some(check) = self.checks.iter().find(|check| check.required && !check.passed) {
            return GateVerdict::Failed {
                reason: FailureReason::GateFailed,
                why: check.summary(),
            };
        }
        match &self.review {
            Some(ReviewVerdict::RequestChanges { issues }) => GateVerdict::Failed {
                reason: FailureReason::ReviewRejected,
                why: if issues.is_empty() {
                    "the reviewer asked for changes without naming them".to_string()
                } else {
                    issues.join("; ")
                },
            },
            _ => GateVerdict::Passed,
        }
    }

    /// Whether the subtask may close.
    #[must_use]
    pub fn is_green(&self) -> bool {
        self.verdict() == GateVerdict::Passed
    }
}

/// The gate's judgement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateVerdict {
    /// Every gate the `DoD` asked for is green.
    Passed,
    /// The subtask does not close, and the attempt failed for this reason.
    Failed { reason: FailureReason, why: String },
}

impl GateVerdict {
    /// Whether this is a green gate.
    #[must_use]
    pub const fn is_passed(&self) -> bool {
        matches!(self, Self::Passed)
    }

    /// The failure, when there is one.
    #[must_use]
    pub const fn failure(&self) -> Option<FailureReason> {
        match self {
            Self::Passed => None,
            Self::Failed { reason, .. } => Some(*reason),
        }
    }

    /// What the next attempt's brief should say about the failure.
    #[must_use]
    pub fn why(&self) -> Option<&str> {
        match self {
            Self::Passed => None,
            Self::Failed { why, .. } => Some(why),
        }
    }
}

#[cfg(test)]
mod tests;
