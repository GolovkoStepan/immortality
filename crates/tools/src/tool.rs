//! The tool trait, its context, and the registry that checks schemas at startup.
//!
//! # Why a trait with four methods and not one
//!
//! The design separates what a tool *is* (`name`, `schema`) from what a call *means*
//! (`effect`, `idempotency`) from what it *found* (`probe`). Recovery needs the middle two
//! without running anything, and the policy needs `effect` before `execute` — so a single
//! `run(args)` would force every caller to execute a tool in order to ask whether it may.
//!
//! # Why the registry validates schemas at startup
//!
//! `schema_defects` catches a `maxLenght` typo that would otherwise enforce nothing forever.
//! Checking at startup means a broken schema stops the daemon before it runs an agent, rather
//! than producing a tool that silently accepts anything. It is a programming error, not a user
//! error, so it is a hard failure.
//!
//! # Why results carry an observation string
//!
//! The model reads text. A typed result would have to be rendered anyway, and rendering at the
//! tool boundary is what lets each tool choose the shape that is clearest (`read_file` numbers
//! lines, `glob` lists paths) instead of the engine deciding for all of them. The structured
//! fields next to it are for the journal, the detectors and the human — not for the model.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use immortality_core::protocol::Idempotency;
use serde_json::Value;

use crate::args::Args;
use crate::policy::PathUse;
use crate::{Effect, PolicyGate};

/// Why a tool could not run.
///
/// Separate from a policy `Denial` on purpose: a denial is a decision the human may reverse, a
/// tool error is a bug or bad input, and the engine counts them in different places.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolError {
    /// The arguments did not fit, discovered after the schema check.
    #[error("invalid arguments: {0}")]
    Arguments(String),
    /// The filesystem refused.
    #[error("io error on {path}: {message}")]
    Io { path: String, message: String },
    /// The call cannot be carried out as written, with a message the model can act on.
    #[error("{0}")]
    Refused(String),
}

impl ToolError {
    /// An IO error with the path that failed, which is the one detail a model needs to retry.
    pub fn io(path: &Path, error: &std::io::Error) -> Self {
        Self::Io {
            path: path.display().to_string(),
            message: error.to_string(),
        }
    }
}

/// What a tool produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResult {
    /// What the model is shown.
    pub observation: String,
    /// The structured form, for the journal and for detectors. `Value::Null` when the tool has
    /// nothing to say beyond the text.
    pub data: Value,
}

impl ToolResult {
    /// A result with only text.
    #[must_use]
    pub fn text(observation: impl Into<String>) -> Self {
        Self {
            observation: observation.into(),
            data: Value::Null,
        }
    }

    /// A result with both forms.
    #[must_use]
    pub fn new(observation: impl Into<String>, data: Value) -> Self {
        Self {
            observation: observation.into(),
            data,
        }
    }
}

/// Everything a tool may touch, decided by the caller.
///
/// The gate is inside so that a tool cannot forget to check: [`ToolCtx::authorise`] is the only
/// way to get a path, and a tool that took a path straight from `Args` would be a tool that ran
/// without a check.
#[derive(Debug, Clone)]
pub struct ToolCtx {
    gate: Arc<PolicyGate>,
    budget: OutputBudget,
    call: Option<String>,
    task: Option<immortality_core::ids::TaskId>,
    depth: u8,
}

impl ToolCtx {
    /// A context for one task.
    ///
    /// No call id: a tool that needs one — `git commit` embeds it as a trailer so a crash-time
    /// probe can find its commit — gets it from [`Self::with_call`]. A default of `None` keeps
    /// the dozens of tests that only exercise the policy from having to invent an id.
    #[must_use]
    pub const fn new(gate: Arc<PolicyGate>, budget: OutputBudget) -> Self {
        Self {
            gate,
            budget,
            call: None,
            task: None,
            depth: 0,
        }
    }

    /// The same context, carrying the id of the call it is about to run.
    ///
    /// The id comes from the journal's `intent`, which is the only identity that survives a
    /// crash. Stamping it into the effect — a commit trailer, a stash message — is what lets
    /// recovery answer "did this happen?" without the arguments, which the journal does not keep.
    #[must_use]
    pub fn with_call(mut self, call: impl Into<String>) -> Self {
        self.call = Some(call.into());
        self
    }

    /// The call id, when the caller supplied one.
    #[must_use]
    pub fn call(&self) -> Option<&str> {
        self.call.as_deref()
    }

    /// The same context, carrying the task it belongs to.
    ///
    /// Most tools address their work by path and never need this. `escalate` does: a question is
    /// owned by a task, and the queue file is per task, so a call without one has nowhere to
    /// write and must be refused rather than guessed at.
    #[must_use]
    pub fn with_task(mut self, task: immortality_core::ids::TaskId) -> Self {
        self.task = Some(task);
        self
    }

    /// The task this call belongs to, when the caller supplied one.
    #[must_use]
    pub const fn task(&self) -> Option<&immortality_core::ids::TaskId> {
        self.task.as_ref()
    }

    /// The same context, marked as belonging to a sub-agent `depth` levels down.
    ///
    /// Depth is a property of the *caller*, not of the tool: the daemon builds a sub-agent's
    /// context this way and the whole registry then treats it as one. Tools that must not nest
    /// — `delegate` — read it instead of being handed a different registry, because a second
    /// registry is a second place for the rule to be forgotten.
    #[must_use]
    pub const fn with_depth(mut self, depth: u8) -> Self {
        self.depth = depth;
        self
    }

    /// How deep in the delegation tree this call sits. `0` is the top-level agent.
    #[must_use]
    pub const fn depth(&self) -> u8 {
        self.depth
    }

    /// The gate, for a tool that needs to ask a question the convenience methods do not cover.
    #[must_use]
    pub fn gate(&self) -> &PolicyGate {
        &self.gate
    }

    /// The truncation limits.
    #[must_use]
    pub const fn budget(&self) -> OutputBudget {
        self.budget
    }

    /// Check a call, returning the policy verdict.
    #[must_use]
    pub fn check(&self, effect: Effect, paths: &[PathUse]) -> crate::Verdict {
        self.gate.check(effect, paths)
    }

    /// The absolute path for a worktree-relative argument, after the gate allowed the read.
    ///
    /// # Errors
    /// A refusal, phrased as the observation the model will see. Returning the message rather
    /// than a bare error means a tool cannot accidentally turn a policy question into a crashed
    /// step.
    pub fn read_path(&self, raw: &str) -> Result<PathBuf, ToolError> {
        self.authorise(Effect::Read, &PathUse::read(raw))
    }

    /// The absolute path for a worktree-relative write, after the gate allowed it.
    ///
    /// # Errors
    /// A refusal, as with [`Self::read_path`].
    pub fn write_path(&self, raw: &str) -> Result<PathBuf, ToolError> {
        self.authorise(Effect::Mutate, &PathUse::write(raw))
    }

    /// Resolve a path that will be entered but not modified (`glob`'s base, a `grep` root).
    ///
    /// # Errors
    /// A refusal, as with [`Self::read_path`].
    pub fn enter_path(&self, raw: &str) -> Result<PathBuf, ToolError> {
        self.authorise(Effect::Read, &PathUse::read(raw))
    }

    fn authorise(&self, effect: Effect, used: &PathUse) -> Result<PathBuf, ToolError> {
        match self.gate.check(effect, std::slice::from_ref(used)) {
            // The gate's resolved path, not `worktree().join(raw)`. Resolving twice is how a
            // symlink gets judged by one answer and opened by another.
            crate::Verdict::Allowed => self.gate.resolve(&used.path).ok_or_else(|| {
                ToolError::Refused(
                    crate::Denial::new("the path could not be resolved", &used.path).observation(),
                )
            }),
            crate::Verdict::Denied(denial) => Err(ToolError::Refused(denial.observation())),
            crate::Verdict::Escalated(escalation) => Err(ToolError::Refused(escalation.observation())),
        }
    }
}

/// How much of a result may reach the model.
///
/// Truncation belongs here and not in the context assembler because a tool knows what part of
/// its output is worth keeping — the tail of a log, the head of a file — and a generic
/// "cut at 8 KiB" loses exactly the error message at the end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputBudget {
    /// Bytes of tool output that may enter a prompt.
    pub max_bytes: usize,
    /// Lines of context a `grep` may show around a match.
    pub max_context: usize,
}

impl Default for OutputBudget {
    fn default() -> Self {
        // The design's `max_tool_output_bytes`, and three lines of context either side: enough to
        // see a function's shape, small enough that ten matches stay readable.
        Self {
            max_bytes: 8192,
            max_context: 3,
        }
    }
}

/// One tool.
pub trait Tool: Send + Sync + std::fmt::Debug {
    /// The name the model calls. Stable: it appears in journals and in goldens.
    fn name(&self) -> &'static str;

    /// The JSON Schema the model was given. Checked at registration.
    fn schema(&self) -> &'static Value;

    /// A one-line description for the prompt's tool list.
    fn description(&self) -> &'static str;

    /// What the call can change.
    ///
    /// Takes the arguments because it genuinely varies: `shell` is read or mutate depending on
    /// the command, and `git` on the subcommand.
    fn effect(&self, args: &Args) -> Effect;

    /// How safe it is to run twice.
    fn idempotency(&self, args: &Args) -> Idempotency;

    /// Which paths the call touches, so the gate can check them.
    ///
    /// Declared by the tool rather than inferred by the gate: only the tool knows that
    /// `edit_file`'s `path` is a write while `read_file`'s is a read.
    fn paths(&self, args: &Args) -> Vec<PathUse>;

    /// Run it.
    ///
    /// # Errors
    /// Any condition that prevents a result. A refusal is an observation, not a crash.
    fn execute(&self, ctx: &ToolCtx, args: Args) -> Result<ToolResult, ToolError>;
}

/// Every tool the agent may call, with its schema already checked.
#[derive(Debug)]
pub struct Registry {
    tools: BTreeMap<&'static str, Arc<dyn Tool>>,
}

/// A schema that will not enforce what it says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryError {
    pub tool: String,
    pub defects: Vec<immortality_llm::schema::Defect>,
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tool `{}` has a broken schema: ", self.tool)?;
        for (index, defect) in self.defects.iter().enumerate() {
            if index > 0 {
                f.write_str("; ")?;
            }
            write!(f, "{defect}")?;
        }
        Ok(())
    }
}

impl std::error::Error for RegistryError {}

impl Registry {
    /// Register a tool, checking its schema.
    ///
    /// # Errors
    /// A defect in the schema, or a duplicate name. Both are programming errors: a duplicated
    /// name would make dispatch depend on registration order.
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Result<Self, RegistryError> {
        let mut map: BTreeMap<&'static str, Arc<dyn Tool>> = BTreeMap::new();
        for tool in tools {
            let defects = immortality_llm::schema::schema_defects(tool.schema());
            if !defects.is_empty() {
                return Err(RegistryError {
                    tool: tool.name().to_string(),
                    defects,
                });
            }
            if map.insert(tool.name(), tool).is_some() {
                return Err(RegistryError {
                    tool: "duplicate".to_string(),
                    defects: Vec::new(),
                });
            }
        }
        Ok(Self { tools: map })
    }

    /// Look a tool up by the name the model wrote.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Every tool, in a stable order.
    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn Tool>> {
        self.tools.values()
    }

    /// How many are registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// The names, sorted, for a prompt or a log line.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.tools.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema() -> &'static Value {
        static SCHEMA: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
        SCHEMA.get_or_init(|| {
            json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": false
            })
        })
    }

    #[derive(Debug)]
    struct Fake {
        name: &'static str,
        schema: &'static Value,
    }

    impl Tool for Fake {
        fn name(&self) -> &'static str {
            self.name
        }
        fn schema(&self) -> &'static Value {
            self.schema
        }
        fn description(&self) -> &'static str {
            "a fake"
        }
        fn effect(&self, _args: &Args) -> Effect {
            Effect::Read
        }
        fn idempotency(&self, _args: &Args) -> Idempotency {
            Idempotency::Idempotent
        }
        fn paths(&self, args: &Args) -> Vec<PathUse> {
            args.opt_str("path")
                .ok()
                .flatten()
                .map(|path| vec![PathUse::read(path)])
                .unwrap_or_default()
        }
        fn execute(&self, _ctx: &ToolCtx, _args: Args) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::text("ok"))
        }
    }

    fn fake(name: &'static str) -> Arc<dyn Tool> {
        Arc::new(Fake {
            name,
            schema: schema(),
        })
    }

    #[test]
    fn a_good_registry_looks_tools_up_by_name() {
        let registry = Registry::new(vec![fake("read_file"), fake("write_file")]).unwrap();
        assert_eq!(registry.len(), 2);
        assert!(!registry.is_empty());
        assert_eq!(registry.names(), vec!["read_file", "write_file"]);
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("missing").is_none());
    }

    #[test]
    fn the_order_is_stable_regardless_of_registration() {
        let first = Registry::new(vec![fake("b"), fake("a")]).unwrap();
        let second = Registry::new(vec![fake("a"), fake("b")]).unwrap();
        assert_eq!(first.names(), second.names());
        assert_eq!(first.names(), vec!["a", "b"]);
    }

    #[test]
    fn a_broken_schema_stops_registration() {
        // The whole point: a `maxLenght` typo would enforce nothing forever.
        let broken: &'static Value = Box::leak(Box::new(json!({
            "type": "object",
            "properties": {"path": {"type": "string", "maxLenght": 3}}
        })));
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(Fake {
            name: "broken",
            schema: broken,
        })];
        let error = Registry::new(tools).unwrap_err();
        assert_eq!(error.tool, "broken");
        assert!(error.to_string().contains("maxLenght"), "{error}");
    }

    #[test]
    fn a_duplicate_name_is_refused() {
        assert!(Registry::new(vec![fake("a"), fake("a")]).is_err());
    }

    #[test]
    fn an_empty_registry_is_valid() {
        let registry = Registry::new(Vec::new()).unwrap();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn the_output_budget_defaults_match_the_design() {
        let budget = OutputBudget::default();
        assert_eq!(budget.max_bytes, 8192);
        assert_eq!(budget.max_context, 3);
    }
}
