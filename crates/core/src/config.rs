//! Layered configuration.
//!
//! Precedence, weakest first:
//!
//! 1. built-in defaults
//! 2. global file (`~/Library/Application Support/immortality/config.toml`)
//! 3. project file (`<repo>/.agent/config.toml`)
//! 4. environment (`IMMORTALITY_*`)
//! 5. command-line overrides
//!
//! # The security rule
//!
//! **Security policy lives only in the global file.** The project file is inside the
//! repository, which the agent can edit, so allowing it to grant permissions would let the
//! agent widen its own rights. A `[policy]` section in the project file is therefore ignored
//! in full and reported as a warning. It cannot even narrow permissions — one rule is easier
//! to verify than two.
//!
//! Every file is parsed with `deny_unknown_fields`, so a typo or a stale key is an error
//! rather than a silent default.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

// ── effective config ────────────────────────────────────────────────────────────

/// Fully resolved configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub backend: Backend,
    pub policy: Policy,
    pub context: ContextBudget,
    pub models: ModelProfiles,
    pub loop_: LoopLimits,
    pub detectors: Detectors,
    pub logging: Logging,
    pub storage: Storage,
}

/// LLM backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Backend {
    /// OpenAI-compatible base URL, including the `/v1` suffix.
    pub base_url: String,
    pub model: String,
    /// Measured on the live backend: a 502 can arrive after 300 s of prefill on a 130K
    /// prompt, and a successful 200K prefill takes 254 s. Below 600 s is not safe.
    pub request_timeout_secs: u64,
    pub connect_timeout_secs: u64,
    /// The backend holds 4 slots, but parallel requests lower per-request throughput.
    pub max_inference_concurrency: u32,
    pub health_interval_secs: u64,
    pub unavailable_escalate_after_secs: u64,
    /// `swift-iq2_xs` emits `reasoning_content` by default and spends the whole `max_tokens`
    /// budget on it: at 400 tokens the answer came back empty with `finish_reason: length`.
    /// Disabling it removes reasoning entirely and cuts latency ~2.8x.
    pub enable_thinking: bool,
}

/// Security policy. Global file only.
///
/// Five independent permission flags is deliberate, not an accident of design: whether the
/// agent may push, touch the network, use sudo or install dependencies are separate decisions
/// a human must be able to answer one at a time. A bitmask would hide that in a single number.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    /// Root under which projects may live. Writes outside any worktree are denied.
    pub workspace_root: PathBuf,
    /// Hosts reachable without an escalation. Must include the model backend.
    pub trusted_hosts: BTreeSet<String>,
    /// Paths inside the task worktree a write may land on.
    ///
    /// A worktree-relative glob list, and the reason it defaults to `**`: the worktree is already
    /// the isolation boundary, so the allow list's job is to *narrow* it for an operator who wants
    /// that, not to enumerate every file the agent may touch.
    pub allow_write_globs: Vec<String>,
    /// Paths no write may land on, checked before the allow list.
    ///
    /// Denies win over allows, always. A list that could be overridden by adding a broader allow
    /// pattern would be a suggestion rather than a rule.
    pub deny_write_globs: Vec<String>,
    pub allow_commit: bool,
    pub allow_push: bool,
    pub allow_network: bool,
    pub allow_sudo: bool,
    pub allow_install_deps: bool,
}

/// Context budget. The server window is 256K, but the working budget is deliberately small:
/// a cold 64K prefill measured 166 s, and quality degrades long before the window fills.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextBudget {
    pub server_window_tokens: u32,
    pub soft_budget_tokens: u32,
    pub hard_budget_tokens: u32,
    pub max_tool_output_bytes: usize,
}

/// Model profiles. All three may point at the same model; the split exists so that a second
/// backend can be introduced by editing config, not code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProfiles {
    pub planner: String,
    pub executor: String,
    pub utility: String,
}

/// Loop limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct LoopLimits {
    pub max_steps_per_attempt: u32,
    pub max_tokens_per_attempt: u32,
    pub max_attempts_per_subtask: u32,
    pub max_restarts_per_subtask: u32,
    pub max_subtask_depth: u8,
    pub stall_steps: u32,
}

/// Detector thresholds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Detectors {
    pub exact_repeat_warn: u32,
    pub exact_repeat_strategy: u32,
    pub exact_repeat_stop: u32,
    pub progress_stall_warn: u32,
    pub progress_stall_stop: u32,
    pub reasoning_repeat_ngram: u32,
}

/// Logging.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Logging {
    pub level: String,
    pub dir: PathBuf,
}

/// Data growth limits, enforced by `gc` and the disk guard.
///
/// Split from [`Logging`] because these are about disk, not about verbosity: an operator tuning
/// log levels should not be able to shorten the artifact TTL by accident.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct Storage {
    /// Days an artifact with no live reference is kept before `gc` may delete it.
    pub artifact_ttl_days: u32,
    /// Months of compressed logs to keep. The current and previous month are never deleted.
    pub log_retention_months: u32,
    /// Free space below this is a warning.
    pub disk_warn_bytes: u64,
    /// Free space below this pauses the task: the next write would fail.
    pub disk_pause_bytes: u64,
}

impl Default for Backend {
    fn default() -> Self {
        Self {
            base_url: "http://llm-server:47832/v1".into(),
            model: "swift-iq2_xs".into(),
            request_timeout_secs: 600,
            connect_timeout_secs: 10,
            max_inference_concurrency: 1,
            health_interval_secs: 30,
            unavailable_escalate_after_secs: 900,
            enable_thinking: false,
        }
    }
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            workspace_root: PathBuf::from("~/rust_projects"),
            trusted_hosts: BTreeSet::from([
                "llm-server".to_string(),
                "localhost".to_string(),
                "127.0.0.1".to_string(),
            ]),
            allow_write_globs: vec!["**".to_string()],
            deny_write_globs: vec!["**/.git/config".to_string(), "**/.env*".to_string()],
            allow_commit: true,
            allow_push: false,
            allow_network: false,
            allow_sudo: false,
            allow_install_deps: true,
        }
    }
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            server_window_tokens: 262_144,
            soft_budget_tokens: 32_000,
            hard_budget_tokens: 64_000,
            max_tool_output_bytes: 8192,
        }
    }
}

impl Default for ModelProfiles {
    fn default() -> Self {
        Self {
            planner: "swift-iq2_xs".into(),
            executor: "swift-iq2_xs".into(),
            utility: "swift-iq2_xs".into(),
        }
    }
}

impl Default for LoopLimits {
    fn default() -> Self {
        Self {
            max_steps_per_attempt: 60,
            max_tokens_per_attempt: 120_000,
            max_attempts_per_subtask: 3,
            max_restarts_per_subtask: 2,
            max_subtask_depth: 2,
            stall_steps: 15,
        }
    }
}

impl Default for Detectors {
    fn default() -> Self {
        Self {
            exact_repeat_warn: 3,
            exact_repeat_strategy: 5,
            exact_repeat_stop: 8,
            progress_stall_warn: 15,
            progress_stall_stop: 25,
            reasoning_repeat_ngram: 40,
        }
    }
}

impl Default for Logging {
    fn default() -> Self {
        Self {
            level: "info".into(),
            dir: PathBuf::from("~/Library/Logs/immortality"),
        }
    }
}

impl Default for Storage {
    fn default() -> Self {
        Self {
            artifact_ttl_days: 30,
            log_retention_months: 12,
            // 1 GiB / 256 MiB: the pause threshold is low enough that a single tool output or a
            // checkpoint cannot be the write that fills the disk, and high enough that a normal
            // build or a git operation still fits.
            disk_warn_bytes: 1024 * 1024 * 1024,
            disk_pause_bytes: 256 * 1024 * 1024,
        }
    }
}

// ── raw (partial) file shapes ───────────────────────────────────────────────────

/// Shape of a TOML config file; every field optional so files may be small.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawConfig {
    pub backend: Option<RawBackend>,
    pub policy: Option<RawPolicy>,
    pub context: Option<RawContext>,
    pub models: Option<RawModels>,
    #[serde(rename = "loop")]
    pub loop_: Option<RawLoop>,
    pub detectors: Option<RawDetectors>,
    pub logging: Option<RawLogging>,
    pub storage: Option<RawStorage>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawBackend {
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub request_timeout_secs: Option<u64>,
    pub connect_timeout_secs: Option<u64>,
    pub max_inference_concurrency: Option<u32>,
    pub health_interval_secs: Option<u64>,
    pub unavailable_escalate_after_secs: Option<u64>,
    pub enable_thinking: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawPolicy {
    pub workspace_root: Option<PathBuf>,
    pub trusted_hosts: Option<Vec<String>>,
    pub allow_write_globs: Option<Vec<String>>,
    pub deny_write_globs: Option<Vec<String>>,
    pub allow_commit: Option<bool>,
    pub allow_push: Option<bool>,
    pub allow_network: Option<bool>,
    pub allow_sudo: Option<bool>,
    pub allow_install_deps: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawContext {
    pub server_window_tokens: Option<u32>,
    pub soft_budget_tokens: Option<u32>,
    pub hard_budget_tokens: Option<u32>,
    pub max_tool_output_bytes: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawModels {
    pub planner: Option<String>,
    pub executor: Option<String>,
    pub utility: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawLoop {
    pub max_steps_per_attempt: Option<u32>,
    pub max_tokens_per_attempt: Option<u32>,
    pub max_attempts_per_subtask: Option<u32>,
    pub max_restarts_per_subtask: Option<u32>,
    pub max_subtask_depth: Option<u8>,
    pub stall_steps: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawDetectors {
    pub exact_repeat_warn: Option<u32>,
    pub exact_repeat_strategy: Option<u32>,
    pub exact_repeat_stop: Option<u32>,
    pub progress_stall_warn: Option<u32>,
    pub progress_stall_stop: Option<u32>,
    pub reasoning_repeat_ngram: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawLogging {
    pub level: Option<String>,
    pub dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawStorage {
    pub artifact_ttl_days: Option<u32>,
    pub log_retention_months: Option<u32>,
    pub disk_warn_bytes: Option<u64>,
    pub disk_pause_bytes: Option<u64>,
}

// ── command-line overrides ──────────────────────────────────────────────────────

/// Highest-precedence overrides.
#[derive(Debug, Clone, Default)]
pub struct Overrides {
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub log_level: Option<String>,
    pub global_config: Option<PathBuf>,
    pub project_config: Option<PathBuf>,
}

// ── loading ────────────────────────────────────────────────────────────────────

/// Result of loading: the effective config plus everything the user should hear about.
#[derive(Debug, Clone)]
pub struct Loaded {
    pub config: Config,
    /// Non-fatal problems: ignored `[policy]` in the project file, unreadable optional file.
    pub warnings: Vec<String>,
    pub global_path: PathBuf,
    pub project_path: Option<PathBuf>,
}

/// Load and merge configuration.
///
/// `global_path` is where security policy is read from. `project_path` must be inside a
/// repository; if it does not exist the layer is simply skipped.
pub fn load(overrides: &Overrides) -> Result<Loaded> {
    let mut warnings = Vec::new();
    let mut config = Config::default();

    let global_path = overrides
        .global_config
        .clone()
        .unwrap_or_else(default_global_path);
    let project_path = overrides.project_config.clone().or_else(discover_project_config);

    // Layer 2: global file. Security policy is read only here.
    if global_path.exists() {
        let mut raw = read_file(&global_path)?;
        if let Some(policy) = raw.policy.take() {
            apply_policy(&mut config.policy, policy);
        }
        apply_non_policy(&mut config, raw);
    } else {
        warnings.push(format!(
            "global config not found at {}; using built-in policy defaults",
            global_path.display()
        ));
    }

    // Expand `~` now that the values are settled, so downstream code never sees a tilde.
    config.policy.workspace_root = expand_tilde(&config.policy.workspace_root);
    config.logging.dir = expand_tilde(&config.logging.dir);

    // Layer 3: project file. Never allowed to grant permissions.
    if let Some(path) = &project_path
        && path.exists()
    {
        let raw = read_file(path)?;
        if raw.policy.is_some() {
            warnings.push(format!(
                "{}: `[policy]` is ignored; security policy can only be set in {}. \
                 Remove the section to silence this warning.",
                path.display(),
                global_path.display()
            ));
        }
        apply_non_policy(&mut config, raw);
    }

    // Layer 4: environment.
    apply_env(&mut config);

    // Layer 5: command line.
    if let Some(value) = &overrides.base_url {
        config.backend.base_url.clone_from(value);
    }
    if let Some(value) = &overrides.model {
        config.backend.model.clone_from(value);
        config.models = ModelProfiles {
            planner: value.clone(),
            executor: value.clone(),
            utility: value.clone(),
        };
    }
    if let Some(value) = &overrides.log_level {
        config.logging.level.clone_from(value);
    }

    validate(&config)?;

    Ok(Loaded {
        config,
        warnings,
        global_path,
        project_path,
    })
}

fn default_global_path() -> PathBuf {
    crate::paths::StatePaths::discover().global_config()
}

fn discover_project_config() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join(".agent").join("config.toml");
        if candidate.exists() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn read_file(path: &Path) -> Result<RawConfig> {
    let text = std::fs::read_to_string(path).map_err(|source| Error::ConfigIo {
        path: path.into(),
        source,
    })?;
    toml::from_str(&text).map_err(|source| Error::ConfigToml {
        path: path.into(),
        source,
    })
}

fn apply_policy(target: &mut Policy, raw: RawPolicy) {
    if let Some(value) = raw.workspace_root {
        target.workspace_root = value;
    }
    if let Some(value) = raw.trusted_hosts {
        target.trusted_hosts = value.into_iter().collect();
    }
    if let Some(value) = raw.allow_write_globs {
        target.allow_write_globs = value;
    }
    if let Some(value) = raw.deny_write_globs {
        target.deny_write_globs = value;
    }
    target.allow_commit = raw.allow_commit.unwrap_or(target.allow_commit);
    target.allow_push = raw.allow_push.unwrap_or(target.allow_push);
    target.allow_network = raw.allow_network.unwrap_or(target.allow_network);
    target.allow_sudo = raw.allow_sudo.unwrap_or(target.allow_sudo);
    target.allow_install_deps = raw.allow_install_deps.unwrap_or(target.allow_install_deps);
}

fn apply_non_policy(config: &mut Config, raw: RawConfig) {
    if let Some(raw) = raw.backend {
        let target = &mut config.backend;
        if let Some(v) = raw.base_url {
            target.base_url = v;
        }
        if let Some(v) = raw.model {
            target.model = v;
        }
        target.request_timeout_secs = raw.request_timeout_secs.unwrap_or(target.request_timeout_secs);
        target.connect_timeout_secs = raw.connect_timeout_secs.unwrap_or(target.connect_timeout_secs);
        target.max_inference_concurrency = raw
            .max_inference_concurrency
            .unwrap_or(target.max_inference_concurrency);
        target.health_interval_secs = raw.health_interval_secs.unwrap_or(target.health_interval_secs);
        target.unavailable_escalate_after_secs = raw
            .unavailable_escalate_after_secs
            .unwrap_or(target.unavailable_escalate_after_secs);
        target.enable_thinking = raw.enable_thinking.unwrap_or(target.enable_thinking);
    }
    if let Some(raw) = raw.context {
        let target = &mut config.context;
        target.server_window_tokens = raw.server_window_tokens.unwrap_or(target.server_window_tokens);
        target.soft_budget_tokens = raw.soft_budget_tokens.unwrap_or(target.soft_budget_tokens);
        target.hard_budget_tokens = raw.hard_budget_tokens.unwrap_or(target.hard_budget_tokens);
        target.max_tool_output_bytes = raw.max_tool_output_bytes.unwrap_or(target.max_tool_output_bytes);
    }
    if let Some(raw) = raw.models {
        let target = &mut config.models;
        if let Some(v) = raw.planner {
            target.planner = v;
        }
        if let Some(v) = raw.executor {
            target.executor = v;
        }
        if let Some(v) = raw.utility {
            target.utility = v;
        }
    }
    if let Some(raw) = raw.loop_ {
        let target = &mut config.loop_;
        target.max_steps_per_attempt = raw.max_steps_per_attempt.unwrap_or(target.max_steps_per_attempt);
        target.max_tokens_per_attempt = raw
            .max_tokens_per_attempt
            .unwrap_or(target.max_tokens_per_attempt);
        target.max_attempts_per_subtask = raw
            .max_attempts_per_subtask
            .unwrap_or(target.max_attempts_per_subtask);
        target.max_restarts_per_subtask = raw
            .max_restarts_per_subtask
            .unwrap_or(target.max_restarts_per_subtask);
        target.max_subtask_depth = raw.max_subtask_depth.unwrap_or(target.max_subtask_depth);
        target.stall_steps = raw.stall_steps.unwrap_or(target.stall_steps);
    }
    if let Some(raw) = raw.detectors {
        let target = &mut config.detectors;
        target.exact_repeat_warn = raw.exact_repeat_warn.unwrap_or(target.exact_repeat_warn);
        target.exact_repeat_strategy = raw.exact_repeat_strategy.unwrap_or(target.exact_repeat_strategy);
        target.exact_repeat_stop = raw.exact_repeat_stop.unwrap_or(target.exact_repeat_stop);
        target.progress_stall_warn = raw.progress_stall_warn.unwrap_or(target.progress_stall_warn);
        target.progress_stall_stop = raw.progress_stall_stop.unwrap_or(target.progress_stall_stop);
        target.reasoning_repeat_ngram = raw
            .reasoning_repeat_ngram
            .unwrap_or(target.reasoning_repeat_ngram);
    }
    if let Some(raw) = raw.logging {
        let target = &mut config.logging;
        if let Some(v) = raw.level {
            target.level = v;
        }
        if let Some(v) = raw.dir {
            target.dir = v;
        }
    }
    if let Some(raw) = raw.storage {
        let target = &mut config.storage;
        target.artifact_ttl_days = raw.artifact_ttl_days.unwrap_or(target.artifact_ttl_days);
        target.log_retention_months = raw.log_retention_months.unwrap_or(target.log_retention_months);
        target.disk_warn_bytes = raw.disk_warn_bytes.unwrap_or(target.disk_warn_bytes);
        target.disk_pause_bytes = raw.disk_pause_bytes.unwrap_or(target.disk_pause_bytes);
    }
}

fn apply_env(config: &mut Config) {
    if let Ok(value) = std::env::var("IMMORTALITY_BASE_URL") {
        config.backend.base_url = value;
    }
    if let Ok(value) = std::env::var("IMMORTALITY_MODEL") {
        config.backend.model.clone_from(&value);
        config.models.planner.clone_from(&value);
        config.models.executor.clone_from(&value);
        config.models.utility = value;
    }
    if let Ok(value) = std::env::var("IMMORTALITY_LOG_LEVEL") {
        config.logging.level = value;
    }
    apply_env_u32(
        "IMMORTALITY_SOFT_BUDGET_TOKENS",
        &mut config.context.soft_budget_tokens,
    );
    apply_env_u32(
        "IMMORTALITY_HARD_BUDGET_TOKENS",
        &mut config.context.hard_budget_tokens,
    );
}

/// Apply a numeric environment override, ignoring values that do not parse.
///
/// Ignoring rather than failing is deliberate: an environment variable is ambient and often
/// stale in a shell profile, so a bad value must not make every command unusable.
fn apply_env_u32(name: &str, target: &mut u32) {
    if let Ok(value) = std::env::var(name)
        && let Ok(parsed) = value.parse()
    {
        *target = parsed;
    }
}

fn validate(config: &Config) -> Result<()> {
    if !config.backend.base_url.starts_with("http://") && !config.backend.base_url.starts_with("https://") {
        return Err(Error::ConfigInvalid(format!(
            "backend.base_url must start with http:// or https://, got `{}`",
            config.backend.base_url
        )));
    }
    if config.backend.base_url.ends_with('/') {
        return Err(Error::ConfigInvalid(
            "backend.base_url must not end with `/` (it is joined with `/chat/completions`)".into(),
        ));
    }
    if config.context.soft_budget_tokens >= config.context.hard_budget_tokens {
        return Err(Error::ConfigInvalid(format!(
            "context.soft_budget_tokens ({}) must be smaller than hard_budget_tokens ({})",
            config.context.soft_budget_tokens, config.context.hard_budget_tokens
        )));
    }
    if config.context.hard_budget_tokens > config.context.server_window_tokens {
        return Err(Error::ConfigInvalid(format!(
            "context.hard_budget_tokens ({}) exceeds the server window ({}); \
             compaction could never fit",
            config.context.hard_budget_tokens, config.context.server_window_tokens
        )));
    }
    if config.backend.max_inference_concurrency == 0 {
        return Err(Error::ConfigInvalid(
            "backend.max_inference_concurrency must be at least 1".into(),
        ));
    }
    if config.loop_.max_subtask_depth > 3 {
        return Err(Error::ConfigInvalid(
            "loop.max_subtask_depth above 3 is not supported: context quality collapses".into(),
        ));
    }
    if config.storage.disk_pause_bytes >= config.storage.disk_warn_bytes {
        return Err(Error::ConfigInvalid(format!(
            "storage.disk_pause_bytes ({}) must be smaller than disk_warn_bytes ({})",
            config.storage.disk_pause_bytes, config.storage.disk_warn_bytes
        )));
    }
    if config.storage.artifact_ttl_days == 0 {
        return Err(Error::ConfigInvalid(
            "storage.artifact_ttl_days must be at least 1: a zero TTL would delete artifacts \
             the next step is about to read"
                .into(),
        ));
    }
    if config.storage.log_retention_months < 2 {
        return Err(Error::ConfigInvalid(
            "storage.log_retention_months must be at least 2: the previous month is still being \
             written when the new one starts"
                .into(),
        ));
    }
    // The backend must be reachable without an escalation, otherwise every model call
    // becomes a human approval.
    let host = host_of(&config.backend.base_url);
    if let Some(host) = host
        && !config.policy.trusted_hosts.contains(&host)
    {
        return Err(Error::ConfigInvalid(format!(
            "backend host `{host}` is not in policy.trusted_hosts; every model call would \
             require approval. Add it to the global config's [policy].trusted_hosts."
        )));
    }
    Ok(())
}

/// Extract the host portion of a URL without pulling in a URL parser.
fn host_of(url: &str) -> Option<String> {
    let rest = url.split_once("://")?.1;
    let authority = rest.split('/').next()?;
    // Strip userinfo and port.
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let host = host.split_once(':').map_or(host, |(h, _)| h);
    if host.is_empty() {
        None
    } else {
        Some(host.to_string())
    }
}

fn expand_tilde(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    let Some(rest) = text.strip_prefix("~/") else {
        return path.to_path_buf();
    };
    std::env::var_os("HOME").map_or_else(|| path.to_path_buf(), |home| PathBuf::from(home).join(rest))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        _dir: tempfile::TempDir,
        global: PathBuf,
        project: PathBuf,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let global = dir.path().join("global.toml");
        let project = dir.path().join(".agent").join("config.toml");
        std::fs::create_dir_all(project.parent().unwrap()).unwrap();
        Fixture {
            _dir: dir,
            global,
            project,
        }
    }

    fn overrides_for(fx: &Fixture) -> Overrides {
        Overrides {
            global_config: Some(fx.global.clone()),
            project_config: Some(fx.project.clone()),
            ..Default::default()
        }
    }

    #[test]
    fn defaults_are_valid() {
        validate(&Config::default()).unwrap();
    }

    #[test]
    fn storage_defaults_match_the_documented_thresholds() {
        let storage = Storage::default();
        assert_eq!(storage.artifact_ttl_days, 30);
        assert_eq!(storage.log_retention_months, 12);
        assert_eq!(storage.disk_warn_bytes, 1024 * 1024 * 1024);
        assert_eq!(storage.disk_pause_bytes, 256 * 1024 * 1024);
    }

    #[test]
    fn storage_is_read_from_the_project_file() {
        // Unlike `[policy]`, `[storage]` is a tuning knob, not a permission: the project may set
        // it, and a shorter TTL there is a developer's choice, not a privilege escalation.
        let fx = fixture();
        std::fs::write(&fx.global, "").unwrap();
        std::fs::write(&fx.project, "[storage]\nartifact_ttl_days = 3\n").unwrap();
        let loaded = load(&overrides_for(&fx)).unwrap();
        assert_eq!(loaded.config.storage.artifact_ttl_days, 3);
        assert_eq!(loaded.config.storage.log_retention_months, 12);
        assert!(
            !loaded.warnings.iter().any(|w| w.contains("[policy]")),
            "a `[storage]` section must not be mistaken for `[policy]`"
        );
    }

    #[test]
    fn a_zero_artifact_ttl_is_rejected() {
        let fx = fixture();
        std::fs::write(&fx.global, "").unwrap();
        std::fs::write(&fx.project, "[storage]\nartifact_ttl_days = 0\n").unwrap();
        let error = load(&overrides_for(&fx)).unwrap_err();
        assert!(error.to_string().contains("artifact_ttl_days"), "{error}");
    }

    #[test]
    fn a_pause_threshold_at_or_above_the_warning_is_rejected() {
        // Equal thresholds would mean the warning and the pause fire at the same moment, and the
        // operator would never see the warning before work stopped.
        let fx = fixture();
        std::fs::write(&fx.global, "").unwrap();
        std::fs::write(
            &fx.project,
            "[storage]\ndisk_warn_bytes = 1000\ndisk_pause_bytes = 1000\n",
        )
        .unwrap();
        let error = load(&overrides_for(&fx)).unwrap_err();
        assert!(error.to_string().contains("disk_pause_bytes"), "{error}");
    }

    #[test]
    fn a_log_retention_shorter_than_two_months_is_rejected() {
        let fx = fixture();
        std::fs::write(&fx.global, "").unwrap();
        std::fs::write(&fx.project, "[storage]\nlog_retention_months = 1\n").unwrap();
        let error = load(&overrides_for(&fx)).unwrap_err();
        assert!(error.to_string().contains("log_retention_months"), "{error}");
    }

    #[test]
    fn project_file_cannot_grant_push() {
        let fx = fixture();
        std::fs::write(&fx.global, "[policy]\nallow_push = false\n").unwrap();
        std::fs::write(&fx.project, "[policy]\nallow_push = true\n").unwrap();

        let loaded = load(&overrides_for(&fx)).unwrap();
        assert!(
            !loaded.config.policy.allow_push,
            "project must not be able to grant push"
        );
        assert!(
            loaded.warnings.iter().any(|w| w.contains("[policy]` is ignored")),
            "expected a warning, got {:?}",
            loaded.warnings
        );
    }

    #[test]
    fn project_file_cannot_widen_the_write_allow_list() {
        // The allow list is a permission, so it lives in the global file only. A project file that
        // could add `**` would be a project file that can grant itself the right to write
        // anywhere in the worktree - including `.git/hooks`.
        let fx = fixture();
        std::fs::write(&fx.global, "[policy]\nallow_write_globs = [\"src/**\"]\n").unwrap();
        std::fs::write(
            &fx.project,
            "[policy]\nallow_write_globs = [\"**\"]\ndeny_write_globs = []\n",
        )
        .unwrap();

        let loaded = load(&overrides_for(&fx)).unwrap();
        assert_eq!(loaded.config.policy.allow_write_globs, vec!["src/**".to_string()]);
        assert_eq!(
            loaded.config.policy.deny_write_globs,
            vec!["**/.git/config".to_string(), "**/.env*".to_string()],
            "the project file must not be able to empty the deny list"
        );
    }

    #[test]
    fn project_file_cannot_grant_network_or_sudo() {
        let fx = fixture();
        std::fs::write(&fx.global, "").unwrap();
        std::fs::write(
            &fx.project,
            "[policy]\nallow_network = true\nallow_sudo = true\ntrusted_hosts = [\"evil.example\"]\n",
        )
        .unwrap();

        let loaded = load(&overrides_for(&fx)).unwrap();
        assert!(!loaded.config.policy.allow_network);
        assert!(!loaded.config.policy.allow_sudo);
        assert!(!loaded.config.policy.trusted_hosts.contains("evil.example"));
    }

    #[test]
    fn global_file_sets_policy() {
        let fx = fixture();
        std::fs::write(
            &fx.global,
            "[policy]\nallow_push = true\nallow_sudo = true\ntrusted_hosts = [\"llm-server\", \"example.test\"]\n",
        )
        .unwrap();

        let loaded = load(&overrides_for(&fx)).unwrap();
        assert!(loaded.config.policy.allow_push);
        assert!(loaded.config.policy.allow_sudo);
        assert!(loaded.config.policy.trusted_hosts.contains("example.test"));
    }

    #[test]
    fn project_overrides_non_policy_values() {
        let fx = fixture();
        std::fs::write(&fx.global, "[context]\nsoft_budget_tokens = 20000\n").unwrap();
        std::fs::write(
            &fx.project,
            "[context]\nsoft_budget_tokens = 16000\n\n[loop]\nmax_steps_per_attempt = 25\n",
        )
        .unwrap();

        let loaded = load(&overrides_for(&fx)).unwrap();
        assert_eq!(loaded.config.context.soft_budget_tokens, 16_000);
        assert_eq!(loaded.config.loop_.max_steps_per_attempt, 25);
    }

    #[test]
    fn unknown_key_is_an_error_not_a_silent_default() {
        let fx = fixture();
        std::fs::write(&fx.project, "[context]\nsoft_budgets_tokens = 1000\n").unwrap();
        let err = load(&overrides_for(&fx)).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("soft_budgets_tokens"),
            "error should name the bad key: {text}"
        );
    }

    #[test]
    fn unknown_section_is_an_error() {
        let fx = fixture();
        std::fs::write(&fx.project, "[policies]\nallow_push = true\n").unwrap();
        assert!(load(&overrides_for(&fx)).is_err());
    }

    #[test]
    fn missing_global_file_warns_but_loads() {
        let fx = fixture();
        let loaded = load(&overrides_for(&fx)).unwrap();
        assert!(
            loaded
                .warnings
                .iter()
                .any(|w| w.contains("global config not found"))
        );
    }

    #[test]
    fn cli_overrides_beat_everything() {
        let fx = fixture();
        std::fs::write(&fx.global, "[backend]\nmodel = \"from-file\"\n").unwrap();
        std::fs::write(&fx.project, "[backend]\nmodel = \"from-project\"\n").unwrap();

        let mut overrides = overrides_for(&fx);
        overrides.model = Some("from-cli".into());
        let loaded = load(&overrides).unwrap();

        assert_eq!(loaded.config.backend.model, "from-cli");
        assert_eq!(
            loaded.config.models.executor, "from-cli",
            "model override must apply to all profiles"
        );
    }

    #[test]
    fn backend_host_must_be_trusted() {
        let fx = fixture();
        std::fs::write(
            &fx.global,
            "[backend]\nbase_url = \"http://untrusted.example:1234/v1\"\n[policy]\ntrusted_hosts = [\"llm-server\"]\n",
        )
        .unwrap();
        let err = load(&overrides_for(&fx)).unwrap_err().to_string();
        assert!(err.contains("untrusted.example"), "{err}");
        assert!(err.contains("trusted_hosts"), "{err}");
    }

    #[test]
    fn soft_budget_must_be_below_hard() {
        let fx = fixture();
        std::fs::write(
            &fx.project,
            "[context]\nsoft_budget_tokens = 100000\nhard_budget_tokens = 100000\n",
        )
        .unwrap();
        let err = load(&overrides_for(&fx)).unwrap_err().to_string();
        assert!(err.contains("must be smaller"), "{err}");
    }

    #[test]
    fn hard_budget_cannot_exceed_server_window() {
        let fx = fixture();
        std::fs::write(
            &fx.project,
            "[context]\nserver_window_tokens = 8000\nhard_budget_tokens = 9000\nsoft_budget_tokens = 4000\n",
        )
        .unwrap();
        let err = load(&overrides_for(&fx)).unwrap_err().to_string();
        assert!(err.contains("exceeds the server window"), "{err}");
    }

    #[test]
    fn base_url_must_not_end_with_slash() {
        let fx = fixture();
        std::fs::write(
            &fx.global,
            "[backend]\nbase_url = \"http://llm-server:47832/v1/\"\n",
        )
        .unwrap();
        let err = load(&overrides_for(&fx)).unwrap_err().to_string();
        assert!(err.contains("must not end with"), "{err}");
    }

    #[test]
    fn host_extraction_handles_port_and_userinfo() {
        assert_eq!(
            host_of("http://llm-server:47832/v1").as_deref(),
            Some("llm-server")
        );
        assert_eq!(
            host_of("https://user:pass@host.example/v1").as_deref(),
            Some("host.example")
        );
        assert_eq!(host_of("http://localhost/v1").as_deref(), Some("localhost"));
        assert_eq!(host_of("not a url"), None);
    }

    #[test]
    fn tilde_is_expanded_for_policy_and_logs() {
        let fx = fixture();
        std::fs::write(&fx.global, "[policy]\nworkspace_root = \"~/work\"\n").unwrap();
        let loaded = load(&overrides_for(&fx)).unwrap();
        let home = std::env::var("HOME").unwrap();
        assert_eq!(
            loaded.config.policy.workspace_root,
            PathBuf::from(home).join("work")
        );
    }

    #[test]
    fn config_roundtrips_through_toml() {
        let config = Config::default();
        let text = toml::to_string(&config).unwrap();
        let back: Config = toml::from_str(&text).unwrap();
        assert_eq!(config, back);
    }

    #[test]
    fn loop_section_uses_toml_key_name() {
        // `loop` is a Rust keyword; the file must still say `[loop]`.
        let fx = fixture();
        std::fs::write(&fx.project, "[loop]\nstall_steps = 7\n").unwrap();
        let loaded = load(&overrides_for(&fx)).unwrap();
        assert_eq!(loaded.config.loop_.stall_steps, 7);
    }
}
