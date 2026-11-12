//! Environment and backend diagnostics behind `immortality doctor`.
//!
//! Why this exists in M0 rather than later: on this project almost every confusing failure
//! is environmental. The backend is on another machine, the prompt cache decides whether a
//! step costs one second or ninety, and a missing trusted host turns every model call into a
//! human approval. `doctor` is the difference between "it hangs" and "the host does not
//! resolve".
//!
//! Each check is independent and never aborts the run: the point is a full picture.

use std::path::Path;
use std::time::{Duration, Instant};

use immortality_core::config::Config;
use serde::Serialize;

/// Outcome of a single check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Ok,
    Warn,
    Fail,
    Skipped,
}

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub name: &'static str,
    pub status: Status,
    pub detail: String,
}

impl Check {
    fn ok(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            status: Status::Ok,
            detail: detail.into(),
        }
    }
    fn warn(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            status: Status::Warn,
            detail: detail.into(),
        }
    }
    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            status: Status::Fail,
            detail: detail.into(),
        }
    }
    fn skipped(name: &'static str, reason: impl Into<String>) -> Self {
        Self {
            name,
            status: Status::Skipped,
            detail: reason.into(),
        }
    }
}

/// Full diagnostic run.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub checks: Vec<Check>,
}

impl Report {
    /// Exit code convention: 0 when nothing failed, 1 when at least one check failed.
    /// Warnings do not fail the run — they are advice, not breakage.
    #[must_use]
    pub fn has_failures(&self) -> bool {
        self.checks.iter().any(|c| c.status == Status::Fail)
    }

    #[must_use]
    pub fn exit_code(&self) -> i32 {
        i32::from(self.has_failures())
    }
}

/// Inputs that come from outside the config.
#[derive(Debug)]
pub struct Context<'a> {
    pub config: &'a Config,
    pub config_warnings: &'a [String],
    pub global_config_path: &'a Path,
    /// Daemon socket, if one is expected to exist.
    pub socket_path: Option<&'a Path>,
}

/// Run every check. Network checks time out quickly; a dead backend must not stall `doctor`.
#[allow(clippy::too_many_lines)]
pub async fn run(context: &Context<'_>) -> Report {
    let mut checks = Vec::new();

    checks.push(config_check(context));
    checks.push(trusted_host_check(context.config));
    checks.extend(backend_checks(context.config).await);
    checks.push(disk_check(context.config));
    checks.push(log_dir_check(context.config));
    checks.push(socket_check(context.socket_path));

    Report { checks }
}

fn config_check(context: &Context<'_>) -> Check {
    if context.config_warnings.is_empty() {
        Check::ok(
            "config",
            format!("loaded from {}", context.global_config_path.display()),
        )
    } else {
        Check::warn(
            "config",
            format!(
                "{} warning(s): {}",
                context.config_warnings.len(),
                context.config_warnings.join(" | ")
            ),
        )
    }
}

fn trusted_host_check(config: &Config) -> Check {
    let Some(host) = host_of(&config.backend.base_url) else {
        return Check::fail(
            "trusted-host",
            format!("cannot parse host from `{}`", config.backend.base_url),
        );
    };
    if config.policy.trusted_hosts.contains(&host) {
        Check::ok(
            "trusted-host",
            format!("`{host}` is trusted; model calls need no approval"),
        )
    } else {
        Check::fail(
            "trusted-host",
            format!("`{host}` is NOT in policy.trusted_hosts; every model call would need approval"),
        )
    }
}

async fn backend_checks(config: &Config) -> Vec<Check> {
    let Ok(client) = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(config.backend.connect_timeout_secs))
        // Short by design: a dead backend must not stall `doctor` for the agent's 600 s.
        .timeout(Duration::from_secs(30))
        .build()
    else {
        return vec![Check::fail("backend-http", "cannot build the HTTP client")];
    };

    let mut out = models_checks(&client, config).await;
    // Only measure latency if the backend answered at all; a second timeout on a dead
    // host would double the wait for no extra information.
    if out.iter().all(|check| check.status != Status::Fail) {
        out.extend(latency_checks(&client, config).await);
    } else {
        out.push(Check::skipped("backend-latency", "backend unreachable"));
        out.push(Check::skipped("backend-cache", "backend unreachable"));
    }
    out
}

/// `GET /models`: is anything listening, and does it serve the configured model?
async fn models_checks(client: &reqwest::Client, config: &Config) -> Vec<Check> {
    let base = &config.backend.base_url;
    let started = Instant::now();
    let response = client.get(format!("{base}/models")).send().await;
    let response = match response {
        Ok(response) if response.status().is_success() => response,
        Ok(response) => {
            let status = response.status();
            return vec![Check::fail(
                "backend-models",
                format!("`{base}/models` returned HTTP {status}"),
            )];
        }
        Err(err) => {
            return vec![Check::fail(
                "backend-models",
                format!(
                    "cannot reach `{base}/models` after {} ms: {err:#}",
                    started.elapsed().as_millis()
                ),
            )];
        }
    };
    let list: ModelsResponse = match response.json().await {
        Ok(list) => list,
        Err(err) => {
            return vec![Check::fail(
                "backend-models",
                format!("unreadable `/models` response: {err}"),
            )];
        }
    };

    let names: Vec<&str> = list.data.iter().map(|m| m.id.as_str()).collect();
    let served = if names.is_empty() {
        Check::fail("backend-model", "server reports no models")
    } else if names.contains(&config.backend.model.as_str()) {
        Check::ok("backend-model", format!("`{}` is served", config.backend.model))
    } else {
        Check::warn(
            "backend-model",
            format!(
                "`{}` is not in the list; available: {}",
                config.backend.model,
                names.join(", ")
            ),
        )
    };

    vec![
        Check::ok(
            "backend-models",
            format!(
                "{} model(s) in {} ms: {}",
                names.len(),
                started.elapsed().as_millis(),
                names.join(", ")
            ),
        ),
        served,
    ]
}

/// Two identical completions: the second must be served from the prefix cache.
///
/// This is the check that catches a prompt layout which defeats caching, which on this
/// backend is the difference between a one-second and a ninety-second step.
async fn latency_checks(client: &reqwest::Client, config: &Config) -> Vec<Check> {
    match measure_latency(client, config).await {
        Ok(measurement) => {
            let mut out = vec![Check::ok(
                "backend-latency",
                format!(
                    "first token after {} ms, {} completion token(s) in {} ms",
                    measurement.first_token_ms, measurement.completion_tokens, measurement.total_ms
                ),
            )];
            let cache_check = cache_check(measurement.cache);
            out.push(cache_check);
            out
        }
        Err(err) => vec![
            Check::fail("backend-latency", format!("completion request failed: {err:#}")),
            Check::skipped("backend-cache", "no completion to measure"),
        ],
    }
}

/// Turn the measured cache counters into a check. A ratio below 50% is a warning, because
/// it means the prompt layout is defeating caching and every step will be slow.
fn cache_check(cache: Option<CacheMeasurement>) -> Check {
    let Some(cache) = cache else {
        return Check::skipped("backend-cache", "backend did not report cache counters");
    };
    let status = if cache.hit_ratio() >= 0.5 {
        Status::Ok
    } else {
        Status::Warn
    };
    Check {
        name: "backend-cache",
        status,
        detail: format!(
            "second identical prefix: {}/{} prompt tokens from cache ({:.0}%)",
            cache.cached_tokens,
            cache.prompt_tokens,
            cache.hit_ratio() * 100.0
        ),
    }
}

fn disk_check(config: &Config) -> Check {
    // The daemon dies of ENOSPC long before it dies of anything else, and a multi-day run
    // makes that a certainty rather than a risk.
    let dir = config.logging.dir.clone();
    match free_bytes(&dir) {
        Some(bytes) if bytes > 2 * 1024 * 1024 * 1024 => Check::ok(
            "disk",
            format!("{:.1} GiB free under {}", bytes as f64 / 1e9, dir.display()),
        ),
        Some(bytes) if bytes > 512 * 1024 * 1024 => Check::warn(
            "disk",
            format!("only {:.2} GiB free under {}", bytes as f64 / 1e9, dir.display()),
        ),
        Some(bytes) => Check::fail(
            "disk",
            format!(
                "{:.0} MiB free under {} — runs will pause",
                bytes as f64 / 1e6,
                dir.display()
            ),
        ),
        None => Check::warn(
            "disk",
            format!("cannot determine free space for {}", dir.display()),
        ),
    }
}

fn log_dir_check(config: &Config) -> Check {
    let dir = &config.logging.dir;
    match std::fs::create_dir_all(dir) {
        Ok(()) => {
            // The probe name includes the pid: two `doctor` processes running at once (a
            // human and the daemon's health check, say) must not delete each other's probe
            // and report a writable directory as broken.
            let probe = dir.join(format!(".doctor-probe-{}", std::process::id()));
            match std::fs::write(&probe, b"ok").and_then(|()| std::fs::remove_file(&probe)) {
                Ok(()) => Check::ok("log-dir", format!("writable: {}", dir.display())),
                Err(err) => Check::fail("log-dir", format!("{} is not writable: {err}", dir.display())),
            }
        }
        Err(err) => Check::fail("log-dir", format!("cannot create {}: {err}", dir.display())),
    }
}

fn socket_check(path: Option<&Path>) -> Check {
    let Some(path) = path else {
        return Check::skipped("socket", "no daemon socket configured");
    };
    if !path.exists() {
        return Check::skipped(
            "socket",
            format!("{} does not exist (daemon not running)", path.display()),
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        match std::fs::metadata(path) {
            Ok(meta) => {
                let mode = meta.permissions().mode() & 0o777;
                if mode == 0o600 {
                    Check::ok("socket", format!("{} exists with mode 0600", path.display()))
                } else {
                    Check::fail(
                        "socket",
                        format!("{} has mode {mode:o}, expected 600", path.display()),
                    )
                }
            }
            Err(err) => Check::fail("socket", format!("cannot stat {}: {err}", path.display())),
        }
    }
    #[cfg(not(unix))]
    {
        Check::ok("socket", format!("{} exists", path.display()))
    }
}

// ── backend measurement ────────────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Debug, serde::Deserialize)]
struct ModelEntry {
    id: String,
}

#[derive(Debug, serde::Deserialize)]
struct ChatResponse {
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    timings: Option<Timings>,
}

#[derive(Debug, serde::Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

#[derive(Debug, serde::Deserialize)]
struct Timings {
    #[serde(default)]
    cache_n: u32,
}

#[derive(Debug, Clone, Copy)]
struct CacheMeasurement {
    prompt_tokens: u32,
    cached_tokens: u32,
}

impl CacheMeasurement {
    fn hit_ratio(self) -> f64 {
        if self.prompt_tokens == 0 {
            0.0
        } else {
            f64::from(self.cached_tokens) / f64::from(self.prompt_tokens)
        }
    }
}

struct Measurement {
    first_token_ms: u128,
    total_ms: u128,
    completion_tokens: u32,
    cache: Option<CacheMeasurement>,
}

async fn measure_latency(client: &reqwest::Client, config: &Config) -> anyhow::Result<Measurement> {
    // A stable, non-trivial prefix: short enough to be quick, long enough that a cache miss
    // is visible in the numbers.
    let prefix = "You are a diagnostics probe. Answer with a single word. ".repeat(20);
    let url = format!("{}/chat/completions", config.backend.base_url);

    let body = serde_json::json!({
        "model": config.backend.model,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": "Reply with the single word: ready"}
        ],
        "max_tokens": 8,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": config.backend.enable_thinking},
    });

    let started = Instant::now();
    let first = post_completion(client, &url, &body).await?;
    let first_token_ms = started.elapsed().as_millis();

    // Second identical request; only the cached-prefix counter differs.
    let second = post_completion(client, &url, &body).await;
    let cache = second.ok().and_then(|response| {
        let timings = response.timings.as_ref()?.cache_n;
        let prompt = response.usage.as_ref()?.prompt_tokens;
        if prompt == 0 && timings == 0 {
            None
        } else {
            Some(CacheMeasurement {
                prompt_tokens: prompt.max(timings),
                cached_tokens: timings,
            })
        }
    });

    Ok(Measurement {
        first_token_ms,
        total_ms: started.elapsed().as_millis(),
        completion_tokens: first.usage.map_or(0, |u| u.completion_tokens),
        cache,
    })
}

async fn post_completion(
    client: &reqwest::Client,
    url: &str,
    body: &serde_json::Value,
) -> anyhow::Result<ChatResponse> {
    let response = client.post(url).json(body).send().await?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("HTTP {status}: {}", text.chars().take(200).collect::<String>());
    }
    Ok(response.json::<ChatResponse>().await?)
}

// ── helpers ────────────────────────────────────────────────────────────────────

fn host_of(url: &str) -> Option<String> {
    let rest = url.split_once("://")?.1;
    let authority = rest.split('/').next()?;
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let host = host.split_once(':').map_or(host, |(h, _)| h);
    if host.is_empty() {
        None
    } else {
        Some(host.to_string())
    }
}

/// Free bytes on the filesystem containing `path`, walking up to the nearest existing parent.
fn free_bytes(path: &Path) -> Option<u64> {
    let mut probe = path;
    loop {
        if probe.exists() {
            break;
        }
        probe = probe.parent()?;
    }
    let output = std::process::Command::new("df")
        .arg("-k")
        .arg(probe)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    // `df -k` columns: Filesystem 1024-blocks Used Available Capacity ...
    text.lines()
        .nth(1)?
        .split_whitespace()
        .nth(3)?
        .parse::<u64>()
        .ok()
        .map(|kib| kib * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_extraction() {
        assert_eq!(
            host_of("http://llm-server:47832/v1").as_deref(),
            Some("llm-server")
        );
        assert_eq!(
            host_of("https://user:pw@example.test/v1").as_deref(),
            Some("example.test")
        );
        assert_eq!(host_of("junk"), None);
    }

    #[test]
    fn cache_hit_ratio_handles_zero_prompt() {
        let m = CacheMeasurement {
            prompt_tokens: 0,
            cached_tokens: 0,
        };
        assert!((m.hit_ratio() - 0.0).abs() < f64::EPSILON);
        let m = CacheMeasurement {
            prompt_tokens: 100,
            cached_tokens: 98,
        };
        assert!((m.hit_ratio() - 0.98).abs() < 1e-9);
    }

    #[test]
    fn exit_code_is_nonzero_only_on_failure() {
        let mut report = Report {
            checks: vec![Check::ok("a", "fine")],
        };
        assert_eq!(report.exit_code(), 0);
        report.checks.push(Check::warn("b", "meh"));
        assert_eq!(report.exit_code(), 0, "warnings must not fail the run");
        report.checks.push(Check::fail("c", "broken"));
        assert_eq!(report.exit_code(), 1);
    }

    #[test]
    fn trusted_host_check_catches_misconfiguration() {
        let mut config = Config::default();
        config.backend.base_url = "http://elsewhere.example:9000/v1".into();
        let check = trusted_host_check(&config);
        assert_eq!(check.status, Status::Fail);
        assert!(check.detail.contains("elsewhere.example"));

        let mut config = Config::default();
        config.policy.trusted_hosts.insert("elsewhere.example".into());
        config.backend.base_url = "http://elsewhere.example:9000/v1".into();
        assert_eq!(trusted_host_check(&config).status, Status::Ok);
    }

    #[test]
    fn log_dir_check_creates_and_writes() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = Config::default();
        config.logging.dir = dir.path().join("logs");
        let check = log_dir_check(&config);
        assert_eq!(check.status, Status::Ok, "{:?}", check.detail);
        assert!(config.logging.dir.exists());
    }

    #[test]
    fn log_dir_check_reports_unwritable_parent() {
        let mut config = Config::default();
        // /dev/null/immortality cannot be created.
        config.logging.dir = Path::new("/dev/null/immortality").to_path_buf();
        assert_eq!(log_dir_check(&config).status, Status::Fail);
    }

    #[test]
    fn socket_check_skips_when_absent() {
        let check = socket_check(Some(Path::new("/tmp/definitely-not-a-real-socket-immortality")));
        assert_eq!(check.status, Status::Skipped);
        assert_eq!(socket_check(None).status, Status::Skipped);
    }

    #[test]
    fn config_check_surfaces_warnings() {
        let config = Config::default();
        let global = Path::new("/tmp/config.toml");
        let clean = Context {
            config: &config,
            config_warnings: &[],
            global_config_path: global,
            socket_path: None,
        };
        assert_eq!(config_check(&clean).status, Status::Ok);

        let warnings = vec!["[policy] ignored".to_string()];
        let dirty = Context {
            config: &config,
            config_warnings: &warnings,
            global_config_path: global,
            socket_path: None,
        };
        let check = config_check(&dirty);
        assert_eq!(check.status, Status::Warn);
        assert!(check.detail.contains("[policy] ignored"));
    }
}
