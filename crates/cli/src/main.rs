//! `immortality` — thin client for the daemon.
//!
//! At M0 the only working command is `doctor`, because diagnostics are what the milestone is
//! for. Everything else is declared so the interface is visible and versioned from day one,
//! and returns a clear "not implemented until Mx" instead of pretending to work.

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

use std::path::PathBuf;

use anyhow::{Context as _, Result};
use clap::{Parser, Subcommand};

use immortality_core::config::{self, Overrides};
use immortality_core::paths::StatePaths;
use immortality_core::{logging, paths::RepoPaths};
use immortality_daemon::doctor::{self, Report, Status};

#[derive(Debug, Parser)]
#[command(
    name = "immortality",
    version,
    about = "Long-lived terminal coding agent for local models"
)]
struct Cli {
    /// Path to the global config file.
    #[arg(long, global = true, env = "IMMORTALITY_GLOBAL_CONFIG")]
    global_config: Option<PathBuf>,

    /// Path to the project config file.
    #[arg(long, global = true)]
    project_config: Option<PathBuf>,

    /// Override the backend base URL.
    #[arg(long, global = true)]
    base_url: Option<String>,

    /// Override the model name for every profile.
    #[arg(long, global = true)]
    model: Option<String>,

    /// Log level for this process.
    #[arg(long, global = true)]
    log_level: Option<String>,

    /// Emit machine-readable JSON instead of human output.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Diagnostic report for config, backend, disk and daemon socket.
    Doctor,
    /// Create the `.agent/` directory and a starter config. (M4)
    Init,
    /// Start a task. (M4)
    Start {
        /// What to accomplish.
        goal: String,
    },
    /// Attach to running events. (M5)
    Attach,
    /// Show task state. (M5)
    Status,
    /// Show or edit the plan. (M4)
    Plan,
    /// List pending escalations. (M3)
    Escalations,
    /// Print a human-readable transcript of a run. (M5)
    Log { task: Option<String> },
    /// Merge a finished task branch into main. Human action only. (M5)
    Merge { task: String },
    /// Rotate logs and prune artefacts. (M1)
    Gc,
    /// Run the daemon in the foreground. (M5)
    Serve,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let overrides = Overrides {
        base_url: cli.base_url,
        model: cli.model,
        log_level: cli.log_level,
        global_config: cli.global_config,
        project_config: cli.project_config,
    };
    let loaded = config::load(&overrides).map_err(|err| anyhow::anyhow!("{err}"))?;

    // Logging is best-effort for short commands: a diagnostic tool must still work when the
    // log directory is the thing that is broken.
    let _ = logging::init(&loaded.config.logging.level, &loaded.config.logging.dir);

    match cli.command {
        Command::Doctor => doctor_command(&loaded, cli.json).await,
        Command::Gc => not_yet("gc", "M1"),
        Command::Escalations => not_yet("escalations", "M3"),
        Command::Init => init_command(&loaded, cli.json),
        Command::Start { goal } => {
            if goal.trim().is_empty() {
                anyhow::bail!("goal must not be empty");
            }
            not_yet("start", "M4")
        }
        Command::Plan => not_yet("plan", "M4"),
        Command::Attach => not_yet("attach", "M5"),
        Command::Status => not_yet("status", "M5"),
        Command::Log { task: _ } => not_yet("log", "M5"),
        Command::Merge { task: _ } => not_yet("merge", "M5"),
        Command::Serve => not_yet("serve", "M5"),
    }
}

fn not_yet(command: &str, milestone: &str) -> Result<()> {
    anyhow::bail!("`{command}` is not implemented until {milestone}; see docs/roadmap.md")
}

async fn doctor_command(loaded: &config::Loaded, json: bool) -> Result<()> {
    let state = StatePaths::discover();
    let socket = state.daemon_socket();
    let context = doctor::Context {
        config: &loaded.config,
        config_warnings: &loaded.warnings,
        global_config_path: &loaded.global_path,
        socket_path: Some(&socket),
    };
    let report = doctor::run(&context).await;

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human(&report);
    }
    if report.has_failures() {
        std::process::exit(report.exit_code());
    }
    Ok(())
}

fn print_human(report: &Report) {
    for check in &report.checks {
        let marker = match check.status {
            Status::Ok => "ok  ",
            Status::Warn => "warn",
            Status::Fail => "FAIL",
            Status::Skipped => "--  ",
        };
        println!("{marker}  {:<14} {}", check.name, check.detail);
    }
    let failed = report.checks.iter().filter(|c| c.status == Status::Fail).count();
    let warned = report.checks.iter().filter(|c| c.status == Status::Warn).count();
    println!();
    if failed == 0 {
        println!("{warned} warning(s), no failures");
    } else {
        println!("{failed} failure(s), {warned} warning(s)");
    }
}

fn init_command(loaded: &config::Loaded, json: bool) -> Result<()> {
    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let repo = RepoPaths::new(&cwd);
    let agent_dir = repo.agent_dir();
    std::fs::create_dir_all(&agent_dir).with_context(|| format!("cannot create {}", agent_dir.display()))?;

    // A project config is scaffolding, not policy: it carries budgets and model names only.
    let project_config = repo.project_config();
    if !project_config.exists() {
        let without_policy = project_config_toml(&loaded.config)?;
        let header = "\
# Project configuration for immortality.
# Security policy is deliberately absent: it lives in the global config only.
# See docs/design.md §3.
";
        std::fs::write(&project_config, format!("{header}\n{without_policy}"))
            .with_context(|| format!("cannot write {}", project_config.display()))?;
    }

    if json {
        println!(
            "{}",
            serde_json::json!({"created": agent_dir.display().to_string(), "config": project_config.display().to_string()})
        );
    } else {
        println!("created {}", agent_dir.display());
        println!("wrote   {}", project_config.display());
        println!(
            "\nnext: edit {} and run `immortality doctor`",
            project_config.display()
        );
    }
    Ok(())
}

/// Serialise the effective config for a project file.
///
/// `toml::to_string` emits the Rust field name `[loop_]`, which the loader (correctly)
/// rejects as an unknown key. Fix that one key here and then verify the result parses with
/// the loader's own type, so `init` can never write a config it cannot read back.
fn project_config_toml(config: &immortality_core::config::Config) -> Result<String> {
    let text = toml::to_string(config).context("cannot serialise config")?;
    let text = text.replace("[loop_]", "[loop]");
    toml::from_str::<immortality_core::config::RawConfig>(&text)
        .context("generated project config does not parse; this is a bug")?;
    Ok(strip_policy_section(&text))
}

/// Remove the `[policy]` table (and its key-value lines) from a serialised config.
///
/// `toml::to_string` emits all tables, including policy; the project file must not contain
/// it, because its presence would be both useless and confusing.
fn strip_policy_section(text: &str) -> String {
    let mut out = String::new();
    let mut in_policy = false;
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('[') {
            in_policy = trimmed.starts_with("[policy");
        }
        if !in_policy {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.trim_end().to_string() + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_policy_removes_table_and_its_keys() {
        let text = "[backend]\nmodel = \"m\"\n\n[policy]\nallow_push = false\ntrusted_hosts = []\n\n[context]\nsoft_budget_tokens = 100\n";
        let stripped = strip_policy_section(text);
        assert!(!stripped.contains("policy"), "{stripped}");
        assert!(!stripped.contains("allow_push"), "{stripped}");
        assert!(stripped.contains("[backend]"));
        assert!(stripped.contains("[context]"));
    }

    #[test]
    fn strip_policy_on_text_without_policy_is_identity() {
        let text = "[backend]\nmodel = \"m\"\n";
        assert_eq!(strip_policy_section(text), text);
    }

    #[test]
    fn generated_project_config_parses_and_uses_loop_key() {
        // Regression: `toml::to_string` writes `[loop_]`, which the loader rejects. `init`
        // used to write a config that made every following command fail to start.
        let text = project_config_toml(&immortality_core::config::Config::default()).unwrap();
        assert!(text.contains("[loop]"), "{text}");
        assert!(!text.contains("loop_"), "{text}");
        assert!(!text.contains("[policy]"), "{text}");
        let raw: immortality_core::config::RawConfig = toml::from_str(&text).unwrap();
        assert_eq!(raw.loop_.and_then(|l| l.stall_steps), Some(15));
    }

    #[test]
    fn not_yet_names_the_milestone() {
        let Err(err) = not_yet("attach", "M5") else {
            panic!("`not_yet` must fail");
        };
        let text = err.to_string();
        assert!(text.contains("attach"), "{text}");
        assert!(text.contains("M5"), "{text}");
    }
}
