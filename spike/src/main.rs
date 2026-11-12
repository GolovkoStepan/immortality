//! S1 spike: measure how a live local model behaves in an agent loop.
//!
//! Throwaway by design, with two exceptions that graduate into `immortality-llm`:
//! `sse.rs` (incremental SSE parser) and the tool-call accumulator in `llm.rs`.
//!
//! Usage:
//!   cargo run --release -- --base-url http://llm-server:47832/v1 --model swift-iq2_xs
//!   cargo run --release -- --scenario failing-tests --verbose

mod agent;
mod llm;
mod scenarios;
mod sse;
mod tools;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};

use agent::LoopConfig;
use llm::Client;
use tools::Sandbox;

#[derive(Debug)]
struct Args {
    base_url: String,
    model: String,
    scenario: Option<String>,
    max_steps: u32,
    max_tokens: u32,
    verbose: bool,
    sandbox_root: PathBuf,
    traces_dir: PathBuf,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self {
            base_url: std::env::var("SPIKE_BASE_URL")
                .unwrap_or_else(|_| "http://llm-server:47832/v1".to_string()),
            model: std::env::var("SPIKE_MODEL").unwrap_or_else(|_| "swift-iq2_xs".to_string()),
            scenario: None,
            max_steps: 40,
            max_tokens: 1024,
            verbose: false,
            sandbox_root: std::env::temp_dir().join("immortality-spike"),
            traces_dir: PathBuf::from("spike/traces"),
        };
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--base-url" => args.base_url = iter.next().context("--base-url needs a value")?,
                "--model" => args.model = iter.next().context("--model needs a value")?,
                "--scenario" => args.scenario = Some(iter.next().context("--scenario needs a value")?),
                "--max-steps" => args.max_steps = iter.next().context("--max-steps needs a value")?.parse()?,
                "--max-tokens" => args.max_tokens = iter.next().context("--max-tokens needs a value")?.parse()?,
                "--sandbox" => args.sandbox_root = PathBuf::from(iter.next().context("--sandbox needs a value")?),
                "--traces" => args.traces_dir = PathBuf::from(iter.next().context("--traces needs a value")?),
                "--verbose" | "-v" => args.verbose = true,
                "--help" | "-h" => {
                    println!(
                        "spike [--base-url URL] [--model NAME] [--scenario NAME] [--max-steps N] \
                         [--max-tokens N] [--sandbox DIR] [--traces DIR] [--verbose]"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument: {other}"),
            }
        }
        Ok(args)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse()?;
    std::fs::create_dir_all(&args.traces_dir)?;

    let client = Client::new(&args.base_url);
    let models = client.models().await.context("backend is not reachable")?;
    println!("backend : {}", args.base_url);
    println!("models  : {}", models.join(", "));
    if !models.iter().any(|m| m == &args.model) {
        eprintln!("warning: `{}` is not in /v1/models", args.model);
    }

    let selected: Vec<scenarios::Scenario> = match &args.scenario {
        Some(name) => {
            let found: Vec<_> = scenarios::all().into_iter().filter(|s| s.name == name).collect();
            if found.is_empty() {
                let names: Vec<&str> = scenarios::all().iter().map(|s| s.name).collect();
                bail!("unknown scenario `{name}`; available: {}", names.join(", "));
            }
            found
        }
        None => scenarios::all(),
    };

    let config = LoopConfig {
        model: args.model.clone(),
        max_steps: args.max_steps,
        max_tokens_per_request: args.max_tokens,
        verbose: args.verbose,
    };

    let mut reports = Vec::new();
    for scenario in &selected {
        println!("\n════════ {} ════════", scenario.name);
        let sandbox = Sandbox::new(args.sandbox_root.join(scenario.name))?;
        let started = Instant::now();
        match agent::run(&client, &sandbox, scenario, config.clone(), &args.traces_dir).await {
            Ok(report) => {
                println!(
                    "  {} in {} steps, {:.1}s — {}",
                    if report.completed { "COMPLETED" } else { "NOT COMPLETED" },
                    report.steps,
                    started.elapsed().as_secs_f64(),
                    report.stop_reason
                );
                reports.push(report);
            }
            Err(err) => {
                println!("  ABORTED: {err:#}");
            }
        }
    }

    // Machine-readable output for the report generator.
    let summary_path = args.traces_dir.join("summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&reports)?)?;
    println!("\nsummary written to {}", summary_path.display());
    print_table(&reports);

    Ok(())
}

fn print_table(reports: &[agent::RunReport]) {
    if reports.is_empty() {
        return;
    }
    println!("\n{:<26} {:>3} {:>4} {:>7} {:>7} {:>7} {:>8} {:>6} {:>6}", "", "st", "ok", "wall", "tok", "cache%", "invalid", "noprog", "empty");
    for r in reports {
        // `prompt_tokens` already includes the cached part; `cache_hits` is the subset.
        let cache_pct = if r.prompt_tokens > 0 {
            (r.cache_hits as f64 / r.prompt_tokens as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "{:<26} {:>3} {:>4} {:>6.0}s {:>7} {:>5.0}% {:>7} {:>6} {:>6}",
            r.scenario,
            r.steps,
            if r.completed { "yes" } else { "no" },
            r.wall_ms as f64 / 1000.0,
            r.prompt_tokens + r.completion_tokens,
            cache_pct,
            r.invalid_args + r.unknown_tools,
            r.no_progress_steps,
            r.empty_responses + r.thinking_truncated,
        );
    }
}
