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

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use clap::{Parser, Subcommand};

use immortality_core::config::{self, Overrides};
use immortality_core::paths::StatePaths;
use immortality_core::{logging, paths::RepoPaths};
use immortality_daemon::doctor::{self, Report, Status};
use immortality_daemon::service::{self, Agent, ServiceSpec};

mod client;
mod render;
mod serve;
mod style;

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
    /// Start a task, or resume one, and run it in the daemon.
    ///
    /// With a goal this admits a new task: a plan already on disk is respected, and one that is
    /// missing is planned first. With `--task` it resumes a task that already exists — the same
    /// path a restarted daemon takes.
    Start {
        /// What to accomplish. Omit only when resuming with `--task`.
        goal: Option<String>,
        /// Resume this task instead of creating one.
        #[arg(long)]
        task: Option<String>,
        /// A registered project path. Defaults to the current directory.
        #[arg(long)]
        project: Option<String>,
        /// Re-plan an existing tree instead of running the one on disk. Needs a goal.
        #[arg(long)]
        force: bool,
    },
    /// Attach to running events. (M3)
    Attach {
        /// The task to watch. Omit when only one task exists.
        task: Option<String>,
        /// Resume after the last sequence number already seen.
        #[arg(long, default_value_t = 0)]
        since_seq: u64,
    },
    /// Show task state. (M3)
    Status {
        /// One task, instead of every task this daemon knows.
        task: Option<String>,
    },
    /// Show the plan as the daemon reads it, or build one from a goal.
    ///
    /// Without `--goal` this reports the tree already on disk. With it, the planner is asked to
    /// decompose the goal and the three files are written — after which a human may edit them
    /// before the run starts.
    Plan {
        /// A registered project path. Defaults to the current directory.
        #[arg(long)]
        project: Option<String>,
        /// Ask the planner to build a plan for this goal.
        #[arg(long)]
        goal: Option<String>,
        /// Replace an existing plan. Without this, a plan that is already there is left alone,
        /// because a human may have edited it.
        #[arg(long)]
        force: bool,
    },
    /// List pending escalations. (M3)
    Escalations {
        /// One task, instead of every task this daemon knows.
        task: Option<String>,
    },
    /// Answer a pending escalation. (M3)
    Approve {
        /// The escalation id, from `immortality escalations`.
        id: String,
        /// One of `allow_once`, `allow_for_task`, `deny`.
        answer: String,
    },
    /// Print a human-readable transcript of a run. (M5)
    Log { task: Option<String> },
    /// Print the v0.1 numbers computed from journals. (M6)
    ///
    /// The figures `docs/metrics-v0.1.md` quotes, computed from the same files rather than
    /// transcribed by hand: closed subtasks, detector firings, escalations and restarts.
    Metrics {
        /// One task, instead of every task this daemon knows.
        task: Option<String>,
    },
    /// Merge a finished task branch into main. Human action only. (M5)
    Merge { task: String },
    /// Rotate logs and prune artefacts.
    Gc {
        /// Report what would be collected without deleting anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Run the daemon in the foreground until killed.
    Serve,
    /// Manage the launchd agent that runs the daemon in the background.
    Service {
        #[command(subcommand)]
        action: ServiceAction,
    },
}

#[derive(Debug, Subcommand)]
enum ServiceAction {
    /// Install and load the agent. Idempotent; a healthy daemon is never restarted.
    Install,
    /// Unload the agent and remove its plist. Idempotent.
    Uninstall,
    /// Show whether the plist is installed and whether launchd is running it.
    Status,
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
        Command::Gc { dry_run } => gc_command(dry_run, cli.json).await,
        Command::Escalations { task } => escalations_command(task, cli.json).await,
        Command::Approve { id, answer } => approve_command(&id, &answer, cli.json).await,
        Command::Init => init_command(&loaded, cli.json),
        Command::Start {
            goal,
            task,
            project,
            force,
        } => start_command(goal, task, project, force, cli.json).await,
        Command::Plan { project, goal, force } => plan_command(project, goal, force, cli.json).await,
        Command::Attach { task, since_seq } => attach_command(task, since_seq, cli.json).await,
        Command::Status { task } => status_command(task, cli.json).await,
        Command::Log { task } => log_command(task, cli.json).await,
        Command::Metrics { task } => metrics_command(task, cli.json).await,
        Command::Merge { task } => merge_command(&task, cli.json).await,
        Command::Serve => serve::run().await,
        Command::Service { action } => service_command(&action, cli.json),
    }
}

/// `service`: the launchd agent behind a background daemon.
///
/// Not `async`: nothing here talks to the daemon. `install`/`uninstall`/`status` drive `launchctl`
/// and the filesystem, and making them async would only obscure that they are blocking calls.
fn service_command(action: &ServiceAction, json: bool) -> Result<()> {
    match action {
        ServiceAction::Install => service_install(json),
        ServiceAction::Uninstall => service_uninstall(json),
        ServiceAction::Status => service_status(json),
    }
}

/// `service install`: write the plist and load the agent, disturbing nothing that works.
fn service_install(json: bool) -> Result<()> {
    let spec = ServiceSpec::discover().map_err(|error| anyhow::anyhow!("{error}"))?;
    let agent = Agent::discover().map_err(|error| anyhow::anyhow!("{error}"))?;
    let outcome = agent.install(&spec).map_err(|error| anyhow::anyhow!("{error}"))?;
    let plist = agent.plist().to_path_buf();
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "installed": match outcome {
                    service::Installed::Fresh => "fresh",
                    service::Installed::Unchanged => "unchanged",
                    service::Installed::Reloaded => "reloaded",
                },
                "plist": plist,
            }))?
        );
        return Ok(());
    }
    match outcome {
        service::Installed::Fresh => {
            println!("installed {} and started the daemon", service::LABEL);
        }
        service::Installed::Unchanged => {
            println!("{} is already installed and running", service::LABEL);
        }
        service::Installed::Reloaded => {
            println!(
                "reinstalled {} and restarted it to pick up the change",
                service::LABEL
            );
        }
    }
    println!("  plist: {}", plist.display());
    println!(
        "  state: {}\n  logs:  {}",
        spec.state_root.display(),
        spec.log_path.display()
    );
    Ok(())
}

/// `service uninstall`: unload and remove. Idempotent, for scripts.
fn service_uninstall(json: bool) -> Result<()> {
    Agent::discover()
        .map_err(|error| anyhow::anyhow!("{error}"))?
        .uninstall()
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    if json {
        println!("{}", serde_json::json!({ "installed": false }));
    } else {
        println!("uninstalled {}", service::LABEL);
    }
    Ok(())
}

/// `service status`: installed and running are two separate facts.
fn service_status(json: bool) -> Result<()> {
    // `discover` failing means there is no agent to report on, which is a different answer from
    // "not installed": the latter is a state a human can fix with `service install`.
    let agent = Agent::discover().map_err(|error| anyhow::anyhow!("{error}"))?;
    let status = agent.status();
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "installed": status.installed,
                "running": status.running,
                "pid": status.pid,
                "plist": status.plist,
            }))?
        );
        return Ok(());
    }
    // Reported separately on purpose: a crashed job inside its throttle window is installed and
    // not running, and answering only "installed" would tell a human the opposite of what they
    // need to know.
    println!("installed: {}", if status.installed { "yes" } else { "no" });
    if status.installed {
        println!("  plist:   {}", status.plist.display());
    }
    println!(
        "running:   {}",
        match (status.running, status.pid) {
            (true, Some(pid)) => format!("yes (pid {pid})"),
            (true, None) => "yes".to_string(),
            (false, _) => "no".to_string(),
        }
    );
    if !status.installed {
        println!("Install it with `immortality service install`.");
    } else if !status.running {
        println!("launchd is not running it. `immortality service install` reloads the agent.");
    }
    Ok(())
}

/// `start`: admit a task, or resume one, and report the admission.
///
/// The run itself belongs to the daemon (D-088's rule, and the milestone's): the client sends a
/// goal or a task id and prints the answer, which is "started", not "finished". Everything that
/// decides whether the run may begin — the plan, the depth limit, the single run slot — is the
/// daemon's, because a client that duplicated those rules would be a second, stale answer.
async fn start_command(
    goal: Option<String>,
    task: Option<String>,
    project: Option<String>,
    force: bool,
    json: bool,
) -> Result<()> {
    let params = start_params(goal, task, project, force)?;
    let mut client = connect().await?;
    let result = take(client.request("start", params).await?)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print!("{}", render::admitted(&result, style()));
    }
    Ok(())
}

/// What `start` sends, or why the two arguments it was given cannot both be honoured.
///
/// Kept out of [`start_command`] so the refusal is testable without a daemon: "a goal and `--task`
/// together" is a usage error the human should hear immediately, not after a socket round trip.
fn start_params(
    goal: Option<String>,
    task: Option<String>,
    project: Option<String>,
    force: bool,
) -> Result<serde_json::Value> {
    match (goal, task) {
        (Some(_), Some(_)) => anyhow::bail!(
            "give either a goal or `--task`, not both: a goal admits a new task, \
             `--task` resumes an existing one"
        ),
        (Some(goal), None) => {
            if goal.trim().is_empty() {
                anyhow::bail!("the goal must not be empty; say what the task should accomplish");
            }
            let project = match project {
                Some(project) => PathBuf::from(project),
                // The daemon cannot see this process's working directory, and its own is `/` under
                // launchd, so the client always says which repository it means.
                None => std::env::current_dir().context("cannot read the current directory")?,
            };
            Ok(serde_json::json!({
                "project": project,
                "goal": goal,
                "force": force,
            }))
        }
        (None, Some(task)) => {
            if force {
                anyhow::bail!(
                    "`--force` re-plans, and re-planning needs a goal; a resume runs the plan \
                     already on disk"
                );
            }
            Ok(serde_json::json!({ "task": task }))
        }
        (None, None) => anyhow::bail!(
            "say what to do — `immortality start \"what to do\"` — or resume a task with \
             `immortality start --task <id>`"
        ),
    }
}

/// `status`: what this daemon knows, as a table or raw JSON.
async fn status_command(task: Option<String>, json: bool) -> Result<()> {
    let mut client = connect().await?;
    let params = task.as_ref().map_or_else(
        || serde_json::json!({}),
        |task| serde_json::json!({ "task": task }),
    );
    let result = client.request("status", params).await?;
    let result = take(result)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print!("{}", render::status(&result, style()));
    }
    Ok(())
}

/// `plan`: the tree as the scheduler reads it, or a tree built from a goal.
///
/// The project defaults to the current directory, which is what a human means when they type
/// `immortality plan` inside a repository. An unregistered directory is an error rather than a
/// guess: the daemon has no task there, so there is no tree for it to read.
///
/// The goal is sent rather than planned here, because the plan must be validated against the
/// daemon's depth limit and written by the process that owns the inference slot (D-088). This
/// client's only extra job is to say which repository it is standing in.
async fn plan_command(project: Option<String>, goal: Option<String>, force: bool, json: bool) -> Result<()> {
    // Refused before the socket is opened: an empty goal is a usage error, and asking a daemon
    // that is not running would report the wrong problem. The daemon refuses it too — it answers
    // other clients — but a human at a terminal should get the answer that is about what they
    // typed.
    if let Some(goal) = &goal
        && goal.trim().is_empty()
    {
        anyhow::bail!("the goal must not be empty; say what the task should accomplish");
    }
    let mut client = connect().await?;
    // The daemon cannot see this process's working directory, and its own is meaningless under
    // launchd, so the client always says which repository it means.
    let project = match project {
        Some(project) => PathBuf::from(project),
        None => std::env::current_dir().context("cannot read the current directory")?,
    };
    let mut params = serde_json::json!({ "project": project });
    if let Some(goal) = goal {
        params["goal"] = serde_json::json!(goal);
        params["force"] = serde_json::json!(force);
    }
    let result = take(client.request("plan", params).await?)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else if result.get("created").and_then(serde_json::Value::as_bool) == Some(true) {
        print!("{}", render::planned(&result, style()));
    } else {
        print!("{}", render::plan(&result, style()));
    }
    Ok(())
}

/// `log`: the transcript of one task.
async fn log_command(task: Option<String>, json: bool) -> Result<()> {
    let mut client = connect().await?;
    let task = match task {
        Some(task) => task,
        None => single_task(&mut client).await?,
    };
    let result = take(client.request("log", serde_json::json!({ "task": task })).await?)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print!("{}", render::log(&result, style()));
    }
    Ok(())
}

/// `metrics`: the v0.1 numbers, computed by the daemon from the journals.
async fn metrics_command(task: Option<String>, json: bool) -> Result<()> {
    let mut client = connect().await?;
    let params = task.map_or_else(
        || serde_json::json!({}),
        |task| serde_json::json!({ "task": task }),
    );
    let result = take(client.request("metrics", params).await?)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print!("{}", render::metrics(&result, style()));
    }
    Ok(())
}

/// `gc`: collect what nothing needs.
async fn gc_command(dry_run: bool, json: bool) -> Result<()> {
    let mut client = connect().await?;
    let result = take(
        client
            .request("gc", serde_json::json!({ "dry_run": dry_run }))
            .await?,
    )?;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print!("{}", render::gc_report(&result, style()));
    }
    Ok(())
}

/// `merge`: the human's one git action.
async fn merge_command(task: &str, json: bool) -> Result<()> {
    let mut client = connect().await?;
    let result = take(
        client
            .request("merge", serde_json::json!({ "task": task }))
            .await?,
    )?;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print!("{}", render::merged(&result, style()));
    }
    Ok(())
}

/// `escalations`: the questions waiting for a human.
async fn escalations_command(task: Option<String>, json: bool) -> Result<()> {
    let mut client = connect().await?;
    let params = task.as_ref().map_or_else(
        || serde_json::json!({}),
        |task| serde_json::json!({ "task": task }),
    );
    let result = client.request("escalations", params).await?;
    let result = take(result)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print!("{}", render::escalations(&result, style()));
    }
    Ok(())
}

/// `approve`: answer one question.
async fn approve_command(id: &str, answer: &str, json: bool) -> Result<()> {
    let mut client = connect().await?;
    let result = client
        .request("approve", serde_json::json!({ "id": id, "answer": answer }))
        .await?;
    let result = take(result)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print!("{}", render::approved(&result, style()));
    }
    Ok(())
}

/// `attach`: stream one task's records until interrupted.
///
/// The task may be omitted only when exactly one exists. Choosing the first of several would
/// silently attach the human to work they did not ask about, which is worse than asking.
async fn attach_command(task: Option<String>, since_seq: u64, json: bool) -> Result<()> {
    let mut client = connect().await?;
    let task = match task {
        Some(task) => task,
        None => single_task(&mut client).await?,
    };

    if json {
        // One record per line, unmodified: this is what a script parses, and any prettifying
        // would be a second format to keep in sync with the daemon's.
        client
            .attach(
                &task,
                since_seq,
                |record| {
                    if let Ok(line) = serde_json::to_string(record) {
                        println!("{line}");
                    }
                },
                // A script consuming `--json` gets records only: a reconnect notice is not a
                // record and would make the stream unparseable. `stderr` is not used either,
                // because the point of `--json` is an exactly reproducible byte stream.
                |_notice| {},
            )
            .await?;
        return Ok(());
    }

    // Bytes, not lines: a renderer that buffers until a newline makes a running task look
    // frozen, and "looks frozen" is how a human concludes the agent hung and kills it.
    let stdout = std::io::stdout();
    let style = style();
    let mut renderer = render::Renderer::new(style);
    client
        .attach(
            &task,
            since_seq,
            |record| {
                write_stdout(&stdout, &renderer.line(record));
            },
            // `Style` is `Copy`, so the notice closure does not need to borrow the renderer —
            // two closures cannot hold `&mut` on it at once.
            |notice| {
                write_stdout(&stdout, &render::notice(notice, style));
            },
        )
        .await?;
    Ok(())
}

/// Write `text` and flush, ignoring a closed pipe.
///
/// A human who pressed Ctrl-C or piped into `head` closed stdout; the command is over, and an
/// error there would only be noise. Bytes are written rather than lines so a block that arrives
/// in pieces still appears as it arrives.
fn write_stdout(stdout: &std::io::Stdout, text: &str) {
    if text.is_empty() {
        // `tool_call_started` carries no news of its own; a write of zero bytes would be a
        // syscall per record for nothing.
        return;
    }
    let mut lock = stdout.lock();
    let _ = std::io::Write::write_all(&mut lock, text.as_bytes());
    let _ = std::io::Write::flush(&mut lock);
}

/// Whether this process may write colour, and the cost of deciding it once.
///
/// Read on every call rather than cached in a global: a CLI process is short-lived, and a global
/// would make the test binary's first call decide for every test after it.
fn style() -> style::Style {
    use std::io::IsTerminal as _;

    let is_tty = std::io::stdout().is_terminal();
    let no_color = std::env::var_os("NO_COLOR");
    let term = std::env::var_os("TERM");
    if style::should_colour(is_tty, no_color.as_deref(), term.as_deref()) {
        style::Style::colour()
    } else {
        style::Style::plain()
    }
}

/// The one task a daemon knows, or an error naming what to do instead.
async fn single_task(client: &mut client::Client) -> Result<String> {
    let result = take(client.request("status", serde_json::json!({})).await?)?;
    let tasks = result
        .get("tasks")
        .and_then(serde_json::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    match tasks {
        [one] => one
            .get("task")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .context("the daemon listed a task with no id"),
        [] => anyhow::bail!(
            "no task exists yet, so there is nothing to attach to.\n\
             Start one with `immortality start <goal>`."
        ),
        many => anyhow::bail!(
            "this daemon has {} tasks; name the one to watch.\n\
             `immortality status` lists them.",
            many.len()
        ),
    }
}

/// Open a connection with the default endpoints.
async fn connect() -> Result<client::Client> {
    let (socket, token) = client::default_endpoints();
    client::Client::connect(&socket, &token).await
}

/// The result of a request, or the daemon's refusal as an error.
fn take(response: immortality_daemon::rpc::Response) -> Result<serde_json::Value> {
    match (response.result, response.error) {
        (Some(result), _) => Ok(result),
        (None, Some(error)) => Err(anyhow::anyhow!("{}", client::explain(&error))),
        (None, None) => anyhow::bail!("the daemon answered with neither a result nor an error"),
    }
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

    // Registering is what links a task id to this repository later; without it the daemon
    // cannot find the journal of a task it is asked about. Best-effort on purpose: `init`
    // creating `.agent/` must not fail because a shared state directory is unwritable, and the
    // warning tells the human what will break.
    let state = StatePaths::discover();
    let registry_note = match register_project(&state, &cwd) {
        Ok(id) => Some(id),
        Err(error) => {
            eprintln!("warning: could not register this repository with the daemon: {error}");
            None
        }
    };

    if json {
        println!(
            "{}",
            serde_json::json!({
                "created": agent_dir.display().to_string(),
                "config": project_config.display().to_string(),
                "project": registry_note,
            })
        );
    } else {
        println!("created {}", agent_dir.display());
        println!("wrote   {}", project_config.display());
        if let Some(id) = &registry_note {
            println!("registered project {id}");
        }
        println!(
            "\nnext: edit {} and run `immortality doctor`",
            project_config.display()
        );
    }
    Ok(())
}

/// Register `path` with the daemon's project registry, returning its id.
///
/// The registry lives beside the daemon's socket, not in the repository: it is machine-local
/// state, and committing it would make two clones fight over the same ids.
fn register_project(state: &StatePaths, path: &Path) -> Result<String> {
    let registry_path = state.registry();
    if let Some(parent) = registry_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let mut registry = immortality_core::registry::Registry::open(&registry_path)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let remote = git_remote(path);
    registry
        .register(path, remote)
        .map_err(|error| anyhow::anyhow!("{error}"))
}

/// The repository's `origin` URL, when it has one.
///
/// Read from `.git/config` by looking for the `[remote "origin"]` section rather than running
/// `git`: `init` must work in a directory git considers broken, and this field is informational
/// (nothing here pushes).
fn git_remote(repo: &Path) -> Option<String> {
    let config = std::fs::read_to_string(repo.join(".git").join("config")).ok()?;
    let mut in_origin = false;
    for line in config.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_origin = trimmed == "[remote \"origin\"]";
            continue;
        }
        if in_origin
            && let Some(url) = trimmed.strip_prefix("url")
            && let Some(url) = url.trim_start().strip_prefix('=')
        {
            return Some(url.trim().to_string());
        }
    }
    None
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
    fn start_refuses_a_goal_with_no_task_and_no_plan() {
        // The daemon knows whether a plan exists; the client only reports what it was told. This
        // test pins the client half: `--task` and a goal together are a usage error, because the
        // two request different things and guessing which was meant would run the wrong one.
        let Err(error) = start_params(Some("do it".to_string()), Some("tsk_x".to_string()), None, false)
        else {
            panic!("a goal and a task together must be refused");
        };
        let text = error.to_string();
        assert!(text.contains("--task"), "{text}");
    }
}
