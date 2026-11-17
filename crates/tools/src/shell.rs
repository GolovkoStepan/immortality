//! `shell` — running a command with a deadline, a closed stdin, and output that cannot flood the
//! context.
//!
//! # Process per command, not a persistent shell
//!
//! D-058 records why. The short version: a persistent shell needs a PTY and `unsafe` signal
//! handling to avoid zombies, and the design's own risk note for this step names that complexity
//! as the danger. `cwd` survives between calls through a marker file; `cd` inside one call works
//! because the whole command shares one `sh`; an `export` does not, and the observation says so
//! rather than pretending.
//!
//! # Why the group is killed and not the child
//!
//! Measured, not assumed: `sh -c 'sleep 300'` `exec`s the sleep, so the direct child *is* the
//! process; but `sh -c 'sleep 301; echo done'` leaves `sh` resident with `sleep` as its child.
//! Killing only the direct child in the second case orphans the grandchild, which keeps running,
//! keeps writing, and outlives the step that was supposed to have been cleaned up. So every
//! command starts in its own process group and the timeout kills the group.
//!
//! # Why stdin is `/dev/null`
//!
//! A command that reads from stdin — `cargo publish`, `git commit` without `-m`, `ssh` — blocks
//! forever on a terminal that will never type. Feeding it EOF makes it fail in one second with a
//! real error message instead of hanging the agent until a disk guard notices nothing is
//! happening. This is the single most valuable line in the file.
//!
//! # Why output goes to an artifact
//!
//! `find /` prints gigabytes, and the model sees 8 KiB of them either way. Keeping everything in
//! a content-addressed artifact means the human can look at the whole thing and the model gets
//! the tail — where the compiler's actual error is — plus a line count that tells it how much it
//! is missing.
//!
//! The output goes straight to a file, not into a pipe drained by a thread. That was the first
//! implementation and it had two defects that only showed up under sabotage. A pipe needs a
//! reader, a full pipe blocks the writer, and a background child inherits the write end, so the
//! reader waits for a process the step never asked to wait for — an unbounded join on a step that
//! is supposed to have a deadline. A file has none of that: the kernel writes, nothing blocks, and
//! `find /` produces its gigabytes without this module holding a byte of them.
//!
//! # Why the timeout result is not an error
//!
//! A timed-out command is a *result*: the model needs to know that it ran for 120 s, produced
//! these lines, and was killed. Returning `Err` would make it indistinguishable from "the command
//! never started", and the model's next move differs completely between those two.

use std::fmt::Write as _;
use std::io::{BufReader, Read as _, Seek as _, SeekFrom};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use immortality_core::artifacts::ArtifactStore;
use immortality_core::domain::dod::masks_exit_code;
use immortality_core::protocol::Idempotency;
use serde_json::{Value, json};

use crate::Effect;
use crate::args::Args;
use crate::policy::PathUse;
use crate::tool::{Tool, ToolCtx, ToolError, ToolResult};

/// Timeout when the model does not ask for one.
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Largest timeout a model may request.
///
/// The design's default is 120 s and the point of a cap is that a model cannot turn one step into
/// an hour of silence. Anything genuinely long-lived belongs in `start_process`, which returns a
/// handle instead of waiting.
pub const MAX_TIMEOUT_SECS: u64 = 600;

/// How long a `TERM`ed process gets before `KILL`.
///
/// Five seconds, from the design. Enough for a process to flush and remove its temp files; short
/// enough that a wedged process does not extend the step.
pub const KILL_GRACE: Duration = Duration::from_secs(5);

/// How much of the end of the output is read to find the lines worth showing.
///
/// Half a megabyte is a few thousand lines even for a long `cargo` log, and it is read with a
/// single `seek`, so the step never walks a hundred megabytes to find the last two hundred lines.
const TAIL_WINDOW: u64 = 512 * 1024;

/// Lines of output shown to the model, from the end.
///
/// The end, because that is where `error[E0308]` is. A build log is 90% warnings and then the
/// failure.
pub const SHOWN_LINES: usize = 200;

fn shell_schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command, run with `/bin/sh -c`. stdin is closed. `cd` in this command affects the next call's directory."
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory, relative to the worktree root. Defaults to the directory left by the previous call."
                },
                "timeout_secs": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_TIMEOUT_SECS,
                    "description": "How long to wait before killing the process group. Defaults to 120."
                },
                "env": {
                    "type": "object",
                    "description": "Extra environment variables for this call only.",
                    "additionalProperties": { "type": "string" }
                }
            },
            "required": ["command"],
            "additionalProperties": false
        })
    })
}

/// The shell state a session carries between calls.
///
/// Only `cwd`, because that is the only thing the marker can honestly carry. Environment changes
/// die with the `sh` that made them, which is a documented limitation rather than a hidden one.
#[derive(Debug, Default)]
pub struct ShellState {
    cwd: Mutex<Option<PathBuf>>,
}

impl ShellState {
    /// A fresh session at the worktree root.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The directory the next command will run in.
    #[must_use]
    pub fn cwd(&self) -> Option<PathBuf> {
        self.cwd.lock().ok().and_then(|guard| guard.clone())
    }
}

/// Run a command.
#[derive(Debug)]
pub struct Shell {
    state: Arc<ShellState>,
    artifacts: Option<ArtifactStore>,
}

impl Shell {
    /// A shell that keeps its `cwd` in `state` and spools large output nowhere.
    ///
    /// A `None` artifact store is the honest default for a test or a read-only session: output
    /// that exceeds [`IN_MEMORY_LIMIT`] is then reported as lost rather than silently dropped.
    #[must_use]
    pub const fn new(state: Arc<ShellState>, artifacts: Option<ArtifactStore>) -> Self {
        Self { state, artifacts }
    }

    /// A shell with a private state.
    #[must_use]
    pub fn standalone(artifacts: Option<ArtifactStore>) -> Self {
        Self::new(Arc::new(ShellState::new()), artifacts)
    }
}

impl Default for Shell {
    fn default() -> Self {
        Self::standalone(None)
    }
}

/// What a finished command produced.
#[derive(Debug)]
struct Finished {
    /// `None` when the process could not be reaped — it was killed and never reported. Distinct
    /// from "exited with no code" and reported as such rather than invented.
    status: Option<ExitStatus>,
    stdout: Captured,
    stderr: Captured,
    timed_out: bool,
    duration: Duration,
    /// A process the command backgrounded and left behind, killed by us because nothing else can.
    leftovers_killed: bool,
    /// Removes the capture files when the result is dropped.
    _scratch: Scratch,
}

/// One stream of output, summarised and still on disk.
#[derive(Debug, Default)]
struct Captured {
    /// Where the whole stream lives until the artifact takes it.
    path: PathBuf,
    bytes: u64,
    total_lines: usize,
    /// The last [`SHOWN_LINES`] lines, already joined.
    tail: String,
    shown: usize,
    omitted: usize,
}

/// The two files a command writes its output into, removed on drop.
///
/// A file rather than a pipe: nothing can block, no thread has to be joined, and a background
/// child holding the write end cannot keep the step open. `Drop` is the cleanup for the normal
/// path, the error path, and a panic alike.
#[derive(Debug, Default)]
struct Scratch {
    stdout: PathBuf,
    stderr: PathBuf,
}

impl Scratch {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let sequence = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let base =
            std::env::temp_dir().join(format!("immortality-shell-out-{}-{sequence}", std::process::id()));
        Self {
            stdout: base.with_extension("stdout"),
            stderr: base.with_extension("stderr"),
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.stdout);
        let _ = std::fs::remove_file(&self.stderr);
    }
}

impl Tool for Shell {
    fn name(&self) -> &'static str {
        "shell"
    }

    fn schema(&self) -> &'static Value {
        shell_schema()
    }

    fn description(&self) -> &'static str {
        "Run a shell command with stdin closed and a timeout. Output is truncated to the last 200 lines."
    }

    /// Delegated to the policy's own classifier, because the answer genuinely depends on the
    /// command: `ls` is a read, `rm` is a mutation, `curl` is external.
    fn effect(&self, args: &Args) -> Effect {
        // An unreadable command is not a command this gate can reason about.
        args.str("command").map_or(Effect::External, |command| {
            crate::classify::classify(command).effect
        })
    }

    /// A command may have run.
    ///
    /// Even a read is not simply replayable here: `shell("cargo test")` mutates `target/`, and a
    /// crashed call cannot be told apart from one that never started. `Unknown` makes recovery
    /// probe — read the tree — rather than replay blindly, which is what D-026 requires.
    fn idempotency(&self, _args: &Args) -> Idempotency {
        Idempotency::Unknown
    }

    fn paths(&self, args: &Args) -> Vec<PathUse> {
        let Ok(command) = args.str("command") else {
            return Vec::new();
        };
        let classification = crate::classify::classify(command);
        let mut paths = classification.paths;
        if let Ok(Some(cwd)) = args.opt_str("cwd") {
            paths.push(PathUse::read(cwd));
        }
        paths
    }

    /// The command itself, quoted, because the classified paths are often empty (`true`, `make`)
    /// and the operator's question is "which command was about to run", not "what did it touch".
    fn expected_effect(&self, args: &Args) -> String {
        let command = args.str("command").unwrap_or("?");
        crate::tool::one_line(&format!("run `{command}`"), crate::tool::EFFECT_LIMIT)
    }

    fn execute(&self, ctx: &ToolCtx, args: Args) -> Result<ToolResult, ToolError> {
        let command = args.str("command")?.trim().to_string();
        if command.is_empty() {
            return Err(ToolError::Arguments("`command` is empty".to_string()));
        }
        let timeout = args
            .opt_u64("timeout_secs")?
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .clamp(1, MAX_TIMEOUT_SECS);
        let env = read_env(&args)?;

        // The policy sees the command, not just the tool name.
        match ctx.gate().check_command(&command) {
            crate::Verdict::Allowed => {}
            crate::Verdict::Denied(denial) => return Err(ToolError::Refused(denial.observation())),
            crate::Verdict::Escalated(escalation) => {
                return Err(ToolError::Refused(escalation.observation()));
            }
        }

        let directory = self.resolve_cwd(ctx, &args)?;
        let marker = marker_path();
        let script = wrap_command(&command, &marker);
        let finished = run(&script, &directory, &env, Duration::from_secs(timeout))?;

        // The marker is the only way `cd` survives, so read it and store it. A command that never
        // reached the marker (killed, or `exec`ed away) leaves the previous directory in place.
        if let Some(found) = std::fs::read_to_string(&marker)
            .ok()
            .map(|text| text.trim().to_string())
            && !found.is_empty()
        {
            self.remember(&found);
        }
        let _ = std::fs::remove_file(&marker);

        Ok(self.report(&command, &finished, self.state.cwd().as_deref()))
    }
}

impl Shell {
    fn resolve_cwd(&self, ctx: &ToolCtx, args: &Args) -> Result<PathBuf, ToolError> {
        if let Some(raw) = args.opt_str("cwd")? {
            return ctx.enter_path(raw);
        }
        // A remembered directory is re-checked every call: it was inside the worktree when it was
        // recorded, and the worktree may have been rebuilt since.
        match self.state.cwd() {
            Some(remembered) if remembered.is_dir() => Ok(remembered),
            _ => ctx.enter_path("."),
        }
    }

    fn remember(&self, cwd: &str) {
        if let Ok(mut guard) = self.state.cwd.lock() {
            *guard = Some(PathBuf::from(cwd));
        }
    }

    fn report(&self, command: &str, finished: &Finished, cwd: Option<&Path>) -> ToolResult {
        let code = finished.status.and_then(|status| status.code());
        let summary = match (finished.status, code) {
            (Some(_), Some(0)) if !finished.timed_out => "exited 0".to_string(),
            (Some(_), Some(code)) => format!("exited {code}"),
            // A real signal death and our failure to reap both land here; the distinction the
            // model can act on is "it was killed", and `timed_out` says which kill it was.
            (Some(_), None) => "killed by a signal".to_string(),
            (None, _) => "killed, did not report".to_string(),
        };
        let masked = masks_exit_code(command);

        // The summary is the first line, not the last. It used to be appended after the output,
        // which put the one fact the model most needs — did this succeed — behind up to 8 KiB of
        // transcript, and made the first line of every result a `--- stdout ---` label. Measured
        // on the `roomly` run: 97 of 100 shell results rendered as `shell: --- stdout ---` in
        // `immortality log`, so the transcript was unreadable.
        // The summary is the first line, not the last. It used to be appended after the output,
        // which put the one fact the model most needs — did this succeed — behind up to 8 KiB of
        // transcript, and made the first line of every result a `--- stdout ---` label. Measured
        // on the `roomly` run: 97 of 100 shell results rendered as `shell: --- stdout ---` in
        // `immortality log`, so the transcript was unreadable.
        let mut observation = String::new();
        let _ = writeln!(
            observation,
            "[{summary} in {:.1} s, {} line(s) of stdout, {} of stderr]",
            finished.duration.as_secs_f64(),
            finished.stdout.total_lines,
            finished.stderr.total_lines
        );
        if masked {
            // `/bin/sh` gives a pipeline the status of its *last* command, so an `exited 0` above
            // can be `tail`'s and not the command's. Measured on the `roomly` run: nine results
            // carried a failure in the text — `16 examples, 16 failures` — next to `exited 0`,
            // because the model wrote `rspec … | tail -30`. D-116 refuses such a command as a
            // `DoD` check; the tool cannot refuse it, so it says what the number means.
            let _ = writeln!(
                observation,
                "This command is a pipeline: `{summary}` is the status of its last command, not \
                 of the whole pipeline. Drop the `|` to have a failure earlier in the pipe judged."
            );
        }
        if finished.leftovers_killed {
            let _ = writeln!(
                observation,
                "The command left a background process running; it has been killed. `shell` keeps \
                 no handle to background jobs — use `start_process` for anything that must \
                 outlive the call."
            );
        }
        if finished.timed_out {
            let _ = writeln!(
                observation,
                "The command was killed after {} s. Output so far follows; it may be incomplete.",
                finished.duration.as_secs()
            );
        }

        let mut shown = 0usize;
        let mut omitted = 0usize;
        for (label, captured) in [("stdout", &finished.stdout), ("stderr", &finished.stderr)] {
            if captured.bytes == 0 {
                continue;
            }
            omitted += captured.omitted;
            shown += captured.shown;
            observation.push('\n');
            let _ = writeln!(observation, "--- {label} ---");
            observation.push_str(&captured.tail);
            if captured.omitted > 0 {
                let _ = writeln!(
                    observation,
                    "… {} earlier line(s) not shown; the full output is in the artifact.",
                    captured.omitted
                );
            }
        }

        let artifact = self.spool(command, finished);

        ToolResult::new(
            observation,
            json!({
                "command": command,
                "exit_code": code,
                "timed_out": finished.timed_out,
                "duration_ms": u64::try_from(finished.duration.as_millis()).unwrap_or(u64::MAX),
                "stdout_lines": finished.stdout.total_lines,
                "stderr_lines": finished.stderr.total_lines,
                "shown_lines": shown,
                "omitted_lines": omitted,
                "stdout_bytes": finished.stdout.bytes,
                "stderr_bytes": finished.stderr.bytes,
                "truncated": omitted > 0,
                "cwd": cwd.map(|path| path.display().to_string()),
                "artifact": artifact,
            }),
        )
    }

    /// Store the full output, if there is a store and anything worth keeping.
    ///
    /// Streamed from the capture files rather than assembled in memory: a 100 MB log is a normal
    /// thing for `find /` to produce, and the point of the artifact is that the *whole* thing is
    /// kept. Each piece is a `Read`, so `put_from` hashes and writes with a fixed buffer.
    fn spool(&self, command: &str, finished: &Finished) -> Option<String> {
        let store = self.artifacts.as_ref()?;
        if finished.stdout.bytes == 0 && finished.stderr.bytes == 0 {
            return None;
        }
        let mut header = String::new();
        let _ = writeln!(header, "$ {command}");
        let _ = writeln!(header, "--- stdout ---");
        let stdout = stream_of(&finished.stdout)?;
        let stderr = stream_of(&finished.stderr)?;

        // `Read::chain` nests, so four pieces are three calls and no trait objects.
        let reader = std::io::Cursor::new(header.into_bytes())
            .chain(stdout)
            .chain(std::io::Cursor::new(b"\n--- stderr ---\n".to_vec()))
            .chain(stderr);
        store
            .put_from(reader)
            .ok()
            // `digest_text`, not bare hex: the observation carries `sha256:<64 hex>`, which is the
            // form `parse_digest` accepts and the form the journal writes.
            .map(|digest| immortality_core::journal::digest_text(&digest))
    }
}

/// The capture file as a reader, or an empty one when the stream was empty.
///
/// `chain` needs a reader for every piece, and an empty stream has a file that may legitimately be
/// zero bytes or may have been removed. Both mean "no bytes", so both produce an empty reader
/// rather than an error — the output is not the thing the step is reporting on.
fn stream_of(captured: &Captured) -> Option<Box<dyn std::io::Read>> {
    if captured.bytes == 0 {
        return Some(Box::new(std::io::empty()));
    }
    std::fs::File::open(&captured.path)
        .ok()
        .map(|file| Box::new(BufReader::new(file)) as Box<dyn std::io::Read>)
}

/// What a command run outside the model's tool loop produced.
///
/// The gate runner and `doctor` need the same guarantees `shell` gives the model — a closed
/// stdin, its own process group, the whole group killed on a deadline, output captured to files
/// instead of pipes — but they are not tool calls: there is no model to hand an observation to and
/// no policy decision to make here, because the caller already made it. Exposing the machinery
/// rather than a tool keeps those two callers from re-implementing the guards, which is how one of
/// them ends up hanging on a credential prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    /// `None` when the process was killed and never reported a code.
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub duration: Duration,
    /// The last [`SHOWN_LINES`] lines of stdout, then of stderr, already labelled.
    pub tail: String,
    pub stdout_lines: usize,
    pub stderr_lines: usize,
}

impl CommandOutput {
    /// Whether the command ran to completion and exited zero.
    #[must_use]
    pub fn succeeded(&self) -> bool {
        !self.timed_out && self.exit_code == Some(0)
    }
}

/// Run one command in `directory` with no stdin, a deadline and a killed process group.
///
/// `timeout` is clamped to [`MAX_TIMEOUT_SECS`]: the cap exists because a single gate command is
/// not allowed to turn a verification step into an hour of silence, and a caller that could pass
/// a larger number would be a caller that can.
///
/// # Errors
/// The command could not be spawned, or its capture files could not be created.
pub fn run_command(command: &str, directory: &Path, timeout: Duration) -> Result<CommandOutput, ToolError> {
    let timeout = timeout.clamp(Duration::from_secs(1), Duration::from_secs(MAX_TIMEOUT_SECS));
    let finished = run(command, directory, &[], timeout)?;
    let mut tail = String::new();
    for (label, captured) in [("stdout", &finished.stdout), ("stderr", &finished.stderr)] {
        if captured.bytes == 0 {
            continue;
        }
        if !tail.is_empty() {
            tail.push('\n');
        }
        let _ = writeln!(tail, "--- {label} ---");
        tail.push_str(&captured.tail);
        if captured.omitted > 0 {
            let _ = writeln!(tail, "… {} earlier line(s) not shown", captured.omitted);
        }
    }
    Ok(CommandOutput {
        exit_code: finished.status.and_then(|status| status.code()),
        timed_out: finished.timed_out,
        duration: finished.duration,
        tail,
        stdout_lines: finished.stdout.total_lines,
        stderr_lines: finished.stderr.total_lines,
    })
}

/// Wrap a command so its `cwd` and exit code survive the `sh` that ran it.
///
/// The braces matter: without them `cmd; __s=$?` in a command ending in `&` would put the
/// assignment in the background and the marker would never be written.
fn wrap_command(command: &str, marker: &Path) -> String {
    format!(
        "{{ {command}\n}}; __immortality_status=$?\nprintf '%s' \"$PWD\" > '{}'\nexit $__immortality_status",
        marker.display()
    )
}

pub(crate) fn read_env(args: &Args) -> Result<Vec<(String, String)>, ToolError> {
    let Some(value) = args.raw().get("env") else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let Some(map) = value.as_object() else {
        return Err(ToolError::Arguments(
            "`env` must be an object of string values".to_string(),
        ));
    };
    let mut out = Vec::with_capacity(map.len());
    for (key, value) in map {
        // A key with `=` or a NUL cannot be passed to `execve`, and silently dropping it would
        // make the command see a different environment than the one it asked for.
        if key.is_empty() || key.contains('=') || key.contains('\0') {
            return Err(ToolError::Arguments(format!(
                "`env` key `{key}` is not a valid environment variable name"
            )));
        }
        let Some(text) = value.as_str() else {
            return Err(ToolError::Arguments(format!("`env.{key}` must be a string")));
        };
        out.push((key.clone(), text.to_string()));
    }
    Ok(out)
}

/// Run the wrapped script, killing the whole process group on timeout.
fn run(
    script: &str,
    directory: &Path,
    env: &[(String, String)],
    timeout: Duration,
) -> Result<Finished, ToolError> {
    let started = Instant::now();
    let scratch = Scratch::new();
    let out_file =
        std::fs::File::create(&scratch.stdout).map_err(|error| ToolError::io(&scratch.stdout, &error))?;
    let err_file =
        std::fs::File::create(&scratch.stderr).map_err(|error| ToolError::io(&scratch.stderr, &error))?;

    let mut command = Command::new("/bin/sh");
    command
        .arg("-c")
        .arg(script)
        .current_dir(directory)
        // The line this file exists for: a command that waits for input gets EOF and fails in a
        // second instead of hanging the agent until something else notices.
        .stdin(Stdio::null())
        // Files, not pipes. A pipe needs a reader thread and blocks once it fills; a background
        // child inherits the write end and then the reader waits for a process the step never
        // asked to wait for. A file cannot do any of that, and 100 MB costs no memory.
        .stdout(Stdio::from(out_file))
        .stderr(Stdio::from(err_file))
        // Its own group, so the timeout can kill grandchildren. Measured: `sh -c 'a; b'` leaves
        // `sh` resident, and killing only the direct child orphans the command it started.
        .process_group(0);
    for (key, value) in env {
        command.env(key, value);
    }
    command.env("IMMORTALITY_SHELL", "1");

    let mut child = command
        .spawn()
        .map_err(|error| ToolError::io(directory, &error))?;
    let pid = child.id();

    let deadline = started + timeout;
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    timed_out = true;
                    terminate_group(pid);
                    break reap(&mut child, directory)?;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => return Err(ToolError::io(directory, &error)),
        }
    };

    // Anything the command backgrounded is still in the group, and `shell` hands the model no
    // handle to it — so the agent could never stop it, and a `cmd &` typo would accumulate
    // processes for the days this daemon is meant to run. It is killed, and the observation says
    // so, because a silently disappearing background job is its own kind of lie.
    //
    // `timed_out` already killed the group, so this only fires on the normal path.
    let leftovers_killed = !timed_out && group_alive(pid);
    if leftovers_killed {
        terminate_group(pid);
    }

    Ok(Finished {
        status,
        stdout: summarise(&scratch.stdout),
        stderr: summarise(&scratch.stderr),
        timed_out,
        duration: started.elapsed(),
        leftovers_killed,
        _scratch: scratch,
    })
}

/// Measure a capture file and read the lines worth showing.
///
/// The tail is found by seeking, not by reading: the whole point of capturing to a file is that a
/// hundred megabytes never enter memory, and reading it all to take the last two hundred lines
/// would throw that away. A single line longer than [`TAIL_WINDOW`] is cut to its end, which is
/// the part with the error in it.
fn summarise(path: &Path) -> Captured {
    // The path is recorded even for an empty stream: `spool` still has to know whether there is
    // a file to open, and a missing path and an empty file are not the same thing.
    let mut captured = Captured {
        path: path.to_path_buf(),
        ..Captured::default()
    };
    let Ok(mut file) = std::fs::File::open(path) else {
        return captured;
    };
    let bytes = file.metadata().map_or(0, |meta| meta.len());
    if bytes == 0 {
        return captured;
    }

    let window = bytes.min(TAIL_WINDOW);
    let mut buffer = Vec::with_capacity(window as usize);
    if file.seek(SeekFrom::Start(bytes - window)).is_ok() {
        let _ = file.take(window).read_to_end(&mut buffer);
    }

    // A window that starts mid-line would show a partial first line as if it were whole, so drop
    // everything up to the first newline — unless the whole file fits, in which case there is no
    // partial line to drop.
    let text = if window == bytes {
        String::from_utf8_lossy(&buffer).into_owned()
    } else {
        buffer.iter().position(|byte| *byte == b'\n').map_or_else(
            || String::from_utf8_lossy(&buffer).into_owned(),
            |newline| String::from_utf8_lossy(&buffer[newline + 1..]).into_owned(),
        )
    };

    let all: Vec<&str> = text.lines().collect();
    let start = all.len().saturating_sub(SHOWN_LINES);
    let mut tail = String::new();
    for line in all.get(start..).unwrap_or_default() {
        tail.push_str(line);
        tail.push('\n');
    }

    captured.bytes = bytes;
    captured.total_lines = all.len();
    captured.shown = all.len() - start;
    captured.omitted = start;
    captured.tail = tail;
    captured
}

/// Wait for the direct child after a kill, without ever blocking on it for long.
///
/// `child.wait()` is unbounded, so a process that survived both signals — stuck in an
/// uninterruptible kernel call, or simply not ours to kill — would hold the step open forever.
/// That is the hang this project exists to prevent, and it must not be reachable by a command a
/// model can write. A status we cannot obtain is reported as a signal death, which is the honest
/// description of "it was killed and did not report".
fn reap(child: &mut std::process::Child, directory: &Path) -> Result<Option<ExitStatus>, ToolError> {
    let deadline = Instant::now() + KILL_GRACE;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(Some(status)),
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(25));
            }
            // Nothing left to do but stop waiting: the group was signalled twice, and blocking
            // here is the hang this function exists to rule out.
            Ok(None) => return Ok(None),
            Err(error) => return Err(ToolError::io(directory, &error)),
        }
    }
}

/// `TERM` the group, wait out the grace period, then `KILL` what is left.
///
/// The group, not the process: see the module docs. `TERM` first so a well-behaved child can
/// flush and remove its temp files; `KILL` after the grace period because a process ignoring
/// `TERM` must not own the step.
///
/// `kill` is spawned rather than called through `libc` because `unsafe_code` is forbidden
/// crate-wide, and a few hundred microseconds once per timeout is not worth an exception.
fn terminate_group(pid: u32) {
    signal_group(pid, "-TERM");
    let grace = Instant::now();
    while grace.elapsed() < KILL_GRACE {
        if !group_alive(pid) {
            return;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    if group_alive(pid) {
        signal_group(pid, "-KILL");
    }
}

fn signal_group(pid: u32, signal: &str) {
    let _ = Command::new("kill")
        .args([signal, "--", &format!("-{pid}")])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

/// Whether any process remains in the group.
///
/// `kill -0` asks about existence without delivering anything, so this cannot itself kill a group
/// that has just been recycled for someone else's use.
///
/// `--` is load-bearing, and this is the same separator [`signal_group`] needs: procps `kill`
/// (Debian, and so the Linux CI image) reads a leading `-` in the operand as a *signal
/// specification* unless it follows `--`. `kill -0 -1234` therefore answers success for a group
/// that does not exist and delivers nothing to one that does. The effect was not a rare race but
/// a constant: `group_alive` was always true, so every finished command looked as if it had left
/// a child behind, the step waited out [`KILL_GRACE`] and the model was told a process had been
/// killed that never was. BSD `kill` on macOS accepts both spellings, so the separator is the
/// portable one.
fn group_alive(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", "--", &format!("-{pid}")])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// A per-call marker file, outside the worktree so it never shows up in `git status`.
fn marker_path() -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("immortality-shell-{}-{sequence}.cwd", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::PolicyGate;
    use std::io::Write as _;

    fn ctx(dir: &Path) -> ToolCtx {
        let gate = PolicyGate::with_defaults(
            dir,
            immortality_core::config::Policy {
                workspace_root: dir.parent().unwrap().to_path_buf(),
                ..immortality_core::config::Policy::default()
            },
        )
        .unwrap();
        ToolCtx::new(Arc::new(gate), crate::OutputBudget::default())
    }

    #[test]
    fn the_expected_effect_quotes_the_command() {
        // The classified paths are empty for `true` or `make`, so a path-derived sentence would
        // say only "shell". The operator's question is which command was about to run.
        let shell = Shell::standalone(None);
        let effect = shell.expected_effect(&Args::new(json!({"command": "cargo test --all"})));
        assert_eq!(effect, "run `cargo test --all`");
        let newlines = shell.expected_effect(&Args::new(json!({"command": "echo a\necho b"})));
        assert_eq!(newlines, "run `echo a echo b`", "one journal line, one line");
    }

    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("a.txt"), b"hello\n").unwrap();
        dir
    }

    fn run_shell(dir: &Path, shell: &Shell, args: Value) -> Result<ToolResult, ToolError> {
        shell.execute(&ctx(dir), Args::new(args))
    }

    #[test]
    fn a_command_runs_and_reports_its_output_and_code() {
        let dir = fixture();
        let shell = Shell::default();
        let result = run_shell(dir.path(), &shell, json!({"command": "echo hi"})).unwrap();
        assert!(result.observation.contains("hi"), "{}", result.observation);
        assert_eq!(result.data["exit_code"], 0);
        assert_eq!(result.data["timed_out"], false);
    }

    #[test]
    fn the_summary_is_the_first_line_so_a_reader_sees_the_verdict_before_the_output() {
        // The defect this pins: the summary used to be appended last, so the first line of every
        // result was a `--- stdout ---` label. `immortality log` renders only the first line, and
        // 97 of 100 shell calls in the `roomly` run therefore looked like they had no output.
        let dir = fixture();
        let shell = Shell::default();
        let result = run_shell(dir.path(), &shell, json!({"command": "echo hi"})).unwrap();
        // The first *non-empty* line, because the renderer skips blank lines too: `immortality
        // log` decides what a step says from the same line, and a leading newline must not be
        // able to hide the regression.
        let first = result
            .observation
            .lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or_default();
        assert!(
            first.starts_with("[exited 0 in "),
            "the verdict must come first, got {first:?}"
        );
        assert!(
            result.observation.contains("--- stdout ---"),
            "{}",
            result.observation
        );
    }

    #[test]
    fn a_pipeline_says_that_its_exit_code_is_the_last_command_s() {
        // Measured on the `roomly` run: nine results carried `16 examples, 16 failures` next to
        // `exited 0`, because the model wrote `rspec … | tail -30` and `/bin/sh` reports the
        // pipeline's *last* status. D-116 refuses such a command as a `DoD` check; the tool has to
        // run what the model asks, so it names the trap instead.
        let dir = fixture();
        let shell = Shell::default();
        let result = run_shell(dir.path(), &shell, json!({"command": "echo hi | tail -1"})).unwrap();
        assert_eq!(result.data["exit_code"], 0);
        assert!(
            result.observation.contains("is a pipeline"),
            "a masked exit code must be named: {}",
            result.observation
        );
        assert!(
            result
                .observation
                .contains("last command, not of the whole pipeline"),
            "{}",
            result.observation
        );
    }

    #[test]
    fn a_command_without_a_pipe_is_not_warned_about() {
        // The warning has to be rare enough to mean something. A `||` is a deliberate short
        // circuit, not a masked status, and must not trigger it either.
        let dir = fixture();
        let shell = Shell::default();
        for command in ["echo hi", "false || echo recovered"] {
            let result = run_shell(dir.path(), &shell, json!({"command": command})).unwrap();
            assert!(
                !result.observation.contains("is a pipeline"),
                "{command} must not be flagged: {}",
                result.observation
            );
        }
    }

    #[test]
    fn a_non_zero_exit_is_a_result_not_an_error() {
        let dir = fixture();
        let shell = Shell::default();
        // `false`, not `exit 3`: an unknown program is escalated by the classifier by design, and
        // a builtin that ends the shell before the marker is written would test two things at once.
        let result = run_shell(dir.path(), &shell, json!({"command": "false"})).unwrap();
        assert_eq!(result.data["exit_code"], 1);
        assert!(result.observation.contains("exited 1"), "{}", result.observation);
    }

    #[test]
    fn stderr_is_reported_separately_from_stdout() {
        let dir = fixture();
        let shell = Shell::default();
        let result = run_shell(dir.path(), &shell, json!({"command": "echo out; echo err >&2"})).unwrap();
        assert!(
            result.observation.contains("--- stdout ---"),
            "{}",
            result.observation
        );
        assert!(
            result.observation.contains("--- stderr ---"),
            "{}",
            result.observation
        );
    }

    #[test]
    fn a_command_that_reads_stdin_gets_eof_instead_of_hanging() {
        // The most valuable behaviour in the tool: `cat` with no file waits for input forever on
        // a terminal. Here it must finish immediately, and the whole step must not take seconds.
        let dir = fixture();
        let shell = Shell::default();
        let started = Instant::now();
        let result = run_shell(dir.path(), &shell, json!({"command": "cat"})).unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "took {:?}",
            started.elapsed()
        );
        assert_eq!(result.data["exit_code"], 0, "cat sees EOF and exits cleanly");
    }

    #[test]
    fn a_sleeping_command_is_killed_at_the_timeout() {
        let dir = fixture();
        let shell = Shell::default();
        let started = Instant::now();
        let result = run_shell(
            dir.path(),
            &shell,
            json!({"command": "sleep 300", "timeout_secs": 1}),
        )
        .unwrap();
        assert_eq!(result.data["timed_out"], true);
        assert!(
            started.elapsed() < Duration::from_secs(15),
            "the grace period leaked: {:?}",
            started.elapsed()
        );
        assert!(
            result.observation.contains("was killed"),
            "{}",
            result.observation
        );
    }

    #[test]
    fn a_killed_grandchild_does_not_survive_the_step() {
        // Measured failure mode: `sh -c 'sleep 400; echo x'` leaves `sh` resident with the sleep as
        // its child, so killing only the direct child orphans the grandchild — which keeps running
        // and keeps holding whatever the step was supposed to have released.
        //
        // The grandchild records its own pid, because `pgrep -f` cannot tell one `sleep 400` from
        // another and a pattern matching the *script* would only prove the `sh` died.
        let dir = fixture();
        let shell = Shell::default();
        let pid_file = dir.path().join("grandchild.pid");
        let command = format!("sleep 400 & echo $! > '{}'; wait", pid_file.display());
        let result = run_shell(dir.path(), &shell, json!({"command": command, "timeout_secs": 1})).unwrap();
        assert_eq!(result.data["timed_out"], true);

        let pid: i32 = std::fs::read_to_string(&pid_file)
            .expect("the grandchild wrote its pid before the timeout")
            .trim()
            .parse()
            .expect("a pid");
        assert!(pid > 0, "pid was {pid}");

        // Reaping takes a moment and is not the thing under test; the thing under test is that
        // the process is not still *running*.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let running = Command::new("ps")
                .args(["-p", &pid.to_string(), "-o", "stat="])
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .output()
                .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
                .unwrap_or_default();
            // Empty means gone; `Z` means dead-but-unreaped, which is also not running.
            if running.is_empty() || running.starts_with('Z') {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the grandchild (pid {pid}) survived the timeout: stat `{running}`"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    #[test]
    fn a_background_child_does_not_hold_the_step_open() {
        // Found by sabotage, not by design: killing only the direct child leaves the grandchild
        // holding the stdout pipe, and an unbounded reader join then waits for it. But the same
        // hang is reachable with *correct* cleanup — `sleep 400 & echo hi` is a legitimate command
        // whose child outlives the shell, and the step must still return in seconds.
        let dir = fixture();
        let shell = Shell::default();
        let pid_file = dir.path().join("bg.pid");
        let started = Instant::now();
        let result = run_shell(
            dir.path(),
            &shell,
            json!({
                "command": format!("sleep 400 & echo $! > '{}'; echo started", pid_file.display()),
                "timeout_secs": 30
            }),
        )
        .unwrap();
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_secs(20),
            "a background child held the step open for {elapsed:?}"
        );
        assert!(
            !result.data["timed_out"].as_bool().unwrap_or(true),
            "the foreground finished, so this must not read as a timeout: {:?}",
            result.data
        );

        // The leaked child is then cleaned up by the group kill that follows the reader timeout,
        // which is the behaviour that makes this safe rather than merely non-blocking.
        if let Ok(text) = std::fs::read_to_string(&pid_file)
            && let Ok(pid) = text.trim().parse::<i32>()
        {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                let alive = Command::new("kill")
                    .args(["-0", &pid.to_string()])
                    .stderr(Stdio::null())
                    .status()
                    .is_ok_and(|status| status.success());
                if !alive || Instant::now() >= deadline {
                    assert!(!alive, "the background child (pid {pid}) was never cleaned up");
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }

    #[test]
    fn the_reported_cwd_is_the_directory_the_command_ran_in() {
        // `cwd` used to be hard-coded to `None`, which made the field a lie rather than a gap.
        let dir = fixture();
        let shell = Shell::standalone(None);
        let result = run_shell(dir.path(), &shell, json!({"command": "cd sub"})).unwrap();
        let reported = result.data["cwd"].as_str().unwrap_or_default();
        assert!(
            reported.ends_with("sub"),
            "the observation should name the directory that survives: {reported:?}"
        );
    }

    #[test]
    fn cwd_is_remembered_between_calls() {
        // The replacement for a persistent shell: `cd` inside one command is captured by the
        // marker and becomes the next call's starting directory.
        let dir = fixture();
        let shell = Shell::standalone(None);
        run_shell(dir.path(), &shell, json!({"command": "cd sub"})).unwrap();
        let result = run_shell(dir.path(), &shell, json!({"command": "pwd"})).unwrap();
        assert!(
            result.observation.contains("sub"),
            "the second call did not inherit cwd: {}",
            result.observation
        );
    }

    #[test]
    fn an_explicit_cwd_wins_over_the_remembered_one() {
        let dir = fixture();
        let shell = Shell::standalone(None);
        run_shell(dir.path(), &shell, json!({"command": "cd sub"})).unwrap();
        let result = run_shell(dir.path(), &shell, json!({"command": "pwd", "cwd": "."})).unwrap();
        assert!(
            !result.observation.contains("/sub"),
            "explicit cwd was ignored: {}",
            result.observation
        );
    }

    #[test]
    fn a_failed_cd_does_not_move_the_session() {
        let dir = fixture();
        let shell = Shell::standalone(None);
        // A *relative* miss, so the failure is `cd`'s and not the policy's: an absolute path
        // outside the worktree is refused before the shell ever runs, which would make this test
        // about the gate instead of about the marker.
        let result = run_shell(dir.path(), &shell, json!({"command": "cd missing-dir"})).unwrap();
        assert_ne!(result.data["exit_code"], 0);
        let after = run_shell(dir.path(), &shell, json!({"command": "pwd"})).unwrap();
        assert!(
            !after.observation.contains("missing-dir"),
            "{}",
            after.observation
        );
    }

    #[test]
    fn env_passed_explicitly_reaches_the_command() {
        let dir = fixture();
        let shell = Shell::default();
        let result = run_shell(
            dir.path(),
            &shell,
            json!({"command": "printf '%s' \"$MY_VAR\"", "env": {"MY_VAR": "value-42"}}),
        )
        .unwrap();
        assert!(result.observation.contains("value-42"), "{}", result.observation);
    }

    #[test]
    fn env_does_not_leak_into_the_next_call() {
        // Documented limitation, pinned so it cannot change silently: `sh -c` cannot export.
        //
        // The marker is a word rather than the digit `1`, because the observation carries a
        // summary of its own — `[exited 0 in 0.1 s, 0 line(s) of stdout, 0 of stderr]` — whose
        // timing is made of digits. Asserting on `'1'` made this test fail whenever the command
        // happened to finish in a tenth of a second, which is on a fast machine nearly always.
        let dir = fixture();
        let shell = Shell::standalone(None);
        run_shell(
            dir.path(),
            &shell,
            json!({"command": "true", "env": {"LEAK": "leaked-value"}}),
        )
        .unwrap();
        let result = run_shell(dir.path(), &shell, json!({"command": "printf '%s' \"$LEAK\""})).unwrap();
        assert!(
            !result.observation.contains("leaked-value"),
            "{}",
            result.observation
        );
        assert_eq!(
            result.data["stdout_bytes"], 0,
            "the second call printed something, so it saw an environment it should not have: {}",
            result.observation
        );
    }

    #[test]
    fn a_bad_env_key_is_refused_by_name() {
        let dir = fixture();
        let shell = Shell::default();
        let error = run_shell(
            dir.path(),
            &shell,
            json!({"command": "true", "env": {"A=B": "x"}}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("`A=B`"), "{error}");
    }

    #[test]
    fn an_empty_command_is_refused() {
        let dir = fixture();
        let shell = Shell::default();
        assert!(run_shell(dir.path(), &shell, json!({"command": "   "})).is_err());
    }

    #[test]
    fn the_timeout_is_clamped_to_the_cap() {
        let dir = fixture();
        let shell = Shell::default();
        // Asking for an hour must not produce an hour-long step; it produces a fast command with
        // the clamped timeout, which is indistinguishable from success here.
        let result = run_shell(
            dir.path(),
            &shell,
            json!({"command": "true", "timeout_secs": 999_999}),
        )
        .unwrap();
        assert_eq!(result.data["exit_code"], 0);
    }

    #[test]
    fn a_large_output_is_truncated_to_the_last_lines_and_says_so() {
        let dir = fixture();
        let store = ArtifactStore::open(dir.path().join("artifacts")).unwrap();
        let shell = Shell::standalone(Some(store));
        // 1000 lines, the error on the last one — the shape of a real build log.
        let result = run_shell(
            dir.path(),
            &shell,
            json!({"command": "seq 1 1000; echo 'error: the last line'"}),
        )
        .unwrap();
        assert!(
            result.observation.contains("the last line"),
            "{}",
            result.observation
        );
        assert!(!result.observation.contains("\n1\n"), "the head must be dropped");
        assert!(
            result.data["omitted_lines"].as_u64().unwrap() > 0,
            "{:?}",
            result.data
        );
        assert!(result.data["artifact"].is_string(), "the full log must be stored");
    }

    #[test]
    fn the_artifact_of_a_large_output_is_readable_and_complete() {
        let dir = fixture();
        let store = ArtifactStore::open(dir.path().join("artifacts")).unwrap();
        let shell = Shell::standalone(Some(store.clone()));
        let result = run_shell(dir.path(), &shell, json!({"command": "seq 1 500; echo tail"})).unwrap();
        let digest = result.data["artifact"].as_str().unwrap();
        let digest = immortality_core::journal::parse_digest(digest).unwrap();
        let text = store.read_text(&digest).unwrap();
        assert!(text.contains("1\n"), "the head is kept in the artifact");
        assert!(text.contains("tail"), "the tail is kept too");
        assert!(
            text.contains("$ seq 1 500"),
            "the command is recorded in the artifact"
        );
    }

    #[test]
    fn without_a_store_a_large_output_still_reports_its_tail() {
        let dir = fixture();
        let shell = Shell::standalone(None);
        let result = run_shell(dir.path(), &shell, json!({"command": "seq 1 400; echo final"})).unwrap();
        assert!(result.observation.contains("final"), "{}", result.observation);
        assert_eq!(result.data["artifact"], Value::Null);
    }

    #[test]
    fn a_command_outside_the_worktree_is_refused_by_the_gate() {
        let dir = fixture();
        let shell = Shell::default();
        let error = run_shell(dir.path(), &shell, json!({"command": "cat ../../etc/passwd"})).unwrap_err();
        assert!(error.to_string().contains("denied_by_policy"), "{error}");
    }

    #[test]
    fn an_external_command_is_refused_not_run() {
        let dir = fixture();
        let shell = Shell::default();
        let error =
            run_shell(dir.path(), &shell, json!({"command": "curl https://example.com"})).unwrap_err();
        assert!(error.to_string().contains("awaiting_human_approval"), "{error}");
    }

    #[test]
    fn an_unknown_command_is_refused_as_external() {
        let dir = fixture();
        let shell = Shell::default();
        let error = run_shell(dir.path(), &shell, json!({"command": "frobnicate --all"})).unwrap_err();
        assert!(error.to_string().contains("awaiting_human_approval"), "{error}");
    }

    #[test]
    fn cwd_outside_the_worktree_is_refused() {
        let dir = fixture();
        let shell = Shell::default();
        let error = run_shell(dir.path(), &shell, json!({"command": "pwd", "cwd": "../.."})).unwrap_err();
        assert!(error.to_string().contains("denied_by_policy"), "{error}");
    }

    #[test]
    fn the_effect_follows_the_command() {
        let shell = Shell::default();
        assert_eq!(shell.effect(&Args::new(json!({"command": "ls"}))), Effect::Read);
        assert_eq!(
            shell.effect(&Args::new(json!({"command": "mkdir x"}))),
            Effect::Mutate
        );
        assert_eq!(
            shell.effect(&Args::new(json!({"command": "curl https://x"}))),
            Effect::External
        );
    }

    #[test]
    fn idempotency_is_unknown_so_recovery_probes() {
        // D-026: a `shell` call that may have run must be probed, never replayed.
        assert_eq!(Shell::default().idempotency(&Args::empty()), Idempotency::Unknown);
    }

    #[test]
    fn the_schema_is_well_formed() {
        assert_eq!(
            immortality_llm::schema::schema_defects(Shell::default().schema()),
            Vec::new()
        );
    }

    #[test]
    fn the_declared_paths_include_the_command_targets() {
        let shell = Shell::default();
        let paths = shell.paths(&Args::new(json!({"command": "rm -rf target"})));
        assert!(
            paths.iter().any(|used| used.path == "target" && used.write),
            "{paths:?}"
        );
    }

    #[test]
    fn summarise_reads_the_tail_without_reading_the_whole_file() {
        // The property that makes the file capture worth having: a hundred megabytes must cost
        // one seek and one window, not a hundred megabytes of memory.
        let dir = fixture();
        let path = dir.path().join("big.log");
        let mut file = std::fs::File::create(&path).unwrap();
        for n in 1..=50_000u32 {
            writeln!(file, "line {n}").unwrap();
        }
        file.write_all(b"error: the one that matters\n").unwrap();
        drop(file);

        let captured = summarise(&path);
        assert!(
            captured.tail.contains("the one that matters"),
            "{}",
            captured.tail
        );
        assert!(!captured.tail.contains("line 1\n"), "the head must not be read");
        assert!(captured.omitted > 0, "{}", captured.omitted);
        assert!(captured.shown <= SHOWN_LINES, "{}", captured.shown);
        assert_eq!(captured.bytes, std::fs::metadata(&path).unwrap().len());
    }

    #[test]
    fn a_finished_command_does_not_look_as_if_it_left_a_child_behind() {
        // The bug this pins: procps `kill` (Debian, and so the Linux CI image) reads a leading
        // `-` in the operand as a signal name unless it follows `--`, so `kill -0 -<pid>` answered
        // success for a group that no longer existed. `group_alive` then said "still running" for
        // every finished command, and each step paid `KILL_GRACE` — five seconds of silence and a
        // false "a background process has been killed" in the observation.
        //
        // The duration assertion is what makes this a test of the separator rather than of
        // `group_alive`'s return value alone: without `--` the code is not merely wrong about
        // liveness, it is slow in a way the model experiences as a stalled tool.
        let dir = fixture();
        let shell = Shell::default();
        let started = Instant::now();
        let result = run_shell(dir.path(), &shell, json!({"command": "printf hi"})).unwrap();
        let elapsed = started.elapsed();
        assert!(
            !result.observation.contains("left a background process running"),
            "a plain command reported a background child it never spawned: {}",
            result.observation
        );
        assert!(
            elapsed < KILL_GRACE,
            "a finished command took {elapsed:?}, which is the group-liveness grace period"
        );
    }

    #[test]
    fn a_terminated_group_is_reported_dead() {
        // The other half of the same separator bug, asserted directly on the helper: a group that
        // cannot exist must not read as alive, on any `kill`.
        let missing = 4_000_000;
        assert!(
            !group_alive(missing),
            "a group that cannot exist is reported alive; `kill` is probably parsing `-{missing}` \
             as a signal specification"
        );
    }

    #[test]
    fn summarise_of_a_short_file_keeps_every_line() {
        let dir = fixture();
        let path = dir.path().join("small.log");
        std::fs::write(&path, b"one\ntwo\nthree\n").unwrap();
        let captured = summarise(&path);
        assert_eq!(captured.omitted, 0);
        assert_eq!(captured.shown, 3);
        assert_eq!(captured.tail, "one\ntwo\nthree\n");
    }

    #[test]
    fn an_empty_output_summarises_to_nothing() {
        let dir = fixture();
        let path = dir.path().join("empty.log");
        std::fs::write(&path, b"").unwrap();
        let captured = summarise(&path);
        assert_eq!(captured.bytes, 0);
        assert_eq!(captured.tail, "");
        assert_eq!(captured.total_lines, 0);
    }

    #[test]
    fn a_hundred_megabytes_is_captured_without_being_held() {
        // The roadmap's explicit case. `yes` is used because it produces output faster than any
        // reader could think about it, and the assertion is that the step still finishes with a
        // correct count and the last line — not that the bytes were buffered somewhere.
        let dir = fixture();
        let store = ArtifactStore::open(dir.path().join("artifacts")).unwrap();
        let shell = Shell::standalone(Some(store.clone()));
        let started = Instant::now();
        let result = run_shell(
            dir.path(),
            &shell,
            json!({
                // `seq`, not `yes`: the classifier knows `seq` is a read, while `yes` is an unknown
                // program and is escalated by design. The point here is capture, not policy.
                "command": "seq 1 20000000; echo END",
                "timeout_secs": 120
            }),
        )
        .unwrap();
        let elapsed = started.elapsed();
        assert!(elapsed < Duration::from_secs(120), "took {elapsed:?}");
        assert_eq!(result.data["timed_out"], false);
        assert!(result.observation.contains("END"), "{}", result.observation);
        let bytes = result.data["stdout_bytes"].as_u64().unwrap();
        assert!(
            bytes > 100_000_000,
            "the capture must hold the whole stream: {bytes} bytes"
        );

        // And the artifact really holds all of it, which is the promise the observation makes.
        let digest = result.data["artifact"].as_str().unwrap();
        let digest = immortality_core::journal::parse_digest(digest).unwrap();
        let bytes = store.read(&digest).unwrap();
        assert!(
            bytes.len() > 100_000_000,
            "the artifact must hold the whole stream, not a window: {} bytes",
            bytes.len()
        );
    }

    #[test]
    fn the_marker_wrapper_preserves_the_exit_code() {
        let marker = marker_path();
        let script = wrap_command("exit 7", &marker);
        let status = Command::new("/bin/sh").arg("-c").arg(&script).status().unwrap();
        assert_eq!(status.code(), Some(7));
        let _ = std::fs::remove_file(&marker);
    }
    // ── run_command: the seam the gate runner uses ────────────────────────────

    #[test]
    fn run_command_reports_the_exit_code_and_the_output_tail() {
        let dir = tempfile::tempdir().unwrap();
        let out = run_command(
            "printf 'error[E0308]: mismatched types\\n' >&2; exit 101",
            dir.path(),
            Duration::from_secs(10),
        )
        .unwrap();
        assert_eq!(out.exit_code, Some(101));
        assert!(!out.succeeded());
        assert!(out.tail.contains("error[E0308]"), "{}", out.tail);
        assert_eq!(out.stderr_lines, 1);
    }

    #[test]
    fn run_command_treats_a_closed_stdin_as_eof() {
        // The whole reason the tool layer owns the spawner: `cat` here must not wait for input.
        let dir = tempfile::tempdir().unwrap();
        let out = run_command("cat; echo done", dir.path(), Duration::from_secs(10)).unwrap();
        assert!(out.succeeded(), "{out:?}");
        assert!(out.duration < Duration::from_secs(5), "{:?}", out.duration);
    }

    #[test]
    fn run_command_kills_the_whole_group_on_timeout() {
        // `sh -c 'sleep 60; echo after'` keeps `sh` resident with `sleep` as a child. Killing only
        // the direct child would orphan the sleep and let it outlive the check.
        let dir = tempfile::tempdir().unwrap();
        let out = run_command("sleep 60; echo never", dir.path(), Duration::from_secs(1)).unwrap();
        assert!(out.timed_out);
        assert!(!out.succeeded());
        assert!(!out.tail.contains("never"), "{}", out.tail);
    }

    #[test]
    fn run_command_clamps_a_timeout_above_the_cap() {
        // A check that can ask for an hour is a check that can hold a verification step for an
        // hour; the cap is the point.
        let dir = tempfile::tempdir().unwrap();
        let out = run_command("true", dir.path(), Duration::from_secs(100_000)).unwrap();
        assert!(out.succeeded());
    }

    #[test]
    fn run_command_runs_in_the_directory_it_was_given() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("marker.txt"), "x").unwrap();
        let out = run_command(
            "test -f marker.txt && echo found",
            dir.path(),
            Duration::from_secs(10),
        )
        .unwrap();
        assert!(out.tail.contains("found"), "{}", out.tail);
    }
}
