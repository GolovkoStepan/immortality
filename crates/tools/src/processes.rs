//! Long-lived processes: `start_process`, `read_process`, `kill_process`.
//!
//! # Why this is a separate tool from `shell`
//!
//! `shell` waits, and a thing that waits has a deadline. A dev server, a watch mode, a test suite
//! that takes forty minutes — none of them fit inside a step, and forcing them into one produces
//! either a killed server or a step that blocks for forty minutes. So a long-lived process gets a
//! handle, and the model polls it.
//!
//! # Why the handle is not the pid
//!
//! A daemon outlives the process it starts, and a pid is recycled by the kernel. Handing a model
//! a pid would mean `kill_process` could kill an unrelated process after a restart. The handle is
//! ours, it is written into a record on disk, and it carries the pid only as an implementation
//! detail. `kill_process` re-checks that the process it is about to signal is still the one the
//! record names — and if that check is impossible, it refuses rather than guesses.
//!
//! # Why the marker is a comment in the command line, not an environment variable
//!
//! Recovery needs to answer "did the start happen?" with no in-memory state, after a crash, from
//! a journal record alone. The record carries the arguments digest, the digest determines a
//! marker string, and the probe searches `ps` for it.
//!
//! The design said "поиск процесса по маркеру в env", and that is what this file did first. It
//! does not work on macOS, and the way it fails is instructive: `ps -E` *appears* to work,
//! because a backgrounded `IMMORTALITY_PROC=x sleep 300` shows the string in `ps -A -E`. But
//! what `ps` was showing is the *wrapping shell's* command line — the literal text the human
//! typed — not the child's environment block. Spawning the child directly, with no shell in
//! between, gives `hit: False` for every `-E` spelling. A probe built on that would have
//! reported `Applied` whenever the agent itself had earlier written the same words in a command,
//! which is worse than no probe.
//!
//! What does work is putting the marker where `ps` genuinely reads: the command line. It cannot
//! be an argument, because arguments belong to the command, and it cannot be `$0`, because the
//! command's own `$0` is its to use. A comment is both: `sh -c '# immortality-proc-<hex>\n<cmd>'`
//! runs exactly `<cmd>` and shows the marker in `ps -A -o command=`. Measured: the marker
//! survives, `kill -TERM -<pgid>` clears it, and the command's `$0` is unchanged.
//!
//! The body is additionally wrapped in braces so `sh` stays resident. Without them a
//! single-command body `exec`s, the `sh` image is replaced, and the command line that carried the
//! marker is gone — the process would be unidentifiable exactly when it is running fine.
//!
//! # Why output is appended, not re-read
//!
//! `read_process` returns new bytes since the cursor, not the whole log again. A server that has
//! been running for a day would otherwise put a day of logs into one context, and the model would
//! spend its budget on the beginning of a log it has already read.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use immortality_core::protocol::Idempotency;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::Effect;
use crate::args::Args;
use crate::policy::PathUse;
use crate::tool::{Tool, ToolCtx, ToolError, ToolResult};

/// Bytes of new output `read_process` returns per call.
///
/// Smaller than the model's budget on purpose: a poll is cheap and frequent, and the point of a
/// cursor is that the model can come back. One call that returns everything makes the first poll
/// expensive and every later one useless.
pub const DEFAULT_READ_BYTES: usize = 16 * 1024;

/// Largest `read_process` window a model may ask for.
pub const MAX_READ_BYTES: usize = 64 * 1024;

/// How long a killed process gets before `KILL`.
pub const KILL_GRACE: Duration = Duration::from_secs(5);

/// The prefix of the marker comment that names a tracked process.
///
/// Appears in the `sh` command line, which is where `ps` can actually see it. Both the spawner and
/// the probe build it, so the constant lives here and not in two places.
pub const MARKER_PREFIX: &str = "immortality-proc-";

/// One tracked process, as stored on disk.
///
/// Persisted rather than held in memory because the daemon restarts: a handle the model wrote
/// down in its plan must still resolve afterwards, and a `kill_process` after a restart must
/// still kill the right process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessRecord {
    /// The model-facing handle, `proc_<n>`.
    pub handle: String,
    /// Kernel pid at the time of the start. Advisory: checked before signalling, never trusted
    /// alone.
    pub pid: u32,
    /// The marker comment written into the child's command line.
    pub marker: String,
    /// What was run, for the human reading `immortality log`.
    pub command: String,
    /// Where it ran.
    pub cwd: String,
    /// Capture files, kept after the process exits so the last output can still be read.
    pub stdout_path: String,
    pub stderr_path: String,
    /// Bytes of stdout and stderr the model has already been shown.
    pub stdout_cursor: u64,
    pub stderr_cursor: u64,
    /// When it started, for ordering and for the human.
    pub started: immortality_core::domain::Timestamp,
}

impl ProcessRecord {
    /// The marker for a call, derived from its arguments digest.
    ///
    /// Derived, not assigned, because a probe runs after a crash with nothing in memory: given
    /// the same journal record it must produce the same string. Shortened to twelve hex digits
    /// because it has to be visible in `ps` output without pushing the command line past a
    /// terminal width.
    #[must_use]
    pub fn marker_for(args_digest: &str) -> String {
        let hex = args_digest.strip_prefix("sha256:").unwrap_or(args_digest);
        format!("{MARKER_PREFIX}{}", hex.get(..16).unwrap_or(hex))
    }
}

/// Where tracked processes live and how they are found.
///
/// A directory of one JSON record per handle plus two capture files. A directory rather than a
/// single file because two processes starting at once would otherwise rewrite each other's entry,
/// and a plain JSON file per handle is something a human can `cat` while debugging.
#[derive(Debug, Clone)]
pub struct ProcessRegistry {
    dir: PathBuf,
    /// Guards handle allocation so two concurrent starts cannot pick the same number.
    next: Arc<Mutex<u64>>,
}

impl ProcessRegistry {
    /// Open (or create) a registry under `dir`.
    ///
    /// The next handle number is the count of existing records plus one, so handles are never
    /// reused for a *different* process: `proc_3` naming a second process would let a stale plan
    /// kill something it never started.
    ///
    /// # Errors
    /// The directory cannot be created or read.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, ToolError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|error| ToolError::io(&dir, &error))?;
        let next = highest_handle(&dir) + 1;
        Ok(Self {
            dir,
            next: Arc::new(Mutex::new(next)),
        })
    }

    /// The directory holding the records.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn record_path(&self, handle: &str) -> PathBuf {
        self.dir.join(format!("{handle}.json"))
    }

    fn stdout_path(&self, handle: &str) -> PathBuf {
        self.dir.join(format!("{handle}.stdout"))
    }

    fn stderr_path(&self, handle: &str) -> PathBuf {
        self.dir.join(format!("{handle}.stderr"))
    }

    /// Allocate the next unused handle.
    fn allocate(&self) -> String {
        let mut guard = self
            .next
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let handle = format!("proc_{}", *guard);
        *guard += 1;
        handle
    }

    /// Load one record.
    ///
    /// # Errors
    /// The handle does not exist, or the record is not readable JSON.
    pub fn load(&self, handle: &str) -> Result<ProcessRecord, ToolError> {
        let path = self.record_path(handle);
        let bytes = std::fs::read(&path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                ToolError::Refused(
                    json!({
                        "error": "unknown_handle",
                        "handle": handle,
                        "hint": "No such tracked process. `read_process` and `kill_process` only \
                                 accept a handle from `start_process`."
                    })
                    .to_string(),
                )
            } else {
                ToolError::io(&path, &error)
            }
        })?;
        serde_json::from_slice(&bytes).map_err(|error| {
            ToolError::Refused(
                json!({
                    "error": "corrupt_record",
                    "handle": handle,
                    "detail": error.to_string(),
                })
                .to_string(),
            )
        })
    }

    /// Persist a record, atomically.
    ///
    /// # Errors
    /// The directory refused the write.
    pub fn save(&self, record: &ProcessRecord) -> Result<(), ToolError> {
        let path = self.record_path(&record.handle);
        let bytes = serde_json::to_vec_pretty(record).map_err(|error| {
            ToolError::Arguments(format!("the process record could not be serialized: {error}"))
        })?;
        crate::write::write_atomically(&path, &bytes)
    }

    /// Every handle, oldest first.
    ///
    /// # Errors
    /// The directory cannot be read.
    pub fn handles(&self) -> Result<Vec<String>, ToolError> {
        let mut handles: Vec<String> = std::fs::read_dir(&self.dir)
            .map_err(|error| ToolError::io(&self.dir, &error))?
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                let stem = name.strip_suffix(".json")?;
                stem.starts_with("proc_").then(|| stem.to_string())
            })
            .collect();
        handles.sort_by_key(|handle| handle_number(handle).unwrap_or(u64::MAX));
        Ok(handles)
    }

    /// Start a command and return its record without waiting for it.
    ///
    /// # Errors
    /// The process cannot be spawned, which is the only failure worth surfacing: everything else
    /// (the command itself failing) is a result the model reads.
    pub fn start(
        &self,
        command: &str,
        directory: &Path,
        env: &[(String, String)],
        args_digest: &str,
    ) -> Result<ProcessRecord, ToolError> {
        let handle = self.allocate();
        let marker = ProcessRecord::marker_for(args_digest);
        let stdout_path = self.stdout_path(&handle);
        let stderr_path = self.stderr_path(&handle);
        let out_file =
            std::fs::File::create(&stdout_path).map_err(|error| ToolError::io(&stdout_path, &error))?;
        let err_file =
            std::fs::File::create(&stderr_path).map_err(|error| ToolError::io(&stderr_path, &error))?;

        let script = marked_script(command, &marker);
        let mut child_command = std::process::Command::new("/bin/sh");
        child_command
            .arg("-c")
            .arg(&script)
            .current_dir(directory)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::from(out_file))
            .stderr(std::process::Stdio::from(err_file))
            // Its own group, so `kill_process` can take the whole tree. A dev server that spawns
            // a watcher is normal, and killing only the server leaves the watcher writing.
            //
            // Nothing here detaches the child from the daemon deliberately: no `wait` is ever
            // called on it, so it is reparented to launchd when the daemon exits, which is what
            // "long-lived" means.
            .process_group(0);
        for (key, value) in env {
            child_command.env(key, value);
        }

        let child = child_command
            .spawn()
            .map_err(|error| ToolError::io(directory, &error))?;
        let pid = child.id();
        // Deliberately not stored and never `wait`ed: dropping the handle leaves the child
        // running, which is what "long-lived" means. Zombies cannot accumulate because the
        // process is reparented to launchd when the daemon exits.
        drop(child);

        let record = ProcessRecord {
            handle,
            pid,
            marker,
            command: command.to_string(),
            cwd: directory.display().to_string(),
            stdout_path: stdout_path.display().to_string(),
            stderr_path: stderr_path.display().to_string(),
            stdout_cursor: 0,
            stderr_cursor: 0,
            started: immortality_core::time::now(),
        };
        self.save(&record)?;
        Ok(record)
    }
}

/// The numeric part of a handle, for ordering.
fn handle_number(handle: &str) -> Option<u64> {
    handle.strip_prefix("proc_")?.parse().ok()
}

/// The highest handle number already on disk, or 0.
fn highest_handle(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            handle_number(name.strip_suffix(".json")?)
        })
        .max()
        .unwrap_or(0)
}

/// Whether the process recorded by `handle` is still running.
///
/// Three questions, in order, and the order is the point:
///
/// 1. Is the pid alive at all? (`kill -0`)
/// 2. Is it still the process the record names? The marker is in its environment, and a recycled
///    pid would not have it. This is the check that makes signalling safe.
/// 3. If the marker cannot be read, the answer is `Unknown` — never "probably it".
#[must_use]
pub fn is_running(record: &ProcessRecord) -> Liveness {
    if !pid_alive(record.pid) {
        return Liveness::Dead;
    }
    match process_has_marker(record.pid, &record.marker) {
        Some(true) => Liveness::Running,
        Some(false) => Liveness::Dead,
        // The pid is alive but `ps` would not tell us what it is. Signalling now could kill a
        // stranger, so the caller is told it cannot tell.
        None => Liveness::Unknown,
    }
}

/// What the registry can say about a process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Liveness {
    /// Alive and identified as ours.
    Running,
    /// Gone, or the pid now belongs to something else.
    Dead,
    /// Alive but unidentifiable; do not signal.
    Unknown,
}

fn pid_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Whether `ps` shows the marker in a pid's environment.
///
/// `Some(false)` means the pid exists but is not ours; `None` means the environment could not be
/// read, which is different from and more dangerous than a negative answer.
fn process_has_marker(pid: u32, marker: &str) -> Option<bool> {
    let output = std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "command="])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Some(text.contains(marker))
}

/// The script actually handed to `sh`: a marker comment, then the body in braces.
///
/// The braces are load-bearing. `sh -c 'sleep 300'` replaces its own image with `sleep`, which
/// discards the command line that carries the marker; `sh -c '{ sleep 300; }'` keeps `sh`
/// resident and the marker visible for as long as the process lives.
#[must_use]
pub fn marked_script(command: &str, marker: &str) -> String {
    format!("# {marker}\n{{ {command}\n}}")
}

/// The digest a start's marker is derived from.
///
/// One definition, because there are two callers and they must agree *exactly*: the tool when it
/// spawns, and recovery when it probes. They disagreed while this was written out twice — the
/// tool hashed the resolved directory (`/private/var/...`) and the test hashed the unresolved one
/// (`/var/...`), so the probe searched for a marker nobody had written. `directory` here is
/// therefore expected to be the resolved path, which [`ToolCtx::enter_path`] returns.
#[must_use]
pub fn start_digest(command: &str, directory: &Path) -> String {
    immortality_core::journal::digest_text(&immortality_core::artifacts::digest_of(
        format!("{command}\u{0}{}", directory.display()).as_bytes(),
    ))
}

/// Probe whether a `start_process` call took effect, for crash recovery.
///
/// Takes the arguments digest rather than a record because recovery has a journal record and no
/// registry entry: that is exactly the state a crash between spawn and `save` leaves behind. The
/// marker is derived from the digest on both sides so the two agree without shared state.
#[must_use]
pub fn probe_started(args_digest: &str) -> immortality_core::protocol::Probe {
    let marker = ProcessRecord::marker_for(args_digest);
    match find_by_marker(&marker) {
        Some(_) => immortality_core::protocol::Probe::Applied,
        None => immortality_core::protocol::Probe::NotApplied,
    }
}

/// Any pid whose environment carries the marker.
fn find_by_marker(marker: &str) -> Option<u32> {
    let output = std::process::Command::new("ps")
        .args(["-A", "-o", "pid=,command="])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout).lines().find_map(|line| {
        if !line.contains(marker) {
            return None;
        }
        line.split_whitespace().next()?.parse().ok()
    })
}

// ── start_process ───────────────────────────────────────────────────────────

fn start_schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to run with `/bin/sh -c`. It keeps running after the call returns."
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory, relative to the worktree root. Defaults to the root."
                },
                "env": {
                    "type": "object",
                    "description": "Extra environment variables.",
                    "additionalProperties": { "type": "string" }
                },
                "name": {
                    "type": "string",
                    "description": "A short human label, echoed back so the human reading the log knows what it is."
                }
            },
            "required": ["command"],
            "additionalProperties": false
        })
    })
}

/// Start a long-lived process.
#[derive(Debug)]
pub struct StartProcess {
    registry: Arc<ProcessRegistry>,
    shell: Arc<crate::shell::ShellState>,
}

impl StartProcess {
    /// A tool backed by `registry`, sharing `shell`'s remembered directory.
    #[must_use]
    pub const fn new(registry: Arc<ProcessRegistry>, shell: Arc<crate::shell::ShellState>) -> Self {
        Self { registry, shell }
    }
}

impl Tool for StartProcess {
    fn name(&self) -> &'static str {
        "start_process"
    }

    fn schema(&self) -> &'static Value {
        start_schema()
    }

    fn description(&self) -> &'static str {
        "Start a long-lived process (a server, a watcher) and return a handle. Use `read_process` to poll it."
    }

    fn effect(&self, args: &Args) -> Effect {
        args.str("command").map_or(Effect::External, |command| {
            crate::classify::classify(command).effect
        })
    }

    /// `Unknown`, not `Idempotent`: a crash between spawn and record-write leaves a process the
    /// agent does not know about, and replaying the call would start a second one on the same
    /// port. Recovery probes by marker instead (see [`probe_started`]).
    fn idempotency(&self, _args: &Args) -> Idempotency {
        Idempotency::Unknown
    }

    fn paths(&self, args: &Args) -> Vec<PathUse> {
        let Ok(command) = args.str("command") else {
            return Vec::new();
        };
        let mut paths = crate::classify::classify(command).paths;
        if let Ok(Some(cwd)) = args.opt_str("cwd") {
            paths.push(PathUse::read(cwd));
        }
        paths
    }

    /// The command that is being started, since the tool's whole point is that it outlives the
    /// call: a handle in the journal is useless without the command behind it.
    fn expected_effect(&self, args: &Args) -> String {
        let command = args.str("command").unwrap_or("?");
        crate::tool::one_line(
            &format!("start a background command: `{command}`"),
            crate::tool::EFFECT_LIMIT,
        )
    }

    fn execute(&self, ctx: &ToolCtx, args: Args) -> Result<ToolResult, ToolError> {
        let command = args.str("command")?.trim().to_string();
        if command.is_empty() {
            return Err(ToolError::Arguments("`command` is empty".to_string()));
        }
        let name = args.opt_str("name")?.unwrap_or("").to_string();
        let env = crate::shell::read_env(&args)?;

        match ctx.gate().check_command(&command) {
            crate::Verdict::Allowed => {}
            crate::Verdict::Denied(denial) => return Err(ToolError::Refused(denial.observation())),
            crate::Verdict::Escalated(escalation) => {
                return Err(ToolError::Refused(escalation.observation()));
            }
        }

        let directory = match args.opt_str("cwd")? {
            Some(raw) => ctx.enter_path(raw)?,
            None => match self.shell.cwd() {
                Some(remembered) if remembered.is_dir() => remembered,
                _ => ctx.enter_path(".")?,
            },
        };

        // Built from the call's own arguments so that recovery, holding only the journal record,
        // computes the same string.
        let digest = start_digest(&command, &directory);
        let record = self.registry.start(&command, &directory, &env, &digest)?;

        let mut observation = String::new();
        // `command` moves into the record above, so the label is rebuilt from the record.
        let label = if name.is_empty() {
            record.command.clone()
        } else {
            format!("{name}: {}", record.command)
        };
        let _ = writeln!(
            observation,
            "Started `{label}` as {} in {}.",
            record.handle, record.cwd
        );
        let _ = writeln!(
            observation,
            "It keeps running. Poll it with `read_process` (handle `{}`), stop it with \
             `kill_process`.",
            record.handle
        );

        Ok(ToolResult::new(
            observation,
            json!({
                "handle": record.handle,
                "name": name,
                "pid": record.pid,
                "command": record.command,
                "cwd": record.cwd,
                "hint": "read_process",
            }),
        ))
    }
}

// ── read_process ────────────────────────────────────────────────────────────

fn read_schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "description": "Handle returned by `start_process`, e.g. `proc_3`."
                },
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_READ_BYTES,
                    "description": "Largest slice of new output to return. Defaults to 16384."
                },
                "stream": {
                    "type": "string",
                    "enum": ["stdout", "stderr", "both"],
                    "description": "Which stream to read. Defaults to `both`."
                }
            },
            "required": ["handle"],
            "additionalProperties": false
        })
    })
}

/// Poll a tracked process.
#[derive(Debug)]
pub struct ReadProcess {
    registry: Arc<ProcessRegistry>,
}

impl ReadProcess {
    /// A tool backed by `registry`.
    #[must_use]
    pub const fn new(registry: Arc<ProcessRegistry>) -> Self {
        Self { registry }
    }
}

impl Tool for ReadProcess {
    fn name(&self) -> &'static str {
        "read_process"
    }

    fn schema(&self) -> &'static Value {
        read_schema()
    }

    fn description(&self) -> &'static str {
        "Read output produced since the last poll, and whether the process is still running."
    }

    /// Reading is a read: it changes no file and the process state is not the tool's doing.
    fn effect(&self, _args: &Args) -> Effect {
        Effect::Read
    }

    /// Reading advances a cursor, so it is *not* idempotent — but the cursor is the daemon's
    /// bookkeeping and replaying a read loses output rather than duplicating an effect, so
    /// `ByContent` is the honest label: the effect is only the cursor, and it is recoverable by
    /// reading further.
    fn idempotency(&self, _args: &Args) -> Idempotency {
        Idempotency::ByContent
    }

    fn paths(&self, _args: &Args) -> Vec<PathUse> {
        Vec::new()
    }

    /// The handle, because there is no path in this call at all and `read_process` alone says
    /// nothing about which of several servers is being polled.
    fn expected_effect(&self, args: &Args) -> String {
        let handle = args.str("handle").unwrap_or("?");
        crate::tool::one_line(
            &format!("read the new output of `{handle}`"),
            crate::tool::EFFECT_LIMIT,
        )
    }

    fn execute(&self, _ctx: &ToolCtx, args: Args) -> Result<ToolResult, ToolError> {
        let handle = args.str("handle")?;
        let limit = args.opt_u64("max_bytes")?.map_or(DEFAULT_READ_BYTES, |value| {
            usize::try_from(value)
                .unwrap_or(DEFAULT_READ_BYTES)
                .min(MAX_READ_BYTES)
        });
        let stream = args.opt_str("stream")?.unwrap_or("both");
        let mut record = self.registry.load(handle)?;

        let liveness = is_running(&record);
        let (out_text, out_new, out_more) = read_since(
            Path::new(&record.stdout_path),
            record.stdout_cursor,
            if stream == "stderr" { 0 } else { limit },
        );
        let (err_text, err_new, err_more) = read_since(
            Path::new(&record.stderr_path),
            record.stderr_cursor,
            if stream == "stdout" { 0 } else { limit },
        );

        // The cursor advances only for bytes actually returned, so a full window does not skip
        // the output that came after it.
        if stream != "stderr" {
            record.stdout_cursor += out_new;
        }
        if stream != "stdout" {
            record.stderr_cursor += err_new;
        }
        // Tolerated: a failed save costs a duplicate read next poll, which is harmless. Refusing
        // the observation instead would hide output the model can already see.
        let _ = self.registry.save(&record);

        let mut observation = String::new();
        match liveness {
            Liveness::Running => {
                let _ = writeln!(observation, "{} is running.", record.handle);
            }
            Liveness::Dead => {
                let _ = writeln!(
                    observation,
                    "{} has exited. Output below is everything unread; further polls return nothing \
                     new.",
                    record.handle
                );
            }
            Liveness::Unknown => {
                let _ = writeln!(
                    observation,
                    "{} could not be identified — the pid is alive but does not carry this \
                     process's marker. Treat it as gone; do not expect live output.",
                    record.handle
                );
            }
        }
        if !out_text.is_empty() {
            let _ = writeln!(observation, "--- stdout ---");
            observation.push_str(&out_text);
            if !observation.ends_with('\n') {
                observation.push('\n');
            }
        }
        if !err_text.is_empty() {
            let _ = writeln!(observation, "--- stderr ---");
            observation.push_str(&err_text);
            if !observation.ends_with('\n') {
                observation.push('\n');
            }
        }
        if out_new == 0 && err_new == 0 {
            let _ = writeln!(observation, "(no new output)");
        }
        if out_more || err_more {
            let _ = writeln!(
                observation,
                "More is waiting: call `read_process` again with the same handle."
            );
        }

        Ok(ToolResult::new(
            observation,
            json!({
                "handle": record.handle,
                "running": liveness == Liveness::Running,
                "liveness": match liveness {
                    Liveness::Running => "running",
                    Liveness::Dead => "dead",
                    Liveness::Unknown => "unknown",
                },
                "stdout_bytes": out_new,
                "stderr_bytes": err_new,
                "stdout_cursor": record.stdout_cursor,
                "stderr_cursor": record.stderr_cursor,
                "has_more": out_more || err_more,
            }),
        ))
    }
}

/// Read from `cursor` to the end, at most `limit` bytes.
///
/// Returns the text, how many bytes it consumed, and whether more remains. Bytes are not split at
/// a character boundary: a limit that lands mid-UTF-8 is rounded down to the start of the next
/// character, so the cursor never leaves half a character behind for the next read to mangle.
fn read_since(path: &Path, cursor: u64, limit: usize) -> (String, u64, bool) {
    if limit == 0 {
        return (String::new(), 0, false);
    }
    let Ok(mut file) = std::fs::File::open(path) else {
        return (String::new(), 0, false);
    };
    let length = file.metadata().map_or(0, |meta| meta.len());
    if cursor >= length {
        return (String::new(), 0, false);
    }
    if file.seek(SeekFrom::Start(cursor)).is_err() {
        return (String::new(), 0, false);
    }
    let available = length - cursor;
    let take = usize::try_from(available.min(limit as u64)).unwrap_or(limit);
    let mut buffer = vec![0u8; take];
    let Ok(read) = file.read(&mut buffer) else {
        return (String::new(), 0, false);
    };
    buffer.truncate(read);

    // Round down to a character boundary so the text is whole and the cursor is not left in the
    // middle of a character.
    let mut consumed = read;
    while consumed > 0 && std::str::from_utf8(&buffer[..consumed]).is_err() {
        consumed -= 1;
    }
    let text = String::from_utf8_lossy(&buffer[..consumed]).into_owned();
    (text, consumed as u64, available > consumed as u64)
}

// ── kill_process ────────────────────────────────────────────────────────────

fn kill_schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "description": "Handle returned by `start_process`."
                }
            },
            "required": ["handle"],
            "additionalProperties": false
        })
    })
}

/// Stop a tracked process.
#[derive(Debug)]
pub struct KillProcess {
    registry: Arc<ProcessRegistry>,
}

impl KillProcess {
    /// A tool backed by `registry`.
    #[must_use]
    pub const fn new(registry: Arc<ProcessRegistry>) -> Self {
        Self { registry }
    }
}

impl Tool for KillProcess {
    fn name(&self) -> &'static str {
        "kill_process"
    }

    fn schema(&self) -> &'static Value {
        kill_schema()
    }

    fn description(&self) -> &'static str {
        "Stop a process started with `start_process`. Killing an already-dead process is not an error."
    }

    fn effect(&self, _args: &Args) -> Effect {
        Effect::Mutate
    }

    /// Idempotent: the second kill finds nothing and says so, which is the definition.
    fn idempotency(&self, _args: &Args) -> Idempotency {
        Idempotency::Idempotent
    }

    fn paths(&self, _args: &Args) -> Vec<PathUse> {
        Vec::new()
    }

    /// The handle: `kill_process` without it does not say what is about to die.
    fn expected_effect(&self, args: &Args) -> String {
        let handle = args.str("handle").unwrap_or("?");
        crate::tool::one_line(&format!("stop the process `{handle}`"), crate::tool::EFFECT_LIMIT)
    }

    fn execute(&self, _ctx: &ToolCtx, args: Args) -> Result<ToolResult, ToolError> {
        let handle = args.str("handle")?;
        let record = self.registry.load(handle)?;

        match is_running(&record) {
            Liveness::Dead => {
                return Ok(ToolResult::new(
                    format!("{} was already gone; nothing to kill.", record.handle),
                    json!({
                        "handle": record.handle,
                        "killed": false,
                        "already_dead": true,
                    }),
                ));
            }
            Liveness::Unknown => {
                // Refusing is the whole reason the marker exists. Signalling a pid we cannot
                // identify could kill a process that merely inherited the number.
                return Err(ToolError::Refused(
                    json!({
                        "error": "cannot_identify",
                        "handle": record.handle,
                        "pid": record.pid,
                        "hint": "The pid is alive but its environment does not carry this \
                                 process's marker, so it is probably an unrelated process that \
                                 reused the pid. Refusing to signal it. If the process is gone, \
                                 ignore this handle."
                    })
                    .to_string(),
                ));
            }
            Liveness::Running => {}
        }

        // The group, not the pid: a dev server that spawned a watcher must not leave the watcher
        // behind. `-pid` addresses the group started by `process_group(0)`.
        signal_group(record.pid, "-TERM");
        let deadline = Instant::now() + KILL_GRACE;
        while Instant::now() < deadline {
            if matches!(is_running(&record), Liveness::Dead) {
                return Ok(killed(&record));
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        signal_group(record.pid, "-KILL");

        Ok(killed(&record))
    }
}

fn killed(record: &ProcessRecord) -> ToolResult {
    ToolResult::new(
        format!(
            "Stopped {}. The recorded output is still readable with `read_process`.",
            record.handle
        ),
        json!({
            "handle": record.handle,
            "killed": true,
            "already_dead": false,
        }),
    )
}

fn signal_group(pid: u32, signal: &str) {
    let _ = std::process::Command::new("kill")
        .args([signal, &format!("-{pid}")])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

/// Everything this module registers, for a caller building the whole tool set.
#[must_use]
pub fn tools(registry: Arc<ProcessRegistry>, shell: Arc<crate::shell::ShellState>) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(StartProcess::new(Arc::clone(&registry), shell)),
        Arc::new(ReadProcess::new(Arc::clone(&registry))),
        Arc::new(KillProcess::new(registry)),
    ]
}

/// A `BTreeMap` of handles to liveness, for `immortality status`.
///
/// # Errors
/// The registry directory cannot be read.
pub fn liveness_of_all(registry: &ProcessRegistry) -> Result<BTreeMap<String, Liveness>, ToolError> {
    let mut out = BTreeMap::new();
    for handle in registry.handles()? {
        if let Ok(record) = registry.load(&handle) {
            out.insert(handle, is_running(&record));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::PolicyGate;

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

    fn fixture() -> (tempfile::TempDir, Arc<ProcessRegistry>) {
        let dir = tempfile::tempdir().unwrap();
        let registry = Arc::new(ProcessRegistry::open(dir.path().join("procs")).unwrap());
        (dir, registry)
    }

    fn start(dir: &Path, registry: &Arc<ProcessRegistry>, args: Value) -> Result<ToolResult, ToolError> {
        StartProcess::new(Arc::clone(registry), Arc::new(crate::shell::ShellState::new()))
            .execute(&ctx(dir), Args::new(args))
    }

    fn read(dir: &Path, registry: &Arc<ProcessRegistry>, args: Value) -> Result<ToolResult, ToolError> {
        ReadProcess::new(Arc::clone(registry)).execute(&ctx(dir), Args::new(args))
    }

    fn kill(dir: &Path, registry: &Arc<ProcessRegistry>, args: Value) -> Result<ToolResult, ToolError> {
        KillProcess::new(Arc::clone(registry)).execute(&ctx(dir), Args::new(args))
    }

    fn handle_of(result: &ToolResult) -> String {
        result.data["handle"].as_str().unwrap().to_string()
    }

    #[test]
    fn a_started_process_returns_a_handle_and_keeps_running() {
        let (dir, registry) = fixture();
        let result = start(
            dir.path(),
            &registry,
            json!({"command": "sleep 30", "name": "sleeper"}),
        )
        .unwrap();
        let handle = handle_of(&result);
        assert_eq!(handle, "proc_1");
        assert!(result.observation.contains("sleeper"), "{}", result.observation);

        let record = registry.load(&handle).unwrap();
        assert_eq!(is_running(&record), Liveness::Running);
        let _ = kill(dir.path(), &registry, json!({"handle": handle}));
    }

    #[test]
    fn the_output_of_a_started_process_is_read_incrementally() {
        // The cursor is the whole feature: a poll returns only what is new, so a process running
        // for an hour does not dump an hour of logs into one context.
        let (dir, registry) = fixture();
        let result = start(dir.path(), &registry, json!({"command": "echo first; sleep 30"})).unwrap();
        let handle = handle_of(&result);
        // Give the shell a moment to produce the line.
        std::thread::sleep(Duration::from_millis(300));

        let first = read(dir.path(), &registry, json!({"handle": handle})).unwrap();
        assert!(first.observation.contains("first"), "{}", first.observation);
        assert!(first.data["stdout_bytes"].as_u64().unwrap() > 0);

        let second = read(dir.path(), &registry, json!({"handle": handle})).unwrap();
        assert!(
            !second.observation.contains("first"),
            "a second poll must not repeat the first: {}",
            second.observation
        );
        assert!(
            second.observation.contains("no new output"),
            "{}",
            second.observation
        );
        let _ = kill(dir.path(), &registry, json!({"handle": handle}));
    }

    #[test]
    fn a_process_that_exits_between_calls_is_reported_dead() {
        let (dir, registry) = fixture();
        let result = start(dir.path(), &registry, json!({"command": "true"})).unwrap();
        let handle = handle_of(&result);

        // `true` exits immediately; the daemon may not have reaped it, so poll briefly.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let read_back = read(dir.path(), &registry, json!({"handle": handle})).unwrap();
            if read_back.data["running"] == serde_json::json!(false) {
                break;
            }
            assert!(Instant::now() < deadline, "the process never reported as dead");
            std::thread::sleep(Duration::from_millis(50));
        }

        let after = read(dir.path(), &registry, json!({"handle": handle})).unwrap();
        assert!(after.observation.contains("has exited"), "{}", after.observation);
    }

    #[test]
    fn killing_twice_is_not_an_error() {
        let (dir, registry) = fixture();
        let result = start(dir.path(), &registry, json!({"command": "sleep 30"})).unwrap();
        let handle = handle_of(&result);

        let first = kill(dir.path(), &registry, json!({"handle": handle})).unwrap();
        assert_eq!(first.data["killed"], true);

        let second = kill(dir.path(), &registry, json!({"handle": handle})).unwrap();
        assert_eq!(second.data["killed"], false);
        assert_eq!(second.data["already_dead"], true);
        assert!(
            second.observation.contains("already gone"),
            "{}",
            second.observation
        );
    }

    #[test]
    fn kill_takes_the_whole_group_so_a_child_does_not_survive() {
        let (dir, registry) = fixture();
        let pid_file = dir.path().join("child.pid");
        let result = start(
            dir.path(),
            &registry,
            json!({"command": format!("sleep 60 & echo $! > '{}'; wait", pid_file.display())}),
        )
        .unwrap();
        let handle = handle_of(&result);
        std::thread::sleep(Duration::from_millis(400));

        let child: i32 = std::fs::read_to_string(&pid_file)
            .expect("the child wrote its pid")
            .trim()
            .parse()
            .unwrap();

        kill(dir.path(), &registry, json!({"handle": handle})).unwrap();

        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let running = pid_alive(u32::try_from(child).unwrap());
            if !running {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "a grandchild (pid {child}) survived kill_process"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    #[test]
    fn an_unknown_handle_is_refused_with_a_usable_message() {
        let (dir, registry) = fixture();
        let error = read(dir.path(), &registry, json!({"handle": "proc_99"})).unwrap_err();
        assert!(error.to_string().contains("unknown_handle"), "{error}");
    }

    #[test]
    fn handles_are_never_reused_after_a_restart() {
        // The registry's reason for existing: a plan that says `proc_1` must not, after a daemon
        // restart, resolve to a different process.
        let (dir, registry) = fixture();
        let first = start(dir.path(), &registry, json!({"command": "sleep 30"})).unwrap();
        assert_eq!(handle_of(&first), "proc_1");
        let _ = kill(dir.path(), &registry, json!({"handle": "proc_1"}));

        // Simulate a restart: a fresh registry over the same directory.
        let reopened = Arc::new(ProcessRegistry::open(registry.dir()).unwrap());
        let second = start(dir.path(), &reopened, json!({"command": "sleep 30"})).unwrap();
        assert_eq!(
            handle_of(&second),
            "proc_2",
            "a restarted daemon must not hand out `proc_1` again"
        );
        let _ = kill(dir.path(), &reopened, json!({"handle": "proc_2"}));
    }

    #[test]
    fn a_record_survives_a_restart_so_a_handle_still_resolves() {
        let (dir, registry) = fixture();
        let result = start(dir.path(), &registry, json!({"command": "sleep 30"})).unwrap();
        let handle = handle_of(&result);

        let reopened = Arc::new(ProcessRegistry::open(registry.dir()).unwrap());
        let record = reopened.load(&handle).unwrap();
        assert_eq!(is_running(&record), Liveness::Running);
        read(dir.path(), &reopened, json!({"handle": handle})).unwrap();
        let _ = kill(dir.path(), &reopened, json!({"handle": handle}));
    }

    #[test]
    fn the_marker_is_derived_from_the_arguments_digest() {
        // Both sides of the probe compute it, with no shared state, so the derivation itself has
        // to be stable and deterministic.
        let digest = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        assert_eq!(
            ProcessRecord::marker_for(digest),
            "immortality-proc-0123456789abcdef"
        );
        // A bare digest, as an older record might hold it, still works.
        assert_eq!(
            ProcessRecord::marker_for("0123456789abcdef"),
            "immortality-proc-0123456789abcdef"
        );
        // Distinct digests stay distinct, or two starts would share one marker.
        assert_ne!(
            ProcessRecord::marker_for("sha256:aaaaaaaaaaaaaaaa"),
            ProcessRecord::marker_for("sha256:bbbbbbbbbbbbbbbb")
        );
    }

    #[test]
    fn a_started_process_carries_the_marker_in_its_environment() {
        // This is the mechanism the probe relies on, so it is asserted directly rather than only
        // through the probe.
        let (dir, registry) = fixture();
        let result = start(dir.path(), &registry, json!({"command": "sleep 30"})).unwrap();
        let handle = handle_of(&result);
        let record = registry.load(&handle).unwrap();

        assert_eq!(
            process_has_marker(record.pid, &record.marker),
            Some(true),
            "`ps` must show the marker comment for the probe to work"
        );
        let _ = kill(dir.path(), &registry, json!({"handle": handle}));
    }

    #[test]
    fn the_probe_says_applied_while_the_process_runs_and_not_applied_after() {
        let (dir, registry) = fixture();
        // A digest for a call that was never made: the honest "before" state.
        let never = start_digest("sleep 999", dir.path());
        assert_eq!(
            probe_started(&never),
            immortality_core::protocol::Probe::NotApplied
        );

        let command = "sleep 30";
        let result = start(dir.path(), &registry, json!({"command": command})).unwrap();
        let handle = handle_of(&result);
        let record = registry.load(&handle).unwrap();

        // Derived from the record's own resolved directory, which is what the tool hashed. Using
        // anything else is the bug this test was written after: the probe and the spawner must
        // agree byte for byte.
        let digest = start_digest(&record.command, Path::new(&record.cwd));
        assert_eq!(digest, start_digest(command, Path::new(&record.cwd)));
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(
            probe_started(&digest),
            immortality_core::protocol::Probe::Applied,
            "a running process must be found by its marker"
        );

        let _ = kill(dir.path(), &registry, json!({"handle": handle}));
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if probe_started(&digest) == immortality_core::protocol::Probe::NotApplied {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the probe never went back to NotApplied"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    #[test]
    fn the_marker_stays_visible_for_a_single_command_body() {
        // The brace wrapper's whole purpose. `sh -c 'sleep 30'` replaces its own image, so the
        // command line carrying the marker disappears while the process keeps running — the probe
        // would then answer NotApplied for a process that is running fine, and recovery would
        // start a second one.
        let (dir, registry) = fixture();
        let result = start(dir.path(), &registry, json!({"command": "sleep 30"})).unwrap();
        let handle = handle_of(&result);
        let record = registry.load(&handle).unwrap();
        std::thread::sleep(Duration::from_millis(300));

        assert_eq!(
            probe_started(&start_digest(&record.command, Path::new(&record.cwd))),
            immortality_core::protocol::Probe::Applied
        );
        let _ = kill(dir.path(), &registry, json!({"handle": handle}));
    }

    #[test]
    fn a_recycled_pid_belonging_to_another_process_is_not_killed() {
        // The dangerous case the marker exists for: the recorded pid is alive but is not ours.
        // `kill_process` must refuse rather than signal a stranger.
        let (dir, registry) = fixture();
        let result = start(dir.path(), &registry, json!({"command": "sleep 30"})).unwrap();
        let handle = handle_of(&result);

        // Rewrite the record with a live pid that is *not* the started process, and a marker that
        // does not match it — precisely a recycled pid.
        let mut record = registry.load(&handle).unwrap();
        let _ = kill(dir.path(), &registry, json!({"handle": handle}));
        record.pid = std::process::id();
        record.marker = "proc_not-ours-at-all".to_string();
        registry.save(&record).unwrap();

        assert_eq!(is_running(&record), Liveness::Dead, "pid alive, marker absent");
        let outcome = kill(dir.path(), &registry, json!({"handle": handle})).unwrap();
        assert_eq!(
            outcome.data["already_dead"], true,
            "a stranger's pid must be reported dead, not killed: {}",
            outcome.observation
        );
    }

    #[test]
    fn a_read_window_does_not_split_a_character() {
        // A limit landing mid-UTF-8 must round down, so the next poll starts at a boundary rather
        // than on a continuation byte that would decode as garbage.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.txt");
        std::fs::write(&path, "aёb").unwrap();
        // "ё" is two bytes, so a limit of 2 lands inside it.
        let (text, consumed, more) = read_since(&path, 0, 2);
        assert_eq!(consumed, 1, "only the ASCII `a` fits whole");
        assert_eq!(text, "a");
        assert!(more);
        let (rest, consumed, more) = read_since(&path, consumed, 16);
        assert_eq!(rest, "ёb");
        assert_eq!(consumed, 3);
        assert!(!more);
    }

    #[test]
    fn reading_past_the_end_returns_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.txt");
        std::fs::write(&path, "abc").unwrap();
        let (text, consumed, more) = read_since(&path, 3, 16);
        assert_eq!(text, "");
        assert_eq!(consumed, 0);
        assert!(!more);
    }

    #[test]
    fn reading_a_stream_that_does_not_exist_is_empty_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let (text, consumed, more) = read_since(&dir.path().join("nope"), 0, 16);
        assert_eq!(text, "");
        assert_eq!(consumed, 0);
        assert!(!more);
    }

    #[test]
    fn a_stream_selector_limits_which_cursor_advances() {
        let (dir, registry) = fixture();
        let result = start(
            dir.path(),
            &registry,
            json!({"command": "echo out; echo err >&2; sleep 30"}),
        )
        .unwrap();
        let handle = handle_of(&result);
        std::thread::sleep(Duration::from_millis(300));

        let only_err = read(
            dir.path(),
            &registry,
            json!({"handle": handle, "stream": "stderr"}),
        )
        .unwrap();
        assert!(only_err.observation.contains("err"), "{}", only_err.observation);
        assert!(!only_err.observation.contains("out"), "{}", only_err.observation);
        let record = registry.load(&handle).unwrap();
        assert_eq!(
            record.stdout_cursor, 0,
            "reading stderr must not advance the stdout cursor"
        );
        let _ = kill(dir.path(), &registry, json!({"handle": handle}));
    }

    #[test]
    fn an_escalated_command_is_not_started() {
        let (dir, registry) = fixture();
        let error = start(
            dir.path(),
            &registry,
            json!({"command": "curl https://example.com"}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("awaiting_human_approval"), "{error}");
        assert!(
            registry.handles().unwrap().is_empty(),
            "a refused start must leave no record"
        );
    }

    #[test]
    fn a_cwd_outside_the_worktree_is_refused() {
        let (dir, registry) = fixture();
        let error = start(
            dir.path(),
            &registry,
            json!({"command": "sleep 30", "cwd": "../.."}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("denied_by_policy"), "{error}");
    }

    #[test]
    fn an_empty_command_is_refused() {
        let (dir, registry) = fixture();
        assert!(start(dir.path(), &registry, json!({"command": "  "})).is_err());
    }

    #[test]
    fn the_schemas_are_well_formed() {
        let (_dir, registry) = fixture();
        let shell = Arc::new(crate::shell::ShellState::new());
        for tool in tools(registry, shell) {
            assert_eq!(
                immortality_llm::schema::schema_defects(tool.schema()),
                Vec::new(),
                "{}",
                tool.name()
            );
        }
    }

    #[test]
    fn the_declared_effects_and_idempotency_are_the_designed_ones() {
        let (_dir, registry) = fixture();
        let shell = Arc::new(crate::shell::ShellState::new());
        let start_tool = StartProcess::new(Arc::clone(&registry), shell);
        let read_tool = ReadProcess::new(Arc::clone(&registry));
        let kill_tool = KillProcess::new(Arc::clone(&registry));

        assert_eq!(
            start_tool.effect(&Args::new(json!({"command": "ls"}))),
            Effect::Read
        );
        assert_eq!(
            start_tool.idempotency(&Args::empty()),
            Idempotency::Unknown,
            "a replay could start a second server on the same port"
        );
        assert_eq!(read_tool.effect(&Args::empty()), Effect::Read);
        assert_eq!(kill_tool.effect(&Args::empty()), Effect::Mutate);
        assert_eq!(kill_tool.idempotency(&Args::empty()), Idempotency::Idempotent);
    }

    #[test]
    fn liveness_of_all_covers_every_handle() {
        let (dir, registry) = fixture();
        let one = start(dir.path(), &registry, json!({"command": "sleep 30"})).unwrap();
        let two = start(dir.path(), &registry, json!({"command": "true"})).unwrap();
        std::thread::sleep(Duration::from_millis(300));

        let all = liveness_of_all(&registry).unwrap();
        assert_eq!(all.len(), 2, "{all:?}");
        assert_eq!(all[handle_of(&one).as_str()], Liveness::Running);
        let _ = two;
        let _ = kill(dir.path(), &registry, json!({"handle": handle_of(&one)}));
    }

    #[test]
    fn a_body_reading_command_still_gets_its_output_captured() {
        // The capture files are the only way to see a started process's output, so this pins that
        // they are actually written.
        let (dir, registry) = fixture();
        let result = start(
            dir.path(),
            &registry,
            json!({"command": "printf 'one\\ntwo\\n'; sleep 30"}),
        )
        .unwrap();
        let handle = handle_of(&result);
        std::thread::sleep(Duration::from_millis(300));
        let polled = read(dir.path(), &registry, json!({"handle": handle})).unwrap();
        assert!(polled.observation.contains("one"), "{}", polled.observation);
        assert!(polled.observation.contains("two"), "{}", polled.observation);
        let _ = kill(dir.path(), &registry, json!({"handle": handle}));
    }

    #[test]
    fn more_output_than_one_window_says_to_call_again() {
        let (dir, registry) = fixture();
        let result = start(dir.path(), &registry, json!({"command": "seq 1 5000; sleep 30"})).unwrap();
        let handle = handle_of(&result);
        std::thread::sleep(Duration::from_millis(500));

        let first = read(dir.path(), &registry, json!({"handle": handle, "max_bytes": 100})).unwrap();
        assert_eq!(first.data["has_more"], true);
        assert!(
            first.observation.contains("call `read_process` again"),
            "{}",
            first.observation
        );

        // Draining the rest must eventually reach the end without repeating bytes.
        let mut seen = first.data["stdout_bytes"].as_u64().unwrap();
        loop {
            let next = read(
                dir.path(),
                &registry,
                json!({"handle": handle, "max_bytes": 8, "stream": "stdout"}),
            )
            .unwrap();
            seen += next.data["stdout_bytes"].as_u64().unwrap();
            if next.data["has_more"] == serde_json::json!(false) {
                break;
            }
        }
        let total = std::fs::metadata(registry.load(&handle).unwrap().stdout_path)
            .unwrap()
            .len();
        assert_eq!(seen, total, "every byte must be delivered exactly once");
        let _ = kill(dir.path(), &registry, json!({"handle": handle}));
    }
}
