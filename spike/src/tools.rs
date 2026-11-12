//! Four tools, deliberately boring. The point of the spike is to observe the model, not to
//! build a good tool layer — that is M3.
//!
//! `edit_file` is the interesting one: it refuses ambiguous edits instead of guessing, which
//! is the behaviour we want to measure the model's reaction to.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

pub struct Sandbox {
    root: PathBuf,
}

impl Sandbox {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve inside the sandbox or fail. Prevents `../` escapes and absolute paths.
    fn resolve(&self, path: &str) -> Result<PathBuf> {
        let candidate = Path::new(path);
        if candidate.is_absolute() {
            bail!("absolute paths are not allowed: {path}");
        }
        let mut out = self.root.clone();
        for part in candidate.components() {
            use std::path::Component::*;
            match part {
                Normal(p) => out.push(p),
                CurDir => {}
                ParentDir => {
                    bail!("path escapes the sandbox: {path}");
                }
                _ => bail!("unsupported path component in {path}"),
            }
        }
        Ok(out)
    }
}

pub fn definitions() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file from the sandbox. Returns contents with 1-based line numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path relative to the sandbox root"}
                    },
                    "required": ["path"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or overwrite a whole file with the given content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace an exact snippet in a file. The snippet must occur exactly once; otherwise the call fails and nothing is changed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old": {"type": "string", "description": "Exact text to find, including indentation"},
                        "new": {"type": "string", "description": "Replacement text"}
                    },
                    "required": ["path", "old", "new"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a shell command inside the sandbox root. Not interactive: stdin is closed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        }),
    ]
}

/// Result of one tool call. Always returned to the model, never converted into a crash.
pub struct ToolOutcome {
    pub ok: bool,
    pub content: String,
}

impl ToolOutcome {
    fn ok(content: impl Into<String>) -> Self {
        Self { ok: true, content: content.into() }
    }
    /// Exposed to the model as an observation starting with `ERROR:`.
    fn err(content: impl Into<String>) -> Self {
        Self { ok: false, content: format!("ERROR: {}", content.into()) }
    }
}

pub async fn execute(sandbox: &Sandbox, name: &str, args: &Value) -> ToolOutcome {
    let path = args.get("path").and_then(|p| p.as_str()).unwrap_or_default();
    match name {
        "read_file" => match read_file(sandbox, path) {
            Ok(text) => ToolOutcome::ok(text),
            Err(e) => ToolOutcome::err(e.to_string()),
        },
        "write_file" => {
            let content = args.get("content").and_then(|c| c.as_str()).unwrap_or_default();
            match write_file(sandbox, path, content) {
                Ok(()) => ToolOutcome::ok(format!("wrote {} bytes to {path}", content.len())),
                Err(e) => ToolOutcome::err(e.to_string()),
            }
        }
        "edit_file" => {
            let old = args.get("old").and_then(|c| c.as_str()).unwrap_or_default();
            let new = args.get("new").and_then(|c| c.as_str()).unwrap_or_default();
            match edit_file(sandbox, path, old, new) {
                Ok(()) => ToolOutcome::ok(format!("edited {path}")),
                Err(e) => ToolOutcome::err(e.to_string()),
            }
        }
        "shell" => {
            let command = args.get("command").and_then(|c| c.as_str()).unwrap_or_default();
            shell(sandbox, command).await
        }
        other => ToolOutcome::err(format!("unknown tool `{other}`")),
    }
}

fn read_file(sandbox: &Sandbox, path: &str) -> Result<String> {
    let full = sandbox.resolve(path)?;
    let bytes = std::fs::read(&full).with_context(|| format!("cannot read {path}"))?;
    let text = String::from_utf8(bytes).map_err(|_| anyhow::anyhow!("{path} is not valid UTF-8"))?;
    let numbered: String = text
        .lines()
        .enumerate()
        .map(|(i, line)| format!("{:>6}\t{line}\n", i + 1))
        .collect();
    Ok(numbered)
}

fn write_file(sandbox: &Sandbox, path: &str, content: &str) -> Result<()> {
    let full = sandbox.resolve(path)?;
    if let Some(parent) = full.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&full, content).with_context(|| format!("cannot write {path}"))?;
    Ok(())
}

fn edit_file(sandbox: &Sandbox, path: &str, old: &str, new: &str) -> Result<()> {
    if old.is_empty() {
        bail!("`old` must not be empty; use write_file to create a file");
    }
    let full = sandbox.resolve(path)?;
    let text = std::fs::read_to_string(&full).with_context(|| format!("cannot read {path}"))?;

    let matches = text.matches(old).count();
    match matches {
        0 => bail!(
            "`old` snippet not found in {path}. Read the file again and copy the exact text, including indentation."
        ),
        1 => {}
        n => bail!(
            "`old` snippet occurs {n} times in {path}, it must be unique. Include more surrounding lines to disambiguate."
        ),
    }

    let updated = text.replacen(old, new, 1);
    std::fs::write(&full, updated)?;
    Ok(())
}

async fn shell(sandbox: &Sandbox, command: &str) -> ToolOutcome {
    if command.trim().is_empty() {
        return ToolOutcome::err("empty command");
    }
    let result = tokio::time::timeout(
        Duration::from_secs(120),
        tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(sandbox.root())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output(),
    )
    .await;

    let output = match result {
        Err(_) => return ToolOutcome::err("command timed out after 120s"),
        Ok(Err(e)) => return ToolOutcome::err(format!("cannot spawn: {e}")),
        Ok(Ok(o)) => o,
    };

    const LIMIT: usize = 4000;
    let trim = |s: &[u8]| -> String {
        let text = String::from_utf8_lossy(s).to_string();
        if text.len() <= LIMIT {
            text
        } else {
            let head: String = text.chars().take(LIMIT / 2).collect();
            let tail: String = text.chars().rev().take(LIMIT / 2).collect::<String>().chars().rev().collect();
            format!("{head}\n... [truncated {} bytes] ...\n{tail}", text.len() - LIMIT)
        }
    };

    let body = format!(
        "exit_code: {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status.code().unwrap_or(-1),
        trim(&output.stdout),
        trim(&output.stderr)
    );
    if output.status.success() {
        ToolOutcome::ok(body)
    } else {
        ToolOutcome { ok: false, content: body }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_sandbox(name: &str) -> Sandbox {
        let dir = std::env::temp_dir().join(format!("spike-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Sandbox::new(dir).unwrap()
    }

    #[test]
    fn rejects_parent_escapes() {
        let sb = temp_sandbox("escape");
        assert!(sb.resolve("../etc/passwd").is_err());
        assert!(sb.resolve("/etc/passwd").is_err());
        assert!(sb.resolve("a/../b").is_err());
        assert!(sb.resolve("a/b").is_ok());
    }

    #[test]
    fn edit_requires_unique_match() {
        let sb = temp_sandbox("edit");
        write_file(&sb, "f.txt", "a\nb\na\n").unwrap();
        let err = edit_file(&sb, "f.txt", "a", "x").unwrap_err().to_string();
        assert!(err.contains("occurs 2 times"), "{err}");
        // File untouched.
        assert_eq!(std::fs::read_to_string(sb.resolve("f.txt").unwrap()).unwrap(), "a\nb\na\n");
    }

    #[test]
    fn edit_reports_missing_snippet() {
        let sb = temp_sandbox("edit-missing");
        write_file(&sb, "f.txt", "hello\n").unwrap();
        let err = edit_file(&sb, "f.txt", "nope", "x").unwrap_err().to_string();
        assert!(err.contains("not found"), "{err}");
    }

    #[test]
    fn edit_replaces_single_occurrence() {
        let sb = temp_sandbox("edit-ok");
        write_file(&sb, "f.txt", "one\ntwo\n").unwrap();
        edit_file(&sb, "f.txt", "two", "three").unwrap();
        assert_eq!(std::fs::read_to_string(sb.resolve("f.txt").unwrap()).unwrap(), "one\nthree\n");
    }

    #[test]
    fn empty_old_is_rejected() {
        let sb = temp_sandbox("edit-empty");
        write_file(&sb, "f.txt", "x\n").unwrap();
        assert!(edit_file(&sb, "f.txt", "", "y").is_err());
    }

    #[tokio::test]
    async fn shell_closes_stdin() {
        let sb = temp_sandbox("shell-stdin");
        // `cat` with no input must terminate instead of hanging.
        let out = shell(&sb, "cat").await;
        assert!(out.ok, "{}", out.content);
    }

    #[tokio::test]
    async fn shell_reports_nonzero_exit() {
        let sb = temp_sandbox("shell-fail");
        let out = shell(&sb, "exit 3").await;
        assert!(!out.ok);
        assert!(out.content.contains("exit_code: 3"), "{}", out.content);
    }

    #[tokio::test]
    async fn shell_truncates_huge_output() {
        let sb = temp_sandbox("shell-big");
        let out = shell(&sb, "yes abcdefgh | head -c 200000").await;
        assert!(out.content.len() < 12000, "len={}", out.content.len());
    }

    #[test]
    fn unknown_tool_is_an_observation_not_a_panic() {
        let sb = temp_sandbox("unknown");
        let outcome = tokio::runtime::Runtime::new().unwrap().block_on(execute(&sb, "nope", &json!({})));
        assert!(!outcome.ok);
        assert!(outcome.content.starts_with("ERROR:"));
    }
}
