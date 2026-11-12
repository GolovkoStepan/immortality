//! Five scenarios, each probing a different failure mode of a weak local model.
//!
//! They are ordered by what they measure, not by difficulty:
//!   1. multi-file-edit  — instruction holding across edits, `edit_file` disambiguation
//!   2. failing-tests    — reaction to repeated tool errors (D1/D3, real data)
//!   3. long-horizon     — degradation of tool-calling with growing context
//!   4. big-output       — handling of huge tool output without losing the thread
//!   5. refactor         — plan holding + rename across files

use anyhow::Result;

use crate::tools::Sandbox;

pub struct Scenario {
    pub name: &'static str,
    pub system: &'static str,
    pub goal: &'static str,
    pub setup: fn(&Sandbox) -> Result<()>,
    pub done: fn(&Sandbox) -> bool,
}

const SYSTEM: &str = "\
You are a coding agent working in an isolated sandbox directory. \
You have exactly four tools: read_file, write_file, edit_file, shell. \
Rules: \
- Use tools, never ask the user for permission or clarification. \
- After each tool result, decide the next action. \
- When the task is complete, reply with a short summary and no tool call. \
- edit_file requires the `old` snippet to occur exactly once; include surrounding lines if needed. \
- Keep edits minimal.";

fn read(sb: &Sandbox, path: &str) -> String {
    std::fs::read_to_string(sb.root().join(path)).unwrap_or_default()
}

fn write(sb: &Sandbox, path: &str, content: &str) -> Result<()> {
    let full = sb.root().join(path);
    if let Some(parent) = full.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(full, content)?;
    Ok(())
}

// ── 1. multi-file-edit ────────────────────────────────────────────────────────────

fn setup_multi(sb: &Sandbox) -> Result<()> {
    write(sb, "src/lib.rs", "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n")?;
    write(sb, "src/util.rs", "pub fn unused_helper() -> u8 {\n    0\n}\n")?;
    write(sb, "Cargo.toml", "[package]\nname = \"demo\"\nversion = \"0.1.0\"\n")?;
    Ok(())
}

fn done_multi(sb: &Sandbox) -> bool {
    let lib = read(sb, "src/lib.rs");
    let util = read(sb, "src/util.rs");
    lib.contains("#[cfg(test)]") && lib.contains("add(2, 2)") && util.contains("pub fn helper") && !util.contains("unused_helper")
}

// ── 2. failing-tests ─────────────────────────────────────────────────────────────

fn setup_tests(sb: &Sandbox) -> Result<()> {
    // The function is buggy on purpose; the model must find it by running the test.
    write(
        sb,
        "src/lib.rs",
        "pub fn average(values: &[i64]) -> i64 {\n    let sum: i64 = values.iter().sum();\n    sum / (values.len() as i64 + 1)\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n    #[test]\n    fn averages() {\n        assert_eq!(average(&[2, 4, 6]), 4);\n    }\n\n    #[test]\n    fn empty_is_zero() {\n        assert_eq!(average(&[]), 0);\n    }\n}\n",
    )?;
    write(sb, "run_test.sh", "#!/bin/sh\n# Stand-in for the real test runner: extracts the expected value and checks the formula.\nif grep -q '/ (values.len() as i64 + 1)' src/lib.rs; then\n  echo 'test averages ... FAILED'\n  echo '  left: 3, right: 4'\n  exit 1\nfi\necho 'test averages ... ok'\nexit 0\n")?;
    Ok(())
}

fn done_tests(sb: &Sandbox) -> bool {
    let lib = read(sb, "src/lib.rs");
    // Fixed divisor and empty-input guard, the two things the failing test implies.
    (lib.contains("values.len() as i64)") || lib.contains("/ values.len() as i64"))
        && !lib.contains("+ 1)")
        && lib.contains("is_empty()")
}

// ── 3. long-horizon ─────────────────────────────────────────────────────────────

fn setup_long(sb: &Sandbox) -> Result<()> {
    write(sb, "src/main.rs", "fn main() {\n    println!(\"start\");\n}\n")?;
    let mut notes = String::from("# Progress log\n\nSteps completed so far:\n");
    for i in 1..=30 {
        notes.push_str(&format!("{i}. checkpoint {i} recorded\n"));
    }
    write(sb, "NOTES.md", &notes)?;
    Ok(())
}

fn done_long(sb: &Sandbox) -> bool {
    // Each of the 15 functions must exist in its own file and be re-exported.
    (1..=15).all(|i| read(sb, &format!("src/mod_{i}.rs")).contains(&format!("pub fn step_{i}")))
        && read(sb, "src/main.rs").contains("mod_15")
}

// ── 4. big-output ───────────────────────────────────────────────────────────────

fn setup_big(sb: &Sandbox) -> Result<()> {
    let mut log = String::new();
    for i in 1..=2000 {
        log.push_str(&format!(
            "2026-11-12T10:{:02}:{:02}Z INFO worker-{} processed item {} status=ok\n",
            i % 60,
            i % 60,
            i % 8,
            i
        ));
    }
    log.push_str("2026-11-12T10:59:59Z ERROR worker-3 processed item 1999 status=FAILED code=E42\n");
    write(sb, "logs/app.log", &log)?;
    Ok(())
}

fn done_big(sb: &Sandbox) -> bool {
    // The one failing line is buried among 2000 ok lines.
    let report = read(sb, "REPORT.md");
    report.contains("1999") && report.contains("E42")
}

// ── 6. iterative-fix ────────────────────────────────────────────────────────────
//
// The only scenario that cannot be batched: each step depends on the output of the
// previous one, so the model must stay coherent across many turns with growing context.

fn setup_iterative(sb: &Sandbox) -> Result<()> {
    write(
        sb,
        "src/lib.rs",
        "pub fn is_palindrome(s: &str) -> bool {\n    let chars: Vec<char> = s.chars().collect();\n    chars == chars.iter().rev().copied().collect::<Vec<_>>()\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn simple() { assert!(is_palindrome(\"racecar\")); }\n}\n",
    )?;
    // A checklist the model must satisfy item by item; the checker reports the first failing one.
    write(
        sb,
        "check.sh",
        r#"#!/bin/sh
check() {
  if ! grep -q "$1" src/lib.rs; then
    echo "MISSING: $2"
    exit 1
  fi
}
check 'fn is_palindrome' 'is_palindrome exists'
check 'to_lowercase' 'case-insensitive comparison'
check 'is_alphanumeric' 'ignores punctuation and spaces'
check 'fn count_palindromes' 'count_palindromes helper'
check 'fn longest_palindromic_prefix' 'longest_palindromic_prefix helper'
check 'fn is_palindrome_recursive' 'recursive variant'
check 'fn normalize' 'normalize helper'
check 'fn palindromic_substrings' 'palindromic_substrings'
check '#\[test\]' 'tests still present'
echo ALL_OK
"#,
    )?;
    Ok(())
}

fn done_iterative(sb: &Sandbox) -> bool {
    matches!(
        std::process::Command::new("sh")
            .arg("check.sh")
            .current_dir(sb.root())
            .output(),
        Ok(o) if o.status.success()
    )
}

// ── 7. conflicting-instructions ─────────────────────────────────────────────────
//
// Instruction holding under pressure: two rules conflict, the model must notice and
// follow the explicit precedence instead of silently dropping one.

fn setup_conflict(sb: &Sandbox) -> Result<()> {
    write(sb, "src/lib.rs", "pub fn value() -> u32 {\n    1\n}\n")?;
    write(sb, "src/other.rs", "pub fn value() -> u32 {\n    7\n}\n")?;
    Ok(())
}

fn done_conflict(sb: &Sandbox) -> bool {
    let lib = read(sb, "src/lib.rs");
    let other = read(sb, "src/other.rs");
    // src/lib.rs must be doubled (1 -> 2); src/other.rs must still return 7.
    lib.contains("    2\n") && !lib.contains("    1\n") && other.contains("    7\n") && !other.contains("    14\n")
}

fn setup_refactor(sb: &Sandbox) -> Result<()> {
    write(
        sb,
        "src/api.rs",
        "pub fn get_user(id: u32) -> String {\n    format!(\"user-{id}\")\n}\n\npub fn get_config(name: &str) -> String {\n    format!(\"config-{name}\")\n}\n",
    )?;
    write(
        sb,
        "src/handlers.rs",
        "use crate::api::{get_user, get_config};\n\npub fn handle_user(id: u32) {\n    let _ = get_user(id);\n}\n\npub fn handle_config() {\n    let _ = get_config(\"default\");\n}\n",
    )?;
    Ok(())
}

fn done_refactor(sb: &Sandbox) -> bool {
    // Rename get_config -> load_config everywhere, and nothing else.
    let api = read(sb, "src/api.rs");
    let handlers = read(sb, "src/handlers.rs");
    api.contains("pub fn load_config")
        && !api.contains("pub fn get_config")
        && handlers.contains("load_config")
        && !handlers.contains("get_config")
        && handlers.contains("get_user")
        && api.contains("pub fn get_user")
}

pub fn all() -> Vec<Scenario> {
    vec![
        Scenario { name: "multi-file-edit", system: SYSTEM, goal: "\
1. Read src/lib.rs.\n\
2. Append a `#[cfg(test)] mod tests` to src/lib.rs with one test asserting add(2, 2) == 4.\n\
3. In src/util.rs rename `unused_helper` to `helper`.\n\
4. Do not touch Cargo.toml.\n\
When done, reply with a one-line summary.", setup: setup_multi, done: done_multi },

        Scenario { name: "failing-tests", system: SYSTEM, goal: "\
The function `average` in src/lib.rs is wrong. Run `sh run_test.sh` to see the failure, \
find the bug by reading the code, fix it so the test passes, and make `average(&[])` return 0. \
Verify by running `sh run_test.sh` again. Then reply with a one-line summary.", setup: setup_tests, done: done_tests },

        Scenario { name: "long-horizon", system: SYSTEM, goal: "\
Create 15 files src/mod_1.rs .. src/mod_15.rs. File src/mod_N.rs must contain exactly \
`pub fn step_N() -> u32 { N }` (substituting the number). Then create src/main.rs declaring \
all 15 modules (`mod mod_1;` .. `mod mod_15;`) and a main that prints the sum of step_1()..step_15(). \
Work file by file with write_file. When all files exist, reply with a one-line summary.", setup: setup_long, done: done_long },

        Scenario { name: "big-output", system: SYSTEM, goal: "\
The file logs/app.log contains ~2000 lines. Find every line with status=FAILED, extract the \
item number and the error code, and write the findings to REPORT.md as a short markdown list. \
If there are no failures, write 'no failures' to REPORT.md. When done, reply with a one-line summary.", setup: setup_big, done: done_big },

        Scenario { name: "refactor", system: SYSTEM, goal: "\
Rename the public function `get_config` to `load_config` across the whole project. \
Update every call site. Do not rename anything else. When done, reply with a one-line summary.", setup: setup_refactor, done: done_refactor },

        Scenario { name: "iterative-fix", system: SYSTEM, goal: "\
Run `sh check.sh` repeatedly. It reports one missing feature at a time. Fix exactly that one \
feature in src/lib.rs with edit_file, then run `sh check.sh` again. Repeat until it prints ALL_OK. \
Do not add all features at once: the checker tests them one by one and you must keep the existing \
tests passing. When it prints ALL_OK, reply with a one-line summary.", setup: setup_iterative, done: done_iterative },

        Scenario { name: "conflicting-instructions", system: SYSTEM, goal: "\
Read src/lib.rs and src/other.rs. Both define a function named `value`. \
Double the return value of `value` in src/lib.rs only. \
IMPORTANT: src/other.rs must remain exactly as it is; do not modify it under any circumstance, \
even though it looks similar. When done, reply with a one-line summary.", setup: setup_conflict, done: done_conflict },
    ]
}
