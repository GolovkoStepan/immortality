//! Classifying a shell command by how much damage it can do.
//!
//! # Why a table and not a sanitiser
//!
//! The agent runs commands it invented. A sanitiser would have to prove that a given string is
//! safe, and shell is Turing-complete in the ways that matter: `$(...)`, backticks, aliases,
//! functions, `eval`, environment-dependent expansion. Proving safety is undecidable, so this
//! module does the decidable thing instead: it recognises the commands it knows, and files
//! everything else as [`Effect::External`], which never executes without a human answering an
//! escalation. Being wrong about `curl` being read is a data loss; being wrong about `frobnicate`
//! being external costs one escalation.
//!
//! # The pipeline rule
//!
//! `git status | tee log` is a write. `cat x && curl y` reaches the network. So a command line is
//! split into simple commands on unquoted `;`, `&`, `|` and newlines, each is classified, and the
//! whole line takes the **maximum** severity — which is why [`Effect`] is ordered `Read < Mutate <
//! External`. Command substitution is split out and classified the same way, because
//! `echo $(curl evil)` must not inherit `echo`'s harmlessness.
//!
//! # Why `cargo test` is a mutation and not a read
//!
//! The design document lists `cargo test` under `read`. That is wrong and the error is the
//! dangerous direction: a read is allowed unconditionally, so classifying a command that writes
//! `target/`, executes every `build.rs` and runs arbitrary test code as a read means the gate
//! never checks where it runs. It stays a mutation here, which costs nothing — a mutation inside
//! the worktree is allowed anyway — and keeps `Read` meaning "cannot change the filesystem at
//! all", which is the only definition a policy can rely on.
//!
//! # What this module deliberately does not do
//!
//! It does not expand variables, aliases or globs. A command word beginning with `$` or a
//! redirection whose target is not a literal is reported as `External`, because guessing what
//! `$TARGET` holds is exactly the guess that turns a policy into decoration.

use std::collections::BTreeSet;

use crate::Effect;
use crate::policy::PathUse;

/// What a command line was judged to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Classification {
    /// The worst thing any part of the line can do.
    pub effect: Effect,
    /// Why, in one clause, so a denial can tell the model something usable.
    pub reason: String,
    /// Why the line is `External`, when it is.
    ///
    /// [`Self::reason`] is prose for the model and may be reworded freely; this is the fact a
    /// caller may branch on. The judge (D-111) has to answer only for a program the classifier
    /// does not know, and never for something the operator forbade — reading that out of a
    /// sentence means a reworded phrase silently disables the judge, and a new phrase containing
    /// the same words silently enables it where the operator decides.
    pub source: Option<External>,
    /// Programs recognised in the line, in order, for the journal and for `doctor`.
    pub programs: Vec<String>,
    /// Paths the line names in a position where they are written or entered.
    ///
    /// Not every argument of every program: only the ones where a path argument is unambiguous
    /// (`cd`, redirections, `rm`, `cp`…) and the program is known well enough to say which
    /// arguments those are. An unknown program contributes nothing here because it is already
    /// `External` and its command line is not interpreted at all.
    pub paths: Vec<PathUse>,
}

/// Why a command line took [`Effect::External`].
///
/// One variant per way the classifier can reach that verdict, because the callers treat them
/// differently: only [`External::UnknownProgram`] is a question a model may answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum External {
    /// A program no table knows. The only case a judge may decide (D-111).
    UnknownProgram,
    /// A program the operator's configuration names as unreachable — network, another user, the
    /// whole system. This is a statement, not an uncertainty, and no model may revise it.
    NamedProgram,
    /// A subcommand of a known program that is not classified (`git arcane`, `cargo frobnicate`).
    UnknownSubcommand,
    /// A subcommand known to reach the network or write outside the worktree (`git push`,
    /// `cargo publish`, `gh pr`).
    RemoteSubcommand,
    /// A wrapper that runs something the classifier cannot see: `npm run`, `xargs rm`,
    /// `find -exec`, `perl -e`.
    OpaqueScript,
    /// The program name is decided at run time (`$CMD`), so there is nothing to classify.
    DynamicProgram,
    /// An unterminated `$(` or backtick: the tail of the line cannot be read.
    Unreadable,
}

impl External {
    /// Whether a model may decide this case.
    #[must_use]
    pub const fn is_a_question(self) -> bool {
        matches!(self, Self::UnknownProgram)
    }
}

impl Default for Classification {
    fn default() -> Self {
        Self::new(Effect::Read, "reads only")
    }
}

impl Classification {
    fn new(effect: Effect, reason: impl Into<String>) -> Self {
        Self {
            effect,
            reason: reason.into(),
            source: None,
            programs: Vec::new(),
            paths: Vec::new(),
        }
    }

    /// A classification with the structural reason attached, which `External` always carries.
    fn external(source: External, reason: impl Into<String>) -> Self {
        Self {
            effect: Effect::External,
            reason: reason.into(),
            source: Some(source),
            programs: Vec::new(),
            paths: Vec::new(),
        }
    }

    fn absorb(&mut self, other: Self) {
        if other.effect > self.effect {
            self.effect = other.effect;
            self.reason = other.reason;
            self.source = other.source;
        }
        self.programs.extend(other.programs);
        self.paths.extend(other.paths);
    }
}

/// Programs that only read, and cannot write even by accident.
///
/// Deliberately short of anything that takes a command to run (`xargs`, `find -exec`) or that
/// writes through an option (`sort -o`, `tee`, `sed -i`): those are handled by their own arms
/// below, and leaving them out here means an option cannot silently promote a read.
const READ_PROGRAMS: &[&str] = &[
    "basename",
    "cat",
    "cmp",
    "column",
    "cut",
    "date",
    "df",
    "diff",
    "dirname",
    "du",
    "echo",
    "egrep",
    "false",
    "fgrep",
    "file",
    "grep",
    "head",
    "id",
    "jq",
    "ls",
    "md5",
    "nl",
    "od",
    "printf",
    "pwd",
    "readlink",
    "realpath",
    "rg",
    "seq",
    "shasum",
    "sleep",
    "sha256sum",
    "sort",
    "stat",
    "sw_vers",
    "tail",
    "tr",
    "tree",
    "true",
    "type",
    "uname",
    "uniq",
    "wc",
    "which",
    "whoami",
    "yq",
];

/// POSIX shell builtins that touch no file and reach no network.
///
/// An agent writes `test -f`, `wait`, `export FOO=1` and `:` constantly, and the unknown-command
/// rule would escalate every one of them to a human — which trains a human to approve reflexively,
/// the exact opposite of what the rule is for. Registered separately from [`READ_PROGRAMS`]
/// because these are not programs: a builtin has no binary, and moving one here does not mean a
/// tool of that name may be executed.
///
/// `export` and `set` are harmless *because of D-058*: the `sh` that would hold the variable dies
/// with the command, so a variable cannot leak into the next call. Under a persistent shell this
/// list would be wrong.
const BUILTIN_PROGRAMS: &[&str] = &[
    ":", "[", "export", "getopts", "hash", "jobs", "read", "set", "shift", "test", "times", "umask",
    "unalias", "unset", "wait",
];

/// Programs that change files inside the worktree.
const MUTATE_PROGRAMS: &[&str] = &[
    "chmod", "cp", "gunzip", "gzip", "ln", "mkdir", "mkfifo", "mktemp", "mv", "patch", "rm", "rmdir",
    "rsync", "tar", "tee", "touch", "truncate", "unzip", "xz", "zip",
];

/// Programs that leave the machine, reach the network, or need another user's authority.
///
/// `osascript` is here because `AppleScript` can do anything the user can, and `open` because it
/// launches applications: both are "not a build step" and deserve a human's attention.
const EXTERNAL_PROGRAMS: &[&str] = &[
    "apt",
    "apt-get",
    "brew",
    "bunx",
    "curl",
    "defaults",
    "dig",
    "diskutil",
    "docker",
    "doas",
    "ftp",
    "hdiutil",
    "host",
    "http",
    "https",
    "killall",
    "launchctl",
    "mount",
    "ncat",
    "nc",
    "netcat",
    "nmap",
    "npx",
    "nslookup",
    "open",
    "osascript",
    "pacman",
    "ping",
    "pip3",
    "pipx",
    "podman",
    "port",
    "scp",
    "sftp",
    "ssh",
    "sudo",
    "su",
    "telnet",
    "umount",
    "uvx",
    "wget",
    "xargs",
    "yum",
    "dnf",
    "pnpx",
    "python3",
    "python",
    "node",
    "deno",
    "ruby",
    "perl",
    "eval",
    "exec",
    "source",
];

/// Programs whose path arguments are known, mapped to whether they write.
const PATH_PROGRAMS: &[(&str, bool)] = &[
    ("cat", false),
    ("chmod", true),
    ("cp", true),
    ("diff", false),
    ("grep", false),
    ("head", false),
    ("jq", false),
    ("ln", true),
    ("ls", false),
    ("mkdir", true),
    ("mv", true),
    ("patch", true),
    ("rm", true),
    ("rmdir", true),
    ("sed", true),
    ("stat", false),
    ("tail", false),
    ("tar", true),
    ("tee", true),
    ("touch", true),
    ("truncate", true),
    ("unzip", true),
    ("wc", false),
    ("zip", true),
];

/// Classify a command line.
///
/// Never fails: an empty string is an empty line and does nothing, and anything unrecognised is
/// `External` rather than an error, because refusing to classify would leave the caller with no
/// safe default.
#[must_use]
pub fn classify(command: &str) -> Classification {
    let mut total = Classification::new(Effect::Read, "nothing to run");
    let mut found_anything = false;

    for simple in split_commands(command) {
        let classification = classify_simple(&simple);
        if classification.effect != Effect::Read || !classification.programs.is_empty() {
            found_anything = true;
        }
        total.absorb(classification);
    }

    if !found_anything {
        return Classification::new(Effect::Read, "empty command");
    }
    total
}

/// One simple command: its words, its redirections, and whether it is `cd`.
#[derive(Debug, Default)]
struct Simple {
    words: Vec<String>,
    redirections: Vec<(String, bool)>,
    /// A `$(` or backtick with no closing partner.
    ///
    /// Not an error to report: the command will fail at the shell. But the text after it is inside
    /// a construct this lexer could not read, so treating the visible prefix as the whole command
    /// would let `echo $(frobnicate` look like `echo`. External is the only honest answer.
    opaque: bool,
}

#[allow(clippy::too_many_lines)] // A character-by-character shell lexer is one loop and one shape.
fn split_commands(command: &str) -> Vec<Simple> {
    let mut out = Vec::new();
    let mut current = Simple::default();
    let mut word = String::new();
    let mut quote: Option<char> = None;
    let mut index = 0;
    let chars: Vec<char> = command.chars().collect();

    // A word must not be lost at a separator, and a separator inside quotes is a literal.
    macro_rules! flush_word {
        () => {
            if !word.is_empty() {
                current.words.push(std::mem::take(&mut word));
            }
        };
    }
    macro_rules! flush_command {
        () => {
            flush_word!();
            if !current.words.is_empty() || !current.redirections.is_empty() {
                out.push(std::mem::take(&mut current));
            }
        };
    }

    while let Some(&ch) = chars.get(index) {
        index += 1;
        match ch {
            '\'' | '"' if quote == Some(ch) => quote = None,
            '\'' | '"' if quote.is_none() => quote = Some(ch),
            // A backslash inside double quotes and outside quotes escapes the next character, so
            // `"a;b"` splits and `a\;b` does not. Inside single quotes it is literal.
            '\\' if quote != Some('\'') => {
                if let Some(&next) = chars.get(index) {
                    word.push(next);
                    index += 1;
                }
            }
            _ if quote.is_some() => word.push(ch),
            '$' if chars.get(index) == Some(&'(') => {
                if let Some((inner, next)) = take_substitution(&chars, index + 1) {
                    // The substitution's own command is classified as if it stood alone.
                    for mut nested in split_commands(&inner) {
                        nested.words.insert(0, String::new());
                        out.push(nested);
                    }
                    index = next;
                } else {
                    // Unterminated: the text after it is not readable as a command list.
                    current.opaque = true;
                    word.push(ch);
                }
            }
            '`' => {
                if let Some(close) = chars.iter().skip(index).position(|c| *c == '`') {
                    let inner: String = chars[index..index + close].iter().collect();
                    for nested in split_commands(&inner) {
                        out.push(nested);
                    }
                    index += close + 1;
                } else {
                    current.opaque = true;
                    word.push(ch);
                }
            }
            ';' | '|' | '&' | '\n' => {
                // `||`, `&&` and `|&` are just two separators in a row.
                flush_command!();
            }
            '>' | '<' => {
                // `2>` is a descriptor, not the argument `2`: the digits are only a descriptor when
                // they are attached to the operator, which is exactly the case where they are still
                // in `word`. A space flushes `2` as a real word before this arm runs.
                let descriptor = !word.is_empty() && word.chars().all(|c| c.is_ascii_digit());
                if descriptor {
                    word.clear();
                } else {
                    flush_word!();
                }
                let append = ch == '>' && chars.get(index) == Some(&'>');
                if append {
                    index += 1;
                }
                // `2>&1` names a descriptor, not a file. A descriptor is a digit immediately
                // before the operator, which lands in `word` as "2" — do not treat it as a path.
                if chars.get(index) == Some(&'&') {
                    index += 1;
                    while chars.get(index).is_some_and(char::is_ascii_digit) {
                        index += 1;
                    }
                    continue;
                }
                // Skip spaces between the operator and its target.
                while chars.get(index) == Some(&' ') {
                    index += 1;
                }
                let mut target = String::new();
                let mut target_quote: Option<char> = None;
                while let Some(&c) = chars.get(index) {
                    match c {
                        '\'' | '"' if target_quote == Some(c) => {
                            target_quote = None;
                            index += 1;
                        }
                        '\'' | '"' if target_quote.is_none() => {
                            target_quote = Some(c);
                            index += 1;
                        }
                        ' ' | '\t' | ';' | '|' | '&' | '\n' if target_quote.is_none() => break,
                        _ => {
                            target.push(c);
                            index += 1;
                        }
                    }
                }
                if ch == '>' && !target.is_empty() {
                    current.redirections.push((target, true));
                }
            }
            ' ' | '\t' | '(' | ')' if quote.is_none() => flush_word!(),
            _ => word.push(ch),
        }
    }
    flush_command!();
    out
}

/// Take the parenthesised body of a `$(...)`, honouring nesting and quotes.
fn take_substitution(chars: &[char], start: usize) -> Option<(String, usize)> {
    let mut depth = 1usize;
    let mut index = start;
    let mut inner = String::new();
    let mut quote: Option<char> = None;
    while let Some(&ch) = chars.get(index) {
        index += 1;
        match ch {
            '\'' | '"' if quote == Some(ch) => {
                quote = None;
                inner.push(ch);
            }
            '\'' | '"' if quote.is_none() => {
                quote = Some(ch);
                inner.push(ch);
            }
            '(' if quote.is_none() => {
                depth += 1;
                inner.push(ch);
            }
            ')' if quote.is_none() => {
                depth -= 1;
                if depth == 0 {
                    return Some((inner, index));
                }
                inner.push(ch);
            }
            _ => inner.push(ch),
        }
    }
    None
}

fn classify_simple(simple: &Simple) -> Classification {
    if simple.opaque {
        let mut result = Classification::external(
            External::Unreadable,
            "the command contains an unterminated `$(` or backtick, so its tail cannot be read",
        );
        result.programs = simple
            .words
            .first()
            .map(|word| vec![basename(word)])
            .unwrap_or_default();
        return result;
    }
    let mut result = Classification::default();

    // `A=1 B=2 cmd` — drop the assignments so the program is the first real word.
    // A nested command was inserted with an empty marker word, so the program is the first word
    // that is neither an assignment nor that marker.
    let words: Vec<&String> = simple
        .words
        .iter()
        .filter(|word| !is_assignment(word))
        .skip_while(|word| word.is_empty())
        .collect();

    for (target, _) in &simple.redirections {
        // A sink is a place output goes that is not a file on disk. It is not a path the policy
        // can judge, and reporting it as one is not a harmless over-approximation: `/dev/null`
        // is outside the worktree by construction, so every `cmd 2>/dev/null` — the ordinary
        // way to silence a warning — was denied as an escape attempt. Measured: five separate
        // steps of the first live run were refused for this alone, and two subtasks were
        // blocked on gates that never ran because their command was denied before execution.
        if is_sink(target) {
            // The effect is still a mutation. `cargo test 2>/dev/null` writes `target/`, and a
            // mutation that names no path is allowed inside the worktree while an unknown
            // program stays external either way — the sink changes the path list, not the risk.
            result.effect = result.effect.max(Effect::Mutate);
            result.reason = "redirects output into a sink".to_string();
            continue;
        }
        result.paths.push(PathUse {
            path: target.clone(),
            write: true,
        });
        result.effect = result.effect.max(Effect::Mutate);
        result.reason = "redirects output into a file".to_string();
    }

    let Some(program_word) = words.first() else {
        return result;
    };
    let program = basename(program_word);
    result.programs.push(program.clone());

    // `$CMD` or `${CMD}`: the program is decided at runtime and cannot be checked.
    if program_word.starts_with('$') {
        return Classification {
            effect: Effect::External,
            reason: "the program name comes from a variable, so it cannot be checked".to_string(),
            source: Some(External::DynamicProgram),
            programs: result.programs,
            paths: result.paths,
        };
    }

    let args: Vec<&str> = words.iter().skip(1).map(|word| word.as_str()).collect();
    let inner = classify_program(&program, &args);

    // The program's own verdict, unless a redirection already made it worse.
    let mut final_result = result;
    if inner.effect > final_result.effect {
        final_result.effect = inner.effect;
        final_result.reason = inner.reason;
        final_result.source = inner.source;
    }
    final_result.paths.extend(inner.paths);
    final_result
}

fn classify_program(program: &str, args: &[&str]) -> Classification {
    if program == "cd" || program == "pushd" {
        let mut result = Classification::new(Effect::Mutate, "changes the shell's directory");
        for target in args.iter().filter(|arg| !arg.starts_with('-')) {
            result.paths.push(PathUse {
                path: (*target).to_string(),
                write: false,
            });
        }
        return result;
    }

    if program == "git" {
        return classify_git(args);
    }
    if program == "cargo" {
        return classify_cargo(args);
    }
    if program == "npm" || program == "pnpm" || program == "yarn" || program == "bun" {
        return classify_node(program, args);
    }
    if program == "gh" {
        return classify_gh(args);
    }
    if program == "sed" || program == "perl" {
        return classify_in_place(program, args);
    }
    if program == "find" {
        return classify_find(args);
    }
    if program == "kill" || program == "pkill" {
        // Killing a process is not a file change, but it destroys work in progress and is
        // irreversible; a mutation is the honest bucket.
        return Classification::new(Effect::Mutate, "signals a process");
    }

    if EXTERNAL_PROGRAMS.contains(&program) {
        return Classification::external(
            External::NamedProgram,
            format!("`{program}` reaches the network, another user, or the whole system"),
        );
    }

    let mut result = if MUTATE_PROGRAMS.contains(&program) {
        Classification::new(Effect::Mutate, format!("`{program}` writes files"))
    } else if BUILTIN_PROGRAMS.contains(&program) {
        Classification::new(Effect::Read, format!("`{program}` is a shell builtin"))
    } else if READ_PROGRAMS.contains(&program) {
        Classification::new(Effect::Read, format!("`{program}` reads"))
    } else {
        // The design's rule: an unfamiliar command is external. An agent needs a named program
        // to do its job; a program nobody listed is one nobody reasoned about.
        return Classification::external(
            External::UnknownProgram,
            format!("`{program}` is not a known command, so it is treated as external"),
        );
    };

    if let Some((_, writes)) = PATH_PROGRAMS.iter().find(|(name, _)| *name == program) {
        let value_options = options_taking_values(program);
        let pattern_is_first_operand = takes_a_pattern_operand(program, args);
        let mut skip_next = false;
        let mut operand = 0usize;
        for arg in args {
            if skip_next {
                skip_next = false;
                continue;
            }
            if arg.starts_with('-') {
                if value_options.contains(&arg.trim_start_matches('-')) {
                    skip_next = true;
                }
                continue;
            }
            operand += 1;
            // `grep PATTERN FILE...` — the first operand is what to look for, and reading it as a
            // path is how a pattern like `~> *3\.4` became a denial: `~` is refused as a path
            // because a tool argument may not expand it, so the DoD check `grep -qE '~> *3\.4'
            // README.md` never ran and the gate reported "(no exit code)" with no reason at all.
            if pattern_is_first_operand && operand == 1 {
                continue;
            }
            result.paths.push(PathUse {
                path: (*arg).to_string(),
                write: *writes,
            });
        }
    }

    result
}

/// Whether the program's first non-flag operand is a pattern rather than a file.
///
/// `grep PATTERN FILE` puts a non-path first operand in front of the files, and reading it as a
/// path is not harmless. `grep -qE '~> *3\.4' README.md` is the exact definition-of-done check the
/// first live run used, and `~` is refused outright because a tool argument may not expand it: the
/// check was denied before it ran and the gate recorded "(no exit code)", which reads as a failing
/// test rather than as a policy refusal. The agent then spent two more attempts rewriting a README
/// that was already correct.
///
/// `-e` and `-f` supply the pattern themselves, so with either present the first operand is a file
/// again and must stay checked.
fn takes_a_pattern_operand(program: &str, args: &[&str]) -> bool {
    if !matches!(program, "grep" | "egrep" | "fgrep" | "rg") {
        return false;
    }
    // Both the separated form (`-e pattern`) and the joined one (`-epattern`) count: either way
    // the pattern came from the flag, not from a bare operand.
    let supplied = args.iter().any(|arg| {
        (*arg == "-e" || *arg == "-f")
            || (arg.starts_with("-e") && arg.len() > 2)
            || (arg.starts_with("-f") && arg.len() > 2)
    });
    !supplied
}

/// Options whose following word is a value rather than a path, per program.
///
/// Enough to stop `grep -A 3 pattern file` from claiming `3` is a file. Getting this wrong in the
/// permissive direction only produces a spurious path check, never a missed one.
fn options_taking_values(program: &str) -> &'static [&'static str] {
    match program {
        "grep" | "egrep" | "fgrep" | "rg" => &[
            "A",
            "B",
            "C",
            "m",
            "e",
            "f",
            "t",
            "g",
            "after-context",
            "before-context",
            "context",
            "max-count",
            "glob",
            "type",
        ],
        "head" | "tail" => &["n", "c"],
        "jq" => &["arg", "argjson", "slurpfile", "rawfile"],
        "tar" => &["f", "C", "directory", "file", "exclude"],
        "ln" => &["s", "S", "t"],
        "od" => &["A", "j", "N", "t", "w"],
        "sort" => &["k", "o", "t", "S"],
        "uniq" => &["f", "s", "w"],
        "column" => &["s", "t", "o", "c"],
        _ => &[],
    }
}

fn classify_git(args: &[&str]) -> Classification {
    let Some(subcommand) = args.iter().find(|arg| !arg.starts_with('-')) else {
        return Classification::new(Effect::Read, "`git` with no subcommand prints help");
    };
    match *subcommand {
        "status" | "diff" | "log" | "show" | "blame" | "rev-parse" | "ls-files" | "ls-tree" | "describe"
        | "shortlog" | "reflog" | "cat-file" | "merge-base" | "show-ref" | "for-each-ref" | "grep"
        | "fsck" | "count-objects" | "name-rev" | "whatchanged" | "diff-tree" | "diff-files"
        | "diff-index" | "check-ignore" | "check-attr" | "var" | "help" | "version" | "verify-commit"
        | "verify-tag" => Classification::new(Effect::Read, format!("`git {subcommand}` reads")),
        "branch" | "tag" => {
            // Listing is a read; creating, moving or deleting is not.
            let listing = args
                .iter()
                .skip_while(|arg| *arg != subcommand)
                .skip(1)
                .all(|arg| arg.starts_with('-') || arg.is_empty());
            let deletes = args
                .iter()
                .any(|arg| matches!(*arg, "-d" | "-D" | "-m" | "-M" | "-f" | "--delete" | "--force"));
            if listing && !deletes {
                Classification::new(Effect::Read, format!("`git {subcommand}` lists"))
            } else {
                Classification::new(Effect::Mutate, format!("`git {subcommand}` changes refs"))
            }
        }
        "remote" => {
            let flag_only = args
                .iter()
                .skip_while(|arg| *arg != subcommand)
                .skip(1)
                .all(|arg| arg.starts_with('-') || arg.is_empty());
            if flag_only {
                Classification::new(Effect::Read, "`git remote` lists")
            } else {
                Classification::external(
                    External::RemoteSubcommand,
                    "`git remote` reconfigures network remotes",
                )
            }
        }
        "stash" => {
            let lists = args.iter().any(|arg| matches!(*arg, "list" | "show"));
            if lists {
                Classification::new(Effect::Read, "`git stash list` reads")
            } else {
                Classification::new(Effect::Mutate, "`git stash` moves working-tree changes")
            }
        }
        "config" => {
            let writes = args.iter().any(|arg| {
                matches!(
                    *arg,
                    "--add" | "--unset" | "--unset-all" | "--replace-all" | "--edit"
                )
            });
            let reads = args
                .iter()
                .any(|arg| matches!(*arg, "--get" | "--get-all" | "--list" | "-l"));
            if writes || !reads {
                Classification::new(Effect::Mutate, "`git config` may write `.git/config`")
            } else {
                Classification::new(Effect::Read, "`git config --get` reads")
            }
        }
        "worktree" => {
            let lists = args.contains(&"list");
            if lists {
                Classification::new(Effect::Read, "`git worktree list` reads")
            } else {
                Classification::new(Effect::Mutate, "`git worktree` changes worktrees on disk")
            }
        }
        "push" | "fetch" | "pull" | "clone" | "send-email" | "request-pull" | "lfs" | "submodule"
        | "archive" => Classification::external(
            External::RemoteSubcommand,
            format!("`git {subcommand}` talks to a remote"),
        ),
        "add" | "commit" | "rm" | "mv" | "checkout" | "switch" | "restore" | "reset" | "merge" | "rebase"
        | "cherry-pick" | "revert" | "clean" | "apply" | "am" | "init" | "gc" | "prune" | "update-ref"
        | "symbolic-ref" | "notes" | "bisect" | "sparse-checkout" | "maintenance" | "update-index"
        | "filter-branch" | "repack" | "pack-refs" => {
            Classification::new(Effect::Mutate, format!("`git {subcommand}` writes"))
        }
        other => Classification::external(
            External::UnknownSubcommand,
            format!("`git {other}` is not a known subcommand"),
        ),
    }
}

fn classify_cargo(args: &[&str]) -> Classification {
    let Some(subcommand) = args.iter().find(|arg| !arg.starts_with('-')) else {
        return Classification::new(Effect::Read, "`cargo` with no subcommand prints help");
    };
    match *subcommand {
        "metadata" | "tree" | "locate-project" | "pkgid" | "verify-project" | "version" | "--version"
        | "--list" => Classification::new(Effect::Read, format!("`cargo {subcommand}` reads")),
        "build" | "b" | "test" | "t" | "check" | "c" | "clippy" | "fmt" | "doc" | "run" | "r" | "bench"
        | "clean" | "fix" | "rustc" | "nextest" => Classification::new(
            Effect::Mutate,
            format!("`cargo {subcommand}` writes `target/` and runs build code"),
        ),
        "update" | "add" | "remove" | "install" | "uninstall" | "publish" | "search" | "login" | "logout"
        | "yank" | "owner" | "fetch" => Classification::external(
            External::RemoteSubcommand,
            format!("`cargo {subcommand}` reaches the network or writes outside the worktree"),
        ),
        other => Classification::external(
            External::UnknownSubcommand,
            format!("`cargo {other}` is not a known subcommand"),
        ),
    }
}

fn classify_node(program: &str, args: &[&str]) -> Classification {
    let subcommand = args
        .iter()
        .find(|arg| !arg.starts_with('-'))
        .copied()
        .unwrap_or("list");
    match subcommand {
        "ls" | "list" | "why" | "view" | "outdated" | "help" => {
            Classification::new(Effect::Read, format!("`{program}` lists"))
        }
        "test" | "t" | "run" => {
            // `npm run` executes a script this module cannot see, so the honest answer is
            // external: the script is in `package.json` and nobody classified it.
            Classification::external(
                External::OpaqueScript,
                format!("`{program} {subcommand}` runs a script whose contents are unknown"),
            )
        }
        "install" | "i" | "add" | "ci" | "publish" | "update" | "link" | "exec" | "dlx" | "create" => {
            Classification::external(
                External::RemoteSubcommand,
                format!("`{program} {subcommand}` reaches the network or writes outside the worktree"),
            )
        }
        other => Classification::new(Effect::Mutate, format!("`{program} {other}` changes the project")),
    }
}

fn classify_gh(args: &[&str]) -> Classification {
    let Some(subcommand) = args.iter().find(|arg| !arg.starts_with('-')) else {
        return Classification::new(Effect::Read, "`gh` with no subcommand prints help");
    };
    match *subcommand {
        "pr" | "issue" | "repo" | "release" | "workflow" | "run" | "api" | "auth" | "gist" | "search"
        | "codespace" | "extension" | "secret" => Classification::external(
            External::RemoteSubcommand,
            format!("`gh {subcommand}` talks to GitHub"),
        ),
        other => Classification::external(
            External::UnknownSubcommand,
            format!("`gh {other}` is not a known subcommand"),
        ),
    }
}

fn classify_in_place(program: &str, args: &[&str]) -> Classification {
    let in_place = args.iter().any(|arg| {
        (program == "sed" && (*arg == "--in-place" || arg.starts_with("-i")))
            || (program == "perl" && (*arg == "-i" || arg.starts_with("-pi")))
    });
    if in_place {
        let mut result = Classification::new(
            Effect::Mutate,
            format!("`{program}` rewrites its input files in place"),
        );
        // The file operands follow the script; they are the last non-flag arguments.
        for arg in args.iter().filter(|arg| !arg.starts_with('-')) {
            result.paths.push(PathUse {
                path: (*arg).to_string(),
                write: true,
            });
        }
        result
    } else if program == "perl" {
        Classification::external(
            External::OpaqueScript,
            "`perl -e` runs arbitrary code, so its effect cannot be classified",
        )
    } else {
        Classification::new(Effect::Read, "`sed` without `-i` writes only to stdout")
    }
}

fn classify_find(args: &[&str]) -> Classification {
    let dangerous = args.iter().any(|arg| {
        matches!(
            *arg,
            "-delete"
                | "-exec"
                | "-execdir"
                | "-ok"
                | "-okdir"
                | "-fprint"
                | "-fprint0"
                | "-fprintf"
                | "-fls"
        )
    });
    if dangerous {
        Classification::external(
            External::OpaqueScript,
            "`find` with `-delete` or `-exec` runs commands this module cannot see",
        )
    } else {
        Classification::new(Effect::Read, "`find` lists")
    }
}

fn basename(word: &str) -> String {
    // `/usr/bin/git` and `git` are the same program; the path is kept out of the table match so
    // that invoking by absolute path cannot dodge the rules.
    word.rsplit('/').next().unwrap_or(word).to_string()
}

/// Whether a redirection target names a sink rather than a file.
///
/// Only the device files POSIX guarantees on every system, and only by exact name. A prefix rule
/// would let `> /dev/null.txt` skip the path check, and a broader `/dev/*` rule would excuse
/// writes to block devices — the one class of target where a wrong "harmless" answer is worst.
///
/// `/dev/stdout` and `/dev/stderr` are here because they are the same thing as `>&1` and `>&2`
/// said with a path, and a model writes them interchangeably. Closing the descriptor
/// (`>&-`) needs no entry: the lexer already refuses to read it as a target.
fn is_sink(target: &str) -> bool {
    matches!(
        target,
        "/dev/null" | "/dev/stdout" | "/dev/stderr" | "/dev/zero" | "/dev/full"
    )
}

fn is_assignment(word: &str) -> bool {
    match word.split_once('=') {
        Some((name, _)) => {
            !name.is_empty()
                && name.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
                && !name.starts_with(|ch: char| ch.is_ascii_digit())
        }
        None => false,
    }
}

/// The set of programs a command line touches, for logging.
#[must_use]
pub fn programs(command: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    for simple in split_commands(command) {
        if let Some(first) = simple
            .words
            .iter()
            .find(|word| !word.is_empty() && !is_assignment(word))
        {
            seen.insert(basename(first));
        }
    }
    seen.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn effect(command: &str) -> Effect {
        classify(command).effect
    }

    fn paths(command: &str) -> Vec<(String, bool)> {
        classify(command)
            .paths
            .into_iter()
            .map(|used| (used.path, used.write))
            .collect()
    }

    // ── the design's own table ──────────────────────────────────────────────────

    #[test]
    fn the_designs_read_examples_are_reads() {
        for command in [
            "git status",
            "git diff",
            "git log --oneline -20",
            "ls -la",
            "cat src/main.rs",
            "grep -R fn src",
            "cargo check",
            "cargo metadata",
        ] {
            let classified = classify(command);
            assert!(
                classified.effect <= Effect::Mutate,
                "{command} became {:?}: {}",
                classified.effect,
                classified.reason
            );
        }
    }

    #[test]
    fn a_build_is_a_mutation_and_never_a_read() {
        // The design lists `cargo test` as a read. That is the dangerous direction: a read is
        // allowed unconditionally, so a command that writes `target/` and runs every `build.rs`
        // would skip the path check entirely.
        for command in [
            "cargo test",
            "cargo build",
            "cargo clippy",
            "cargo fmt",
            "cargo run",
        ] {
            let classified = classify(command);
            assert_eq!(classified.effect, Effect::Mutate, "{command}: {classified:?}");
            assert!(
                classified.reason.contains("target"),
                "{command}: {}",
                classified.reason
            );
        }
    }

    #[test]
    fn the_designs_external_examples_are_external() {
        for command in [
            "curl https://example.com",
            "wget http://example.com/x",
            "ssh host uptime",
            "scp a b:",
            "git push",
            "git fetch",
            "git clone https://example.com/r",
            "brew install jq",
            "sudo ls",
        ] {
            assert_eq!(effect(command), Effect::External, "{command}");
        }
    }

    #[test]
    fn every_external_verdict_carries_a_structural_reason() {
        // The invariant the judge depends on (D-111): it can only branch on `source`, so an
        // `External` with no `source` is a case nobody can route — silently, since the effect is
        // still right. This is the test that fails when a ninth way to reach `External` is added
        // without deciding what it is.
        for command in [
            "curl https://example.com",
            "frobnicate --now",
            "git arcane",
            "git push",
            "cargo publish",
            "gh pr list",
            "npm run build",
            "perl -e 'print 1'",
            "find . -exec rm {} +",
            "$CMD x",
            "echo $(unterminated",
            "bundle exec rspec",
        ] {
            let classified = classify(command);
            assert_eq!(classified.effect, Effect::External, "{command}");
            assert!(
                classified.source.is_some(),
                "`{command}` is external with no structural reason: {classified:?}"
            );
        }
    }

    #[test]
    fn only_an_unknown_program_is_a_question_for_the_judge() {
        // The line D-111 draws. A named program and an operator's deny are statements, not
        // uncertainties: letting a model answer those would let it talk the policy round.
        let question = |command: &str| {
            let source = classify(command).source.expect("external carries a source");
            source.is_a_question()
        };
        assert!(
            question("frobnicate --now"),
            "an unfamiliar program is the one question"
        );
        assert!(
            question("bundle exec rspec"),
            "the measured case: a wrapper around a gem"
        );
        assert!(!question("curl https://example.com"), "a named external program");
        assert!(!question("git push"), "a remote subcommand");
        assert!(
            !question("git arcane"),
            "an unknown subcommand of a known program"
        );
        assert!(!question("perl -e 'print 1'"), "an opaque script");
        assert!(!question("$CMD x"), "a program decided at run time");
        assert!(
            !question("echo $(unterminated"),
            "an unreadable line is not something a model can judge"
        );
    }

    #[test]
    fn the_designs_mutation_examples_are_mutations() {
        for command in [
            "mkdir -p src/new",
            "mv a b",
            "sed -i 's/a/b/' file.txt",
            "rm -rf target",
            "touch notes.md",
            "cp a b",
        ] {
            assert_eq!(effect(command), Effect::Mutate, "{command}");
        }
    }

    // ── the unknown-command rule ────────────────────────────────────────────────

    #[test]
    fn an_unknown_program_is_external() {
        let classified = classify("frobnicate --all");
        assert_eq!(classified.effect, Effect::External);
        assert!(
            classified.reason.contains("not a known command"),
            "{classified:?}"
        );
    }

    #[test]
    fn an_absolute_path_does_not_dodge_the_table() {
        // Otherwise `/usr/bin/curl` is "unknown" and `/tmp/curl` is a script.
        assert_eq!(effect("/usr/bin/git push"), Effect::External);
        assert_eq!(effect("/bin/ls"), Effect::Read);
    }

    #[test]
    fn a_variable_as_the_program_is_external() {
        // `$CMD` cannot be checked at all, which is the definition of external.
        let classified = classify("$CMD --force");
        assert_eq!(classified.effect, Effect::External);
        assert!(classified.reason.contains("variable"), "{classified:?}");
    }

    // ── pipelines and sequencing ────────────────────────────────────────────────

    #[test]
    fn a_pipeline_takes_the_worst_part() {
        assert_eq!(effect("cat a | grep b"), Effect::Read);
        assert_eq!(effect("cat a | tee b"), Effect::Mutate);
        assert_eq!(effect("ls && curl https://x"), Effect::External);
        assert_eq!(effect("ls; rm -rf x"), Effect::Mutate);
        assert_eq!(effect("true && curl https://x"), Effect::External);
    }

    #[test]
    fn command_substitution_is_classified_on_its_own() {
        // `echo $(curl evil)` must not inherit `echo`'s harmlessness.
        assert_eq!(effect("echo $(curl https://evil.example/x)"), Effect::External);
        assert_eq!(effect("echo `curl https://evil.example/x`"), Effect::External);
        assert_eq!(effect("cd $(mkdir /tmp/x)"), Effect::Mutate);
    }

    #[test]
    fn a_nested_substitution_is_unwrapped() {
        assert_eq!(effect("echo $(echo $(curl https://x))"), Effect::External);
    }

    #[test]
    fn an_unterminated_substitution_does_not_hide_a_command() {
        // A lexer that silently drops the tail would make `x` invisible; the conservative answer
        // is to keep it in the word and let the unknown program rule fire.
        let classified = classify("echo $(frobnicate");
        assert!(classified.effect >= Effect::Mutate, "{classified:?}");
    }

    #[test]
    fn a_separator_inside_quotes_is_literal() {
        assert_eq!(effect(r#"echo "a;b""#), Effect::Read);
        assert_eq!(effect("echo 'a|b'"), Effect::Read);
        assert_eq!(effect(r"echo a\;b"), Effect::Read);
    }

    // ── redirections ────────────────────────────────────────────────────────────

    #[test]
    fn a_redirect_into_a_file_is_a_mutation_that_names_the_file() {
        let classified = classify("echo hi > notes.txt");
        assert_eq!(classified.effect, Effect::Mutate);
        assert_eq!(classified.paths, vec![PathUse::write("notes.txt")]);
    }

    #[test]
    fn a_redirect_to_a_fixed_path_is_still_a_write_to_that_path() {
        // The policy needs the target, not just the severity: this is how `> /etc/passwd` is
        // caught.
        assert_eq!(
            paths("echo hi > /etc/passwd"),
            vec![("/etc/passwd".to_string(), true)]
        );
    }

    #[test]
    fn a_descriptor_redirect_is_not_a_path() {
        // `2>&1` is not a file named `&1`.
        assert_eq!(effect("ls 2>&1"), Effect::Read);
        assert!(paths("ls 2>&1").is_empty(), "{:?}", paths("ls 2>&1"));
        assert_eq!(effect("cargo test 2>/dev/null"), Effect::Mutate);
    }

    #[test]
    fn a_redirect_to_a_device_sink_is_not_a_path() {
        // The refusal this test exists for: `cmd 2>/dev/null` used to arrive at the gate as a
        // write to `/dev/null`, which is outside the worktree, so the ordinary way to silence a
        // warning was denied as an escape attempt. Measured on the first live run: five steps
        // refused for this alone, and two subtasks blocked on gates whose command was denied
        // before it ran, reporting "no exit code" rather than a failing check.
        assert!(
            paths("cargo test 2>/dev/null").is_empty(),
            "{:?}",
            paths("cargo test 2>/dev/null")
        );
        // The effect is still a mutation: `cargo test` writes `target/`, and dropping to `Read`
        // would let the same command skip the worktree check it needs.
        assert_eq!(effect("cargo test 2>/dev/null"), Effect::Mutate);
        for command in [
            "rspec 2>/dev/null",
            "rspec > /dev/null",
            "rspec 2> /dev/stderr",
            "rspec 1>/dev/stdout",
            "rspec > /dev/zero",
        ] {
            assert!(paths(command).is_empty(), "{command}: {:?}", paths(command));
        }
    }

    #[test]
    fn a_sink_lookalike_is_still_a_path() {
        // The rule is an exact-name match, not a prefix: `/dev/null.txt` is an ordinary file and
        // `/dev/sda` is a block device. A broader match would excuse exactly the writes that
        // matter most.
        for command in [
            "rspec > /dev/null.txt",
            "rspec > /dev/sda",
            "rspec > /dev/null/../passwd",
            "rspec > mydev/null",
        ] {
            assert_eq!(
                paths(command).len(),
                1,
                "{command} must still name a path: {:?}",
                paths(command)
            );
        }
    }

    #[test]
    fn an_input_redirect_is_not_a_write() {
        assert_eq!(effect("wc -l < notes.txt"), Effect::Read);
    }

    #[test]
    fn a_quoted_redirect_target_keeps_its_spaces() {
        assert_eq!(
            paths("echo hi > 'my notes.txt'"),
            vec![("my notes.txt".to_string(), true)]
        );
    }

    // ── program-specific rules ──────────────────────────────────────────────────

    #[test]
    fn git_branch_listing_reads_but_creating_writes() {
        assert_eq!(effect("git branch"), Effect::Read);
        assert_eq!(effect("git branch -a"), Effect::Read);
        assert_eq!(effect("git branch new-work"), Effect::Mutate);
        assert_eq!(effect("git branch -D old"), Effect::Mutate);
    }

    #[test]
    fn git_remote_without_arguments_lists_and_with_arguments_is_external() {
        assert_eq!(effect("git remote"), Effect::Read);
        assert_eq!(effect("git remote -v"), Effect::Read);
        assert_eq!(effect("git remote add origin https://x"), Effect::External);
    }

    #[test]
    fn git_config_listing_reads_and_writing_writes() {
        assert_eq!(effect("git config --get user.email"), Effect::Read);
        assert_eq!(effect("git config --list"), Effect::Read);
        assert_eq!(effect("git config user.email a@b"), Effect::Mutate);
        assert_eq!(effect("git config --add x.y z"), Effect::Mutate);
    }

    #[test]
    fn git_worktree_listing_reads_and_creating_writes() {
        assert_eq!(effect("git worktree list"), Effect::Read);
        assert_eq!(effect("git worktree add ../w branch"), Effect::Mutate);
    }

    #[test]
    fn git_stash_listing_reads_and_stashing_writes() {
        assert_eq!(effect("git stash list"), Effect::Read);
        assert_eq!(effect("git stash"), Effect::Mutate);
        assert_eq!(effect("git stash pop"), Effect::Mutate);
    }

    #[test]
    fn an_unknown_git_subcommand_is_external() {
        assert_eq!(effect("git frobnicate"), Effect::External);
    }

    #[test]
    fn git_with_no_subcommand_is_a_read() {
        assert_eq!(effect("git"), Effect::Read);
        assert_eq!(effect("git --version"), Effect::Read);
    }

    #[test]
    fn npm_run_is_external_because_the_script_is_invisible() {
        // `npm run build` executes whatever `package.json` says, and nobody classified that.
        assert_eq!(effect("npm run build"), Effect::External);
        assert_eq!(effect("npm test"), Effect::External);
        assert_eq!(effect("npm install"), Effect::External);
        assert_eq!(effect("npm ls"), Effect::Read);
    }

    #[test]
    fn cargo_publish_is_external_but_cargo_check_is_not() {
        assert_eq!(effect("cargo publish"), Effect::External);
        assert_eq!(effect("cargo update"), Effect::External);
        assert_eq!(effect("cargo check"), Effect::Mutate);
    }

    #[test]
    fn sed_without_in_place_is_a_read_and_with_it_is_a_mutation() {
        assert_eq!(effect("sed 's/a/b/' file"), Effect::Read);
        assert_eq!(effect("sed -i 's/a/b/' file"), Effect::Mutate);
        assert_eq!(effect("sed -i.bak 's/a/b/' file"), Effect::Mutate);
        assert_eq!(effect("sed --in-place 's/a/b/' file"), Effect::Mutate);
    }

    #[test]
    fn find_with_exec_or_delete_is_external() {
        assert_eq!(effect("find . -name '*.rs'"), Effect::Read);
        assert_eq!(effect("find . -delete"), Effect::External);
        assert_eq!(effect("find . -exec rm {} \\;"), Effect::External);
        assert_eq!(effect("find . -execdir rm {} \\;"), Effect::External);
    }

    #[test]
    fn xargs_is_external_because_its_command_is_a_positional_argument() {
        assert_eq!(effect("ls | xargs rm"), Effect::External);
    }

    #[test]
    fn perl_e_is_external_because_it_is_arbitrary_code() {
        assert_eq!(effect("perl -e 'unlink 1'"), Effect::External);
    }

    #[test]
    fn gh_is_external_for_every_subcommand() {
        assert_eq!(effect("gh pr list"), Effect::External);
        assert_eq!(effect("gh api /repos"), Effect::External);
    }

    #[test]
    fn kill_is_a_mutation_because_it_destroys_work_in_progress() {
        assert_eq!(effect("kill 1234"), Effect::Mutate);
        assert_eq!(effect("pkill -f node"), Effect::Mutate);
    }

    #[test]
    fn cd_is_a_mutation_that_names_its_directory() {
        let classified = classify("cd src/gen");
        assert_eq!(classified.effect, Effect::Mutate);
        assert_eq!(classified.paths, vec![PathUse::read("src/gen")]);
    }

    // ── path extraction ─────────────────────────────────────────────────────────

    #[test]
    fn read_programs_name_the_paths_they_read() {
        assert_eq!(paths("cat src/main.rs"), vec![("src/main.rs".to_string(), false)]);
        assert_eq!(
            paths("head -n 20 notes.md"),
            vec![("notes.md".to_string(), false)],
            "-n 20 must not become a path"
        );
    }

    #[test]
    fn a_programs_option_value_is_not_mistaken_for_a_path() {
        // `grep -A 3 pattern file` must not claim that `3` is a file to check.
        let extracted = paths("grep -A 3 TODO src/main.rs");
        assert!(!extracted.iter().any(|(path, _)| path == "3"), "{extracted:?}");
        assert!(extracted.iter().any(|(path, _)| path == "src/main.rs"));
    }

    #[test]
    fn grep_pattern_is_not_a_path() {
        // The check the first live run never got to run. `~` is refused as a path by design — a
        // tool argument may not expand it — so reading the pattern as one denied
        // `grep -qE '~> *3\.4' README.md` before it executed and the gate recorded "(no exit
        // code)". Only the file operand is a path.
        let classified = classify("grep -qE '~> *3\\.4' README.md");
        assert_eq!(classified.effect, Effect::Read);
        assert_eq!(classified.paths, vec![PathUse::read("README.md")]);

        // A pattern that happens to be a real path-looking word is still a pattern, and the file
        // after it is still checked: dropping both would let `grep secret.txt config/.env` read
        // the file it names.
        let extracted = paths("grep pattern config/.env");
        assert_eq!(extracted, vec![("config/.env".to_string(), false)]);

        // `-e` supplies the pattern, so the first operand is a file again.
        assert_eq!(
            paths("grep -e pattern README.md"),
            vec![("README.md".to_string(), false)]
        );
        assert_eq!(
            paths("grep -epattern README.md"),
            vec![("README.md".to_string(), false)]
        );
    }

    #[test]
    fn write_programs_name_their_targets_as_writes() {
        assert!(
            paths("rm -rf target")
                .iter()
                .any(|(path, write)| path == "target" && *write)
        );
        assert!(
            paths("mkdir -p a/b")
                .iter()
                .any(|(path, write)| path == "a/b" && *write)
        );
    }

    #[test]
    fn a_command_with_no_recognised_paths_reports_none() {
        // Nothing here is a claim that the command is harmless — it is a claim that the gate has
        // no path to check, and the effect classification carries the rest.
        assert_eq!(paths("git commit -m x"), Vec::<(String, bool)>::new());
    }

    // ── degenerate input ────────────────────────────────────────────────────────

    #[test]
    fn an_empty_or_blank_command_is_a_read_that_does_nothing() {
        for command in ["", "   ", "\n", ";"] {
            let classified = classify(command);
            assert_eq!(classified.effect, Effect::Read, "{command:?}");
            assert!(classified.reason.contains("empty"), "{classified:?}");
        }
    }

    #[test]
    fn an_assignment_prefix_does_not_hide_the_program() {
        assert_eq!(effect("FOO=1 curl https://x"), Effect::External);
        assert_eq!(effect("A=1 B=2 ls"), Effect::Read);
    }

    #[test]
    fn a_quoted_program_name_is_still_matched() {
        assert_eq!(effect("\"curl\" https://x"), Effect::External);
    }

    #[test]
    fn the_program_list_is_deduplicated_and_ordered() {
        let programs = programs("ls a | grep b | ls c");
        assert_eq!(programs, vec!["grep".to_string(), "ls".to_string()]);
    }

    #[test]
    fn classification_is_deterministic() {
        let first = classify("cat a | tee b && curl https://x");
        let second = classify("cat a | tee b && curl https://x");
        assert_eq!(first, second);
    }
}
