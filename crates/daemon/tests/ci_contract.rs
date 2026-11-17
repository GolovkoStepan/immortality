//! The local gate must be the CI gate, and a suppressed advisory must carry its decision.
//!
//! # Why this is a test and not a line in `AGENTS.md`
//!
//! `v0.1.3` was tagged while the `dependencies` job was red. Nothing was missed locally in the
//! sense of a check that failed and was ignored: the audit was never run, because `just ci`
//! defined itself as `fmt-check lint test` while the workflow had four jobs. The `justfile`
//! carried a comment claiming it ran "exactly what GitHub Actions runs", which was true of the
//! three recipes it named and false about the file — and a comment cannot fail.
//!
//! A suppression has the same shape of problem. `ignore = ["RUSTSEC-…"]` makes the audit green
//! and, once merged, is indistinguishable from "we checked and it is fine". The debt it stands
//! for survives only if something reads it. So the rule enforced here: every ignored advisory
//! must name itself as an advisory and must be attached to a recorded decision (`D-NNN`). An
//! entry added to silence a failure, with no reasoning above it, fails the suite.
//!
//! # What this deliberately does not check
//!
//! It compares *names*, not the command text inside each recipe. `test` could drift from the
//! workflow's `cargo test --workspace --all-features` and this test would still pass. Checking
//! that would mean executing the recipes, which is what `just ci` itself is for; the property
//! worth pinning cheaply is that a job cannot appear in CI without a local counterpart at all.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Which recipe stands in for which workflow job.
///
/// The workflow and this table are compared in **both** directions: a job with no row fails,
/// and a row naming a recipe that `ci` does not depend on fails. Adding a fifth job to CI
/// therefore forces a decision here instead of silently widening the gap.
const JOB_COVERAGE: &[(&str, &[&str])] = &[
    // Format, clippy and test all run inside this job.
    ("test-linux", &["fmt-check", "lint", "test"]),
    // Same commands; only the runner differs, and that difference is not expressible as a
    // recipe.
    ("test-macos", &["test"]),
    ("deny", &["audit"]),
    ("layering", &["layering"]),
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn read(relative: &str) -> String {
    let path = repo_root().join(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("{} should be readable: {err}", path.display()))
}

/// The job keys of `jobs:` in the workflow, at exactly two spaces of indentation.
///
/// Parsed rather than grepped: `name: test (linux)` and `runs-on:` sit at four spaces, and the
/// step keys at six, so the indentation is what distinguishes a job from its attributes.
fn ci_jobs(workflow: &str) -> Vec<String> {
    let mut jobs = Vec::new();
    let mut in_jobs = false;
    for line in workflow.lines() {
        if line == "jobs:" {
            in_jobs = true;
            continue;
        }
        if !in_jobs {
            continue;
        }
        if !line.is_empty() && !line.starts_with(' ') {
            // Back at the document level (`env:`, `on:` in a reordered file).
            in_jobs = false;
            continue;
        }
        let Some(rest) = line.strip_prefix("  ") else {
            continue;
        };
        if rest.starts_with(' ') {
            continue;
        }
        if let Some(name) = rest.strip_suffix(':')
            && !name.is_empty()
            && !name.contains(' ')
        {
            jobs.push(name.to_owned());
        }
    }
    jobs
}

/// A recipe: the dependencies on its header line, and its indented body.
struct Recipe {
    deps: Vec<String>,
    body: Vec<String>,
}

/// Recipes are top-level `name: deps` lines followed by indented lines.
fn just_recipes(justfile: &str) -> BTreeMap<String, Recipe> {
    let mut recipes: BTreeMap<String, Recipe> = BTreeMap::new();
    let mut current: Option<String> = None;
    for line in justfile.lines() {
        let indented = line.starts_with(' ') || line.starts_with('\t');
        if indented {
            if let Some(name) = &current
                && let Some(recipe) = recipes.get_mut(name)
            {
                recipe.body.push(line.trim().to_owned());
            }
            continue;
        }
        current = None;
        if line.is_empty() || line.starts_with('#') || !line.contains(':') {
            continue;
        }
        // Settings (`set shell := [...]`) are not recipes, and `:=` would otherwise read as a
        // recipe named `set shell`.
        if line.starts_with("set ") || line.starts_with("export ") {
            continue;
        }
        let (header, tail) = line.split_once(':').expect("checked above");
        let Some(name) = header.split_whitespace().next() else {
            continue;
        };
        // `name: dep-a dep-b` — dependencies are to the *right* of the colon. Right-hand words
        // carrying `=` are recipe arguments, not dependencies.
        let deps: Vec<String> = tail
            .split_whitespace()
            .filter(|word| !word.contains('='))
            .map(ToOwned::to_owned)
            .collect();
        current = Some(name.to_owned());
        recipes.insert(
            name.to_owned(),
            Recipe {
                deps,
                body: Vec::new(),
            },
        );
    }
    recipes
}

#[test]
fn every_ci_job_has_a_recipe_that_the_local_gate_runs() {
    let jobs = ci_jobs(&read(".github/workflows/ci.yml"));
    assert!(
        jobs.len() >= 4,
        "the workflow should have at least the four known jobs, parsed: {jobs:?}"
    );

    let recipes = just_recipes(&read("justfile"));
    let ci = recipes
        .get("ci")
        .expect("the justfile should define a `ci` recipe");

    for job in &jobs {
        let recipes_for_job = JOB_COVERAGE
            .iter()
            .find(|(name, _)| name == job)
            .unwrap_or_else(|| {
                panic!(
                    "ci.yml has job `{job}` with no local counterpart. Add a recipe, make `ci` \
                     depend on it, and add a JOB_COVERAGE row — a job nobody can run before a \
                     push is how v0.1.3 shipped red."
                )
            })
            .1;
        for recipe in recipes_for_job {
            assert!(
                recipes.contains_key(*recipe),
                "JOB_COVERAGE maps job `{job}` to recipe `{recipe}`, which the justfile does \
                 not define"
            );
            assert!(
                ci.deps.iter().any(|dep| dep == recipe),
                "`just ci` does not depend on `{recipe}` (it runs {:?}), but CI job `{job}` \
                 does. Run `just ci` before tagging and it would still be green while CI is \
                 red.",
                ci.deps
            );
        }
    }
}

#[test]
fn the_local_gate_names_only_recipes_that_exist() {
    let recipes = just_recipes(&read("justfile"));
    let ci = recipes.get("ci").expect("`ci` recipe");
    for dep in &ci.deps {
        assert!(
            recipes.contains_key(dep),
            "`just ci` depends on `{dep}`, which no recipe defines — `just ci` would fail on a \
             typo rather than run the check"
        );
    }
}

#[test]
fn the_dependency_audit_is_part_of_the_local_gate() {
    // Singled out from the coverage table because this is the exact check that was missing,
    // and a reader of the file should find the reason here rather than reconstruct it.
    let recipes = just_recipes(&read("justfile"));
    let audit = recipes
        .get("audit")
        .expect("the justfile should define an `audit` recipe");
    let command = audit.body.join(" ");
    assert!(
        command.contains("cargo deny check advisories bans licenses sources"),
        "the `audit` recipe should run the same `cargo deny` selection as the `dependencies` \
         job, found: {command}"
    );
}

#[test]
fn an_ignored_advisory_names_itself_and_the_decision_that_owns_it() {
    let deny = read("deny.toml");
    let ignored = ignored_advisories(&deny);

    for (id, leading_comments) in &ignored {
        assert!(
            id.starts_with("RUSTSEC-"),
            "`ignore` in deny.toml should hold advisory ids, found `{id}`. Silencing a lint or \
             a duplicate-version warning belongs in its own section with its own reason."
        );
        assert!(
            leading_comments.iter().any(|line| mentions_decision(line)),
            "the ignored advisory `{id}` has no `D-NNN` in the comment above it. A suppression \
             without a recorded decision is indistinguishable from \"checked, not a problem\" \
             long before anyone re-reads it — name the decision, or take the update."
        );
        assert!(
            leading_comments.iter().any(|line| line.contains(id)),
            "the ignored advisory `{id}` is not named in the comment above it, so the reason \
             cannot be read next to the id it explains"
        );
    }
}

fn mentions_decision(line: &str) -> bool {
    // `D-143` anywhere in a comment line. Digits are required so a stray `D-` in prose does
    // not count.
    let bytes = line.as_bytes();
    bytes
        .windows(2)
        .enumerate()
        .any(|(index, pair)| pair == b"D-" && bytes.get(index + 2).is_some_and(u8::is_ascii_digit))
}

/// The `ignore = [...]` entries, each with the comment block that precedes it.
fn ignored_advisories(deny: &str) -> Vec<(String, Vec<String>)> {
    let mut found = Vec::new();
    let mut in_ignore = false;
    let mut comments: Vec<String> = Vec::new();
    for line in deny.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            comments.push(trimmed.to_owned());
            continue;
        }
        if !in_ignore && trimmed.starts_with("ignore = [") {
            in_ignore = !trimmed.contains(']');
            continue;
        }
        if !in_ignore {
            comments.clear();
            continue;
        }
        if trimmed.starts_with(']') {
            in_ignore = false;
            comments.clear();
            continue;
        }
        if let Some(value) = trimmed.strip_prefix('"')
            && let Some(id) = value.split('"').next()
        {
            found.push((id.to_owned(), comments.clone()));
        }
        comments.clear();
    }
    found
}

#[test]
fn the_parser_finds_the_advisory_that_is_actually_ignored() {
    // Guards the three tests above from passing because the parser returned nothing: an
    // empty list satisfies every `for` loop. The count is not asserted (ignoring more is a
    // decision), only that the known entry is seen with its reasoning.
    let ignored = ignored_advisories(&read("deny.toml"));
    assert!(
        !ignored.is_empty(),
        "deny.toml should ignore at least RUSTSEC-2026-0543 (chrono, D-143); if that is no \
         longer true, delete this assertion deliberately rather than removing the entry's \
         reason"
    );
    let (id, comments) = &ignored[0];
    assert!(
        comments.iter().any(|line| mentions_decision(line)),
        "the parser lost the comment block for `{id}`: {comments:?}"
    );
}
