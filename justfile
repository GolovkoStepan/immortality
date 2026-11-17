# `just ci` runs every check GitHub Actions runs, and a test enforces that it keeps doing so.
#
# The point is that a red CI run should never be the first time a problem is seen. That claim
# used to be a comment on this file while `ci` skipped two of the four jobs, which is how
# `v0.1.3` was tagged with a red `dependencies` job: the advisory was published that morning,
# and nothing local would have said so. `crates/daemon/tests/ci_contract.rs` now parses this
# file and `.github/workflows/ci.yml` and fails when they drift apart, so adding a job without
# a recipe — or dropping a recipe from `ci` — breaks the suite instead of the release.
#
# `just --list` uses the *last* comment line before a recipe as its description, so each recipe
# has a one-line summary directly above it and any longer reasoning above that, after a blank
# line.

set shell := ["bash", "-euo", "pipefail", "-c"]

default: ci

# Everything CI checks.
#
# All four jobs of the workflow. `test-linux` and `test-macos` share the `test` recipe; the
# runner difference is a platform thing, not a different command.

# `fmt-check`, `lint`, `test`, `audit` and `layering` — all four CI jobs.
ci: fmt-check lint test audit layering

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all --check

# Clippy with `-D warnings`, exactly as the workflow runs it.
lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
    cargo test --workspace --all-features

# The `dependencies` job.
#
# Needs the network the first time and a `RUSTSEC` database refresh after that; there is no
# `--offline` because the whole point is news arriving without a push.

# `cargo deny check advisories bans licenses sources`.
audit:
    cargo deny check advisories bans licenses sources

# The `crate layering` job: core must stay free of async runtimes and HTTP clients.
layering:
    if grep -E '^(tokio|reqwest|hyper|immortality-daemon|immortality-tools)' crates/core/Cargo.toml; then \
        echo "core must stay free of async/HTTP dependencies"; exit 1; \
    fi

# Diagnostics against the configured backend. Not part of `ci`: it needs the network.
doctor:
    cargo run -q -p immortality-cli -- doctor

build:
    cargo build --release

# Format, lint and test in one go, fixing what can be fixed automatically.
fix:
    cargo fmt --all
    cargo clippy --workspace --all-targets --all-features --fix --allow-dirty -- -D warnings
    cargo test --workspace --all-features
