# `just ci` runs exactly what GitHub Actions runs.
#
# The point is that a red CI run should never be the first time a problem is seen. Every
# check below is the same command the workflow invokes; if the two drift, CI is decoration.

set shell := ["bash", "-euo", "pipefail", "-c"]

default: ci

# Everything CI checks, in the order that fails fastest.
ci: fmt-check lint test

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all --check

# `-D warnings` matches the workspace lint config, so local and CI agree.
lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
    cargo test --workspace --all-features

# Full local gate including dependency and licence policy.
audit:
    cargo deny check advisories bans licenses sources

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
