# One-time full run in Linux, to reproduce the CI job that `just ci` cannot.

`just ci` runs on macOS and says nothing about the Linux job; the workflow does, but only
after a push. When a push goes red that the local gate cannot reproduce, this is the
container that does.

`CARGO_TARGET_DIR` is mounted separately on purpose: sharing `./target` writes Linux
artifacts over the macOS ones, and the next local `cargo` then fails in ways that have
nothing to do with the code.

```sh
docker run --rm \
  -v "$PWD":/w \
  -v immortality-cargo:/cargo \
  -v immortality-target:/target \
  -w /w \
  -e CARGO_HOME=/cargo \
  -e CARGO_TARGET_DIR=/target \
  rust:1.99.1-bookworm \
  cargo test --workspace --all-features
```

The first run compiles the whole tree (~10 min on an M-series machine); later runs reuse
`immortality-cargo` and `immortality-target`.

Recreate the volumes with `docker volume rm immortality-cargo immortality-target` if a
toolchain bump makes the caches stale.
