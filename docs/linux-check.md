# One-time full run in Linux, to reproduce the CI job that `just ci` cannot.

`just ci` runs on macOS and says nothing about the Linux job; the workflow does, but only
after a push. When a push goes red that the local gate cannot reproduce, this is the
container that does.

```sh
docker run --rm \
  -v "$PWD":/src:ro \
  -v immortality-cargo:/cargo \
  -e CARGO_HOME=/cargo \
  rust:1.99.1-bookworm \
  bash -c 'cp -R /src /work && cd /work && rm -rf target .git && \
           cargo test --workspace --all-features'
```

Three details are load-bearing, and getting any of them wrong produces failures that look
like defects in the code:

- **`/src` is read-only, and the run happens in a copy.** `cargo test` writes `target/` and
  the dogfood tests create git repositories under a temp dir; running in the mount keeps
  Linux artifacts from landing in the macOS checkout.
- **No shared `CARGO_TARGET_DIR`.** The dogfood tests admit three fixture crates
  (`tally`) into separate temp directories and run `cargo test` in each, in parallel. Point
  all of them at one target directory and they overwrite each other's build of a crate with
  the same name and version but different source: gates that must be red come back green,
  and which test fails changes from run to run. Leave the target directory inside the copy.
- **`--all-features`, no `--offline`.** That is what the workflow passes. `--offline`
  works here only because the local cache happens to be warm; CI resolves from the network.

The first run compiles the whole tree; later runs reuse the `immortality-cargo` volume.
Recreate it with `docker volume rm immortality-cargo` if a toolchain bump makes the cache
stale.

## The class of failure this catches

A test that shells out to a macOS-only binary. `plutil` (plist validation) and `launchctl`
both exist only on macOS; on Linux the test fails on `NotFound`, which says nothing about
the behaviour under test. Such tests belong under `#![cfg(target_os = "macos")]` — the
macOS job runs on every push, so the coverage stays where the toolchain does.
