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

`systemd`-backend is the deliberate exception: `tests/service_systemd.rs` runs on **both**
platforms, against a scripted `systemctl`. It is not gated on `cfg(target_os = "linux")` for
the reason D-110 gives — a `cfg` on those tests would mean the sequence (write the unit,
create the enablement symlink, `daemon-reload`, `restart`) is only ever exercised after a
push. What genuinely needs Linux is `systemd` itself, and that is outside this repository's
test scope the same way a real `launchctl` is.

## The class of failure this catches

A test that shells out to a macOS-only binary. `plutil` (plist validation) and `launchctl`
both exist only on macOS; on Linux the test fails on `NotFound`, which says nothing about
the behaviour under test. Such tests belong under `#![cfg(target_os = "macos")]` — the
macOS job runs on every push, so the coverage stays where the toolchain does.

The second class is subtler and was found this way: a utility with the same name but
different argument parsing. `kill` is BSD `kill` on macOS and procps `kill` on Debian, and
they disagree about a leading `-` in the operand:

```sh
kill -0 -1234      # procps reads `-1234` as a signal spec, answers success, signals nothing
kill -0 -- -1234   # both read it as process group 1234
```

The code used the first form to probe and to kill process groups, so on Linux `group_alive`
answered "still running" for every finished command: each `shell` call waited out the
5-second grace period and told the model it had killed a background process that never
existed. On macOS the same tests were green. Any call that takes a negative argument needs
`--`, and the separator is portable — both implementations accept it.
