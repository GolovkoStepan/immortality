//! `doctor` against a fake backend and against a dead one.
//!
//! These are integration tests because the interesting behaviour is the HTTP interaction:
//! which status codes count as reachable, whether a failure skips the expensive latency
//! probe, and what exit code results. The unit tests in `doctor.rs` cover the pure parts.
//!
//! The mock is hand-rolled on purpose. A framework would test the framework's HTTP stack;
//! what needs testing is that `doctor` copes with a plain, occasionally wrong, server.

// Integration tests: `expect` and `unwrap` are the right tool here — a failed assertion
// should name the exact line, and there is no caller to propagate to.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::net::SocketAddr;
use std::path::PathBuf;

use immortality_core::config::Config;
use immortality_daemon::doctor::{self, Status};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// What the mock should do for the next requests.
#[derive(Debug, Clone, Copy)]
enum Mode {
    /// Serve a valid model list and two chat completions with cache counters.
    Healthy,
    /// Answer `/models` with 500.
    ServerError,
    /// Accept the connection and close it without a response.
    HangUp,
}

/// Minimal HTTP/1.1 server: one request per connection, `Connection: close`.
///
/// Binds port 0 so tests can run in parallel without coordinating ports.
async fn spawn_mock(mode: Mode) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let handle = tokio::spawn(async move {
        // Three requests at most: /models, then two completions. Extra connections are
        // accepted and closed so a retry never hangs the test.
        for _ in 0..4 {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let mut buffer = vec![0u8; 8192];
            let Ok(read) = stream.read(&mut buffer).await else {
                continue;
            };
            let request = String::from_utf8_lossy(&buffer[..read]).to_string();
            let path = request
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or("");

            let body = match mode {
                Mode::HangUp => continue,
                Mode::ServerError => {
                    respond(&mut stream, 500, "application/json", r#"{"error":"boom"}"#).await;
                    continue;
                }
                Mode::Healthy if path.ends_with("/models") => {
                    r#"{"object":"list","data":[{"id":"swift-iq2_xs","object":"model"}]}"#.to_string()
                }
                Mode::Healthy => {
                    // Second identical call is served from cache: 250 of 260 tokens.
                    r#"{
                        "choices":[{"index":0,"message":{"role":"assistant","content":"ready"}}],
                        "usage":{"prompt_tokens":260,"completion_tokens":2,"total_tokens":262},
                        "timings":{"cache_n":250}
                    }"#
                    .to_string()
                }
            };
            respond(&mut stream, 200, "application/json", &body).await;
        }
    });
    (addr, handle)
}

async fn respond(stream: &mut tokio::net::TcpStream, code: u16, content_type: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {code} OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes()).await;
    let _ = stream.flush().await;
}

fn config_for(addr: SocketAddr) -> Config {
    let mut config = Config::default();
    config.backend.base_url = format!("http://{addr}/v1");
    config.backend.connect_timeout_secs = 5;
    config.policy.trusted_hosts.insert("127.0.0.1".into());
    config.logging.dir = PathBuf::from("/tmp/immortality-doctor-test-logs");
    config
}

async fn run(config: &Config) -> Vec<doctor::Check> {
    let warnings = Vec::new();
    let global = PathBuf::from("/tmp/immortality-doctor-test.toml");
    let context = doctor::Context {
        config,
        config_warnings: &warnings,
        global_config_path: &global,
        socket_path: None,
    };
    doctor::run(&context).await.checks
}

fn find<'a>(checks: &'a [doctor::Check], name: &str) -> &'a doctor::Check {
    checks
        .iter()
        .find(|check| check.name == name)
        .unwrap_or_else(|| panic!("no `{name}` check in {checks:?}"))
}

#[tokio::test]
async fn healthy_backend_passes_every_backend_check() {
    let (addr, server) = spawn_mock(Mode::Healthy).await;
    let config = config_for(addr);
    let checks = run(&config).await;

    assert_eq!(find(&checks, "backend-models").status, Status::Ok);
    assert_eq!(find(&checks, "backend-model").status, Status::Ok);
    assert_eq!(find(&checks, "backend-latency").status, Status::Ok);
    let cache = find(&checks, "backend-cache");
    assert_eq!(cache.status, Status::Ok, "{}", cache.detail);
    assert!(cache.detail.contains("250/260"), "{}", cache.detail);

    server.abort();
}

#[tokio::test]
async fn server_error_fails_models_and_skips_latency() {
    let (addr, server) = spawn_mock(Mode::ServerError).await;
    let config = config_for(addr);
    let checks = run(&config).await;

    let models = find(&checks, "backend-models");
    assert_eq!(models.status, Status::Fail, "{}", models.detail);
    assert!(models.detail.contains("500"), "{}", models.detail);
    // The expensive probes must be skipped, not attempted: a broken backend should not
    // make `doctor` wait for two completion timeouts.
    assert_eq!(find(&checks, "backend-latency").status, Status::Skipped);
    assert_eq!(find(&checks, "backend-cache").status, Status::Skipped);

    server.abort();
}

#[tokio::test]
async fn connection_drop_is_a_failure_not_a_hang() {
    let (addr, server) = spawn_mock(Mode::HangUp).await;
    let config = config_for(addr);
    let checks = run(&config).await;

    assert_eq!(find(&checks, "backend-models").status, Status::Fail);
    assert_eq!(find(&checks, "backend-latency").status, Status::Skipped);

    server.abort();
}

#[tokio::test]
async fn unreachable_backend_reports_failure_and_finishes() {
    // Port 1 on loopback: reserved, nothing listens, connection refused immediately.
    let mut config = Config::default();
    config.backend.base_url = "http://127.0.0.1:1/v1".into();
    config.backend.connect_timeout_secs = 2;
    config.policy.trusted_hosts.insert("127.0.0.1".into());
    config.logging.dir = PathBuf::from("/tmp/immortality-doctor-test-logs");

    let checks = run(&config).await;
    assert_eq!(find(&checks, "backend-models").status, Status::Fail);
    assert_eq!(find(&checks, "backend-latency").status, Status::Skipped);
    assert_eq!(find(&checks, "backend-cache").status, Status::Skipped);
    // Local checks must still run: the point of `doctor` is the whole picture.
    assert_eq!(find(&checks, "trusted-host").status, Status::Ok);
    assert!(checks.iter().any(|check| check.name == "disk"));
}

#[tokio::test]
async fn untrusted_host_is_reported_even_when_backend_answers() {
    let (addr, server) = spawn_mock(Mode::Healthy).await;
    let mut config = config_for(addr);
    config.policy.trusted_hosts.remove("127.0.0.1");

    let checks = run(&config).await;
    assert_eq!(find(&checks, "trusted-host").status, Status::Fail);
    // A human-facing misconfiguration must not stop the diagnostic from checking the backend.
    assert_eq!(find(&checks, "backend-models").status, Status::Ok);

    server.abort();
}

#[tokio::test]
async fn exit_code_reflects_the_worst_check() {
    let (addr, server) = spawn_mock(Mode::Healthy).await;
    let config = config_for(addr);
    let warnings = Vec::new();
    let global = PathBuf::from("/tmp/immortality-doctor-test.toml");
    let report = doctor::run(&doctor::Context {
        config: &config,
        config_warnings: &warnings,
        global_config_path: &global,
        socket_path: None,
    })
    .await;
    assert_eq!(report.exit_code(), 0, "{:?}", report.checks);

    let mut broken = config.clone();
    broken.backend.base_url = "http://127.0.0.1:1/v1".into();
    let report = doctor::run(&doctor::Context {
        config: &broken,
        config_warnings: &warnings,
        global_config_path: &global,
        socket_path: None,
    })
    .await;
    assert_eq!(report.exit_code(), 1);

    server.abort();
}
