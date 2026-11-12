//! Logging setup: human-readable on stderr, JSON lines in monthly files.
//!
//! Logs are operational noise, not the audit trail — that is `JOURNAL.jsonl` in the
//! repository. Consequently logs rotate freely: the current month is plain, previous months
//! are compressed to `YYYY-MM.log.gz`.

use chrono::Datelike as _;
use std::path::{Path, PathBuf};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, fmt};

use crate::error::{Error, Result};

/// Log file name for a given year and month: `2026-11.log`.
///
/// The archive form is the same name with `.gz` appended.
pub fn log_file_name(year: i32, month: u32) -> String {
    format!("{year:04}-{month:02}.log")
}

/// Path of the active log file for the current month.
pub fn current_log_path(dir: &Path) -> PathBuf {
    let now = chrono::Local::now();
    dir.join(log_file_name(now.year(), now.month()))
}

/// Plain `.log` files that belong to a month other than `now` and should be compressed.
pub fn stale_logs(dir: &Path, now: chrono::DateTime<chrono::Local>) -> Vec<PathBuf> {
    stale_logs_named(dir, &log_file_name(now.year(), now.month()))
}

fn stale_logs_named(dir: &Path, current: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().is_some_and(|ext| ext == "log")
                && path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n != current)
                && looks_like_month_file(path)
        })
        .collect();
    out.sort();
    out
}

/// `2026-10.log` matches; `immortality.log` does not, so an operator's own files are safe.
fn looks_like_month_file(path: &Path) -> bool {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return false;
    };
    let Some((year, month)) = stem.split_once('-') else {
        return false;
    };
    year.len() == 4
        && month.len() == 2
        && year.chars().all(|c| c.is_ascii_digit())
        && month.chars().all(|c| c.is_ascii_digit())
        && month.parse::<u32>().is_ok_and(|m| (1..=12).contains(&m))
}

/// Install the global subscriber.
///
/// Must be called at most once per process. A second call returns an error instead of
/// panicking so that `doctor` and tests can share setup code without special-casing.
pub fn init(level: &str, dir: &Path) -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(level))
        .map_err(|e| Error::Logging(format!("invalid log level `{level}`: {e}")))?;

    std::fs::create_dir_all(dir).map_err(|source| Error::io(dir, source))?;

    let stderr_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_target(false)
        .compact();

    // The file layer is wired to the monthly file of the current month. Rotation into
    // `.gz` archives is `gc`'s job, not the logger's: a logger that rotates files while
    // writing to them is a source of silent data loss.
    let file_path = current_log_path(dir);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .map_err(|source| Error::io(&file_path, source))?;
    let file_layer = fmt::layer()
        .json()
        .with_writer(std::sync::Mutex::new(file))
        .with_ansi(false);

    tracing_subscriber::registry()
        .with(filter)
        .with(stderr_layer)
        .with(file_layer)
        .try_init()
        .map_err(|e| Error::Logging(format!("subscriber already installed: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone as _;

    fn date(year: i32, month: u32) -> chrono::DateTime<chrono::Local> {
        chrono::Local.with_ymd_and_hms(year, month, 15, 12, 0, 0).unwrap()
    }

    #[test]
    fn log_names_are_zero_padded() {
        assert_eq!(log_file_name(2026, 1), "2026-01.log");
        assert_eq!(log_file_name(2026, 11), "2026-11.log");
    }

    #[test]
    fn stale_logs_exclude_current_month() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "2026-10.log",
            "2026-11.log",
            "2026-09.log.gz",
            "notes.txt",
            "immortality.log",
        ] {
            std::fs::write(dir.path().join(name), "").unwrap();
        }
        let stale: Vec<String> = stale_logs(dir.path(), date(2026, 11))
            .into_iter()
            .filter_map(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(std::string::ToString::to_string)
            })
            .collect();
        assert_eq!(
            stale,
            vec!["2026-10.log"],
            "only plain month files of other months rotate"
        );
    }

    #[test]
    fn stale_logs_on_missing_dir_is_empty() {
        assert_eq!(
            stale_logs(Path::new("/nonexistent/immortality/logs"), date(2026, 11)),
            Vec::<PathBuf>::new()
        );
    }

    #[test]
    fn current_log_path_uses_today() {
        let path = current_log_path(Path::new("/tmp/logs"));
        let now = chrono::Local::now();
        assert!(path.ends_with(log_file_name(now.year(), now.month())));
    }

    #[test]
    fn month_file_detection_rejects_junk() {
        assert!(looks_like_month_file(Path::new("2026-10.log")));
        assert!(looks_like_month_file(Path::new("/logs/2026-12.log")));
        assert!(!looks_like_month_file(Path::new("immortality.log")));
        assert!(!looks_like_month_file(Path::new("2026-13.log")));
        assert!(!looks_like_month_file(Path::new("2026-1.log")));
    }
}
