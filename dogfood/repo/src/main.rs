//! Count words, lines and unique words in a file.
//!
//! Deliberately small and dependency-free: the point is that the agent can build, test and
//! change it with no network, so every gate in a dogfood run runs in seconds.

use std::process::ExitCode;

use tally::count;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: tally <file>");
        return ExitCode::FAILURE;
    };
    if args.next().is_some() {
        eprintln!("usage: tally <file>");
        return ExitCode::FAILURE;
    }
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("tally: cannot read {path}: {error}");
            return ExitCode::FAILURE;
        }
    };

    println!("lines: {}", count::lines(&text));
    println!("words: {}", count::words(&text));
    println!("unique: {}", count::unique_words(&text));
    ExitCode::SUCCESS
}
