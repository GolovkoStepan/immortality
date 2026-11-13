//! Turning `git diff` output into the shape the gates reason about.
//!
//! # Why parse instead of asking git three times
//!
//! The reviewer needs the rendered diff, the sanity check needs the added lines and the binary
//! flags, and the journal needs the file list. Three git invocations would be three chances to
//! describe the same change differently — and the one that disagreed would be the one nobody
//! tested. One parse, three views.
//!
//! # What this deliberately does not parse
//!
//! Not a full unified-diff implementation. It reads the headers git actually emits
//! (`diff --git`, `new file mode`, `deleted file mode`, `rename from`, `Binary files`, and the
//! `+++`/`---` pair), and everything inside a hunk it keeps verbatim. A file whose diff does not
//! parse is not dropped: it is kept with an empty body and the raw text still reaches the reviewer
//! through [`Diff::text`], so an unfamiliar git version degrades to "reviewed, not scanned"
//! rather than "silently ignored".

use std::path::PathBuf;

use super::{ChangedFile, Diff, FileStatus, MAX_REVIEW_BYTES};

/// Parse the output of `git diff --no-color`.
///
/// `truncated` says the caller already dropped part of the diff before passing it in; it is
/// carried through so the sanity check can refuse to call a half-read change approved.
#[must_use]
pub fn parse(text: &str, truncated: bool) -> Diff {
    let mut files: Vec<ChangedFile> = Vec::new();
    // The headers of the file currently being parsed.
    let mut current: Option<Builder> = None;

    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("diff --git ") {
            if let Some(builder) = current.take() {
                files.push(builder.finish());
            }
            current = Some(Builder::new(rest));
            continue;
        }
        let Some(builder) = current.as_mut() else {
            continue;
        };
        builder.absorb(line);
    }
    if let Some(builder) = current.take() {
        files.push(builder.finish());
    }

    Diff {
        files,
        text: text.to_string(),
        truncated: truncated || text.len() > MAX_REVIEW_BYTES,
    }
}

/// The file being accumulated.
#[derive(Debug)]
struct Builder {
    /// The `b/` side of the `diff --git a/x b/y` line, unquoted and unprefixed.
    ///
    /// Always the post-change path: a rename's new name, an addition's name, a modification's
    /// only name. The `a/` side is needed by git to find the old blob and by nothing here.
    head: PathBuf,
    status: Option<FileStatus>,
    binary: bool,
    added: Vec<String>,
}

impl Builder {
    fn new(header: &str) -> Self {
        Self {
            // The fallback is the raw header, which is only reached when the line is not in git's
            // own shape; keeping it visible beats dropping the file.
            head: paths_of(header).map_or_else(|| unquote(header), |(_, after)| after),
            status: None,
            binary: false,
            added: Vec::new(),
        }
    }

    /// Take one line that is not a `diff --git` header.
    fn absorb(&mut self, line: &str) {
        if line.starts_with("new file mode") {
            self.status = Some(FileStatus::Added);
            return;
        }
        if line.starts_with("deleted file mode") {
            self.status = Some(FileStatus::Deleted);
            return;
        }
        if line.starts_with("rename from ") {
            self.status = Some(FileStatus::Renamed);
            return;
        }
        if line.starts_with("rename to ") || line.starts_with("similarity index") {
            if self.status.is_none() {
                self.status = Some(FileStatus::Renamed);
            }
            return;
        }
        if line.starts_with("Binary files ") || line == "GIT binary patch" {
            // `git diff` says "Binary files a/x and b/x differ"; a repository that sets
            // `diff.git` emits a patch form. Both mean the same thing to the gate.
            self.binary = true;
            return;
        }
        if line.starts_with("--- ") || line.starts_with("+++ ") || line.starts_with("index ") {
            return;
        }
        if let Some(added) = line.strip_prefix('+') {
            self.added.push(added.to_string());
        }
    }

    fn finish(self) -> ChangedFile {
        ChangedFile {
            path: self.head,
            // A file whose mode line git omitted is modified: every other status is announced by
            // its own header, so "no header" is the default, not a guess.
            status: self.status.unwrap_or(FileStatus::Modified),
            binary: self.binary,
            added_lines: self.added,
        }
    }
}

/// The two paths on a `diff --git` line.
///
/// The line is ambiguous when a path contains a space — `a/x y b/x y` could split three ways — so
/// the prefix is what disambiguates: the first ` b/` after an `a/` starts the second path. Git
/// quotes a path with awkward characters, and the quotes are stripped here so the path matches
/// what `git status` reports.
fn paths_of(header: &str) -> Option<(PathBuf, PathBuf)> {
    let header = header.trim();
    // A quoted path is git's answer to awkward characters, and it quotes the *whole* path,
    // prefix included: `"a/x y" "b/x y"`. Splitting that on ` b/` would find nothing, so the
    // quoting is handled first and only the plain form falls through.
    if header.starts_with('"') {
        let (first, rest) = quoted(header)?;
        let second = rest.trim_start();
        let (second, _) = quoted(second).or_else(|| plain_tail(second))?;
        return Some((strip_prefix(&first, "a/"), strip_prefix(&second, "b/")));
    }
    let rest = header.strip_prefix("a/")?;
    // The second path begins at the last ` b/` that leaves both sides non-empty. Git writes the
    // same path twice for a plain modification, so the common case is unambiguous either way.
    let mut split = None;
    let mut from = 0;
    while let Some(offset) = rest[from..].find(" b/") {
        let at = from + offset;
        if at > 0 && at + 3 < rest.len() {
            split = Some(at);
        }
        from = at + 1;
    }
    let at = split?;
    Some((unquote(&rest[..at]), unquote(&rest[at + 3..])))
}

/// The first quoted token and what follows it.
fn quoted(text: &str) -> Option<(String, &str)> {
    let inner = text.strip_prefix('"')?;
    let mut chars = inner.char_indices();
    while let Some((index, ch)) = chars.next() {
        match ch {
            '\\' => {
                chars.next();
            }
            '"' => {
                let (token, rest) = inner.split_at(index);
                return Some((unescape(token), &rest[1..]));
            }
            _ => {}
        }
    }
    None
}

/// A trailing unquoted path, for the mixed form git does not normally emit.
fn plain_tail(text: &str) -> Option<(String, &str)> {
    let text = text.trim_start();
    let token = text.split_whitespace().next().filter(|token| !token.is_empty())?;
    Some((token.to_string(), &text[token.len()..]))
}

/// Drop the `a/` or `b/` side prefix, if it is there.
fn strip_prefix(text: &str, prefix: &str) -> PathBuf {
    PathBuf::from(text.strip_prefix(prefix).unwrap_or(text))
}

/// Strip git's surrounding quotes from a path, if it has any.
///
/// Git's quoted form escapes with C-style sequences; the ones that appear in a real path are the
/// quote and the backslash, and a partial unescape would produce a path that matches nothing.
fn unquote(text: &str) -> PathBuf {
    let text = text.trim();
    let Some(inner) = text.strip_prefix('"').and_then(|rest| rest.strip_suffix('"')) else {
        return PathBuf::from(text);
    };
    PathBuf::from(unescape(inner))
}

/// C-style unescape of a quoted path body.
fn unescape(inner: &str) -> String {
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some(other) => out.push(other),
            None => out.push('\\'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODIFIED: &str = "\
diff --git a/src/main.rs b/src/main.rs
index 1a2b3c4..5d6e7f8 100644
--- a/src/main.rs
+++ b/src/main.rs
@@ -1,3 +1,4 @@
 fn main() {
-    println!(\"hi\");
+    println!(\"hello\");
 }
";

    #[test]
    fn a_modified_file_yields_its_added_lines() {
        let diff = parse(MODIFIED, false);
        assert_eq!(diff.files.len(), 1);
        let file = &diff.files[0];
        assert_eq!(file.path, PathBuf::from("src/main.rs"));
        assert_eq!(file.status, FileStatus::Modified);
        assert!(!file.binary);
        assert_eq!(file.added_lines, vec![r#"    println!("hello");"#]);
    }

    #[test]
    fn the_context_lines_are_not_added_lines() {
        // The hunk header and the unchanged lines both start with a space; only `+` is an addition.
        assert_eq!(parse(MODIFIED, false).files[0].added_lines.len(), 1);
    }

    #[test]
    fn an_added_file_is_added_and_a_deleted_one_is_deleted() {
        let text = "\
diff --git a/src/new.rs b/src/new.rs
new file mode 100644
--- /dev/null
+++ b/src/new.rs
@@ -0,0 +1 @@
+let x = 1;
diff --git a/src/gone.rs b/src/gone.rs
deleted file mode 100644
--- a/src/gone.rs
+++ /dev/null
@@ -1 +0,0 @@
-let y = 2;
";
        let diff = parse(text, false);
        assert_eq!(diff.files[0].status, FileStatus::Added);
        assert_eq!(diff.files[0].added_lines, vec!["let x = 1;"]);
        assert_eq!(diff.files[1].status, FileStatus::Deleted);
        assert_eq!(diff.files[1].added_lines, Vec::<String>::new());
    }

    #[test]
    fn a_binary_file_is_marked_binary_and_keeps_no_lines() {
        let text = "\
diff --git a/assets/logo.png b/assets/logo.png
new file mode 100644
Binary files /dev/null and b/assets/logo.png differ
";
        let diff = parse(text, false);
        assert!(diff.files[0].binary);
        assert_eq!(diff.files[0].status, FileStatus::Added);
        assert_eq!(diff.files[0].added_lines, Vec::<String>::new());
    }

    #[test]
    fn a_rename_keeps_the_new_path() {
        let text = "\
diff --git a/src/old.rs b/src/new.rs
similarity index 100%
rename from src/old.rs
rename to src/new.rs
";
        let diff = parse(text, false);
        assert_eq!(diff.files[0].status, FileStatus::Renamed);
        assert_eq!(diff.files[0].path, PathBuf::from("src/new.rs"));
    }

    #[test]
    fn two_files_are_two_entries() {
        let text = format!("{MODIFIED}{MODIFIED}");
        assert_eq!(parse(&text, false).files.len(), 2);
    }

    #[test]
    fn a_path_with_a_space_parses_as_one_path() {
        let text = "\
diff --git a/src/my file.rs b/src/my file.rs
--- a/src/my file.rs
+++ b/src/my file.rs
@@ -1 +1 @@
-a
+b
";
        let diff = parse(text, false);
        assert_eq!(diff.files[0].path, PathBuf::from("src/my file.rs"));
    }

    #[test]
    fn a_quoted_path_is_unquoted() {
        let text = "\
diff --git \"a/src/tab\\there.rs\" \"b/src/tab\\there.rs\"
--- \"a/src/tab\\there.rs\"
+++ \"b/src/tab\\there.rs\"
@@ -1 +1 @@
-a
+b
";
        let diff = parse(text, false);
        assert_eq!(diff.files[0].path, PathBuf::from("src/tab\there.rs"));
    }

    #[test]
    fn an_unfamiliar_header_does_not_lose_the_file() {
        // A future git emits a header this parser does not know. The file still reaches the
        // reviewer through `text`, and it is listed as modified rather than dropped.
        let text = "\
diff --git a/src/x.rs b/src/x.rs
some future header
--- a/src/x.rs
+++ b/src/x.rs
@@ -1 +1 @@
-a
+b
";
        let diff = parse(text, false);
        assert_eq!(diff.files.len(), 1);
        assert_eq!(diff.files[0].status, FileStatus::Modified);
        assert_eq!(diff.files[0].added_lines, vec!["b"]);
    }

    #[test]
    fn the_diff_text_is_kept_verbatim_for_the_reviewer() {
        let diff = parse(MODIFIED, false);
        assert_eq!(diff.text, MODIFIED);
        assert!(!diff.truncated);
    }

    #[test]
    fn a_caller_supplied_truncation_is_carried_through() {
        assert!(parse(MODIFIED, true).truncated);
    }

    #[test]
    fn an_oversized_diff_is_truncated_even_if_the_caller_says_otherwise() {
        // The caller that forgot to cap is exactly the one this guards.
        let text = format!("{MODIFIED}{}", "x".repeat(MAX_REVIEW_BYTES));
        assert!(parse(&text, false).truncated);
    }

    #[test]
    fn an_empty_diff_has_no_files() {
        let diff = parse("", false);
        assert!(diff.is_empty());
        assert_eq!(diff.paths(), Vec::<PathBuf>::new());
    }
}
