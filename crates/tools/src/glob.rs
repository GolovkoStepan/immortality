//! Glob matching for the policy's allow/deny lists.
//!
//! # Why not a `glob` crate
//!
//! The patterns come from a config file the human writes, and the question they answer is
//! security-shaped: "may this write land here?" A matcher that quietly does nothing is worse
//! than no matcher, because the deny list is what stops an agent from rewriting `.git/config`.
//! So [`Glob::parse`] rejects syntax it does not implement instead of matching nothing against
//! it — a `{a,b}` brace or a backslash escape is a startup error, not a silent hole.
//!
//! # Why matching is case-insensitive
//!
//! APFS on macOS is case-insensitive by default. A deny glob of `**/.git/config` that matched
//! case-sensitively would let `.GIT/config` through on the same machine that would happily open
//! it. Over-denying a differently-cased path costs nothing; under-denying costs the repository.
//!
//! # Supported syntax
//!
//! `*` matches any run of characters within one path segment, `**` matches across segments,
//! `?` matches one character, and `[abc]` / `[a-z]` / `[!a-z]` match a character class. `\`,
//! `{` and `}` are rejected: each means something in other glob dialects, and answering them
//! with a literal comparison is how a deny list silently stops denying.

use std::fmt;

/// Why a pattern could not be compiled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobError {
    pub pattern: String,
    pub detail: String,
}

impl fmt::Display for GlobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}`: {}", self.pattern, self.detail)
    }
}

impl std::error::Error for GlobError {}

/// One compiled glob pattern.
///
/// Segments rather than a flat token list: `**` is the only construct that crosses a separator,
/// so the recursive match can treat a segment as an atom and never has to reason about a `*`
/// that swallowed a slash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Glob {
    pattern: String,
    segments: Vec<Segment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Segment {
    /// `**`: zero or more path segments.
    Any,
    /// A single segment, matched character by character.
    One(Vec<Token>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Literal(char),
    /// `?`
    AnyOne,
    /// `*`
    AnyRun,
    /// `[a-z]`, `[!a-z]`, lowercased.
    Class {
        ranges: Vec<(char, char)>,
        negated: bool,
    },
}

impl Glob {
    /// Compile a pattern.
    ///
    /// `**` is only meaningful as a whole segment: `a**b` is rejected because every other glob
    /// dialect reads it differently, and a rule whose meaning depends on which dialect the reader
    /// has in mind is not a security rule.
    ///
    /// # Errors
    /// A pattern containing syntax this module does not implement.
    pub fn parse(pattern: &str) -> Result<Self, GlobError> {
        let error = |detail: &str| GlobError {
            pattern: pattern.to_string(),
            detail: detail.to_string(),
        };

        if pattern.is_empty() {
            return Err(error("an empty pattern matches nothing"));
        }
        if pattern.contains('\\') {
            return Err(error(
                "backslash escapes are not supported; a `\\` in a pattern would match itself here",
            ));
        }
        if pattern.contains('{') || pattern.contains('}') {
            return Err(error("brace alternation is not supported"));
        }
        // An absolute pattern would silently never match a worktree-relative path, which is the
        // failure mode this whole module exists to prevent.
        if pattern.starts_with('/') {
            return Err(error(
                "patterns are relative to the worktree; a leading `/` would never match",
            ));
        }

        let mut segments = Vec::new();
        for raw in pattern.split('/') {
            if raw == "**" {
                // Collapse `**/**` — it is the same rule and the recursive matcher would do
                // twice the work for no extra meaning.
                if segments.last() != Some(&Segment::Any) {
                    segments.push(Segment::Any);
                }
                continue;
            }
            if raw.contains("**") {
                return Err(error(
                    "`**` must be a whole segment; `a**b` is ambiguous across glob dialects",
                ));
            }
            if raw.is_empty() {
                return Err(error("an empty path segment: did a `//` slip in?"));
            }
            segments.push(Segment::One(parse_segment(raw, &error)?));
        }

        Ok(Self {
            pattern: pattern.to_string(),
            segments,
        })
    }

    /// The pattern as written.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.pattern
    }

    /// Whether `path` matches. Case-insensitive; see the module docs.
    #[must_use]
    pub fn matches(&self, path: &str) -> bool {
        // Trailing slashes are noise from a directory listing, not a different path.
        let path = path.trim_end_matches('/');
        let parts: Vec<&str> = path.split('/').filter(|part| !part.is_empty()).collect();
        match_segments(&self.segments, &parts)
    }
}

impl fmt::Display for Glob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.pattern)
    }
}

fn parse_segment<F>(raw: &str, error: &F) -> Result<Vec<Token>, GlobError>
where
    F: Fn(&str) -> GlobError,
{
    let chars: Vec<char> = raw.chars().collect();
    let mut tokens = Vec::with_capacity(chars.len());
    let mut index = 0;

    while let Some(&ch) = chars.get(index) {
        index += 1;
        match ch {
            '*' => {
                // `**` was already excluded for this segment; a run of `*` is still one `*`.
                if tokens.last() != Some(&Token::AnyRun) {
                    tokens.push(Token::AnyRun);
                }
            }
            '?' => tokens.push(Token::AnyOne),
            '[' => {
                let (class, next) = parse_class(&chars, index, error)?;
                tokens.push(class);
                index = next;
            }
            ']' => return Err(error("a `]` without a matching `[`")),
            other => tokens.push(Token::Literal(other.to_ascii_lowercase())),
        }
    }

    Ok(tokens)
}

/// Parse a character class starting just after its `[`. Returns the class and the index of the
/// first character after its `]`.
fn parse_class<F>(chars: &[char], mut index: usize, error: &F) -> Result<(Token, usize), GlobError>
where
    F: Fn(&str) -> GlobError,
{
    let mut negated = false;
    if chars.get(index) == Some(&'!') || chars.get(index) == Some(&'^') {
        negated = true;
        index += 1;
    }

    let mut ranges: Vec<(char, char)> = Vec::new();
    loop {
        let Some(&low) = chars.get(index) else {
            return Err(error("unterminated character class"));
        };
        index += 1;
        if low == ']' {
            if ranges.is_empty() {
                return Err(error("an empty character class matches nothing"));
            }
            return Ok((Token::Class { ranges, negated }, index));
        }

        // `a-z`, but a trailing `-` before `]` is a literal hyphen.
        if chars.get(index) == Some(&'-') && chars.get(index + 1).is_some_and(|ch| *ch != ']') {
            let high = chars[index + 1];
            if high < low {
                return Err(error("a character range runs backwards, as in `[z-a]`"));
            }
            ranges.push((low.to_ascii_lowercase(), high.to_ascii_lowercase()));
            index += 2;
        } else {
            let letter = low.to_ascii_lowercase();
            ranges.push((letter, letter));
        }
    }
}

fn match_segments(pattern: &[Segment], path: &[&str]) -> bool {
    // Memoised on (segments left, components left). Without it, a pattern like
    // `**/a/**/a/**/a/**/a/**/b` is `components ^ stars` branches; with it the answer costs
    // `segments × components`, which is a few hundred cells. The key is the remaining lengths
    // because the recursion only ever slices from the front.
    let mut memo = Memo::new(pattern.len(), path.len());
    segments_at(pattern, path, &mut memo)
}

fn segments_at(pattern: &[Segment], path: &[&str], memo: &mut Memo) -> bool {
    if let Some(known) = memo.get(pattern.len(), path.len()) {
        return known;
    }
    let answer = match pattern.split_first() {
        // `**` at the end matches the prefix itself and everything under it, which is the
        // recursion below with an empty tail.
        Some((Segment::Any, rest)) => {
            (0..=path.len()).any(|skip| segments_at(rest, path.get(skip..).unwrap_or(&[]), memo))
        }
        Some((Segment::One(tokens), rest)) => match path.split_first() {
            Some((part, remaining)) => match_tokens(tokens, part) && segments_at(rest, remaining, memo),
            None => false,
        },
        None => path.is_empty(),
    };
    memo.put(pattern.len(), path.len(), answer);
    answer
}

/// A small boolean memo, keyed by two remaining lengths.
///
/// Deliberately tiny: the row count is the number of pattern segments and the column count the
/// number of path components, both of which a real path keeps in the dozens.
#[derive(Debug)]
struct Memo {
    width: usize,
    cells: Vec<Cell>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cell {
    Unknown,
    No,
    Yes,
}

impl Memo {
    fn new(rows: usize, columns: usize) -> Self {
        Self {
            width: columns + 1,
            cells: vec![Cell::Unknown; (rows + 1) * (columns + 1)],
        }
    }

    const fn index(&self, row: usize, column: usize) -> usize {
        row * self.width + column
    }

    fn get(&self, row: usize, column: usize) -> Option<bool> {
        match self.cells[self.index(row, column)] {
            Cell::Unknown => None,
            Cell::No => Some(false),
            Cell::Yes => Some(true),
        }
    }

    fn put(&mut self, row: usize, column: usize, value: bool) {
        let index = self.index(row, column);
        self.cells[index] = if value { Cell::Yes } else { Cell::No };
    }
}

fn match_tokens(tokens: &[Token], text: &str) -> bool {
    // Lowercased on both sides once, so the comparison inside the loop is one `char` compare
    // rather than a case-fold per candidate.
    let text: Vec<char> = text.chars().map(|ch| ch.to_ascii_lowercase()).collect();

    // Two pointers and one remembered star, not recursion. A recursive matcher tries every split
    // of every `*`, so `*a*a*a*a*a*a*a*a` against a long name is exponential — and the pattern
    // comes from the model, which means that is a hang an answer can cause. This form is
    // `tokens × characters` worst case and allocates nothing.
    let mut token = 0usize;
    let mut at = 0usize;
    // `(token after the star, character index to resume from)` — the last star seen, because a
    // later star can always absorb what an earlier one failed to.
    let mut star: Option<(usize, usize)> = None;

    while at < text.len() {
        let Some(current) = tokens.get(token) else {
            // Tokens exhausted but characters left: only a star can rescue this.
            match star {
                Some((resume_token, resume_at)) => {
                    token = resume_token;
                    at = resume_at + 1;
                    star = Some((resume_token, resume_at + 1));
                }
                None => return false,
            }
            continue;
        };

        if matches!(current, Token::AnyRun) {
            star = Some((token + 1, at));
            token += 1;
            continue;
        }
        if matches_one(current, text[at]) {
            token += 1;
            at += 1;
            continue;
        }
        match star {
            Some((resume_token, resume_at)) => {
                token = resume_token;
                at = resume_at + 1;
                star = Some((resume_token, resume_at + 1));
            }
            None => return false,
        }
    }

    // A trailing `*` consumes nothing and still matches, so skip the ones that remain.
    while matches!(tokens.get(token), Some(Token::AnyRun)) {
        token += 1;
    }
    token == tokens.len()
}

/// Whether one non-`*` token matches one character.
fn matches_one(token: &Token, ch: char) -> bool {
    match token {
        Token::Literal(expected) => *expected == ch,
        Token::AnyOne => true,
        Token::Class { ranges, negated } => {
            let inside = ranges.iter().any(|(low, high)| (*low..=*high).contains(&ch));
            inside != *negated
        }
        // The caller handles `*` itself, because it is the only token that can consume nothing.
        Token::AnyRun => false,
    }
}

/// A compiled list of patterns, all of which are candidates.
///
/// A newtype rather than a bare `Vec<Glob>` so that "does this path match any pattern?" and
/// "which pattern matched?" cannot be confused, and so the answer can name the rule that fired —
/// an observation saying only "denied" leaves the model guessing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PatternSet {
    globs: Vec<Glob>,
}

impl PatternSet {
    /// Compile every pattern.
    ///
    /// # Errors
    /// The first pattern that does not compile, with its position, so a config typo is named
    /// rather than counted.
    pub fn parse<S: AsRef<str>>(patterns: &[S]) -> Result<Self, GlobError> {
        let mut globs = Vec::with_capacity(patterns.len());
        for raw in patterns {
            globs.push(Glob::parse(raw.as_ref())?);
        }
        Ok(Self { globs })
    }

    /// Whether any pattern matches.
    #[must_use]
    pub fn matches(&self, path: &str) -> bool {
        self.globs.iter().any(|glob| glob.matches(path))
    }

    /// The first pattern that matches, for the observation's "because of X" clause.
    #[must_use]
    pub fn matching(&self, path: &str) -> Option<&Glob> {
        self.globs.iter().find(|glob| glob.matches(path))
    }

    /// Whether the set is empty, which lets a caller skip the check entirely.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.globs.is_empty()
    }

    /// How many patterns are compiled.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.globs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glob(pattern: &str) -> Glob {
        Glob::parse(pattern).expect("a valid pattern")
    }

    #[test]
    fn a_literal_pattern_matches_exactly() {
        assert!(glob("src/main.rs").matches("src/main.rs"));
        assert!(!glob("src/main.rs").matches("src/lib.rs"));
        assert!(!glob("src/main.rs").matches("src/main.rs/extra"));
    }

    #[test]
    fn a_star_stays_inside_one_segment() {
        let pattern = glob("*.rs");
        assert!(pattern.matches("main.rs"));
        assert!(!pattern.matches("src/main.rs"), "`*` must not cross a slash");
    }

    #[test]
    fn a_double_star_crosses_segments_and_matches_the_prefix_itself() {
        let pattern = glob("**/.env*");
        assert!(pattern.matches(".env"));
        assert!(pattern.matches("a/.env"));
        assert!(pattern.matches("a/b/.env.local"));
        assert!(!pattern.matches("a/b/env"));
    }

    #[test]
    fn a_trailing_double_star_matches_everything_below() {
        let pattern = glob("target/**");
        assert!(pattern.matches("target"));
        assert!(pattern.matches("target/debug/build/x"));
        assert!(!pattern.matches("targeted/x"));
    }

    #[test]
    fn a_leading_double_star_with_more_segments_still_works() {
        let pattern = glob("**/config.toml");
        assert!(pattern.matches("config.toml"));
        assert!(pattern.matches(".agent/config.toml"));
        assert!(pattern.matches("a/b/c/config.toml"));
        assert!(!pattern.matches("a/b/config.toml.bak"));
    }

    #[test]
    fn question_matches_exactly_one_character() {
        let pattern = glob("a?c");
        assert!(pattern.matches("abc"));
        assert!(!pattern.matches("ac"));
        assert!(!pattern.matches("abbc"));
    }

    #[test]
    fn a_character_class_matches_a_set_or_a_range() {
        assert!(glob("file[123].txt").matches("file2.txt"));
        assert!(!glob("file[123].txt").matches("file4.txt"));
        assert!(glob("[a-c]x").matches("bx"));
        assert!(!glob("[a-c]x").matches("dx"));
    }

    #[test]
    fn a_negated_class_inverts_the_match() {
        let pattern = glob("file[!0-9].txt");
        assert!(pattern.matches("filea.txt"));
        assert!(!pattern.matches("file1.txt"));
    }

    #[test]
    fn matching_ignores_case_because_apfs_does() {
        // The bug this prevents: `**/.git/config` not matching `.GIT/config` on a filesystem
        // that would happily open it.
        assert!(glob("**/.git/config").matches(".GIT/config"));
        assert!(glob("**/.env*").matches("a/.ENV.local"));
        assert!(glob("[A-C]x").matches("ax"));
    }

    #[test]
    fn a_path_with_trailing_or_doubled_slashes_still_matches() {
        assert!(glob("src/*.rs").matches("src/main.rs/"));
        assert!(glob("src/*.rs").matches("src//main.rs"));
    }

    #[test]
    fn repeated_double_stars_collapse() {
        let pattern = glob("**/**/x");
        assert!(pattern.matches("x"));
        assert!(pattern.matches("a/b/x"));
    }

    #[test]
    fn unsupported_syntax_is_rejected_rather_than_ignored() {
        // Each of these means something in some glob dialect. Matching them literally would
        // make a deny rule silently fail to deny.
        assert!(Glob::parse("**/{a,b}").is_err());
        assert!(Glob::parse("a\\*").is_err());
        assert!(Glob::parse("a**b").is_err());
        assert!(Glob::parse("").is_err());
        assert!(Glob::parse("/abs/path").is_err());
        assert!(Glob::parse("a//b").is_err());
    }

    #[test]
    fn a_malformed_class_is_rejected() {
        assert!(Glob::parse("[abc").is_err(), "unterminated");
        assert!(Glob::parse("[]").is_err(), "empty");
        assert!(Glob::parse("[z-a]").is_err(), "backwards range");
        assert!(Glob::parse("a]b").is_err(), "stray close bracket");
    }

    #[test]
    fn the_error_names_the_pattern_and_the_problem() {
        let error = Glob::parse("**/{a,b}").unwrap_err();
        assert_eq!(error.pattern, "**/{a,b}");
        assert!(error.to_string().contains("brace"), "{error}");
    }

    #[test]
    fn a_pattern_set_names_the_rule_that_fired() {
        let set = PatternSet::parse(&["**/.git/config", "**/.env*"]).unwrap();
        assert!(set.matches(".env"));
        assert_eq!(set.matching(".env").unwrap().as_str(), "**/.env*");
        assert!(!set.matches("src/main.rs"));
        assert_eq!(set.len(), 2);
        assert!(!set.is_empty());
    }

    #[test]
    fn an_empty_set_matches_nothing() {
        let set = PatternSet::parse::<&str>(&[]).unwrap();
        assert!(set.is_empty());
        assert!(!set.matches("anything"));
    }

    /// The allow list from the design: everything inside the worktree.
    #[test]
    fn the_designs_allow_pattern_covers_ordinary_paths() {
        let allow = PatternSet::parse(&["**"]).unwrap();
        for path in ["a", "a/b/c.rs", ".agent/PLAN.md", "Cargo.toml"] {
            assert!(allow.matches(path), "{path}");
        }
    }

    /// The deny list from the design, checked against the paths it exists to stop.
    #[test]
    fn pathological_star_patterns_finish_in_linear_time() {
        // The bug this prevents is a hang, not a wrong answer: a backtracking matcher tries every
        // split of every `*`, so this pair is 2^40 branches. A test that only checked the answer
        // would pass slowly on a short name and never finish on a real one, so the name is long.
        let name = "a".repeat(400);
        assert!(!glob("*a*a*a*a*a*a*a*a*a*a*b").matches(&name));
        let pattern = glob("*a*a*a*a*a*a*a*a*a*a");
        assert!(pattern.matches(&name), "ten stars must still match a run of `a`");
        assert!(!pattern.matches("bbb"));
    }

    #[test]
    fn many_double_stars_do_not_multiply() {
        // Each `**` is a choice, so a naive recursive segment matcher is `components^stars`.
        let path = (0..40).map(|n| format!("d{n}")).collect::<Vec<_>>().join("/");
        let pattern = glob("**/a/**/a/**/a/**/a/**/a/**/a/**/z");
        assert!(!pattern.matches(&path));
        assert!(glob("**/d39").matches(&path), "trailing `**` must still work");
    }

    #[test]
    fn a_trailing_star_matches_the_prefix_and_the_whole_name() {
        // Regression guard for the non-recursive matcher: a `*` at the end consumes nothing and
        // still has to count as matched.
        assert!(glob("*.rs").matches("a.rs"));
        assert!(glob("*").matches("anything"));
        assert!(glob("a*").matches("a"));
        assert!(Glob::parse("a**b").is_err(), "`a**b` is still rejected");
        assert!(glob("ab*").matches("ab"));
    }

    #[test]
    fn the_designs_deny_patterns_stop_secret_and_git_files() {
        let deny = PatternSet::parse(&["**/.git/config", "**/.env*"]).unwrap();
        for path in [".git/config", ".env", "sub/.env.local", ".env.production"] {
            assert!(deny.matches(path), "{path}");
        }
        for path in [".gitignore", "src/env.rs", "docs/env.md"] {
            assert!(!deny.matches(path), "{path}");
        }
        // `.envrc` *is* matched: `.env*` is a prefix rule and direnv's file is exactly the file
        // that holds secrets. Denying more than intended costs one escalation; denying less
        // ships the secret.
        assert!(deny.matches(".envrc.example.bak"));
    }
}

#[cfg(test)]
mod differential {
    //! The non-recursive matcher against a reference one.
    //!
    //! The two-pointer loop replaced a recursion that was obviously correct but exponential. That
    //! trade is only safe if the fast version answers every input the same way, and no hand-picked
    //! example can establish that. So the recursion lives on here, in the test module, small and
    //! obviously correct, and `proptest` compares them over generated patterns and names.
    //!
    //! The reference is exponential and that is fine: the generator keeps inputs short, and a
    //! reference that also had to be clever would be a second implementation to be wrong in.

    use super::*;
    use proptest::prelude::*;

    /// Reference matcher: try every split of every `*`. Exponential, obviously correct.
    fn reference(tokens: &[Token], text: &[char]) -> bool {
        match tokens.split_first() {
            None => text.is_empty(),
            Some((Token::AnyRun, rest)) => {
                (0..=text.len()).any(|take| reference(rest, text.get(take..).unwrap_or(&[])))
            }
            Some((Token::AnyOne, rest)) => !text.is_empty() && reference(rest, text.get(1..).unwrap_or(&[])),
            Some((Token::Class { ranges, negated }, rest)) => {
                let Some((&first, remaining)) = text.split_first() else {
                    return false;
                };
                let inside = ranges.iter().any(|(low, high)| (*low..=*high).contains(&first));
                (inside != *negated) && reference(rest, remaining)
            }
            Some((Token::Literal(expected), rest)) => match text.split_first() {
                Some((&first, remaining)) if first == *expected => reference(rest, remaining),
                _ => false,
            },
        }
    }

    /// A pattern segment built from the tokens a model actually writes, with repeated `*` runs
    /// over-represented because that is where the two algorithms could diverge.
    ///
    /// Adjacent stars are filtered out: `**` inside a segment is a deliberate parse error, so
    /// generating it would test the parser's rejection rather than the matcher's agreement.
    fn segment() -> impl Strategy<Value = String> {
        proptest::collection::vec(
            prop_oneof![
                Just("*".to_string()),
                Just("?".to_string()),
                Just("[ab]".to_string()),
                Just("[!a]".to_string()),
                Just("a".to_string()),
                Just("b".to_string()),
                Just("c".to_string()),
            ],
            1..6,
        )
        .prop_map(|parts| parts.concat())
        .prop_filter("the parser must accept it as one ordinary segment", |text| {
            Glob::parse(text).is_ok()
        })
    }

    fn name() -> impl Strategy<Value = String> {
        proptest::collection::vec(prop_oneof![Just('a'), Just('b'), Just('c')], 0..10)
            .prop_map(|chars| chars.into_iter().collect())
    }

    proptest! {
        #[test]
        fn the_fast_matcher_agrees_with_the_reference(pattern in segment(), text in name()) {
            let compiled = Glob::parse(&pattern).expect("the generator filters to valid patterns");
            let tokens = compiled.segments.first().and_then(|segment| match segment {
                Segment::One(tokens) => Some(tokens.clone()),
                Segment::Any => None,
            });
            let Some(tokens) = tokens else {
                // A bare `**` is handled by the segment matcher, not the token matcher.
                return Ok(());
            };
            let chars: Vec<char> = text.chars().map(|ch| ch.to_ascii_lowercase()).collect();
            prop_assert_eq!(
                match_tokens(&tokens, &text),
                reference(&tokens, &chars),
                "pattern `{}` against `{}`",
                pattern,
                text
            );
        }
    }
}
