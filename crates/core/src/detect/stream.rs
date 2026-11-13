//! D4: a model repeating itself inside one message.
//!
//! # Why this cannot wait for the turn to finish
//!
//! Local models get stuck mid-sentence. The measured shape is: a paragraph, then the same
//! paragraph again, then a third time, and by then the turn has spent thousands of tokens on
//! nothing and the loop has to act on a message that is already written. The design's rule is to
//! cut the stream on the third occurrence and pay for a forced-strategy step instead — which is
//! what the streaming API's abort callback exists for.
//!
//! # Why n-grams of words and not of tokens
//!
//! Counting real tokens needs the model's tokenizer, which this process does not have: the
//! backend tokenizes. Dividing characters is the same approximation the budget uses, and for a
//! repetition detector the approximation is harmless — a phrase of forty "tokens" is roughly one
//! hundred and forty characters, and a paragraph that repeats will repeat at any of those sizes.
//! The threshold is therefore expressed in words and documented as such.
//!
//! # Why `SimHash` is not here
//!
//! The design lists a second signal: three paragraphs with `SimHash` similarity above 0.9. `SimHash`
//! needs a hash per shingle and a distance threshold, and both are tuned against real transcripts
//! this project does not have yet. Shipping an untuned similarity detector would either never
//! fire or fire on ordinary prose, and a detector that fires on ordinary prose trains the reader
//! to ignore `detector_fired`. The n-gram rule is exact, cheap, and enough to stop the measured
//! failure; the similarity rule is deferred with its reason written down (D-077).

use std::collections::HashMap;

use crate::config::Detectors;

/// Words kept for n-gram counting.
///
/// The design says "in the last 1200 tokens"; at roughly 1.4 tokens per word that is about 850
/// words, rounded to 1 200 so the window is generous rather than the detector blind. Each word is
/// hashed, so the memory is 8 bytes per word plus the n-gram index, not the text.
const WINDOW_WORDS: usize = 1_200;

/// What the scanner says about the stream so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamVerdict {
    /// Nothing yet; keep reading.
    Clear,
    /// The same n-gram has appeared often enough: stop generating.
    Looping {
        /// How many times the n-gram has appeared.
        occurrences: u32,
    },
}

impl StreamVerdict {
    /// Whether the client's abort callback must return `false`.
    #[must_use]
    pub const fn must_abort(self) -> bool {
        matches!(self, Self::Looping { .. })
    }
}

/// A rolling scanner over one message's text.
///
/// Fed one delta at a time by the streaming client. It keeps a ring of word hashes, so a model
/// that writes a hundred thousand words costs one fixed-size buffer and a hash per word.
#[derive(Debug)]
pub struct Stream {
    /// Hash of each word currently in the window, oldest first.
    window: std::collections::VecDeque<u64>,
    /// How many times each n-gram occurs in the window.
    grams: HashMap<u64, u32>,
    /// The n-gram length, in words.
    ngram: usize,
    /// How many occurrences abort the stream.
    limit: u32,
    /// The partial word left over from the previous delta.
    ///
    /// A delta boundary can split a word, and counting `hel` and `lo` as two words would make the
    /// n-grams sensitive to chunking — which is exactly what the detector must not be: the same
    /// message split differently by the transport must produce the same verdict.
    partial: String,
    /// The highest count reached so far, for the journal after an abort.
    peak: u32,
}

impl Stream {
    /// A scanner configured from the validated detector thresholds.
    #[must_use]
    pub fn new(detectors: &Detectors) -> Self {
        let ngram = usize::try_from(detectors.reasoning_repeat_ngram)
            .unwrap_or(usize::MAX)
            .max(1);
        Self {
            window: std::collections::VecDeque::with_capacity(WINDOW_WORDS),
            grams: HashMap::new(),
            ngram,
            limit: detectors.reasoning_repeat_count,
            partial: String::new(),
            peak: 0,
        }
    }

    /// Feed a delta; returns whether the stream must be aborted.
    pub fn push(&mut self, delta: &str) -> StreamVerdict {
        self.partial.push_str(delta);
        // Every complete word: the text up to the last whitespace. The tail after it stays in
        // `partial` because the next delta may continue it.
        let complete = match self.partial.rfind(char::is_whitespace) {
            Some(index) => {
                let tail = self.partial.split_off(index + 1);
                std::mem::replace(&mut self.partial, tail)
            }
            None => return StreamVerdict::Clear,
        };
        for word in complete.split_whitespace() {
            self.push_word(hash_word(word));
        }
        self.verdict()
    }

    /// The highest n-gram count seen in this message.
    #[must_use]
    pub const fn peak(&self) -> u32 {
        self.peak
    }

    /// How many words are in the window.
    #[must_use]
    pub fn window(&self) -> usize {
        self.window.len()
    }

    /// Add one word and count the n-gram that ends at it.
    fn push_word(&mut self, word: u64) {
        self.window.push_back(word);
        if self.window.len() > WINDOW_WORDS {
            self.window.pop_front();
        }
        if self.window.len() < self.ngram {
            return;
        }
        // Only the n-gram ending at the new word can change; every other one is unchanged.
        let start = self.window.len() - self.ngram;
        let hash = self.gram_hash(start);
        let count = self.grams.entry(hash).or_insert(0);
        *count = count.saturating_add(1);
        self.peak = self.peak.max(*count);
    }

    /// FNV-1a over the n-gram's word hashes.
    ///
    /// No extra dependency, deterministic, and the collision rate over a few thousand n-grams is
    /// not a concern for a threshold counter. Counts are never forgotten when the window slides,
    /// so a phrase repeated early stays counted: that makes the detector slightly more eager than
    /// a strict sliding window, which is the right error for one whose job is to stop a runaway.
    fn gram_hash(&self, start: usize) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for word in self.window.iter().skip(start) {
            for byte in word.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    const fn verdict(&self) -> StreamVerdict {
        if self.peak >= self.limit {
            StreamVerdict::Looping {
                occurrences: self.peak,
            }
        } else {
            StreamVerdict::Clear
        }
    }
}

/// FNV-1a of one word's bytes.
fn hash_word(word: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in word.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detectors(ngram: u32, count: u32) -> Detectors {
        Detectors {
            reasoning_repeat_ngram: ngram,
            reasoning_repeat_count: count,
            ..Detectors::default()
        }
    }

    /// A phrase of `words` distinct words, numbered so repetitions are unambiguous.
    fn phrase(words: usize, tag: &str) -> String {
        (0..words)
            .map(|index| format!("{tag}{index}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn a_phrase_repeated_the_configured_number_of_times_aborts_the_stream() {
        let mut stream = Stream::new(&detectors(3, 3));
        let phrase = phrase(3, "w");
        assert!(!stream.push(&format!("{phrase} ")).must_abort());
        assert!(!stream.push(&format!("{phrase} ")).must_abort());
        assert!(stream.push(&format!("{phrase} ")).must_abort());
        assert_eq!(stream.peak(), 3);
    }

    #[test]
    fn ordinary_prose_does_not_abort() {
        // The failure that matters more than a missed loop: a detector that fires on a normal
        // answer teaches everyone to ignore `detector_fired`.
        let mut stream = Stream::new(&detectors(4, 3));
        let text = "the parser handles trailing commas now and the tests pass; \
                    next I will look at the lexer and then at the error messages";
        assert!(!stream.push(text).must_abort());
        assert!(!stream.push(" and then the documentation").must_abort());
    }

    #[test]
    fn the_verdict_does_not_depend_on_where_the_deltas_split() {
        // Chunking is a transport detail; a detector sensitive to it would fire on one machine
        // and not another for the same model output.
        // The trailing space matters: a word is counted when the next whitespace completes it,
        // and a stream's last word is genuinely incomplete. Every variant below therefore ends
        // with one, which is what a real stream eventually delivers.
        let text = format!("{} {} {} ", phrase(3, "a"), phrase(3, "a"), phrase(3, "a"));
        let whole = {
            let mut stream = Stream::new(&detectors(3, 3));
            stream.push(&text).must_abort()
        };
        let split = {
            let mut stream = Stream::new(&detectors(3, 3));
            let mut aborted = false;
            for chunk in text.split_inclusive(' ') {
                aborted |= stream.push(chunk).must_abort();
            }
            aborted
        };
        let bytewise = {
            let mut stream = Stream::new(&detectors(3, 3));
            let mut aborted = false;
            for character in text.chars() {
                aborted |= stream.push(&character.to_string()).must_abort();
            }
            aborted
        };
        assert!(whole);
        assert_eq!(split, whole);
        assert_eq!(bytewise, whole);
    }

    #[test]
    fn a_partial_word_is_not_counted_twice() {
        // `hello` arriving as `hel` + `lo` is one word; counting it as two would shift every
        // n-gram by one and make the window depend on chunk size.
        let mut stream = Stream::new(&detectors(2, 2));
        stream.push("hel");
        stream.push("lo ");
        assert_eq!(stream.window(), 1);
    }

    #[test]
    fn the_window_is_bounded() {
        // A model that writes for an hour must not grow this process's memory.
        let mut stream = Stream::new(&detectors(40, 3));
        for index in 0..5_000 {
            stream.push(&format!("word{index} "));
        }
        assert_eq!(stream.window(), WINDOW_WORDS);
    }

    #[test]
    fn a_long_repeat_is_found_in_a_large_window() {
        let mut stream = Stream::new(&detectors(40, 3));
        let filler = phrase(200, "f");
        let repeat = phrase(40, "r");
        stream.push(&format!("{filler} {repeat} "));
        assert!(!stream.push(&format!("{repeat} ")).must_abort(), "second");
        assert!(stream.push(&format!("{repeat} ")).must_abort(), "third");
    }

    #[test]
    fn one_occurrence_short_of_the_limit_is_clear() {
        let mut stream = Stream::new(&detectors(2, 4));
        let phrase = phrase(2, "p");
        for _ in 0..3 {
            assert!(!stream.push(&format!("{phrase} ")).must_abort());
        }
        assert!(stream.push(&format!("{phrase} ")).must_abort());
    }
}
