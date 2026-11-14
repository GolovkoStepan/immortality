# Plan — the `once` family in `tally`

`tally` prints lines, words and unique words. A reader comparing two files wants one more number:
how many words appear exactly once. This task adds that count, makes the binary print it, and
exposes the words themselves in sorted order.

## Why this shape

Three subtasks, in a line:

1. the counting rule in `src/count.rs` (`once_words`);
2. the output in `src/main.rs` (`once: N`);
3. the listing in `src/count.rs` (`once_words_sorted`).

Each is a separate gate, and each gate is a separate test target, so the green of one does not
make the next one green. That is not tidiness: `word-count` is the task that has to show three
consecutive closures, and a plan whose later subtasks are already green when they start would let
the milestone's bar be met by a run that changed one file.

## Why three and not one

Splitting the count from the printing from the listing would be needless if the only goal were the
feature. The goal is also the bar: three subtasks, each closing because its own gate went from red
to green. Two subtasks could not demonstrate three in a row, and one shared gate could not
demonstrate anything past the first.
