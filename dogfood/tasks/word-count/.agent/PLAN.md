# Plan — add a `unique` subcommand

`tally` already prints lines, words and unique words. The number a reader actually wants when
comparing two files is the number of words that appear exactly once, so add a `unique` line to
the output and a `--unique` flag that prints only that number.

## Why this shape

The counting rule is one function in `src/count.rs`, and the output is one block in `src/main.rs`.
Splitting the work any finer would produce two subtasks that each touch every file, which is how
a plan turns into merge conflicts against itself.
