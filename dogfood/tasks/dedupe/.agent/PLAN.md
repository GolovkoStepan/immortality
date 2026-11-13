# Plan — dedupe the word set

`count::unique_words` stores seen words in a `Vec` and calls `contains` on every word, which is
quadratic. It is correct today and the existing tests pin that. Replace the `Vec` with a
`BTreeSet<String>` so the behaviour is unchanged and the lookup is logarithmic.

## Why this shape

A refactor is only a refactor if the old behaviour is pinned first. The single subtask therefore
adds nothing new: it changes the container, keeps every existing test green, and adds one test
that the ordering of the output cannot depend on the container's iteration order.
