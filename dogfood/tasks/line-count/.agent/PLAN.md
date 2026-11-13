# Plan — fix the line count

`tally` reports one line too many for any file ending in a newline. `lines("a\nb\n")` returns 3
because `str::split` yields a trailing empty field after the final separator, and the empty check
only guards the empty-file case.

## Why this shape

The fix is in one function and the regression test is in the same module. A second subtask would
only re-run the same commands.
