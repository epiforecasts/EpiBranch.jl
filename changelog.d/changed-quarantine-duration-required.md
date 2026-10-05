`Quarantine`'s `duration` is now required, with no default: `Quarantine(;
duration = Inf)` reproduced the previous behaviour silently, so a caller who
did not know the keyword existed got an indefinite quarantine anyway.
`ContactTracing` now takes its `action` the same way, with no default, on
both the keyword and the terse positional constructor. A contact traced and
quarantined before it was ever infected should not go on blocking its own
transmission after a route reaches it later; requiring the duration is what
makes that a choice rather than an accident. Pass `Quarantine(duration =
Inf)` to keep the previous behaviour explicitly.
