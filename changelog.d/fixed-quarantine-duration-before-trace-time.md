`Quarantine(duration = f)` called `f(rng, contact)` before the contact's
`trace_time` was recorded, so a duration function had no way to read it and
compute a value relative to the trace. `_trace_pair!` and `SourceTracing`'s
`trace_contacts!` now record `trace_time` before calling `apply_trace!`.
`duration` can then read `ind.state[:trace_time]` (or other state such as
`onset_time`) set during the same trace.
