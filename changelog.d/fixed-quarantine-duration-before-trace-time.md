`Quarantine(duration = f)` called `f(rng, contact)` before the contact's
`trace_time` was recorded, so a duration function could not read it to
compute a value relative to the trace. `_trace_pair!` and `SourceTracing`'s
`trace_contacts!` now record `trace_time` before calling `apply_trace!`, so
`duration` can read `ind.state[:trace_time]` (or other state such as
`onset_time`) set during the same trace.
