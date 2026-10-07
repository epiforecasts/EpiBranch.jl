`ContactTracing`'s `quarantine_on_trace` keyword is deprecated in favour of
`action`: `quarantine_on_trace = true` maps onto `Quarantine(duration = Inf)`
and `quarantine_on_trace = false` onto `FlagOnly()`, with a deprecation
warning either way. The `Bool` hid which action a trace takes behind a
switch, where `action` makes it a value the caller supplies, as a
`TraceAction` isn't limited to those two built-ins.
