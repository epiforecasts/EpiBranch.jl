`Scheduled` and `CapacityConstrained` contact tracing and `SourceTracing` are
now judged at the time of the trace itself, which each tracer reports through
`intervention_time`, rather than at the contact's isolation time. A
`Scheduled(ContactTracing(...; action = FlagOnly()))` used to flag, ring-grow
from and vaccinate or isolate contacts before its `start_time`, because a
`FlagOnly` trace never writes an isolation time. `reset!` on either tracer now
undoes only the trace time, removal entries and flags that tracer recorded.
Resetting one no longer wipes the quarantine of the other. A `FlagOnly` reset
also clears the isolation time recorded for the trace.
