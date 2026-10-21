`reset!` on `ContactTracing` and `SourceTracing` now undoes only the
quarantine a `Quarantine` action itself recorded, restoring whatever
isolation stood before it rather than clearing the case's isolation
outright. A `Scheduled` contact tracing that has not yet started previously
cleared a case's isolation set by a separate `Isolation` intervention,
because `:isolated`/`:isolation_time` are shared state; isolation combined
with tracing that starts late could then give worse control than isolation
alone.
