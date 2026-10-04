`linelist` reported `Isolation`'s own bookkeeping — `isolated_by_isolation`,
the flag recording which intervention set an isolation, and
`isolation_time_before_isolation`, the quarantine time a self-report
superseded — as ordinary columns, because it only knew to drop a few such
keys by name and missed these two. Internal keys an intervention keeps
purely for its own bookkeeping now start with an underscore, as
`:_intervention_actions` already did, and `linelist` drops every key with
that prefix instead of naming each one, so a column appears only for state
that a composed component records as output.
