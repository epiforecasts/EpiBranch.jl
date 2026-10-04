`linelist` reported `Isolation`'s own bookkeeping — the flag recording which
intervention set an isolation, and the quarantine time a self-report
superseded — as ordinary columns, because it only knew to drop a few such keys
by name and missed these two. It now drops every key whose name marks it as
internal, so a column appears only for state that a composed component records
as output.
