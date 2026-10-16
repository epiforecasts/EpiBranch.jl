`SourceTracing` traces a case back to its own infector rather than forward to
the people it infects, through the same eligibility, rate, delay and action
protocol as `ContactTracing`. Both engines offer a case's own infector to any
intervention declaring the new `EpiBranch.traces_source` trait, through a
second call to `trace_contacts!`; `ContactTracing` itself is unaffected. On a
continuous-time model the infector's own pending proposals are still checked
against the usual competing risks, so quarantining it blocks its remaining
transmission without undoing anything already settled; on the generation
engine, where a case's secondary cases are drawn in one step before it can
itself be traced back, `SourceTracing` only records the link.
