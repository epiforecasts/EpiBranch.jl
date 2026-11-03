`SourceTracing` traces a case back to its own infector rather than forward to
the people it infects, through the same eligibility, rate, delay and action
protocol as `ContactTracing`. The race offers a case's own infector to any
intervention declaring the new `EpiBranch.traces_source` trait, through a
second call to `trace_contacts!`; `ContactTracing` itself is unaffected. The
infector's own pending proposals are still checked against the usual
competing risks, so quarantining it blocks its remaining transmission
without undoing anything already settled.
