The tracing ring walk now holds its frontier and pending contacts in vectors
with a concrete element type, so each `trace_contacts!` call is dispatched at
compile time rather than at run time. Seeded results are unchanged.
