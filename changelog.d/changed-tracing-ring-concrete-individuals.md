The tracing ring walk now holds its frontier and pending contacts in vectors
with a concrete element type. Each `trace_contacts!` call is therefore
dispatched at compile time instead of at run time. Seeded results are
unchanged.
