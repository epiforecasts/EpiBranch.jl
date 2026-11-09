`ContactTracing` now records every case whose trace reached a person in
`:tracers`, and `compute_trace_level!` returns the smallest number of tracing
steps from any index case, found by a multi-source breadth-first search. On a
network with cycles the level was counted along the path of the last tracer
and could exceed the shortest distance. `:traced_by` is still set, and names
the tracer that reached the person first, by the earliest trace time.
