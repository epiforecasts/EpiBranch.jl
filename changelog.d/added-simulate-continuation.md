`simulate(model, state; max_time)` continues a `SimulationState` returned by
`simulate` up to a later `max_time`, picking the race up from wherever it
stopped rather than redrawing anything already decided. A continued run
agrees bit for bit with one run straight through to the same `max_time` from
the same seed, on every built-in model; the one exception, mass action with
an infectious window that never closes, agrees in distribution. No
observation model runs on a continuation, since a caller such as a particle
filter scores each step itself. `state` is returned unchanged once it has
reached extinction, or stopped for some other reason than `max_time`.
