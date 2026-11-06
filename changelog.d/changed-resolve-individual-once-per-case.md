`resolve_individual!` runs once for each infection, when the case is
settled: after its clinical transitions and before it infects anyone. While
it runs, `state.max_infection_time` is the case's infection time,
`state.current_generation` its generation and `state.cumulative_cases` the
count of cases up to and including it. Clinical transitions see the same
values; they used to see the previous case's clock and count instead, and a
run's returned state had `current_generation` left at 0 rather than set to
the latest generation reached.
