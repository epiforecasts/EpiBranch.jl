`resolve_individual!` runs once for each infection on both engines, when the
case is settled: after its clinical transitions and before it infects anyone.
While it runs, `state.max_infection_time` is the case's infection time,
`state.current_generation` its generation and `state.cumulative_cases` the
count of cases up to and including it. Clinical transitions see the same
values. On the generation engine this changes three things:

- Cases of the last generation are resolved too. Before, a run stopped by
  `max_generations` or `max_cases` left them without an isolation time.
- It is no longer called on uninfected contacts that `keep_active` keeps in
  the simulation to generate contacts of their own, such as the ring members of
  `ContactTracing(depth = 2)`.
- A `Scheduled` gate is evaluated against the case's own infection time.
  Before, it read the latest infection time of the case's whole generation, so
  `Scheduled(iso; start_time = 10.0)` could isolate a case infected on day 9.

On the continuous-time models the clinical transitions see the case's own
clock and count. They used to see the previous case's. The state such a
model returns has `current_generation` set to the latest generation reached,
where it was left at 0.

A seeded run on the generation engine can give different results when an
intervention's `resolve_individual!` draws random numbers, as `Isolation` does
with a random `onset_to_isolation_delay`: the last generation's cases now make
draws of their own, and ring members kept by `keep_active` no longer do. The
observation model's draws shift with them, as do the later runs of a
multi-run `simulate`.
