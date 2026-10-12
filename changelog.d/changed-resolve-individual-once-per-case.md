`resolve_individual!` runs once for each infection on both engines, when the
case is settled: after its clinical transitions and before it infects anyone.
While it runs, `state.max_infection_time` is the case's infection time,
`state.current_generation` its generation and `state.cumulative_cases` the
count of cases up to and including it. Clinical transitions see the same
values. On the generation engine this changes three things:

- Cases of the last generation are resolved too. Before, a run stopped by
  `max_generations` or `max_cases` left them without an isolation time.
- Uninfected contacts that `keep_active` keeps generating contacts, such as
  the ring members of `ContactTracing(depth = 2)`, are no longer passed to it.
- A `Scheduled` gate is evaluated against the case's own infection time.
  Before, it read the latest infection time of the case's whole generation, so
  `Scheduled(iso; start_time = 10.0)` could isolate a case infected on day 9.

On the continuous-time models the clinical transitions now see the case's
clock and count, where they saw those of the previous case. A seeded run on
the generation engine with tracing deeper than one generation can draw a
different trajectory than before.
