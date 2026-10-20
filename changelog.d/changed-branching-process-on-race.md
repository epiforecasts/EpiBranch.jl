`BranchingProcess`, multi-type included, now runs on the same continuous-time
race as the models of a fixed population. Each case creates its contacts when
it is settled. The race then decides every contact in order of infection time,
across generations. What this changes in a run:

- `max_cases` stops at exactly that many cases, and `max_time` leaves out every
  infection after it. Before, the generation that crossed either limit was
  completed first. The last cases of such a run have not yet made all their
  contacts, so per-case quantities such as offspring counts and realised
  generation intervals are right-truncated; bound a run by `max_generations`
  to measure them.
- A contact whose infection was still to be decided when a run stopped is left
  out of the output. One that tracing or an intervention's actions had already
  reached stays a contact of its case, with no infection time, and is not
  listed by `contacts`. Before, the generation engine created every contact of
  the last generation.
- Intervention hooks run case by case in order of infection time, with the
  state as it stands at each case, rather than once per generation. An action
  that `intervention_actions` dates before the infection of the case whose
  settlement proposed it is dropped; before, any action with a finite time
  was admitted.
- A ring of depth 2 or more reaches the contacts of a ring member when the
  ring reaches it, including a member infected later, whose contacts are then
  the ones it goes on to expose. Where the offspring or generation time is a
  function, which may read what settling the member writes, or a window opens
  on a later state, a member's contacts are drawn only once it settles as a
  case, on every window, and the ring does not grow past a member who is
  never infected. A member who is infected has its contacts traced from when
  the ring reached it. This includes
  `incubation_linked_generation_time`. An offspring specification of your own
  that reads nothing settling writes says so with
  `EpiBranch.reads_settled_state`.
- `keep_active` is asked only by tracing, to choose which traced contacts a
  ring grows from.
- A contact drawn for a time before its infector was infected, which a window
  timed from onset can give, is dropped: no exposure happens then. Before, it
  was infected before its infector.
- `risk_applies` is asked about each `Infectiousness` window of a branching
  process, as a route named `:transmission` with `INTERVENTION_REMOVAL` in
  its `until`. An intervention answering `false` for it no longer blocks
  contacts there; before, a branching process applied every risk.
- The line list of a branching process has the `index` column the other
  models already had.

On every model the race now offers the `intervention_actions` of each
intervention once a case settles, to the case and the contacts the model
names. A
network or household model now stops with an error when `CapacityConstrained`
wraps an intervention with nothing to ration, such as `Isolation` or
`ContactTracing`, as a branching process already did. Before, it warned or ran
the wrapped intervention without a budget.
