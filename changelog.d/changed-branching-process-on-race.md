`BranchingProcess`, multi-type included, now runs on the same continuous-time
race as the models of a fixed population. Each case creates its contacts when
it is settled. The race then decides every contact in order of infection time,
across generations. What this changes in a run:

- `max_cases` stops at exactly that many cases, and `max_time` leaves out every
  infection after it. Before, the generation that crossed either limit was
  completed first.
- A contact whose infection was still to be decided when a run stopped is left
  out of the output. One that tracing or an intervention's actions had already
  reached stays a contact of its case, with no infection time, and is not
  listed by `contacts`. Before, the generation engine created every contact of
  the last generation.
- Intervention hooks run case by case in order of infection time, with the
  state as it stands at each case, rather than once per generation.
- A ring of depth 2 or more reaches the contacts of a ring member when the
  ring reaches it, including a member infected later, whose contacts are then
  the ones it goes on to expose. Where the offspring or generation time is a
  function, which may read what settling the member writes, or a window opens
  on a later state, a member's contacts are drawn only once it settles as a
  case, on every window, and an uninfected member has none. A member that is
  eligible for tracing in its own right then has its contacts traced from its
  own trigger, as before, so the deeper ring adds nothing there; it still
  reaches the contacts of members that are not eligible. This includes
  `incubation_linked_generation_time`. An offspring specification of your own that reads nothing settling
  writes says so with `EpiBranch.reads_settled_state`.
- `keep_active` is asked only by tracing, to choose which traced contacts a
  ring grows from. An intervention answering it without tracing is named in a
  warning.
- A contact drawn for a time before its infector was infected, which a window
  timed from onset can give, is dropped: no exposure happens then. Before, it
  was infected before its infector.
- `risk_applies` is asked about each `Infectiousness` window of a branching
  process, as a route named `:transmission` with `INTERVENTION_REMOVAL` in
  its `until`. An intervention answering `false` for it no longer blocks
  contacts there; before, a branching process applied every risk.
- The deprecated `apply_post_transmission!` is no longer called on a
  branching process.
- The line list of a branching process has the `index` column the other
  models already had.

On every model the race now offers the `intervention_actions` of each
intervention once a case settles, to the case and the contacts the model
names, whether or not the intervention declares `continuous_actions`. A
network or household model now stops with an error when `CapacityConstrained`
wraps an intervention with nothing to ration, such as `Isolation` or
`ContactTracing`, as a branching process already did. Before, it warned or ran
the wrapped intervention without a budget.
