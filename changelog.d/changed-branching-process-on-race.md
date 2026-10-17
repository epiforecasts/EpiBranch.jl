`BranchingProcess`, multi-type included, now runs on the same continuous-time
race as the models of a fixed population. Each case creates its contacts when
it is settled. The race then decides every contact in order of infection time,
across generations. What this changes in a run:

- `max_cases` stops at exactly that many cases, and `max_time` leaves out every
  infection after it. Before, the generation that crossed either limit was
  completed first.
- A contact whose infection was still to be decided when a run stopped stays a
  contact of its case, with no infection time, and is not listed by
  `contacts`.
- Intervention hooks run case by case in order of infection time, with the
  state as it stands at each case, rather than once per generation.
- A ring of depth 2 or more reaches the contacts of a ring member when the
  ring reaches it, including a member infected later, whose contacts are then
  the ones it goes on to expose.
- `keep_active` is asked only by tracing, to choose which traced contacts a
  ring grows from. An intervention answering it without tracing is named in a
  warning.
- The deprecated `apply_post_transmission!` is no longer called on a
  branching process.
- The line list of a branching process has the `index` column the other
  models already had.

On every model the race now offers the `intervention_actions` of each
intervention once a case settles, to the case and the contacts the model
names, whether or not the intervention declares `continuous_actions`.
