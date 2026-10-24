`EpiBranch.action_candidates(iv, state, current, traced, is_pending, members, contacts)`
chooses who an intervention's `intervention_actions` is offered to once a case
settles. The default is the case and the contacts it can still reach;
`GroupVaccination` writes its own method, offering the members of the case's
group instead. An intervention written outside the package can narrow or
widen its own candidates the same way, without the simulation having to walk
everyone in the run.
