`EpiBranch.action_candidates(iv, state, current, traced, is_pending, members, contacts)`
chooses who an intervention's `intervention_actions` is offered once a case
settles. The default is the case and the contacts it can still reach;
`GroupVaccination` writes its own method, offering the members of the case's
group. An intervention written outside the package narrows or widens its own
candidates the same way, without the simulation having to walk everyone in
the run.
