`EpiBranch.continuous_candidates(iv, state, current, traced, is_pending, members, contacts)`
is now the documented way for a `continuous_actions` intervention to choose who
`intervention_actions` is offered once a case settles on a network or household
model. Without a method of it, an intervention still gets the default, every
settled case plus everyone else still pending, which costs the size of the
race; `RingVaccination` and `GroupVaccination` now write their own methods of
it, narrowing their own candidates the same way they always did.
