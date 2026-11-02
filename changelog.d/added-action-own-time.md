An admitted `InterventionAction`'s `effect!` now runs at `action.time`,
against the state as it then stands, rather than the moment `_admit_actions!`
accepts it: on a continuous-time race, that means the race's own event queue,
interleaved with the cases still settling. It may return a vector of further
`InterventionAction`s in place of `nothing`, admitted through the same
funnel, so an effect can schedule what follows from it instead of needing a
second action proposed and accepted up front. `InterventionAction` takes an
optional `claim!(individual, time, state)`, run once, synchronously, at
admission, for the rarer case where a later proposal needs to read something
back straight away rather than wait for a deferred `effect!`; the built-in
vaccinations use it to keep recording a dose immediately, exactly as before.
