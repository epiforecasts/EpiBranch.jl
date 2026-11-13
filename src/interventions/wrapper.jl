"""
    InterventionWrapper <: AbstractIntervention

An intervention that modifies another intervention, such as `Scheduled`
(when it may act) or `CapacityConstrained` (how many it may reach). A
subtype stores the wrapped intervention in an `intervention` field and
inherits plain delegation of every hook and overrides only the hooks it
restricts. `trace_contacts!` has no default, so each wrapper defines it.
`keep_active` forwards to the wrapped intervention unless the wrapper
restricts it.
"""
abstract type InterventionWrapper <: AbstractIntervention end

initialise_individual!(w::InterventionWrapper, ind, state) = initialise_individual!(w.intervention, ind, state)
resolve_individual!(w::InterventionWrapper, ind, state) = resolve_individual!(w.intervention, ind, state)
competing_risk(w::InterventionWrapper, parent, contact, state) = competing_risk(w.intervention, parent, contact, state)
is_active(w::InterventionWrapper, state::SimulationState) = is_active(w.intervention, state)
required_fields(w::InterventionWrapper) = required_fields(w.intervention)
intervention_time(w::InterventionWrapper, ind::Individual) = intervention_time(w.intervention, ind)
reset!(w::InterventionWrapper, ind::Individual) = reset!(w.intervention, ind)
function infectious_removal_time(w::InterventionWrapper, ind::Individual)
    t = infectious_removal_time(w.intervention, ind)
    isempty(removal_gap_host_times(w)) || return t
    # A stretch with its own release is still read by the per-contact risk,
    # which re-checks the wrapper's gate at every proposal and already honours
    # a block the wrapper later withdraws; narrowing the window for it here
    # would turn a removal due to lapse into one that never does. Only a
    # stretch with no release, standing on the wrapped intervention's own
    # terms, needs the window closed here, since nothing is left to hand the
    # host back.
    for key in removal_gap_host_times(w.intervention)
        t = min(t, permanent_removal_time(ind, key))
    end
    return t
end
traces_contacts(w::InterventionWrapper) = traces_contacts(w.intervention)
traces_source(w::InterventionWrapper) = traces_source(w.intervention)
function on_infection_settled!(w::InterventionWrapper, ind, state, rng)
    return on_infection_settled!(w.intervention, ind, state, rng)
end
function keep_active(w::InterventionWrapper, state, targets, is_new)
    return keep_active(w.intervention, state, targets, is_new)
end

binding_release(w::InterventionWrapper) = binding_release(w.intervention)
risk_applies(w::InterventionWrapper, route) = risk_applies(w.intervention, route)

# Read through any wrapper, so `CapacityConstrained` composes with `Scheduled`
# in either order.
capacity_key(w::InterventionWrapper) = capacity_key(w.intervention)
capacity_time_key(w::InterventionWrapper) = capacity_time_key(w.intervention)
held_until(w::InterventionWrapper, ind) = held_until(w.intervention, ind)
intervention_actions(w::InterventionWrapper, state, candidates) = intervention_actions(w.intervention, state, candidates)
may_revise(w::InterventionWrapper, prior, new) = may_revise(w.intervention, prior, new)
acts_on_population(w::InterventionWrapper) = acts_on_population(w.intervention)
needs_contacts(w::InterventionWrapper) = needs_contacts(w.intervention)
persistent_competing_risks(w::InterventionWrapper) = persistent_competing_risks(w.intervention)
_may_lapse(w::InterventionWrapper) = _may_lapse(w.intervention)
_triggers_of(w::InterventionWrapper) = _triggers_of(w.intervention)
standing_block(w::InterventionWrapper) = standing_block(w.intervention)
susceptibility_host_times(w::InterventionWrapper) = susceptibility_host_times(w.intervention)
susceptibility_components(w::InterventionWrapper, host) = susceptibility_components(w.intervention, host)
_innermost(w::InterventionWrapper) = _innermost(w.intervention)
action_candidates(w::InterventionWrapper, state, current, traced, is_pending, members, contacts) =
    action_candidates(w.intervention, state, current, traced, is_pending, members, contacts)
# A wrapper that can withdraw the inner block part-way through a stretch
# already recorded cannot have those stretches read back: one per-host record
# says nothing about when the gating closed. The default is therefore to
# declare none, which narrows the infectious window instead (above), and a
# wrapper whose gating cannot withdraw a block forwards the declaration.

# A wrapper that declares no key of its own (the default above, or one that
# only sometimes forwards, such as a lapsing `Scheduled`) is the one case
# `removal_gap_host_times` leaves silent on purpose: whatever the intervention
# it holds would have named, or already counts as withdrawn further down, is
# withdrawn here too, so the gap stays unnamed without looking forgotten. A
# key the wrapper forwards stays named rather than withdrawn, but the wrapped
# intervention's withdrawals (declared for reasons of its own, such as
# `ContactTracing`'s shared history) still need to reach the model: forwarding
# one key is no reason to drop another.
function removal_gap_host_times_withdrawn(w::InterventionWrapper)
    named = removal_gap_host_times(w)
    inner_withdrawn = removal_gap_host_times_withdrawn(w.intervention)
    isempty(named) || return Tuple(setdiff(inner_withdrawn, named))
    inner_named = removal_gap_host_times(w.intervention)
    return Tuple(union(inner_withdrawn, inner_named))
end

"""
    gates_on_population_clock(intervention) -> Bool

For extension authors: whether the effect of this intervention depends on the
clock or case count of the population, as [`Scheduled`](@ref) does through its
start condition, [`CapacityConstrained`](@ref) through its budget per
period and [`HeldCapacity`](@ref) through its cap on places held at once. A
model with no such clock, such as the branching process over households,
refuses an intervention that declares it. The default is `false`.
A wrapper written outside the package should forward the answer for the wrapped
intervention, as every `InterventionWrapper` here does. The answer is then
found through any nesting.
"""
gates_on_population_clock(::AbstractIntervention) = false
gates_on_population_clock(w::InterventionWrapper) = gates_on_population_clock(w.intervention)
