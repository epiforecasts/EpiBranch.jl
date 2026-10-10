# An intervention that modifies another intervention, such as `Scheduled`
# (when it may act) or `CapacityConstrained` (how many it may reach). A
# subtype stores the wrapped intervention in an `intervention` field and
# inherits the plain delegation of every hook below, then overrides only the
# hooks it gates. `apply_post_transmission!`, `keep_active` and
# `trace_contacts!` have no default here, so each wrapper states what it does
# with them.
abstract type InterventionWrapper <: AbstractIntervention end

function initialise_individual!(w::InterventionWrapper, ind, state)
    return initialise_individual!(w.intervention, ind, state)
end
function resolve_individual!(w::InterventionWrapper, ind, state)
    return resolve_individual!(w.intervention, ind, state)
end
function competing_risk(w::InterventionWrapper, parent, contact, state)
    return competing_risk(w.intervention, parent, contact, state)
end
is_active(w::InterventionWrapper, state::SimulationState) = is_active(w.intervention, state)
required_fields(w::InterventionWrapper) = required_fields(w.intervention)
function intervention_time(w::InterventionWrapper, ind::Individual)
    return intervention_time(w.intervention, ind)
end
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
function on_infection_settled!(w::InterventionWrapper, ind, state, rng)
    return on_infection_settled!(w.intervention, ind, state, rng)
end
_unwrap_scheduled(w::InterventionWrapper) = _unwrap_scheduled(w.intervention)

binding_release(w::InterventionWrapper) = binding_release(w.intervention)
risk_applies(w::InterventionWrapper, route) = risk_applies(w.intervention, route)
reads_population_state(w::InterventionWrapper) = reads_population_state(w.intervention)
# A wrapper that can withdraw the inner block part-way through a stretch
# already recorded cannot have those stretches read back: one per-host record
# says nothing about when the gating closed. The default is therefore to
# declare none, which narrows the infectious window instead (above), and a
# wrapper whose gating cannot withdraw a block forwards the declaration.
