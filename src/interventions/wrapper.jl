# An intervention that modifies another intervention, such as `Scheduled`
# (when it may act) or `CapacityConstrained` (how many it may reach). A
# subtype stores the wrapped intervention in an `intervention` field and
# inherits the plain delegation of every hook below, then overrides only the
# hooks it gates. `apply_post_transmission!`, `keep_active` and
# `trace_contacts!` have no default here, so each wrapper states what it does
# with them.
abstract type InterventionWrapper <: AbstractIntervention end

function initialise_individual!(w::InterventionWrapper, ind, state)
    initialise_individual!(w.intervention, ind, state)
end
function resolve_individual!(w::InterventionWrapper, ind, state)
    resolve_individual!(w.intervention, ind, state)
end
function competing_risk(w::InterventionWrapper, parent, contact, state)
    competing_risk(w.intervention, parent, contact, state)
end
is_active(w::InterventionWrapper, state::SimulationState) = is_active(w.intervention, state)
required_fields(w::InterventionWrapper) = required_fields(w.intervention)
function intervention_time(w::InterventionWrapper, ind::Individual)
    intervention_time(w.intervention, ind)
end
reset!(w::InterventionWrapper, ind::Individual) = reset!(w.intervention, ind)
function infectious_removal_time(w::InterventionWrapper, ind::Individual)
    infectious_removal_time(w.intervention, ind)
end
traces_contacts(w::InterventionWrapper) = traces_contacts(w.intervention)
_unwrap_scheduled(w::InterventionWrapper) = _unwrap_scheduled(w.intervention)

risk_applies(w::InterventionWrapper, route) = risk_applies(w.intervention, route)
