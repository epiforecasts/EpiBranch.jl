# Test adapter: accept a `transitions` kwarg and attach it to the model's
# `progression`, then call the real kwarg-free `simulate`. Lets the
# transition-behaviour tests keep their shape while exercising the new
# model.progression API (the natural history now lives on the model).
function tsim(model::BranchingProcess;
        transitions = EpiBranch.AbstractClinicalTransition[],
        attributes = NoAttributes(),
        interventions = AbstractIntervention[],
        kwargs...)
    spec = ModelSpec(model; progression = transitions, attributes = attributes,
        interventions = interventions)
    return simulate(spec; kwargs...)
end
