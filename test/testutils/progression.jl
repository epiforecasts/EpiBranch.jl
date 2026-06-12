# Test adapter: accept a `transitions` kwarg and attach it to the model's
# `progression`, then call the real kwarg-free `simulate`. Lets the
# transition-behaviour tests keep their shape while exercising the new
# model.progression API (the natural history now lives on the model).
function tsim(model::BranchingProcess;
        transitions = EpiBranch.AbstractClinicalTransition[],
        attributes = NoAttributes(),
        interventions = AbstractIntervention[],
        kwargs...)
    m = BranchingProcess(model.infectiousness;
        population_size = model.population_size,
        n_types = model.n_types,
        type_labels = model.type_labels,
        progression = transitions,
        attributes = attributes,
        interventions = interventions)
    return simulate(m; kwargs...)
end
