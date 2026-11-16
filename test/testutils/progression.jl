# Test adapter: accept a `transitions` kwarg and attach it to the model's
# `progression`, then call the real kwarg-free `simulate`. Lets the
# transition-behaviour tests keep their shape while exercising the new
# model.progression API (the natural history now lives on the model).
function tsim(
        model::BranchingProcess;
        transitions = EpiBranch.AbstractClinicalTransition[],
        attributes = NoAttributes(),
        interventions = AbstractIntervention[],
        kwargs...
    )
    spec = ModelSpec(
        model; progression = transitions, attributes = attributes,
        interventions = interventions
    )
    return simulate(spec; kwargs...)
end

# A state with no individuals, for tests that drive one hook directly. The
# defaults match the model most of those tests use.
function empty_state(
        model = BranchingProcess(Poisson(1.0), Exponential(5.0));
        progression = EpiBranch.AbstractClinicalTransition[],
        attributes = NoAttributes(),
        rng = MersenneTwister(1)
    )
    return EpiBranch.new_state(model, progression, attributes, rng)
end

# The SIR removal: infected cases recover after `delay` and leave the process.
function sir(delay)
    return [Transition(:recovered; from = :infection, delay = delay, terminal = true)]
end
