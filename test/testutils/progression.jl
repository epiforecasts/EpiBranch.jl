# Test adapter: accept a `transitions` kwarg and attach it to the model's
# `progression`, then call the real kwarg-free `simulate`. Lets the
# transition-behaviour tests keep their shape while exercising the new
# model.progression API (the natural history now lives on the model).
function tsim(
        model::BranchingProcess;
        transitions = EpiBranch.AbstractClinicalTransition[],
        progression = transitions,
        attributes = NoAttributes(),
        interventions = AbstractIntervention[],
        observation = NoObservation(),
        kwargs...
    )
    spec = ModelSpec(
        model; progression, attributes, interventions, observation
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

# A state whose population has two named groups, for tests of a budget per
# group. Individuals fall in the group named by their `:type`.
function _grouped_state(groups = ["A", "B"])
    model = BranchingProcess(
        [1.2 0.3; 0.3 1.2], R -> Poisson(R), Dirac(14.0);
        population = Population([100, 100]; groups)
    )
    return empty_state(model)
end

# The SIR removal: infected cases recover after `delay` and leave the process.
function sir(delay)
    return [Transition(:recovered; from = :infection, delay = delay, terminal = true)]
end

# The attributes most tests use: a log-normal incubation period.
# `clinical_presentation` defaults to no asymptomatic cases.
const CLINICAL = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

# Fraction of `n` outbreaks simulated from a `ModelSpec` that
# `containment_probability` counts as contained. Extra keywords go to `simulate`.
function contained(
        spec::ModelSpec, n = 200;
        max_cases = 200, seed = 11, kwargs...
    )
    states = simulate(spec, n; max_cases, rng = StableRNG(seed), kwargs...)
    return containment_probability(states; max_cases).proportion
end

# Run the Sellke race over `members` (a vector of ids, or a count `n` for
# `1:n`), with the cases in `seed` (an id or ids) infected at time zero. This is
# the set-up shared by the hand-built race tests. Tests that seed the race
# differently pass `seed!`, which replaces `seed`. Other keywords go to the race.
function race!(state, members, rng; seed = 1, seed! = nothing, kwargs...)
    ids = members isa Integer ? collect(1:members) : members
    seeded = seed! === nothing ?
        (best, _, _) -> (best[collect(seed)] .= 0.0) : seed!
    return EpiBranch._sellke_race!(state, ids, rng; seed! = seeded, kwargs...)
end
