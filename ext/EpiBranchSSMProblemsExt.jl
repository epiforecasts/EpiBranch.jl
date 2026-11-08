module EpiBranchSSMProblemsExt

# An outbreak as an SSMProblems.jl state-space model. The latent state is the
# `SimulationState`, and the dynamics are `simulate(model, state; max_time)`,
# the engine's continuation seam, so a particle filter from a package built on
# SSMProblems can run on any model that supports it.

using EpiBranch
using EpiBranch: TransmissionModel, ModelSpec, SimulationState
using Random: AbstractRNG
using SSMProblems: StatePrior, LatentDynamics, StateSpaceModel, DistributionObservation
import SSMProblems

# A bare process or a `ModelSpec` is a model for `simulate`.
const _Model = Union{TransmissionModel, ModelSpec}

# Starts the outbreak at time zero, before any step of the dynamics has run.
struct OutbreakPrior{M <: _Model, K <: NamedTuple} <: StatePrior
    model::M
    kwargs::K
end

# Continues a copy of the previous state by one step of length `dt`. The copy
# is needed because a filter resamples by reusing one state for several
# particles, and continuation mutates the state it is given.
struct OutbreakDynamics{M <: _Model, T <: Real} <: LatentDynamics
    model::M
    dt::T
end

function SSMProblems.simulate(rng::AbstractRNG, prior::OutbreakPrior)
    return simulate(prior.model; prior.kwargs..., max_time = 0.0, rng)
end

function SSMProblems.simulate(
        rng::AbstractRNG, dyn::OutbreakDynamics, t::Integer, x::SimulationState
    )
    return simulate(dyn.model, deepcopy(x); max_time = t * dyn.dt, rng)
end

function EpiBranch.outbreak_state_space(
        model::_Model, observe; dt::Real = 1.0, simulate_kwargs...
    )
    dt > 0 || throw(ArgumentError("dt must be positive, got $dt"))
    haskey(simulate_kwargs, :rng) &&
        throw(ArgumentError("rng is supplied by the filter, not to outbreak_state_space"))
    haskey(simulate_kwargs, :max_time) &&
        throw(ArgumentError("max_time is set by each step of the dynamics"))
    return StateSpaceModel(
        OutbreakPrior(model, NamedTuple(simulate_kwargs)),
        OutbreakDynamics(model, dt),
        DistributionObservation(observe)
    )
end

end # module
