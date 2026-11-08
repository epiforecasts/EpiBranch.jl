"""
    outbreak_state_space(model, observe; dt::Real = 1.0,
                         simulate_kwargs...)

Treat an outbreak as a state-space model in the sense of
[SSMProblems.jl](https://github.com/TuringLang/SSMProblems.jl), so that
particle filters from packages built on it (such as GeneralisedFilters.jl) can
condition `model`, a [`TransmissionModel`](@ref) or [`ModelSpec`](@ref), on observations over time. Available once SSMProblems is
loaded; this function is defined by a package extension.

The latent state is the [`SimulationState`](@ref) of the outbreak. The prior
starts the outbreak with [`simulate`](@ref) at time zero, taking any of its
keywords (`n_initial`, `initial_cases`, `index_types`, `max_cases` and so on)
from `simulate_kwargs`; `rng` is not one of them, because the filter supplies
its own. The dynamics at step `t` copy the previous state and continue the
copy to time `t * dt` with `simulate(model, state; max_time)`, so `model` is
held once and only the state is copied for each particle. A `max_cases` in
`simulate_kwargs` therefore stops every particle at that size.

`observe(t, state)` returns the distribution of the data at step `t` given the
latent `state`, for example a `Poisson` around a count. Step `t` covers the
interval `((t - 1) * dt, t * dt)`, and the state holds the whole history, so
a count over the step is read with a signal: `TransitionCount(:onset)(state,
((t - 1) * dt, t * dt))` is the number of onsets in it (see
[`TransitionCount`](@ref) and [`Prevalence`](@ref)).

Returns a `SSMProblems.StateSpaceModel`.
"""
function outbreak_state_space end
