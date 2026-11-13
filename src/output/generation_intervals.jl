"""
    realised_generation_interval(ind::Individual, state::SimulationState)

The realised generation interval of one case: the days from its infector's
infection to its own. Index cases and people who were never infected return
`NaN`.

The interval is shaped by the epidemic. Susceptible depletion and
interventions decide which transmissions happen, so realised intervals
are distributed differently from the intrinsic `generation_time` you
give the model.
"""
function realised_generation_interval(ind::Individual, state::SimulationState)
    (is_infected(ind) && ind.parent_id != 0) || return NaN
    parent = state.individuals[ind.parent_id]
    return ind.infection_time - parent.infection_time
end

"""
    realised_generation_intervals(state::SimulationState)

The realised generation intervals (days) of every case in one simulation
except the index cases, as a vector. See
[`realised_generation_interval`](@ref).

For a model with a fixed `generation_time`, the intrinsic interval is
that distribution itself. For a state-dependent `generation_time`, run
the model without interventions and the realised intervals coincide
with the intrinsic ones, since nothing blocks transmission, as long as
every case's contacts were decided. A run stopped by `max_cases` or
`max_time` leaves out the late infections of its last cases, which shortens
the realised intervals; bound it by `max_generations` instead.

# Examples

```julia
using Statistics
state = simulate(ModelSpec(BranchingProcess(Poisson(2.0), Gamma(2.0, 3.0)));
    n_initial = 5, max_cases = nothing, max_generations = 6)
mean(realised_generation_intervals(state))   # compare with mean(Gamma(2.0, 3.0)) = 6
```
"""
function realised_generation_intervals(state::SimulationState)
    gts = (realised_generation_interval(ind, state) for ind in state.individuals)
    return Float64[g for g in gts if !isnan(g)]
end

"""
    realised_generation_intervals(states::Vector{<:SimulationState})

The realised generation intervals (days) across several simulations, pooled
into one vector.
"""
function realised_generation_intervals(states::Vector{<:SimulationState})
    return reduce(
        vcat, (realised_generation_intervals(s) for s in states);
        init = Float64[]
    )
end

"""
    realised_serial_interval(ind::Individual, state::SimulationState)

The realised serial interval of one case: the days from its infector's
symptom onset to its own. Index cases, people who were never infected and
pairs where either case has no onset (asymptomatic cases, or a model without
onset times) return `NaN`.

Serial intervals are what is observed. Unlike generation intervals, they
can be negative when the infectee's incubation period is shorter than the
infector's. Isolation and tracing shift them away from the intrinsic
generation time.
"""
function realised_serial_interval(ind::Individual, state::SimulationState)
    (is_infected(ind) && ind.parent_id != 0) || return NaN
    return onset_time(ind) - onset_time(state.individuals[ind.parent_id])
end

"""
    realised_serial_intervals(state::SimulationState)

The realised serial intervals (days) of every infector-infectee pair in one
simulation where both cases have an onset, as a vector. Pairs with a missing
onset on either side are skipped. See [`realised_serial_interval`](@ref) and,
for the infection-to-infection counterpart,
[`realised_generation_intervals`](@ref).

# Examples

```julia
state = simulate(
    ModelSpec(BranchingProcess(Poisson(2.0), Gamma(2.0, 3.0));
        attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.4)));
    n_initial = 5, max_cases = nothing, max_generations = 6)
realised_serial_intervals(state)
```
"""
function realised_serial_intervals(state::SimulationState)
    sis = (realised_serial_interval(ind, state) for ind in state.individuals)
    return Float64[s for s in sis if !isnan(s)]
end

"""
    realised_serial_intervals(states::Vector{<:SimulationState})

The realised serial intervals (days) across several simulations, pooled into
one vector.
"""
function realised_serial_intervals(states::Vector{<:SimulationState})
    return reduce(
        vcat, (realised_serial_intervals(s) for s in states);
        init = Float64[]
    )
end
