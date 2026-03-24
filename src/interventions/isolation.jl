"""
    Isolation(; delay, start_time=0.0, residual_transmission=0.0, test_sensitivity=1.0)

Isolate symptomatic, test-positive individuals after a delay from symptom onset.

Requires `:onset_time` and `:asymptomatic` on individuals (set via
[`Disease`](@ref) or [`clinical_presentation`](@ref)).

`test_sensitivity` is the probability that a symptomatic individual tests
positive and is therefore eligible for isolation.

Initialises: `:isolated`, `:isolation_time`, `:test_positive`.
"""
Base.@kwdef struct Isolation <: AbstractIntervention
    delay::Distribution
    start_time::Float64 = 0.0
    residual_transmission::Float64 = 0.0
    test_sensitivity::Float64 = 1.0
end

required_fields(::Isolation) = [:onset_time, :asymptomatic]
residual_transmission(iso::Isolation) = iso.residual_transmission
start_time(iso::Isolation) = iso.start_time
intervention_time(::Isolation, ind::Individual) = isolation_time(ind)

function reset!(::Isolation, ind::Individual)
    ind.state[:isolated] = false
    ind.state[:isolation_time] = Inf
    return nothing
end

function initialise_individual!(iso::Isolation, individual, state)
    individual.state[:isolated] = false
    individual.state[:isolation_time] = Inf
    # Test result determined by isolation's test sensitivity
    is_asymp = get(individual.state, :asymptomatic, false)
    individual.state[:test_positive] = !is_asymp && rand(state.rng) < iso.test_sensitivity
end

function resolve_individual!(iso::Isolation, individual, state)
    is_isolated(individual) && return nothing
    is_asymptomatic(individual) && return nothing
    !is_test_positive(individual) && return nothing

    iso_delay = rand(state.rng, iso.delay)
    iso_time = onset_time(individual) + iso_delay

    # If contact tracing has already computed a traced isolation time,
    # take the earlier of self-reporting and tracing
    traced_time = get(individual.state, :traced_isolation_time, Inf)
    set_isolated!(individual, min(iso_time, traced_time))
    return nothing
end
