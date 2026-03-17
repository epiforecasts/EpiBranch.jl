"""
    Isolation(; delay, start_time=0.0, residual_transmission=0.0)

Isolate symptomatic, test-positive individuals after a delay from symptom onset.

Requires fields on individuals (set via `clinical_presentation()`):
`:onset_time`, `:asymptomatic`, `:test_positive`.

Initialises: `:isolated`, `:isolation_time`.
"""
Base.@kwdef struct Isolation <: AbstractIntervention
    delay::Distribution
    start_time::Float64 = 0.0
    residual_transmission::Float64 = 0.0
end

required_fields(::Isolation) = [:onset_time, :asymptomatic, :test_positive]

function initialise_individual!(iso::Isolation, individual, state)
    individual.state[:isolated] = false
    individual.state[:isolation_time] = Inf
    return nothing
end

function resolve_individual!(iso::Isolation, individual, state)
    is_isolated(individual) && return nothing
    individual.infection_time < iso.start_time && return nothing
    is_asymptomatic(individual) && return nothing
    !is_test_positive(individual) && return nothing

    iso_delay = rand(state.rng, iso.delay)
    iso_time = onset_time(individual) + iso_delay

    set_isolated!(individual, iso_time)

    return nothing
end
