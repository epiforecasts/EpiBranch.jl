"""
    Isolation(; delay, start_time=0.0, test_sensitivity=1.0, residual_transmission=0.0)

Isolate symptomatic, test-positive individuals after a delay from symptom onset.

The reduction in transmission is determined by the generation time CDF
evaluated at the isolation time (hazard-based formulation). With leaky
isolation (`residual_transmission > 0`), some transmission continues
after isolation:

    effective_R = R₀ · G(t_iso) + residual_transmission · R₀ · (1 - G(t_iso))

- `delay`: distribution of time from symptom onset to isolation
- `start_time`: simulation time when isolation policy begins
- `test_sensitivity`: probability a symptomatic case tests positive
- `residual_transmission`: fraction of transmission that continues while
  isolated (0 = perfect, 1 = no effect). Models household transmission etc.
"""
Base.@kwdef struct Isolation <: AbstractIntervention
    delay::Distribution
    start_time::Float64 = 0.0
    test_sensitivity::Float64 = 1.0
    residual_transmission::Float64 = 0.0
end

function resolve_individual!(iso::Isolation, individual, state)
    # Already resolved
    individual.isolated && return nothing

    # Policy not yet active
    individual.infection_time < iso.start_time && return nothing

    # Asymptomatic cases are never isolated via symptom-based surveillance
    individual.asymptomatic && return nothing

    # Test sensitivity: determine if this individual tests positive
    if !individual.test_positive
        return nothing
    end

    # Compute isolation time = onset + delay
    iso_delay = rand(state.rng, iso.delay)
    iso_time = individual.onset_time + iso_delay

    individual.isolated = true
    individual.isolation_time = iso_time

    return nothing
end
