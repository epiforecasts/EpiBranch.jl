"""
Options controlling simulation termination and setup. Contains only
simulation control parameters -- clinical and demographic properties
are set via `attributes` functions and interventions.
"""
Base.@kwdef struct SimOpts
    max_cases::Int = 10_000
    max_generations::Int = 100
    max_time::Float64 = Inf
    n_initial::Int = 1
end

# ── Output options ───────────────────────────────────────────────────

"""
Distributions for delays used in line list generation.
"""
Base.@kwdef struct DelayOpts{R, A, O}
    onset_to_reporting::R = NoDelay()
    onset_to_admission::A = NoDelay()
    onset_to_outcome::O = NoDelay()
end

"""
Parameters controlling case outcomes (hospitalisation and death).
"""
Base.@kwdef struct OutcomeOpts{C}
    prob_hospitalisation::Float64 = 0.2
    prob_death::Float64 = 0.05
    age_specific_cfr::C = NoCFR()
end

"""
Parameters for generating demographic data (age, sex).
"""
Base.@kwdef struct DemographicOpts{D}
    age_distribution::D = NoAgeDistribution()
    age_range::Tuple{Int, Int} = (0, 90)
    prob_female::Float64 = 0.5
end
