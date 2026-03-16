"""
    SimOpts(; kwargs...)

Options controlling simulation termination and setup.
"""
Base.@kwdef struct SimOpts
    max_cases::Int = 10_000
    max_generations::Int = 100
    max_time::Float64 = Inf
    n_initial::Int = 1
    incubation_period::Union{Distribution, Nothing} = nothing
    prob_asymptomatic::Float64 = 0.0
    asymptomatic_R_scaling::Float64 = 1.0
    test_sensitivity::Float64 = 1.0
    latent_period::Float64 = 0.0
end

# ── Phase 2: Output options ──────────────────────────────────────────

"""
    DelayOpts(; kwargs...)

Distributions for delays used in line list generation.
"""
Base.@kwdef struct DelayOpts
    onset_to_reporting::Union{Distribution, Nothing} = nothing
    onset_to_admission::Union{Distribution, Nothing} = nothing
    onset_to_outcome::Union{Distribution, Nothing} = nothing
end

"""
    OutcomeOpts(; kwargs...)

Parameters controlling case outcomes (hospitalisation and death).
"""
Base.@kwdef struct OutcomeOpts
    prob_hospitalisation::Float64 = 0.2
    prob_death::Float64 = 0.05
    age_specific_cfr::Union{Dict{Tuple{Int,Int}, Float64}, Nothing} = nothing
end

"""
    DemographicOpts(; kwargs...)

Parameters for generating demographic data (age, sex).
"""
Base.@kwdef struct DemographicOpts
    age_distribution::Union{Distribution, Nothing} = nothing
    age_range::Tuple{Int, Int} = (0, 90)
    prob_female::Float64 = 0.5
end
