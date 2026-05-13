"""
Options controlling simulation termination and setup. Contains only
simulation control parameters — clinical and demographic properties
are set via `attributes` functions, [`AbstractClinicalTransition`](@ref)s,
and interventions.
"""
Base.@kwdef struct SimOpts
    max_cases::Int = 10_000
    max_generations::Int = 100
    max_time::Float64 = Inf
    n_initial::Int = 1
end
