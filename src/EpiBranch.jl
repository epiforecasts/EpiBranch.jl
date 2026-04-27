module EpiBranch

using DataFrames
using Dates
using Distributions
using QuadGK
using Random
using SpecialFunctions

# Docstring templates (must come before any docstrings)
include("docstrings.jl")

# Core types
include("types.jl")
include("options.jl")
include("distributions.jl")
include("utils.jl")

# Intervention interface (must come before models that use it)
include("interventions/interface.jl")
include("interventions/isolation.jl")
include("interventions/contact_tracing.jl")
include("interventions/vaccination.jl")
include("interventions/scheduled.jl")

# Public API declarations (Julia 1.11+)
@static if VERSION >= v"1.11"
    include("public.jl")
end

# Transmission models
include("models/branching_process.jl")

# Observation models (wrappers around TransmissionModel)
include("observation.jl")

# Simulation engine
include("simulation.jl")

# Output
include("output/linelist.jl")
include("output/chains.jl")
include("output/summary.jl")

# Analytical
include("analytical/extinction.jl")
include("analytical/chain_distributions.jl")
include("analytical/data_types.jl")
include("analytical/likelihood.jl")
include("analytical/superspreading.jl")
include("analytical/fitting.jl")
include("analytical/cluster_mixed.jl")
include("analytical/real_time.jl")

# Exports — types
export TransmissionModel, BranchingProcess
export Individual, SimulationState
export SimOpts, DelayOpts, OutcomeOpts, DemographicOpts
export AbstractIntervention, Isolation, ContactTracing, RingVaccination, Scheduled
export is_active, intervention_time
export PartiallyObserved

# Exports — sentinel types
export NoPopulation, NoAttributes, NoTypeLabels, NoDelay, NoCFR
export NoAgeDistribution, NoOutcomes, NoDemographics, NoCases

# Exports — accessors
export onset_time, is_isolated, isolation_time, is_traced, is_quarantined
export is_vaccinated, is_asymptomatic, is_test_positive, is_infected
export individual_type, set_isolated!

# Exports — distributions
export NegBin, scale_distribution, ringbp_generation_time
export Borel, GammaBorel, PoissonGammaChainSize, ThinnedChainSize, chain_size_distribution

# Exports — attributes functions
export Disease, clinical_presentation, demographics, compose

# Exports — simulation
export simulate, simulate_batch

# Exports — output
export linelist, contacts, chain_statistics
export containment_probability, is_extinct, generation_R, weekly_incidence, scenario_sweep

# Exports — analytical
export extinction_probability, epidemic_probability
export proportion_transmission, proportion_cluster_size, network_R
export probability_contain

# Exports — unified inference interface
export OffspringCounts, ChainSizes, ChainLengths
export ClusterMixed, ChainSizeMixture
export RealTimeChainSizes, end_of_outbreak_probability

end # module
