module EpiBranch

using DataFrames
using Dates
using Distributions
using Random
using SpecialFunctions

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

# Transmission models
include("models/branching_process.jl")

# Simulation engine
include("simulation.jl")

# Output
include("output/linelist.jl")
include("output/chains.jl")
include("output/summary.jl")

# Analytical
include("analytical/extinction.jl")
include("analytical/chain_distributions.jl")
include("analytical/likelihood.jl")
include("analytical/superspreading.jl")

# Exports — types
export TransmissionModel, BranchingProcess
export Individual, SimulationState
export SimOpts, DelayOpts, OutcomeOpts, DemographicOpts
export AbstractIntervention, Isolation, ContactTracing, RingVaccination

# Exports — accessors
export onset_time, is_isolated, isolation_time, is_traced, is_quarantined
export is_vaccinated, is_asymptomatic, is_test_positive, is_infected
export individual_type, set_isolated!

# Exports — distributions
export NegBin, scale_distribution, ringbp_generation_time
export Borel, GammaBorel, chain_size_distribution

# Exports — attributes functions
export clinical_presentation, testing, demographics, compose

# Exports — simulation
export simulate, simulate_batch, simulate_conditioned

# Exports — output
export linelist, contacts, chain_statistics
export containment_probability, is_extinct, generation_R, weekly_incidence

# Exports — analytical
export extinction_probability, epidemic_probability
export chain_size_ll, chain_length_ll
export proportion_transmission

end # module
