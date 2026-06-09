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
include("state_accessors.jl")
include("options.jl")
include("distributions.jl")
include("utils.jl")

# Intervention interface (must come before models that use it)
include("interventions/interface.jl")
include("interventions/isolation.jl")
include("interventions/contact_tracing.jl")
include("interventions/vaccination.jl")
include("interventions/scheduled.jl")

# Clinical transitions — case-state Markov chain layered on the
# intervention framework. Same hook shape as interventions; sibling
# abstract type so the public API can keep `interventions=` and
# `transitions=` namespaces distinct.
include("transitions/interface.jl")
include("transitions/reporting.jl")
include("transitions/hospitalisation.jl")
include("transitions/outcome.jl")

# Public API declarations (Julia 1.11+)
@static if VERSION >= v"1.11"
    include("public.jl")
end

# Transmission models
include("models/branching_process.jl")
include("models/network_process.jl")

# Observation models (state-space slot)
include("observation_models.jl")

# Observation models (wrappers around TransmissionModel)
include("observation.jl")

# Simulation engine
include("simulation.jl")

# Network process needs the engine helpers (_decide_infected,
# _resolve_transitions!, …), so its simulate loop is included after.
include("models/network_simulate.jl")

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
include("analytical/end_of_outbreak_probability.jl")

# Distribution wrappers so models work directly with Turing's `~`.
include("likelihood_dists.jl")

# Exports — types
export TransmissionModel, BranchingProcess, NetworkProcess
export Individual, SimulationState
export SimOpts
export AbstractStoppingRule, Extinction, MaxCases, MaxGenerations, MaxTime, should_stop
export AbstractIntervention, Isolation, ContactTracing
export IsolationEligibility, SymptomaticOnly, AllCases
export is_eligible_for_isolation
export TraceEligibility, AlwaysEligible, SymptomaticParent
export OnSymptomOnset, OnLabConfirmation, OnIsolation, TraceEveryone, TraceNobody, NoTracing
export AnyOf, AllOf, NoneOf
export TraceRate, ConstantRate
export TraceDelay, ConstantDelay
export TraceAction, Quarantine, FlagOnly
export AbstractVaccination, RingVaccination, MassVaccination
export AbstractEffectMode, LeakyMode, AllOrNothingMode
export Scheduled, Risk
export is_active, intervention_time
export AbstractClinicalTransition, Reporting, Hospitalisation, Death, Recovery
export is_terminal, terminal_event
export ObservationModel, PerCaseObservation, Observed
export single_type_offspring

# Exports — sentinel types
export NoPopulation, NoAttributes, NoTypeLabels
export NoAgeDistribution, NoCases

# Exports — accessors
export onset_time, is_isolated, isolation_time, is_traced, is_quarantined
export is_vaccinated, is_asymptomatic, is_test_positive, is_infected
export individual_type, set_isolated!

# Exports — distributions
export NegBin, scale_distribution, incubation_linked_generation_time
# Process-side chain-size distributions (closed forms for offspring specs)
export Borel, GammaBorel, PoissonGammaChainSize, chain_size_distribution
# Observation-side chain-size distribution (binomial thinning of any base)
export ThinnedChainSize

# Exports — attributes functions
export clinical_presentation, demographics, transmission_traits, compose

# Exports — simulation
export simulate, make_contact!, susceptible_fraction

# Exports — output
export linelist, contacts, chain_statistics
export containment_probability, is_extinct, generation_R, weekly_incidence, scenario_sweep

# Exports — analytical
export extinction_probability, epidemic_probability
export proportion_transmission, proportion_cluster_size, network_R
export probability_contain

# Exports — unified inference interface
export OffspringCounts, ChainSizes, ChainLengths
# Entry points returning a Distribution that wraps a model; use with
# Turing's `~`. `chain_size_distribution` is already exported above.
export chain_length_distribution, offspring_distribution
export ClusterMixed, ChainSizeMixture
# Real-time mixture: per-cluster "is finished?" weight
export end_of_outbreak_probability

end # module
