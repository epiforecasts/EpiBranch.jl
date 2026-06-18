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
include("timing.jl")
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
include("transitions/transition.jl")
include("transitions/reporting.jl")
include("transitions/hospitalisation.jl")
include("transitions/outcome.jl")

"""
    generate_offspring(model, parent, state) -> count

The offspring-driven transmission seam: how many contacts `parent` makes
this generation, as a single count (single-type) or a count per type
(multi-type). The engine calls it once per active parent, then creates
that many fresh contacts and assigns each an infection time from the
model's `generation_time` — so the model builds no `Individual`s, assigns
no timing, and takes no `interventions` argument.

This is the path for tree-like models (a branching process), where every
contact is fresh. Structure-driven models whose contacts are existing
nodes (a contact network, say) define [`contacts_of`](@ref) and
override [`collect_exposures`](@ref) instead.
"""
function generate_offspring end

# Public API declarations (Julia 1.11+)
@static if VERSION >= v"1.11"
    include("public.jl")
end

# Observation models (state-space slot) and the model inputs
# (interventions, attributes, observation) that every process carries.
# Defined before the process types so their constructors can store them.
include("observation_models.jl")
include("model_inputs.jl")

# Transmission models
include("models/branching_process.jl")

# Observation helpers (thinned chain-size distribution, dispatch on the
# model's observation model)
include("observation.jl")

# Simulation engine
include("simulation.jl")

# Output
include("output/linelist.jl")
include("output/tracing.jl")
include("output/chains.jl")
include("output/generation_intervals.jl")
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
export TransmissionModel, BranchingProcess, Infectiousness
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
export AbstractClinicalTransition, Transition, Reporting, Hospitalisation, Death, Recovery
export is_terminal, terminal_event
export ObservationModel, PerCaseObservation, NoObservation, observe
export single_type_offspring

# Exports — sentinel types
export NoPopulation, NoAttributes, NoTypeLabels
export NoAgeDistribution, NoCases

# Exports — accessors
export onset_time, incubation_period, is_isolated, isolation_time, is_traced, is_quarantined
export is_vaccinated, is_asymptomatic, is_test_positive, is_infected
export individual_type, set_isolated!

# Exports — distributions
export NegBin, scale_distribution, incubation_linked_generation_time
# Process-side chain-size distributions (closed forms for offspring specs).
# Reach `GammaBorel`/`PoissonGammaChainSize` via `chain_size_distribution`;
# only the standalone `Borel` and the dispatch entry point are exported.
export Borel, chain_size_distribution
# Observation-side chain-size distribution (binomial thinning of any base)
export ThinnedChainSize

# Exports — attributes functions
export clinical_presentation, demographics, transmission_traits, compose

# Exports — simulation
export simulate, make_contact!, susceptible_fraction
export generate_offspring, contacts_of, collect_exposures, gather_by_target

# Exports — output
export linelist, contacts, chain_statistics, compute_trace_level!
export realised_generation_interval, realised_generation_intervals
export containment_probability, is_extinct, generation_R, weekly_incidence, scenario_sweep

# Exports — analytical
export extinction_probability, epidemic_probability
export proportion_transmission, proportion_cluster_size, heterogeneous_contact_R
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
