"""
EpiBranch.jl — branching-process simulation and inference for
infectious-disease outbreaks.

The package is organised into seven submodules. `using EpiBranch` brings
the public surface of all of them into scope so most users do not need
to import submodules directly.

- [`EpiBranchBase`](@ref EpiBranch.EpiBranchBase): types, sentinels,
  state accessors, options, the intervention / transition / observation
  interfaces.
- [`EpiBranchInterventions`](@ref EpiBranch.EpiBranchInterventions):
  `Isolation`, `ContactTracing`, vaccination, `Scheduled`.
- [`EpiBranchTransitions`](@ref EpiBranch.EpiBranchTransitions):
  `Reporting`, `Hospitalisation`, `Death`, `Recovery`.
- [`EpiBranchEngine`](@ref EpiBranch.EpiBranchEngine): the simulation
  engine and the canonical `BranchingProcess` model.
- [`EpiBranchObservation`](@ref EpiBranch.EpiBranchObservation):
  observation models and the `Observed` wrapper.
- [`EpiBranchOutput`](@ref EpiBranch.EpiBranchOutput): DataFrame
  projections (line list, contacts, chain statistics).
- [`EpiBranchAnalytics`](@ref EpiBranch.EpiBranchAnalytics): chain-size
  distributions, likelihoods, fitting, end-of-outbreak probability.
"""
module EpiBranch

include("docstrings.jl")

# Submodules — include in dependency order.
include("EpiBranchBase/EpiBranchBase.jl")
include("EpiBranchInterventions/EpiBranchInterventions.jl")
include("EpiBranchTransitions/EpiBranchTransitions.jl")
include("EpiBranchEngine/EpiBranchEngine.jl")
include("EpiBranchObservation/EpiBranchObservation.jl")
include("EpiBranchOutput/EpiBranchOutput.jl")
include("EpiBranchAnalytics/EpiBranchAnalytics.jl")

# Bring each submodule's public surface into the top-level namespace.
using .EpiBranchBase
using .EpiBranchInterventions
using .EpiBranchTransitions
using .EpiBranchEngine
using .EpiBranchObservation
using .EpiBranchOutput
using .EpiBranchAnalytics

# ── Re-export the public surface ────────────────────────────────────
# Mirror of each submodule's `export` block so `using EpiBranch` is a
# drop-in replacement for the previous flat module.

# Base
export TransmissionModel
export Individual, SimulationState
export NoPopulation, NoAttributes, NoTypeLabels
export NoAgeDistribution, NoCases, NoGenerationTime
export AbstractClinicalTransition
export onset_time, is_isolated, isolation_time, is_traced, is_quarantined
export is_vaccinated, is_asymptomatic, is_test_positive, is_infected
export individual_type, set_isolated!
export population_size, latent_period, n_types, single_type_offspring
export SimOpts
export AbstractStoppingRule, Extinction, MaxCases, MaxGenerations, MaxTime
export should_stop
export NegBin, scale_distribution, incubation_linked_generation_time
export AbstractIntervention, Risk
export initialise_individual!, resolve_individual!, apply_post_transmission!
export competing_risk, is_active, intervention_time, reset!, required_fields
export is_terminal, terminal_event
export ObservationModel

# Interventions
export Isolation, ContactTracing
export IsolationEligibility, SymptomaticOnly, AllCases
export is_eligible_for_isolation
export TraceEligibility, AlwaysEligible, SymptomaticParent
export TraceRate, ConstantRate
export TraceDelay, ConstantDelay
export TraceAction, Quarantine, FlagOnly
export AbstractVaccination, RingVaccination, MassVaccination
export AbstractEffectMode, LeakyMode, AllOrNothingMode
export Scheduled

# Transitions
export Reporting, Hospitalisation, Death, Recovery

# Engine
export BranchingProcess
export simulate, simulate_batch, make_contact!
export clinical_presentation, demographics, transmission_traits, compose

# Observation
export PerCaseObservation, Observed, ThinnedChainSize

# Output
export linelist, contacts, chain_statistics
export containment_probability, is_extinct, generation_R, weekly_incidence
export scenario_sweep

# Analytics
export Borel, GammaBorel, PoissonGammaChainSize, chain_size_distribution
export OffspringCounts, ChainSizes, ChainLengths
export ClusterMixed, ChainSizeMixture
export extinction_probability, epidemic_probability
export proportion_transmission, proportion_cluster_size, network_R
export probability_contain
export end_of_outbreak_probability

end # module
