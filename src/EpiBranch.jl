"""
EpiBranch.jl — branching-process simulation and inference for
infectious-disease outbreaks.

Organised into seven submodules. `using EpiBranch` brings the public
surface of all of them into scope, so most users do not need to import
submodules directly.

- `EpiBranchBase`: data types (`Individual`, `SimulationState`,
  `Risk`), abstract types (`TransmissionModel`, `AbstractIntervention`,
  `AbstractClinicalTransition`, `ObservationModel`), hook protocols,
  and engine/analytics extension contracts.
- `Engine`: the simulation engine, `BranchingProcess`, stopping rules,
  attribute builders, distribution helpers.
- `Interventions`: `Isolation`, `ContactTracing`, vaccinations,
  `Scheduled` + the per-intervention accessors and seam traits.
- `Transitions`: `Reporting`, `Hospitalisation`, `Death`, `Recovery`.
- `Observation`: `PerCaseObservation`, `Observed`, `ThinnedChainSize`.
- `Output`: DataFrame projections (`linelist`, `contacts`,
  `chain_statistics`, summary helpers).
- `Analytics`: chain-size distributions, likelihoods, fitting,
  end-of-outbreak probability.
"""
module EpiBranch

include("docstrings.jl")

# Submodules — include in dependency order.
include("EpiBranchBase/EpiBranchBase.jl")
include("Interventions/Interventions.jl")
include("Transitions/Transitions.jl")
include("Engine/Engine.jl")
include("Observation/Observation.jl")
include("Output/Output.jl")
include("Analytics/Analytics.jl")

# Bring each submodule's public surface into the top-level namespace.
using .EpiBranchBase
using .Interventions
using .Transitions
using .Engine
using .Observation
using .Output
using .Analytics

# ── Re-export the public surface ────────────────────────────────────
# Mirror of each submodule's `export` block so `using EpiBranch` is a
# drop-in replacement for the previous flat module.

# Base — protocol layer
export TransmissionModel
export Individual, SimulationState
export NoPopulation, NoAttributes, NoTypeLabels
export NoAgeDistribution, NoCases, NoGenerationTime
export AbstractClinicalTransition, AbstractIntervention, ObservationModel, Risk
export is_infected, individual_type, onset_time, is_asymptomatic
export population_size, latent_period, n_types, single_type_offspring
export initialise_individual!, resolve_individual!, apply_post_transmission!
export competing_risk, is_active, intervention_time, reset!, required_fields
export is_terminal, terminal_event
export step!, make_contact!, chain_size_distribution

# Engine
export BranchingProcess
export simulate, simulate_batch
export clinical_presentation, demographics, transmission_traits, compose
export SimOpts
export AbstractStoppingRule, Extinction, MaxCases, MaxGenerations, MaxTime
export should_stop
export NegBin, scale_distribution, incubation_linked_generation_time

# Interventions
export Isolation, ContactTracing
export IsolationEligibility, SymptomaticOnly, AllCases
export TraceEligibility, AlwaysEligible, SymptomaticParent
export TraceRate, ConstantRate
export TraceDelay, ConstantDelay
export TraceAction, Quarantine, FlagOnly
export AbstractVaccination, RingVaccination, MassVaccination
export AbstractEffectMode, LeakyMode, AllOrNothingMode
export Scheduled
export is_isolated, isolation_time, set_isolated!, is_test_positive
export is_traced, is_quarantined, is_vaccinated
export is_eligible, is_eligible_for_isolation
export traces, draw_trace_delay, apply_trace!
export required_for_eligibility, required_for_ct_eligibility

# Transitions
export Reporting, Hospitalisation, Death, Recovery

# Observation
export PerCaseObservation, Observed, ThinnedChainSize, scalar_detection_prob

# Output
export linelist, contacts, chain_statistics
export containment_probability, is_extinct, generation_R, weekly_incidence
export scenario_sweep

# Analytics
export Borel, GammaBorel, PoissonGammaChainSize
export OffspringCounts, ChainSizes, ChainLengths
export ClusterMixed, ChainSizeMixture
export extinction_probability, epidemic_probability
export proportion_transmission, proportion_cluster_size, network_R
export probability_contain
export end_of_outbreak_probability

end # module
