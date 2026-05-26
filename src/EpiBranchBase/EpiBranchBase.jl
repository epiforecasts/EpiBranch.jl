"""
Foundation: types, sentinels, state accessors, options/stopping rules,
distributions, interfaces (intervention, transition, observation).
Every downstream submodule depends on this one.
"""
module EpiBranchBase

using Distributions
using Random
using SpecialFunctions

# Cross-submodule generic functions: declared here as empty multimethods
# so that downstream submodules add methods via `import ..EpiBranchBase:
# <name>` rather than creating namespace-local stand-ins. Each is also
# exported below, which means `EpiBranch.x(::MyType) = …` from user code
# extends the canonical method table.
function chain_size_distribution end
function step! end
function is_eligible end
function is_eligible_for_isolation end
function traces end
function draw_trace_delay end
function apply_trace! end
function required_for_eligibility end
function required_for_ct_eligibility end
function scalar_detection_prob end

# Core types and sentinels
export Individual, SimulationState, TransmissionModel
export AbstractClinicalTransition
export NoPopulation, NoAttributes, NoTypeLabels
export NoAgeDistribution, NoCases, NoGenerationTime

# State accessors
export onset_time, is_isolated, isolation_time, is_traced, is_quarantined
export is_vaccinated, is_asymptomatic, is_test_positive, is_infected
export individual_type, set_isolated!

# Process accessors / interface defaults
export population_size, latent_period, n_types, single_type_offspring

# Stopping rules and options
export SimOpts
export AbstractStoppingRule, Extinction, MaxCases, MaxGenerations, MaxTime
export should_stop

# Distributions helpers
export NegBin, scale_distribution, incubation_linked_generation_time

# Intervention interface
export AbstractIntervention, Risk
export initialise_individual!, resolve_individual!, apply_post_transmission!
export competing_risk, is_active, intervention_time, reset!, required_fields

# Intervention extension points (seam traits)
export is_eligible, is_eligible_for_isolation
export traces, draw_trace_delay, apply_trace!
export required_for_eligibility, required_for_ct_eligibility

# Engine extension point
export step!

# Observation extension point
export scalar_detection_prob

# Chain-size distribution generic (extended in Observation and Analytics)
export chain_size_distribution

# Transition interface
export is_terminal, terminal_event

# Observation interface (just the abstract type lives in Base for
# extensibility; concrete types live in EpiBranchObservation).
export ObservationModel

include("types.jl")
include("state_accessors.jl")
include("distributions.jl")
include("utils.jl")
include("options.jl")
include("intervention_interface.jl")
include("transition_interface.jl")
include("observation_interface.jl")

end
