"""
EpiBranchCore — protocol layer for the EpiBranch.jl ecosystem.

Owns the data types, abstract types, hook protocols, distribution
helpers, and state accessors that every slot-in package depends on.
Deliberately thin: the simulator lives in `EpiBranchProcess`;
concrete interventions, transitions, observations, output, and
analytics each live in their own package and add methods to the
generics declared here.

A downstream package adding a new intervention, transition,
observation model, or transmission model depends only on this
package plus the engine (`EpiBranchProcess`) if it needs to run
simulations.
"""
module EpiBranchCore

using Distributions
using DocStringExtensions
using Random

# Docstring templates (used elsewhere in the package surface)
include("docstrings.jl")

# Data and abstract types
include("types.jl")

# State accessors
include("state_accessors.jl")

# Distribution helpers (NegBin, incubation_linked_generation_time,
# _sample_value, scale_distribution)
include("distributions.jl")
include("utils.jl")

# Protocol layer
include("intervention_interface.jl")
include("transition_interface.jl")
include("observation_interface.jl")

# Cross-package extension contract — empty generics that more than one
# slot-in package extends.
include("extension_contract.jl")

# ── Exports ─────────────────────────────────────────────────────────

# Types
export Individual, SimulationState, TransmissionModel

# Sentinels
export NoPopulation, NoAttributes, NoTypeLabels
export NoAgeDistribution, NoCases, NoGenerationTime

# Abstract types and protocol structs
export AbstractIntervention, AbstractClinicalTransition, ObservationModel
export Risk

# TransmissionModel interface (model-level metadata)
export population_size, latent_period, n_types, single_type_offspring

# Intervention hook protocol
export initialise_individual!, resolve_individual!, apply_post_transmission!
export competing_risk, is_active, intervention_time, reset!, required_fields

# Transition hook protocol
export is_terminal, terminal_event

# Generic clinical / engine accessors
export onset_time, is_asymptomatic, is_infected, individual_type

# Intervention-specific accessors (live here until slot-in packages
# claim ownership)
export is_isolated, isolation_time, set_isolated!
export is_traced, is_quarantined, is_vaccinated, is_test_positive

# Distribution helpers
export NegBin, scale_distribution, incubation_linked_generation_time

# Engine and analytics seams (concrete methods in downstream packages)
export simulate, simulate_batch, step!, make_contact!, draw_offspring
export chain_size_distribution

end # module
