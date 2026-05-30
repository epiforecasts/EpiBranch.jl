"""
Foundation: data types, abstract types, the intervention and
transition hook protocols, and the engine/analytics extension
contracts. Everything every other submodule depends on.

`Base` is deliberately small. Concrete intervention seam traits,
distribution helpers, stopping rules, attribute builders, and the
accessors for intervention-specific state keys all live in the
submodule that owns them — not here.
"""
module EpiBranchBase

using Distributions
using Random

# ── Cross-submodule extension contracts ─────────────────────────────
# Empty generics; downstream submodules `import ..EpiBranchBase: <name>`
# and add methods. Declaring them here lets user code do
# `EpiBranch.step!(::MyModel, …) = …` (etc.) and extend the canonical
# method table rather than create a namespace-local stand-in.
function step! end
function make_contact! end
function chain_size_distribution end

# Data and abstract types
export Individual, SimulationState
export TransmissionModel
export AbstractClinicalTransition, AbstractIntervention, ObservationModel, Risk
export NoPopulation, NoAttributes, NoTypeLabels, NoGenerationTime
export NoAgeDistribution, NoCases

# Engine-owned and generic clinical accessors
export is_infected, individual_type
export onset_time, is_asymptomatic

# Process interface (defaults dispatched on `TransmissionModel`)
export population_size, latent_period, n_types, single_type_offspring

# Intervention hook protocol
export initialise_individual!, resolve_individual!, apply_post_transmission!
export competing_risk, is_active, intervention_time, reset!, required_fields

# Transition hook protocol
export is_terminal, terminal_event

# Engine + analytics extension contracts
export step!, make_contact!, chain_size_distribution

include("types.jl")
include("state_accessors.jl")
include("distributions.jl")
include("intervention_interface.jl")
include("transition_interface.jl")
include("observation_interface.jl")

end
