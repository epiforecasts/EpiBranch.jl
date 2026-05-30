"""
The simulation engine plus the canonical `BranchingProcess` model.
Owns `simulate`, `simulate_batch`, `step!`, `make_contact!`,
stopping rules, attribute builders, and distribution helpers. A
custom `TransmissionModel` extends `step!` and uses `make_contact!`.
"""
module Engine

using Distributions
using Random
using ..EpiBranchBase

# Internal helpers from Base
using ..EpiBranchBase: _sample_value

# Terminal-arbitration helper from Transitions
using ..Transitions: _finalise_terminal!

# Generics from Base that we add methods to.
import ..EpiBranchBase: population_size, latent_period, n_types,
                        required_fields, initialise_individual!,
                        resolve_individual!, apply_post_transmission!,
                        competing_risk, step!, make_contact!

# Models
export BranchingProcess

# Simulation
export simulate, simulate_batch, make_contact!

# Attributes
export clinical_presentation, demographics, transmission_traits, compose

# Stopping rules
export SimOpts
export AbstractStoppingRule, Extinction, MaxCases, MaxGenerations, MaxTime
export should_stop

# Distribution helpers
export NegBin, scale_distribution, incubation_linked_generation_time

include("distributions.jl")
include("options.jl")
include("branching_process.jl")
include("simulation.jl")

end
