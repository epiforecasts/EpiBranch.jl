"""
The simulation engine plus the canonical `BranchingProcess` model.
Owns `simulate`, `simulate_batch`, `step!`, `make_contact!`,
competing-risks resolution, and attribute builders. A custom
`TransmissionModel` extends `step!` and uses `make_contact!`.
"""
module EpiBranchEngine

using Distributions
using Random
using ..EpiBranchBase

# Internal helpers + types from Base
using ..EpiBranchBase: _sample_value, _finalise_terminal!

# Generics from Base that we add methods to.
import ..EpiBranchBase: population_size, latent_period, n_types,
                        required_fields, initialise_individual!,
                        resolve_individual!, apply_post_transmission!,
                        competing_risk, step!

# Models
export BranchingProcess

# Simulation
export simulate, simulate_batch, make_contact!

# Attributes
export clinical_presentation, demographics, transmission_traits, compose

include("branching_process.jl")
include("simulation.jl")

end
