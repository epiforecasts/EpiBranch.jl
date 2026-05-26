"""
Concrete observation models and the `Observed` wrapper. Extends
`simulate` (from `EpiBranchEngine`) and `chain_size_distribution`
(declared in `EpiBranchBase`) so that observation models compose with
the simulation and analytical paths through dispatch.
"""
module EpiBranchObservation

using Distributions
using Random
using ..EpiBranchBase
using ..EpiBranchEngine
import ..EpiBranchBase: chain_size_distribution, single_type_offspring,
                        population_size, latent_period, n_types
import ..EpiBranchEngine: simulate

# Internal helpers from Base
using ..EpiBranchBase: _sample_value

export PerCaseObservation, Observed
export ThinnedChainSize
export scalar_detection_prob

include("observation_models.jl")
include("observation.jl")

end
