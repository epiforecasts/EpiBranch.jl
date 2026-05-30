"""
Analytical helpers: chain-size and chain-length likelihoods, fitting,
extinction / epidemic / containment probabilities, end-of-outbreak
probability, superspreading helpers, cluster-mixed compound
distributions. Extends `chain_size_distribution` from `EpiBranchBase`,
`loglikelihood` and `fit` from `Distributions`.
"""
module Analytics

using Distributions
using QuadGK
using Random
using SpecialFunctions
using ..EpiBranchBase
using ..Engine
using ..Observation
using ..Output: chain_statistics
import ..EpiBranchBase: chain_size_distribution, single_type_offspring
import ..Engine: BranchingProcess, _draw_offspring
import Distributions: loglikelihood, fit

# Internal helpers from Engine / Observation
using ..Engine: max_cases
using ..Observation: scalar_detection_prob

# Process-side chain-size distributions
export Borel, GammaBorel, PoissonGammaChainSize
export chain_size_distribution

# Data wrappers
export OffspringCounts, ChainSizes, ChainLengths

# Cluster mixing
export ClusterMixed, ChainSizeMixture

# Likelihood and inference helpers
export extinction_probability, epidemic_probability
export proportion_transmission, proportion_cluster_size, network_R
export probability_contain
export end_of_outbreak_probability

# Include order matters: extinction + data types + chain distributions
# first, then likelihoods/fitting that depend on them, then EOO.
include("extinction.jl")
include("chain_distributions.jl")
include("data_types.jl")
include("likelihood.jl")
include("superspreading.jl")
include("fitting.jl")
include("cluster_mixed.jl")
include("end_of_outbreak_probability.jl")

end
