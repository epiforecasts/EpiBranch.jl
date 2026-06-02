"""
EpiBranchAnalytics — chain-size distributions, likelihoods, fitting, and
end-of-outbreak probability for `EpiBranch`.

Owns the closed-form chain-size distributions (`Borel`, `GammaBorel`,
`PoissonGammaChainSize`), `ClusterMixed` (Gamma-Borel-style chain-size
mixture), the data wrappers (`OffspringCounts`, `ChainSizes`,
`ChainLengths`), `fit` for offspring distributions, and the
superspreading helpers (`proportion_transmission`,
`proportion_cluster_size`, `network_R`).

Extends `EpiBranchCore.chain_size_distribution` (closed-form chain
sizes for Poisson/NegativeBinomial offspring and the `ClusterMixed`
offspring spec), `EpiBranchCore.draw_offspring` (sampling
`ClusterMixed` chains), and `Distributions.loglikelihood` (analytical
and simulation-based likelihoods).
"""
module EpiBranchAnalytics

using Distributions
using DocStringExtensions
using QuadGK
using Random
using SpecialFunctions
using EpiBranchCore
using EpiBranchCore: _sample_value
using EpiBranchProcess
using EpiBranchProcess: max_cases
using EpiBranchObservation
using EpiBranchOutput: chain_statistics

include("data_types.jl")
include("chain_distributions.jl")
include("cluster_mixed.jl")
include("extinction.jl")
include("superspreading.jl")
include("fitting.jl")
include("likelihood.jl")
include("end_of_outbreak_probability.jl")

# ── Exports ─────────────────────────────────────────────────────────

# Closed-form chain-size distributions
export Borel, GammaBorel, PoissonGammaChainSize

# Offspring spec — cluster-level heterogeneity
export ClusterMixed, ChainSizeMixture

# Data wrappers for the unified inference interface
export OffspringCounts, ChainSizes, ChainLengths

# Analytical helpers
export extinction_probability, epidemic_probability, probability_contain
export end_of_outbreak_probability
export proportion_transmission, proportion_cluster_size, network_R

end # module
