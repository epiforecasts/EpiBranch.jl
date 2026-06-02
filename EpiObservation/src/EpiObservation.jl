"""
EpiObservation — observation models for `EpiBranch`.

Owns `PerCaseObservation`, the `Observed{P, O}` wrapper that
combines a `TransmissionModel` with an `ObservationModel`, and
`ThinnedChainSize` (binomial thinning of a chain-size distribution).

Extends `simulate` and `chain_size_distribution` from `EpiBranchCore`
so that an `Observed` model plugs straight into the engine and into
the analytical chain-size machinery.
"""
module EpiObservation

using Distributions
using DocStringExtensions
using Random
using EpiBranchCore
using EpiBranchCore: _sample_value

include("observation_models.jl")
include("observation.jl")

# ── Exports ─────────────────────────────────────────────────────────

export PerCaseObservation, Observed, ThinnedChainSize
export scalar_detection_prob

end # module
