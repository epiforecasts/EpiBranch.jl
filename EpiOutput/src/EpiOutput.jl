"""
EpiOutput — extract analyses, line lists, and summaries from
completed `EpiBranch` simulations.

Owns `linelist`, `contacts`, `chain_statistics`, plus the summary
helpers `containment_probability`, `is_extinct`, `generation_R`,
`weekly_incidence`, and `scenario_sweep`.

Operates on `SimulationState` (from `EpiBranchCore`) and reads
intervention/transition/observation keys written into
`Individual.state` by the relevant slot-in packages.
"""
module EpiOutput

using DataFrames
using Dates
using DocStringExtensions
using Random
using EpiBranchCore
using EpiBranchProcess

include("linelist.jl")
include("chains.jl")
include("summary.jl")

# ── Exports ─────────────────────────────────────────────────────────

export linelist, contacts, chain_statistics
export containment_probability, is_extinct, generation_R
export weekly_incidence, scenario_sweep

end # module
