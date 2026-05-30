"""
Post-simulation projections: `linelist`, `contacts`, `chain_statistics`,
and summary helpers. All operate on `SimulationState` and produce
DataFrames or scalar summaries.
"""
module Output

using DataFrames
using Dates
using Distributions
using Random
using ..EpiBranchBase
using ..Engine

export linelist, contacts, chain_statistics
export containment_probability, is_extinct, generation_R, weekly_incidence
export scenario_sweep

include("linelist.jl")
include("chains.jl")
include("summary.jl")

end
