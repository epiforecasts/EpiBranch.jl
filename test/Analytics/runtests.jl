using Test
using EpiBranch
using Distributions
using Random
using StableRNGs

# Sim ↔ analytical consistency helper, used by test_analytical.jl.
include(joinpath(@__DIR__, "..", "testutils", "sim_analytical_consistency.jl"))

@testset "Analytics" begin
    include("test_analytical.jl")
    include("test_r_targets.jl")
end
