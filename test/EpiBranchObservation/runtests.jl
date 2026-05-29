using Test
using EpiBranch
using Distributions
using Random
using StableRNGs

@testset "EpiBranchObservation" begin
    include("test_pc_observation.jl")
end
