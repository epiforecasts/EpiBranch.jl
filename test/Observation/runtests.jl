using Test
using EpiBranch
using Distributions
using Random
using StableRNGs

@testset "Observation" begin
    include("test_pc_observation.jl")
end
