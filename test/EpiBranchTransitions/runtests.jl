using Test
using EpiBranch
using Distributions
using Random
using StableRNGs

@testset "EpiBranchTransitions" begin
    include("test_transitions.jl")
end
