using Test
using EpiBranch
using Distributions
using Random
using StableRNGs

@testset "Transitions" begin
    include("test_transitions.jl")
end
