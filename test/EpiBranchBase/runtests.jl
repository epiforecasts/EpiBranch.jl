using Test
using EpiBranch
using Distributions
using Random
using StableRNGs

@testset "EpiBranchBase" begin
    include("test_types.jl")
    include("test_attributes.jl")
end
