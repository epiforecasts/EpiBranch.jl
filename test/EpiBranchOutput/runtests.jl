using Test
using EpiBranch
using Distributions
using Random
using StableRNGs

@testset "EpiBranchOutput" begin
    include("test_linelist.jl")
    include("test_chains.jl")
end
