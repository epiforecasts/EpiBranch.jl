using Test
using EpiBranch
using Distributions
using Random
using StableRNGs

@testset "Output" begin
    include("test_linelist.jl")
    include("test_chains.jl")
end
