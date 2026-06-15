using Test
using epiNetwork
using EpiBranch
using Distributions
using StableRNGs

@testset "epiNetwork.jl" begin
    include("test_network_process.jl")
end
