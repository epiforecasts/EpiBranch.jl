using Test
using EpiNetwork
using EpiBranch
using Distributions
using StableRNGs

@testset "EpiNetwork.jl" begin
    include("test_network_process.jl")
end
