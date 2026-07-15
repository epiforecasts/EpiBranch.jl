using Test
using EpiNetwork
using EpiBranch
using Distributions
using StableRNGs
using Graphs

@testset "EpiNetwork.jl" begin
    include("test_network_process.jl")
    include("test_graphs_ext.jl")
end
