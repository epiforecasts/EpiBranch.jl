using Test
using EpiBranch
using Distributions
using Random
using StableRNGs

@testset "Interventions" begin
    include("test_interventions.jl")
    include("test_isolation_seams.jl")
    include("test_ct_seams.jl")
end
