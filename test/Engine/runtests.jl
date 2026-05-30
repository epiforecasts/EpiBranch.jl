using Test
using EpiBranch
using Distributions
using Random
using StableRNGs

@testset "Engine" begin
    include("test_branching_process.jl")
    include("test_competing_risks.jl")
    include("test_density_dependent.jl")
    include("test_stopping_rules.jl")
    include("test_multitype.jl")
end
