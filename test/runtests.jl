using Test
using EpiBranch
using Distributions
using Random
using StableRNGs

include("testutils/sim_analytical_consistency.jl")

@testset "EpiBranch" begin
    include("test_quality.jl")
    include("test_types.jl")
    include("test_branching_process.jl")
    include("test_attributes.jl")
    include("test_interventions.jl")
    include("test_competing_risks.jl")
    include("test_ct_seams.jl")
    include("test_pc_observation.jl")
    include("test_stopping_rules.jl")
    include("test_transitions.jl")
    include("test_density_dependent.jl")
    include("test_linelist.jl")
    include("test_chains.jl")
    include("test_analytical.jl")
    include("test_multitype.jl")
    include("test_integration.jl")
    include("test_r_targets.jl")
end
