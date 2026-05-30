using Test
using EpiBranch
using Distributions
using Random
using StableRNGs

# Sim ↔ analytical consistency helper is needed by Analytics.
# Include it at the top level so both the orchestrated run and any
# per-submodule runtests pick it up.
include("testutils/sim_analytical_consistency.jl")

@testset "EpiBranch" begin
    include("test_quality.jl")
    include("EpiBranchBase/runtests.jl")
    include("Interventions/runtests.jl")
    include("Transitions/runtests.jl")
    include("Engine/runtests.jl")
    include("Observation/runtests.jl")
    include("Output/runtests.jl")
    include("Analytics/runtests.jl")
    include("test_integration.jl")
end
