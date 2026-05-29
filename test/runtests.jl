using Test
using EpiBranch
using Distributions
using Random
using StableRNGs

# Sim ↔ analytical consistency helper is needed by EpiBranchAnalytics.
# Include it at the top level so both the orchestrated run and any
# per-submodule runtests pick it up.
include("testutils/sim_analytical_consistency.jl")

@testset "EpiBranch" begin
    include("test_quality.jl")
    include("EpiBranchBase/runtests.jl")
    include("EpiBranchInterventions/runtests.jl")
    include("EpiBranchTransitions/runtests.jl")
    include("EpiBranchEngine/runtests.jl")
    include("EpiBranchObservation/runtests.jl")
    include("EpiBranchOutput/runtests.jl")
    include("EpiBranchAnalytics/runtests.jl")
    include("test_integration.jl")
end
