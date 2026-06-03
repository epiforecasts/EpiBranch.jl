using ADTypes
using DifferentiationInterface
using ForwardDiff

# AD smoke tests: differentiate the analytical chain-size and chain-length
# log-likelihoods through their distribution parameters.
#
# Scope: a starting baseline. ForwardDiff is the lowest-friction backend
# and exercises the bulk of the analytical path. Other backends (Mooncake,
# Enzyme, Zygote) can be added behind `@test_skip` markers as they're
# verified. This is intentionally minimal — see #105 for the broader plan.

@testset "AD" begin
    @testset "ForwardDiff" begin
        backend = AutoForwardDiff()

        # ChainSizes / Poisson
        sizes = ChainSizes([1, 2, 1, 3, 1, 5, 2])
        ll_poisson = R -> loglikelihood(sizes, Poisson(R))
        @test isfinite(ll_poisson(0.5))
        g = derivative(ll_poisson, backend, 0.5)
        @test isfinite(g)

        # ChainSizes / NegBin (two-parameter — differentiate through R)
        ll_negbin_R = R -> loglikelihood(sizes, NegBin(R, 0.5))
        @test isfinite(ll_negbin_R(0.5))
        gR = derivative(ll_negbin_R, backend, 0.5)
        @test isfinite(gR)

        # ChainLengths / Poisson
        lengths = ChainLengths([0, 1, 2, 0, 3, 1])
        ll_lp = R -> loglikelihood(lengths, Poisson(R))
        @test isfinite(ll_lp(0.5))
        gL = derivative(ll_lp, backend, 0.5)
        @test isfinite(gL)

        # OffspringCounts / Poisson — covered by Distributions.jl directly
        counts = OffspringCounts([0, 1, 2, 0, 3, 1, 0])
        ll_oc = R -> loglikelihood(counts, Poisson(R))
        @test isfinite(ll_oc(0.5))
        gOC = derivative(ll_oc, backend, 0.5)
        @test isfinite(gOC)
    end
end
