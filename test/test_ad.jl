using ADTypes
using DifferentiationInterface
import DifferentiationInterfaceTest as DIT
using FiniteDifferences
using ForwardDiff

# AD correctness tests for the analytical chain-size, chain-length, and
# offspring-count log-likelihoods. Uses `DifferentiationInterfaceTest`
# (DIT) to compare each backend's gradient against a finite-difference
# reference (`res1`) computed up-front with FiniteDifferences.jl.
#
# Scope: ForwardDiff only for now — a starting baseline. The right end
# state is the scenario-based pattern in CensoredDistributions.jl
# (per-backend tags, ADFixtures path package, Mooncake/Enzyme/ReverseDiff
# coverage); this PR is the smallest step that uses DIT rather than
# ad-hoc `derivative` calls. See #105 for the broader plan.

@testset "AD" begin
    sizes_data = [1, 2, 1, 3, 1, 5, 2]
    lengths_data = [0, 1, 2, 0, 3, 1]
    counts_data = [0, 1, 2, 0, 3, 1, 0]

    fdm = central_fdm(5, 1)
    fd_grad(f, x) = first(FiniteDifferences.grad(fdm, f, x))

    # Each scenario pairs a function-of-parameters with its starting
    # point. `Constant`-wrapped data is passed positionally so DIT
    # differentiates only through the model parameters. `res1` is the
    # reference gradient that DIT compares each backend against.
    function scenario(f, x, data)
        g = fd_grad(p -> f(p, data), x)
        return DIT.Scenario{:gradient, :out}(f, x, Constant(data); res1 = g)
    end

    scenarios = [
        scenario((R, d) -> loglikelihood(ChainSizes(d), Poisson(R[1])),
            [0.5], sizes_data),
        scenario((θ, d) -> loglikelihood(ChainSizes(d), NegBin(θ[1], θ[2])),
            [0.5, 0.5], sizes_data),
        scenario((R, d) -> loglikelihood(ChainLengths(d), Poisson(R[1])),
            [0.5], lengths_data),
        scenario((R, d) -> loglikelihood(OffspringCounts(d), Poisson(R[1])),
            [0.5], counts_data)
    ]

    DIT.test_differentiation(
        [AutoForwardDiff()], scenarios;
        correctness = true,
        type_stability = :none,
        logging = false,
        rtol = 1e-5,
        atol = 1e-8)
end
