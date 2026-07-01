using ADTypes
using DifferentiationInterface
import DifferentiationInterfaceTest as DIT
using FiniteDifferences
using ForwardDiff
using StableRNGs

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

# A gradient of an outbreak summary through the forward simulator, w.r.t. a
# generation-time parameter — the capability the eltype-generic state opens up.
# The tree is held fixed (same seed, offspring independent of the timing
# parameter), so only the times vary with μ. For LogNormal(μ, σ) contact
# intervals each interval's ∂/∂μ equals the interval itself, so the derivative
# of the total infection time equals that total — a check with a known answer.
@testset "AD through the forward simulator (timing gradient)" begin
    function total_infection_time(μ)
        model = BranchingProcess(NegBin(3.0, 0.5), LogNormal(μ, 0.5))
        state = simulate(model; n_initial = 5, rng = StableRNG(20260701),
            stopping_rules = [Extinction(), MaxGenerations(6)])
        return sum(ind.infection_time for ind in state.individuals)
    end

    μ0 = 1.6
    value = total_infection_time(μ0)
    grad = ForwardDiff.derivative(total_infection_time, μ0)

    @test isfinite(grad)
    @test grad > 0
    @test isapprox(grad, value; rtol = 1e-8)        # ∂/∂μ equals the total itself
    fd = central_fdm(5, 1)(total_infection_time, μ0)
    @test isapprox(grad, fd; rtol = 1e-4)           # finite-difference cross-check

    # The default (non-AD) run is unchanged: the state carries Float64.
    plain = simulate(BranchingProcess(NegBin(3.0, 0.5), LogNormal(1.6, 0.5));
        n_initial = 1, rng = StableRNG(1),
        stopping_rules = [Extinction(), MaxGenerations(3)])
    @test EpiBranch._timetype(plain) === Float64
end
