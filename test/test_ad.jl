using ADTypes
using DifferentiationInterface
import DifferentiationInterfaceTest as DIT
using FiniteDifferences
using ForwardDiff
import Mooncake
using StableRNGs

# AD correctness tests for the analytical chain-size, chain-length, and
# offspring-count log-likelihoods. Uses `DifferentiationInterfaceTest`
# (DIT) to compare each backend's gradient against a finite-difference
# reference (`res1`) computed up-front with FiniteDifferences.jl.
#
# Scope: ForwardDiff through DIT, plus a Mooncake reverse-mode pass in a
# separate testset below. Mooncake is the reverse-mode backend the package
# targets, so the analytical likelihoods are checked both ways. It is not run
# through DIT because DIT's correctness harness builds its reverse rule
# through the `ChainSizes`/etc. data constructor (including its validation
# branch) and errors there; a direct gradient comparison against ForwardDiff
# avoids that while still exercising the reverse path over the likelihood.
# ReverseDiff is left out — no rule for the `logabsgamma` in the NegBin
# chain-size path. Enzyme and the per-backend-tag scenario pattern from
# CensoredDistributions.jl are the remaining end state; see #105.

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

# Reverse-mode (Mooncake) coverage of the same analytical likelihoods. Each
# gradient is taken with respect to the distribution parameters and must match
# the ForwardDiff gradient — so the reverse path is exercised end to end without
# the DIT harness that trips over the data constructor.
@testset "AD (Mooncake reverse mode)" begin
    sizes_data = [1, 2, 1, 3, 1, 5, 2]
    lengths_data = [0, 1, 2, 0, 3, 1]
    counts_data = [0, 1, 2, 0, 3, 1, 0]
    mooncake = AutoMooncake(; config = nothing)

    cases = [
        (R -> loglikelihood(ChainSizes(sizes_data), Poisson(R[1])), [0.5]),
        (θ -> loglikelihood(ChainSizes(sizes_data), NegBin(θ[1], θ[2])), [0.5, 0.5]),
        (R -> loglikelihood(ChainLengths(lengths_data), Poisson(R[1])), [0.5]),
        (R -> loglikelihood(OffspringCounts(counts_data), Poisson(R[1])), [0.5])
    ]

    for (f, x) in cases
        g_reverse = DifferentiationInterface.gradient(f, mooncake, x)
        g_forward = DifferentiationInterface.gradient(f, AutoForwardDiff(), x)
        @test g_reverse ≈ g_forward rtol=1e-6
    end
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

# The same capability for the fixed-size Sellke pool: a dual β flows through the
# crossing times, so the derivative of a smooth outbreak summary with respect to
# β falls out by forward mode. The infected set is held fixed here — a small step
# keeps every crossing on the same side of every event — so the total infection
# time is smooth in β with a within-piece derivative a tiny finite difference
# confirms. How the final size itself moves with β is discontinuous and is the
# unbiased estimator's job (StochasticAD), not forward mode.
@testset "AD through the Sellke pool (timing gradient)" begin
    seed = 20260701
    build(β) = HomogeneousProcess(; transmission_rate = β, population_size = 500,
        infectious_period = Exponential(1.0))
    function total_infection_time(β)
        state = simulate(build(β); n_initial = 5, rng = StableRNG(seed))
        return sum(ind.infection_time
        for ind in state.individuals
        if get(ind.state, :infected, false))
    end

    β0 = 2.0
    plain = simulate(build(β0); n_initial = 5, rng = StableRNG(seed))
    dual = simulate(build(ForwardDiff.Dual(β0, 1.0)); n_initial = 5, rng = StableRNG(seed))

    # A dual β makes an Individual{Dual} pool; the primal trajectory is unchanged.
    @test EpiBranch._timetype(plain) === Float64
    @test EpiBranch._timetype(dual) <: ForwardDiff.Dual
    @test dual.cumulative_cases == plain.cumulative_cases
    @test all(ForwardDiff.value(d.infection_time) == p.infection_time
    for (d, p) in zip(dual.individuals, plain.individuals))

    grad = ForwardDiff.derivative(total_infection_time, β0)
    @test isfinite(grad)
    @test grad < 0        # faster spread pulls the same infections earlier

    h = 1e-6              # a step small enough to keep the infected set fixed
    fd = (total_infection_time(β0 + h) - total_infection_time(β0 - h)) / (2h)
    @test isapprox(grad, fd; rtol = 1e-4)
end
