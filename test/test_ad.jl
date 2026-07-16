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
    build(β) = ModelSpec(
        HomogeneousProcess(; transmission_rate = β, population_size = 500);
        progression = [Transition(:recovered; from = :infection,
            delay = Exponential(1.0), terminal = true)])
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

# Interventions act on the timing layer, so a run carrying an `Isolation`
# intervention must still differentiate: the isolation time is `onset + delay`,
# a dual under AD, and `set_isolated!`/`isolation_time` have to carry it rather
# than pin `Float64`. Before the eltype-generic accessors this threw a
# MethodError at `set_isolated!(::Individual, ::Dual)`.
@testset "AD through the forward simulator (isolation timing)" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.0)
    iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
    function total_infection_time(μ)
        model = BranchingProcess(NegBin(2.0, 0.5), LogNormal(μ, 0.5))
        state = simulate(ModelSpec(model; interventions = [iso], attributes = clinical);
            n_initial = 20, rng = StableRNG(20260701),
            stopping_rules = [Extinction(), MaxGenerations(5)])
        return sum(ind.infection_time for ind in state.individuals if is_infected(ind))
    end

    μ0 = 1.6
    @test isfinite(total_infection_time(μ0))
    grad = ForwardDiff.derivative(total_infection_time, μ0)
    @test isfinite(grad)
end

# The `until`-window censor reads the infector's removal time (`:died_time` /
# `:recovered_time`), which is a dual once a timing parameter is differentiated.
# Before dropping the `::Float64` assertion in `WindowCensor.competing_risk`,
# any censored/funeral-route model threw a TypeError under AD.
@testset "AD through the forward simulator (until-window censoring)" begin
    history = [
        Transition(:infectious, from = :infection, delay = LogNormal(1.0, 0.3)),
        Transition(:onset, from = :infection, delay = LogNormal(1.6, 0.4)),
        Transition(:died, from = :onset, delay = Gamma(2.0, 3.0),
            probability = 0.6, terminal = true),
        Transition(:recovered, from = :onset, delay = Gamma(2.0, 5.0), terminal = true)
    ]
    function total_infection_time(θ)
        community = Infectiousness(NegBin(1.5, 0.5);
            from = :infectious, until = (:recovered, :died), kernel = Gamma(2.0, θ))
        state = simulate(ModelSpec(BranchingProcess(community); progression = history);
            n_initial = 20, rng = StableRNG(20260701),
            stopping_rules = [Extinction(), MaxGenerations(5)])
        return sum(ind.infection_time for ind in state.individuals if is_infected(ind))
    end

    θ0 = 2.0
    @test isfinite(total_infection_time(θ0))
    grad = ForwardDiff.derivative(total_infection_time, θ0)
    @test isfinite(grad)
end

# Contact tracing carries dual timing too: FlagOnly writes a dual
# `:traced_isolation_time` (from onset) that Isolation reads, and depth-2 tracing
# reads the infector's dual `:trace_time`. Exercises the eltype-generic reads in
# isolation.jl and contact_tracing.jl that the isolation-only test above misses.
@testset "AD through the forward simulator (contact tracing + isolation)" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.0)
    iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
    ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(0.5),
        quarantine_on_trace = false, depth = 2)
    function total_infection_time(μ)
        model = BranchingProcess(NegBin(2.0, 0.5), LogNormal(μ, 0.5))
        state = simulate(
            ModelSpec(model; interventions = [iso, ct], attributes = clinical);
            n_initial = 20, rng = StableRNG(20260701),
            stopping_rules = [Extinction(), MaxGenerations(5)])
        return sum(ind.infection_time for ind in state.individuals if is_infected(ind))
    end

    μ0 = 1.6
    @test isfinite(total_infection_time(μ0))
    grad = ForwardDiff.derivative(total_infection_time, μ0)
    @test isfinite(grad)
end

# Ring vaccination stores a dual vaccination time (the trace-driven isolation
# time) and reads it back on both the susceptibility and onward-transmission
# sides — the `:vaccination_time` reads that were still pinned to Float64.
@testset "AD through the forward simulator (ring vaccination)" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.0)
    iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
    ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Exponential(0.5))
    rv = RingVaccination(efficacy = 0.8, onward_efficacy = 0.5)
    function total_infection_time(μ)
        model = BranchingProcess(NegBin(2.0, 0.5), LogNormal(μ, 0.5))
        state = simulate(
            ModelSpec(model; interventions = [iso, ct, rv], attributes = clinical);
            n_initial = 20, rng = StableRNG(20260701),
            stopping_rules = [Extinction(), MaxGenerations(5)])
        return sum(ind.infection_time for ind in state.individuals if is_infected(ind))
    end

    μ0 = 1.6
    @test isfinite(total_infection_time(μ0))
    grad = ForwardDiff.derivative(total_infection_time, μ0)
    @test isfinite(grad)
end
