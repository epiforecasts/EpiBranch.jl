using StochasticAD
using Statistics
using StableRNGs

# Gradients of simulated counts with respect to a parameter are zero almost
# everywhere under ordinary (forward- or reverse-mode) AD: a discrete draw is
# a step function of its parameter, so differentiating straight through it
# gives no signal. StochasticAD.jl instead gives an *unbiased estimator* of
# the derivative of an expectation over the discrete choice; each call draws
# its own sample, so the estimate is noisy and only its average over many
# calls should match a reference derivative.
#
# This file checks two things against finite differences (via
# `StochasticAD.derivative_estimate`, averaged over many calls):
#
# 1. The offspring distributions (`Poisson`, `NegBin`) already compose with
#    StochasticAD with no change to the package: `draw_offspring` and
#    `generate_offspring` call `rand(rng, offspring)` on a plain
#    `Distributions.jl` object, and StochasticAD overloads `rand` for exactly
#    these distributions when a parameter is its `StochasticTriple` number
#    type. `NegBin` draws via `NegativeBinomial`'s own Gamma-Poisson mixture
#    (a continuous reparameterisation feeding a native-Poisson draw), which
#    composes the same way.
# 2. The engine's competing-risk draw (`_risks_block`, used for host
#    susceptibility, infector infectiousness and every intervention's leaky
#    effect) now draws from `Bernoulli(prob)` rather than comparing a raw
#    uniform against `prob` — necessary groundwork, since StochasticAD only
#    recognises a handful of `Distributions.jl` types as differentiable
#    discrete choices, not an arbitrary comparison. That alone does not yet
#    give a gradient through `simulate`: the `&&`/`return` short-circuiting
#    around the draw still has to decide, for a given run, whether the
#    contact was blocked, and a decision that depends on the parameter is
#    itself non-smooth. `StochasticAD.propagate` is the documented way to
#    make a non-smooth decision like that produce a derivative — it reruns the
#    decision on each alternative the draw's `StochasticTriple` represents,
#    rather than branching on the triple directly, which StochasticAD
#    rejects ("non-boolean used in boolean context"/"boolean predicate
#    cannot depend on input"). The second testset below checks that pattern
#    end to end on a small model built the same way `_risks_block` resolves
#    one risk; wiring it into the engine's own short-circuiting loop, so a
#    `ModelSpec` with interventions gets a working gradient through
#    `simulate` itself, is follow-up work (see
#    docs/src/tutorials/inference.md, "Gradients of simulated outcomes").

@testset "Offspring distributions compose with StochasticAD" begin
    R0, k0 = 2.5, 0.5
    n_replicates = 8000

    # E[Poisson(R)] = R, so d/dR = 1 exactly; StochasticAD's own estimator
    # for this case has zero variance.
    poisson_grad = mean(
        StochasticAD.derivative_estimate(R -> rand(StableRNG(20261030), Poisson(R)), R0)
            for _ in 1:500
    )
    @test poisson_grad ≈ 1.0

    # NegBin(R, k) has mean R for every k, so d/dR = 1 here too, but the
    # Gamma-Poisson mixture draw makes the estimator noisy: average over
    # enough replicates for the Monte Carlo error to stay a few standard
    # errors below the tolerance.
    rng_R = StableRNG(20261030)
    negbin_grad = mean(
        StochasticAD.derivative_estimate(R -> rand(rng_R, NegBin(R, k0)), R0)
            for _ in 1:n_replicates
    )
    @test negbin_grad ≈ 1.0 rtol = 0.1

    # d/dk is 0: k does not change the mean. The estimator is noisier here
    # (no reparameterisation shortens its variance the way the Poisson-only
    # case above has none at all), so this only checks it is small next to
    # the d/dR magnitude just confirmed, not that it is tight around zero.
    rng_k = StableRNG(20261030)
    negbin_grad_k = mean(
        StochasticAD.derivative_estimate(k -> rand(rng_k, NegBin(R0, k)), k0)
            for _ in 1:n_replicates
    )
    @test abs(negbin_grad_k) < 0.3
end

@testset "A competing-risk draw composes with StochasticAD via propagate" begin
    # `_draw_blocked` mirrors the engine's own `rand(rng, Bernoulli(prob))`
    # draw (one per active risk source). Its native StochasticAD dispatch
    # fires automatically once `prob` is a `StochasticTriple` — nothing here
    # is StochasticAD-specific.
    _draw_blocked(rng, prob) = rand(rng, Bernoulli(prob))

    # Whether *any* source blocked decides the contact's outcome — the
    # non-smooth step that `_risks_block`'s `&&`/`return` loop also makes. Wrapped
    # in `propagate` so each of the draw's alternatives reruns this decision
    # instead of being compared against directly.
    _survives(any_blocked::Integer) = any_blocked == 0 ? 1.0 : 0.0
    _survives(x::StochasticAD.StochasticTriple) = StochasticAD.propagate(_survives, x)

    function contact_survives(rng, probs)
        blocked = reduce(max, (_draw_blocked(rng, p) for p in probs))
        return _survives(blocked)
    end

    rng = StableRNG(20261030)
    n_contacts = 300
    n_replicates = 500

    @testset "one risk source" begin
        p0 = 0.3
        total(p) = sum(contact_survives(rng, (p,)) for _ in 1:n_contacts)
        grad = mean(
            StochasticAD.derivative_estimate(total, p0) for _ in 1:n_replicates
        )
        # E[# survivors] = n_contacts * (1 - p); d/dp = -n_contacts
        @test grad ≈ -n_contacts rtol = 0.02
    end

    @testset "two independent risk sources" begin
        p1_0, p2_0 = 0.3, 0.2
        total(p1) = sum(contact_survives(rng, (p1, p2_0)) for _ in 1:n_contacts)
        grad = mean(
            StochasticAD.derivative_estimate(total, p1_0) for _ in 1:n_replicates
        )
        # E[# survivors] = n_contacts * (1 - p1) * (1 - p2); d/dp1 = -n_contacts * (1 - p2)
        @test grad ≈ -n_contacts * (1 - p2_0) rtol = 0.02
    end
end
