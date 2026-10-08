# ── Observation protocol ────────────────────────────────────────────
# A process carries its observation model alongside its dynamics (see
# model_inputs.jl); the engine and the likelihood read it with
# `observation(model)` and dispatch on the returned object. An
# observation model sits alongside `AbstractIntervention`: it joins in by
# implementing two methods, dispatched on the observation type,
# `apply_observation!` (simulation side) and `observe` (analytical side).
# The dispatch is on the observation value, so there is no model type
# parameter.

# Interventions, attributes and observation are set once, on the model
# constructor. A process reads them back through the `interventions(m)`,
# `attributes(m)` and `observation(m)` accessors (model_inputs.jl). There
# is no in-place "replace one input" API: to vary one of them, construct
# the model with the input you want, stating all of them, so a model's
# interventions, attributes and observation are always explicit at
# construction.

"""
    apply_observation!(obs::ObservationModel, state, rng)

Simulation side of the observation protocol: apply `obs` to a finished
`SimulationState` in place (e.g. mark `:reported` cases and set
`:report_time`). Called by [`simulate`](@ref) after the run. The default
[`NoObservation`](@ref) leaves the latent cases untouched.
"""
apply_observation!(::NoObservation, state, rng) = state
# A minimum recorded size selects whole clusters, which the analytical and
# simulation chain-size paths each apply where they see sizes.
apply_observation!(::MinimumSize, state, rng) = state

function apply_observation!(o::PerCaseObservation, state, rng)
    for ind in state.individuals
        # Observation is a property of cases: skip uninfected contact nodes
        # (kept for the contact-tracing table) so no RNG is spent on non-cases
        # and no `:reported` flag lands on a record that isn't a case.
        is_infected(ind) || continue
        ρ = _sample_value(o.detection_prob, rng, ind)
        d = _sample_value(o.delay, rng, ind)
        anchor = _percase_anchor(o.from, ind)
        ind.state[:reported] = rand(rng) < ρ
        ind.state[:report_time] = anchor + d
    end
    return state
end

_percase_anchor(s::Symbol, ind) =
let v = get(ind.state, s, NaN)
    isnan(v) ? ind.infection_time : v
end
_percase_anchor(f, ind) =
let v = float(f(ind))
    isnan(v) ? ind.infection_time : v
end

"""
    ThinnedChainSize(base, detection_prob)

Distribution of observed chain sizes when each case in a `base` chain
is detected with probability `detection_prob`, conditioned on at least
one case being detected: a chain with no detected case leaves no trace
in the data, so it cannot be one of the observed sizes.

`logpdf(d, obs)` sums `logpdf(base, n) + logpdf(Binomial(n, p), obs)`
over `n >= obs` until the tail is negligible, then subtracts
`log(1 - P(0 detected))`, with `P(0 detected) = Σ_n P(base = n) (1 - p)^n`
summed the same way. The computation only needs `logpdf` on the base,
so this composes without specialised methods. `P(0 detected)` depends
only on `base` and `detection_prob`, which lets it be computed once
at construction rather than on every `logpdf` call. It is stored with
whatever element type `base`'s parameters give `logpdf`, rather than
fixed to `Float64`, so differentiating through `base` (e.g. fitting an
offspring parameter with `ForwardDiff`) passes a `Dual` through this
field instead of hitting a conversion error.
"""
struct ThinnedChainSize{D <: DiscreteUnivariateDistribution, T <: Real} <:
    DiscreteUnivariateDistribution
    base::D
    detection_prob::Float64
    log_prob_any_detected::T
end

function ThinnedChainSize(base::D, detection_prob) where {D <: DiscreteUnivariateDistribution}
    p = Float64(detection_prob)
    log_prob_any_detected = _log_prob_any_detected(base, p)
    return ThinnedChainSize{D, typeof(log_prob_any_detected)}(base, p, log_prob_any_detected)
end

Distributions.minimum(::ThinnedChainSize) = 1
Distributions.maximum(::ThinnedChainSize) = Inf
Distributions.insupport(::ThinnedChainSize, n::Integer) = n >= 1

"""
Streaming log-sum-exp of `term(n)` for `n = n_start, n_start + 1, …`:
maintain a running max `m` and `S = Σ exp(xᵢ - m)`. Stop when the
accumulated value stops changing (within `tol`) and at least 20 further
terms have been added, to avoid false early convergence on heavy-tailed
bases (e.g. GammaBorel with low k).
"""
function _streaming_logsumexp(
        term, n_start::Integer; tol::Float64 = 1.0e-12, max_n::Int = 100_000
    )
    m = term(n_start)
    S = 1.0
    prev = m
    n = n_start + 1
    while n <= max_n
        x = term(n)
        if x > m
            S = 1.0 + S * exp(m - x)
            m = x
        else
            S += exp(x - m)
        end
        cur = m + log(S)
        if isfinite(cur) && isfinite(prev) && abs(cur - prev) < tol &&
                n - n_start >= 20
            return cur
        end
        prev = cur
        n += 1
    end
    return prev
end

_log1mexp(x::Real) = x > -log(2) ? log(-expm1(x)) : log1p(-exp(x))

# log P(no case in the chain is detected) = log Σ_n P(base = n) (1 - p)^n,
# summed over the base's support (chains always have at least one case).
# At p = 1 every term is -Inf (zero mass), and the streaming sum can't
# subtract two infinite terms; short-circuit to the same answer directly.
function _log_prob_none_detected(base, detection_prob::Float64)
    detection_prob >= 1.0 && return -Inf
    lq = log1p(-detection_prob)
    return _streaming_logsumexp(n -> logpdf(base, n) + n * lq, 1)
end

_log_prob_any_detected(base, detection_prob::Float64) =
    _log1mexp(_log_prob_none_detected(base, detection_prob))

function Distributions.logpdf(d::ThinnedChainSize, obs::Integer)
    obs < 1 && return -Inf
    p = d.detection_prob
    unconditioned = _streaming_logsumexp(
        n -> logpdf(d.base, n) + logpdf(Binomial(n, p), obs), obs
    )
    return unconditioned - d.log_prob_any_detected
end

Distributions.pdf(d::ThinnedChainSize, n::Integer) = exp(logpdf(d, n))

"""
    observe(base_distribution, obs::ObservationModel)

Analytical side of the observation protocol: transform the latent
`base_distribution` (e.g. a chain-size distribution) into the
distribution of the *observed* quantity under `obs`, returning a
`Distribution`. Because the result is itself a distribution, it slots
into the same likelihood machinery as the latent law (see the design
notes on why observation models return distributions). The default
[`NoObservation`](@ref) returns the base unchanged;
[`PerCaseObservation`](@ref) thins it with [`ThinnedChainSize`](@ref), and
[`MinimumSize`](@ref) conditions it with [`TruncatedChainSize`](@ref).
"""
observe(base, ::NoObservation) = base
observe(base, o::MinimumSize) = TruncatedChainSize(base, o.min_size)
function observe(base, o::PerCaseObservation)
    p = scalar_detection_prob(o)
    # ρ = 1 is a no-op; skip the wrap so multi-seed likelihoods route
    # directly to the base distribution's own multi-seed implementation.
    return p == 1.0 ? base : ThinnedChainSize(base, p)
end
