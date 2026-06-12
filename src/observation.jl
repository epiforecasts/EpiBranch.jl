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

function apply_observation!(o::PerCaseObservation, state, rng)
    for ind in state.individuals
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
is detected with probability `detection_prob`.

`logpdf(d, obs)` sums `logpdf(base, n) + logpdf(Binomial(n, p), obs)`
over `n >= obs` until the tail is negligible. The computation only
needs `logpdf` on the base, so this composes without specialised
methods.
"""
struct ThinnedChainSize{D <: DiscreteUnivariateDistribution} <:
       DiscreteUnivariateDistribution
    base::D
    detection_prob::Float64
end

Distributions.minimum(::ThinnedChainSize) = 1
Distributions.maximum(::ThinnedChainSize) = Inf
Distributions.insupport(::ThinnedChainSize, n::Integer) = n >= 1

function Distributions.logpdf(d::ThinnedChainSize, obs::Integer)
    obs < 1 && return -Inf
    p = d.detection_prob
    # Streaming log-sum-exp: maintain running max `m` and `S = Σ exp(xᵢ - m)`.
    # Stop when the accumulated value stops changing (within `tol`) and at
    # least 20 further terms have been added, to avoid false early
    # convergence on heavy-tailed bases (e.g. GammaBorel with low k).
    tol = 1e-12
    max_n = 100_000
    first = logpdf(d.base, obs) + logpdf(Binomial(obs, p), obs)
    m = first
    S = 1.0
    prev = m
    n = obs + 1
    while n <= max_n
        x = logpdf(d.base, n) + logpdf(Binomial(n, p), obs)
        if x > m
            S = 1.0 + S * exp(m - x)
            m = x
        else
            S += exp(x - m)
        end
        cur = m + log(S)
        if isfinite(cur) && isfinite(prev) && abs(cur - prev) < tol &&
           n - obs >= 20
            return cur
        end
        prev = cur
        n += 1
    end
    return prev
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
[`PerCaseObservation`](@ref) thins it with [`ThinnedChainSize`](@ref).
"""
observe(base, ::NoObservation) = base
function observe(base, o::PerCaseObservation)
    p = scalar_detection_prob(o)
    # ρ = 1 is a no-op; skip the wrap so multi-seed likelihoods route
    # directly to the base distribution's own multi-seed implementation.
    return p == 1.0 ? base : ThinnedChainSize(base, p)
end
