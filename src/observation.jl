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

Mark which simulated cases are reported, and when, under an observation
model. [`simulate`](@ref) calls it once the outbreak has been simulated; it
changes `state` in place. Under [`PerCaseObservation`](@ref) each case
records `:reported` (`true`/`false`) and `:report_time`.
[`NoObservation`](@ref) and [`MinimumSize`](@ref) leave the cases unchanged.

To add a new observation model, define this together with [`observe`](@ref).
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

Distribution of observed chain sizes under under-reporting: each case in a
chain whose true size follows `base` is detected independently with
probability `detection_prob`, and the observed size is the number detected.
The probability of each observed size sums over the possible true sizes.
Usually built for you by [`observe`](@ref) with a
[`PerCaseObservation`](@ref).
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
    tol = 1.0e-12
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

Turn the true distribution of chain sizes into the distribution a
surveillance system would see under the observation model `obs`. The result
is itself a distribution, so it can be used in a likelihood in place of the
true one.

- [`NoObservation`](@ref): returned unchanged.
- [`PerCaseObservation`](@ref): under-reporting, each case detected with
  probability `detection_prob` ([`ThinnedChainSize`](@ref)).
- [`MinimumSize`](@ref): only chains of at least `min_size` cases are
  recorded ([`TruncatedChainSize`](@ref)).

To add a new observation model, define this together with
[`apply_observation!`](@ref EpiBranch.apply_observation!).

# Examples

```julia
observe(chain_size_distribution(NegBin(0.8, 0.5)), PerCaseObservation(detection_prob = 0.6))
```
"""
observe(base, ::NoObservation) = base
observe(base, o::MinimumSize) = TruncatedChainSize(base, o.min_size)
function observe(base, o::PerCaseObservation)
    p = scalar_detection_prob(o)
    # ρ = 1 is a no-op; skip the wrap so multi-seed likelihoods route
    # directly to the base distribution's own multi-seed implementation.
    return p == 1.0 ? base : ThinnedChainSize(base, p)
end
