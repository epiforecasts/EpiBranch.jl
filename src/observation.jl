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
records `:reported` (`true`/`false`) and `:reporting_time`. Under
[`ChainDetection`](@ref) every case in a chain shares the same `:reported`
flag. [`NoObservation`](@ref) and [`MinimumSize`](@ref) leave the cases
unchanged.

To add a new observation model, define this together with [`observe`](@ref).
"""
apply_observation!(::NoObservation, state, rng) = state
# A minimum recorded size selects whole clusters, which the analytical and
# simulation chain-size paths each apply where they see sizes.
apply_observation!(::MinimumSize, state, rng) = state

function apply_observation!(o::ChainDetection, state, rng)
    cases = filter(is_infected, state.individuals)
    # A progression transition (such as `Reporting`) that records `:reported`
    # owns that key, and its likelihood reads what it wrote.
    any(case -> haskey(case.state, :reported), cases) && throw(
        ArgumentError(
            "ChainDetection cannot be combined with a progression " *
                "transition that records `:reported`, such as `Reporting`: " *
                "both would write the case's report. Report cases with one " *
                "of them."
        )
    )
    chain_size = Dict{Int, Int}()
    for case in cases
        chain_size[case.chain_id] = get(chain_size, case.chain_id, 0) + 1
    end
    # One draw per chain, not per case: a chain is found and investigated in
    # full, or missed altogether, never partly reported.
    recorded = Dict(
        cid => rand(rng) < o.recording_probability(n) for (cid, n) in chain_size
    )
    for case in cases
        case.state[:reported] = recorded[case.chain_id]
    end
    return state
end

function apply_observation!(o::PerCaseObservation, state, rng)
    for ind in state.individuals
        # Observation is a property of cases: skip uninfected contact nodes
        # (kept for the contact-tracing table) so no RNG is spent on non-cases
        # and no `:reported` flag lands on a record that isn't a case.
        is_infected(ind) || continue
        # A progression transition (such as `Reporting`) that records
        # `:reported` owns that key, and its likelihood reads what it wrote.
        haskey(ind.state, :reported) && throw(
            ArgumentError(
                "PerCaseObservation cannot be combined with a progression " *
                    "transition that records `:reported`, such as `Reporting`: " *
                    "both would write the case's report. Report cases with one " *
                    "of them."
            )
        )
        ρ = sample_value(o.detection_prob, rng, ind)
        d = sample_value(o.delay, rng, ind)
        anchor = _percase_anchor(o.from, ind)
        reported = rand(rng) < ρ
        ind.state[:reported] = reported
        # Shares :reporting_time with the Reporting transition, which leaves
        # it at Inf for a case that was never reported.
        ind.state[:reporting_time] = reported ? anchor + d : Inf
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
A chain with nobody detected leaves no trace in the data, so the distribution
is conditioned on at least one case being detected. Usually built for you by
[`observe`](@ref) with a [`PerCaseObservation`](@ref).

`logpdf(d, obs)` adds `logpdf(base, n) + logpdf(Binomial(n, p), obs)` over
`n >= obs` until the remaining tail is negligible, then subtracts
`log(1 - P(0 detected))`, where `P(0 detected) = Σ_n P(base = n) (1 - p)^n` is
summed the same way. Only `logpdf` on the base is needed, so any base
distribution works without a method of its own. `P(0 detected)` depends on
`base` and `detection_prob` alone, so it is worked out once when the
distribution is built. Its field takes whatever element type `base`'s
parameters give `logpdf`, rather than `Float64`, so a `Dual` can pass through
it when something differentiates through `base`, such as fitting an offspring
parameter with `ForwardDiff`.
"""
struct ThinnedChainSize{D <: DiscreteUnivariateDistribution, T <: Real} <:
    ChainSizeLaw
    base::D
    detection_prob::Float64
    log_prob_any_detected::T
end

function ThinnedChainSize(base::D, detection_prob) where {D <: DiscreteUnivariateDistribution}
    p = Float64(detection_prob)
    0.0 < p <= 1.0 || throw(ArgumentError("detection_prob must be in (0, 1], got $p"))
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
        elseif isfinite(x - m)
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

Distributions.cdf(d::ThinnedChainSize, n::Integer) = _chain_size_cdf(d, n)
function Distributions.logccdf(d::ThinnedChainSize, n::Integer)
    return _chain_size_right_tail_logprob(d, n + 1, 1)
end

"""
Sample an observed chain size: draw a true size from `base`, thin it to a
`Binomial(n, detection_prob)` number of detected cases, and redraw when none
are detected, matching the conditioning `logpdf` applies.
"""
function Base.rand(rng::AbstractRNG, d::ThinnedChainSize)
    obs = 0
    while obs == 0
        n = rand(rng, d.base)
        obs = rand(rng, Binomial(n, d.detection_prob))
    end
    return obs
end

"""
    DetectedChainSize(base, recording_probability)

Distribution of observed chain sizes when whole chains are recorded with a
probability that depends on their size: the chain-size distribution `base`,
reweighted by `recording_probability` and renormalised,

    P(N = n) = P(base = n) * recording_probability(n) / Σ_m P(base = m) * recording_probability(m)

Usually built for you by [`observe`](@ref) with a [`ChainDetection`](@ref).
`logpdf(d, obs)` is `logpdf(base, obs) + log(recording_probability(obs))`
less the log normalising constant, which is summed once, the same way
[`ThinnedChainSize`](@ref) sums its conditioning term: a streaming
log-sum-exp over `base`'s support, stopping once the tail is negligible.
Only `logpdf` on `base` is needed, so any base distribution works without a
method of its own.
"""
struct DetectedChainSize{D <: DiscreteUnivariateDistribution, F, T <: Real} <:
    ChainSizeLaw
    base::D
    recording_probability::F
    log_prob_recorded::T
end

function DetectedChainSize(base::D, recording_probability::F) where
    {D <: DiscreteUnivariateDistribution, F}
    log_prob_recorded = _streaming_logsumexp(
        n -> logpdf(base, n) + _log_recording_probability(recording_probability, n), 1
    )
    return DetectedChainSize{D, F, typeof(log_prob_recorded)}(
        base, recording_probability, log_prob_recorded
    )
end

Distributions.minimum(::DetectedChainSize) = 1
Distributions.maximum(::DetectedChainSize) = Inf
Distributions.insupport(::DetectedChainSize, n::Integer) = n >= 1

_log_recording_probability(f, n) =
let p = f(n)
    p > 0 ? log(p) : oftype(float(p), -Inf)
end

function Distributions.logpdf(d::DetectedChainSize, n::Integer)
    n < 1 && return -Inf
    return logpdf(d.base, n) + _log_recording_probability(d.recording_probability, n) -
        d.log_prob_recorded
end

Distributions.pdf(d::DetectedChainSize, n::Integer) = exp(logpdf(d, n))

Distributions.cdf(d::DetectedChainSize, n::Integer) = _chain_size_cdf(d, n)
function Distributions.logccdf(d::DetectedChainSize, n::Integer)
    return _chain_size_right_tail_logprob(d, n + 1, 1)
end

"""
    observe(base_distribution, obs::ObservationModel)

Turn the true distribution of chain sizes into the distribution a
surveillance system would see under the observation model `obs`. The result
is itself a distribution, so it can be used in a likelihood in place of the
true one.

- [`NoObservation`](@ref): returned unchanged.
- [`PerCaseObservation`](@ref) (deprecated): under-reporting, each case
  detected with probability `detection_prob` ([`ThinnedChainSize`](@ref)).
- [`MinimumSize`](@ref): only chains of at least `min_size` cases are
  recorded ([`TruncatedChainSize`](@ref)).
- [`ChainDetection`](@ref): a whole chain is recorded with a probability
  depending on its size ([`DetectedChainSize`](@ref)).

To add a new observation model, define this together with
[`apply_observation!`](@ref EpiBranch.apply_observation!).

# Examples

```julia
# only chains of at least 2 cases are recorded
observe(chain_size_distribution(NegBin(0.8, 0.5)), MinimumSize(2))
```
"""
observe(base, ::NoObservation) = base
observe(base, o::MinimumSize) = TruncatedChainSize(base, o.min_size)
observe(base, o::ChainDetection) = DetectedChainSize(base, o.recording_probability)
function observe(base, o::PerCaseObservation)
    p = scalar_detection_prob(o)
    # ρ = 1 is a no-op; skip the wrap so multi-seed likelihoods route
    # directly to the base distribution's own multi-seed implementation.
    return p == 1.0 ? base : ThinnedChainSize(base, p)
end

"""
Scalar detection probability from a model's `Reporting` step in
`progression`, or `nothing` when it has none. Shared by the closed-form
chain-size law, `chain_size_distribution` and `end_of_outbreak_probability`,
which all read the same per-case reporting source.
"""
function _reporting_probability(progression)
    reporting = filter(t -> t isa Reporting, progression)
    isempty(reporting) && return nothing
    length(reporting) == 1 || throw(
        ArgumentError(
            "at most one `Reporting` step in `progression` is supported here; " *
                "got $(length(reporting))"
        )
    )
    p = only(reporting).probability
    p isa Real || throw(
        ArgumentError(
            "a scalar `Reporting` probability is needed here; got $(typeof(p))"
        )
    )
    return float(p)
end

"""
Thin `base` by a `Reporting` step's detection probability, read straight from
`progression`, so the closed-form chain-size law and `chain_size_distribution`
read the same source `simulate` and [`progression_loglik`](@ref) do. Returns
`base` unchanged when `progression` has no `Reporting` step, or one with
probability `1`. Throws the same `ArgumentError` `simulate` does (via
`apply_observation!`) when `obs` is a [`PerCaseObservation`](@ref) or a
[`ChainDetection`](@ref) and `progression` also has a `Reporting` step,
rather than silently thinning by both.
"""
function _fold_reporting(progression, base, obs)
    ρ = _reporting_probability(progression)
    ρ === nothing && return base
    obs isa Union{PerCaseObservation, ChainDetection} && throw(
        ArgumentError(
            "$(nameof(typeof(obs))) cannot be combined with a progression " *
                "transition that records `:reported`, such as `Reporting`: " *
                "both would write the case's report. Report cases with one " *
                "of them."
        )
    )
    ρ == 1.0 && return base
    return ThinnedChainSize(base, ρ)
end
