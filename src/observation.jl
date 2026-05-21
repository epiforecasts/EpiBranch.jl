# ── Observation models ──────────────────────────────────────────────
# Wrappers around a TransmissionModel that describe how it is observed.
# Dispatch uses the analytical likelihood when one is defined; otherwise
# it falls back to simulation.

"""
    Observed(process, observation)

State-space combiner: pair a `TransmissionModel` with an
[`ObservationModel`](@ref). Dispatch on `Observed{P, O}` is the single
surface for likelihoods that need to know about both the latent
dynamics and how the data was observed.

Forwards process-model accessors (`population_size`, `latent_period`,
`n_types`, `single_type_offspring`) to `process`.
"""
struct Observed{P <: TransmissionModel, O <: ObservationModel} <:
       TransmissionModel
    process::P
    observation::O
end

function Base.show(io::IO, m::Observed)
    print(io, "Observed($(m.process), $(m.observation))")
end

population_size(m::Observed) = population_size(m.process)
latent_period(m::Observed) = latent_period(m.process)
n_types(m::Observed) = n_types(m.process)
single_type_offspring(m::Observed) = single_type_offspring(m.process)

"""
    simulate(m::Observed; kwargs...)

Run the underlying process simulation, then apply the observation
model in place. For `PerCaseObservation(ρ, D)`, each individual
gets:

- `state[:reported] = true` with probability `ρ`, else `false`.
- `state[:report_time] = infection_time + d` where `d` is an
  independent draw from `D`.

Downstream output (line list, chain statistics) can then filter on
`:reported` and read `:report_time`.
"""
function simulate(m::Observed{<:TransmissionModel, <:PerCaseObservation};
        rng::AbstractRNG = Random.default_rng(), kwargs...)
    state = simulate(m.process; rng = rng, kwargs...)
    for ind in state.individuals
        ρ = _sample_value(m.observation.detection_prob, rng, ind)
        d = _sample_value(m.observation.delay, rng, ind)
        ind.state[:reported] = rand(rng) < ρ
        ind.state[:report_time] = ind.infection_time + d
    end
    return state
end

"""
    ThinnedChainSize(base, detection_prob)

Distribution of observed chain sizes when each case in a `base` chain
is detected with probability `detection_prob`. Nesting applies a second
round of Binomial thinning.

`logpdf(d, obs)` sums `logpdf(base, n) + logpdf(Binomial(n, p), obs)`
over `n >= obs` until the tail is negligible. The computation only
needs `logpdf` on the base, so a nested `ThinnedChainSize` gives the
same answer as a single thinning with a compounded probability without
any specialised method.
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
    chain_size_distribution(m::Observed{<:Any, <:PerCaseObservation})

Return `ThinnedChainSize` wrapping the chain size distribution of the
inner process. Recurses through any nested `Observed` wrappers, so a
double per-case observation (e.g. surveillance plus a separate audit)
gives the right nested-thinning likelihood without pairwise dispatch.
"""
function chain_size_distribution(m::Observed{<:Any, <:PerCaseObservation})
    p = scalar_detection_prob(m.observation)
    base = chain_size_distribution(m.process)
    # ρ = 1 is a no-op; skip the ThinnedChainSize wrap so multi-seed
    # likelihoods route directly to the underlying distribution's
    # multi-seed implementation (which ThinnedChainSize lacks).
    return p == 1.0 ? base : ThinnedChainSize(base, p)
end
