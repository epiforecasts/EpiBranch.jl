# ── Observation protocol ────────────────────────────────────────────
# A process carries its observation model in its forcings (see
# forcings.jl); the engine and the likelihood read it with
# `_observation(model)` and dispatch on the returned object. An
# observation model sits alongside `AbstractIntervention`: it joins in by
# implementing two methods, dispatched on the observation type,
# `apply_observation!` (simulation side) and `observe` (analytical side).
# The dispatch is on the observation value, so there is no model type
# parameter.

# Rebuild a process with a replacement `Forcings`, keeping its core
# dynamics. Each process type implements this; the `with_*` helpers below
# are then defined once against the abstract model.
function _rebuild(m::BranchingProcess, f::Forcings)
    BranchingProcess(m.infectiousness, m.population_size, m.n_types,
        m.type_labels, m.progression, f)
end

function _rebuild(m::NetworkProcess, f::Forcings)
    NetworkProcess(m.adjacency, m.edge_probability, m.generation_time;
        attributes = f.attributes, interventions = f.interventions,
        observation = f.observation)
end

"""
    with_interventions(model, interventions)
    with_attributes(model, attributes)
    with_observation(model, observation)

Return a copy of `model` with one forcing replaced, keeping the others.
Equivalent to passing `interventions = …` / `attributes = …` /
`observation = …` to the process constructor. Use it to derive a
counterfactual model (e.g. a fitted model under a policy it was not fitted
with): a different forcing means a different model.
"""
function with_interventions(m::TransmissionModel, interventions)
    _rebuild(m, _mk_forcings(; attributes = _attributes(m),
        interventions, observation = _observation(m)))
end

function with_attributes(m::TransmissionModel, attributes)
    _rebuild(m,
        _mk_forcings(; attributes,
            interventions = _interventions(m), observation = _observation(m)))
end

function with_observation(m::TransmissionModel, observation::ObservationModel)
    _rebuild(m, _with_observation(_forcings(m), observation))
end

# All three share the docstring attached to `with_interventions`.
@doc (@doc with_interventions) with_attributes
@doc (@doc with_interventions) with_observation

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
