"""
Cases are admitted to hospital with probability `probability` after a
`delay` drawn per case, measured from `from`. `from` defaults to
`:onset_time` but accepts any `Symbol` (state-dict key) or
`Function (ind) -> Real` — see [`Reporting`](@ref) for the anchor
semantics. If the anchor is not finite, the case is skipped.

Both `probability` and `delay` accept the heterogeneity shapes shared
across transitions: `Real`/`Distribution` for constants,
`Function (rng, ind) -> Real` for per-individual rules.

For *prerequisite-gated* admission (e.g. admit only cases that have
been reported, tested, contact-traced, vaccinated, or that satisfy any
other predicate on `ind.state`), express the gate inside the
`probability` function — return `0.0` when the gate is closed:

```julia
Hospitalisation(
    delay = LogNormal(2.0, 0.5),
    probability = (rng, ind) -> get(ind.state, :reported, false) ? 0.2 : 0.0
)
```

The same idiom covers any composite condition; no per-prerequisite
field is needed.

An admission drawn after the case's outcome (death or recovery) is reset to
"did not occur": see [`censor_after_outcome!`](@ref
EpiBranch.censor_after_outcome!). Chain `from = :admission_time` on a
terminal transition if admission should instead delay or otherwise change
the outcome.

Initialises: `:admitted = false`, `:admission_time = Inf`.
"""
Base.@kwdef struct Hospitalisation{D, P, F} <: AbstractClinicalTransition
    delay::D
    probability::P = 0.2
    from::F = :onset_time
end

required_fields(h::Hospitalisation) = _from_required(h.from)

function initialise_individual!(::Hospitalisation, individual, state)
    individual.state[:admitted] = false
    individual.state[:admission_time] = Inf
    return nothing
end

function resolve_individual!(h::Hospitalisation, individual, state)
    anchor = _resolve_anchor(h.from, individual)
    time = transition_time(
        state.rng, individual, anchor, h.delay;
        probability = h.probability
    )
    time === nothing && return nothing
    individual.state[:admitted] = true
    individual.state[:admission_time] = time
    return nothing
end

function censor_after_outcome!(::Hospitalisation, individual)
    _censor_event!(individual, :admitted, :admission_time)
    return nothing
end

function transition_loglik(h::Hospitalisation, individual::Individual)
    anchor = _resolve_anchor(h.from, individual)
    _anchor_ok(anchor) || return 0.0
    flag = individual.state[:admitted]::Bool
    time = flag ? individual.state[:admission_time] : Inf
    occurred = flag && time <= outcome_time(individual)
    ll = transition_term(
        h.probability, h.delay, individual, anchor, occurred;
        censor = min(infection_aborted_time(individual), outcome_time(individual))
    )
    occurred || return ll
    return ll + _delay_loglik(h.delay, time - anchor)
end
