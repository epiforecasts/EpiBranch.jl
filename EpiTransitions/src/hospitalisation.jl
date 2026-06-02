"""
Cases are admitted to hospital with probability `probability` after a
`delay` drawn per case, measured from `from`. `from` defaults to
`:onset_time` but accepts any `Symbol` (state-dict key) or
`Function (ind) -> Real` — see [`Reporting`](@ref) for the anchor
semantics. If the anchor is `NaN`, the case is skipped.

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

Initialises: `:admitted = false`, `:admission_time = Inf`.
"""
Base.@kwdef struct Hospitalisation{D, P, F} <: AbstractClinicalTransition
    delay::D
    probability::P = 0.2
    from::F = :onset_time
end

EpiBranchCore.required_fields(h::Hospitalisation) = _from_required(h.from)

function EpiBranchCore.initialise_individual!(::Hospitalisation, individual, state)
    individual.state[:admitted] = false
    individual.state[:admission_time] = Inf
    return nothing
end

function EpiBranchCore.resolve_individual!(h::Hospitalisation, individual, state)
    anchor = _resolve_anchor(h.from, individual)
    isnan(anchor) && return nothing
    p = _resolve_probability(h.probability, state.rng, individual)
    rand(state.rng) < p || return nothing
    delay = _resolve_delay(h.delay, state.rng, individual)
    individual.state[:admitted] = true
    individual.state[:admission_time] = anchor + delay
    return nothing
end
