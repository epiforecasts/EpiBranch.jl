"""
Symptomatic cases are admitted to hospital with probability `probability`
after a `delay` drawn per case, measured from `:onset_time`. Asymptomatic
cases are never admitted.

Both `probability` and `delay` accept the heterogeneity shapes shared
across transitions: `probability` is a `Real` or `Function (rng, ind) -> Real`,
`delay` is a `Distribution` or `Function (rng, ind) -> Real`. Use the
function form for age-dependent admission rates, capacity-aware delays,
or anything else that should vary across cases.

For *prerequisite-gated* admission (e.g. admit only cases that have been
reported, tested, contact-traced, vaccinated, or that satisfy any other
predicate on `ind.state`), express the gate inside the `probability`
function — return `0.0` when the gate is closed:

```julia
Hospitalisation(
    delay = LogNormal(2.0, 0.5),
    probability = (rng, ind) -> get(ind.state, :reported, false) ? 0.2 : 0.0
)
```

The same idiom covers any composite condition; no per-prerequisite field
is needed.

Initialises: `:admitted = false`, `:admission_time = Inf`.

Requires `:onset_time` and `:asymptomatic`.
"""
Base.@kwdef struct Hospitalisation{D, P} <: AbstractClinicalTransition
    delay::D
    probability::P = 0.2
end

required_fields(::Hospitalisation) = [:onset_time, :asymptomatic]

function initialise_individual!(::Hospitalisation, individual, state)
    individual.state[:admitted] = false
    individual.state[:admission_time] = Inf
    return nothing
end

function resolve_individual!(h::Hospitalisation, individual, state)
    is_asymptomatic(individual) && return nothing
    ot = onset_time(individual)
    isnan(ot) && return nothing
    p = _resolve_probability(h.probability, state.rng, individual)
    rand(state.rng) < p || return nothing
    individual.state[:admitted] = true
    individual.state[:admission_time] = ot +
                                        _resolve_delay(h.delay, state.rng,
        individual)
    return nothing
end
