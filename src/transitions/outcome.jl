"""
Terminal transition: the case recovers. A candidate recovery time is
drawn from `delay` and added to the value of `from`. `from` defaults
to `:onset_time` but accepts any `Symbol` (state-dict key) or
`Function (ind) -> Real` — see [`Reporting`](@ref) for the anchor
semantics. If the anchor is not finite, no recovery candidate is produced.

`delay` is a `Distribution` or a `Function (rng, ind) -> Real` for
per-individual heterogeneity (e.g. age-conditional recovery delay).

Initialises `:recovery_candidate_time = Inf`.

`Recovery` and [`Death`](@ref) compose as competing terminal events:
whichever has the earliest candidate time becomes the case's `:outcome`.
Other user-defined terminal transitions (with `is_terminal = true` and
a `terminal_event` method) participate in the same arbitration.
"""
Base.@kwdef struct Recovery{D, F} <: AbstractClinicalTransition
    delay::D
    from::F = :onset_time
end

required_fields(r::Recovery) = _from_required(r.from)
is_terminal(::Recovery) = true
_terminal_target(::Recovery) = :recovered

function initialise_individual!(::Recovery, individual, state)
    individual.state[:recovery_candidate_time] = Inf
    return nothing
end

function resolve_individual!(r::Recovery, individual, state)
    anchor = _resolve_anchor(r.from, individual)
    time = transition_time(state.rng, individual, anchor, r.delay)
    time === nothing && return nothing
    individual.state[:recovery_candidate_time] = time
    return nothing
end

function terminal_event(::Recovery, individual::Individual{T}) where {T}
    t = convert(T, get(individual.state, :recovery_candidate_time, T(Inf)))
    return isfinite(t) ? (t, :recovered) : nothing
end

# Recovery has no `probability` gate: once its anchor is reached, a candidate
# time is drawn unconditionally, so a non-finite candidate there is
# impossible under the model.
function transition_loglik(r::Recovery, individual::Individual)
    anchor = _resolve_anchor(r.from, individual)
    _anchor_ok(anchor) || return 0.0
    t = individual.state[:recovery_candidate_time]
    if !isfinite(t)
        abort = _infection_aborted_time(individual)
        isinf(abort) && return -Inf
        return _censored_loglik(1.0, r.delay, individual, anchor, abort)
    end
    return _delay_loglik(r.delay, t - anchor)
end

"""
Terminal transition: the case dies. When death is drawn, a candidate
death time is produced by adding a sample from `delay` to the value of
`from`. `from` defaults to `:onset_time` but accepts any `Symbol` or
`Function (ind) -> Real` — see [`Reporting`](@ref) for the anchor
semantics.

`probability` is required (no default) and accepts a `Real`, a
`Distribution`, or a `Function (rng, ind) -> Real`. The probability is
too pathogen-specific for a sensible default — pass an explicit value,
even if it is `0.0`. Use the function form for age- or risk-conditional
rates:

```julia
Death(delay = LogNormal(2.5, 0.4),
      probability = (rng, ind) -> ind.state[:age] >= 80 ? 0.3 : 0.02)
```

`delay` accepts a `Distribution` or `Function (rng, ind) -> Real`,
making time-to-death heterogeneity available the same way.

A vaccine that lowers mortality rather than blocking transmission (a
[`RingVaccination`](@ref) or [`MassVaccination`](@ref) with a
`severity_efficacy`) is read the same way, via the
[`severity_efficacy`](@ref) and [`immunity_time`](@ref) accessors:

```julia
Death(delay = LogNormal(2.5, 0.4),
      probability = (rng, ind) ->
          immunity_time(ind) <= onset_time(ind) ?
              0.7 * (1 - severity_efficacy(ind)) : 0.7)
```

`immunity_time(ind) <= onset_time(ind)` is what makes a dose whose
immunity has not yet developed by onset confer no protection; comparing
against `is_vaccinated(ind)` alone would count it as protective anyway.
It composes with an age-conditional CFR the same way: multiply whatever
base probability applies by `1 - severity_efficacy(ind)` once immune.

Initialises `:death_candidate_time = Inf`.

`Death` and [`Recovery`](@ref) compose as competing terminal events, resolved
by earliest candidate time. `probability` is the probability death *enters*
that race, so the realised fraction dying equals it only when death's candidate
time reliably precedes any competing recovery/removal — otherwise the realised
case-fatality is lower (with equal delays and a competing `Recovery`, roughly
halved). Gating `Death` and a second terminal transition independently — say
at `CFR` and `1 - CFR` — does not fix this either: each draws its own
Bernoulli, so about `CFR * (1 - CFR)` of cases fire both (resolved by
whichever candidate time is earlier) and another `CFR * (1 - CFR)` fire
neither, leaving `:outcome` unset. `Recovery` has no `probability` of its own
to gate this way in any case — it always fires once its anchor is reached.
For an exact CFR, replace the competing `Recovery` with a `Transition`
carrying its own `probability`, and build both probabilities with
[`exclusive_probabilities`](@ref), which shares one draw between the two so
exactly one of them fires; or make death's delay dominate the competing one.
"""
Base.@kwdef struct Death{D, P, F} <: AbstractClinicalTransition
    delay::D
    probability::P
    from::F = :onset_time
end

required_fields(d::Death) = _from_required(d.from)
is_terminal(::Death) = true
_terminal_target(::Death) = :died

function initialise_individual!(::Death, individual, state)
    individual.state[:death_candidate_time] = Inf
    return nothing
end

function resolve_individual!(d::Death, individual, state)
    anchor = _resolve_anchor(d.from, individual)
    time = transition_time(
        state.rng, individual, anchor, d.delay;
        probability = d.probability
    )
    time === nothing && return nothing
    individual.state[:death_candidate_time] = time
    return nothing
end

function terminal_event(::Death, individual::Individual{T}) where {T}
    t = convert(T, get(individual.state, :death_candidate_time, T(Inf)))
    return isfinite(t) ? (t, :died) : nothing
end

function transition_loglik(d::Death, individual::Individual)
    anchor = _resolve_anchor(d.from, individual)
    _anchor_ok(anchor) || return 0.0
    t = individual.state[:death_candidate_time]
    candidate = isfinite(t)
    abort = _infection_aborted_time(individual)
    candidate || isinf(abort) ||
        return _censored_loglik(d.probability, d.delay, individual, anchor, abort)
    ll = _probability_loglik(d.probability, candidate, individual)
    candidate || return ll
    return ll + _delay_loglik(d.delay, t - anchor)
end
