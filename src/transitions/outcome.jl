"""
    Recovery(; delay, from = :onset_time)

Recovery, a step that ends the case: every case that reaches `from` (symptom
onset by default) recovers `delay` days later, unless an earlier terminal step
such as [`Death`](@ref) ends the case first. The earliest terminal step
becomes the case's `:outcome` and `:outcome_time`.

`delay` is a distribution, a fixed number of days or a function of the random
number generator and the individual, `(rng, ind) -> ...` (for example a
recovery delay that depends on age). `from` takes the same forms as in
[`Reporting`](@ref); cases that never reached it get no recovery time.
`Recovery` has no `probability`: it always happens once `from` is reached.

# Examples

```julia
# recover a mean of 10 days after onset, unless death comes first
Recovery(delay = Gamma(4.0, 2.5))
```
"""
Base.@kwdef struct Recovery{D, F} <: AbstractClinicalTransition
    delay::D
    from::F = :onset_time
end

required_fields(r::Recovery) = _from_required(r.from)
is_terminal(::Recovery) = true
terminal_target(::Recovery) = :recovered
# No `probability` field at all: unconditional once the anchor is reached.
terminal_certainty(::Recovery) = true

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
        # No gate of its own, so a candidate is missing only where an abort
        # undid it; anything else rules the individual out.
        isinf(infection_aborted_time(individual)) && return -Inf
        return transition_term(1.0, r.delay, individual, anchor, false)
    end
    return _delay_loglik(r.delay, t - anchor)
end

"""
    Death(; delay, probability, from = :onset_time)

Death, a step that ends the case: a case dies with probability `probability`,
`delay` days after symptom onset (or after `from`), unless an earlier terminal
step ends the case first.

!!! warning "The realised case fatality ratio can be lower than `probability`"
    `Death` competes with other terminal steps, and the earliest wins. With a
    competing [`Recovery`](@ref), a case drawn to die still recovers if its
    recovery time comes first, so fewer than `probability` of cases die (with
    equal delays, about half as many). For an exact case fatality ratio,
    replace `Recovery` with a terminal [`Transition`](@ref) and split the two
    outcomes with [`exclusive_probabilities`](@ref), so each case gets
    exactly one:

    ```julia
    death_p, recovered_p = exclusive_probabilities([0.64, 0.36])
    progression = [
        Death(delay = LogNormal(2.5, 0.4), probability = death_p),
        Transition(:recovered, from = :onset, delay = LogNormal(2.0, 0.4),
            probability = recovered_p, terminal = true),
    ]
    ```

    Giving death and recovery independent probabilities `CFR` and `1 - CFR`
    does not work: about `CFR * (1 - CFR)` of cases would qualify for both and
    another `CFR * (1 - CFR)` for neither, leaving them with no outcome. The
    alternative is to make the delay to death reliably shorter than any
    competing delay.

`probability` has no default, because the case fatality ratio is too
disease-specific for one; pass it explicitly, even if it is `0.0`. It is a
number or a function of the random number generator and the individual,
`(rng, ind) -> ...`, for example to make it depend on age:

```julia
Death(delay = LogNormal(2.5, 0.4),
      probability = (rng, ind) -> ind.state[:age] >= 80 ? 0.3 : 0.02)
```

`delay` is a distribution, a fixed number of days or such a function. `from`
takes the same forms as in [`Reporting`](@ref).

A vaccine that lowers mortality (a [`RingVaccination`](@ref) or
[`MassVaccination`](@ref) with `severity_efficacy`) acts through the same
function, using [`severity_efficacy`](@ref) and [`immunity_time`](@ref):

```julia
Death(delay = LogNormal(2.5, 0.4),
      probability = (rng, ind) ->
          immunity_time(ind) <= onset_time(ind) ?
              0.7 * (1 - severity_efficacy(ind)) : 0.7)
```

Comparing `immunity_time(ind)` with `onset_time(ind)` means a dose whose
protection had not developed by symptom onset gives none; testing
`is_vaccinated(ind)` alone would count it as protective. Combine it with an
age-dependent case fatality ratio by multiplying that ratio by
`1 - severity_efficacy(ind)` once the case is immune.
"""
Base.@kwdef struct Death{D, P, F} <: AbstractClinicalTransition
    delay::D
    probability::P
    from::F = :onset_time
end

required_fields(d::Death) = _from_required(d.from)
is_terminal(::Death) = true
terminal_target(::Death) = :died
terminal_certainty(d::Death) = _certain_probability(d.probability)

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
    ll = transition_term(d.probability, d.delay, individual, anchor, candidate)
    candidate || return ll
    return ll + _delay_loglik(d.delay, t - anchor)
end
