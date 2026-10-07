"""
    Transition(state; from = :infection, delay = …, rate = …, probability = 1.0, terminal = false)

One step in a case's natural history, such as becoming infectious after a
latent period or recovering after an infectious period: the case reaches
`state` a `delay` (in days) after it reached `from`, with probability
`probability`. If the step does not happen, because it was not drawn or the
case never reached `from`, the case simply never reaches `state`.

- `from`: the earlier state the delay is measured from. `:infection` (the
  default) is the time of infection; `from = :onset` measures from symptom
  onset, and in general `from = :s` measures from the time recorded for state
  `s`. A function of the individual returning a time is also accepted. If the
  case never reached `from` (including asymptomatic cases, whose onset time is
  `NaN`), the step is skipped.
- `delay` or `rate` (give exactly one): `delay` is a fixed number of days, a
  distribution, or a function of the random number generator and the
  individual, `(rng, ind) -> ...`, drawn for each case. `rate = r` is the
  compartmental-model alternative, an exponentially distributed delay with
  mean `1 / r` days (`delay = Exponential(1 / r)`).
- `probability`: the chance the step happens, a number or a function
  `(rng, ind) -> ...`.
- `terminal = true` marks a step that ends the case. When a case can reach
  several terminal steps, the earliest becomes its outcome (`:outcome`,
  `:outcome_time`).

Each case records `state` as reached (`true`/`false`) and the time it was
reached (`Inf` if never), under the name `Symbol(state, :_time)`, so
`Transition(:infectious, ...)` gives `:infectious_time`. [`Reporting`](@ref),
[`Hospitalisation`](@ref), [`Recovery`](@ref) and [`Death`](@ref) are ready-made
versions of this step with fixed names.

# Examples

```julia
# latent period: infection → onset of infectiousness
Transition(:infectious, from = :infection, delay = LogNormal(1.0, 0.4))

# infectious period as a recovery rate (exponential, mean 1/γ)
Transition(:recovered, from = :infectious, rate = 1 / 6, terminal = true)

# severity branch, then death from the severe state
Transition(:severe, from = :onset,  delay = Gamma(2, 2), probability = 0.3)
Transition(:died,   from = :severe, delay = Gamma(2, 3), probability = 0.6, terminal = true)
```
"""
struct Transition{D, P, F} <: AbstractClinicalTransition
    state::Symbol
    time_key::Symbol
    delay::D
    probability::P
    from::F
    terminal::Bool
end

function Transition(
        state::Symbol; delay = nothing, rate = nothing,
        from = :infection, probability = 1.0, terminal::Bool = false
    )
    d = _transition_delay(delay, rate)
    return Transition(state, Symbol(state, :_time), d, probability, from, terminal)
end

# Resolve a transition's timing from exactly one of `delay` or `rate`. A rate
# `r` is an exponential (Markovian) transition with hazard `r`, so the delay is
# `Exponential(1 / r)` with mean `1 / r`.
function _transition_delay(delay, rate)
    (delay === nothing) == (rate === nothing) && throw(
        ArgumentError(
            "Transition needs exactly one of `delay` or `rate`"
        )
    )
    rate === nothing && return delay
    rate > 0 || throw(ArgumentError("rate must be positive, got $rate"))
    return Exponential(1 / rate)
end

# Time of the `from` state for this individual. `:infection` is the
# infection time (a field, not a state key); any other state name `s` is
# `Symbol(s, :_time)` in `ind.state`; a function is evaluated directly.
function _state_time(ind::Individual{T}, from::Symbol) where {T}
    from === :infection && return ind.infection_time
    return convert(T, get(ind.state, Symbol(from, :_time), T(NaN)))
end
_state_time(ind, from) = float(from(ind))

# `from` states other than `:infection` are usually produced by an upstream
# transition rather than by `attributes`, so the start-up validator cannot
# see them; an unreached state simply yields a non-finite time and the
# transition skips. So no fields are required up front.
required_fields(::Transition) = Symbol[]

function initialise_individual!(t::Transition, individual, state)
    individual.state[t.state] = false
    individual.state[t.time_key] = Inf
    return nothing
end

function resolve_individual!(t::Transition, individual, state)
    anchor = _state_time(individual, t.from)
    _transition_selected(state.rng, individual, anchor, t.probability) || return nothing
    # Delay callbacks can read the newly reached state.
    individual.state[t.state] = true
    individual.state[t.time_key] = anchor + _resolve_delay(t.delay, state.rng, individual)
    return nothing
end

is_terminal(t::Transition) = t.terminal
terminal_target(t::Transition) = t.terminal ? t.state : nothing
terminal_certainty(t::Transition) = t.terminal ? _certain_probability(t.probability) : missing
function terminal_event(t::Transition, individual::Individual{T}) where {T}
    t.terminal || return nothing
    tm = convert(T, get(individual.state, t.time_key, T(Inf)))
    return isfinite(tm) ? (tm, t.state) : nothing
end

function transition_loglik(t::Transition, individual::Individual)
    anchor = _state_time(individual, t.from)
    _anchor_ok(anchor) || return 0.0
    occurred = individual.state[t.state]::Bool
    ll = transition_term(t.probability, t.delay, individual, anchor, occurred)
    occurred || return ll
    return ll + _delay_loglik(t.delay, individual.state[t.time_key] - anchor)
end
