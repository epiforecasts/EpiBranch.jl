"""
    Transition(state; from = :infection, delay = …, rate = …, probability = 1.0, terminal = false)

A timed transition in a case's natural history: the case reaches `state`
a `delay` after it reached `from`, with probability `probability`. On each
individual it writes `state => true` and `Symbol(state, :_time) =>
from_time + delay`. When the transition does not happen — the probability
gate fails, or the `from` state was never reached — the flag stays `false`
and the time stays `Inf`.

`from` names an earlier state whose time this one is measured from:
`:infection` (the default) resolves to the individual's `infection_time`;
any other symbol `s` resolves to `Symbol(s, :_time)` in `ind.state` (so
`from = :onset` reads `:onset_time`). A `Function (ind) -> Real` is also
accepted, for anchors held as fields on the `Individual`. If the `from`
state's time is not finite (the upstream state was never reached, or an
asymptomatic case has a `NaN` onset), the transition is skipped.

The timing is given as exactly one of `delay` or `rate`. `delay` is a fixed
`Real`, a `Distribution`, or an `(rng, ind) -> Real`, resolved per individual.
`rate = r` is the compartmental alternative: an exponential (Markovian)
transition with hazard `r`, i.e. `delay = Exponential(1 / r)` (mean `1 / r`).
`probability` is a `Real` or `(rng, ind) -> Real`, resolved per individual.

`terminal = true` marks the transition as ending the case: it joins the
competing-terminal arbitration, where `:outcome` and `:outcome_time` take
the earliest terminal that happened. This generalises [`Reporting`](@ref),
[`Death`](@ref) and the rest, which are this transition with a fixed state
and bespoke key names.

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

function Transition(state::Symbol; delay = nothing, rate = nothing,
        from = :infection, probability = 1.0, terminal::Bool = false)
    d = _transition_delay(delay, rate)
    return Transition(state, Symbol(state, :_time), d, probability, from, terminal)
end

# Resolve a transition's timing from exactly one of `delay` or `rate`. A rate
# `r` is an exponential (Markovian) transition with hazard `r`, so the delay is
# `Exponential(1 / r)` with mean `1 / r`.
function _transition_delay(delay, rate)
    (delay === nothing) == (rate === nothing) && throw(ArgumentError(
        "Transition needs exactly one of `delay` or `rate`"))
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
    _anchor_ok(anchor) || return nothing
    p = _resolve_probability(t.probability, state.rng, individual)
    rand(state.rng) < p || return nothing
    individual.state[t.state] = true
    individual.state[t.time_key] = anchor + _resolve_delay(t.delay, state.rng, individual)
    return nothing
end

is_terminal(t::Transition) = t.terminal
function terminal_event(t::Transition, individual::Individual{T}) where {T}
    t.terminal || return nothing
    tm = convert(T, get(individual.state, t.time_key, T(Inf)))
    return isfinite(tm) ? (tm, t.state) : nothing
end
