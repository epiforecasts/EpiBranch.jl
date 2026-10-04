"""
Base type for clinical-state transitions. Subtypes implement
`initialise_individual!` (set default state on a new case) and
`resolve_individual!` (draw the transition's timing and probability).

A transition writes its outcome to one or more keys on `individual.state`,
under names it owns. Other transitions and the line-list projection read
from these keys. Transitions are composable: stack them in a vector and
the engine applies them in order at case-creation time, after attributes
and interventions have run.

Terminal transitions — those that end the case — declare themselves by
returning `true` from [`is_terminal`](@ref) and implement
[`terminal_event`](@ref). After all transitions resolve for an
individual, the engine collects every terminal candidate (across every
transition that declared itself terminal) and assigns `:outcome` and
`:outcome_time` from the earliest. [`Death`](@ref) and [`Recovery`](@ref)
are the built-in pair, but the framework is open: a user-defined
`LostToFollowUp`, `MovedAway`, or disease-specific terminal state plugs
in by adding the same two methods and dropping the struct into the
transitions vector. Competing-risks arbitration handles the rest. Also
implement [`terminal_target`](@ref) so a window's `until`-coverage check
can see the new terminal state (see the Extending guide for a worked
example); without it, the check simply cannot tell the state apart from
one no transition reaches.

See also [`AbstractIntervention`](@ref) — transitions are the clinical
analogue: where interventions are policy applied to a case, transitions
are biology happening to a case.

The abstract type itself is declared in `src/types.jl` to allow
`SimulationState` to hold a typed `transitions` vector; the interface
methods live here.
"""
AbstractClinicalTransition

"""Set up transition-specific fields on a newly created individual. Default: no-op."""
initialise_individual!(::AbstractClinicalTransition, individual, state) = nothing

"""Draw the transition's timing/probability and write its outcome to state. Default: no-op."""
resolve_individual!(::AbstractClinicalTransition, individual, state) = nothing

"""Whether this transition is terminal (i.e. ends the case). Default: false."""
is_terminal(::AbstractClinicalTransition) = false

"""
    terminal_target(transition::AbstractClinicalTransition) -> Union{Nothing, Symbol}

The state label a terminal transition writes, known without an
individual (unlike [`terminal_event`](@ref), which needs one to resolve
the *time*). The `until`-coverage check in a process's progression
validation reads this to warn when a terminal state is missing from a
window's `until`; a terminal transition that does not override this
(default `nothing`) is simply not checkable there and stays silently
exempt from that warning. Non-terminal transitions never need it.
"""
terminal_target(::AbstractClinicalTransition) = nothing

# Whether a terminal transition, taken alone, is guaranteed to occur once its
# anchor is reached: `true` when nothing can stop it, `false` when a constant
# gate sits below certainty (including a gate fixed at 0, which never lets a
# case through), `missing` when it depends on the individual (a `Function`-valued
# gate, say) and so cannot be judged without one. Used only
# by `_warn_incomplete_terminal_coverage` in model_spec.jl, to spot a
# progression whose terminal transitions are all independently gated below
# certainty. A transition that does not override this is simply treated as
# unknown rather than guessed at from its field layout; override it alongside
# `is_terminal` and `terminal_event` to make that check see past it. Not part
# of the transition interface documented in `extending.md`.
terminal_certainty(::AbstractClinicalTransition) = missing

# Whether a constant `probability` guarantees the gate it sits behind passes;
# a `Function`-valued one depends on the individual and so is unknowable here.
_certain_probability(p::Real) = isone(p)
_certain_probability(p) = missing

"""
    terminal_event(transition, individual) -> Union{Nothing, Tuple{Float64, Symbol}}

For terminal transitions, return `(time, label)` if this transition would
end the case (e.g. `(11.3, :died)`), or `nothing` if it does not occur for
this case. Called after all `resolve_individual!`s have run. The engine
takes the earliest terminal candidate across all transitions and writes
`:outcome` (Symbol) and `:outcome_time` (Float64) to the individual's
state.

Non-terminal transitions never see this method called.
"""
terminal_event(::AbstractClinicalTransition, individual) = nothing

"""Fields a transition requires on individuals (set by `attributes`). Default: none."""
required_fields(::AbstractClinicalTransition) = Symbol[]

# ── Heterogeneity helpers ───────────────────────────────────────────
#
# Transition fields (`probability`, `delay`) accept three shapes,
# resolved per individual at `resolve_individual!` time:
#
# - a `Real` / `Distribution`: constant across the population.
# - a `Function (rng, ind) -> value`: arbitrary per-individual rule.
#   Use this for age-dependent CFRs, vulnerability-conditioned delays,
#   risk-group-specific reporting, etc. The function is called with the
#   simulation RNG and the individual; return the probability (as a
#   `Real`) or the delay (as a `Real` time, typically days).
#
# Distribution-valued delays sample from the distribution. Function
# delays return a sample directly. The pattern matches `transmission_traits`
# and `clinical_presentation` so heterogeneity is configured the same way
# across the package.
_resolve_probability(p::Real, rng, ind) = float(p)
_resolve_probability(f, rng, ind) = float(f(rng, ind))

_resolve_delay(d::Distribution, rng, ind) = float(rand(rng, d))
_resolve_delay(x::Real, rng, ind) = float(x)         # a fixed, deterministic delay
_resolve_delay(f, rng, ind) = float(f(rng, ind))

# ── Evaluating ──────────────────────────────────────────────────────
#
# The reverse of `_resolve_delay`/`_resolve_probability`: given a resolved
# outcome, the log-density of having drawn it. A `Function` delay has no
# density family to evaluate, so it is rejected rather than silently ignored.
_delay_loglik(d::Distribution, dt) = logpdf(d, dt)
_delay_loglik(x::Real, dt) = isapprox(dt, x) ? 0.0 : -Inf
function _delay_loglik(f, dt)
    throw(
        ArgumentError(
            "a `Function` delay has no density to evaluate; `progression_loglik` " *
                "needs a `Distribution` (or a fixed `Real`) for every delay it evaluates"
        )
    )
end

# Used only to evaluate a `probability` gate: it deliberately implements no
# generation methods, so a callable that actually draws from its `rng`
# argument (rather than merely accepting it, as every deterministic
# per-individual gate does) fails loudly here instead of drawing a fresh,
# uncontrolled value and silently caching it on the individual.
struct _NoRandRNG <: Random.AbstractRNG end

# One bucket of a shared draw: the transition is selected when `lo <= u < hi`.
# Returning 0.0/1.0 (rather than deciding directly) keeps this a `probability`
# callable like any other, so it composes with `_transition_selected`'s own
# `rand(rng) < p` unchanged — that draw is now deterministic, since `u` alone
# decided the outcome. A type rather than a closure, so the likelihood can read
# the bucket's width: the probability the shared draw selects it, which a run's
# 0 or 1 hides.
struct _ExclusiveGate{T <: Real}
    key::Symbol
    lo::T
    hi::T
    total::T
    owns_shortfall::Bool   # the one bucket that holds the group's shortfall
end
function (g::_ExclusiveGate)(rng, ind)
    u = get!(() -> rand(rng), ind.state, g.key)
    return (g.lo <= u < g.hi) ? 1.0 : 0.0
end

# A gate's own probability. `probability` resolves with `_NoRandRNG()`: a
# callable gate is expected to be a deterministic function of the individual
# (as every built-in and documented example is), not of the RNG draw that also
# consumes it during simulation; one that does draw is rejected rather than
# evaluated with an arbitrary, non-reproducible value.
function _probability_value(probability, ind)
    return try
        _resolve_probability(probability, _NoRandRNG(), ind)
    catch e
        e isa MethodError && parentmodule(e.f) === Random || rethrow()
        throw(
            ArgumentError(
                "a `probability` callable drew from its `rng` argument while being " *
                    "evaluated; `progression_loglik` needs `probability` to be a " *
                    "deterministic function of the individual alone. A callable " *
                    "gate has to replay from the individual alone; evaluate " *
                    "simulated individuals, not hand-built ones, with one that " *
                    "reads state a simulation wrote"
            )
        )
    end
end

# The log-likelihood contribution of a probability gate, given whether it
# passed.
function _probability_loglik(probability, passed, ind)
    p = _probability_value(probability, ind)
    return passed ? log(p) : log1p(-p)
end

"""
    transition_term(probability, delay, individual, anchor, occurred)

A transition's gate term for [`transition_loglik`](@ref
EpiBranch.transition_loglik), given whether the transition happened: the
log-probability of the gate either way, to which the caller adds the delay
density when it did.

Call it rather than reading `probability` directly. It handles the two cases a
custom transition would otherwise get wrong: a gate built by
[`exclusive_probabilities`](@ref) contributes the width of the bucket its
group's shared draw selected, not the 0 or 1 the gate itself returns; and a
transition an abort undid is censored at
[`infection_aborted_time`](@ref EpiBranch.infection_aborted_time) rather than
read as a gate that failed.
"""
function transition_term(probability, delay, ind, anchor, occurred)
    occurred && return _probability_loglik(probability, true, ind)
    abort = infection_aborted_time(ind)
    isinf(abort) && return _probability_loglik(probability, false, ind)
    return _censored_loglik(probability, delay, ind, anchor, abort)
end

# A transition an abort undid: `_resolve_before_abort!` restored its flag and
# cleared its time, so what the individual records is that the transition would
# have taken effect at or after the abort. Its contribution is the probability
# of exactly that — the gate failing, or the gate passing and the delay landing
# no earlier than the abort — which censors the transition at the abort instead
# of reading an undone transition as a gate that failed.
function _censored_loglik(probability, delay, ind, anchor, abort)
    p = _probability_value(probability, ind)
    elapsed = abort - anchor
    isone(p) && return _delay_logccdf(delay, elapsed)
    return log1p(-p * _delay_cdf(delay, elapsed))
end

# A shared draw decides the group once, so the censored term reads the draw
# rather than the gate's own 0 or 1: the bucket it selected keeps its width and
# censors its delay at the abort, and a bucket it passed over says what it says
# in any other case.
function _censored_loglik(g::_ExclusiveGate, delay, ind, anchor, abort)
    _exclusive_selected(g, ind) || return _probability_loglik(g, false, ind)
    return log(g.hi - g.lo) + _delay_logccdf(delay, abort - anchor)
end

_delay_cdf(d::Distribution, t) = cdf(d, t)
_delay_cdf(x::Real, t) = x < t ? 1.0 : 0.0
_delay_cdf(f, t) = _delay_loglik(f, t)             # a `Function` delay throws
_delay_logccdf(d::Distribution, t) = logccdf(d, t)
_delay_logccdf(x::Real, t) = x < t ? -Inf : 0.0
_delay_logccdf(f, t) = _delay_loglik(f, t)

"""
    transition_loglik(t::AbstractClinicalTransition, individual) -> Float64

The log-likelihood contribution of `individual`'s outcome under transition
`t`: the probability of the gate it passed or failed, plus the delay density
at the time the transition occurred. Called by [`progression_loglik`](@ref) once per
transition per individual; `0.0` when the transition's anchor was never
reached (it took no part in the individual's history).

Implemented for the built-in transitions ([`Transition`](@ref),
[`Reporting`](@ref), [`Hospitalisation`](@ref), [`Death`](@ref),
[`Recovery`](@ref)). A custom `<: AbstractClinicalTransition` used with
[`progression_loglik`](@ref) needs its own method, reading back the state
keys its `resolve_individual!` writes; `delay` must be a `Distribution` or a
fixed `Real` — a raw `Function` delay has no density.
"""
function transition_loglik(t::AbstractClinicalTransition, individual)
    throw(
        ArgumentError(
            "$(nameof(typeof(t))) needs a method for `EpiBranch.transition_loglik` " *
                "evaluating the outcome its `resolve_individual!` writes to " *
                "`individual.state`; see `progression_loglik`"
        )
    )
end

"""
    exclusive_probabilities(ps::AbstractVector{<:Real}) -> Vector

Build matched `probability` callables for `length(ps)` sibling transitions
whose outcomes are meant to be mutually exclusive — an exact case-fatality
ratio split between death and recovery, say.

Passing raw probabilities straight to each sibling's own `probability` field
draws an *independent* Bernoulli per transition (`_transition_selected`
consumes its own `rand(rng)`): two terminal transitions gated at `p` and
`1 - p` then both occur (resolved by whichever candidate time comes first) on
about `p(1 - p)` of cases, and neither occurs — leaving `:outcome` unset — on
another `p(1 - p)`.

Each gate this returns reads a single shared uniform draw per case instead:
the first sibling to resolve draws it and caches it on the individual, and
every sibling reads the same value. The case's draw lands in exactly one of the
`ps`-sized buckets, so the outcomes partition the population in the given
proportions, with no case counted twice and none dropped except by design
(see below).

Assign each gate to its sibling's `probability` as it comes. A gate wrapped in
a callable of your own is a plain 0-or-1 probability to the rest of the
package: the likelihood reads that instead of the bucket's width, and an
aborted infection drops the group's draw. Fold a per-individual modifier into
`ps` instead of around the gate.

`ps` must be non-negative and sum to at most `1`; a shortfall between
`sum(ps)` and `1` is the (intentional) probability that none of the siblings
occurs — pair it with an unconditional terminal transition, or expect some
cases to reach no terminal state.

[`progression_loglik`](@ref) evaluates such a group as one event: the bucket the
draw selected contributes the log of its own width, and the siblings it passed
over contribute nothing. A group with a shortfall also needs the draw a
simulation cached, so keep every one of its gates in the `progression`.

# Examples

```julia
death_p, recovered_p = exclusive_probabilities([0.64, 0.36])
progression = [
    Death(delay = LogNormal(2.5, 0.4), probability = death_p),
    Transition(:recovered, from = :onset, delay = LogNormal(2.0, 0.4),
        probability = recovered_p, terminal = true),
]
```
"""
function exclusive_probabilities(ps::AbstractVector{<:Real})
    all(>=(0), ps) || throw(
        ArgumentError(
            "exclusive_probabilities needs non-negative probabilities, got $ps"
        )
    )
    total = sum(ps)
    slack = sqrt(eps(float(total)))
    total <= 1 + slack || throw(
        ArgumentError(
            "exclusive_probabilities needs probabilities summing to at most 1, got $total"
        )
    )
    # A sum that reaches 1 up to its own rounding leaves no shortfall at all.
    shortfall = total < 1 - slack
    key = gensym(:exclusive_draw)
    bounds = cumsum(ps)
    return [
        _ExclusiveGate(
            key, i == 1 ? zero(total) : bounds[i - 1], bounds[i], total,
            shortfall && i == lastindex(ps)
        )
            for i in eachindex(ps)
    ]
end


# The state keys a transition reads that belong to a group of siblings rather
# than to itself, which an aborted infection's undo therefore keeps. Only a
# shared-draw gate has one, so this asks the transition's own gate — including
# a custom transition's, which keeps it under `probability` as every documented
# one does. A transition holding its gate anywhere else declares its own
# method.
function _shared_draw_keys(t::AbstractClinicalTransition)
    return hasproperty(t, :probability) ? _shared_draw_keys(t.probability) : ()
end
_shared_draw_keys(g::_ExclusiveGate) = (g.key,)
_shared_draw_keys(probability) = ()

# A shared draw is one event, so the bucket it selected holds the whole gate
# term and the siblings it passed over say nothing more. The group's shortfall
# — the probability that it selected none of them, when `sum(ps) < 1` — belongs
# to the group once, so the last bucket is the one that holds it, reading the
# draw a simulation cached.
function _probability_loglik(g::_ExclusiveGate, passed, ind)
    passed && return log(g.hi - g.lo)
    g.owns_shortfall || return 0.0
    return _exclusive_draw(g, ind) >= g.total ? log1p(-g.total) : 0.0
end

# The draw the group shared, which only a simulation can have cached.
function _exclusive_draw(g::_ExclusiveGate, ind)
    u = get(ind.state, g.key, nothing)
    u === nothing && throw(
        ArgumentError(
            "a shared-draw gate needs the draw `resolve_individual!` cached " *
                "under `:$(g.key)`, which this individual does not hold. A " *
                "group whose probabilities leave a shortfall, or one censored " *
                "by an aborted infection, reads that draw, so build the " *
                "individual through a simulation or give the group " *
                "probabilities summing to 1"
        )
    )
    return u
end
_exclusive_selected(g::_ExclusiveGate, ind) = g.lo <= _exclusive_draw(g, ind) < g.hi

"""
    transition_time(rng, individual, start_time, delay; probability = nothing)

Sample a clinical event time from a finite `start_time`, returning `nothing`
when the event is absent. `delay` accepts a real number, distribution or callable
`(rng, individual) -> Real`. `probability` accepts a real number or callable with
the same arguments. A supplied probability consumes one uniform draw, including
when it is zero or one; `nothing` skips that draw. An absent starting event
(non-finite `start_time`) consumes no random draws.

The caller resolves the starting event and writes the returned time to its own
state keys. Terminal-event arbitration remains the caller's responsibility.
"""
function transition_time(rng, individual, start_time, delay; probability = nothing)
    _transition_selected(rng, individual, start_time, probability) || return nothing
    return start_time + _resolve_delay(delay, rng, individual)
end

function _transition_selected(rng, individual, start_time, probability)
    _anchor_ok(start_time) || return false
    probability === nothing && return true
    p = _resolve_probability(probability, rng, individual)
    return rand(rng) < p
end

# Anchor for a transition's `delay`. A `Symbol` is looked up in
# `ind.state` (e.g. `:onset_time`, `:test_time`, `:admission_time`); a
# `Function (ind) -> Real` is called directly (use this for fields on
# `Individual` itself, e.g. `ind.infection_time`, or for composite
# anchors). Returning `NaN` signals "no anchor" and the transition is
# skipped.
function _resolve_anchor(s::Symbol, ind::Individual{T}) where {T}
    return convert(T, get(ind.state, s, T(NaN)))
end
_resolve_anchor(f, ind) = float(f(ind))

# A transition occurs only from a finite anchor. An anchor is missing (`NaN`)
# when the `from` key was never written, and non-finite (`Inf`) when the
# upstream transition initialised the key but never occurred — both mean "the
# `from` state was not reached", so the transition must be skipped. Guarding
# on `isfinite` (not `isnan`) keeps the two subsystems in step; the generic
# `Transition` uses the same check.
_anchor_ok(anchor) = isfinite(anchor)

# By default each transition's `required_fields` returns `[:onset_time]`
# so the simulation start-up validator catches missing
# `clinical_presentation` setup. When the user overrides `from` to
# anchor on something other than onset, the required field is the
# user's responsibility — typically a downstream transition's output
# key (e.g. `:test_time`) that isn't set by attributes anyway.
_from_required(s::Symbol) = s === :onset_time ? [:onset_time] : Symbol[]
_from_required(_) = Symbol[]

"""
    _finalise_terminal!(individual, transitions)

After all `resolve_individual!`s have run, collect terminal candidates
across all terminal transitions and set `:outcome` and `:outcome_time`
to the earliest. If no terminal transition occurs, neither key is set.
"""
function _finalise_terminal!(individual, transitions)
    best_time = Inf
    best_label = :none
    has_any = false
    for t in transitions
        is_terminal(t) || continue
        ev = terminal_event(t, individual)
        ev === nothing && continue
        time, label = ev
        if time < best_time
            best_time = time
            best_label = label
            has_any = true
        end
    end
    if has_any
        individual.state[:outcome_time] = best_time
        individual.state[:outcome] = best_label
    end
    return nothing
end
