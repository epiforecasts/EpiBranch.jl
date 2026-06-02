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
`:outcome_time` from the earliest. [`Death`](@ref EpiTransitions.Death) and [`Recovery`](@ref EpiTransitions.Recovery)
are the built-in pair, but the framework is open: a user-defined
`LostToFollowUp`, `MovedAway`, or disease-specific terminal state plugs
in by adding the same two methods and dropping the struct into the
transitions vector. Competing-risks arbitration handles the rest.

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
    terminal_event(transition, individual) -> Union{Nothing, Tuple{Float64, Symbol}}

For terminal transitions, return `(time, label)` if this transition would
end the case (e.g. `(11.3, :died)`), or `nothing` if it does not fire for
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
_resolve_delay(f, rng, ind) = float(f(rng, ind))

# Anchor for a transition's `delay`. A `Symbol` is looked up in
# `ind.state` (e.g. `:onset_time`, `:test_time`, `:admission_time`); a
# `Function (ind) -> Real` is called directly (use this for fields on
# `Individual` itself, e.g. `ind.infection_time`, or for composite
# anchors). Returning `NaN` signals "no anchor" and the transition is
# skipped.
_resolve_anchor(s::Symbol, ind) = get(ind.state, s, NaN)::Float64
_resolve_anchor(f, ind) = float(f(ind))

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
to the earliest. If no terminal transition fires, neither key is set.
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
