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
`:outcome_time` from the earliest. [`Death`](@ref EpiBranch.Transitions.Death) and [`Recovery`](@ref EpiBranch.Transitions.Recovery)
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
