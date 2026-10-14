"""
Base type for all interventions. Subtypes implement one or more of:
`initialise_individual!`, `resolve_individual!`, `trace_contacts!`,
`intervention_actions`, `competing_risk`.

To support time-based scheduling (`Scheduled(iv; start_time=...)`), also
implement [`intervention_time`](@ref) and [`reset!`](@ref). The default
implementations return `-Inf` and a no-op respectively, which is correct
for interventions whose effect time is always considered "now".

Tree-shaping interventions (a hard cap on offspring per parent,
gathering-size limits, etc.) are expressed by passing a state-aware
function-form offspring distribution to [`BranchingProcess`](@ref) —
not via the intervention protocol. See
[Change how many people each case infects](@ref) for an example.
"""
abstract type AbstractIntervention end

"""Set up intervention-specific fields on a newly created individual. Default: no-op."""
initialise_individual!(::AbstractIntervention, individual, state) = nothing

"""
    resolve_individual!(intervention, case, state)

Set the intervention's state on a case before it transmits. `Isolation`, for
example, records when the case isolates. Default: no-op.

Both engines call it once for each infection, as the case is settled: after
its infection time is fixed and its clinical transitions have run, and before
it infects anyone. It is never called on an uninfected individual, even a
contact that `keep_active` keeps in the simulation to generate contacts of
its own. While it runs,
`state.max_infection_time` is the case's infection time,
`state.current_generation` is the case's generation and
`state.cumulative_cases` counts the case and every case settled before it. A
[`Scheduled`](@ref) gate is evaluated against these values.

The continuous-time models settle cases in order of infection time. The
generation engine settles a generation's cases together, once its infections
are decided. There `state.cumulative_cases` counts the cases of earlier
generations and those of the case's own generation infected no later than it;
a case of a later generation is not counted, even one infected earlier.
"""
resolve_individual!(::AbstractIntervention, individual, state) = nothing

"""
    apply_post_transmission!(intervention, state, new_contacts)

Deprecated: implement [`trace_contacts!`](@ref) for what an intervention does
to the contacts a case reached, and [`intervention_actions`](@ref) for actions
with a delivery time. Both engines call those once per case.

The generation engine used to hand this hook a whole generation's new
contacts at once. It still does for an intervention that defines a method of
its own, with a deprecation warning, so an intervention written against it
keeps working there. Called on an intervention without one, it traces each
contact from its `parent_id` and then admits the intervention's actions for
the batch, as the built-in methods it replaces did.
"""
function apply_post_transmission!(iv::AbstractIntervention, state, new_contacts)
    Base.depwarn(
        "`apply_post_transmission!` is deprecated. Call `trace_contacts!` with " *
            "each infector and the contacts it reached, and `apply_actions!` for " *
            "an intervention's actions.",
        :apply_post_transmission!; force = true
    )
    by_infector = Dict{Int, Vector{eltype(new_contacts)}}()
    for contact in new_contacts
        1 <= contact.parent_id <= length(state.individuals) || continue
        push!(get!(Vector{eltype(new_contacts)}, by_infector, contact.parent_id), contact)
    end
    for pid in sort!(collect(keys(by_infector)))
        trace_contacts!(iv, state, state.individuals[pid], by_infector[pid])
    end
    actions = intervention_actions(iv, state, new_contacts)
    actions === nothing || _admit_actions!(iv, state, actions)
    return nothing
end

# Whether the generation engine should hand `iv` a whole generation's contacts
# through the deprecated `apply_post_transmission!` rather than call its
# per-case hooks: true for an intervention with a method of its own for it.
# The built-in wrappers keep a method for wrapping such an intervention, so
# they answer for the one they wrap.
function _uses_batch_hook(iv::AbstractIntervention)
    return _has_own_method(apply_post_transmission!, typeof(iv), AbstractIntervention)
end

# Logged once per intervention type, so each one still on the hook is named.
function _warn_batch_hook(iv::AbstractIntervention)
    T = typeof(_unwrap_scheduled(iv))
    msg = "$(nameof(T)) implements `apply_post_transmission!`, which is " *
        "deprecated. Implement `trace_contacts!(intervention, state, infector, " *
        "contacts)` for what it does to the contacts a case reached, and " *
        "`intervention_actions(intervention, state, candidates)` for actions " *
        "with a delivery time; both engines call these once per case."
    id = (:apply_post_transmission!, T)
    @warn msg maxlog = 1 _id = id _group = :depwarn
    return nothing
end

"""
    trace_contacts!(intervention, state, infector, contacts[, not_before])

Act on the `contacts` that `infector` reached and that it can still affect.
Default: no-op.

Both engines call it once per case, after the case is settled and before any
of these contacts is decided infected or not, and before any intervention's
[`intervention_actions`](@ref) for the same case. While it runs,
`state.max_infection_time` is the infector's infection time, as in
[`resolve_individual!`](@ref). The continuous-time models call it as the race
settles the case, with its neighbours or household members not yet settled.
The generation engine calls it for each case of the previous generation, with
the contacts it made in this one. A contact several cases reached is passed to
each of them. The generation engine also calls it on an uninfected individual
that [`keep_active`](@ref) kept generating contacts, with the state as the
generation left it.

`not_before[i]`, when given, is the earliest time `contacts[i]` can be sought:
when that person became a contact of `infector`. A contact met only at a
funeral cannot be traced before the funeral, while a household member is a
contact from the start and has `not_before = -Inf`. On `RoutedNetwork` it is the
time of the linking route's `contacts_from` state. The race calls this method
whenever the model yields `(id, time)` contacts and the four-argument method
otherwise; by default the five-argument method calls the four-argument one, so
an intervention that does not time its action from the contact need not handle
it.

Pair with [`traces_contacts`](@ref EpiBranch.traces_contacts), which tells the
race whether an intervention needs this hook at all.
"""
trace_contacts!(::AbstractIntervention, state, infector, contacts) = nothing
function trace_contacts!(iv::AbstractIntervention, state, infector, contacts, not_before)
    return trace_contacts!(iv, state, infector, contacts)
end

"""
    traces_contacts(intervention) -> Bool

Whether `intervention` implements [`trace_contacts!`](@ref
EpiBranch.trace_contacts!). The continuous-time race gathers a case's
reachable contacts only when some intervention says `true`, so an
intervention that does not trace costs nothing. The default reads the methods
themselves: `true` for a type with a `trace_contacts!` method of its own, so an
intervention written outside the package need not declare it. A type with a
`trace_contacts!` method that should not count, such as one that only opts out
of tracing as [`CapacityConstrained`](@ref)'s does, defines
`traces_contacts(::T) = false` itself.
"""
function traces_contacts(iv::AbstractIntervention)
    return _has_own_method(trace_contacts!, typeof(iv), AbstractIntervention)
end

"""
    keep_active(intervention, state, targets, is_new) -> iterable of Int

The ids of the `targets` a case reached that should go on generating contacts
of their own although they are not infected. Default: none.

Both engines ask it once per case, right after the case's
[`trace_contacts!`](@ref) and [`intervention_actions`](@ref), with the same
contacts and the same state. An intervention still implementing the deprecated
[`apply_post_transmission!`](@ref) is asked instead once per generation, about
the whole batch, after that hook has run. An intervention that grows its reach through
uninfected people, such as contact tracing reaching contacts-of-contacts,
returns those people's ids here. The generation engine keeps them active for
one more generation, so their contacts are offered to them through the same
hooks in turn, and the `InfectiousSource` risk source (a default) stops them
infecting anyone. The continuous-time race walks the model's contact structure
from them instead.

`is_new[i]` flags which of `targets` were freshly created. A contact returned
here that turns out to be infected is active as a case anyway."""
keep_active(::AbstractIntervention, state, targets, is_new) = ()

"""Whether an intervention is currently active given the simulation state. Default: always."""
is_active(::AbstractIntervention, ::SimulationState) = true

"""
    Risk(event_time, block_probability, release_time)

A competing risk contributed by an intervention against a single
contact's transmission. The risk's event is in force at transmission time `T`
if `event_time <= T < release_time`; while in force, transmission is blocked
with probability `block_probability`. A contact is infected iff no
intervention's risk blocks it.

All three fields accept either a `Real` or a function
`(rng, parent, contact, state) -> Real`. The function form lets the
event time, block probability, or release time depend on per-individual
state, e.g. age-conditional vaccine efficacy, or a quarantine's duration.

Use `event_time = -Inf` (the default) for risks that are not
time-tagged — pop_suscept, per-individual susceptibility,
infectiousness, and the like. Use `release_time = Inf` (the default) for a
block that, once in force, never lapses, such as a quarantine or isolation
given an infinite duration.

Returned by [`competing_risk`](@ref).
"""
struct Risk{T, P, R}
    event_time::T
    block_probability::P
    release_time::R
end
Risk(; event_time = -Inf, block_probability, release_time = Inf) =
    Risk(event_time, block_probability, release_time)
# The two-argument positional form a `competing_risk` method written before
# there was a release still builds a risk that never lapses.
Risk(event_time, block_probability) = Risk(event_time, block_probability, Inf)

"""
    competing_risk(intervention, parent, contact, state)
        -> Union{Nothing, Risk, NTuple{N, Risk}}

Return the [`Risk`](@ref)(s) this intervention contributes against the
parent → contact transmission, or `nothing` if the intervention does
not gate this transmission. Default: `nothing`.

Most interventions gate transmission through a single mechanism and
return one `Risk`. Interventions that gate it through more than one
mechanism — e.g. ring vaccination's susceptibility reduction on the
contact *and* its onward-infectiousness reduction on the parent — may
return a tuple of risks instead; the engine applies each independently.

A community introduction on a continuous-time model has no infector, and the
person being introduced stands in for one, so a risk that reads the infector
sees the contact itself. Return `nothing` when `parent === contact` if that is
not what your risk means, as [`RingVaccination`](@ref)'s onward-transmission
risk does.

Both engines resolve risks after the infector's [`trace_contacts!`](@ref)
and [`intervention_actions`](@ref) have run, so a risk can read state that
other interventions have written on the contact (e.g. `:vaccination_time` set
by tracing-driven vaccination). The generation engine resolves a generation's
contacts once every case of the previous generation has been through those
hooks; the continuous-time models resolve each infection as they propose
it.
"""
competing_risk(::AbstractIntervention, parent, contact, state) = nothing

"""
    intervention_time(intervention, individual)

Time at which this intervention's effect occurs for an individual. Used
by [`Scheduled`](@ref) to enforce `start_time`: if the intervention time
is earlier than `Scheduled`'s `start_time`, the effect is undone via
[`reset!`](@ref).

Default: `-Inf` (effect always applies).
"""
intervention_time(::AbstractIntervention, ::Individual) = -Inf

"""
    infectious_removal_time(intervention, individual) -> Real

The time at which this intervention takes `individual` out of onward
transmission. The continuous-time (Sellke) transmission models
([`HomogeneousProcess`](@ref), and the network/household processes) close the
infectious window at the earliest removal time across the interventions.
`Isolation` removes a case at its isolation time, and `ContactTracing` removes a
quarantined contact at its trace time. The default is `Inf` (no removal), which
is what an intervention whose effect is a per-contact block rather than a
removal wants — a leaky vaccination, say: those reach the continuous-time models
through [`competing_risk`](@ref) instead, resolved against each infection the
model proposes. Not read by the generation-based engine.
"""
infectious_removal_time(::AbstractIntervention, ::Individual) = Inf

"""
    on_infection_settled!(intervention, individual, state, rng)

Continuous-time models only: called on each case the moment the race fixes its
infection time, before the onset derived from it, the clinical transitions, or
the other intervention hooks read it.

The place for an effect that has to be reconsidered against the exposure the
race has just chosen, rather than the one standing when the intervention acted.
A dose given to a still-uninfected member of the race is the worked example:
when it was given the member had no infection time to abort, and
[`RingVaccination`](@ref) uses this hook to decide the abort once it does.

The race's own `rng` is passed rather than taken from `state`, so a draw here
stays in the stream the race threads. Default: no-op.
"""
on_infection_settled!(::AbstractIntervention, individual, state, rng) = nothing

"""
    reset!(intervention, individual)

Undo the effect of an intervention on an individual. Called by
[`Scheduled`](@ref) when `intervention_time` falls before `start_time`.

Default: no-op.
"""
reset!(::AbstractIntervention, ::Individual) = nothing

"""
    risk_applies(intervention, route) -> Bool

Whether an intervention's [`competing_risk`](@ref) applies to a continuous-time
route. `route` is the existing [`RouteWindow`](@ref), or `nothing` for a
community introduction whose source is outside the population. Models using
the single-route shorthand, including `HomogeneousProcess`, supply a window
named `:transmission`. The default is
`true`, so protection follows a person across routes. Wrappers delegate to their
wrapped intervention.

[`Isolation`](@ref) and [`ContactTracing`](@ref) apply only to routes listing
[`EpiBranch.INTERVENTION_REMOVAL`](@ref) in `until`; they do not protect against
community introductions. External interventions may select routes by any window
property. This predicate does not filter model-provided risk sources or the
generation-based engine's contacts.
"""
risk_applies(::AbstractIntervention, route) = true

# A removal's duration, drawn or given. A negative or `NaN` one puts the
# release before its own start, which the generation engine reads as no removal
# and the continuous-time engines as a permanent one, so it is refused here
# rather than left to split them. Zero is allowed and means no removal on
# either engine: the generation engine needs `event_t <= t < release_t` and so
# blocks nothing, and `infectious_removal_time` reports nothing to close a
# window with.
function _removal_duration(duration, rng, individual, what)
    d = _sample_value(duration, rng, individual)
    (isnan(d) || d < 0) && throw(
        ArgumentError(
            "$what must not be negative, got $d for individual $(individual.id)"
        )
    )
    return d
end

# One risk per stretch a removal took the host out for, a release of `Inf`
# standing for a removal that never ends. A likelihood takes the same
# stretches out of each pair's exposure, which is what keeps the two in step.
function _removal_risks(parent, block_probability, key = REMOVAL_STRETCHES_KEY)
    stretches = removal_stretches(parent, key)
    isempty(stretches) && return nothing
    return Tuple(
        Risk(event_time = a, block_probability = block_probability, release_time = b)
            for (a, b) in stretches
    )
end

# The start and release holding two removals at once. Where they overlap or
# touch, the smallest interval covering both is their union and loses nothing,
# and keeping one side's release alone would drop a removal still in force.
# Where they are disjoint, one pair cannot hold both: the later removal is kept,
# the earlier one being spent before the other begins.
function _combine_removal(standing_start, standing_release, new_start, new_release)
    if new_start <= standing_release && standing_start <= new_release
        return (min(new_start, standing_start), max(new_release, standing_release))
    end
    return standing_start < new_start ? (new_start, new_release) :
        (standing_start, standing_release)
end

"""
    reads_population_state(intervention) -> Bool

Whether `intervention`'s delivery can depend on population-wide state — a
running case count, a capacity budget shared across every individual — rather
than only on the individual it is resolving. A structure-driven model that
races a clique (a household) at a time, rather than the whole population on
one clock, gives each clique's race the state left by whichever clique raced
before it, in race order rather than calendar order; an intervention that
reads population-wide state through that race therefore needs every clique on
one shared clock instead.

The default is the conservative `true`. An intervention written outside the
package therefore reads population-wide state until it says otherwise.

`RingVaccination` and `MassVaccination` return `false`, each delivering
against the individual it resolves. `GroupVaccination` returns `true`, since a
group's trigger is the earliest eligible time among members who may live
anywhere in the population. `CapacityConstrained` returns `true`, because its
admission decision reads how much of the shared budget every other individual
has used. A `Scheduled` built with `start_after_cases` or `start_after`
returns `true`, the case count or trigger being exactly such a read; one
built with only `start_time`/`end_time` compares against the time of the case
being resolved and answers for the intervention it wraps, returning `false`
only while that intervention does.

`Isolation` and `ContactTracing` answer for the components they are given —
`ContactTracing` its eligibility, rate, delay and action, `Isolation` its
eligibility — each defaulting to `false`, and an eligibility combinator
answers for what it wraps. A component of your own that reads population-wide
state declares `true` for itself, which lifts the intervention holding it. Wrappers without a read of their own delegate to the intervention
they wrap.
"""
reads_population_state(::AbstractIntervention) = true

"""
    analytic_offspring_effect(intervention, offspring) -> offspring

Return the offspring distribution a case draws from under `intervention`, given
the one a bare process would use (a `Distribution` for a single-type process, a
[`MultiTypeOffspring`](@ref) for a multi-type one). Define a method only when
the effect is exact in closed form on its own, and return a law the analytical
functions take: a `Poisson` or `NegativeBinomial` for a single-type process.
A population-level reduction of R by a factor `p`, for example, turns
`NegBin(R, k)` into `NegBin(p * R, k)`.

An effect that depends on how the intervention's timing overlaps the
generation-time distribution has no such form. Isolation's effect on R is one
of these, and `Isolation` has no method. The generic method throws an error,
which makes [`extinction_probability`](@ref), [`epidemic_probability`](@ref),
[`probability_contain`](@ref), [`reproduction_number`](@ref),
[`proportion_transmission`](@ref), [`proportion_cases_individual`](@ref),
[`proportion_cases_offspring`](@ref), [`proportion_cluster_size`](@ref),
`end_of_outbreak_probability` and [`offspring_distribution`](@ref) throw
too when called on a [`ModelSpec`](@ref), or a process that stores its own
interventions, holding an intervention that has no method.
"""
function analytic_offspring_effect(iv::AbstractIntervention, offspring)
    throw(
        ArgumentError(
            "$(typeof(iv)) has no closed-form effect on the offspring distribution " *
                "(no `analytic_offspring_effect` method), so this analytical function " *
                "cannot account for it. Compute the quantity from simulated outbreaks, " *
                "simulate(spec, n), instead. For containment, " *
                "containment_probability(simulate(spec, n)) gives the simulated answer " *
                "and probability_contain(R, k; ind_control, pop_control) an approximation."
        )
    )
end
