"""
Base type for all interventions. Subtypes implement one or more of:
`initialise_individual!`, `resolve_individual!`, `apply_post_transmission!`,
`competing_risk`.

To support time-based scheduling (`Scheduled(iv; start_time=...)`), also
implement [`intervention_time`](@ref) and [`reset!`](@ref). The default
implementations return `-Inf` and a no-op respectively, which is correct
for interventions whose effect time is always considered "now".

Tree-shaping interventions (a hard cap on offspring per parent,
gathering-size limits, etc.) are expressed by passing a state-aware
function-form offspring distribution to [`BranchingProcess`](@ref) —
not via the intervention protocol. See the
[Extending guide](@ref "Extending EpiBranch") for an example.
"""
abstract type AbstractIntervention end

"""Set up intervention-specific fields on a newly created individual. Default: no-op."""
initialise_individual!(::AbstractIntervention, individual, state) = nothing

"""Determine intervention state before transmission. Default: no-op."""
resolve_individual!(::AbstractIntervention, individual, state) = nothing

"""Act on contacts after creation. All contacts are received. Default: no-op."""
apply_post_transmission!(::AbstractIntervention, state, new_contacts) = nothing

"""
    trace_contacts!(intervention, state, infector, contacts[, not_before])

Act on the `contacts` that `infector` reaches, on the continuous-time (Sellke)
path. The counterpart of [`apply_post_transmission!`](@ref), which the
generation engine calls with freshly created contact individuals whose
`parent_id` names their infector. The continuous-time models have no such
objects: every node exists from the start and the race only settles when each
is infected, so the infector has to be passed explicitly.

Called once per case, when the race finalises it and its timeline is therefore
known, with the contacts it can still affect. Default: no-op.

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
intervention that does not trace costs nothing. Default: `false`.
"""
traces_contacts(::AbstractIntervention) = false

"""
    keep_active(intervention, state, targets, is_new) -> iterable of Int

The ids of this generation's contacts that should stay *active* into the
next generation (keep generating contacts of their own), beyond the newly
infected cases, which always do. Default: none.

The engine unions these into the next active set, so who keeps generating
contacts is not a special built-in rule. An intervention that needs the
engine to keep growing contacts from *uninfected* nodes, such as contact
tracing reaching contacts-of-contacts, returns those nodes' ids here. Pair
it with the `InfectiousSource` risk source (a default) so those uninfected
nodes generate contacts without infecting them.

`targets` are this generation's contacts and `is_new[i]` flags which were
freshly created. Return ids of nodes that are *not* already infected; the
infected ones stay active anyway."""
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

On the generation-based engine, resolution happens after
`apply_post_transmission!` so that risks can read state that other
interventions have written on the contact (e.g. `:vaccination_time`
set by tracing-driven vaccination). The continuous-time models resolve
the same risks against each infection they propose, after the infector
has been traced, which is where their own tracing-driven state is
written.
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
the single-route shorthand, including the homogeneous pool, supply a window
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

"""
    risk_depends_on_infector(intervention) -> Bool

Whether an intervention's [`competing_risk`](@ref) can block a contact
differently depending on who infected it. A fixed-size pool with more than one
mixing group draws each contact's infector in proportion to infectiousness,
without regard to which groups mix with which. That draw is exact only for
risks that ignore the infector, and the pool refuses an intervention for which
this is `true`. Every other engine ignores it.

The default is `true` for an intervention with a `competing_risk` method of its
own and `false` for one without. Return `false` from an intervention whose risk
reads only the contact, such as a vaccine's protection of the person exposed.
Wrappers delegate to their wrapped intervention.
"""
function risk_depends_on_infector(iv::AbstractIntervention)
    return _has_own_method(competing_risk, typeof(iv), AbstractIntervention)
end

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
has used. A `Scheduled` built with `start_after_cases` returns `true`, the
case count being exactly such a read; one built with only
`start_time`/`end_time` compares against the time of the case being resolved
and answers for the intervention it wraps, returning `false` only while that
intervention does.

`Isolation` and `ContactTracing` answer for the components they are given —
`ContactTracing` its eligibility, rate, delay and action, `Isolation` its
eligibility — each defaulting to `false`, and an eligibility combinator
answers for what it wraps. A component of your own that reads population-wide
state declares `true` for itself, which lifts the intervention holding it. Wrappers without a read of their own delegate to the intervention
they wrap.
"""
reads_population_state(::AbstractIntervention) = true
