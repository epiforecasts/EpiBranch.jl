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
    trace_contacts!(iv, state, infector, contacts)
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
    Risk(event_time, block_probability)

A competing risk contributed by an intervention against a single
contact's transmission. The risk has fired by transmission time `T` if
`event_time <= T`; when it has fired, transmission is blocked with
probability `block_probability`. A contact is infected iff no
intervention's risk blocks it.

Both fields accept either a `Real` or a function
`(rng, parent, contact, state) -> Real`. The function form lets the
event time or block probability depend on per-individual state, e.g.
age-conditional vaccine efficacy.

Use `event_time = -Inf` (the default) for risks that are not
time-tagged — pop_suscept, per-individual susceptibility,
infectiousness, and the like.

Returned by [`competing_risk`](@ref).
"""
struct Risk{T, P}
    event_time::T
    block_probability::P
end
Risk(; event_time = -Inf, block_probability) = Risk(event_time, block_probability)

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
    reset!(intervention, individual)

Undo the effect of an intervention on an individual. Called by
[`Scheduled`](@ref) when `intervention_time` falls before `start_time`.

Default: no-op.
"""
reset!(::AbstractIntervention, ::Individual) = nothing

"""
    RiskScope

Which of a continuous-time model's transmission routes an intervention's
[`competing_risk`](@ref)s apply on, as returned by
[`risk_scope`](@ref EpiBranch.risk_scope). The rule is per route: a route that
lists [`EpiBranch.INTERVENTION_REMOVAL`](@ref) in its `until` resolves every
risk, and one that does not resolves only those scoped to every route. The
single-route shorthand the network, household and homogeneous processes use
lists it, so every risk applies there, as every risk applies to every contact on
the generation-based engine; a [`RouteWindow`](@ref) of a `RoutedNetwork` need
not, whether the model has one route or several. A community introduction
resolves only the risks scoped to every route, on any model, because its source
is outside the population.

  - [`EveryRoute`](@ref EpiBranch.EveryRoute): the risks apply on every route.
  - [`RemovalRoutes`](@ref EpiBranch.RemovalRoutes): the risks apply only on
    routes that list [`EpiBranch.INTERVENTION_REMOVAL`](@ref) in their `until`.
"""
abstract type RiskScope end

"""
    EveryRoute()

The [`RiskScope`](@ref EpiBranch.RiskScope) of a risk that belongs to the people
in a contact whatever route it travels along, such as a vaccine's protection of
the contact or its reduction of the infector's onward transmission. A vaccinated
person is then as protected at home as in the community.
"""
struct EveryRoute <: RiskScope end

"""
    RemovalRoutes()

The [`RiskScope`](@ref EpiBranch.RiskScope) of a risk that stands in for taking
the infector out of circulation, such as leaky isolation. It applies only on the
routes the removal itself would cut, which are those listing
[`EpiBranch.INTERVENTION_REMOVAL`](@ref), so a household route that runs on
through an isolation is not blocked by that isolation's risk either.
"""
struct RemovalRoutes <: RiskScope end

"""
    risk_scope(intervention) -> RiskScope

The routes on which `intervention`'s [`competing_risk`](@ref)s apply on a
continuous-time model — those listing [`EpiBranch.INTERVENTION_REMOVAL`](@ref)
in their `until`, or all of them — and whether they reach a community
introduction, which only those scoped to every route do. Default:
[`EveryRoute`](@ref EpiBranch.EveryRoute), which matches the generation-based
engine, where every risk applies to every contact. [`Isolation`](@ref) and
[`ContactTracing`](@ref) return [`RemovalRoutes`](@ref EpiBranch.RemovalRoutes),
because their effect is a removal and a route opts into removal. An intervention
whose risk expresses a removal, typically one that also defines
[`infectious_removal_time`](@ref), should return `RemovalRoutes()` too.
"""
risk_scope(::AbstractIntervention) = EveryRoute()
