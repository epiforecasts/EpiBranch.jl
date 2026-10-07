"""
Parent type of all interventions: [`Isolation`](@ref),
[`ContactTracing`](@ref), [`RingVaccination`](@ref),
[`MassVaccination`](@ref) and [`GroupVaccination`](@ref). Wrap one in
[`Scheduled`](@ref) to start or stop it at a given time or case count, or in
[`CapacityConstrained`](@ref) to limit how many people it can reach per day.
Pass a vector of them as `interventions` to [`ModelSpec`](@ref).

Measures that limit the number of secondary cases directly (a cap per case, a
limit on gathering size) are written as an offspring distribution that depends
on the simulation state, passed to [`BranchingProcess`](@ref).

For extension authors: a new intervention is a subtype that defines one or
more of [`initialise_individual!`](@ref EpiBranch.initialise_individual!),
[`resolve_individual!`](@ref EpiBranch.resolve_individual!),
[`apply_post_transmission!`](@ref EpiBranch.apply_post_transmission!) and
[`competing_risk`](@ref EpiBranch.competing_risk). To work with `Scheduled`
it also defines [`intervention_time`](@ref EpiBranch.intervention_time) and
[`reset!`](@ref EpiBranch.reset!), whose defaults (`-Inf` and doing nothing)
suit an intervention that always acts immediately. See the
[Extending guide](@ref "Extending EpiBranch").
"""
abstract type AbstractIntervention end

"""Record this intervention's starting information on a person when they
enter the simulation (for example "not yet isolated"). Default: does nothing."""
initialise_individual!(::AbstractIntervention, individual, state) = nothing

"""Decide what the intervention does to a case before it transmits (for
example when it is isolated). Default: does nothing."""
resolve_individual!(::AbstractIntervention, individual, state) = nothing

"""Act on the contacts a generation of cases has just made, infected or not
(for example tracing or vaccinating them). Default: does nothing."""
apply_post_transmission!(::AbstractIntervention, state, new_contacts) = nothing

"""
    trace_contacts!(intervention, state, infector, contacts[, not_before])

Act on the people `infector` has been in contact with, in a continuous-time
model that knows each case's contacts (network or household, not
[`HomogeneousProcess`](@ref)), for example by tracing them. It is
the continuous-time counterpart of
[`apply_post_transmission!`](@ref EpiBranch.apply_post_transmission!).
In these models everyone in the population exists from the start and infection
times are settled one at a time (the "race" between possible infections), so
the infector is passed explicitly. Called when a case's infection and course
are known, with the contacts it can still affect. When tracing reaches further
than a case's own contacts, it is called again for each contact it carries on
from, with that contact as `infector` and their own contacts. Default: does nothing.

`not_before[i]`, when given, is the earliest time (days) `contacts[i]` can be
reached: when that person became a contact of `infector`. A contact met only
at a funeral cannot be traced before the funeral, while a household member is
a contact from the start (`not_before = -Inf`). On `RoutedNetwork` it is the
time of the linking route's `contacts_from` state. The five-argument method
is called when the model provides contact times, the four-argument one
otherwise; by default the five-argument method calls the four-argument one, so
an intervention that does not time its action from the contact need not
handle it.

Define [`traces_contacts`](@ref EpiBranch.traces_contacts) as `true` too, or
this method is not called.
"""
trace_contacts!(::AbstractIntervention, state, infector, contacts) = nothing
function trace_contacts!(iv::AbstractIntervention, state, infector, contacts, not_before)
    return trace_contacts!(iv, state, infector, contacts)
end

"""
    traces_contacts(intervention) -> Bool

Whether `intervention` acts on a case's contacts in continuous-time models
through [`trace_contacts!`](@ref EpiBranch.trace_contacts!). Contacts are
collected only when some intervention returns `true`. Default: `false`.
"""
traces_contacts(::AbstractIntervention) = false

"""
    keep_active(intervention, state, targets, is_new) -> iterable of Int

Which uninfected contacts from this generation should go on making contacts
of their own in the next generation, as ids. Newly infected cases always do.
Default: none.

Contact tracing uses this to reach contacts of contacts: a traced but
uninfected person keeps making contacts so they can be traced, and the default
[`InfectiousSource`](@ref EpiBranch.InfectiousSource) rule stops an uninfected
person infecting anyone.

`targets` are this generation's contacts and `is_new[i]` flags which were
newly created. Return only ids of people who are not infected."""
keep_active(::AbstractIntervention, state, targets, is_new) = ()

"""Whether an intervention is in effect at this point of the simulation (for
example after a [`Scheduled`](@ref) start). Default: always."""
is_active(::AbstractIntervention, ::SimulationState) = true

"""
    Risk(event_time, block_probability, release_time)

How an intervention can prevent one infection: from `event_time` until
`release_time` (days), an infection that would happen is prevented with
probability `block_probability`. A contact is infected only if no
intervention prevents it; each intervention's chance acts independently, so
whichever applies first (competing risks) decides.

Each field is a number or a function `(rng, parent, contact, state) -> Real`,
for example vaccine efficacy that depends on age, or a quarantine's duration.
`parent` is the infector.

Leave `event_time = -Inf` (the default) for protection that does not start at
a particular time, such as reduced susceptibility. Leave `release_time = Inf`
(the default) for protection that never lapses, such as isolation or
quarantine with an infinite duration.

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

How this intervention can prevent the infection of `contact` by `parent`
(the infector): one or more [`Risk`](@ref)s giving the probability and the
time window, or `nothing` if it has no effect on this infection. Default:
`nothing`.

An intervention that acts in more than one way, such as ring vaccination
reducing both the contact's susceptibility and the infector's onward
transmission, returns a tuple of risks, each applied independently.

A community introduction on a continuous-time model has no infector, and the
person being introduced stands in for one, so a risk that reads the infector
sees the contact itself. Return `nothing` when `parent === contact` if that is
not what your risk means, as [`RingVaccination`](@ref)'s onward-transmission
risk does.

In a branching process the risks are evaluated after
[`apply_post_transmission!`](@ref EpiBranch.apply_post_transmission!), so they
can read what other interventions recorded on the contact (such as a
vaccination time set after tracing). Continuous-time models evaluate them
against each possible infection after the infector's contacts have been
traced.
"""
competing_risk(::AbstractIntervention, parent, contact, state) = nothing

"""
    intervention_time(intervention, individual)

Time (days) at which this intervention acts on a person, such as their
isolation time. [`Scheduled`](@ref) uses it to enforce `start_time`: an
effect timed before the start is undone with
[`reset!`](@ref EpiBranch.reset!).

Default: `-Inf` (effect always applies).
"""
intervention_time(::AbstractIntervention, ::Individual) = -Inf

"""
    infectious_removal_time(intervention, individual) -> Real

The time (days) at which this intervention ends `individual`'s infectious
period for good. Continuous-time models ([`HomogeneousProcess`](@ref) and the
network and household models) end a case's infectious period at the earliest
such time across interventions: `Isolation` at the isolation time (perfect
isolation only, `post_isolation_transmission = 0`) and `ContactTracing` at a
quarantined contact's trace time, when the isolation or quarantine has no end
(`duration = Inf`). The default `Inf`
(no removal) suits an intervention that reduces transmission per contact,
such as a leaky vaccine, which acts through
[`competing_risk`](@ref EpiBranch.competing_risk) instead. Branching processes
do not use it.
"""
infectious_removal_time(::AbstractIntervention, ::Individual) = Inf

"""
    on_infection_settled!(intervention, individual, state, rng)

Continuous-time models only: called on each case as soon as its infection
time is fixed, before its onset, clinical course or other interventions are
worked out from it.

Use it for an effect that depends on when the person was actually infected.
For example, [`RingVaccination`](@ref) can vaccinate someone before they are
infected; whether post-exposure protection stops the infection is decided
here, once the infection time is known. Draw random numbers from the `rng`
passed in, not from `state`. Default: does nothing.
"""
on_infection_settled!(::AbstractIntervention, individual, state, rng) = nothing

"""
    reset!(intervention, individual)

Undo this intervention's effect on a person. [`Scheduled`](@ref) calls it
when the effect would have happened before `start_time`. Default: does
nothing.
"""
reset!(::AbstractIntervention, ::Individual) = nothing

"""
    risk_applies(intervention, route) -> Bool

Whether this intervention's [`competing_risk`](@ref EpiBranch.competing_risk)
acts on infections through a given transmission route in a continuous-time
model. `route` is a [`RouteWindow`](@ref), or `nothing` for an infection from
outside the population. A model with a single route names it
`:transmission`. The default is `true`, so protection such as vaccination
follows a person across routes. [`Scheduled`](@ref) and
[`CapacityConstrained`](@ref) answer for the intervention they contain.

[`Isolation`](@ref) and [`ContactTracing`](@ref) act only on routes listing
[`EpiBranch.INTERVENTION_REMOVAL`](@ref) in `until`, and not on infections
from outside the population. Other interventions may select routes by any
property of the route. This does not affect protection set by the model
itself or contacts in a branching process.
"""
risk_applies(::AbstractIntervention, route) = true

"""
    risk_depends_on_infector(intervention) -> Bool

Whether this intervention's protection against an infection can depend on
who the infector is (for example an isolated infector). A
[`HomogeneousProcess`](@ref) with more than one mixing type picks each
infection's infector in proportion to infectiousness, which is exact only for
protection that ignores the infector, so it refuses an intervention for which
this is `true`. Other models ignore it.

The default is `true` for an intervention with its own
[`competing_risk`](@ref EpiBranch.competing_risk) method and `false` otherwise.
Return `false` when the protection depends only on the person exposed, such
as a vaccine's. [`Scheduled`](@ref) and [`CapacityConstrained`](@ref) answer
for the intervention they contain.
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

Whether who receives this intervention, or when, can depend on the whole
population (a running case count, a capacity shared by everyone) rather than
only on the person concerned. A household model can simulate one household at
a time only when no intervention reads population-wide information; otherwise
all households are simulated together on one clock so the information is in
calendar order.

The default is `true`, the safe choice for an intervention written outside the
package.

`RingVaccination` and `MassVaccination` return `false`, each acting on one
person at a time. `GroupVaccination` returns `true`, since a
group's trigger is the earliest eligible time among members who may live
anywhere in the population. `CapacityConstrained` returns `true`, because
whether a person is served depends on how much of the shared capacity others
have used. A `Scheduled` built with `start_after_cases` returns `true`, the
case count being exactly such a read; one built with only
`start_time`/`end_time` compares against the time of the case concerned
and answers for the intervention it contains.

`Isolation` and `ContactTracing` answer for their parts (`ContactTracing` its
eligibility, rate, delay and action; `Isolation` its eligibility), each
`false` by default, and a combined eligibility rule answers for its
conditions. A part of your own that reads population-wide information returns
`true`, which makes the intervention containing it return `true`.
"""
reads_population_state(::AbstractIntervention) = true
