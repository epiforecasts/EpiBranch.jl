"""
Parent type of all interventions: [`Isolation`](@ref),
[`ContactTracing`](@ref), [`RingVaccination`](@ref),
[`MassVaccination`](@ref) and [`GroupVaccination`](@ref). Wrap one in
[`Scheduled`](@ref) to start or stop it at a given time or case count, or in
[`CapacityConstrained`](@ref) to limit how many people it can reach, per
period or in total.
Pass a vector of them as `interventions` to [`ModelSpec`](@ref).

Measures that limit the number of secondary cases directly (a cap per case, a
limit on gathering size) are written as an offspring distribution that depends
on the simulation state, passed to [`BranchingProcess`](@ref).

For extension authors: a new intervention is a subtype that defines one or
more of [`initialise_individual!`](@ref EpiBranch.initialise_individual!),
[`resolve_individual!`](@ref EpiBranch.resolve_individual!),
[`trace_contacts!`](@ref EpiBranch.trace_contacts!),
[`intervention_actions`](@ref EpiBranch.intervention_actions) and
[`competing_risk`](@ref EpiBranch.competing_risk). To work with `Scheduled`
it also defines [`intervention_time`](@ref EpiBranch.intervention_time) and
[`reset!`](@ref EpiBranch.reset!), whose defaults (`-Inf` and doing nothing)
suit an intervention that always acts immediately. See the Extending EpiBranch
guide.
"""
abstract type AbstractIntervention end

"""Record this intervention's starting information on a person when they
enter the simulation (for example "not yet isolated"). Default: does nothing."""
initialise_individual!(::AbstractIntervention, individual, state) = nothing

"""
    resolve_individual!(intervention, case, state)

Decide what the intervention does to a case before it transmits, for example
when it is isolated. Default: does nothing.

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
an intervention that does not time its action from the contact need not
handle it.

Contacts are collected only once some intervention in the stack returns
`true` from [`traces_contacts`](@ref EpiBranch.traces_contacts). Once they
are, this method is called on every intervention, so declaring `false` keeps
the walk from starting but does not keep this method dormant in a stack that
holds something which does trace.

Both engines also call it a second time per case, for an intervention
declaring [`traces_source`](@ref EpiBranch.traces_source): once with `infector`
the case itself and `contacts` its own infector, the single-element collection
`(source,)`, so a backward-tracing intervention such as [`SourceTracing`](@ref)
can act on the person who infected this case rather than the people it
infected. An index case or a community introduction has no infector, so this
second call is skipped for it.

A backward-tracing intervention still receives the *first*, forward call too,
with `case`'s real contacts: declaring `traces_source` only adds the second
call, it does not withdraw the intervention from the first. Check that the
`contacts` offered actually is `case`'s own infector, for example
`only(contacts).id == case.parent_id`, before acting, so that call is a no-op
on a stack's ordinary forward round; [`SourceTracing`](@ref) does this.
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
    traces_source(intervention) -> Bool

Whether `intervention` traces a case back to its own infector rather than
forward to the people it infected: both engines then call its
[`trace_contacts!`](@ref EpiBranch.trace_contacts!) a second time per case, as
documented there. Default: `false`, so writing `trace_contacts!` for forward
tracing (as [`ContactTracing`](@ref) does) never also triggers a backward
call; a backward-tracing intervention such as [`SourceTracing`](@ref) declares
`traces_source(::T) = true` itself. Unlike [`traces_contacts`](@ref
EpiBranch.traces_contacts), this is not inferred from the methods an
intervention defines, since both directions are written through the same
`trace_contacts!` verb.
"""
traces_source(::AbstractIntervention) = false

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

"""Whether an intervention is in effect at this point of the simulation (for
example after a [`Scheduled`](@ref) start). Default: always."""
is_active(::AbstractIntervention, ::SimulationState) = true

"""
    Risk(event_time, block_probability, release_time)

How an intervention can prevent one infection: from `event_time` until
`release_time` (days), an infection that would happen is prevented with
probability `block_probability`. Every risk in force at the time of the
infection is drawn independently, and the contact is infected only if none of
them prevents it.

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
        -> Union{Nothing, Risk, collection of Risk}

How this intervention can prevent the infection of `contact` by `parent`
(the infector): one or more [`Risk`](@ref)s giving the probability and the
time window, or `nothing` if it has no effect on this infection. Default:
`nothing`.

An intervention that acts in more than one way, such as ring vaccination
reducing both the contact's susceptibility and the infector's onward
transmission, returns several risks, each applied independently: a tuple of
them, or any other collection that can be iterated and indexed, as the
built-in removals return a vector with one risk per stretch of removal.

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

Time (days) at which this intervention acts on a person, such as their
isolation time. [`Scheduled`](@ref) runs an intervention that defines this
whether or not its schedule is active at the case's or contact's own clock,
checks the schedule again at `intervention_time`, and undoes the effect with
[`reset!`](@ref EpiBranch.reset!) if that falls outside it.

Default: `-Inf`. An intervention relying on the default is instead gated
before it runs: with no named time, `reset!` has nothing to undo afterwards.
"""
intervention_time(::AbstractIntervention, ::Individual) = -Inf

"""
    infectious_removal_time(intervention, individual) -> Real

The time (days) at which this intervention ends `individual`'s infectious
period for good. Continuous-time models ([`HomogeneousProcess`](@ref),
[`MixingProcess`](@ref) and the network and household models) end a case's
infectious period at the earliest
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
function _removal_risks(
        parent::Individual{T}, block_probability, key = REMOVAL_STRETCHES_KEY
    ) where {T}
    stretches = get(parent.state, key, nothing)
    # Recorded in the host's own number type, as `record_removal!` does, the
    # stretches are walked without dispatching on each; anything else is
    # walked as `Any`.
    if stretches isa Vector{Tuple{T, T}}
        return isempty(stretches) ? nothing : _StretchRisks(stretches, block_probability)
    end
    (stretches === nothing || isempty(stretches)) && return nothing
    return _StretchRisks(collect(Any, stretches)::Vector{Any}, block_probability)
end

# The risks of `_removal_risks`, a vector of them made one at a time as they
# are read rather than gathered first.
struct _StretchRisks{R, V, P} <: AbstractVector{R}
    stretches::V
    block_probability::P
end
function _StretchRisks(stretches::Vector{Tuple{T, T}}, p::P) where {T, P}
    return _StretchRisks{Risk{T, P, T}, typeof(stretches), P}(stretches, p)
end
_StretchRisks(stretches::Vector{Any}, p::P) where {P} =
    _StretchRisks{Risk, Vector{Any}, P}(stretches, p)
Base.size(r::_StretchRisks) = size(r.stretches)
Base.@propagate_inbounds function Base.getindex(r::_StretchRisks, i::Int)
    a, b = r.stretches[i]
    return Risk(event_time = a, block_probability = r.block_probability, release_time = b)
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
have used. A `Scheduled` built with `start_after_cases` or `start_after`
returns `true`, the case count or trigger being exactly such a read; one built
with only
`start_time`/`end_time` compares against the time of the case concerned
and answers for the intervention it contains.

`Isolation` and `ContactTracing` answer for their parts (`ContactTracing` its
eligibility, rate, delay and action; `Isolation` its eligibility), each
`false` by default, and a combined eligibility rule answers for its
conditions. A part of your own that reads population-wide information returns
`true`, which makes the intervention containing it return `true`.
"""
reads_population_state(::AbstractIntervention) = true

"""
    analytic_offspring_effect(intervention, offspring) -> offspring
    analytic_offspring_effect(intervention, offspring, spec) -> offspring

Return the offspring distribution a case draws from under `intervention`, given
the one a bare process would use (a `Distribution` for a single-type process, a
[`MultiTypeOffspring`](@ref) for a multi-type one). Define a method only when
the effect is exact in closed form on its own, and return a law the analytical
functions take: a `Poisson`, a `NegativeBinomial`, or a
[`GeneratingFunctionOffspring`](@ref) built from the offspring's generating
function or probability mass function. A population-level reduction of R by a
factor `p`, for example, turns `NegBin(R, k)` into `NegBin(p * R, k)`, as
[`MassVaccination`](@ref)'s method does for a campaign already complete
before the simulation starts.

The three-argument form also receives the [`ModelSpec`](@ref) (or process that
stores its own interventions), for an effect that needs more than the
offspring law itself, such as the generation-time distribution or the
incubation period. Its default falls back to the two-argument method, so an
intervention whose effect does not depend on the rest of the model need only
define that one; [`Isolation`](@ref) defines the three-argument form, since its
effect on R depends on how its timing overlaps the generation time.

An effect with no closed form at all, such as contact tracing's or ring and
group vaccination's, has no method of either arity. The generic method throws
an error, which makes [`extinction_probability`](@ref),
[`epidemic_probability`](@ref), [`probability_contain`](@ref),
[`reproduction_number`](@ref), [`proportion_transmission`](@ref),
[`proportion_cases_individual`](@ref), [`proportion_cases_offspring`](@ref),
[`proportion_cluster_size`](@ref), `end_of_outbreak_probability` and
[`offspring_distribution`](@ref) throw too when called on a [`ModelSpec`](@ref),
or a process that stores its own interventions, holding an intervention that
has no method.
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

function analytic_offspring_effect(iv::AbstractIntervention, offspring, spec)
    return analytic_offspring_effect(iv, offspring)
end

"""
    has_effect(intervention) -> Bool

Whether `intervention` can change a simulation or a likelihood. [`ModelSpec`](@ref)
asks this of each intervention it is given and warns when one answers `false`,
so an inert intervention does not run unnoticed.

The default reads the method tables: `true` for a type with a method of its own,
at the number of arguments the engine calls it with, for one of the hooks the
package dispatches on an intervention. An intervention that reaches the model
some other way, through a seam a package of its own defines for instance,
defines `EpiBranch.has_effect(::MyIntervention) = true` to say so. The question
is asked once per intervention type, so the answer should not depend on the
instance.
"""
function has_effect(iv::AbstractIntervention)
    T = typeof(iv)
    return any(_effect_hooks()) do (f, n, _)
        _has_own_method(f, T, AbstractIntervention, n)
    end
end
