# Writing an intervention

This page is for a control measure the package does not include: a treatment
that ends infections early, a vaccination campaign aimed at a particular group,
a rule for whose contacts are traced, a policy that starts on a given date or
is limited by the doses available. It also covers clinical events of your own,
such as testing, treatment or loss to follow-up, which use the same hooks. It
builds on the border closure in [Extending EpiBranch](extending.md), which used
a single method; here are all the steps of a simulation an intervention can act
at, and what each is for. The [words used in these pages](@ref "Words used in
these pages") are defined in that guide.

An intervention is a `struct` that subtypes `AbstractIntervention`. You write
methods only for the hooks it needs; every other hook leaves the simulation
unchanged.

## What happens to each case

Every built-in model, the branching process as much as the network,
household, homogeneous and mixing models, settles one case at a time, in
order of infection time. Once a case's infection time is fixed,
`on_infection_settled!` runs, then its clinical transitions and every
intervention's `resolve_individual!`. This is where `Isolation` works out when
a case is isolated, from its onset time.

A branching process then draws the case's contacts, each with a potential
infection time. A contact becomes an `Individual` when its infection is
decided, or earlier when tracing or an intervention's actions reach it.
`initialise_individual!` then sets the values the intervention keeps for it
(`ContactTracing` sets `:traced` to `false`). The network, household,
homogeneous and mixing models (the fixed-population models) create everyone at
the start.

On a model that can name a case's contacts (a branching process, network or
household model, not the homogeneous or mixing one), `trace_contacts!` acts on
the case's contacts not yet infected. Contact tracing happens here, and
`keep_active` chooses which traced contacts a ring grows from. Every
intervention with an `intervention_actions` method is then offered the case and
those contacts, and proposes actions on them, such as doses of a vaccine, which
go ahead only if their schedule and capacity limits allow. The case's potential
infections are then decided in order of infection time across the whole
outbreak, and `competing_risk` is asked about each when the simulation reaches
it, with the state as it stands then: a dose or quarantine recorded on the
contact since the infection time was drawn is in force. The contact is infected
only if nothing blocks it.

The index cases are settled in the same way first.

A model written against `generate_offspring` or `contacts_of` (see [New
transmission structures](new-structures.md)) is simulated the same way.

While a case's clinical transitions and intervention hooks run, from
`resolve_individual!` on, `state.max_infection_time` is that case's infection
time and `state.cumulative_cases` counts the case and those settled before it.
`on_infection_settled!` runs before that, and still sees the previous case's
values. A [`Scheduled`](@ref) condition reads these: `cumulative_cases` for a
`start_after_cases` count. `start_time`/`end_time` are checked differently,
against `isolation_time`/`intervention_time` rather than this clock; see
below.

Interventions are applied in the order they appear in `interventions = [...]`,
and each sees what earlier ones wrote for the same case.

A [`Risk`](@ref) is what `competing_risk` returns.
`Risk(event_time = t, block_probability = p)` blocks a transmission happening
at or after time `t` with probability `p`. An optional `release_time` ends the
block, for a removal such as quarantine that ends. Return `nothing` when the
intervention does not affect a pair, or a tuple of risks when it acts in more
than one way: `RingVaccination` returns a risk on the contact (lower
susceptibility) alongside one on the infector (lower onward infectiousness).
A contact is infected only if no risk blocks it.

A measure that changes *how many* people a case infects, such as a cap on
gathering size, belongs in the offspring distribution instead; see [Change how
many people each case infects](@ref).

## All hooks

| Hook | Called | Receives | Returns |
|---|---|---|---|
| `initialise_individual!(iv, individual, state)` | Once, when each person is created | An `Individual` whose `state` holds `:infected = false`, the values its population characteristics set and any the model sets, such as `:type` or `:household`; use `get!` for a key that may already be set | `nothing`; sets values in `individual.state` |
| `resolve_individual!(iv, individual, state)` | Once per case, when it is settled: after its clinical transitions, before it infects anyone. Never for a person who is not infected | The case, with `state.max_infection_time` at its infection time | `nothing`; sets values in `individual.state` |
| `trace_contacts!(iv, state, infector, contacts[, not_before])` | Once per case, after it is settled and before any of its contacts is known to be infected or not; also once per ring member a ring of contact tracing grows from; a second time per case, for an intervention declaring `traces_source`, with the case's own infector | The case, the contacts it reached that it can still affect, and, from a model whose contacts can arise after the case's infection, when each became a contact (the four-argument method is called when the model gives no times, and by default for interventions that ignore them); on the second, backward call, `contacts` is the single-element `(source,)` | `nothing`; sets values on the contacts |
| `intervention_actions(iv, state, candidates)` | Once per case, after every intervention's `trace_contacts!` for it | The people [`EpiBranch.action_candidates`](@ref) chooses: by default the case and the contacts it can still reach (those tracing reached from it, where there was tracing). [`GroupVaccination`](@ref) is offered the members of the case's group | `nothing`, or a vector of [`EpiBranch.InterventionAction`](@ref)s to put through the schedule and capacity limits (see [Intervention actions](@ref)) |
| `keep_active(iv, state, targets, is_new)` | During tracing, straight after `trace_contacts!` | The contacts `trace_contacts!` received, and an `is_new` flag for each, always `false` | The ids of the contacts the ring is traced onward from (default: none) |
| `competing_risk(iv, parent, contact, state)` | Per infector–contact pair, after the infector's own hooks, when the simulation reaches each potential infection, in order of infection time | The infector and one contact | `nothing`, one [`Risk`](@ref), or a collection of them (a tuple, or any indexable iterable such as a vector) for an intervention that blocks transmission in more than one way |
| `traces_contacts(iv)` | Whenever a model decides whether to collect a case's contacts at all | Nothing | `true` if this intervention traces. The default is `true` for a type with a `trace_contacts!` method of its own; you need not write it |
| `traces_source(iv)` | Whenever the simulation decides whether to make the second, backward `trace_contacts!` call | Nothing | `true` if this intervention traces a case back to its own infector. Default `false`; [`SourceTracing`](@ref) declares it, since this is not inferred the way `traces_contacts` is |
| `EpiBranch.action_candidates(iv, state, current, traced, is_pending, members, contacts)` | Once per case, for an intervention with an `intervention_actions` method, to choose its candidates | The settled case, the people tracing reached from it (or `nothing`), `is_pending(id)` to ask whether a person is not yet settled, the ids of everyone in the run and the model's contacts | A vector of candidates. The default is the case and the contacts it can still reach; [`GroupVaccination`](@ref) writes its own method, offering the members of the case's group. An intervention that can reach anyone not yet settled walks `members`, at the cost of the size of the run per case |
| `acts_on_population(iv)` | Once, before the simulation processes any case, and for each contact a branching process creates | Nothing | `Bool`: whether this intervention's actions belong to the whole population rather than a case's contacts or group, so `intervention_actions` is called once, upfront, with every member. Default `false`; [`MassVaccination`](@ref) declares it |
| `needs_contacts(iv)` | Whenever the simulation decides whether an intervention can be offered actions on a model that names no contacts | Nothing | `Bool`: whether this intervention's candidates come from the model's named contacts. Default `true`, the conservative choice; [`GroupVaccination`](@ref) declares `false`, since its candidates come from a case's group |
| `has_effect(iv)` | Once per intervention type, when a `ModelSpec` is built | Nothing | `Bool`: whether the intervention can change a simulation or a likelihood. The default is `true` for a type with a method of its own, with the right number of arguments, for a hook it can act through, such as `competing_risk` or `trace_contacts!`; `ModelSpec` warns on `false` |
| `infectious_removal_time(iv, individual)` | Every built-in model: when a case's infectious period is closed | A person | The time this intervention removes them from onward transmission (default `Inf`) |
| `on_infection_settled!(iv, individual, state, rng)` | Every built-in model: once a case's infection time is fixed, before its onset or clinical transitions read it | The case, and the random number generator to use | `nothing`; sets values on the case (default: does nothing) |
| `risk_applies(iv, route)` | Every built-in model choosing the risks for a route or infectiousness window | The route's `RouteWindow`, or `nothing` for an introduction from outside | `Bool`; default `true` |
| `standing_block(iv)` | Fixed-population models deciding whether a certain block ends a pair's contacts for good | Nothing | `Bool`; default `false` |
| `analytic_offspring_effect(iv, offspring)`, `analytic_offspring_effect(iv, offspring, spec)` | When an analytical function such as [`extinction_probability`](@ref) is called on a `ModelSpec` holding the intervention | The offspring distribution without the intervention (the three-argument form also receives the `ModelSpec`, for an effect that needs more than the offspring law) | The offspring distribution a case has under the intervention. Without a method of either arity the function stops with an error |

`analytic_offspring_effect` is for the rare intervention whose effect on the
offspring distribution is exact in closed form. One that cuts every case's R
by a fixed proportion `p` turns `NegBin(R, k)` into `NegBin((1 - p) * R, k)`,
and its method returns that; return a `Poisson`, a negative binomial, or a
[`GeneratingFunctionOffspring`](@ref) built from the effect's own generating
function, the forms the analytical functions accept. [`MassVaccination`](@ref)
has such a method for a campaign already complete before the simulation
starts. Most interventions still have no such form: contact tracing's and
ring and group vaccination's effects depend on who they reach and when, and
have no method. `Isolation`'s effect on R depends on how its timing overlaps
the generation time, which is a mixture over the isolation time rather than a
fixed proportion; it defines the three-argument form to build that mixture
from the model's generation time and incubation period, and its own docstring
sets out the limits of this (a custom eligibility rule, say, still has no
closed form). [`extinction_probability`](@ref),
[`epidemic_probability`](@ref), [`probability_contain`](@ref) and the other
[analytical functions](analytical.md) stop with an error on a `ModelSpec` that
holds an intervention with no method of either arity, and their error points
to simulating the model instead. The border closure from [Extending
EpiBranch](@ref "A custom intervention: closing a border") is one of these.

The [`AbstractIntervention`](@ref) docstring and the
[Extension reference](@ref "Continuous-time models: further details") have the
remaining details for the continuous-time models.

## Which hooks run on which model

Every model, built in or written against `generate_offspring` or
`contacts_of`, decides each potential infection, a drawn time for an infector
and a contact, in order of infection time, and calls every hook. A `Risk` can
act on that infection, as can the end of the infectious period. Forward
contact tracing needs a model that can name each case's contacts: every model
can except the mass-action pools, homogeneous and mixing.

In practice:

- An intervention that **removes** a case from transmission (isolation,
  quarantine once traced, hospitalisation) works on every model, because it
  shortens the infectious period, which every model has.
- An intervention that blocks each contact with some probability (a leaky
  vaccine, partially effective prophylaxis) also works everywhere, and so do
  per-person susceptibility and infectiousness. Each potential infection is
  put to the risks when the simulation reaches it, in order of infection
  time, at the time drawn for it. On a fixed-population model a blocked
  contact does not transmit, and the pair keeps meeting: the pair's next
  contact is drawn later. On a branching process, or for an exposure a model
  names through `contacts_of`, the blocked contact is lost.
- [`SourceTracing`](@ref) quarantines whichever case it finds, so it is a
  removal too, and reaches every built-in model, including the
  mass-action pools [`ContactTracing`](@ref) cannot: a case's infector is read
  off the transmission tree rather than the model's own contact structure. It
  works on a branching process too: the infector's contacts are already drawn
  by the time it is found, but those not yet infected are blocked once it is
  quarantined.

!!! warning "The same efficacy means different things on the two kinds of model"
    On a fixed-population model a pair that escapes infection keeps meeting, so
    blocking each contact with probability `p` multiplies the rate of infection
    by `1 - p`. A pair with a one-day mean contact interval and a two-day
    infectious period then infects `1 - exp(-1) = 0.63` of the time at
    efficacy 0.5. On a branching process, or a model written against
    `contacts_of`, a case's contacts are a fixed set of draws, so a blocked contact is a transmission lost, and the same efficacy
    of 0.5 halves that pair's chance of transmission (0.43 in this example,
    against 0.63). The per-exposure reading on fixed-population models is what a
    leaky vaccine means there; check which one your scenario needs.

- On a network or household model, `loglikelihood(data, spec)` refuses any
  part of `spec` the likelihood cannot represent: a per-contact block that
  does not come from a recorded removal stretch (`RingVaccination`'s
  `onward_efficacy`, say), or an effect not declared through
  `infection_likelihood_compatible` or `susceptibility_components`. Complete
  and leaky isolation are both fitted exactly (the latter through
  [`EpiBranch.removal_block_probability`](@ref EpiBranch.removal_block_probability)),
  and so is vaccination within the limits listed in [Likelihood
  compatibility](@ref), when the fit is given the same `spec` as the
  simulation. Fitting with the bare model, or with a `spec` that leaves the
  intervention out, gives no error and disagrees with the simulation.
- An intervention with an `intervention_actions` method is offered its
  actions once each case settles, narrowed to the case and the contacts it can
  still reach. One that chooses its own candidates, such as everyone not yet
  infected, writes a method of `EpiBranch.action_candidates`. Such actions
  must work without knowing the infection time of a
  contact not yet infected, and must not change a case already settled. That
  narrowing needs a model that can name contacts, unless the intervention
  declares `EpiBranch.needs_contacts(iv) = false`, as group vaccination does,
  since its candidates come from a case's group rather than its contacts. A
  branching process also offers such an intervention each contact as it is
  created, with that contact alone as the candidate, so a group's dose reaches
  a member who joins it later. Those actions are admitted at any finite time,
  even one before the clock, since nothing about the new person is settled. Ring
  vaccination admits a dose for a contact not yet exposed as though any
  `eligibility_window` were open, and withdraws it through
  `EpiBranch.on_infection_settled!` once that contact's own infection settles
  and the window turns out not to cover it.
  `EpiBranch.acts_on_population(iv) = true` instead offers the actions once,
  before any case settles, to every member, and to each contact a branching
  process creates; [`MassVaccination`](@ref) declares it, since its rollout
  depends on nothing about who infected whom.
- **Contact tracing** needs both: its action is a removal, so it applies on
  every model, but it has to know who a case's contacts were. A branching
  process knows the contacts each case made. A fixed-population model must
  supply them: the model reports `EpiBranch.supplies_contacts(model) = true`
  and passes a `contacts` function. That function returns contact ids, or
  `(id, time)` pairs when some contacts arise only after the case's infection,
  such as at a funeral (`contacts_from = :died` on a `RoutedNetwork` route).
  `time` is when each person became a contact, and `ContactTracing` starts that
  contact's trace delay no earlier than then. A network names a person's
  neighbours and a household its members; the homogeneous and mixing models
  have no list of who met whom, so tracing does not happen there.

### Order of steps on the two kinds of model

On both kinds of model a contact is traced when its *infector* is settled,
before the contact's own clinical transitions and `resolve_individual!` run. An
intervention that writes onto a contact must therefore not assume the contact
has nothing written yet, and one that reads a contact's own values must not
assume they are already set. On the fixed-population models even the contact's
infection time and onset are not yet known. `Isolation` therefore treats a
quarantine already in place as a competing route to isolation and keeps the
earlier time.

On the built-in models tracing reaches only contacts whose own infection is not
yet settled. Once a case is settled, its infectious period and onward
transmissions are fixed, so a trace arriving afterwards has nothing left to
shorten. Backward tracing, to the person who infected a case, is not supported
on either kind of model.

What an intervention can rely on:

- `resolve_individual!` runs on a case before it infects anyone, so a risk can
  read what `resolve_individual!` wrote on the infector.
- A case's `trace_contacts!`, `intervention_actions` and `keep_active` run
  before any of its contacts is known to be infected or not, so a risk can read
  what they wrote on the contact (for example `:vaccination_time`).
- Every intervention's `trace_contacts!` for a case runs before any
  intervention's `intervention_actions` for it, so ring vaccination reads what
  tracing wrote, whatever the order of the list.
- Otherwise interventions apply in the order of `interventions = [...]`, and
  each sees what earlier interventions wrote for the same case.
- `keep_active` runs before its contacts are known to be infected, so choose by
  what tracing wrote, as `ContactTracing` does, not by `:infected`. A contact
  it keeps that is then infected goes on as a case anyway.

On the built-in models, the transmission time a `Risk` is compared with is the
potential infection time drawn for that pair.

## The hooks in the built-in interventions

These simplified versions of the package source show one hook each. They call
internal functions and do not run on their own. The full source is in
`src/interventions/`.

`initialise_individual!`: `ContactTracing` sets the two values it keeps on
every new person, where code reading them later always finds a value.

```julia
function initialise_individual!(::ContactTracing, individual, state)
    individual.state[:traced] = false
    individual.state[:quarantined] = false
    return nothing
end
```

`resolve_individual!`: `Isolation` works out when the case is isolated, from
its onset time plus a sampled delay, and takes an earlier isolation time if
contact tracing found the case first. It then draws how long the isolation
lasts from its `duration` and passes the end to `set_isolated!` as
`release_time`. `x || return nothing` means "stop here unless `x` is true".

```julia
function resolve_individual!(iso::Isolation, individual, state)
    onset = onset_time(individual)
    # Self-isolation after a positive test.
    test_time = if is_test_positive(individual)
        onset + _sample_value(iso.onset_to_isolation_delay, state.rng, individual)
    else
        Inf
    end
    # A contact traced before its onset was known has only the bare trace
    # time, so hold it back to the onset.
    traced_time = max(get(individual.state, :_traced_isolation_time, Inf), onset)
    start = min(test_time, traced_time)
    isfinite(start) || return nothing
    duration = _removal_duration(
        iso.duration, state.rng, individual, "`Isolation`'s `duration`"
    )
    set_isolated!(individual, start; release_time = start + duration)
    return nothing
end
```

`trace_contacts!`: `ContactTracing` goes through the contacts a case reached
and applies its trace action (`Quarantine` or `FlagOnly`) when the contact is
eligible and is reached. The trace is timed from
[`trigger_time`](@ref EpiBranch.trigger_time), which for a policy based on
isolation is the recorded isolation, so an isolation that does not count as a
detection starts no trace. Leaving out the depth of the ring and the contacts'
`not_before` times, it reads:

```julia
function trace_contacts!(ct::ContactTracing, state, infector, contacts)
    rng = state.rng
    for ind in contacts
        is_eligible(ct.eligibility, infector, ind, state) || continue
        traces(ct.trace_rate, infector, ind, state, rng) || continue
        trace_delay = draw_trace_delay(ct.isolation_to_trace_delay, infector, ind, state, rng)
        trace_time = trigger_time(ct.eligibility, infector, ind, state) + trace_delay
        apply_trace!(ct.action, ind, state, trace_time, rng)
    end
    return nothing
end
```

`competing_risk`: see [A custom intervention: closing a border](@ref).

## Checking your intervention

Every hook has a default, so an intervention defines only the hooks it needs.
Building a `ModelSpec` warns, naming the hook where it can, about a method
written with a number of arguments no call uses, and about one written without
the `EpiBranch.` prefix, which defines a new function of the same name instead
of a method of the package's own. An intervention with no hook method at all
gets a warning listing every hook. One that acts through something the check
does not read, such as a function of your own package, defines
`EpiBranch.has_effect(::MyIntervention) = true` to silence it. The warning
appears once for each intervention type and set of methods.

None of this can tell an intervention that leaves a hook out on purpose from one
missing a hook it needs. Three checks for the rest:

- Run a small simulation (`max_cases = 50`) with and without your intervention.
  If the results look the same, your `competing_risk` or `trace_contacts!` is
  probably not being called for the people you expect.
- Declare the values your intervention needs with `required_fields` (see
  [Requiring values on each person](@ref)), so the simulation stops with an
  error at the start when nothing sets them.
- After a small run, look at `state.individuals[1].state` to confirm your hook
  wrote the values that later steps read.

## Combining with built-in interventions

A custom intervention combines with the built-in ones. Here the border closure
from [Extending EpiBranch](@ref "A custom intervention: closing a border") is added
to isolation of symptomatic cases. Each case infects a number of people drawn
from `NegBin(2.5, 0.16)` (mean R = 2.5, dispersion k = 0.16), the generation
time is `Exponential(5.0)` (mean 5 days), the incubation period is
`LogNormal(1.5, 0.5)` (mean and standard deviation of the log, in days) and
isolation starts 2 days after onset on average:

```@example interventions_dev
using EpiBranch
using Distributions
using StableRNGs

struct BorderClosure <: AbstractIntervention
    start_time::Float64
    leakage::Float64
end

function EpiBranch.competing_risk(bc::BorderClosure, parent, contact, state)
    parent.state[:region] == contact.state[:region] && return nothing
    return Risk(event_time = bc.start_time, block_probability = 1.0 - bc.leakage)
end

attrs = [
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    (rng, ind) -> (ind.state[:region] = rand(rng, (:north, :south))),
]
iso = Isolation(onset_to_isolation_delay = Exponential(2.0), duration = 7.0)
process = BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0))

isolation_only = simulate(ModelSpec(process; interventions = [iso], attributes = attrs),
    200; max_cases = 500, rng = StableRNG(42))
with_closure = simulate(
    ModelSpec(process; interventions = [iso, BorderClosure(10.0, 0.05)], attributes = attrs),
    200; max_cases = 500, rng = StableRNG(42))
(isolation_only = containment_probability(isolation_only),
 with_closure = containment_probability(with_closure))
```

Adding the closure raises the proportion of outbreaks contained, because from
day 10 most transmission between regions is blocked on top of what isolation
prevents.

## Who triggers contact tracing

[`ContactTracing`](@ref) decides which cases have their contacts traced
through an eligibility policy, such as `OnSymptomOnset()` or
`OnLabConfirmation()`, and policies combine with `&`, `|` and `!` (see
[Isolation and contact tracing](interventions.md)). A policy the built-ins cannot express is a
type subtyping `EpiBranch.TraceEligibility` with one `is_eligible` method,
which receives the infector, the contact and the simulation state. It then
combines with the operators like a built-in policy.

Here only symptomatic infectors aged 65 or over have their contacts traced:

```@example tracing_dev
using EpiBranch
using Distributions
using StableRNGs

struct SymptomaticOver65 <: EpiBranch.TraceEligibility end

function EpiBranch.is_eligible(::SymptomaticOver65, infector, contact, state)
    !EpiBranch.is_asymptomatic(infector) && get(infector.state, :age, 0) >= 65
end

attrs = [clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    demographics(age_distribution = Normal(50, 20))]
iso = Isolation(onset_to_isolation_delay = Exponential(2.0), duration = 7.0)
process = BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0))
n_traced(runs) = sum(s -> count(ind -> get(ind.state, :traced, false), s.individuals), runs)

everyone = ContactTracing(OnSymptomOnset(), 0.7, Exponential(1.0), Quarantine(duration = Inf))
older_only = ContactTracing(SymptomaticOver65(), 0.7, Exponential(1.0), Quarantine(duration = Inf))
traced_all = simulate(ModelSpec(process; interventions = [iso, everyone], attributes = attrs),
    100; max_cases = nothing, max_generations = 4, rng = StableRNG(5))
traced_older = simulate(ModelSpec(process; interventions = [iso, older_only], attributes = attrs),
    100; max_cases = nothing, max_generations = 4, rng = StableRNG(5))
(all_symptomatic = n_traced(traced_all), symptomatic_over_65 = n_traced(traced_older))
```

The runs stop after four generations (`max_generations = 4`, with
`max_cases = nothing`), which leaves every contact of the earlier cases
decided; a run stopped at `max_cases` would also count contacts traced before
their infection was decided. With ages drawn from `Normal(50, 20)`, nearly a
quarter of infectors are 65 or over, so far fewer contacts are traced, even
though the outbreaks under the narrower policy grow larger. Combined with a built-in policy,
`SymptomaticOver65() | OnLabConfirmation()` traces the contacts of older
symptomatic cases and of every laboratory-confirmed case.

## Ending an infection early

A treatment that ends an infection before symptom onset, such as a
post-exposure antiviral, calls [`EpiBranch.abort_infection!`](@ref) with the
time the infection ends. On every transmission model the case then transmits
nothing from that time, never develops symptoms (so it is never isolated on
symptoms) and has no clinical transitions from that time on (see [Individual
state and reserved keys](@ref)).

Here every exposed contact is treated and its infection ends `delay` days after
exposure, unless symptoms would start first:

```@example interventions_dev
struct Antiviral <: AbstractIntervention
    delay::Float64
end

function treat!(av::Antiviral, ind)
    incubation = get(ind.state, :incubation_period, NaN)
    ends = ind.infection_time + av.delay
    if !isnan(incubation) && ends >= ind.infection_time + incubation
        return nothing    # symptoms start before the treatment would work
    end
    return EpiBranch.abort_infection!(ind, ends)
end

# Each case, once its infection time is fixed.
function EpiBranch.on_infection_settled!(av::Antiviral, ind, state, rng)
    return treat!(av, ind)
end
```

Every built-in model calls `on_infection_settled!` once a case's infection time
is fixed, before its onset or clinical transitions read it. A model written
against `generate_offspring` or `contacts_of` does not call it. To treat
contacts there, add a `trace_contacts!` method that calls `treat!` on each
contact at its potential infection time, before the contact is known to be
infected; a contact that turns out to be infected at or after the time the
treatment ends it, or not at all, keeps no record of the treatment.

## Susceptibility and infectiousness are risks too

A person's own susceptibility and an infector's infectiousness are handled the
same way as an intervention: as risks that can block a transmission. The
simulation puts every transmission to the built-in risks and your
interventions alike. `competing_risk` is the single way to block a
transmission, whether the reason is a vaccine, a border closure or the
person's own susceptibility.

Six built-in risks are always present:

- [`EpiBranch.HostSusceptibility`](@ref) blocks with probability
  `1 - susceptibility` of the contact. A contact with susceptibility 0.3
  escapes 70% of exposures.
- [`EpiBranch.InfectorInfectiousness`](@ref) blocks with probability
  `1 - infectiousness` of the infector.
- [`EpiBranch.InfectiousSource`](@ref) blocks every transmission from someone
  who is not infected. A ring of contact tracing can reach the contacts of
  someone never infected (see [Following up contacts who were not
  infected](@ref)), but only infected people make exposures, so it has nothing
  to block on the built-in models.
- The end of an infectiousness window blocks a contact timed at or after the
  earliest `until` state of the infector's window, such as their death or
  recovery (see [Infectiousness windows](@ref)). A window with no `until`
  blocks nothing.
- [`EpiBranch.AbortedInfection`](@ref) blocks every transmission an infector
  makes from its `:infection_aborted_time`, so an infection ended by
  [`EpiBranch.abort_infection!`](@ref) stays ended after the intervention that
  ended it stops being active.
- [`EpiBranch.HostImmunity`](@ref) blocks every exposure of a contact who has
  already been infected, until [`susceptible_again_time`](@ref) is in the
  past. It does nothing for a model whose `progression` never reaches
  `:susceptible_again`. A model that does gets a new, separately recorded infection episode
  (see [Individual state and reserved keys](@ref)) instead of a reinfection
  overwriting the earlier one.

A susceptibility or infectiousness of `1.0` blocks nothing, and these risks have
no effect unless a population characteristic sets a value below one. You can
add to them with a `competing_risk` of your own. They always apply, so a risk of
your own should not apply susceptibility or infectiousness a second time.

`HostSusceptibility` and `InfectorInfectiousness` apply contact by contact on
branching processes and on models written against `contacts_of`. The other
fixed-population models apply the same two values as multipliers on the rate
of transmission instead (see [Continuous-time models: further details](@ref)),
so they do not decide them contact by contact. Of the remaining four,
`InfectiousSource` never has anything to block on any model, since the race
proposes only from cases that have settled and are infected by construction, and the end of
an infectiousness window is enforced by the window closing rather than by
`competing_risk`. `AbortedInfection` and `HostImmunity`, by contrast, are both
resolved through `competing_risk` there: a settled member can become a target
again and is blocked by `HostImmunity` until their immunity has waned, exactly
when the `progression` reaches `:susceptible_again`, with no change needed to
the model's own `targets`/`contacts`. A risk of your own applies there as it
does on a branching process.

## Following up contacts who were not infected

Contact tracing beyond direct contacts needs the contacts of a case's
contacts, even of those never infected. `keep_active` is asked during tracing,
straight after `trace_contacts!`: return the ids of the traced contacts the
ring is traced onward from.

On a branching process whose window opens at infection, the default, a
contact's own contacts are drawn when the ring reaches it, from its potential
infection time and before its infection is decided. The ring grows through
them whether or not the contact is infected, and if it is, they are the
contacts it goes on to expose.

An offspring or generation time given as a function may read what settling the
contact writes, such as its severity, and a generation time from
[`incubation_linked_generation_time`](@ref) is such a function. With one, a
contact's own contacts are drawn only once it settles as a case, and the ring
stops at a contact who is never infected (see
[`EpiBranch.reads_settled_state`](@ref)). Once an infected contact settles,
its contacts are traced from when the ring reached it. If it is eligible for
tracing in its own right, they are traced again from its own trigger, as they
would be had they been drawn early. A window that opens on a later state, such
as `:infectious` after a latent period, has no opening time yet and waits the
same way. When any of a contact's windows waits, all of them do: the ring grows
past a contact only once, so every contact an infected contact makes has to be
known by then.

A model written against `generate_offspring` or `contacts_of` waits the same
way, unless it declares that its count, or its `contacts_of`, reads nothing
settling writes (see [`EpiBranch.reads_settled_state`](@ref)). A
`generate_offspring` model that declares it has its count drawn when the ring
reaches a person, as on a [`BranchingProcess`](@ref) with a `Distribution` of
offspring.

[`ContactTracing`](@ref) with `depth > 1` is the built-in intervention that
uses this: it grows the ring from its members until the ring's depth is
reached. At `depth = 2` the ring reaches the contacts of contacts, whom ring
vaccination then targets.

## Scheduling an intervention

[`Scheduled`](@ref) starts any intervention on a given day:
`Scheduled(iv; start_time = ...)`. Built-in interventions have no start-time
field of their own. `Scheduled` does not hold back `on_infection_settled!`: an
intervention that acts through that hook acts from the start of the outbreak
even when scheduled.

For an intervention that acts on each case through `resolve_individual!` or
`trace_contacts!` and defines its own `intervention_time`, `Scheduled` runs it
whether or not the schedule is active at the case's or contact's own clock,
then checks the schedule again at the time `intervention_time` reports and
undoes the effect with `reset!` if that falls outside it. An index case,
infected on day 0, is therefore still isolated by a policy starting on day 10
once their own isolation, after onset and a delay, falls on or after it (see
[Interventions that start or stop during the outbreak](@ref)). An intervention
that proposes actions (see [Proposing actions of your own](@ref)) is checked
against each action's own date the same way. One with no `intervention_time`
of its own is instead gated before it runs, exactly as `Scheduled` always
gated every hook: its default, `-Inf`, leaves `reset!` nothing to undo once
the hook has already acted.

To opt into the action's-own-time check, the intervention defines two
methods:

- `EpiBranch.intervention_time(intervention, individual)`: the time at which the
  intervention acts on the person (for example the isolation time).
- `EpiBranch.reset!(intervention, individual)`: undo its effect on the person.

A simplified version of the pair `Isolation` defines:

```julia
EpiBranch.intervention_time(::Isolation, ind::Individual) = isolation_time(ind)

function EpiBranch.reset!(::Isolation, ind::Individual)
    clear_isolated!(ind)
    return nothing
end
```

A user then schedules the intervention as any other:

```julia
# The closure has no date of its own (day 0); Scheduled starts it on day 10.
Scheduled(BorderClosure(0.0, 0.05); start_time = 10.0)
```

The border closure's own `start_time` is compared with each transmission time.
`Scheduled` instead switches the whole policy on when the simulation reaches
day 10, which is how built-in interventions are started.

### A trigger of your own

`start_after` opens a schedule on a condition instead of a date, and takes any
subtype of [`AbstractTrigger`](@ref EpiBranch.AbstractTrigger). The built-in
[`Infections`](@ref) and [`ReportedCases`](@ref) are two; a trigger of your own
works without any change to `Scheduled`. It needs one method, and has two
traits with defaults:

- `EpiBranch.is_triggered!(trigger, state)` says whether the condition holds
  now. The same trigger is reused across the runs of `simulate(model, n)`, so
  keep nothing on it between calls and work out any count from `state` each
  time. It is asked about 1.7 times per person, so a count that goes through
  the whole population on every call, as `ReportedCases` does, makes a run
  take time that grows with the square of the number of people. A count the
  simulation already keeps, as `Infections` reads `state.cumulative_cases`,
  costs nothing.
- `EpiBranch.trigger_can_lapse(trigger)` says whether the condition, once true,
  can turn false again. The default is `true`, the safe choice. Return `false`
  for a condition that holds for good once met, as both built-in triggers do:
  neither the case count nor the number of reports already made can fall. The
  answer matters for fitting: a schedule that can close again gives up the
  isolation releases the likelihood would otherwise use.
- `EpiBranch.required_transitions(trigger)` names the clinical transitions the
  model's `progression` must include for the trigger to work, `()` by default.
  Building a `ModelSpec` without one of them gives a warning naming the
  trigger, because without the transition that writes what the trigger reads
  the schedule never opens. `ReportedCases` names `Reporting`, except at
  `n = 0`, when the schedule opens before any report exists.
- `EpiBranch.required_fields(trigger)` names the population characteristics
  the trigger reads from `Individual.state`, `Symbol[]` by default.
  `Scheduled` adds these to the fields it already requires of the
  intervention it wraps (see [Requiring values on each person](@ref)), so the
  simulation's startup check catches a model missing either.

Here the border closes once 20 cases have had symptom onset:

```julia
struct TwentyOnsets <: EpiBranch.AbstractTrigger end

function EpiBranch.is_triggered!(::TwentyOnsets, state)
    return count(
        ind -> is_infected(ind) && onset_time(ind) <= state.max_infection_time,
        state.individuals
    ) >= 20
end
EpiBranch.required_fields(::TwentyOnsets) = [:onset_time]

Scheduled(BorderClosure(0.0, 0.05); start_after = TwentyOnsets())
```

`TwentyOnsets` keeps the default `trigger_can_lapse`: a person infected a second
time draws a new onset, so the count can fall. It reads `:onset_time`, which
[`clinical_presentation`](@ref) sets, and names it with `required_fields`: a
model built without that attribute now stops at the start with an error
naming it, instead of leaving the border open for the whole run with nothing
to say why.

## Capacity limits

[`CapacityConstrained`](@ref) limits how many actions an intervention can take
per period, such as the doses available or the appointments a clinic can offer.
Contact tracing itself cannot be limited this way. Your intervention defines `EpiBranch.capacity_key(iv)`, the name of the
`true`/`false` value recording that a person used the resource, and
`EpiBranch.capacity_time_key(iv)`, the name of the value recording when. This
places usage in the period its date falls in, whichever intervention sharing
the same capacity gave it. Ring, group and mass vaccination use their
dose-label names.

Every proposed recipient is known before admission, including group members
found by searching a whole group. See [Intervention actions](@ref) for how to
propose actions and the rules on timing.

By default everyone draws on the same budget. Define
`EpiBranch.resource_key(iv, individual, state)` to split it instead, returning
whichever value identifies the group `individual` belongs to — a district, a
treatment centre. Each distinct value gets its own budget, tracked under the
same `budget_per_period`, `period`, `carry_over` and `priority`, independently
of every other. It dispatches on the individual rather than a proposed action,
so someone already served can be placed in their group again from their
recorded state alone, with nothing of the original proposal kept around.
`capacity_usage(cc, state, key)` reads one group's own usage.

## Requiring values on each person

If your intervention depends on values set by a population characteristic
(such as `:onset_time`), declare them with `EpiBranch.required_fields`, and the
simulation stops with a clear error at the start when they are missing:

```julia
EpiBranch.required_fields(::MyIntervention) = [:onset_time, :asymptomatic]
```

## A custom vaccination

A new vaccination differs from the built-in ones in whom it reaches and when.
What a dose does once given (`efficacy`, `severity_efficacy`,
`delay_to_immunity`, `mode` and `dose_label`) is described by a
[`VaccineEffect`](@ref), which every [`AbstractVaccination`](@ref) holds. A new
vaccination type stores one and returns it from `EpiBranch.vaccine_effect`; the
rest of the vaccination code reads these parameters only through that method.
The new type then gets, without further code:

- `initialise_individual!`, which records every person as unvaccinated
  (`:vaccinated` and `:vaccination_time`, named after the `dose_label`) unless
  a population characteristic already recorded a dose, such as one from an
  earlier campaign;
- `competing_risk`, the reduced susceptibility described in
  [`AbstractVaccination`](@ref);
- the checks on dose schedules made when a `ModelSpec` is built, so a later
  [`RingVaccination`](@ref) can name its dose in `requires_dose`.

It adds an `intervention_actions` method choosing whom to vaccinate and when,
returning one `EpiBranch.dose_action(v, ind, time)` for each person it will
dose. That is the same action protocol the built-in vaccinations use, so
[`Scheduled`](@ref) wraps your vaccination as it wraps a
[`MassVaccination`](@ref), and [`CapacityConstrained`](@ref) does once the type
defines the two keys described in [Capacity limits](@ref). The simulation puts
the actions through those limits itself. A dose
admitted this way writes the per-dose values listed in [Individual state and
reserved keys](@ref) and draws `efficacy`, `severity_efficacy` and
`delay_to_immunity` for that person, whether they were given as numbers,
distributions or functions.

Here a campaign on day 10 reaches everyone aged 60 or over. `kwargs...` passes
any further keywords on, as `...` does in R:

```@example interventions_dev
struct OlderAdultVaccination{V <: VaccineEffect, B} <: AbstractVaccination
    effect::V
    min_age::Int
    campaign_time::Float64
    booster_uptake::B
end

function OlderAdultVaccination(; min_age, campaign_time, booster_uptake = 0.0,
        kwargs...)
    OlderAdultVaccination(VaccineEffect(; kwargs...), min_age, campaign_time,
        booster_uptake)
end

EpiBranch.vaccine_effect(v::OlderAdultVaccination) = v.effect
EpiBranch.required_fields(::OlderAdultVaccination) = [:age]

function EpiBranch.intervention_actions(v::OlderAdultVaccination, state, candidates)
    actions = EpiBranch.InterventionAction[]
    for ind in candidates
        ind.state[:age] >= v.min_age || continue
        push!(actions, EpiBranch.dose_action(v, ind, v.campaign_time))
    end
    return actions
end

older = OlderAdultVaccination(min_age = 60, campaign_time = 10.0,
    efficacy = 0.8, delay_to_immunity = 14.0)
older_model = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0));
    interventions = [older], attributes = demographics())
older_results = simulate(older_model, 50; max_cases = 200, rng = StableRNG(1))
n_people = sum(s -> length(s.individuals), older_results)
n_vaccinated = sum(s -> count(is_vaccinated, s.individuals), older_results)
(vaccinated = n_vaccinated, people = n_people)
```

Only contacts aged 60 or over are vaccinated, and the vaccinated make up a
minority of the people in these outbreaks.

Passing the keywords on to `VaccineEffect` lets the constructor take the same
effect keywords as the built-in vaccinations. A parameter describing what a
dose does belongs in `VaccineEffect`, where every vaccination gains it at once;
a parameter describing whom a dose reaches belongs on the new type.

An effect only your vaccination has is a field of the type (`booster_uptake`
above). Its per-dose draw is recorded by a method of
`EpiBranch.record_effect_draws!`, which every dose calls. `RingVaccination`
records `post_exposure_efficacy` and `onward_efficacy` that way:

```julia
_booster_uptake_key(label) = Symbol("booster_uptake_", label)

function EpiBranch.record_effect_draws!(v::OlderAdultVaccination, contact, label, rng)
    EpiBranch.store_draw!(v.booster_uptake, _booster_uptake_key, label, contact, rng)
    return nothing
end
```

For a number, `store_draw!` stores nothing and `EpiBranch.dose_value` reads
the value from the vaccination itself; for a distribution or a function it
stores the draw.

### A custom effect mode

A vaccine is leaky (`LeakyMode`: every vaccinee's risk is reduced) or
all-or-nothing (`AllOrNothingMode`: a proportion of vaccinees are fully
protected and the rest not at all). A third kind subtypes
[`AbstractEffectMode`](@ref) and has a method of
[`EpiBranch.realised_efficacy`](@ref), which turns the efficacy a dose was
given into the value stored on the person. Nothing else in the vaccination code
needs to change.

Here a "partial responder" mode gives a proportion `efficacy` of vaccinees full
protection, as all-or-nothing does, but gives the rest a fixed lower level of
leaky protection instead of none. The example records eight doses directly,
with the same dose functions the recipe above uses:

```@example interventions_dev
struct PartialResponseMode <: AbstractEffectMode
    non_responder_efficacy::Float64
end

function EpiBranch.realised_efficacy(mode::PartialResponseMode, eff, rng)
    rand(rng, Bernoulli(eff)) && return 1.0
    return mode.non_responder_efficacy
end

partial = RingVaccination(efficacy = 0.6, mode = PartialResponseMode(0.2))
draws = map(1:8) do i
    contact = Individual(id = i, parent_id = 0, infection_time = 10.0)
    EpiBranch.record_dose!(partial, contact, 0.0, StableRNG(i))
    vaccine_efficacy(contact)
end
draws
```

Every stored value is `1.0` (a responder) or `0.2` (the protection given to a
non-responder), unlike the `0.6` a leaky vaccine would store or the `0.0` an
all-or-nothing vaccine gives a non-responder.

## Intervention actions

An intervention proposes actions, and `Scheduled` and `CapacityConstrained`
decide which go ahead. Each proposal names a person, a date and a function that
records the effect. `Scheduled` checks the date and `CapacityConstrained`
checks the budget, both before the effect is recorded.

### Schedule delivery and retain its protection

Suppose an index case makes two contacts on day 20 (`Dirac(2)` always gives 2
contacts, `Dirac(20.0)` always a generation time of 20 days). Both contacts can
be vaccinated on day 10, while the campaign runs:

```@example action_delivery
using EpiBranch, Distributions, Random

process = BranchingProcess(Dirac(2), Dirac(20.0))
vaccine = MassVaccination(efficacy = 1.0, eligibility_time = 10.0)
campaign = Scheduled(vaccine; start_time = 10.0, end_time = 10.0)
model = ModelSpec(process; interventions = [campaign])
state = simulate(model; max_generations = 1, rng = Xoshiro(42))

(cases = state.cumulative_cases,
 doses = count(is_vaccinated, state.individuals))
```

The result is one case and two doses: the index case remains infected, and both
contacts are protected before their exposures on day 20. The campaign ends on
day 10, which stops new doses; the protection already given lasts.

Now limit the campaign to one dose:

```@example action_delivery
limited = CapacityConstrained(campaign; budget_per_period = 1.0)
limited_model = ModelSpec(process; interventions = [limited])
limited_state = simulate(limited_model; max_generations = 1, rng = Xoshiro(42))

(cases = limited_state.cumulative_cases,
 doses = count(is_vaccinated, limited_state.individuals))
```

This gives two cases and one dose: only one of the two contacts is protected.
`Scheduled(CapacityConstrained(vaccine; budget_per_period = 1.0);
start_time = 10.0, end_time = 10.0)` gives the same result, because both orders
check the proposed delivery date and count only doses that go ahead.

Capacity counts a dose against the budget of the day it is dated, not the
clock of the case that proposes it. Here both contacts' doses are dated day
10, so they draw on the same budget whatever day the index case was processed
on; `period = 7.0` would charge the second week, `[7, 14)`, not the first. The
default `period = Inf` gives one budget for the whole simulation.

### Proposing actions of your own

A clinic appointment can use the same scheduling and capacity limits. This
example offers each new contact an appointment on a fixed date and records
attendance only if the appointment goes ahead:

```@example clinic_action
using EpiBranch, Distributions, Random

struct ClinicAppointment <: EpiBranch.AbstractIntervention
    time::Float64
end

function record_attendance!(person, time, state)
    person.state[:attended] = true
    person.state[:appointment_time] = time
    return nothing
end

function EpiBranch.intervention_actions(visit::ClinicAppointment, state, candidates)
    [EpiBranch.InterventionAction(person, visit.time, record_attendance!)
     for person in candidates if !get(person.state, :attended, false)]
end

EpiBranch.capacity_key(::ClinicAppointment) = :attended
EpiBranch.capacity_time_key(::ClinicAppointment) = :appointment_time

appointments = CapacityConstrained(
    Scheduled(ClinicAppointment(10.0); start_time = 9.0, end_time = 11.0);
    budget_per_period = 1.0)
model = ModelSpec(BranchingProcess(Dirac(2), Dirac(20.0));
    interventions = [appointments])
state = simulate(model; max_generations = 1, rng = Xoshiro(42))

[person.state[:appointment_time] for person in state.individuals
 if get(person.state, :attended, false)]
```

The output is `[10.0]`. Two appointments are proposed and the budget allows
one. `record_attendance!` records that the resource was used and when.
Attending has no effect on transmission, and all three people are infected here.

With the appointment on day 12 instead, nobody attends, because the schedule
ends on day 11. A proposal that `Scheduled` or `CapacityConstrained` turns down
is considered again only if the intervention proposes it again; nothing keeps
a queue of appointments. The rules for proposing, admitting and keeping effects
are in [Proposing, admitting and keeping actions](@ref).

## Custom clinical transitions

A case's own progression (testing, reporting, admission, treatment, recovery or
death) is described by clinical transitions, given as the `progression` of a
`ModelSpec`; the [Clinical transitions](transitions.md) tutorial covers the
built-in ones. A transition of your own subtypes
[`AbstractClinicalTransition`](@ref) and uses the same hooks as an
intervention, called once per case as it is settled: `initialise_individual!`
to set defaults, then `resolve_individual!` to work out what happens to the
case and when. A person who is never infected never has the transition's
values set, so code reading them uses `get` with a default.

### A milestone other transitions can use

Suppose a case is reported only after a positive test, and the reporting delay
counts from the test, not from onset. A `Testing` transition records
`:test_time`, and the built-in `Reporting` then starts from it with
`from = :test_time`. The incubation period is `LogNormal(1.5, 0.5)` and the
test follows onset after `LogNormal(0.5, 0.3)` days (mean and standard
deviation of the log):

```@example transitions_dev
using EpiBranch
using Distributions
using StableRNGs

clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

struct Testing <: AbstractClinicalTransition
    delay::Distribution
    sensitivity::Float64
end

EpiBranch.required_fields(::Testing) = [:onset_time]

function EpiBranch.initialise_individual!(::Testing, ind, state)
    ind.state[:tested] = false
    ind.state[:test_time] = Inf
    return nothing
end

function EpiBranch.resolve_individual!(t::Testing, ind, state)
    ot = onset_time(ind)
    isnan(ot) && return nothing                        # no symptoms, no test
    rand(state.rng) < t.sensitivity || return nothing  # test misses the case
    ind.state[:tested] = true
    ind.state[:test_time] = ot + rand(state.rng, t.delay)
    return nothing
end

testing = Testing(LogNormal(0.5, 0.3), 0.9)
reporting_after_test = Reporting(delay = LogNormal(0.0, 0.2), from = :test_time)

model = ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
    progression = [testing, reporting_after_test], attributes = clinical)
state = simulate(model; max_cases = 100, rng = StableRNG(42))

ind = state.individuals[end]
(onset = onset_time(ind), tested = ind.state[:test_time],
 reported = ind.state[:reporting_time])
```

For a tested case the report follows the test, which follows onset. `Reporting`
skips a case whose `:test_time` is still `Inf` because it was never tested, in
the same way as it skips an asymptomatic case when timed from onset.
Transitions are worked out in the order of the `progression` list, so a
transition that reads another's values must come after it.

### Competing clinical outcomes

[`Death`](@ref) and [`Recovery`](@ref) end the case: they are terminal. Any
transition for which `is_terminal` returns `true`, and which has a
`terminal_event(t, ind)` method returning `(time, label)`, takes part. After
every transition is worked out, the earliest terminal event across the list
becomes the case's outcome, recorded as `:outcome` (the label) and
`:outcome_time`.

A third outcome, loss to follow-up, is another type with these methods.
`terminal_target` names the state it ends in, so EpiBranch can check that an
infectious period ends there (see [Terminal transitions and the end of
transmission](@ref)):

```@example transitions_dev
struct LostToFollowUp <: AbstractClinicalTransition
    delay::Distribution
    probability::Float64
end

EpiBranch.required_fields(::LostToFollowUp) = [:onset_time]
EpiBranch.is_terminal(::LostToFollowUp) = true
EpiBranch.terminal_target(::LostToFollowUp) = :lost

function EpiBranch.initialise_individual!(::LostToFollowUp, ind, state)
    ind.state[:lost_candidate_time] = Inf
    return nothing
end

function EpiBranch.resolve_individual!(t::LostToFollowUp, ind, state)
    ot = onset_time(ind)
    isnan(ot) && return nothing
    rand(state.rng) < t.probability || return nothing
    ind.state[:lost_candidate_time] = ot + rand(state.rng, t.delay)
    return nothing
end

function EpiBranch.terminal_event(::LostToFollowUp, ind)
    t = get(ind.state, :lost_candidate_time, Inf)
    return isfinite(t) ? (t, :lost) : nothing
end

progression = [
    Death(delay = LogNormal(2.5, 0.4), probability = 0.05),
    Recovery(delay = LogNormal(2.0, 0.4)),
    LostToFollowUp(LogNormal(1.5, 0.5), 0.1),
]
model = ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
    progression = progression, attributes = clinical)
state = simulate(model; max_cases = 200, rng = StableRNG(42))

outcomes = [ind.state[:outcome] for ind in state.individuals if haskey(ind.state, :outcome)]
(died = count(==(:died), outcomes), recovered = count(==(:recovered), outcomes),
 lost = count(==(:lost), outcomes))
```

Recovery has no probability, and most symptomatic cases recover. Loss
to follow-up, after about 4.5 days from onset (the median of
`LogNormal(1.5, 0.5)`), tends to come before recovery, about 7.4 days, and
death, about 12 days. The cases lost are mostly ones who would otherwise
have recovered. The same pattern works for admission to intensive care, or
outcomes that depend on treatment.

### A transition that does not end the case

A transition that is not terminal leaves out `is_terminal` and
`terminal_event`. It marks a point on the timeline that other transitions or
observation can read. Here reported cases are treated with an antiviral with
some probability:

```@example transitions_dev
struct AntiviralTreatment <: AbstractClinicalTransition
    delay::Distribution
    probability::Float64
end

EpiBranch.required_fields(::AntiviralTreatment) = [:onset_time]

function EpiBranch.initialise_individual!(::AntiviralTreatment, ind, state)
    ind.state[:treated] = false
    ind.state[:treatment_time] = Inf
    return nothing
end

function EpiBranch.resolve_individual!(t::AntiviralTreatment, ind, state)
    ot = onset_time(ind)
    isnan(ot) && return nothing
    get(ind.state, :reported, false) || return nothing   # only reported cases
    rand(state.rng) < t.probability || return nothing
    ind.state[:treated] = true
    ind.state[:treatment_time] = ot + rand(state.rng, t.delay)
    return nothing
end
```

A later `Death` transition can read `:treated` in its probability, so that
treated cases die less often:

```julia
Death(
    delay = LogNormal(2.5, 0.4),
    probability = (rng, ind) -> ind.state[:treated] ? 0.02 : 0.08,
)
```

To let [`progression_loglik`](@ref) evaluate the clinical timeline, the
transition needs a [`EpiBranch.transition_loglik`](@ref) method; without one,
`progression_loglik` stops with an error. It reads back the values
`resolve_individual!` wrote and returns `0.0` when the transition could not
have happened:

```@example transitions_dev
function EpiBranch.transition_loglik(t::AntiviralTreatment, ind)
    ot = onset_time(ind)
    isnan(ot) && return 0.0
    get(ind.state, :reported, false) || return 0.0
    treated = ind.state[:treated]
    time = treated ? ind.state[:treatment_time] : Inf
    occurred = treated && time <= outcome_time(ind)
    ll = EpiBranch.transition_term(
        t.probability, t.delay, ind, ot, occurred;
        censor = min(EpiBranch.infection_aborted_time(ind), outcome_time(ind))
    )
    occurred || return ll
    return ll + logpdf(t.delay, time - ot)
end

treated = ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
    progression = [Reporting(delay = LogNormal(1.0, 0.3), probability = 0.7),
        AntiviralTreatment(LogNormal(0.5, 0.3), 0.6)],
    attributes = clinical)
treated_state = simulate(treated; max_cases = 200, rng = StableRNG(42))
progression_loglik(treated, treated_state)
```

[`EpiBranch.transition_term`](@ref) gives the term for whether the transition
happened. Reading `t.probability` directly would be wrong in two cases: a
probability built by [`exclusive_probabilities`](@ref), whose alternatives
share one random draw, and a case whose infection ended early, before the
transition could take effect. [The likelihood of a custom transition](@ref)
explains both.

A treatment drawn after the case has died or recovered did not happen. The
`censor` passed to `transition_term` above reads it that way, and the
transition says so on the case too with a
[`EpiBranch.censor_after_outcome!`](@ref) method, as the built-in `Reporting`
and `Hospitalisation` do. The package calls it once the case's outcome is
known:

```@example transitions_dev
function EpiBranch.censor_after_outcome!(::AntiviralTreatment, ind)
    if ind.state[:treatment_time] > outcome_time(ind)
        ind.state[:treated] = false
        ind.state[:treatment_time] = Inf
    end
    return nothing
end
```
