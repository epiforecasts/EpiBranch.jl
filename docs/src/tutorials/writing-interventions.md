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
methods only for the hooks it needs; every other hook does nothing.

## What happens in one generation

On a branching process, each generation of the simulation runs these steps in
order, and each step calls one hook on every intervention:

1. Each case about to transmit is prepared with `resolve_individual!`. This is
   where `Isolation` works out when a case is isolated, from its onset time.
2. The transmission model gives each case its potential secondary cases and a
   potential transmission time for each.
3. Each new person is created, and `initialise_individual!` sets the values the
   intervention keeps for them (`ContactTracing` sets `:traced` to `false`).
4. Once the whole generation's contacts exist, `apply_post_transmission!`
   receives them all. Contact tracing and vaccination of contacts happen here.
5. For each pair of infector and contact, `competing_risk` says whether
   anything stops the transmission. The contact is infected only if nothing
   does.
6. `keep_active` can keep contacts who were not infected in the simulation, so
   their own contacts are generated too (needed for tracing contacts of
   contacts).

Interventions are applied in the order they appear in `interventions = [...]`,
and each sees what earlier ones wrote in the same generation.

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
| `resolve_individual!(iv, individual, state)` | Once per active case at the start of each generation, before its contacts are drawn | The infector for the coming step | `nothing`; sets values in `individual.state` |
| `apply_post_transmission!(iv, state, new_contacts)` | Once per generation, after every active case's contacts for that generation exist | A `Vector{Individual}` of the new contacts | `nothing`; sets values on any of the contacts |
| `competing_risk(iv, parent, contact, state)` | Per infector–contact pair: on branching processes when infection is decided, after `apply_post_transmission!`; on continuous-time models as each infection is proposed | The infector and one contact | `nothing`, one [`Risk`](@ref), or a tuple of `Risk`s for an intervention that blocks transmission in more than one way |
| `keep_active(iv, state, targets, is_new)` | Once per generation after infection is decided, while the next generation's active people are chosen | This generation's contacts and an `is_new` flag for each | The ids of contacts that should keep generating contacts in the next generation (default: none) |
| `trace_contacts!(iv, state, infector, contacts[, not_before])` | Continuous-time models only: once per case, when its infection time is final | The case, the contacts it reached whose infection is not yet final, and, from a model whose contacts can arise after the case's infection, when each became a contact (the four-argument method is called when the model gives no times, and by default for interventions that ignore them) | `nothing`; sets values on the contacts |
| `traces_contacts(iv)` | Whenever a continuous-time model decides whether to collect a case's contacts at all | Nothing | `true` if this intervention has a `trace_contacts!` method (default `false`) |
| `infectious_removal_time(iv, individual)` | Continuous-time models only: when a case's infectious period is closed | A person | The time this intervention removes them from onward transmission (default `Inf`) |
| `on_infection_settled!(iv, individual, state, rng)` | Network and household models only: once a case's infection time is final, before its onset or clinical transitions read it | The case, and the random number generator to use | `nothing`; sets values on the case (default: does nothing) |
| `risk_applies(iv, route)` | Continuous-time models choosing the risks for a route (`nothing` for an introduction from outside) | Nothing | `Bool`; default `true` |
| `standing_block(iv)` | Continuous-time models deciding whether a certain block ends a pair's contacts for good | Nothing | `Bool`; default `false` |
| `risk_depends_on_infector(iv)` | Before a closed population with more than one mixing group runs | Nothing | `Bool`: whether `competing_risk` can block a contact differently depending on its infector (default `true` when the type has its own `competing_risk`) |
| `reads_population_state(iv)` | Before a structure-driven model (such as `HouseholdProcess`) decides whether to simulate each household separately or all on one shared clock | Nothing | `Bool`: whether what the intervention does can depend on population-wide quantities such as a running case count or a shared budget (default `true`, the safe choice) |

`reads_population_state` covers everything an intervention hands work to: a
component or function you supply counts towards its owner's answer. A built-in
intervention asks its own components (`ContactTracing` its eligibility, rate,
delay and action; `Isolation` its eligibility), each answering `false` by
default. An eligibility, rate, delay or action of yours that tests a running
case count, or anything else beyond the person being considered, declares
`true` for itself, and its owner then answers `true` too. A plain function
passed as a parameter has nowhere to declare this, and such a test belongs in a
component type.

The [`AbstractIntervention`](@ref) docstring and the
[Extension reference](@ref "Continuous-time models: further details") have the
remaining details for the continuous-time models.

## Which hooks run on which model

Not every hook is called by every model. A branching process creates a new
`Individual` for every contact, infected or not, and can hand an intervention
a batch of contacts. The continuous-time models (network, household and the
homogeneous pool) create everyone at the start and only work out *when* each
person is infected, from draws of the contact interval. What they do have is
each potential infection, a drawn time for a named pair, and a `Risk` can act
on that, as can the end of the infectious period.

| Hook | Branching process | Network / household | Homogeneous pool |
|---|---|---|---|
| `initialise_individual!` | yes | yes | yes |
| `resolve_individual!` | yes | yes | yes |
| `competing_risk` | yes | yes | yes |
| `infectious_removal_time` | not read | yes | yes |
| `on_infection_settled!` | not called | yes | not called |
| `trace_contacts!` | not called | yes | no contacts to trace |
| `apply_post_transmission!` | yes | not called | not called |
| `keep_active` | yes | not called | not called |

In practice:

- An intervention that **removes** a case from transmission (isolation,
  quarantine once traced, hospitalisation) works on every model, because it
  shortens the infectious period, which every model has.
- An intervention that blocks each contact with some probability (a leaky
  vaccine, partially effective prophylaxis) also works everywhere, and so do
  per-person susceptibility and infectiousness. The continuous-time models put
  each potential infection to the risks when they propose it, at the time
  proposed. A blocked contact does not transmit, and the pair keeps meeting:
  on a network the pair's next contact is drawn later, and in the homogeneous
  pool the susceptible waits for its next contact.

!!! warning "The same efficacy means different things on the two kinds of model"
    On a continuous-time model a pair that escapes infection keeps meeting, so
    blocking each contact with probability `p` multiplies the rate of infection
    by `1 - p`. A two-person household with a one-day mean contact interval
    and a two-day infectious period behaves like a homogeneous pool of two at
    `β = 2`, and at efficacy 0.5 both infect `1 - exp(-1) = 0.63` of the time.
    On a branching process a case's contacts are a fixed set of draws, so a
    blocked contact is a transmission lost, and the same efficacy of 0.5
    halves that pair's chance of transmission (0.43 in this example, against
    0.63). The per-exposure reading on continuous-time models is what a leaky
    vaccine means there; check which one your scenario needs.

- Simulating with a partial block that starts partway through the infectious
  period (a leaky isolation, say), or with an effect not declared through
  `infection_likelihood_compatible` or `susceptibility_components`, and then
  fitting the result with `loglikelihood` on a network or household model will
  disagree. Complete isolation is fitted exactly, and so is vaccination within
  the limits listed in [Likelihood compatibility](@ref).
- An intervention that reaches people only through `apply_post_transmission!`
  or `keep_active` (`MassVaccination`'s rollout vaccinates each new contact as
  it is created) has nothing to act on in a model that creates no contacts. You
  need not declare this: when your type has its own method for either hook,
  the continuous-time models name it in their warning. The exception is an
  intervention that also traces contacts (`traces_contacts` returns `true`),
  whose `trace_contacts!` is taken as the continuous-time version of those
  hooks. It is honoured on a model that can name a case's contacts and
  reported on one that cannot, such as the homogeneous pool.
- **Contact tracing** needs both: its action is a removal, so it applies on
  every model, but it has to know who a case's contacts were. A branching
  process reads that from each contact's `parent_id`. A continuous-time model
  must supply it: the model reports `EpiBranch.supplies_contacts(model) = true`
  and passes a `contacts` function. That function returns contact ids, or
  `(id, time)` pairs when some contacts arise only after the case's infection,
  such as at a funeral (`contacts_from = :died` on a `RoutedNetwork` route).
  `time` is when each person became a contact, and `ContactTracing` starts that
  contact's trace delay no earlier than then. A network names a person's
  neighbours and a household its members; the homogeneous pool has no list of
  who met whom, so tracing does not happen there.

### Order of steps on the two kinds of model

Two differences in order matter when your intervention has to work on both:

- On both kinds of model a contact is traced before its own
  `resolve_individual!` runs. On a branching process the contact's population
  characteristics and onset are already set when it is traced, and its
  `resolve_individual!` runs in the next generation. On the continuous-time
  models a contact is traced when its *infector*'s infection becomes final,
  before the contact's own infection time, and so its onset, is final. An
  intervention that writes onto a contact must therefore not assume the contact
  has nothing written yet, and one that reads a contact's own values must not
  assume they are already set. This is why `Isolation` treats a
  quarantine already in place as a competing route to isolation and keeps the
  earliest time.
- On the continuous-time models tracing reaches only contacts whose own
  infection is not yet final. Once a case's infection is final, its infectious
  period and onward transmissions are fixed, so a trace arriving afterwards has
  nothing left to shorten. Backward tracing, to the person who infected a case,
  is not supported on either kind of model.

What an intervention can rely on:

- `resolve_individual!` runs before any `competing_risk` call in that
  generation, so a risk can read what `resolve_individual!` wrote on the
  infector.
- `apply_post_transmission!` runs before any `competing_risk` call, so a risk
  can read what it wrote on the contact (for example `:vaccination_time`).
- `keep_active` runs after infection is decided, so it can read each contact's
  `:infected` and anything `apply_post_transmission!` wrote this generation.
- Interventions apply in the order of `interventions = [...]`. For
  `apply_post_transmission!` and `competing_risk`, each sees what earlier
  interventions wrote in the same generation.
- On the continuous-time models a case is traced when its infection becomes
  final, before it proposes any infection of its own, so a risk can read what
  `trace_contacts!` wrote on a contact. Ring and group vaccination actions are
  found after tracing and then pass through their scheduling and capacity
  limits.

On the continuous-time models, the transmission time a `Risk` is compared with
is the potential infection time just drawn for that pair.

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

`resolve_individual!`: `Isolation` works out when the case about to transmit
is isolated, from its onset time plus a sampled delay, and takes an earlier
isolation time if contact tracing found the case first. It then draws how
long the isolation lasts from its `duration` and passes the end to
`set_isolated!` as `release_time`. `x || return nothing`
means "stop here unless `x` is true".

```julia
function resolve_individual!(iso::Isolation, individual, state)
    is_test_positive(individual) || return nothing

    iso_delay = _sample_value(iso.onset_to_isolation_delay, state.rng, individual)
    iso_time = onset_time(individual) + iso_delay

    # A contact traced before its onset was known has only the bare trace
    # time, so hold it back to the onset.
    traced_time = max(get(individual.state, :_traced_isolation_time, Inf), onset_time(individual))
    start = min(iso_time, traced_time)
    duration = _removal_duration(
        iso.duration, state.rng, individual, "`Isolation`'s `duration`"
    )
    set_isolated!(individual, start; release_time = start + duration)
    return nothing
end
```

`apply_post_transmission!`: `ContactTracing` goes through the new contacts,
looks up each one's infector, and applies its trace action (`Quarantine` or
`FlagOnly`) when the contact is eligible and is reached. The trace is timed
from [`trigger_time`](@ref EpiBranch.trigger_time), which for a policy based on
isolation is the recorded isolation, so an isolation that does not count as a
detection starts no trace.

```julia
function apply_post_transmission!(ct::ContactTracing, state, new_contacts)
    rng = state.rng
    for ind in new_contacts
        ind.parent_id == 0 && continue
        parent = state.individuals[ind.parent_id]
        is_eligible(ct.eligibility, parent, ind, state) || continue
        traces(ct.trace_rate, parent, ind, state, rng) || continue
        trace_delay = draw_trace_delay(ct.isolation_to_trace_delay, parent, ind, state, rng)
        trace_time = trigger_time(ct.eligibility, parent, ind, state) + trace_delay
        apply_trace!(ct.action, ind, state, trace_time, rng)
    end
    return nothing
end
```

`competing_risk`: see [A custom intervention: closing a border](@ref).

## Checking your intervention

A missing hook raises no error, because every hook does nothing by default.
That makes partial interventions easy to write, but a forgotten or misnamed
method fails silently. Three checks:

- Run a small simulation (`max_cases = 50`) with and without your intervention.
  If the results look the same, your `competing_risk` or
  `apply_post_transmission!` is probably not being called for the people you
  expect.
- Declare the values your intervention needs with `required_fields` (see
  [Requiring values on each person](@ref)), so the simulation stops with an
  error at the start when nothing sets them.
- After a small run, look at `state.individuals[1].state` to confirm your hook
  wrote the values that later steps read.

## Combining with built-in interventions

A custom intervention combines with the built-in ones. Here the border closure
from the [quick start](@ref "A custom intervention: closing a border") is added
to isolation of symptomatic cases, with the incubation period
`LogNormal(1.5, 0.5)` (mean and standard deviation of the log, in days) and
isolation 2 days after onset on average:

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
[Interventions](interventions.md)). A policy the built-ins cannot express is a
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
    100; max_cases = 200, rng = StableRNG(5))
traced_older = simulate(ModelSpec(process; interventions = [iso, older_only], attributes = attrs),
    100; max_cases = 200, rng = StableRNG(5))
(all_symptomatic = n_traced(traced_all), symptomatic_over_65 = n_traced(traced_older))
```

With ages drawn from `Normal(50, 20)`, about a fifth of infectors are 65 or
over, so far fewer contacts are traced. Combined with a built-in policy,
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

# Branching processes: each contact, at its potential exposure time.
function EpiBranch.apply_post_transmission!(av::Antiviral, state, contacts)
    foreach(c -> treat!(av, c), contacts)
    return nothing
end

# Network and household models: each case once its infection time is final.
function EpiBranch.on_infection_settled!(av::Antiviral, ind, state, rng)
    return treat!(av, ind)
end
```

Two methods are needed because the two kinds of model reach a new case at
different steps: branching processes through the batch of new contacts, the
network and household models through each case once its infection time is
final. On those models the simulation still warns that `Antiviral` "will have
no effect", because it checks only for the `apply_post_transmission!` method;
`on_infection_settled!` does the work there, so the warning can be ignored.

## Susceptibility and infectiousness are risks too

A person's own susceptibility and an infector's infectiousness are handled the
same way as an intervention: as risks that can block a transmission. The
simulation puts every transmission to the built-in risks and your
interventions alike. `competing_risk` is the single way to block a
transmission, whether the reason is a vaccine, a border closure or the
person's own susceptibility.

Five built-in risks are always present:

- [`EpiBranch.HostSusceptibility`](@ref) blocks with probability
  `1 - susceptibility` of the contact. A contact with susceptibility 0.3
  escapes 70% of exposures.
- [`EpiBranch.InfectorInfectiousness`](@ref) blocks with probability
  `1 - infectiousness` of the infector.
- [`EpiBranch.InfectiousSource`](@ref) blocks every transmission from someone
  who is not infected, so a contact kept in the simulation without being
  infected (see [Following up contacts who were not infected](@ref)) can
  generate contacts without infecting them. Usually every active person is
  infected and it does nothing.
- [`EpiBranch.AbortedInfection`](@ref) blocks every transmission an infector
  makes from its `:infection_aborted_time`, so an infection ended by
  [`EpiBranch.abort_infection!`](@ref) stays ended after the intervention that
  ended it stops being active.
- [`EpiBranch.HostImmunity`](@ref) blocks every exposure of a contact who has
  already been infected, until [`susceptible_again_time`](@ref) is in the
  past. It does nothing for a model whose `contacts_of` never offers an
  already-infected person as a contact, which is every built-in model. A model
  that does gets a new, separately recorded infection episode (see [Individual
  state and reserved keys](@ref)) instead of a reinfection overwriting the
  earlier one.

A susceptibility or infectiousness of `1.0` blocks nothing, and these risks have
no effect unless a population characteristic sets a value below one. You can
replace or add to them with a `competing_risk` of your own.

`HostSusceptibility` and `InfectorInfectiousness` apply on branching processes.
The continuous-time models apply the same two values as multipliers on the rate
of transmission instead (see [Continuous-time models: further details](@ref)),
so they do not decide them contact by contact. Every other risk, yours
included, applies there as it does on a branching process.

## Following up contacts who were not infected

By default only the people infected in a generation go on to the next one and
generate their own contacts; a contact who was not infected goes no further.
`keep_active` lets an intervention keep other contacts in the simulation:
return the ids of this generation's contacts that should keep generating
contacts, and they are added to the next generation.

This is needed for contact tracing beyond direct contacts. To reach the
contacts of contacts, the simulation has to generate the contacts of an
infected case's contacts even when those contacts were never infected. Keep
them here, and the built-in `InfectiousSource` risk lets them generate contacts
without infecting anyone. `[t.id for t in targets if !is_infected(t)]` collects
the ids of the contacts that were not infected:

```julia
struct KeepUninfectedActive <: AbstractIntervention end

function EpiBranch.keep_active(::KeepUninfectedActive, state, targets, is_new)
    [t.id for t in targets if !is_infected(t)]
end
```

[`ContactTracing`](@ref) with `depth > 1` is the built-in intervention that
uses this: it keeps the uninfected members of a ring in the simulation for as
many steps as the ring's depth, so a second-level ring reaches the contacts of
contacts that ring vaccination then targets.

## Scheduling an intervention

[`Scheduled`](@ref) starts any intervention on a given day:
`Scheduled(iv; start_time = ...)`. Built-in interventions have no start-time
field of their own.

`Scheduled` checks the time at which the intervention would act on each person,
not when they were infected, so someone infected before the start can still be
affected if, say, their isolation would fall after it. When a person's action
time would fall before the start, `Scheduled` undoes it. For this the
intervention defines two methods:

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

## Capacity limits

[`CapacityConstrained`](@ref) limits how many actions an intervention can take
per period, such as the doses available or the number of contacts tracers can
reach. Your intervention defines `EpiBranch.capacity_key(iv)`, the name of the
`true`/`false` value recording that a person used the resource, and
`EpiBranch.capacity_time_key(iv)`, the name of the value recording when. The
time places use by other interventions within a period when
`carry_over = false`; actions admitted by `CapacityConstrained` itself use the
time they were admitted. Ring, group and mass vaccination use their dose-label
names.

Every proposed recipient is known before admission, including group members
found by searching a whole group. See [Intervention actions](@ref) for how to
propose actions and the rules on timing. An older intervention that acts on a
batch of people can still use the capacity names, provided its batch hook only
changes the people it receives.

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

It adds an `apply_post_transmission!` method choosing whom to vaccinate and
when. That method records each dose with `EpiBranch._record_vaccination!(v,
ind, vaccination_time, rng)`, which writes the per-dose values listed in
[Individual state and reserved keys](@ref) and draws `efficacy`,
`severity_efficacy` and `delay_to_immunity` for that person, whether they were
given as numbers, distributions or functions.

!!! note "This recipe uses internal functions"
    `_record_vaccination!`, and `_record_effect_draws!`, `_store_draw!`,
    `_dose_value` and `_vaccine_efficacy` below, start with an underscore: they
    are not part of the public interface and may change in a later release.
    There is no public way to record a dose yet. If you build on them, fix the
    EpiBranch version your project uses.

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

function EpiBranch.apply_post_transmission!(v::OlderAdultVaccination, state, new_contacts)
    for ind in new_contacts
        ind.state[:age] >= v.min_age || continue
        EpiBranch._record_vaccination!(v, ind, v.campaign_time, state.rng)
    end
    return nothing
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
above). Its per-dose draw is recorded by a method of `_record_effect_draws!`,
which `_record_vaccination!` calls for every vaccination. `RingVaccination`
records `post_exposure_efficacy` and `onward_efficacy` that way:

```julia
_booster_uptake_key(label) = Symbol("booster_uptake_", label)

function EpiBranch._record_effect_draws!(v::OlderAdultVaccination, contact, label, rng)
    EpiBranch._store_draw!(v.booster_uptake, _booster_uptake_key, label, contact, rng)
    return nothing
end
```

For a number, `_store_draw!` stores nothing and `EpiBranch._dose_value` reads
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
leaky protection instead of none. The example records eight doses directly
with the internal functions noted above:

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
    EpiBranch._record_vaccination!(partial, contact, 0.0, StableRNG(i))
    EpiBranch._vaccine_efficacy(partial, contact)
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

Capacity counts decisions to admit actions. In this example the index case is
processed at time zero, so the dose for day 10 uses the budget available at
time zero, and `period = 7.0` would not move that charge into the second week.
The default `period = Inf` gives one budget for the whole simulation.

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

function EpiBranch.apply_post_transmission!(visit::ClinicAppointment, state, candidates)
    EpiBranch.apply_actions!(visit, state, candidates)
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
intervention: `initialise_individual!` to set defaults on each new person, and
`resolve_individual!` to work out what happens to the case and when.

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
    occurred = ind.state[:treated]
    ll = EpiBranch.transition_term(t.probability, t.delay, ind, ot, occurred)
    occurred || return ll
    return ll + logpdf(t.delay, ind.state[:treatment_time] - ot)
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
