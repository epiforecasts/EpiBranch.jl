# Vaccination

Can vaccinating the contacts of cases, or the whole population, contain an
outbreak that isolation and contact tracing alone do not? This page covers
ring vaccination of traced contacts, post-exposure prophylaxis, mass and group
vaccination campaigns, multi-dose schedules and limited vaccine supply.

It uses the outbreak model from
[Isolation and contact tracing](interventions.md): a Poisson offspring
distribution with mean 3 (R = 3), an exponential generation time with mean 5
days, and an incubation period of `LogNormal(1.5, 0.5)` (mean and standard
deviation of the log, median about 4.5 days). Infectiousness starts at
infection. All times are in days, counted from the infection of the first
index case. Each scenario simulates a number of outbreaks, stopped at
`max_cases` cases, and [`containment_probability`](@ref) is the proportion
that died out before reaching that cap (see the [glossary](../glossary.md)).

```@example vaccination
using EpiBranch
using Distributions
using StableRNGs

clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

scenario(interventions = AbstractIntervention[], attributes = clinical) =
    ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0)); interventions, attributes)

iso = Isolation(onset_to_isolation_delay = Exponential(2.0), duration = Inf)
ct = ContactTracing(
    probability = 0.7, isolation_to_trace_delay = Exponential(1.0),
    action = Quarantine(duration = Inf)
)
nothing # hide
```

`iso` isolates symptomatic cases a mean of 2 days after onset, for good. `ct`
traces 70% of their contacts a mean of 1 day after the infector's isolation
and quarantines them, also for good.

## How a dose protects

A dose protects from `delay_to_immunity` days after it is given (0 by
default). It has up to three effects, each a probability:

| Parameter | What it prevents | Acts if protection starts |
|---|---|---|
| `efficacy` | the vaccinated person being infected | before the exposure |
| `onward_efficacy` | the vaccinated person infecting others | before each of their transmissions |
| `post_exposure_efficacy` | an infection the person already has | before symptom onset (if before the exposure, it blocks the infection itself) |

`efficacy` is leaky by default: each exposure after protection starts is
blocked with that probability. `mode = AllOrNothingMode()` instead makes that
proportion of vaccinated people fully protected and leaves the rest
unprotected. In a branching process every exposure is a different person, and
the two give the same results on average. They differ in models where a
person can be exposed more than once, such as households and networks.

!!! note "Compare many replicates"
    Every scenario uses its own random draws. Two scenarios run once each
    can differ by chance, and adding a dose can even seem to lower
    containment. Compare scenarios over many simulated outbreaks, and check
    that a difference is larger than the Monte Carlo error: for a
    containment probability `p` estimated from `n` outbreaks the standard
    error is about `sqrt(p * (1 - p) / n)`.

## Ring vaccination

[`RingVaccination`](@ref) vaccinates traced contacts and needs
[`ContactTracing`](@ref) in the same scenario:

```@example vaccination
rv = RingVaccination(efficacy = 0.8)

rng = StableRNG(42)
results = simulate(scenario([iso, ct]), 200; max_cases = 500, rng = rng)
println("Isolation + tracing: $(round(containment_probability(results), digits=3))")

rng = StableRNG(42)
results = simulate(scenario([iso, ct, rv]), 200; max_cases = 500, rng = rng)
println("Isolation + tracing + ring vaccination: $(round(containment_probability(results), digits=3))")
```

!!! warning "Under this tracing, vaccination makes no difference"
    The two numbers are identical, by construction. Tracing starts after the
    infector is isolated, and isolation already stops any further exposure
    of their contacts. A dose that protects against future exposure
    (`efficacy`) has nothing left to prevent, with or without quarantine.

    `efficacy` matters only where a contact can still be exposed after being
    traced: under leaky isolation (`post_isolation_transmission > 0`), with a
    finite isolation `duration` that releases the infector while the contact
    is still susceptible, when tracing starts before the infector is isolated
    (for example `eligibility = OnSymptomOnset()`), or in a ring wider than
    direct contacts (`depth > 1`) that passes through people who keep
    transmitting after they are traced. To protect contacts who have already
    been exposed, use `post_exposure_efficacy` or `onward_efficacy` (see
    [Protecting a contact who has already been exposed](#Protecting-a-contact-who-has-already-been-exposed)).

Tracing that starts at the infector's symptom onset, before isolation, and
does not quarantine, leaves contacts exposed after the trace, and there the
dose acts:

```@example vaccination
ct_onset = ContactTracing(probability = 0.7,
    isolation_to_trace_delay = Exponential(1.0),
    eligibility = OnSymptomOnset(), action = FlagOnly())

for (label, interventions) in (
        ("no vaccination", [iso, ct_onset]),
        ("ring vaccination", [iso, ct_onset, rv]))
    runs = simulate(scenario(interventions), 1000; max_cases = 500, rng = StableRNG(42))
    println(rpad(label, 20), round(containment_probability(runs), digits = 3))
end
```

### Delay to immunity

A vaccine that takes time to protect can be too late. Here protection starts
7 days after the dose:

```@example vaccination
rv_delayed = RingVaccination(efficacy = 0.8, delay_to_immunity = 7.0)

rng = StableRNG(42)
results = simulate(scenario([iso, ct_onset, rv_delayed]), 1000; max_cases = 500, rng = rng)
println("With a 7-day delay to immunity: $(round(containment_probability(results), digits=3))")
```

Compare this with the same-day protection above. Contacts infected after the
trace are infected within days of it, by infectors who have not yet been
isolated, so a dose that protects a week later comes too late for most of
them.

### Counting doses

`condition = 50:200` keeps simulating until an outbreak stops with between 50
and 200 cases and returns it. It stops either because it died out or because
it reached the 200-case cap, as the one here did.
`count(is_vaccinated, state.individuals)` then counts everyone vaccinated,
like `sum(is_vaccinated(x))` in R:

```@example vaccination
rng = StableRNG(42)
state = simulate(scenario([iso, ct, rv]); condition = 50:200, max_cases = 200, rng = rng)
n_vaccinated = count(is_vaccinated, state.individuals)
n_infected = count(is_infected, state.individuals)
println("Vaccinated: $n_vaccinated, Infected: $n_infected")
```

The ratio of the two is the number of doses used per case in this outbreak.

### Rings beyond direct contacts

By default tracing reaches a case's direct contacts. `depth = 2` also traces
the contacts of those contacts, the second ring that Ebola ring vaccination
protocols vaccinate around a confirmed case. An infected contact who meets the
tracing trigger starts a ring of their own, and contacts who were not infected
are followed one step further so the ring can extend past them. The same
`RingVaccination` vaccinates everyone in the ring:

```@example vaccination
ct2 = ContactTracing(
    probability = 0.7, isolation_to_trace_delay = Exponential(1.0),
    action = Quarantine(duration = Inf), depth = 2
)

rng = StableRNG(42)
state = simulate(scenario([iso, ct2, rv]); condition = 50:200, max_cases = 200, rng = rng)
doses_depth2 = count(is_vaccinated, state.individuals)
println("Doses with a second ring: $doses_depth2")
```

A wider ring reaches more people and uses more doses, but the extra doses
protect only people still exposed after they are traced. Here every infected
ring member with symptoms is isolated and starts a ring of their own. Their
contacts are traced after that isolation, when the second ring's doses have
nothing left to protect against. The second ring prevents infections when it passes
through people who keep transmitting after they are traced, such as
asymptomatic contacts under tracing without quarantine. Compare dose counts
and containment across depths for your own setting.

## Protecting a contact who has already been exposed

By the time a contact is traced under the default tracing, their infector is
already isolated. The contact faces no new exposure from that infector, and
whether they were infected before is already decided. `efficacy` then has
nothing to block, but the dose can still act on an infection the contact
already has:

- `post_exposure_efficacy` stops the infection, with that probability, if
  protection starts before the contact's symptom onset. The contact transmits
  as usual until then and not at all afterwards. They never develop
  symptoms, and nothing else in their clinical course (hospitalisation,
  death, recovery) happens after that point. They still count as a case, and
  appear in the line list with a `date_infection_aborted` and no onset date.
- `onward_efficacy` leaves the infection and its disease alone, and blocks
  each of the contact's transmissions after protection starts with that
  probability, whenever their onset falls.

A dose can set both: it then stops some infections and makes the rest less
infectious. Both act only on transmission after protection starts. Neither does
much for a contact who has already infected most of their own contacts.
Quarantine from the trace already stops that later transmission, and the
examples here trace without quarantine. The clearest measure of the effect is
the mean number of secondary cases per traced case, an effective reproduction
number among traced cases. The function below computes it over cases whose
own contacts were simulated before the run stopped:

```@example vaccination
ct_noquarantine = ContactTracing(probability = 0.7,
    isolation_to_trace_delay = Exponential(1.0), action = FlagOnly())

function secondary_per_traced(runs)
    mean(count(id -> is_infected(s.individuals[id]), ind.secondary_case_ids)
    for s in runs for ind in s.individuals
    if is_infected(ind) && is_traced(ind) && ind.generation < s.current_generation)
end

for (label, vaccination) in [
    ("no vaccine", nothing),
    ("efficacy = 0.9", RingVaccination(efficacy = 0.9)),
    ("onward_efficacy = 0.9",
        RingVaccination(efficacy = 0.0, onward_efficacy = 0.9)),
    ("post_exposure_efficacy = 0.9",
        RingVaccination(efficacy = 0.0, post_exposure_efficacy = 0.9)),
]
    interventions = vaccination === nothing ? [iso, ct_noquarantine] :
        [iso, ct_noquarantine, vaccination]
    runs = simulate(scenario(interventions), 400; max_cases = 500, rng = StableRNG(42))
    println(rpad(label, 30), "secondary cases per traced case ",
        round(secondary_per_traced(runs), digits = 2),
        ", containment ", round(containment_probability(runs), digits = 3))
end
```

`onward_efficacy` and `post_exposure_efficacy` both reduce the secondary cases
per traced case, while `efficacy` changes nothing. Containment moves much
less, because a traced contact has usually made a good share of their
transmissions before the trace; at 400 outbreaks per scenario the
differences in containment are close to the Monte Carlo error. A single run
can place the scenarios in either order.

### Post-exposure prophylaxis

There is no separate post-exposure prophylaxis (PEP) intervention: antivirals
or antibiotics given to traced contacts are a `RingVaccination` dose with
`delay_to_immunity = 0` (the default) and a `post_exposure_efficacy`. How much
it achieves depends on speed, because protection has to start before symptom
onset, and the incubation period here averages about five days:

```@example vaccination
for days in [0.0, 2.0, 5.0, 21.0]
    let pep = RingVaccination(efficacy = 0.0, post_exposure_efficacy = 0.9,
            delay_to_immunity = days),
        rng = StableRNG(42)
        results = simulate(scenario([iso, ct_noquarantine, pep]), 400;
            max_cases = 500, rng = rng)
        cases = [ind for s in results for ind in s.individuals
                 if is_infected(ind) && is_traced(ind)]
        stopped = count(ind -> isfinite(EpiBranch.infection_aborted_time(ind)), cases)
        println("Protection after $(lpad(Int(days), 2)) days: ",
            round(Int, 100 * stopped / length(cases)), "% of traced cases stopped, ",
            "secondary cases per traced case ",
            round(secondary_per_traced(results), digits = 2))
    end
end
```

The share of infections stopped, and the reduction in secondary cases, fall
quickly as the delay grows; with protection three weeks after the trace
nothing is left.

!!! warning "Do not set `efficacy` and `post_exposure_efficacy` together"
    Where protection is already in place at the exposure,
    `post_exposure_efficacy` blocks the infection itself. It covers every
    contact that `efficacy` would protect, and setting both counts that
    protection twice. For an asymptomatic contact, who has no onset,
    blocking at exposure is all it can do. `post_exposure_efficacy` needs an
    incubation period, set by [`clinical_presentation`](@ref).

!!! warning "Stopped infections are not traced"
    Under quarantine neither `post_exposure_efficacy` nor `onward_efficacy`
    has transmission left to block, and `post_exposure_efficacy` can even
    lower containment. A contact whose infection is stopped never develops
    symptoms, so they are never isolated and the people they infected before
    their dose are not traced from them. A programme that relies on symptom
    onset to trigger tracing loses that trigger for treated contacts.

## Clustered vaccine refusal

If each traced contact decides independently whether to accept a dose, every
community ends up with about the same coverage. In practice refusal clusters
by household or community: the people who avoid tracing also tend to decline
vaccination.

[`vaccine_acceptance`](@ref) gives each community its own acceptance
probability, drawn from a distribution, and everyone in the community shares
it. Communities come from [`groups`](@ref): list `groups(...)` before
`vaccine_acceptance(...)` in the population characteristics. `coverage` on
the dose is the probability a traced contact accepts it; the function
`(rng, ind) -> ind.state[:vaccine_acceptance]` reads each person's community
value:

```@example vaccination
community = groups(20)  # 20 communities
acceptance = vaccine_acceptance(propensity = Beta(2, 2))  # mean 0.5, varies by community

rv_clustered = RingVaccination(efficacy = 0.8,
    coverage = (rng, ind) -> ind.state[:vaccine_acceptance])
rv_independent = RingVaccination(efficacy = 0.8, coverage = 0.5)
nothing # hide
```

The function below takes each community with at least 5 traced members, in
each simulated outbreak, and records the share of them vaccinated:

```@example vaccination
function coverage_by_community(states)
    out = Float64[]
    for state in states, g in 1:20
        members = filter(state.individuals) do ind
            is_traced(ind) && get(ind.state, :group, 0) == g
        end
        length(members) >= 5 || continue
        push!(out, count(is_vaccinated, members) / length(members))
    end
    return out
end

rng = StableRNG(42)
clustered = coverage_by_community(simulate(
    scenario([iso, ct, rv_clustered], [clinical, community, acceptance]), 50;
    max_cases = 300, rng = rng))

rng = StableRNG(42)
independent = coverage_by_community(simulate(
    scenario([iso, ct, rv_independent], [clinical, community]), 50;
    max_cases = 300, rng = rng))

share_extreme(x) = count(c -> c < 0.2 || c > 0.8, x) / length(x)
println("Mean community coverage: clustered $(round(mean(clustered), digits=3)), independent $(round(mean(independent), digits=3))")
println("Communities under 20% or over 80% covered: clustered $(round(share_extreme(clustered), digits=3)), independent $(round(share_extreme(independent), digits=3))")
```

Average coverage is about the same either way. Clustering spreads it out,
leaving some communities almost fully covered and others almost untouched.

If acceptance is exactly 0 or 1 in each community, as with
`(rng, ind) -> Float64(rand(rng, Bernoulli(0.5)))`, each community accepts or
refuses as a whole. With a `Beta` distribution, as here, communities differ
but people within one still decide individually.

A community's acceptance lasts the whole outbreak, so a community that
refuses keeps refusing in later generations. Give [`GroupVaccination`](@ref)
the same `coverage` function to cluster refusal within the groups it
vaccinates, or use the value in [`MassVaccination`](@ref)'s `eligibility_time`
in the same way.

### Clustered refusal and containment

Under `ct`, ring vaccination cannot change whether an outbreak is contained
(see the warning in [Ring vaccination](@ref)). Measuring the effect of
clustering on containment needs a scenario where vaccination acts. Here tracing does not
quarantine, and `onward_efficacy` reduces a vaccinated contact's own onward
transmission. Every scenario draws the same population characteristics, leaving
vaccination as the only difference between them:

```@example vaccination
rv_clustered_onward = RingVaccination(efficacy = 0.8, onward_efficacy = 0.8,
    coverage = (rng, ind) -> ind.state[:vaccine_acceptance])
rv_independent_onward = RingVaccination(efficacy = 0.8, onward_efficacy = 0.8,
    coverage = 0.5)

n_outbreaks = 10000
containment(interventions) = containment_probability(simulate(
    scenario(interventions, [clinical, community, acceptance]), n_outbreaks;
    max_cases = 200, rng = StableRNG(42)))

for (label, interventions) in (
        ("no vaccination", [iso, ct_noquarantine]),
        ("clustered refusal", [iso, ct_noquarantine, rv_clustered_onward]),
        ("independent refusal", [iso, ct_noquarantine, rv_independent_onward]))
    p = containment(interventions)
    println(rpad(label, 20), round(p, digits = 3),
        " (standard error ", round(sqrt(p * (1 - p) / n_outbreaks), digits = 3), ")")
end
```

Vaccinating about half the traced contacts raises containment a little. The
gap between clustered and independent refusal is within the Monte Carlo error
(the standard error of a difference between two scenarios is about 1.4 times
that of each). These runs cannot tell the two apart.

Clustering has little effect here because `groups(20)` puts each person in a
community independently of who infected them. A case's contacts are spread
across communities. A community's acceptance rarely lines up with the people
that case goes on to infect. Clustering matters more when communities
follow transmission, as households or a contact network do: a community that
refuses then keeps transmitting within itself, which can lower containment at
the same average coverage. Measure the effect in the model you are using.

## Mass vaccination

[`MassVaccination`](@ref) offers vaccine to everyone in the model regardless
of tracing. `eligibility_time` is the day a person can be vaccinated, and
protection starts `delay_to_immunity` days later; a person is protected
against an exposure only if both have passed by then. Here everyone becomes
eligible on day 30 and protection starts 14 days later:

```@example vaccination
mv = MassVaccination(efficacy = 0.85, eligibility_time = 30.0,
    delay_to_immunity = 14.0)

rng = StableRNG(42)
results = simulate(scenario([mv]), 200; max_cases = 500, rng = rng)
println("Mass vaccination from day 30: $(round(containment_probability(results), digits=3))")
```

With R = 3, an outbreak that takes off reaches the case cap well before
protection starts on day 44. Containment stays close to the baseline without
interventions.

For a gradual rollout in which each person becomes eligible at a random
time, pass a distribution. `Exponential(60.0)` makes the mean wait 60 days:

```@example vaccination
mv_random = MassVaccination(efficacy = 0.85,
    eligibility_time = Exponential(60.0),
    delay_to_immunity = 14.0)
```

For a rollout by age, pass a function of the individual. This needs ages: add
[`demographics`](@ref) to the population characteristics. The rule
`ind.state[:age] >= 65 ? 30.0 : 90.0` reads "day 30 if aged 65 or over,
otherwise day 90":

```@example vaccination
mv_age = MassVaccination(
    efficacy = 0.85,
    eligibility_time = (rng, ind) -> ind.state[:age] >= 65 ? 30.0 : 90.0,
    delay_to_immunity = 14.0,
)

attrs = [clinical, demographics(age_distribution = Uniform(0, 90))]
rng = StableRNG(42)
results = simulate(scenario([mv_age], attrs), 200; max_cases = 500, rng = rng)
println("Age-stratified rollout: $(round(containment_probability(results), digits=3))")
```

### Efficacy that varies between people

`efficacy` can be one number for everyone, a distribution (people differ at
random, with one draw per person, as for a varied immune response), or a
function of the individual such as one that depends on age. Here older people
respond less well:

```@example vaccination
mv_heterogeneous = MassVaccination(
    efficacy = (rng, ind) -> ind.state[:age] >= 65 ? 0.7 : 0.9,
    eligibility_time = (rng, ind) -> ind.state[:age] >= 65 ? 30.0 : 90.0,
    delay_to_immunity = 14.0,
)
```

## Multi-dose schedules

For prime and boost (or longer) schedules, list one `MassVaccination` or
`RingVaccination` per dose, each with its own `dose_label`. A dose protects
against an exposure only if its protection has started by then. `:prime` and
`:boost` are labels; the leading colon makes them Julia symbols.

```@example vaccination
prime = MassVaccination(efficacy = 0.6, eligibility_time = 30.0,
    delay_to_immunity = 14.0, dose_label = :prime)
boost = MassVaccination(efficacy = 0.5, eligibility_time = 60.0,
    delay_to_immunity = 14.0, dose_label = :boost)

rng = StableRNG(42)
results = simulate(scenario([prime, boost]), 200; max_cases = 500, rng = rng)
println("Two-dose rollout: $(round(containment_probability(results), digits=3))")
```

!!! warning "Enter the efficacy each dose adds"
    Doses protect independently: an exposure is blocked unless every dose
    whose protection has started fails to block it. With efficacies `e1` and
    `e2`, overall protection is `1 - (1 - e1) * (1 - e2)`. The boost's
    `efficacy` is therefore the protection it adds among people the prime
    left unprotected. For 60% protection after one dose and 80% after two,
    set the prime to 0.6 and the boost to `(0.8 - 0.6) / (1 - 0.6) = 0.5`,
    as above. Setting the boost to 0.8 would give 92% overall.

`is_vaccinated(ind; dose_label = :boost)` tells whether a person had a given
dose, and [`linelist`](@ref) has one date column per dose, such as
`date_vaccination_prime` and `date_immunity_boost`.

### Two doses in a ring

Ring doses are given at the trace. A second dose sets `dose_delay`, the days
from the trace to that dose, and names the dose it follows with
`requires_dose`. Only contacts who have had the earlier dose by the time the
later one is due receive it. The boost's `coverage` is then the share
retained between doses. List each dose after the dose it requires; a boost listed
first finds no one primed and is never given.

```@example vaccination
prime_ring = RingVaccination(efficacy = 0.6, delay_to_immunity = 21.0,
    coverage = 0.8, dose_label = :prime)
boost_ring = RingVaccination(efficacy = 0.5, dose_delay = 28.0,
    delay_to_immunity = 14.0, coverage = 0.9,
    requires_dose = :prime, dose_label = :boost)

rng = StableRNG(42)
results = simulate(scenario([iso, ct, prime_ring, boost_ring]), 200;
    max_cases = 500, rng = rng)
println("Two-dose ring: $(round(containment_probability(results), digits=3))")
```

A protocol that gives the boost four to six weeks after the trace uses a
distribution, `dose_delay = Uniform(28.0, 42.0)`, drawn once per contact.
`delay_to_immunity`, `post_exposure_efficacy` and `onward_efficacy` also take
a distribution or a function of the individual, each drawn once per contact
when the dose is given. A contact keeps the same protection against every
exposure.

The mean number of each dose per simulated outbreak shows what the second
dose costs:

```@example vaccination
doses(label) = mean(count(i -> is_vaccinated(i; dose_label = label), s.individuals)
    for s in results)
println("Primed: $(round(doses(:prime), digits = 1)), ",
    "boosted: $(round(doses(:boost), digits = 1))")
```

A later dose is recorded when it falls due, even if the contact was infected
in the meantime. The boost count is a count of doses scheduled.

Under this tracing neither dose changes the outbreak: no ring member is
infected after being traced (see the warning in
[Ring vaccination](@ref)). Even under `ct_onset`, where infections do
follow the trace, they follow within days. Both doses here start protecting
weeks after the trace, too late for those infections (see
[Delay to immunity](@ref)). Removing the boost and re-running is not a
controlled comparison either. The two runs use different random draws, and
their containment can differ by chance in either direction (see the note on
comparing many replicates above).

## Repeat campaign visits

[`GroupVaccination`](@ref) vaccinates every member of a group, such as a
household or community, once a case in it is detected (lab-confirmed by
default), `dose_delay` days later. A campaign that visits each group several
times, and reaches a person missed on one visit on a later one, can be set up
with its `coverage` and `dose_delay`. People who refuse outright are a
separate population characteristic.

With three visits, each reaching a willing person with probability `c`, the
probabilities that the first, second and third visit is the first to reach
them are `c`, `(1-c)*c` and `(1-c)^2*c`. Their sum is the probability of
ever being reached, and the coverage among willing people. Rescaled to sum to
one, they give the distribution of the delay from the group's trigger to the
visit that reaches them:

```@example vaccination
visit_times = [0.0, 7.0, 14.0]
reach_per_visit = 0.6
first_reached = [reach_per_visit * (1 - reach_per_visit)^(i - 1)
                 for i in eachindex(visit_times)]
campaign_reach = sum(first_reached)
visit_delay = DiscreteNonParametric(visit_times, first_reached ./ campaign_reach)

# Each person's willingness is drawn once, when they are created, and stored
# as :willing; it is separate from whether a visit reaches them.
prob_willing = 0.9
willingness = (rng, ind) -> (ind.state[:willing] = rand(rng) < prob_willing)
repeated_campaign = GroupVaccination(efficacy = 0.8,
    coverage = (rng, ind) -> ind.state[:willing] ? campaign_reach : 0.0,
    dose_delay = visit_delay)

campaign_model = ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
    attributes = [clinical, groups(3), willingness],
    interventions = [Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf),
        repeated_campaign])

println("Reached among willing people: $(round(campaign_reach, digits = 3)); ",
    "expected overall coverage: $(round(prob_willing * campaign_reach, digits = 3))")
```

Overall coverage is the share willing times the share of willing people
reached. A person with `:willing == false` refuses throughout the campaign,
and a willing person left unvaccinated was never reached. For a reach
probability of zero, use `coverage = 0.0` directly. A single visit gives the
usual campaign with one opportunity. Use a different `dose_label` for another
dose; willingness can be shared across doses, as here, or stored separately
for each.

!!! note "Limitations of this campaign recipe"
    Each person's eventual dose and its date are decided once, and protection
    starts the usual `delay_to_immunity` after that date. Visits are not
    simulated one by one. The campaign cannot respond to limited team
    capacity or to how the outbreak develops, and failed visits are not
    recorded. Visits are assumed to reach people independently, with a fixed
    probability.

## Limited vaccine supply

The scenarios above vaccinate everyone they reach. Real capacity is finite:
a fixed number of teams, a daily dose limit, a stockpile.
[`CapacityConstrained`](@ref) limits how many people a ring, group or mass
vaccination reaches: at most `budget_per_period` people every `period` days.
People beyond the limit are not vaccinated, and by default those traced first
are served first:

```@example vaccination
rv_capped = CapacityConstrained(rv; budget_per_period = 5.0, period = 1.0)

state_uncapped = simulate(scenario([iso, ct, rv]); condition = 50:200, max_cases = 200, rng = StableRNG(42))
state_capped = simulate(scenario([iso, ct, rv_capped]); condition = 50:200, max_cases = 200, rng = StableRNG(42))

println("Doses without a limit: $(count(is_vaccinated, state_uncapped.individuals))")
println("Doses with at most 5 people admitted a day: $(count(is_vaccinated, state_capped.individuals))")
```

!!! warning "The limit is on people admitted each day"
    The limit applies when people are selected for a dose, which happens
    when contacts are traced; the doses themselves are dated later, at the
    trace time plus `dose_delay`. Doses can therefore bunch up on other days
    than the one whose allowance paid for them. One day can see several
    times `budget_per_period` doses given: the limit does not cap the
    doses given per day.

[`capacity_usage`](@ref) reports how much of the allowance a simulated
outbreak had used, and how much was available, by the time it stopped:

```@example vaccination
capacity_usage(rv_capped, state_capped)
```

`priority` decides who is served when more people are waiting than the
allowance allows: it is a function of the individual and the simulation
state, and people with lower values are served first. The default orders by
trace time. A lottery among the people waiting:

```@example vaccination
rv_lottery = CapacityConstrained(rv; budget_per_period = 5.0, period = 1.0,
    priority = (ind, state) -> rand(state.rng))
nothing # hide
```

`period = Inf` (the default) makes the budget a single stockpile for the whole
outbreak:

```@example vaccination
rv_stockpile = CapacityConstrained(rv; budget_per_period = 200.0)
nothing # hide
```

With a finite `period`, `carry_over = true` (the default) adds a day's unused
allowance to the next day's; `carry_over = false` discards it, so only that
day's own `budget_per_period` is ever available.

!!! note "Which models support capacity limits"
    On the network and household models, capacity limits work for ring and
    group vaccination but not for mass vaccination. To limit an intervention
    of your own, see [Extending EpiBranch](extending.md).
