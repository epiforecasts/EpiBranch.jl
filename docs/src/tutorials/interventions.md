# Isolation and contact tracing

How much do isolating cases and tracing their contacts reduce the chance that
an outbreak takes off? This page builds up from no interventions to isolation,
contact tracing and interventions that start part-way through an outbreak, and
compares the probability that each scenario is contained. Vaccination is on
the next page, [Vaccination](vaccination.md).

Each case infects its secondary cases at times drawn from the generation time
distribution, counted from the case's own infection: infectiousness starts at
infection, with no separate latent period. Isolation stops transmission from
the moment a case is isolated. A secondary case is infected only if their
infection would have happened before their infector was isolated: whichever
happens first decides. Isolation follows symptom onset, and how much it
prevents depends on how the incubation period compares with the generation
time.

All times on this page are in days. Everything here needs only keyword
arguments; if Julia syntax such as `;` in a function call or `x -> ...` is new,
see [Julia for R users](../julia-for-r-users.md).

## Without interventions

The outbreak model throughout has a Poisson offspring distribution with mean
3 (R = 3, well above the threshold of 1, with no superspreading) and an
exponential generation time with mean 5 days. Isolation needs to know when
each case develops symptoms, and that takes an incubation period.
[`clinical_presentation`](@ref) adds one. `LogNormal(1.5, 0.5)` takes the
mean and standard deviation of the log and has a median of about 4.5 days. These values illustrate the method and do not describe a particular
pathogen.

A scenario is the outbreak model plus its interventions. The helper
`scenario` below builds one: `scenario([iso])` is the model with isolation,
and `scenario()` is the model with none. `AbstractIntervention[]` is an empty
list of interventions. Copy the helper as it is.

```@example interventions
using EpiBranch
using Distributions
using StableRNGs

clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

scenario(interventions = AbstractIntervention[], attributes = clinical) =
    ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0)); interventions, attributes)

rng = StableRNG(42)
results_baseline = simulate(scenario(), 200; max_cases = 500, rng = rng)
println("Containment (no interventions): $(round(containment_probability(results_baseline), digits=3))")
```

This simulates 200 outbreaks, each starting from one index case and stopped
once it reaches `max_cases = 500` cases. `StableRNG(42)` sets the random seed,
and the page gives the same numbers every time it is built.
[`containment_probability`](@ref) is the proportion of simulated outbreaks
that died out before reaching that cap (see the
[glossary](../glossary.md)). With R = 3 most outbreaks are not contained.

## Isolation

[`Isolation`](@ref) isolates symptomatic cases after a delay from symptom
onset. `onset_to_isolation_delay = Exponential(2.0)` is a delay with mean 2
days, and `duration = 7.0` releases each case after 7 days:

```@example interventions
iso = Isolation(onset_to_isolation_delay = Exponential(2.0), duration = 7.0)

rng = StableRNG(42)
results = simulate(scenario([iso]), 200; max_cases = 500, rng = rng)
println("Containment (isolation): $(round(containment_probability(results), digits=3))")
```

The sooner a case is isolated after symptom onset, the fewer of their
secondary cases are infected. Varying the mean delay:

```@example interventions
for d in [0.5, 2.0, 10.0]
    let iso = Isolation(onset_to_isolation_delay = Exponential(d), duration = 7.0),
        rng = StableRNG(42)
        results = simulate(scenario([iso]), 200; max_cases = 500, rng = rng)
        println("Mean delay $d days: containment = $(round(containment_probability(results), digits=3))")
    end
end
```

Containment falls as the delay grows. With a mean delay of 10 days most
cases have done their transmitting before they are isolated.

### Leaky isolation

Isolation is rarely perfect: household members, for example, are still
exposed. `post_isolation_transmission = 0.3` makes an isolated case 30% as
infectious as before. Imperfect isolation like this is called leaky:

```@example interventions
iso_leaky = Isolation(onset_to_isolation_delay = Exponential(2.0), post_isolation_transmission = 0.3, duration = 7.0)

rng = StableRNG(42)
results = simulate(scenario([iso_leaky]), 200; max_cases = 500, rng = rng)
println("Leaky isolation: $(round(containment_probability(results), digits=3))")
```

### Isolation duration

`duration` sets how long a case stays isolated. It has no default, because
the length of isolation is a modelling choice. It takes a number of days,
a distribution, or a function of the random number generator and the
individual, `(rng, ind) -> ...`, in the same way as
`onset_to_isolation_delay`. `Inf` keeps a case isolated for good. A finite
value releases them. A released case who is still infectious transmits again:
a secondary case whose infection would fall after the release is not
prevented. A duration of 0 stops no transmission at all.

A finite duration means "isolated for the rest of the infectious period"
only when infectiousness ends before the duration runs out. The
generation time here has no upper limit, so cases released after the 7 days
above can still infect people. Isolation follows onset and always starts after
infection. Removing a contact before they are infected is quarantine, below.

With a finite duration, the line list from [`linelist`](@ref) gains a
`date_isolation_release` column, the date each isolated case was released.

!!! warning "Release ends leaky isolation too"
    A released leaky case goes back to full infectiousness. A finite
    duration can change a leaky model's final outbreak size by an order of
    magnitude.

!!! note "Release on the network, household and homogeneous models"
    The network, household and homogeneous-mixing models (see
    [Networks](networks.md), [Households](households.md) and
    [Homogeneous models](homogeneous.md)) release isolated cases in the same way. The
    same holds for an isolation switched on or off with [`Scheduled`](@ref),
    below.
    A case isolated for a week part-way through a month of infectiousness
    transmits again for the rest of that month.

## Contact tracing

[`ContactTracing`](@ref) finds the contacts of a case. Each contact is found
with `probability` 0.7, and `isolation_to_trace_delay` (mean 1 day) is the
time from the infector's isolation until the contact is traced.

Isolation applies to cases; quarantine applies to traced contacts who are not
yet known to be cases. The `action` keyword, which has no default, says what
happens to a traced contact:

- `action = Quarantine(duration = Inf)` quarantines the contact at the trace.
  They stop transmitting from then on, even before any symptoms.
- `action = FlagOnly()` only flags the contact as traced. They keep
  transmitting. If they develop symptoms, they are isolated at onset or at
  the trace, whichever is later, unless the usual onset-to-isolation delay
  would isolate them sooner. Asymptomatic contacts are never isolated.

Until the section on scheduled interventions, isolation and quarantine last
indefinitely (`duration = Inf`), to keep the comparisons about tracing.

```@example interventions
iso = Isolation(onset_to_isolation_delay = Exponential(2.0), duration = Inf)
ct = ContactTracing(
    probability = 0.7, isolation_to_trace_delay = Exponential(1.0),
    action = Quarantine(duration = Inf)
)

rng = StableRNG(42)
results = simulate(scenario([iso]), 200; max_cases = 500, rng = rng)
println("Isolation alone: $(round(containment_probability(results), digits=3))")

rng = StableRNG(42)
results = simulate(scenario([iso, ct]), 200; max_cases = 500, rng = rng)
println("Isolation + tracing: $(round(containment_probability(results), digits=3))")
```

Quarantine stops transmission from secondary cases before they show
symptoms, and containment rises above isolation alone with the same
indefinite isolation.

[`Quarantine`](@ref)'s `duration` takes the same values as isolation's, and
has no default either. With `Inf`, a traced contact who escaped infection by
the traced case but is infected later through another route stays blocked
for good. A finite duration releases the contact. Once released, an infected
contact transmits like anyone else. Here each quarantine lasts on
average 5 days:

```julia
ContactTracing(
    probability = 0.7, isolation_to_trace_delay = Exponential(1.0),
    action = Quarantine(duration = Exponential(5.0))
)
```

!!! note "`quarantine_on_trace` is deprecated"
    Older code may use `quarantine_on_trace = true` or `false`. These still
    run with a warning: `true` means `action = Quarantine(duration = Inf)` and
    `false` means `action = FlagOnly()`. Write the `action` instead.

### What triggers contact tracing

By default, a contact is traced once their infector has developed symptoms
and been isolated, and the delay counts from the isolation. Real
programmes start tracing on different events. The `eligibility` keyword sets
the trigger:

| Trigger | Traces once the infector… | Delay counts from |
|---|---|---|
| [`SymptomaticParent`](@ref) (default) | has developed symptoms and been isolated | isolation |
| [`OnSymptomOnset`](@ref) | develops symptoms | symptom onset |
| [`OnLabConfirmation`](@ref) | has tested positive | isolation |
| [`OnIsolation`](@ref) | has been isolated | isolation |
| [`TraceEveryone`](@ref) / [`TraceNobody`](@ref) | has been isolated / never | isolation |

Every trigger except `OnSymptomOnset` waits for the infector's isolation, so
even under `TraceEveryone` the contacts of an infector who is never isolated
(asymptomatic and never quarantined, or symptomatic but missed by the test
and never traced) are never quarantined or isolated through tracing.
[`is_traced`](@ref) still marks them as traced.

Under `OnSymptomOnset` the delay keeps its keyword name,
`isolation_to_trace_delay`, but counts from onset:

```@example interventions
# Begin tracing as soon as the infector shows symptoms, without waiting
# for a positive test or for isolation.
ct_fast = ContactTracing(probability = 0.7, isolation_to_trace_delay = Exponential(1.0),
    action = Quarantine(duration = Inf), eligibility = OnSymptomOnset())
nothing # hide
```

Triggers combine with and (`&`), or (`|`) and not (`!`):

```@example interventions
# Trace suspected or lab-confirmed cases.
elig = OnSymptomOnset() | OnLabConfirmation()

# Trace symptomatic infectors who are never isolated (for example, those
# who test negative), from their onset.
elig_gap = OnSymptomOnset() & !OnIsolation()

ct_combined = ContactTracing(probability = 0.7, isolation_to_trace_delay = Exponential(1.0),
    action = Quarantine(duration = Inf), eligibility = elig)
nothing # hide
```

To write a trigger of your own, such as one that depends on the infector's
age, see [Extending EpiBranch](extending.md).

## Asymptomatic cases and test sensitivity

A fraction `prob_asymptomatic` of cases never develop symptoms, and
symptom-driven isolation misses them. Here asymptomatic cases are as
infectious as symptomatic ones. `test_sensitivity` is the probability that a
symptomatic case tests positive and is isolated; the rest are missed:

```@example interventions
disease_hard = clinical_presentation(
    incubation_period = LogNormal(1.5, 0.5),
    prob_asymptomatic = 0.3,
)
iso_imperfect = Isolation(onset_to_isolation_delay = Exponential(2.0), test_sensitivity = 0.8, duration = Inf)
rng = StableRNG(42)
results = simulate(scenario([iso_imperfect, ct], disease_hard), 200; max_cases = 500, rng = rng)
println("30% asymptomatic, 80% test sensitivity: $(round(containment_probability(results), digits=3))")
```

Containment is lower than with isolation and tracing when every case is
symptomatic and detected, above: missed cases transmit unchecked, and their
contacts are never traced.

## Contact tracing workload

The simulation keeps everyone each case could have infected, including those
who were not infected. It can therefore count how many contacts tracing reached as a
measure of workload. `condition = 50:200` keeps simulating until an outbreak
stops with between 50 and 200 cases, and returns that one outbreak. It stops
either because it died out or because it reached the 200-case cap, as the
one here did:

```@example interventions
rng = StableRNG(42)
state = simulate(scenario([iso, ct]); condition = 50:200, max_cases = 200, rng = rng)

contacts = length(state.individuals)
infected = count(is_infected, state.individuals)
traced = count(is_traced, state.individuals)
println("Contacts: $contacts, Infections: $infected, Traced: $traced")
println("Traced contacts per case: $(round(traced / infected, digits=1))")
```

`count(is_traced, state.individuals)` counts the people for whom `is_traced`
is true, like `sum(is_traced(x))` in R. Every contact generated is
infected unless an intervention prevents it. The gap between contacts and
infections is the number of infections isolation and quarantine prevented.

## Interventions that start or stop during the outbreak

In real outbreaks interventions are rarely in place from the start. Testing
may begin on day 14, or contact tracing once cases pass a threshold.
[`Scheduled`](@ref) sets when an intervention begins or ends. The examples
in this section go back to the 7-day isolation from the start of the page,
and compare with its result.

Two checks decide whether a scheduled isolation or contact tracing reaches a
case. The simulation sets up each case at the start of the generation in
which they transmit, and the intervention applies to them only if some case
in the outbreak has by then been infected on or after the start day. The action's
own time must then also fall on or after the start day. With testing from
day 10, someone whose isolation would fall on day 9 is never isolated.
Someone infected on day 8 with symptom onset on day 9 and a 2-day delay would
be isolated on day 11, which happens only if another case had been infected
on or after day 10 by the time they were set up.

!!! note "The earliest cases are missed"
    Index cases are set up on day 0, so isolation or tracing scheduled to
    start later never reaches them. The same holds for any case set up before the
    outbreak's latest infection reaches the start day, even if their isolation
    would fall after it. A scheduled [vaccination](vaccination.md) is checked
    differently: each dose counts if its own date falls within the schedule,
    whenever the case was set up.

```@example interventions
# Testing starts on day 10
iso_delayed = Scheduled(Isolation(onset_to_isolation_delay = Exponential(2.0), duration = 7.0); start_time = 10.0)

rng = StableRNG(42)
results = simulate(scenario([iso_delayed]), 200; max_cases = 500, rng = rng)
println("Isolation from day 10: $(round(containment_probability(results), digits=3))")
```

Containment is lower than with isolation from the start, because the first
cases, when the outbreak is smallest and easiest to stop, are not isolated.

A start can also follow the case count:

```@example interventions
# Start contact tracing after 20 cumulative cases
iso = Isolation(onset_to_isolation_delay = Exponential(2.0), duration = 7.0)
ct_triggered = Scheduled(
    ContactTracing(probability = 0.7, isolation_to_trace_delay = Exponential(1.0),
        action = Quarantine(duration = Inf));
    start_after_cases = 20,
)

rng = StableRNG(42)
results = simulate(scenario([iso, ct_triggered]), 200; max_cases = 500, rng = rng)
println("Tracing after 20 cases: $(round(containment_probability(results), digits=3))")
```

Tracing that waits for 20 cases misses the contacts of the first cases, and
it changes nothing in outbreaks that die out before reaching 20. The result
is therefore close to 7-day isolation alone; with 200 simulations, a difference of
a few percentage points either way is within simulation noise.

!!! note "The case count includes undetected cases"
    `start_after_cases` counts every infection in the simulation, including
    asymptomatic and missed cases that a surveillance system would not see.
    A programme triggered by reported cases starts later than this.

Start and end times can be combined:

```@example interventions
# Active only between day 5 and day 30
iso_window = Scheduled(Isolation(onset_to_isolation_delay = Exponential(1.0), duration = 7.0);
    start_time = 5.0, end_time = 30.0)
```

For any other trigger, pass a function of the [`SimulationState`](@ref) that
returns `true` while the intervention should be on. This one switches
isolation on from the third generation of transmission, a quantity no real
programme observes but which is useful for exploring the model:

```@example interventions
iso_gen3 = Scheduled(
    Isolation(onset_to_isolation_delay = Exponential(2.0), duration = 7.0),
    state -> state.current_generation >= 3,
)
```

## Writing your own intervention

To write an intervention that is not built in, or to see how the built-in
ones work, see [Extending EpiBranch](extending.md).
