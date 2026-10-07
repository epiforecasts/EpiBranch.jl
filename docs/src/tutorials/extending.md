# Extending EpiBranch

Sometimes the outbreak you want to model has a feature the built-in models and
interventions lack: a reproduction number that falls once a gathering ban
starts, a control measure the package does not include, a ward where patients
can only infect the people in the next beds. This guide shows how to add such a
feature in a few lines of Julia, in your own script, without editing the
package.

If what you need is already built in (isolation, contact tracing, ring or mass
vaccination), start with [Isolation and contact tracing](interventions.md) or
[Vaccination](vaccination.md) instead: using the built-in measures needs
nothing beyond keyword arguments.

The recipes below assume a little Julia: writing a function, and the
`(rng, ind) -> ...` form of an anonymous function. [Julia for R
users](../julia-for-r-users.md) covers what you need.

## Which tool do I need?

| What you want to model | What you write | Where |
|---|---|---|
| R that changes over time, between people or once a policy starts; a cap on how many people one case can infect | a function in place of the offspring distribution | [Change how many people each case infects](@ref) |
| A generation time that depends on the case, such as one linked to its incubation period | a function returning a distribution | [Generation time that depends on the case](@ref) |
| Population characteristics the built-in builders do not set (risk group, region, a household's reporting rate) | a function of the random number generator and the individual | [Population characteristics of your own](@ref) |
| A rule of your own for when a simulation stops | a small type and one method | [Stopping rules](#Stopping-rules) |
| A control measure that blocks some transmissions (a border closure, prophylaxis) | a type and a `competing_risk` method | [A custom intervention: closing a border](@ref) |
| Treatment that ends infections early, vaccination aimed at a group, a measure limited by a start date or by capacity | an intervention type with further methods | [Writing an intervention](writing-interventions.md) |
| A rule of your own for which cases have their contacts traced | a small type and one `is_eligible` method | [Who triggers contact tracing](@ref) |
| A clinical event of your own (testing, treatment, loss to follow-up) | a clinical transition type | [Custom clinical transitions](@ref) |
| A latent period, several routes of transmission (community, household, funeral), seasonal transmission | infectiousness windows and routes | [New transmission structures](new-structures.md) |
| A contact structure the package lacks (a ward, a school, a network you build), or a closed population with structured mixing | a transmission model type | [New transmission structures](new-structures.md) |
| A new way cases are observed, or new data to fit | an observation model or a likelihood method | [New transmission structures](new-structures.md) |
| The per-person values the package reserves, which interventions can be fitted exactly, numerical caveats | nothing; look things up | [Extension reference](extending-reference.md) |

## Words used in these pages

Writing a new piece of a model means using a few programming words. Each has a
plain meaning here.

| Word | Meaning in these pages |
|---|---|
| type (`struct`) | A named record holding the parameters of something you add, such as a border closure's leakage. `struct BorderClosure <: AbstractIntervention ... end` declares one; `<: AbstractIntervention` tells the package it is an intervention and where to use it. |
| method | One version of a package function, written for your type. `EpiBranch.competing_risk(bc::BorderClosure, ...) = ...` tells the package's `competing_risk` what a border closure does. Writing methods like this is how you extend EpiBranch: you never edit its source. |
| hook | A package function the simulation calls at a fixed step: when a person is created, before a case's contacts are drawn, when a transmission is about to happen. You write a method of it for your intervention. A hook you leave out does nothing. |
| risk (`Risk`) | Something that can stop one transmission from a given time on, with a given probability. Isolation on day 5 is a risk with probability 1 from day 5. |
| state | Each simulated person (`ind`, an [`Individual`](@ref)) has a dictionary `ind.state` of named values, such as `ind.state[:age]` or `ind.state[:isolated]`. The whole outbreak so far is a [`SimulationState`](@ref), usually called `state`. |
| model (`ModelSpec`) | A transmission model together with its natural history (`progression`), population characteristics (`attributes`), interventions and observation. This is what you pass to `simulate` and `loglikelihood`. |
| transmission model | Who can infect whom, and when: a branching process, a network, a household model. |
| generation-based and continuous-time models | Branching processes are simulated one generation at a time. The network, household and homogeneous models simulate a fixed population in continuous time: everyone exists from the start, and the simulation works out when each person is infected (by the Sellke construction). Some hooks are called by only one kind. |
| contact interval (`kernel` in code) | The time from the start of a case's infectiousness to a contact that would infect if nothing intervened. |
| `!` at the end of a name | The function changes its argument in place, as `abort_infection!(ind, t)` changes `ind`. |

## Change how many people each case infects

Gathering limits, event-size caps and a reproduction number that changes over
time all change how many people a case infects. Pass a function in place of the
offspring distribution to [`BranchingProcess`](@ref). The function receives the
random number generator `rng` (pass it to every `rand` call so runs are
reproducible) and the infector `ind`, and returns the number of secondary cases.

Here no case can infect more than five people, as under a limit on gathering
size. `NegBin(2.5, 0.16)` has mean R = 2.5 and dispersion k = 0.16, and the
generation time `Exponential(5.0)` has a mean of 5 days:

```@example extending
using EpiBranch
using Distributions
using StableRNGs

capped_offspring(rng, ind) = min(rand(rng, NegBin(2.5, 0.16)), 5)

uncapped = simulate(BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0)), 500;
    max_cases = 500, rng = StableRNG(1))
capped = simulate(BranchingProcess(capped_offspring, Exponential(5.0)), 500;
    max_cases = 500, rng = StableRNG(1))
(uncapped = containment_probability(uncapped),
 capped = containment_probability(capped))
```

The containment probability is the proportion of the 500 simulated outbreaks
that ended before reaching 500 cases. Removing the largest
superspreading events lets more outbreaks die out.

A function with a third argument, `(rng, ind, state)`, can also read the
outbreak so far. Here the cap only applies once there have been 20 cases, as a
policy brought in partway through an outbreak would. `c ? a : b` is Julia for
R's `if (c) a else b`:

```@example extending
function policy_offspring(rng, ind, state)
    n = rand(rng, NegBin(2.5, 0.16))
    return state.cumulative_cases >= 20 ? min(n, 5) : n
end
model_policy = BranchingProcess(policy_offspring, Exponential(5.0))
```

A reproduction number that changes over time reads the infector's infection
time, `ind.infection_time`, in days. Here R falls from 3 to 1 over the first 50
days:

```@example extending
r_at_time(t) = max(1.0, 3.0 - 2.0 * t / 50.0)
time_varying = (rng, ind) -> rand(rng, Poisson(r_at_time(ind.infection_time)))
model_rt = BranchingProcess(time_varying, Exponential(5.0))
```

Use `ind.infection_time` when R follows each infector's own infection time, and
`state.max_infection_time` (with the three-argument form) when R follows the
outbreak's own clock.

R can also depend on who the infector is. Below it depends on a risk group set
by a population characteristic (see [Population characteristics of your
own](@ref) for how `:risk_group` is set):

```@example extending
function risk_group!(rng, ind)
    if rand(rng) < 0.2
        ind.state[:risk_group] = :high
    else
        ind.state[:risk_group] = :low
    end
    return nothing
end

function risk_offspring(rng, ind)
    R = ind.state[:risk_group] == :high ? 4.0 : 1.5
    return rand(rng, Poisson(R))
end

model_risk = ModelSpec(BranchingProcess(risk_offspring, Exponential(5.0); n_types = 1);
    attributes = risk_group!)
results = simulate(model_risk, 200; max_cases = 500, rng = StableRNG(42))
containment_probability(results)
```

A fifth of cases are high-risk, giving an average R of 0.2 × 4 + 0.8 × 1.5 = 2,
and most outbreaks grow past 500 cases.

Or on the infector's generation, here for transmission that wanes as the
outbreak goes on:

```@example extending
function waning_offspring(rng, ind)
    R = 3.0 * exp(-0.1 * ind.generation)
    return rand(rng, Poisson(R))
end

model_waning = BranchingProcess(waning_offspring, Exponential(5.0); n_types = 1)
results = simulate(model_waning, 200; max_cases = 500, rng = StableRNG(42))
containment_probability(results)
```

R drops below 1 after about 11 generations, by which time most outbreaks have
already passed 500 cases.

!!! note "Closed-form results need a distribution"
    A function works for simulation only. Extinction probability, chain-size
    distributions and the other [analytical functions](analytical.md) need an
    offspring distribution they can work with; see [Offspring distributions of
    your own](@ref) for writing one.

## Generation time that depends on the case

`generation_time` can be one `Distribution` for everyone, or a function of the
infector that returns a distribution. The function can read anything stored on
the infector. A common use links the generation time to the case's own
incubation period, read with [`incubation_period`](@ref). `LogNormal(1.5, 0.5)`
takes the mean and standard deviation of the log incubation period, in days, and
`Gamma(2.0, θ)` has shape 2 and scale θ:

```@example extending
gt = ind -> Gamma(2.0, incubation_period(ind) / 2)
linked = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), gt);
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)))

state = simulate(linked; max_cases = 500, rng = StableRNG(42))
state.cumulative_cases
```

Each case's incubation period is drawn once and its generation time is built
from it. The two are therefore correlated: a case with a late onset also tends to
transmit late. [`incubation_linked_generation_time`](@ref) is a ready-made
version, the skew-normal model of Hellewell et al. (2020).

The generation time can depend on any value a population characteristic has
stored, not only the incubation period. Here one per-person draw sets both the
onset time and the mean generation time, in days:

```@example extending
function host!(rng, ind)
    scale = 3.0 + rand(rng)
    ind.state[:gt_scale] = scale
    ind.state[:onset_time] = ind.infection_time + scale
    return nothing
end

scaled = ModelSpec(
    BranchingProcess(Poisson(2.0), ind -> Exponential(ind.state[:gt_scale]));
    attributes = host!)

state = simulate(scaled; max_cases = 500, rng = StableRNG(42))
state.cumulative_cases
```

Use this whenever generation time and onset should come from one per-person
draw instead of two independent ones.

## Population characteristics of your own

The `attributes` argument of a model is a function `(rng, ind) -> ...` that
sets values on each person when they are created, before any intervention
acts. The built-in builders `clinical_presentation`, `demographics` and
`transmission_traits` return such functions. For any other value, write your
own, as `risk_group!` above does.

Characteristics can feed each other. Here `transmission_traits` sets each
person's susceptibility (a number between 0 and 1) from the risk group:

```@example extending
attrs = [
    risk_group!,
    transmission_traits(
        susceptibility = (rng, ind) -> ind.state[:risk_group] == :high ? 0.8 : 0.3,
    ),
]

model = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0)); attributes = attrs)
runs = simulate(model, 200; max_cases = 100, rng = StableRNG(42))
people = reduce(vcat, [s.individuals for s in runs])
high_share(group) = count(ind -> ind.state[:risk_group] == :high, group) / length(group)
(all_contacts = high_share(people), infected = high_share(filter(is_infected, people)))
```

The output keeps contacts who were exposed but not infected. About a fifth of
all contacts are high-risk, but because they are more susceptible they make up a
larger share of those infected (0.2 × 0.8 / (0.2 × 0.8 + 0.8 × 0.3) = 0.4,
ignoring the index cases). The list is applied in order, so a later entry can
read what an earlier one set. Built-in builders and
your own functions mix freely:

```@example extending
combined = [
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    demographics(age_distribution = Normal(40, 15)),
    risk_group!,
    transmission_traits(
        susceptibility = (rng, ind) -> ind.state[:risk_group] == :high ? 0.8 : 0.3,
    ),
]

model_combined = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0));
    attributes = combined)
state = simulate(model_combined; max_cases = 100, rng = StableRNG(42))
ind = state.individuals[1]
(age = ind.state[:age], sex = ind.state[:sex], risk = ind.state[:risk_group])
```

### Sharing a value within groups

[`group_attribute`](@ref) gives every member of a household, community or other
group the same value. It draws once for the first member of each group and
keeps the value for the run. Here each household has its own reporting
probability, drawn from `Beta(6, 4)` (mean 0.6):

```@example extending
reporting_attributes = [groups(50; key = :household),
    group_attribute(:reporting_probability; value = Beta(6, 4),
        group_key = :household)]
reporting_model = ModelSpec(BranchingProcess(Poisson(0.5), Exponential(5.0));
    attributes = reporting_attributes,
    observation = PerCaseObservation(
        detection_prob = (rng, ind) -> ind.state[:reporting_probability],
        from = :infection_time))
reporting_state = simulate(reporting_model; n_initial = 10, rng = StableRNG(42))
```

Set the group label before the shared value, as the order above does. Reusing
these builders in further simulations draws fresh values, including when
simulations run in parallel.

### Rules with parameters of their own

Observation parameters, the parameters of population characteristics and the
conditions of interventions accept a type with parameters wherever they accept a
function, called with the same arguments. A detection rule can then keep its
threshold as a parameter. `(rule::AgeDetection)(rng, ind) = ...` makes an
`AgeDetection` usable as a function:

```@example extending
struct AgeDetection
    minimum_age::Float64
end
(rule::AgeDetection)(rng, ind) = ind.state[:age] >= rule.minimum_age ? 1.0 : 0.0
age_observation = PerCaseObservation(detection_prob = AgeDetection(50.0))
```

An observation's reference time (`from`) takes only `ind`, and a condition for
[`Scheduled`](@ref) takes the simulation state. Numbers and distributions keep
their usual meaning wherever those are accepted.

## Stopping rules

A simulation stops at the first step where any of its stopping rules says so.
The built-in rules [`Extinction`](@ref), [`MaxCases`](@ref),
[`MaxGenerations`](@ref) and [`MaxTime`](@ref) cover most needs, and the
`max_cases`, `max_generations` and `max_time` keywords build them (see
[`SimOpts`](@ref)). A rule of your own is the smallest type you will write: a
`struct` subtyping [`AbstractStoppingRule`](@ref), and a
[`should_stop`](@ref) method that returns `true` when the simulation should end.

Here a rule stops once any chain of transmission reaches a given number of
generations, whatever the case count:

```@example extending
struct MaxChainLength <: AbstractStoppingRule
    n::Int
end
EpiBranch.should_stop(r::MaxChainLength, state::SimulationState) =
    maximum(ind.generation for ind in state.individuals; init = 0) >= r.n

chain_model = BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0))
chain_state = simulate(
    chain_model; stopping_rules = [Extinction(), MaxChainLength(5)], rng = StableRNG(3)
)
maximum(ind.generation for ind in chain_state.individuals; init = 0)
```

The printed value is the last generation reached: 5 if the rule stopped the
run, fewer if transmission died out first. `init = 0` gives the maximum a
value when there are no individuals.

[`Extinction`](@ref) is added automatically unless your `stopping_rules`
already include it. A simulation therefore still ends when transmission dies out:
`MaxChainLength` alone would never stop a chain that goes extinct before 5
generations.

Write the method as `EpiBranch.should_stop(...)`, with the `EpiBranch.` prefix
(or after `import EpiBranch: should_stop`). Without it, Julia creates a new
function of your own called `should_stop` that the simulation never calls.
Type the state argument as `state::SimulationState`, as above, so your method
does not clash with the package's default one.

!!! note "Stopping rules on continuous-time models"
    The network, household and homogeneous models do not check `should_stop`
    as they go: they run until transmission dies out or a time limit is
    reached. A rule
    that should be able to end such a run also defines
    [`EpiBranch.time_bound`](@ref)`(rule)`, the latest infection time at which
    it could still want the simulation to continue, as [`MaxTime`](@ref) does.
    If reaching that time is all the rule tests, it also declares
    `EpiBranch.honoured_without_should_stop(rule) = true`; otherwise these
    models warn that the rule was ignored. A rule that sets a time limit and
    also tests something else, such as a case count, keeps the default
    `false`, because only its time limit is applied there.

## A custom intervention: closing a border

An intervention that stops some transmissions needs one type and one method of
[`competing_risk`](@ref EpiBranch.competing_risk). The simulation asks every intervention, for each pair
of infector and contact, whether anything blocks that transmission. Whatever
blocks it first wins (survival analysts call these competing risks).

Here two regions close the border between them on day 10. After that, a
transmission between people in different regions goes ahead only with a small
probability, the leakage.

```@example border
using EpiBranch
using Distributions
using StableRNGs

struct BorderClosure <: AbstractIntervention
    start_time::Float64   # day the border closes
    leakage::Float64      # proportion of cross-border transmissions still happening
end

function EpiBranch.competing_risk(bc::BorderClosure, parent, contact, state)
    if parent.state[:region] == contact.state[:region]
        return nothing    # same region: the closure does not apply
    end
    return Risk(event_time = bc.start_time, block_probability = 1.0 - bc.leakage)
end
```

`parent` is the infector and `contact` the person who would be infected.
Returning `nothing` means the intervention does not block this transmission.
`Risk(event_time = 10.0, block_probability = 0.95)` blocks a transmission
happening on or after day 10 with probability 0.95; transmissions before the
closure are unaffected.

Each person needs a region. In this example people are equally likely to live
in either, and about half of all contacts cross the border:

```@example border
region! = (rng, ind) -> (ind.state[:region] = rand(rng, (:north, :south)))

process = BranchingProcess(NegBin(1.6, 0.5), Gamma(2.0, 2.5))
open_border = ModelSpec(process; attributes = region!)
closed_border = ModelSpec(process; attributes = region!,
    interventions = [BorderClosure(10.0, 0.05)])

open_runs = simulate(open_border, 500; max_cases = 500, rng = StableRNG(1))
closed_runs = simulate(closed_border, 500; max_cases = 500, rng = StableRNG(1))
(open = containment_probability(open_runs),
 closed = containment_probability(closed_runs))
```

With R = 1.6 and the border open, a sizeable share of outbreaks grow past 500
cases. After the closure, cross-border transmission falls by 95%. The
effective reproduction number is then about 1.6 × (0.5 + 0.5 × 0.05) ≈ 0.84 and
nearly every outbreak dies out.

Your intervention combines with the built-in ones: pass
`interventions = [Isolation(...), BorderClosure(10.0, 0.05)]` and both apply.

!!! warning "A missing method fails silently"
    Every hook does nothing unless you write a method for it. A typo in a
    method name, or a method written for the wrong type, gives results that
    look like no intervention at all. Compare a small simulation with and
    without your intervention, as above, before relying on it.

## Where next

- [Writing an intervention](writing-interventions.md): every hook an
  intervention can use, the order in which they are called, ending an infection
  early, scheduling, capacity limits, custom vaccinations.
- [New transmission structures](new-structures.md): latent periods and
  infectiousness windows, several routes of transmission, seasonal
  transmission, new transmission models and observation models.
- [Extension reference](extending-reference.md): reserved per-person values,
  which interventions the likelihoods can fit exactly, and details of the
  continuous-time models.
