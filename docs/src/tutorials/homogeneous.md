# Homogeneous models

`HomogeneousProcess` simulates a closed, homogeneously-mixing population.
Everyone in the population is equally likely to meet everyone else, so
each infectious person exerts the same force of infection on every
susceptible. The population is finite and depletes as the outbreak grows,
so infections saturate at the population size rather than running on
without bound.

It is simulated by the Sellke threshold construction. Each susceptible is
given a resistance threshold drawn from an `Exponential(1)`, and the
pressure of infection accumulates over time as infectious people come and
go. A susceptible is infected the moment that accumulated pressure crosses
its threshold. This recovers the exact stochastic SIR final-size law, and
because it runs in continuous time it gives every case an infection time,
not just a final count.

## Defining a model

`HomogeneousProcess` is a pure transmission kernel: the reproduction number
(or per-infective rate) and the population size are its only inputs. The
natural history, here just an infectious period, is a `progression` of
[`Transition`](@ref)s attached with a [`ModelSpec`](@ref). `simulate` seeds
`n_initial` index cases at time 0 and returns a `SimulationState`.

```@example homogeneous
using EpiBranch
using Distributions
using StableRNGs

model = ModelSpec(HomogeneousProcess(; R0 = 2.0, population_size = 3000);
    progression = [Transition(:recovered; from = :infection,
        delay = Exponential(1.0), terminal = true)])

state = simulate(model; n_initial = 5, rng = StableRNG(1))
state.cumulative_cases
```

With `R0`, β is resolved at simulate time as `R0` divided by the mean
infectious period read from the progression. This means `R0` needs a
progression with a removal transition whose mean is defined: a scalar delay
or a distribution, not a raw function.

[`linelist`](@ref) renders the outbreak as a one-row-per-case DataFrame,
carrying the infection and recovery times the model stamps on each case.

```@example homogeneous
df = linelist(state)
first(df, 5)
```

The attack rate is the share of the population that was infected. At
`R0 = 2` a major outbreak infects about 80% of the population, the value
the deterministic final-size equation `z = 1 - exp(-R0 z)` gives.

```@example homogeneous
N = 3000
round(count(is_infected, state.individuals) / N, digits = 2)
```

Transmission can be set as `R0` or, equivalently, as a per-infective rate
`transmission_rate`. `R0` is the rate times the mean infectious period, so
the two models below are the same. Set as `transmission_rate`, β is fixed on
the kernel directly; set as `R0`, it is resolved from the progression at
simulate time.

```@example homogeneous
model_rate = ModelSpec(
    HomogeneousProcess(; transmission_rate = 2.0, population_size = 3000);
    progression = [Transition(:recovered; from = :infection,
        delay = Exponential(1.0), terminal = true)])
model_rate.process.rate   # β, held directly (set as transmission_rate)
```

## An exposed period

Adding an `:infectious` transition inserts an exposed period between
infection and infectiousness, turning the SIR model into an SEIR one. The
infectious period then runs `from = :infectious` instead of from infection.
The final size is governed by `R0` and is unchanged by the latent period;
what changes is the timing, since a case is now infectious only after its
exposed period has passed.

```@example homogeneous
seir = ModelSpec(HomogeneousProcess(; R0 = 2.0, population_size = 3000);
    progression = [
        Transition(:infectious; from = :infection, delay = Exponential(2.0)),
        Transition(:recovered; from = :infectious,
            delay = Exponential(4.0), terminal = true)])
seir_state = simulate(seir; n_initial = 10, rng = StableRNG(3))
sort(propertynames(linelist(seir_state)))
```

`date_infectious` now appears alongside `date_infection`, because the
progression writes an `:infectious_time` onto each case. The natural
history is a `progression` of [`Transition`](@ref)s, exactly as for
[`BranchingProcess`](@ref), so symptom onset, hospitalisation and death
come from the same mechanism and appear as their own line-list columns. The
kernel's `from`, the state its infectious window opens at, is derived from
the progression: `:infectious` when a latent transition produces it,
otherwise `:infection`.

## Isolation shortens the outbreak

Isolation acts by closing a case's infectious window early: an isolated
case stops contributing to the force of infection, so fewer of its
would-be contacts ever cross their threshold. Adding an `:isolated`
transition to the progression, one of the removal states that ends the
infectious window, isolates each case a fixed time after infection.

Here the same population runs to a large outbreak when nothing intervenes,
but isolating each case a day after infection holds it well back.

```@example homogeneous
baseline = ModelSpec(
    HomogeneousProcess(; transmission_rate = 2.0, population_size = 2000);
    progression = [Transition(:recovered; from = :infection,
        delay = Exponential(1.0), terminal = true)])

isolating = ModelSpec(
    HomogeneousProcess(; transmission_rate = 2.0, population_size = 2000);
    progression = [
        Transition(:recovered; from = :infection,
            delay = Exponential(1.0), terminal = true),
        Transition(:isolated; from = :infection, delay = (rng, ind) -> 1.0)])

base_sizes = [simulate(baseline; n_initial = 5, rng = StableRNG(s)).cumulative_cases
              for s in 1:30]
iso_sizes = [simulate(isolating; n_initial = 5, rng = StableRNG(s)).cumulative_cases
             for s in 1:30]

println("Mean size, no isolation:   ",
    round(sum(base_sizes) / length(base_sizes), digits = 1))
println("Mean size, with isolation: ",
    round(sum(iso_sizes) / length(iso_sizes), digits = 1))
```

## Structured mixing

`HomogeneousProcess` assumes everyone mixes with everyone else at the same
rate. When mixing is uneven, for example age bands, spatial patches or
demographic strata that contact each other at different rates, the same
Sellke pool takes a contact structure without rewriting the simulation.
The [Extending](extending.md) guide shows how to write such a model.
