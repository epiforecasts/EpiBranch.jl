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

The reproduction number, the population size and the infectious period are
the three inputs. `simulate` seeds `n_initial` index cases at time 0 and
returns a `SimulationState`.

```@example homogeneous
using EpiBranch
using Distributions
using StableRNGs

model = HomogeneousProcess(; R0 = 2.0, population_size = 3000,
    infectious_period = Exponential(1.0))

state = simulate(model; n_initial = 5, rng = StableRNG(1))
state.cumulative_cases
```

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
the two models below are the same.

```@example homogeneous
model_rate = HomogeneousProcess(; transmission_rate = 2.0, population_size = 3000,
    infectious_period = Exponential(1.0))
model_rate.β
```

## An exposed period

A `latent_period` inserts an exposed period between infection and
infectiousness, turning the SIR model into an SEIR one. The final size is
governed by `R0` and is unchanged by the latent period; what changes is
the timing, since a case is now infectious only after its exposed period
has passed.

```@example homogeneous
seir = HomogeneousProcess(; R0 = 2.0, population_size = 3000,
    latent_period = Exponential(2.0), infectious_period = Exponential(4.0))
seir_state = simulate(seir; n_initial = 10, rng = StableRNG(3))
sort(propertynames(linelist(seir_state)))
```

`date_infectious` now appears alongside `date_infection`, because the
progression writes an `:infectious_time` onto each case. The natural
history is a `progression` of [`Transition`](@ref)s, exactly as for
[`BranchingProcess`](@ref), so symptom onset, hospitalisation and death
come from the same mechanism and appear as their own line-list columns.

## Isolation shortens the outbreak

Isolation acts by closing a case's infectious window early: an isolated
case stops contributing to the force of infection, so fewer of its
would-be contacts ever cross their threshold. Adding an `:isolated`
transition to the progression, one of the removal states that ends the
infectious window, isolates each case a fixed time after infection.

Here the same population runs to a large outbreak when nothing intervenes,
but isolating each case a day after infection holds it well back.

```@example homogeneous
baseline = HomogeneousProcess(; transmission_rate = 2.0, population_size = 2000,
    infectious_period = Exponential(1.0))

isolating = HomogeneousProcess(; transmission_rate = 2.0, population_size = 2000,
    infectious_period = Exponential(1.0),
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
