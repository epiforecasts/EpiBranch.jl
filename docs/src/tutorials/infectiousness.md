# Making some cases more or less infectious

Every route below sets one quantity: the rate at which a case infects others at each time since its infection. That rate is a baseline, multiplied by a factor for the person and by factors that events switch on. The package has four places to set it. Which one applies depends on what makes the case different.

| What makes the case different? | Where to set it |
|---|---|
| A property known when the case is created (age, or whether it is asymptomatic), acting for the whole infection | [`transmission_traits`](@ref)`(infectiousness = ...)` |
| The state of its disease, such as only between onset and recovery, or at a funeral | [`Infectiousness`](@ref) windows (`from`, `until`, `name`) |
| An event that happens later (isolation, quarantine, tracing, vaccination) | an intervention, through [`transmission_risks`](@ref EpiBranch.transmission_risks) and [`competing_risk`](@ref EpiBranch.competing_risk) |
| When in the infection transmission happens, rather than how much | the generation time, or an `Infectiousness` window's `kernel` |

## Fixed at infection: `transmission_traits`

`transmission_traits(infectiousness = ...)` draws a factor once, when the case is created, and scales onward transmission by it for the whole infection. The factor can be a number, a distribution or a function `(rng, ind) -> value` that reads what is already known about the case. Asymptomatic cases half as infectious throughout:

```julia
attributes = [
    clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.3,
    ),
    transmission_traits(
        infectiousness = (rng, ind) -> ind.state[:asymptomatic] ? 0.5 : 1.0,
    ),
]
```

The factor cannot depend on anything that happens after the case is created. For that, use an intervention.

## Tied to disease state: `Infectiousness` windows

An [`Infectiousness`](@ref) window says between which states of the natural history a case transmits by one route, with its own offspring distribution. A case that transmits only once infectious, or by a funeral route after death, is described here. Several windows with different names give several routes.

## Triggered by an event: interventions

An intervention reduces transmission from the time an event occurs, through its `transmission_risks` and `competing_risk` methods. [`Isolation`](@ref) and quarantine work this way: the factor applies from isolation onwards. A case traced at day 4 and made half as infectious from then on needs an intervention, because `transmission_traits` cannot see the tracing. See [Writing an intervention](@ref) for how to write one, and [Isolation and contact tracing](@ref) for those that exist.

## About timing: the generation time

The generation time, or the `kernel` of an `Infectiousness` window, can be any distribution or any function `ind -> Distribution`. It sets when the infections of a case happen and leaves their number unchanged. Use it for infectiousness that peaks early or late, or that differs by person in its timing.

## Combining them

The routes multiply. A case that is asymptomatic (halved by `transmission_traits`), and traced and isolated at day 4 (by an intervention from day 4), transmits at half the baseline until day 4 and at the isolation-reduced rate after it. Use `transmission_traits` for what is true of a case from the start, and an intervention for what is true from an event on; a single factor cannot do both.
