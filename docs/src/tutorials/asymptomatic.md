# Asymptomatic cases and how symptoms link to infectiousness

How symptoms, infectiousness and the within-host course of an infection are related differs between pathogens, and between studies of the same pathogen. EpiBranch builds none of these links in. This page states each one as a short model, using the seams that are already there: an attribute builder (a function `(rng, ind) -> ...`), a generation time or offspring distribution that is a function of the individual, a [`Transition`](@ref) whose delay is a function of the individual, and a [`HazardKernel`](@ref) for a within-host trajectory.

The simple version stays simple. [`clinical_presentation`](@ref) with `prob_asymptomatic` makes a proportion of cases asymptomatic and links nothing else to it. An asymptomatic case then transmits exactly as a symptomatic one does. Everything below adds one explicit statement to that model. All times are in days, and the numbers illustrate the method without describing a pathogen.

```@example asymptomatic
using EpiBranch
using Distributions
using StableRNGs
```

To see what each link does, the helper below simulates the first three generations of 400 outbreaks and reports, for the cases whose offspring are all in the run, the mean number of secondary cases of asymptomatic and symptomatic cases, and the share of a symptomatic case's secondary cases infected before its own onset. Copy it as it is.

```@example asymptomatic
using Statistics

function compare(model; n = 400)
    states = simulate(model, n; max_generations = 3, max_cases = 5000, rng = StableRNG(1))
    asymptomatic, secondary, before_onset = Bool[], Int[], Float64[]
    for state in states
        by_id = Dict(ind.id => ind for ind in state.individuals)
        for ind in state.individuals
            EpiBranch.is_infected(ind) && ind.generation < 2 || continue
            infected = [by_id[id] for id in ind.secondary_case_ids
                        if EpiBranch.is_infected(by_id[id])]
            push!(asymptomatic, is_asymptomatic(ind))
            push!(secondary, length(infected))
            if !is_asymptomatic(ind) && !isempty(infected)
                push!(before_onset, mean(c.infection_time < onset_time(ind) for c in infected))
            end
        end
    end
    return (
        R_asymptomatic = round(mean(secondary[asymptomatic]), digits = 2),
        R_symptomatic = round(mean(secondary[.!asymptomatic]), digits = 2),
        before_onset = round(mean(before_onset), digits = 2),
    )
end
nothing # hide
```

## A proportion asymptomatic, nothing linked

A shared incubation period and a proportion asymptomatic. Both kinds of case have the same reproduction number and the same generation time:

```@example asymptomatic
clinical = clinical_presentation(
    incubation_period = LogNormal(1.5, 0.5),
    prob_asymptomatic = 0.3,
)
independent = ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0)); attributes = clinical)
compare(independent)
```

The generation time here is independent of the incubation period, so about 60% of a symptomatic case's secondary cases are infected before its onset.

## Asymptomatic infection is less infectious

Less shedding is a statement about biology, and it lowers each contact's chance of infecting. List [`transmission_traits`](@ref) after `clinical_presentation` so that `:asymptomatic` is set when its `infectiousness` is drawn:

```@example asymptomatic
less_shedding = [
    clinical,
    transmission_traits(infectiousness = (rng, ind) -> ind.state[:asymptomatic] ? 0.5 : 1.0),
]
biology = ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0)); attributes = less_shedding)
compare(biology)
```

## Asymptomatic cases make more contacts

People without symptoms do not stay at home, which is a statement about behaviour. It acts on the number of contacts, so it belongs in the offspring distribution rather than in the infectiousness. The offspring can be a function of the random number generator and the individual. Here asymptomatic cases make 50% more contacts, and the lower infectiousness from above is kept as a separate statement, so the two multiply:

```@example asymptomatic
more_contacts = (rng, ind) -> rand(rng, Poisson(ind.state[:asymptomatic] ? 4.5 : 3.0))
both = ModelSpec(BranchingProcess(more_contacts, Exponential(5.0)); attributes = less_shedding)
compare(both)
```

Dropping `transmission_traits` leaves behaviour alone, and dropping the offspring function leaves biology alone. Neither choice is wired to the other.

## Asymptomatic infection is shorter

A shorter infectious period is a delay that depends on the case. Recovery is a [`Transition`](@ref) whose delay is a function of the individual, and the contacts of the transmission window stop at recovery with `until`:

```@example asymptomatic
progression = [
    Transition(
        :recovered, from = :infection, terminal = true,
        delay = (rng, ind) -> ind.state[:asymptomatic] ? 3.0 : 10.0,
    ),
]
window = Infectiousness(Poisson(3.0); kernel = Exponential(5.0), until = (:recovered,))
shorter = ModelSpec(BranchingProcess(window); attributes = clinical, progression)
compare(shorter)
```

## Onset linked to infectiousness, with no trajectory

[`incubation_linked_generation_time`](@ref) centres each case's generation time on its own incubation period, so that a set share of transmission comes before onset. An asymptomatic case has no onset, and by default it is centred on the incubation period it would have had. That is one choice among several. A generation time is any function of the individual, so a different one for asymptomatic cases is a one-line wrapper. Here they transmit on the plain exponential timescale instead:

```@example asymptomatic
linked = incubation_linked_generation_time(presymptomatic_fraction = 0.3)
onset_linked = ModelSpec(BranchingProcess(Poisson(3.0), linked); attributes = clinical)
compare(onset_linked)
```

```@example asymptomatic
own_timing = ind -> is_asymptomatic(ind) ? Exponential(5.0) : linked(ind)
asymptomatic_own_timing = ModelSpec(BranchingProcess(Poisson(3.0), own_timing); attributes = clinical)
compare(asymptomatic_own_timing)
```

Both put about 30% of a symptomatic case's transmission before its onset, as set. They differ only in when asymptomatic cases transmit, which has no onset to be measured from.

## Onset and infectiousness both follow a trajectory

A within-host trajectory is a function of time since infection that the user draws once per case and stores on the individual. Everything linked to it is then derived from that one draw. Here the load rises to a peak and clears. A case develops symptoms if its peak passes a threshold, at the time the load crosses it, and a case whose peak stays below is asymptomatic. The same load, scaled, is the case's rate of transmission through a [`HazardKernel`](@ref). The builder sets `:asymptomatic` and `:incubation_period` (`NaN` for an asymptomatic case), which is what isolation and the onset time are worked out from, and `:latent_incubation_period`, which [`incubation_linked_generation_time`](@ref) reads:

```@example asymptomatic
function draw_trajectory(rng)
    peak_time = rand(rng, Gamma(6.0, 0.7))
    peak = rand(rng, LogNormal(0.0, 0.6))
    clearance = rand(rng, Gamma(8.0, 0.1))
    load = t -> t < peak_time ? peak * t / peak_time : peak * exp(-clearance * (t - peak_time))
    return (; load, peak_time, peak)
end

threshold = 0.8
function both_follow(rng, ind)
    trajectory = draw_trajectory(rng)
    ind.state[:trajectory] = trajectory.load
    symptomatic = trajectory.peak > threshold
    onset = trajectory.peak_time * threshold / trajectory.peak
    ind.state[:asymptomatic] = !symptomatic
    ind.state[:latent_incubation_period] = onset
    ind.state[:incubation_period] = symptomatic ? onset : NaN
    return nothing
end

from_trajectory = Infectiousness(
    Poisson(6.0);
    kernel = ind -> HazardKernel(t -> 0.4 * ind.state[:trajectory](t); upper = 60.0),
)
trajectory_model = ModelSpec(BranchingProcess(from_trajectory); attributes = both_follow)
compare(trajectory_model)
```

Asymptomatic cases, with the lower peaks, are less infectious without any statement saying so (a reproduction number of about 3 against about 5), and about 40% of a symptomatic case's secondary cases are infected before its onset.

## Only infectiousness follows the trajectory

Keep the trajectory for transmission and draw symptoms from `clinical_presentation`, independently of it. The builder stores the trajectory only:

```@example asymptomatic
function infectiousness_only(rng, ind)
    return ind.state[:trajectory] = draw_trajectory(rng).load
end
infectiousness_follows = ModelSpec(
    BranchingProcess(from_trajectory);
    attributes = [clinical, infectiousness_only],
)
compare(infectiousness_follows)
```

## Only onset follows the trajectory

The reverse keeps `both_follow`, so that symptoms come from the trajectory, and gives transmission a plain generation time that ignores it:

```@example asymptomatic
onset_follows = ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0)); attributes = both_follow)
compare(onset_follows)
```

## Which seam states which link

| Link | Written as |
|---|---|
| Proportion asymptomatic | `prob_asymptomatic` of `clinical_presentation`, a number, a distribution or `(rng, ind) -> probability` |
| Asymptomatic cases shed less | `infectiousness = (rng, ind) -> ...` of `transmission_traits`, listed after `clinical_presentation` |
| Asymptomatic cases make more contacts | the offspring of the `Infectiousness`, `(rng, ind) -> n` |
| Asymptomatic infection is shorter | the `delay` of a `Transition`, `(rng, ind) -> delay`, with `until` on the window |
| Onset linked to infectiousness, no trajectory | `incubation_linked_generation_time`, or any `ind -> Distribution` as the generation time |
| Trajectory drives onset, infectiousness or both | an attribute builder that stores the trajectory, sets `:asymptomatic` and `:incubation_period`, and a `HazardKernel` read from it |

Whether a case is noticed is a separate statement from whether it has symptoms. [`Isolation`](@ref) isolates symptomatic cases by default, and its eligibility decides who else is picked up. A case definition belongs there, not in these links.
