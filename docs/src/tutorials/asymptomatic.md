# Asymptomatic cases and how symptoms link to infectiousness

How symptoms, infectiousness and the within-host course of an infection are related differs between pathogens, and between studies of the same pathogen. EpiBranch builds none of these links in. This page states each one as a short model, using the seams that are already there: an attribute builder (a function `(rng, ind) -> ...`), a generation time or offspring distribution that is a function of the individual, a [`Transition`](@ref) whose delay is a function of the individual, and a [`HazardKernel`](@ref) for a within-host trajectory.

[`clinical_presentation`](@ref) with `prob_asymptomatic` makes a proportion of cases asymptomatic and links nothing else to it. An asymptomatic case then transmits exactly as a symptomatic one does. Everything below adds one explicit statement to that model. All times are in days, and the numbers illustrate the method without describing a pathogen.

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

Because the generation time here is independent of the incubation period, about 60% of the secondary cases of a symptomatic case are infected before its onset.

## Asymptomatic infection is less infectious

Less shedding is a statement about biology. It lowers the chance that each contact is infected. List [`transmission_traits`](@ref) after `clinical_presentation` so that `:asymptomatic` is set when its `infectiousness` is drawn:

```@example asymptomatic
less_shedding = [
    clinical,
    transmission_traits(infectiousness = (rng, ind) -> ind.state[:asymptomatic] ? 0.5 : 1.0),
]
biology = ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0)); attributes = less_shedding)
compare(biology)
```

## Asymptomatic cases make more contacts

People without symptoms do not stay at home. That is behaviour, and it changes the number of contacts, so it belongs in the offspring distribution instead of the infectiousness. The offspring can be a function of the random number generator and the individual. Here asymptomatic cases make 50% more contacts. The lower infectiousness from above stays as a separate statement, and the two multiply:

```@example asymptomatic
more_contacts = (rng, ind) -> rand(rng, Poisson(ind.state[:asymptomatic] ? 4.5 : 3.0))
both = ModelSpec(BranchingProcess(more_contacts, Exponential(5.0)); attributes = less_shedding)
compare(both)
```

Dropping `transmission_traits` leaves the behaviour unchanged, and dropping the offspring function leaves the biology unchanged.

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

[`incubation_linked_generation_time`](@ref) centres each case's generation time on its own incubation period, so that a set share of transmission comes before onset. An asymptomatic case has no onset, but [`clinical_presentation`](@ref) still draws it an incubation period, the time onset would have happened (`incubation_period(ind)`), and by default its generation time is centred on that. A generation time can be any function of the individual, so giving asymptomatic cases a different one takes a one-line wrapper. Here they transmit on the plain exponential timescale instead:

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

The incubation period an asymptomatic case draws can also differ from a symptomatic case's. Give `incubation_period` a function of the random number generator and the individual. `:asymptomatic` is set before the draw, so the function can read it:

```@example asymptomatic
own_incubation = clinical_presentation(
    incubation_period = (rng, ind) -> rand(
        rng, ind.state[:asymptomatic] ? LogNormal(1.2, 0.5) : LogNormal(1.5, 0.5)
    ),
    prob_asymptomatic = 0.3,
)
shorter_incubation = ModelSpec(BranchingProcess(Poisson(3.0), linked); attributes = own_incubation)
compare(shorter_incubation)
```

Asymptomatic cases now transmit earlier, centred on their shorter incubation period. The closed forms that read the incubation period directly, such as the analytic effect of [`Isolation`](@ref), need a distribution and refuse a function, so simulate instead.

## Onset and infectiousness both follow a trajectory

A within-host trajectory is a function of time since infection, drawn once per case and stored on the individual. [`within_host`](@ref) is the builder that stores it under `:trajectory`, and `EpiBranch.trajectory_value(ind, t)` reads the load at `t` days after infection. Everything linked to it is derived from that one draw. Here the load rises to a peak and clears. A case develops symptoms if its peak passes a threshold, at the time the load crosses it; a case whose peak stays below is asymptomatic. The same load, scaled, is the rate of transmission of the case through a [`HazardKernel`](@ref). Because symptoms depend on the same draw, the builder draws the trajectory itself and stores it under `:trajectory`. It also sets `:asymptomatic` and `:incubation_period` (the time onset happens or would have happened), from which the onset time follows:

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
    ind.state[:asymptomatic] = trajectory.peak <= threshold
    ind.state[:incubation_period] = trajectory.peak_time * threshold / trajectory.peak
    return nothing
end

from_trajectory = Infectiousness(
    Poisson(6.0);
    kernel = ind -> HazardKernel(t -> 0.4 * EpiBranch.trajectory_value(ind, t); upper = 60.0),
)
trajectory_model = ModelSpec(BranchingProcess(from_trajectory); attributes = both_follow)
compare(trajectory_model)
```

Asymptomatic cases, with the lower peaks, are less infectious without any statement saying so (a reproduction number of about 3 against about 5). About 40% of the secondary cases of a symptomatic case are infected before its onset.

## Only infectiousness follows the trajectory

Keep the trajectory for transmission and draw symptoms from `clinical_presentation`, independently of it. [`within_host`](@ref) stores the trajectory only:

```@example asymptomatic
infectiousness_only = within_host((rng, ind) -> draw_trajectory(rng).load)
infectiousness_follows = ModelSpec(
    BranchingProcess(from_trajectory);
    attributes = [clinical, infectiousness_only],
)
compare(infectiousness_follows)
```

## Only onset follows the trajectory

Here `both_follow` still draws symptoms from the trajectory, but transmission uses a plain generation time that ignores it:

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
| Onset linked to infectiousness, no trajectory | `incubation_linked_generation_time`, or any `ind -> Distribution` as the generation time; `incubation_period` of `clinical_presentation` can be `(rng, ind) -> value` to give asymptomatic cases their own |
| Trajectory drives onset, infectiousness or both | `within_host` to store the trajectory, an attribute builder that also sets `:asymptomatic` and `:incubation_period` where symptoms follow it, and a `HazardKernel` that reads it with `trajectory_value` |

## Who is tested and classified

Whether a case is noticed is a separate statement from whether it has symptoms. [`Isolation`](@ref) isolates symptomatic cases by default, and its eligibility decides who else is picked up. A [`Testing`](@ref) step samples the same trajectory through an [`Assay`](@ref), so a test result follows from the load that drives infectiousness. By default a test is taken a delay after onset, and an asymptomatic case has no onset, so it is not tested. Timing the sample from infection tests every case, whether or not it has symptoms. A `case_definition` on the [`ModelSpec`](@ref) then classifies each case from what has happened to it. [`ClinicalCaseDefinition`](@ref) calls a case confirmed at a positive test, and a symptomatic case with no positive test suspected:

```@example asymptomatic
assay = Assay(value = load -> 40.0 - 3.3 * load, detect = 0.5)
screening = Testing(assay = assay, from = :infection, delay = Uniform(3.0, 8.0))
classified = ModelSpec(
    BranchingProcess(from_trajectory);
    attributes = both_follow,
    progression = [screening],
    case_definition = ClinicalCaseDefinition(),
)
states = simulate(classified, 100; max_generations = 3, max_cases = 2000, rng = StableRNG(2))
cases = [ind for state in states for ind in state.individuals if EpiBranch.is_infected(ind)]
asymptomatic_cases = filter(is_asymptomatic, cases)
(
    asymptomatic_tested = count(ind -> ind.state[:test_tested], asymptomatic_cases),
    asymptomatic_confirmed = count(
        ind -> isequal(get(ind.state, :case_classification, missing), :confirmed),
        asymptomatic_cases,
    ),
)
```
