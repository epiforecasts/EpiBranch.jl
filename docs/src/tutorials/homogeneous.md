# Homogeneous models

How large does an epidemic get in a closed population, once it starts running
out of people to infect? `HomogeneousProcess` is a stochastic SIR or SEIR
model in a finite, well-mixed population: everyone is equally likely to meet
everyone else, and each infectious person exerts the same force of infection
on every susceptible. As susceptibles are used up, transmission slows and the
epidemic stops at a final size below the population size.

## When to use this instead of a branching process

A [`BranchingProcess`](@ref) assumes an unlimited supply of susceptibles: each
case's expected number of secondary cases stays at R however large the
outbreak gets. That is a good description of the early phase, of small
outbreaks, and of outbreaks brought under control before they reach a
noticeable share of the population. Choose `HomogeneousProcess` when depletion
of susceptibles matters: a closed population such as a ship, a
school, a care home or a small island, or any question about the final size,
the peak or the end of an epidemic that is not contained.

The two models are parameterised differently. A branching process takes the
reproduction number R0 directly. `HomogeneousProcess` takes the transmission
rate β, the rate per day at which one infectious person makes contacts that
would infect a susceptible. The two are linked by R0 = β × mean infectious
period in days.

## Defining a model

Two pieces describe the outbreak. The transmission process holds β and the
population size N. The `progression` holds the natural history, here only the
infectious period: a recovery event, timed from infection, with
`terminal = true` marking it as the case's final outcome. `simulate` starts
the outbreak with `n_initial` index cases at day 0. A call that starts with a
semicolon, `HomogeneousProcess(; ...)`, takes only named arguments.

```@example homogeneous
using EpiBranch
using Distributions
using StableRNGs

model = ModelSpec(HomogeneousProcess(; transmission_rate = 2.0, population_size = 3000);
    progression = [Transition(:recovered; from = :infection,
        delay = Exponential(1.0), terminal = true)])

state = simulate(model; n_initial = 5, rng = StableRNG(1))
state.cumulative_cases
```

`state` holds the simulated outbreak, and `cumulative_cases` is its final
size: the number of people ever infected. Here β = 2 per day and the
infectious period is exponential with mean 1 day (`Exponential(θ)` has mean
θ), which gives R0 = 2. Each susceptible receives infectious contacts at rate β/N from
each infectious person.

[`linelist`](@ref) returns a table (a `DataFrame`, the Julia equivalent of an
R data frame) with one row per case, including the day of infection and of
recovery:

```@example homogeneous
df = linelist(state)
first(df, 5)
```

The attack rate is the proportion of the population that was infected. At
R0 = 2 a major outbreak infects about 80% of the population, the value the
deterministic final-size equation `z = 1 - exp(-R0 z)` gives:

```@example homogeneous
N = 3000
round(count(is_infected, state.individuals) / N, digits = 2)
```

The simulated attack rate is close to the final-size prediction. With 5 index
cases a major outbreak is almost certain; starting from a single case, some
simulations die out early by chance.

## An exposed period

Adding an `:infectious` event between infection and the start of
infectiousness turns the SIR model into an SEIR one. The recovery event is
then timed from the start of infectiousness, `from = :infectious`. Here the
latent period has a mean of 2 days and the infectious period a mean of 4 days.
The final size depends on β and the infectious period. The latent period
changes only the timing: a case can infect others only once its
latent period has passed.

```@example homogeneous
seir = ModelSpec(HomogeneousProcess(; transmission_rate = 2.0, population_size = 3000);
    progression = [
        Transition(:infectious; from = :infection, delay = Exponential(2.0)),
        Transition(:recovered; from = :infectious,
            delay = Exponential(4.0), terminal = true)])
seir_state = simulate(seir; n_initial = 10, rng = StableRNG(3))
seir_df = linelist(seir_state)
first(seir_df[:, [:id, :date_infection, :date_infectious, :date_recovered]], 5)
```

Each case's line-list row now has the day it became infectious as well as the
day it was infected. The model takes the start of infectiousness from the
natural history: the `:infectious` event when there is one, otherwise the
moment of infection. Symptom onset, hospital admission and death can be added
as further events, as for a [`BranchingProcess`](@ref) (see
[Clinical transitions](transitions.md)), and appear as their own line-list
columns.

## Isolation shortens the outbreak

An isolated case stops contributing to the force of infection. One way to
model this is an `:isolated` event in the natural history: like recovery or
death, it ends the case's infectious period. Here each case isolates one day
after infection. Most cases would not yet have symptoms by then. The
scenario stands for very effective case finding.

The same population runs to a large outbreak when nothing intervenes:

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
        Transition(:isolated; from = :infection, delay = 1.0)])

# repeat each simulation with 30 different random seeds
base_sizes = [simulate(baseline; n_initial = 5, rng = StableRNG(s)).cumulative_cases
              for s in 1:30]
iso_sizes = [simulate(isolating; n_initial = 5, rng = StableRNG(s)).cumulative_cases
             for s in 1:30]

println("Mean size, no isolation:   ", round(mean(base_sizes), digits = 1))
println("Mean size, with isolation: ", round(mean(iso_sizes), digits = 1))
```

Isolating each case a day after infection cuts the mean infectious period from
1 day to about 0.63 days. R0 falls from 2 to about 1.3 and the outbreaks are
much smaller. The [`Isolation`](@ref) intervention, with isolation a delay
after symptom onset, works with this model too.

## Partial protection

Many controls reduce transmission without removing anyone from it: a vaccine
that halves the chance of infection, or isolation that people only partly
keep to. If a fraction p of infectious contacts is blocked, the force of
infection is multiplied by (1 - p).

Relative susceptibility and infectiousness, set with
[`transmission_traits`](@ref), work the same way. A relative susceptibility of
0.75 means a person is infected at 75% of the rate of a fully susceptible
person: R0 falls from 2 to 1.5. Relative infectiousness scales how much
each infected person transmits. Either can be a single number for everyone or
a distribution drawn separately for each person; here everyone has a relative
susceptibility of 0.75 in one scenario, and in the other it varies between
people with the same mean (`Beta(0.75, 0.25)` has mean 0.75, with many people
nearly immune and many nearly fully susceptible). Averages are over 200
simulations:

```@example homogeneous
function mean_size(attributes)
    protected = ModelSpec(
        HomogeneousProcess(; transmission_rate = 2.0, population_size = 2000);
        progression = [Transition(:recovered; from = :infection,
            delay = Exponential(1.0), terminal = true)],
        attributes = attributes)
    sizes = [simulate(protected; n_initial = 5, rng = StableRNG(s)).cumulative_cases
             for s in 1:200]
    return round(mean(sizes), digits = 1)
end

println("Mean size, susceptibility 0.75 for everyone: ",
    mean_size(transmission_traits(susceptibility = 0.75)))
println("Mean size, susceptibility varying, mean 0.75: ",
    mean_size(transmission_traits(susceptibility = Beta(0.75, 0.25))))
```

Lower susceptibility gives smaller epidemics than the baseline above. When
susceptibility varies, the most susceptible people tend to be infected first
and the remaining population is less susceptible on average. The final size is
smaller again than with the same mean susceptibility for everyone. At
a relative susceptibility of 0.5, R0 is exactly 1 and outbreaks are no longer
expected to grow.

An [`Isolation`](@ref) that leaves some transmission after isolating
(`post_isolation_transmission > 0`) reduces transmission in the same way.

!!! warning "Contact tracing does not apply"
    [`ContactTracing`](@ref) and [`RingVaccination`](@ref) need named
    contacts, which a well-mixed population does not have, and `simulate`
    warns if you include either. [`MassVaccination`](@ref) and
    [`GroupVaccination`](@ref) need none: the first doses the whole
    population before the race starts, and the second finds a case's group
    through its `group_key` rather than through a contact, so both work here.
    Isolation, reduced susceptibility or infectiousness, and events in the
    natural history all apply too. To model existing protection rather than a
    rollout, lower susceptibility with `transmission_traits` instead.

!!! note "Stopping the simulation"
    A `HomogeneousProcess` runs until no one is infectious or until
    `max_time`, whichever comes first. The `max_cases` and `max_generations`
    keyword shortcuts are disabled by default. A run continues to
    extinction or `max_time` unless an explicit value or a `stopping_rules`
    entry sets a cap.

## Approximations for large populations

Simulating every person is exact but its cost grows with the outbreak. For a population of tens of thousands or more, or a likelihood that needs many runs, an approximation is often accurate enough. Three models share one interface with the exact simulation, `simulate(model; n_initial, max_time)`, and return the compartment counts at `max_time` rather than a line list:

- [`MeanFieldProcess`](@ref) integrates the deterministic SIR equations. It has no randomness: every call returns the same path.
- [`LinearNoiseProcess`](@ref) follows the same mean path and adds a Gaussian covariance of the susceptible and infectious counts around it.
- [`HybridProcess`](@ref) runs the exact simulation while fewer than `threshold` people are infectious, then switches to an approximation, keeping the counts.

They take the removal rate γ (`removal_rate`) rather than a recovery `Transition`. An exponential infectious period of mean 5 days is γ = 0.2. Here they are set up alongside an exact model with the same β and γ:

```@example homogeneous
N = 5000
exact = ModelSpec(HomogeneousProcess(; transmission_rate = 0.4, population_size = N);
    progression = [Transition(:recovered; rate = 0.2, terminal = true)])
mean_field = MeanFieldProcess(;
    population_size = N, transmission_rate = 0.4, removal_rate = 0.2)
linear_noise = LinearNoiseProcess(;
    population_size = N, transmission_rate = 0.4, removal_rate = 0.2)
hybrid = HybridProcess(exact, mean_field; threshold = 50)

mf_state = simulate(mean_field; n_initial = 5, max_time = 20.0)
```

The state holds `S`, `I` and `R` as continuous numbers, and `t`. The number ever infected is `N - S`. Compare it with 200 exact runs stopped at day 20, for 5 index cases and for 100. The linear-noise standard deviation of the infected count is read from the covariance of `(S, I)`, and the hybrid model, which is still exact if it has not reached its threshold by day 20, is run with the same seeds:

```@example homogeneous
# A hybrid run that has not reached the threshold by `max_time` is still exact
infected(h) = h.phase == :approx ? N - h.approx.S : count(is_infected, h.exact.individuals)

function compare(n_initial; max_time = 20.0)
    exact_runs = [simulate(exact; n_initial, max_time, rng = StableRNG(s))
                  for s in 1:200]
    exact_cases = [count(is_infected, r.individuals) for r in exact_runs]
    hybrid_cases = [infected(simulate(hybrid; n_initial, max_time, rng = StableRNG(s)))
                    for s in 1:200]
    mf = simulate(mean_field; n_initial, max_time)
    ln = simulate(linear_noise; n_initial, max_time)
    return (n_initial = n_initial,
        exact_mean = round(mean(exact_cases), digits = 0),
        exact_sd = round(std(exact_cases), digits = 0),
        hybrid_mean = round(mean(hybrid_cases), digits = 0),
        mean_field = round(N - mf.S, digits = 0),
        linear_noise_sd = round(sqrt(ln.covariance[1, 1]), digits = 0))
end

[compare(5), compare(100)]
```

With 100 index cases the mean-field count and the exact mean agree closely, and the linear-noise standard deviation matches the exact one. With 5 the mean-field count is too high, by about 6% here, and the standard deviation is too wide, 295 against 252 for the exact runs. Chance decides how quickly a small outbreak takes off, and the deterministic path cannot represent that. The hybrid model runs the early, chance-dominated stage exactly and lands closer (460), but not on the exact mean, because with a threshold of 50 infectious people it hands over while chance still affects the timing. A higher threshold moves it closer to the exact simulation at more cost.

The final size is much less sensitive to that early randomness, because it is set by R0 and the population:

```@example homogeneous
final_exact = [simulate(exact; n_initial = 5, rng = StableRNG(s)).cumulative_cases
               for s in 1:100]
final_mf = N - simulate(mean_field; n_initial = 5, max_time = 200.0).S

println("Exact mean final size: ", round(mean(final_exact), digits = 0))
println("Mean-field final size: ", round(final_mf, digits = 0))
```

As a guide, use the mean-field model for a quick deterministic trajectory or final size when the outbreak starts with many infections. Use the linear-noise model when you need the spread around that path, as in a Kalman filter. Use the hybrid model when the outbreak starts from a few cases and the population is large, so that chance in the early stage matters but running the whole outbreak exactly is too slow. Use the exact model when the population is small, or when you need a line list or interventions that act on individuals. The approximations track only the compartment counts, and none of them returns individuals.

For a population with several mixing groups, [`MixingMeanFieldProcess`](@ref) and [`MixingLinearNoiseProcess`](@ref) are the equivalents of the first two for a [`MixingProcess`](@ref) `ModelSpec`, and `HybridProcess` pairs the exact mixing model with either. See [Age- or group-structured mixing in a closed population](@ref).

## A branching process by generation

The same idea applies to a [`BranchingProcess`](@ref) in a finite population. [`simulate_by_generation`](@ref) draws each generation's total number of cases at once from the counts in the previous one, rather than person by person, so its cost depends on the number of generations and not on the number of cases. Its interface mirrors `simulate`'s. Here is a negative binomial offspring distribution with mean 2 in a population of 200,000, simulated for 30 generations from 10 index cases. The exact simulation stops at `max_cases` (10,000 by default), so the limit is raised above the population:

```@example homogeneous
bp = BranchingProcess(NegBin(2.0, 0.5); population_size = 200_000)

by_generation = [simulate_by_generation(bp; n_initial = 10, max_generations = 30,
                     rng = StableRNG(s)).cumulative_cases for s in 1:20]
individual = [simulate(bp; n_initial = 10, max_generations = 30, max_cases = 500_000,
                  rng = StableRNG(s)).cumulative_cases for s in 1:20]

println("By generation: ", round(mean(by_generation)), " (sd ", round(std(by_generation)), ")")
println("Individual:    ", round(mean(individual)), " (sd ", round(std(individual)), ")")
```

The two agree on the final size and its spread. The generation-based run keeps no individuals, so there is no line list, and individual-level interventions such as isolation or contact tracing do not apply. The [`simulate_by_generation`](@ref) docstring lists what does: immunity at the start, the depletion rule within a generation and vaccination campaigns. Choose it for large, otherwise unstructured outbreaks where you need counts, and the exact simulation when you need who infected whom.

## Structured mixing

`HomogeneousProcess` assumes everyone mixes with everyone else at the same
rate. For transmission that differs between age groups or other groups, see
[Multi-type models](multi-type.md); for contact networks and households, see
[Network models](networks.md) and [Household models](households.md). When
mixing is uneven, for example age bands, spatial patches or demographic
strata that contact each other at different rates, [`MixingProcess`](@ref)
runs the same simulation over a contact structure instead, with no
simulation code of your own; [Age- or group-structured mixing in a closed
population](@ref) works through an age-structured example.

!!! note "How the simulation works"
    The simulation gives the exact stochastic SIR final-size distribution in
    continuous time, with an infection time for every case.
