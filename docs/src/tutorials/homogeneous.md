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

!!! warning "Contact tracing and vaccination do not apply"
    Contact tracing needs named contacts, and the built-in vaccinations dose
    people as transmission reaches them; a well-mixed population has
    neither. [`ContactTracing`](@ref), [`MassVaccination`](@ref) and
    [`GroupVaccination`](@ref) are not applied to a `HomogeneousProcess`;
    `simulate` warns if you include one. Isolation, reduced susceptibility or
    infectiousness, and events in the natural history all apply. To model
    vaccine protection, lower susceptibility with `transmission_traits`.

!!! note "Stopping the simulation"
    A `HomogeneousProcess` runs until no one is infectious or until
    `max_time`, whichever comes first. `max_cases`, `max_generations` and
    stopping rules other than [`MaxTime`](@ref) do not apply, and `simulate`
    warns if you set one.

## Structured mixing

`HomogeneousProcess` assumes everyone mixes with everyone else at the same
rate. For transmission that differs between age groups or other groups, see
[Multi-type models](multi-type.md); for contact networks and households, see
[Network models](networks.md) and [Household models](households.md). Writing
your own mixing structure for a closed population is covered in
[Extending EpiBranch](extending.md).

!!! note "How the simulation works"
    The simulation gives the exact stochastic SIR final-size distribution in
    continuous time, with an infection time for every case.
