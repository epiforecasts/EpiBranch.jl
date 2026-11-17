# Coming from simulist, epichains, ringbp and superspreading

EpiBranch covers what four R packages do: simulist (line lists and contacts), epichains (transmission chains), ringbp (containment under isolation and tracing) and superspreading (closed-form results). Each package has its own section, headed with the release the mapping was checked against. The tables are organised by what you want to model, with the R function or argument in one column and the EpiBranch call in the other. An argument that is not in a table has no row yet. The recipes after the tables cover mappings that no other page shows. If Julia syntax is new to you, start with [Julia for R users](julia-for-r-users.md).

## simulist 0.7.1

| To model | simulist | EpiBranch |
|---|---|---|
| A line list | `sim_linelist()` | [`simulate`](@ref) a [`ModelSpec`](@ref), then [`linelist`](@ref) |
| A contact table | `sim_contacts()` | `simulate`, then [`contacts`](@ref) |
| Both from one outbreak | `sim_outbreak()` | one `simulate`, then both `linelist` and `contacts` |
| Contacts per case | `contact_distribution` | the offspring distribution, the first argument of `BranchingProcess` |
| Contacts that are not always infected | `prob_infection` | `transmission_traits(susceptibility = p)`, see the recipe below |
| The infectious period | `infectious_period` | the generation time, the second argument of `BranchingProcess` |
| Delay from onset to hospitalisation, and the risk of it | `onset_to_hosp`, `hosp_risk` | [`Hospitalisation`](@ref)`(delay = ..., probability = ...)` |
| Delay from onset to death, and the risk of it | `onset_to_death`, `hosp_death_risk`, `non_hosp_death_risk` | [`Death`](@ref)`(delay = ..., probability = ...)`; the probability can be a function of the case, see [Clinical transitions](tutorials/transitions.md) |
| Delay from onset to recovery | `onset_to_recovery` | [`Recovery`](@ref)`(delay = ...)` |
| Delay from onset to reporting | `reporting_delay` | [`Reporting`](@ref)`(delay = ...)` |
| The date of the first case | `outbreak_start_date` | `linelist(state; reference_date = ...)` |
| The size of the outbreak | `outbreak_size` | `simulate(model; condition = lo:hi, max_cases = hi)` |
| Ages in the population | `population_age` | [`demographics`](@ref) |

## epichains 0.1.1

| To model | epichains | EpiBranch |
|---|---|---|
| Transmission chains, with who infected whom | `simulate_chains()` | `simulate` a [`BranchingProcess`](@ref), then `linelist` (`parent_id` is the infector) |
| One row per chain: size and length | `simulate_chain_stats()`, `summary()` | `simulate`, then [`chain_statistics`](@ref) |
| The likelihood of observed chains | `likelihood()` | `loglikelihood(ChainSizes(sizes), offspring)`, see [Chain statistics, likelihood, and fitting](tutorials/chains.md) |
| The number of chains | `n_chains` | `n_initial` in `simulate(model; n_initial = ...)`, or `simulate(model, n)` for `n` separate runs |
| The offspring distribution | `offspring_dist` | the first argument of `BranchingProcess`, a Distributions.jl distribution such as `NegBin(R, k)` or `Poisson(R)` |
| The chain statistic and the threshold at which to stop | `statistic`, `stat_threshold` | `chain_statistics` gives `size` and `length` for each chain; `max_cases` and `max_generations` in `simulate` set the threshold |
| The generation time | `generation_time` | the second argument of `BranchingProcess` |
| An immune share of the population | `percent_immune` | see the recipe below |

## ringbp 1.0.0

ringbp takes its distributions as sampling functions, such as `\(n) rnbinom(n = n, mu = 2.5, size = 0.16)`. EpiBranch takes the distribution itself, here `NegBin(2.5, 0.16)`.

| To model | ringbp | EpiBranch |
|---|---|---|
| One scenario, run `n` times | `scenario_sim(n = ...)` | `simulate(model, n)` |
| The share of outbreaks that die out | `extinct_prob()` | [`containment_probability`](@ref)`(model, n)` |
| The number of initial cases | `scenario_sim(initial_cases = ...)` | `n_initial` |
| Offspring of community cases | `offspring_opts(community = ...)` | `NegBin(R, k)` as the offspring distribution |
| Offspring of isolated cases | `offspring_opts(isolated = ...)` | [`Isolation`](@ref) with `post_isolation_transmission` |
| Asymptomatic cases with a different mean number of offspring | `offspring_opts(asymptomatic = ...)` | `transmission_traits(infectiousness = ...)`, see the first recipe below |
| Asymptomatic cases with their own offspring distribution, with its own dispersion | `offspring_opts(asymptomatic = ...)` | a two-type process, see the second recipe below |
| The incubation period | `delay_opts(incubation_period = ...)` | `clinical_presentation(incubation_period = ...)` |
| A delay from onset to isolation | `delay_opts(onset_to_isolation = ...)` | `Isolation(onset_to_isolation_delay = ...)` |
| The share of cases that are asymptomatic | `event_prob_opts(asymptomatic = ...)` | `clinical_presentation(prob_asymptomatic = ...)` |
| The share of symptomatic cases whose contacts are traced | `event_prob_opts(symptomatic_traced = ...)` | `ContactTracing(probability = ...)` |
| Quarantine of traced contacts | `intervention_opts(quarantine = ...)` | `ContactTracing(action = Quarantine(duration = ...))` |
| Limits on the run | `sim_opts(cap_max_days = ..., cap_cases = ...)` | `max_time`, `max_cases` in `simulate` |

## superspreading 0.4.0

| To model | superspreading | EpiBranch |
|---|---|---|
| The probability of a large outbreak | `probability_epidemic()` | [`epidemic_probability`](@ref) for one introduction, or `1 - extinction_probability(R, k)^n` for `n` |
| The probability that an outbreak dies out | `probability_extinct()` | [`extinction_probability`](@ref) |
| The probability that an outbreak is contained | `probability_contain()` | [`probability_contain`](@ref) |
| The number of introductions | `num_init_infect` | `n_initial` in `probability_contain` |
| Control that acts on individuals, or on the population | `ind_control`, `pop_control` | `ind_control`, `pop_control` in `probability_contain` |
| The share of transmission from the top fraction of cases | `proportion_transmission()` | [`proportion_transmission`](@ref) |
| The fraction of cases behind a share of transmission | `proportion_transmission()` | [`proportion_cases_individual`](@ref) or [`proportion_cases_offspring`](@ref) |
| The share of cases in clusters of a given size | `proportion_cluster_size()` | [`proportion_cluster_size`](@ref) |

## Recipes

The recipes use these packages:

```@example fromr
using EpiBranch
using Distributions
using DataFrames
using Dates
using StableRNGs
```

### Asymptomatic cases that transmit less

In ringbp, `offspring_opts(asymptomatic = ...)` gives asymptomatic cases their own offspring distribution, and it equals the community distribution if left out. In EpiBranch, leave it out in the same way, or scale the mean as below.

List `clinical_presentation` before `transmission_traits`, so that the infectiousness function can ask whether the case is asymptomatic. Here asymptomatic cases infect half as many people as symptomatic ones:

```@example fromr
asymptomatic_spec = ModelSpec(
    BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5));
    interventions = [Isolation(onset_to_isolation_delay = Exponential(3.0), duration = Inf)],
    attributes = [
        clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.3
        ),
        transmission_traits(
            infectiousness = (rng, ind) -> is_asymptomatic(ind) ? 0.5 : 1.0
        ),
    ]
)
containment_probability(asymptomatic_spec, 200; max_cases = 500, rng = StableRNG(1))
```

An infectiousness of 0.5 halves the number of people a case infects. To match an asymptomatic mean `R_a` in ringbp, use the ratio `R_a / R_community`. Only the mean is matched: the dispersion stays that of the community offspring distribution.

### Asymptomatic cases with their own offspring distribution

To give asymptomatic cases their own dispersion as well, make symptom status a type in a two-type process. Column `j` of the next-generation matrix is the offspring of a type-`j` case, split between the two types in the proportions of that column. With the same split in both columns, a share `a` of every case's offspring is asymptomatic. A vector of distributions, one per type, sets the offspring law of each type. Then `prob_asymptomatic` reads the type, so the asymptomatic cases are exactly the second type:

```@example fromr
a = 0.3
R_community, R_asymptomatic = 2.5, 1.0
next_generation = [
    R_community * (1 - a) R_asymptomatic * (1 - a)
    R_community * a R_asymptomatic * a
]
two_type_spec = ModelSpec(
    BranchingProcess(
        next_generation,
        [NegBin(R_community, 0.16), NegBin(R_asymptomatic, 0.5)],
        LogNormal(1.6, 0.5);
        type_labels = ["symptomatic", "asymptomatic"]
    );
    interventions = [Isolation(onset_to_isolation_delay = Exponential(3.0), duration = Inf)],
    attributes = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5),
        prob_asymptomatic = (rng, ind) -> individual_type(ind) == 2 ? 1.0 : 0.0
    )
)
containment_probability(two_type_spec, 200; max_cases = 500, rng = StableRNG(1))
```

Here the community and asymptomatic offspring distributions are `NegBin(2.5, 0.16)` and `NegBin(1.0, 0.5)`. The share `a` plays the part of `event_prob_opts(asymptomatic = ...)` for the cases infected in the simulation. See [Multi-type models](tutorials/multi-type.md) for the matrix layout.

### Immune individuals (epichains `percent_immune`)

Either give every contact a probability `1 - p` of being susceptible:

```@example fromr
p = 0.3
immune_spec = ModelSpec(
    BranchingProcess(Poisson(2.0), Exponential(5.0));
    attributes = transmission_traits(susceptibility = Bernoulli(1 - p))
)
immune_state = simulate(immune_spec; n_initial = 5, max_cases = 500, rng = StableRNG(2))
immune_state.cumulative_cases
```

or vaccinate a share `p` of the population before the outbreak starts, with a [`MassVaccination`](@ref) that is complete at time 0:

```@example fromr
vaccinated_spec = ModelSpec(
    BranchingProcess(Poisson(2.0), Exponential(5.0));
    interventions = [
        MassVaccination(
            coverage = p, efficacy = 1.0, eligibility_time = 0.0, delay_to_immunity = 0.0
        ),
    ]
)
vaccinated_state = simulate(vaccinated_spec; n_initial = 5, max_cases = 500, rng = StableRNG(2))
vaccinated_state.cumulative_cases
```

The first suits a constant, unmodelled level of immunity. The second records who was vaccinated and lets you vary efficacy and timing, as in the [vaccination tutorial](tutorials/vaccination.md). In both, contacts who are protected appear in [`contacts`](@ref) with `infected = false`.

### Contacts that are not always infected (simulist `prob_infection`)

In simulist the offspring are contacts, each infected with probability `prob_infection`. Draw the offspring as contacts, and give each of them that probability as their susceptibility:

```@example fromr
contact_spec = ModelSpec(
    BranchingProcess(Poisson(3.0), Exponential(5.0));
    attributes = [
        clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
        transmission_traits(susceptibility = 0.5),
    ]
)
contact_state = simulate(contact_spec; condition = 50:200, max_cases = 200, rng = StableRNG(3))
contact_table = contacts(contact_state; reference_date = Date(2024, 1, 1))
(contacts = nrow(contact_table), infected = count(contact_table.infected))
```

The mean of the offspring distribution is now the mean number of contacts. R is that mean times `prob_infection`. Contacts who were not infected are rows of `contacts` with `infected = false`. `linelist` lists only the infected ones unless you pass `infected_only = false`.

## Writing line lists to file

`linelist` and `contacts` return DataFrames, which CSV.jl writes:

```julia
using CSV
CSV.write("linelist.csv", linelist(state; reference_date = Date(2024, 1, 1)))
CSV.write("contacts.csv", contacts(state; reference_date = Date(2024, 1, 1)))
```
