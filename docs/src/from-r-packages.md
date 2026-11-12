# Coming from simulist, epichains, ringbp and superspreading

EpiBranch covers what four R packages do: simulist (line lists and contacts), epichains (transmission chains), ringbp (containment under isolation and tracing) and superspreading (closed-form results). The tables below give the EpiBranch call for each function and argument. Argument names differ between versions of the R packages, so check them against the version you use. The recipes after the tables cover three arguments that have an equivalent no other page shows. If Julia syntax is new to you, start with [Julia for R users](julia-for-r-users.md).

## Functions

| R function | EpiBranch |
|---|---|
| simulist `sim_linelist()` | [`simulate`](@ref) a [`ModelSpec`](@ref), then [`linelist`](@ref) |
| simulist `sim_contacts()` | `simulate`, then [`contacts`](@ref) |
| simulist `sim_outbreak()` | one `simulate`, then both `linelist` and `contacts` |
| epichains `simulate_chains()` | `simulate` a [`BranchingProcess`](@ref), then [`chain_statistics`](@ref) for one row per chain |
| epichains `simulate_tree()` | `simulate`, then `linelist` (`parent_id` is the infector) |
| epichains `summary()` | `chain_statistics` |
| epichains `likelihood()` | `loglikelihood(ChainSizes(sizes), offspring)`, see [Chain statistics, likelihood, and fitting](tutorials/chains.md) |
| ringbp `scenario_sim()` | `simulate(model, n)`, or [`containment_probability`](@ref)`(model, n)` for the share of outbreaks that die out |
| ringbp `extinct_prob()` | `containment_probability` |
| superspreading `probability_epidemic()` | [`epidemic_probability`](@ref) for one introduction, or `1 - extinction_probability(R, k)^n` for `n` |
| superspreading `probability_extinct()` | [`extinction_probability`](@ref) |
| superspreading `probability_contain()` | [`probability_contain`](@ref) |
| superspreading `proportion_transmission()` | [`proportion_transmission`](@ref) for the share of transmission from the top fraction of cases, [`proportion_cases_individual`](@ref) or [`proportion_cases_offspring`](@ref) for the fraction of cases behind a share of transmission |
| superspreading `proportion_cluster_size()` | [`proportion_cluster_size`](@ref) |

## Arguments

| R argument | EpiBranch |
|---|---|
| simulist `contact_distribution` | the offspring distribution, the first argument of `BranchingProcess`; see the recipe below for `prob_infection` |
| simulist `infectious_period` | the generation time, the second argument of `BranchingProcess` |
| simulist `prob_infection` | `transmission_traits(susceptibility = p)` |
| simulist `onset_to_hosp`, `hosp_risk` | [`Hospitalisation`](@ref)`(delay = ..., probability = ...)` |
| simulist `onset_to_death`, `hosp_death_risk`, `non_hosp_death_risk` | [`Death`](@ref)`(delay = ..., probability = ...)`; the probability can be a function of the case, see [Clinical transitions](tutorials/transitions.md) |
| simulist `outbreak_start_date` | `linelist(state; reference_date = ...)` |
| simulist `outbreak_size` | `simulate(model; condition = lo:hi, max_cases = hi)` |
| simulist population age | [`demographics`](@ref) |
| epichains `n_chains` | `n_initial` in `simulate(model; n_initial = ...)`, or `simulate(model, n)` for `n` separate runs |
| epichains `offspring_dist` | the first argument of `BranchingProcess`, a Distributions.jl distribution such as `NegBin(R, k)` or `Poisson(R)` |
| epichains `statistic`, `stat_threshold` | `chain_statistics` gives `size` and `length` for each chain; `max_cases` and `max_generations` in `simulate` set the threshold |
| epichains `generation_time` | the second argument of `BranchingProcess` |
| epichains `percent_immune` | see the recipe below |
| ringbp `initial_cases` | `n_initial` |
| ringbp `r0_community`, `disp_community` | `NegBin(R, k)` as the offspring distribution |
| ringbp `r0_isolated`, `disp_isolated` | [`Isolation`](@ref) with `post_isolation_transmission` |
| ringbp `r0_asymptomatic` | see the recipe below |
| ringbp `prop_asymptomatic` | `clinical_presentation(prob_asymptomatic = ...)` |
| ringbp `prop_ascertain` | `ContactTracing(probability = ...)` |
| ringbp onset-to-isolation delay | `Isolation(onset_to_isolation_delay = ...)` |
| ringbp `quarantine` | `ContactTracing(action = Quarantine(duration = ...))` |
| ringbp `cap_max_days`, `cap_cases` | `max_time`, `max_cases` in `simulate` |
| superspreading `num_init_infect` | `n_initial` in `probability_contain` |
| superspreading `ind_control`, `pop_control` | `ind_control`, `pop_control` in `probability_contain` |

## Recipes

The recipes use these packages:

```@example fromr
using EpiBranch
using Distributions
using DataFrames
using Dates
using StableRNGs
```

### Asymptomatic cases that transmit less (`r0_asymptomatic`)

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

An infectiousness of 0.5 halves the number of people a case infects. To match a ringbp `r0_asymptomatic`, use the ratio `r0_asymptomatic / r0_community`; only the mean is matched, and the dispersion stays that of the community offspring distribution.

### Immune individuals (`percent_immune`)

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

### Contacts that are not always infected (`contact_distribution` with `prob_infection`)

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

The mean of the offspring distribution is now the mean number of contacts, not the reproduction number: R is that mean times `prob_infection`. Contacts who were not infected are rows of `contacts` with `infected = false`, and `linelist` lists only the infected ones unless you pass `infected_only = false`.

## Writing line lists to file

`linelist` and `contacts` return DataFrames, which CSV.jl writes:

```julia
using CSV
CSV.write("linelist.csv", linelist(state; reference_date = Date(2024, 1, 1)))
CSV.write("contacts.csv", contacts(state; reference_date = Date(2024, 1, 1)))
```
