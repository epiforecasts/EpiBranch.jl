# Line lists and contacts

Simulated epidemiological data can be generated from any branching process simulation.
Output generation is model-agnostic and works with single-type, multi-type,
with or without interventions.

## Line list

A simulation state is converted to a DataFrame with one row per case using [`linelist`](@ref):

```@example linelist
using EpiBranch
using Distributions
using DataFrames
using Dates
using StableRNGs

model = BranchingProcess(NegBin(1.5, 0.5), LogNormal(1.6, 0.5))

rng = StableRNG(42)
state = simulate(model;
    condition = 50:200,
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    sim_opts = SimOpts(max_cases = 200),
    rng = rng,
)

ll = linelist(state;
    reference_date = Date(2024, 1, 1),
    delays = DelayOpts(
        onset_to_reporting = Exponential(3.0),
        onset_to_admission = Exponential(5.0),
        onset_to_outcome = Exponential(14.0),
    ),
    outcomes = OutcomeOpts(prob_hospitalisation = 0.2, prob_death = 0.05),
    rng = StableRNG(99),
)
first(ll, 5)
```

## Demographics

Control age distribution, age range, and sex ratio:

```@example linelist
ll = linelist(state;
    reference_date = Date(2024, 1, 1),
    demographics = DemographicOpts(
        age_distribution = Normal(40, 15),
        age_range = (0, 90),
        prob_female = 0.55,
    ),
    rng = StableRNG(99),
)
println("Age range: $(minimum(ll.age)) - $(maximum(ll.age))")
println("Female: $(round(count(==("female"), ll.sex) / nrow(ll) * 100, digits=1))%")
```

## Age-stratified risks

Provide age-specific case fatality risk as a dictionary mapping
`(lower, upper)` age bounds to risk values:

```@example linelist
age_cfr = Dict((0, 14) => 0.001, (15, 64) => 0.01, (65, 90) => 0.15)

ll = linelist(state;
    reference_date = Date(2024, 1, 1),
    outcomes = OutcomeOpts(age_specific_cfr = age_cfr),
    demographics = DemographicOpts(age_range = (0, 90)),
    rng = StableRNG(99),
)

for (lo, hi) in [(0, 14), (15, 64), (65, 90)]
    group = filter(r -> lo <= r.age <= hi, ll)
    n_died = count(==("died"), group.outcome)
    pct = nrow(group) > 0 ? round(n_died / nrow(group) * 100, digits=1) : 0.0
    println("Age $lo-$hi: $(nrow(group)) cases, $n_died deaths ($pct%)")
end
```

## Contacts table

All contacts (infected and non-infected) are returned by [`contacts`](@ref),
with a `was_case` flag matching the **simulist** R package output format:

```@example linelist
ct = contacts(state; reference_date = Date(2024, 1, 1))
println("Total: $(nrow(ct)), Infected: $(count(ct.was_case)), Not infected: $(count(.!ct.was_case))")
first(ct, 5)
```

## Conditioned simulation

Generate outbreaks of a specific size range:

```@example linelist
rng = StableRNG(42)
state = simulate(model;
    condition = 100:150,
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    sim_opts = SimOpts(max_cases = 200),
    rng = rng,
)
println("Outbreak size: $(state.cumulative_cases) (target: 100-150)")
```
