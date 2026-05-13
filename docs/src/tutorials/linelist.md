# Line lists and contacts

`linelist(state)` is a pure projection of `state` into a DataFrame.
All randomness (reporting timing, hospitalisation probability, outcome
draws, demographic sampling) happens during [`simulate`](@ref) via
attributes and [`AbstractClinicalTransition`](@ref)s. The line list
just reads what's there.

## Line list

A simulation state is converted to a DataFrame with one row per
infected case using [`linelist`](@ref):

```@example linelist
using EpiBranch
using Distributions
using DataFrames
using Dates
using StableRNGs

model = BranchingProcess(NegBin(1.5, 0.5), LogNormal(1.6, 0.5))

attrs = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

transitions = [
    Reporting(delay = Exponential(3.0)),
    Hospitalisation(delay = Exponential(5.0), probability = 0.2),
    Death(delay = Exponential(14.0), probability = 0.05),
    Recovery(delay = Exponential(14.0)),
]

rng = StableRNG(42)
state = simulate(model;
    condition = 50:200,
    attributes = attrs,
    transitions = transitions,
    sim_opts = SimOpts(max_cases = 200),
    rng = rng,
)

ll = linelist(state; reference_date = Date(2024, 1, 1))
first(ll, 5)
```

Columns appear only when the relevant state keys are set. Drop the
`Hospitalisation` transition and `date_admission` disappears from the
output. Drop `clinical_presentation` and `date_onset`, `date_reporting`,
`date_admission`, `date_outcome` and `outcome` all disappear — the
transitions can't anchor on a missing onset.

## Demographics

Demographics are an attribute, set at simulation time via the
[`demographics`](@ref) builder. They appear in the line list as `age`
and `sex` columns:

```@example linelist
attrs_demo = compose(
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    demographics(age_distribution = Normal(40, 15), prob_female = 0.55),
)

rng = StableRNG(42)
state = simulate(model;
    condition = 50:200,
    attributes = attrs_demo,
    transitions = transitions,
    sim_opts = SimOpts(max_cases = 200),
    rng = rng,
)

ll = linelist(state; reference_date = Date(2024, 1, 1))
println("Age range: $(minimum(ll.age)) - $(maximum(ll.age))")
println("Female: $(round(count(==("female"), ll.sex) / nrow(ll) * 100, digits=1))%")
```

## Age-stratified risks

Age-conditional case fatality risk is expressed as a closure on the
`Death` transition's `probability`, reading `ind.state[:age]`:

```@example linelist
attrs_demo = compose(
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    demographics(age_distribution = Uniform(0, 90)),
)

cfr_by_age = ind -> begin
    age = ind.state[:age]
    age <= 14 ? 0.001 : age <= 64 ? 0.01 : 0.15
end

age_stratified = [
    Death(delay = Exponential(14.0),
        probability = (rng, ind) -> cfr_by_age(ind)),
    Recovery(delay = Exponential(14.0)),
]

rng = StableRNG(42)
state = simulate(model;
    condition = 100:500,
    attributes = attrs_demo,
    transitions = age_stratified,
    sim_opts = SimOpts(max_cases = 500),
    rng = rng,
)

ll = linelist(state; reference_date = Date(2024, 1, 1))

for (lo, hi) in [(0, 14), (15, 64), (65, 90)]
    group = filter(r -> lo <= r.age <= hi, ll)
    n_died = count(==("died"), group.outcome)
    pct = nrow(group) > 0 ? round(n_died / nrow(group) * 100, digits=1) : 0.0
    println("Age $lo-$hi: $(nrow(group)) cases, $n_died deaths ($pct%)")
end
```

The same closure pattern covers risk groups, comorbidities, or any
state field set by your attributes function. See the
[transitions tutorial](transitions.md) for the full menu.

## Contacts table

All contacts (infected and non-infected) are returned by
[`contacts`](@ref), with a `was_case` flag matching the **simulist** R
package output format:

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
    attributes = attrs,
    sim_opts = SimOpts(max_cases = 200),
    rng = rng,
)
println("Outbreak size: $(state.cumulative_cases) (target: 100-150)")
```
