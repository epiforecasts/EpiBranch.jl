# Multi-type models

Multi-type branching processes are supported, where individuals
belong to types (age groups, risk groups, spatial patches) and transmission
between types is governed by an offspring matrix.

## Offspring matrix

`M[i,j]` is the expected number of type-`i` offspring from a type-`j`
parent — the next-generation matrix.

```@example multitype
using EpiBranch
using Distributions
using DataFrames
using StableRNGs

# 3 age groups: children, adults, elderly
# Contact matrix × transmission probability × infectious duration
contacts = [8.0 3.0 1.0;
            3.0 6.0 2.0;
            1.0 2.0 4.0]
M = contacts .* 0.05 .* 5.0  # q = 0.05, D = 5 days

println("Type-specific R:")
for (i, label) in enumerate(["Children", "Adults", "Elderly"])
    println("  $label: $(sum(M[:, i]))")
end
```

## Simulation

Pass the matrix and a function that maps each type's R to an offspring distribution:

```@example multitype
model = BranchingProcess(
    M,
    R_j -> NegBin(R_j, 0.5),    # distribution family (user's choice)
    LogNormal(1.6, 0.5);         # generation time
    type_labels = ["0-14", "15-64", "65+"],
)

rng = StableRNG(42)
state = simulate(model; sim_opts = SimOpts(max_cases = 500), rng = rng)

infected = filter(is_infected, state.individuals)
for (i, label) in enumerate(["0-14", "15-64", "65+"])
    n = count(ind -> individual_type(ind) == i, infected)
    println("$label: $n cases")
end
```

## Custom offspring function

For full control, pass a function `(rng, parent_type) → Vector{Int}`:

```@example multitype
function heterogeneous_offspring(rng, parent_type)
    if parent_type == 1  # high-risk type
        n = rand(rng, NegBin(5.0, 0.1))
        return [rand(rng, Binomial(n, 0.3)), n - rand(rng, Binomial(n, 0.3))]
    else  # low-risk type
        n = rand(rng, Poisson(1.0))
        return [rand(rng, Binomial(n, 0.1)), n - rand(rng, Binomial(n, 0.1))]
    end
end

model = BranchingProcess(heterogeneous_offspring, Exponential(5.0); n_types = 2)
rng = StableRNG(42)
state = simulate(model; sim_opts = SimOpts(max_cases = 100), rng = rng)
println("Cases: $(state.cumulative_cases)")
```

## Interventions compose with multi-type

Interventions operate on individual state, not types — they work unchanged:

```@example multitype
model = BranchingProcess(M, R_j -> NegBin(R_j, 0.5), LogNormal(1.6, 0.5))
iso = Isolation(delay = Exponential(2.0))

rng = StableRNG(42)
results = simulate_batch(model, 200;
    interventions = [iso],
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    sim_opts = SimOpts(max_cases = 500),
    rng = rng,
)
println("Containment: $(round(containment_probability(results), digits=3))")
```
