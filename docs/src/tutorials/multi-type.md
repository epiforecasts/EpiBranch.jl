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

Column `j` gives the expected offspring of a type-`j` parent. The symmetric
contact matrix above has each row equal to its corresponding column. In this asymmetric
example, adults infect children more often than children infect adults:

```@example multitype
asymmetric = [1.0 1.2;
              0.3 0.9]
println("an adult infects $(asymmetric[1, 2]) children")
println("a child infects $(asymmetric[2, 1]) adults")
println("R for a child: $(sum(asymmetric[:, 1])), for an adult: $(sum(asymmetric[:, 2]))")
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
state = simulate(model; max_cases = 500, rng = rng)

infected = filter(is_infected, state.individuals)
for (i, label) in enumerate(["0-14", "15-64", "65+"])
    n = count(ind -> individual_type(ind) == i, infected)
    println("$label: $n cases")
end
```

## Threshold and extinction probability

The analytical calculations use the matrix and distribution family stored in
the model. `reproduction_number` returns R\*, the dominant eigenvalue of the
next-generation matrix. An outbreak can take off only if R\* exceeds 1.
`extinction_probability` returns one value per type:
the probability that an outbreak seeded by a single case of that type dies out.

```@example multitype
println("R* = $(round(reproduction_number(model), digits = 2))")
q = extinction_probability(model)
for (label, q_j) in zip(["0-14", "15-64", "65+"], q)
    println("  extinction from one $label case: $(round(q_j, digits = 3))")
end
```

A parent draws its total offspring from the distribution family and splits it
across types in proportion to its column of `M`. The extinction probability
accounts for that joint draw. Each simulated run starts from one case of a
random type. The simulated containment probability therefore estimates the
average of the per-type values and agrees with the analytical result.

```@example multitype
results = simulate(model, 1000; max_cases = 200, rng = StableRNG(1))
println("Analytical: $(round(sum(q) / length(q), digits = 3))")
println("Simulated:  $(round(containment_probability(results), digits = 3))")
```

## Custom offspring function

For full control over who infects whom, pass your own function in place of the
matrix. It takes the random number generator and the infector,
`(rng, individual) -> counts`, and returns a vector of whole numbers with one
entry per type: entry `i` is the number of secondary cases of type `i` that this
infector causes. For example, `[3, 1]` means three type-1 cases and one type-2
case. Use `rng` for every random draw, so that runs are reproducible.

Here a type-1 (high-risk) infector causes a highly overdispersed number of
secondary cases, 30% of them high-risk; a type-2 (low-risk) infector causes a
Poisson number with mean 1, 10% of them high-risk. Each case draws its total
once and splits it between the two types with a single binomial draw, so the
two entries always add up to the total:

```@example multitype
function heterogeneous_offspring(rng, individual)
    if individual_type(individual) == 1  # high-risk infector
        n = rand(rng, NegBin(5.0, 0.1))
        h = rand(rng, Binomial(n, 0.3))
    else  # low-risk infector
        n = rand(rng, Poisson(1.0))
        h = rand(rng, Binomial(n, 0.1))
    end
    return [h, n - h]  # [high-risk cases, low-risk cases]
end

model = BranchingProcess(heterogeneous_offspring, Exponential(5.0); n_types = 2)
rng = StableRNG(42)
state = simulate(model; max_cases = 100, rng = rng)
println("Cases: $(state.cumulative_cases)")
```

## Interventions work with multi-type models

Interventions operate on individual state, not types — they work unchanged:

```@example multitype
iso = Isolation(onset_to_isolation_delay = Exponential(2.0), isolation_duration = 7.0)
model = ModelSpec(BranchingProcess(M, R_j -> NegBin(R_j, 0.5), LogNormal(1.6, 0.5));
    interventions = [iso],
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)))

rng = StableRNG(42)
results = simulate(model, 200; max_cases = 500, rng = rng)
println("Containment: $(round(containment_probability(results), digits=3))")
```
