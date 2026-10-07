# Multi-type models

How does an outbreak spread when transmission differs between groups, such as
age groups, risk groups or spatial patches? A multi-type branching process
gives each case a type and describes transmission between types with the
next-generation matrix: the expected number of secondary cases in each group
caused by one case in each group.

## The next-generation matrix

`M[i, j]` is the expected number of type-`i` secondary cases caused by one
type-`j` case. Each column is an infector type and each row an infectee type.
The sum of column `j` is the reproduction number of a type-`j` case.

!!! warning "Columns are infectors, rows are infectees"
    Contact matrices from surveys are often laid out the other way round, with
    rows for the participant. Check the orientation of your matrix, and
    transpose it (`permutedims(M)`) if its rows are the infectors. A matrix
    the wrong way round runs without error and gives the wrong type-specific
    reproduction numbers.

A common way to build the next-generation matrix is contact rate per day ×
probability of transmission per contact × mean infectious period in days. In
Julia, `.*` multiplies element by element, as `*` does on a matrix in R:

```@example multitype
using EpiBranch
using Distributions
using DataFrames
using StableRNGs

# 3 age groups: children, adults, elderly
contacts = [8.0 3.0 1.0;
            3.0 6.0 2.0;
            1.0 2.0 4.0]  # daily contacts between groups
M = contacts .* 0.05 .* 5.0  # transmission probability 0.05, infectious for 5 days

DataFrame(type = ["Children", "Adults", "Elderly"], R = vec(sum(M, dims = 1)))
```

The `R` column is the column sum of `M`: the expected number of secondary
cases, of any age, caused by one case in that age group. Children have the
highest reproduction number because they have the most contacts.

Because this contact matrix is symmetric, it does not show which way round
the indices go. In the next matrix, an adult (column 2) causes 1.2 secondary cases
among children (row 1), but a child (column 1) causes only 0.3 among adults
(row 2):

```@example multitype
asymmetric = [1.0 1.2;
              0.3 0.9]
println("an adult infects $(asymmetric[1, 2]) children")
println("a child infects $(asymmetric[2, 1]) adults")
println("R for a child: $(sum(asymmetric[:, 1])), for an adult: $(sum(asymmetric[:, 2]))")
```

## Simulation

Pass the next-generation matrix, the distribution of the number of secondary
cases per case, and the generation time. The second argument is a function of
a type's reproduction number: `R_j -> NegBin(R_j, 0.5)` reads "for a type
whose column sums to `R_j`, use a negative binomial with mean `R_j` and
dispersion k = 0.5". Options after the semicolon are named arguments.

```@example multitype
model = BranchingProcess(
    M,
    R_j -> NegBin(R_j, 0.5),  # secondary cases: mean R_j, dispersion k = 0.5
    LogNormal(1.6, 0.5);      # generation time, mean about 5.6 days
    type_labels = ["0-14", "15-64", "65+"],
)

rng = StableRNG(42)
state = simulate(model; max_cases = 500, rng = rng)

cases = linelist(state)
combine(groupby(cases, :type), nrow => :cases)
```

`LogNormal(μ, σ)` takes the mean and standard deviation of the logarithm of
the delay, not of the delay itself. The line list numbers types in matrix
order (1 = 0-14, 2 = 15-64, 3 = 65+). The case counts by age group follow the
structure of `M`: children and adults, who have the most contacts, make up
most of the cases.

Each case first draws its total number of secondary cases from the negative
binomial, then divides them between types in proportion to its column of `M`.
This is a modelling assumption: overdispersion applies to a case's total, with
a multinomial split between types given that total.

## Reproduction number and extinction probability

[`reproduction_number`](@ref) returns the reproduction number of the whole
system, the dominant eigenvalue of the next-generation matrix (often written
R\*). An outbreak can take off only if it exceeds 1.
[`extinction_probability`](@ref) returns one value per type: the probability
that transmission from a single index case of that type dies out without
interventions. Both are calculated from the matrix and the distribution of
secondary cases, without simulation.

```@example multitype
println("R* = $(round(reproduction_number(model), digits = 2))")
q = extinction_probability(model)
for (label, q_j) in zip(["0-14", "15-64", "65+"], q)
    println("  extinction from one $label case: $(round(q_j, digits = 3))")
end
```

An outbreak started by an older person is the most likely to die out, because
older people have the fewest contacts.

The closed-form extinction probability can be checked against simulation. The
containment probability is the proportion of simulated outbreaks that end
before reaching the case cap, here 200 cases. Each simulated outbreak starts
from one index case of a random type. The containment probability then
estimates the average of the per-type extinction probabilities:

```@example multitype
results = simulate(model, 1000; max_cases = 200, rng = StableRNG(1))
println("Analytical: $(round(sum(q) / length(q), digits = 3))")
println("Simulated:  $(round(containment_probability(results), digits = 3))")
```

The two agree up to simulation noise and the small chance that an outbreak
passes 200 cases and would still have died out.

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
once and splits it between the two types with a single binomial draw. The two entries
always add up to the total:

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

## Interventions in multi-type models

Interventions apply to every case, whatever its type. Here each symptomatic
case isolates after a delay from symptom onset with mean 2 days
(`Exponential(θ)` has mean θ) and stays isolated for 7 days; the incubation
period, from infection to onset, has a mean of about 5 days:

```@example multitype
iso = Isolation(onset_to_isolation_delay = Exponential(2.0), isolation_duration = 7.0)
model = ModelSpec(BranchingProcess(M, R_j -> NegBin(R_j, 0.5), LogNormal(1.6, 0.5));
    interventions = [iso],
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)))

rng = StableRNG(42)
results = simulate(model, 200; max_cases = 500, rng = rng)
println("Containment: $(round(containment_probability(results), digits=3))")
```

Isolation raises the proportion of outbreaks that are contained above the
extinction probabilities without control.

For control that differs by type, give the delay as a function of the random
number generator and the case, `(rng, ind) -> ...`, that reads the case's type
with `individual_type(ind)`: for example, a shorter delay to isolation for
older people.
