`simulate_by_generation` simulates a `BranchingProcess` with discrete
generations and a finite `population_size`, using case counts. The offspring
for each generation are drawn and thinned in bulk by the susceptible fraction
of the population, without creating each contact as an `Individual`. The resulting `GenerationCounts` continues with
`simulate(model, state; max_generations)`. An optional `multiplier`
distribution, drawn once per generation and shared by every case in it,
scales the mean of their offspring law before each draws from it; with
`Poisson` offspring and a `Gamma` multiplier, this gives the generation's
total a negative-binomial distribution with a shape that stays fixed as the
outbreak grows, instead of increasing with the case count.
