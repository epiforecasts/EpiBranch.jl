`simulate_by_generation` simulates a `BranchingProcess` with discrete
generations and a finite `population_size` by generation, as case counts
rather than individuals: each generation's offspring is drawn and thinned by
the population's susceptible fraction in bulk, without creating a contact as
an `Individual`. The resulting `GenerationCounts` continues with
`simulate(model, state; max_generations)`. An optional `multiplier`
distribution, drawn once per generation and shared by every case in it,
scales the mean of their offspring law before each draws from it; with
`Poisson` offspring and a `Gamma` multiplier, this gives the generation's
total a negative-binomial distribution whose shape stays fixed as the
outbreak grows, rather than one that widens with its case count.
