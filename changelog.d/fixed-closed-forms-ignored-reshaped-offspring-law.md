`end_of_outbreak_probability(model, τ)`, `probability_contain(model)` and the
model methods of `proportion_transmission`, `proportion_cases_individual`,
`proportion_cases_offspring` and `proportion_cluster_size` returned the value
for the bare offspring law when a finite `population_size`, an
`Infectiousness` window with `until`, or `transmission_traits` away from their
neutral values had reshaped it. They now check `EpiBranch.offspring_law_unchanged`,
as the chain-size likelihoods do, and raise an `ArgumentError` that points to
simulation.
