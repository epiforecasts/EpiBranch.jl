`extinction_probability`, `epidemic_probability`, `probability_contain`,
`reproduction_number`, `proportion_transmission`,
`proportion_cases_individual`, `proportion_cases_offspring`,
`proportion_cluster_size`, `end_of_outbreak_probability` and
`offspring_distribution` read only the offspring distribution of a
`ModelSpec`'s process, so a spec with interventions gave the same answer as
one without them. They now apply each intervention's closed-form effect on the
offspring distribution, declared through the new
`EpiBranch.analytic_offspring_effect`. On a spec with an intervention that
declares none, which for now includes every built-in intervention, they throw
an error that points to simulating the spec. For containment it names
`containment_probability(simulate(spec, n))` for the simulated answer and
`probability_contain(R, k; ind_control, pop_control)` for an approximation.
