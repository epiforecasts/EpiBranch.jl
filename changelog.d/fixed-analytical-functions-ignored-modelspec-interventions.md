`extinction_probability`, `epidemic_probability`, `probability_contain`,
`reproduction_number`, `proportion_transmission` and `offspring_distribution`
read only the offspring law of a `ModelSpec`'s process, so a spec carrying
interventions got the same answer as the same spec without them. They now
refuse on a spec whose interventions have no declared closed-form effect on
the offspring law (`EpiBranch.analytic_offspring_effect`, a new dispatched
trait with no useful default), naming `probability_contain(R, k; ind_control,
pop_control)` for an approximation and
`containment_probability(simulate(spec, n))` for the simulated answer,
instead of silently ignoring the interventions.
