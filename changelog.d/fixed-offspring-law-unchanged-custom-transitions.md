`EpiBranch.offspring_law_unchanged` answered `true` for any progression
transition it had no specific method for, including a custom
`AbstractClinicalTransition`, even though `initialise_individual!` and
`resolve_individual!` receive the case directly and can set
`individual.susceptibility` or `individual.infectiousness` just as a bare
attributes closure can. A custom transition now answers the conservative
`false`, sending `loglikelihood(::ChainSizes, model)`, `chain_size_distribution`
and the closed-form incidence likelihood to simulation unless it opts back in.
`Transition`, `Reporting`, `Hospitalisation`, `Recovery` and `Death` touch
neither field, so each overrides it back to `true`.
