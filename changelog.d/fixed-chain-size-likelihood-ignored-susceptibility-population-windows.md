`loglikelihood(::ChainSizes, model)`, `chain_size_distribution` and the
incidence likelihood took their analytical path from the bare offspring
law whenever it had a closed form, ignoring a finite `population_size`, an
`Infectiousness` window with `until`, and `transmission_traits` away from
its neutral susceptibility or infectiousness of 1 — each of which `simulate`
already honoured, so the two disagreed. They now ask the new
`EpiBranch.offspring_law_unchanged` of the process, the attributes and the
progression first, and fall back to simulation (or, for the incidence
likelihood, raise an `ArgumentError` naming the reason) whenever one of them
says no.
