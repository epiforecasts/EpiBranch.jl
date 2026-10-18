`EpiBranch.offspring_law_unchanged` answered `true` for a bare `attributes`
closure, even one that sets `ind.susceptibility` or `ind.infectiousness`
directly — the escape hatch `transmission_traits`'s own docstring recommends
for rules it doesn't cover — so such a closure wrongly kept the analytical
path in `loglikelihood(::ChainSizes, model)`, `chain_size_distribution` and
the closed-form incidence likelihood. A bare function or closure now answers
the conservative `false`, since the trait cannot tell what it does.
`demographics` and `groups` return a dedicated callable type rather than a
closure, so they keep the analytical path.
