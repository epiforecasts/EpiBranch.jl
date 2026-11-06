The Markov recursion and the exact mean for within-household person-time in
`household_offspring` assumed unit susceptibility and infectiousness. When the
shared kernel and infectious window were both exponential, an attribute builder
such as `transmission_traits` had no effect on R*, and the exact mean could
override a simulated result. Both paths now fall back to simulation whenever
`EpiBranch.offspring_law_unchanged(spec.attributes)` cannot rule out non-unit
traits.
