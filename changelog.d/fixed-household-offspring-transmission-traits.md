`household_offspring`'s Markov recursion and its exact mean for
within-household person-time assumed unit susceptibility and infectiousness,
so an attribute builder such as `transmission_traits` had no effect on R* when
the shared kernel and infectious window were both exponential, and the exact
mean could override a simulated result. Both paths now fall back to simulation
whenever `EpiBranch.offspring_law_unchanged(spec.attributes)` cannot rule out
non-unit traits.
