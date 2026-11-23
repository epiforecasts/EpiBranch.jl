The inference tutorial has a new section, "Gradients of simulated outcomes",
showing that the offspring distributions already compose with
StochasticAD.jl with no change needed, and the pattern
(`StochasticAD.propagate`) that a discrete decision such as the simulation's
competing-risk draw needs to produce a gradient too, checked against finite
differences on small models. The section does not cover gradients of
`simulate` under interventions. For gradient-based sampling it points to the
likelihood functions with Enzyme or Mooncake.
