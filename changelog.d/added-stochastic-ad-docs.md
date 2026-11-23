The inference tutorial has a new section, "Gradients of simulated outcomes",
showing that the offspring distributions already compose with
StochasticAD.jl with no change needed, and the pattern
(`StochasticAD.propagate`) that a discrete decision such as the simulation's
competing-risk draw needs to produce a gradient too, checked against finite
differences on small models. Gradients of `simulate` under interventions are
out of scope; the section points to the likelihood functions with Enzyme or
Mooncake for gradient-based sampling.
