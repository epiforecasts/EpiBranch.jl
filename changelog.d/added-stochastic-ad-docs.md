The inference tutorial has a new section, "Gradients of simulated outcomes",
showing that the offspring distributions already compose with
StochasticAD.jl with no change needed, and the pattern
(`StochasticAD.propagate`) that a discrete decision such as the engine's
competing-risk draw needs to produce a gradient too, checked against finite
differences on small models. Wiring this into `simulate`'s own
short-circuiting risk loop, so a gradient-based sampler can fit a
chain-size likelihood under interventions, is follow-up work.
