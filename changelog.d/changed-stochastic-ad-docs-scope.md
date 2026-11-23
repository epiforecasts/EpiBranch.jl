The "Gradients of simulated outcomes" section of the inference tutorial no
longer lists wiring StochasticAD.jl into `simulate`'s competing-risk loop as
follow-up work. It says gradients of `simulate` under interventions are out of
scope, and points to the likelihood functions with Enzyme or Mooncake for
gradient-based sampling.
