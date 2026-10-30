The competing-risk draw that decides whether an intervention, host
susceptibility or infector infectiousness blocks a transmission now draws
from `Bernoulli(prob)` rather than comparing a raw uniform against `prob`.
The outcome is unchanged; `prob` is now free to carry a number type a
differentiation backend recognises as a distribution parameter, which
`StochasticAD.jl` needs to estimate a gradient through this discrete choice.
See `docs/src/tutorials/inference.md`, "Gradients of simulated outcomes".
