`MeanFieldProcess` and `LinearNoiseProcess` approximate a closed,
homogeneously-mixing SIR population as a state-space model: the deterministic
mean-field ODE, or the same path with a Gaussian process of uncertainty
around it (the linear noise approximation). Both share `simulate(model,
state; max_time)` with the exact simulation, so an external Kalman or
particle filter can run on either. `HybridProcess` combines exact simulation
with one of them: it runs exact while the number of currently infectious
people is below a threshold, and switches to the approximation once and for
all above it, carrying the compartment counts across.
