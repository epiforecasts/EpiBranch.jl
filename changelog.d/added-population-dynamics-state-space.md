`MeanFieldProcess` and `LinearNoiseProcess` approximate a closed,
homogeneously-mixing SIR population as a state-space model: the deterministic
mean-field ODE, or the same path with a Gaussian process of uncertainty
around it (the linear noise approximation). Both share `simulate(model,
state; max_time)` with the exact simulation. An external Kalman or
particle filter can run on either. `HybridProcess` runs the exact simulation
while the number of currently infectious people is below a threshold, then
switches permanently to the approximation, retaining the compartment counts.
