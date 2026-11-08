`outbreak_state_space(model, observe; dt, n_initial, ...)` treats an outbreak as
an SSMProblems.jl state-space model, defined by a package extension that loads
with SSMProblems. The prior starts the outbreak, and each step continues a copy
of the previous state with `simulate(model, state; max_time)`, so a particle
filter such as GeneralisedFilters.jl's bootstrap filter runs on any model that
supports continuation. The inference page has an example that filters an
outbreak and compares log marginal likelihoods across transmission rates.
