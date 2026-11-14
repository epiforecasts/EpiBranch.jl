`scenario_sweep(build, rows; summary, n_sim, rng, ...)` runs a scenario once per
row of parameters, such as posterior or literature draws. `build` turns a row
into any `ModelSpec` or process, so the progression, the process type and
parameters inside delay distributions can vary, and `summary` reduces each
row's simulated states to any outcome, containment probability by default.
Passing a function as `rng` gives each row a fresh generator, for example
`() -> CommonRandomNumbers(1)`, so the rows are compared on the same random
numbers.
