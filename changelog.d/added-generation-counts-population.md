`simulate_by_generation` now tracks the susceptibles of each type, set from
`population_size` (or `population_by_type` for a multi-type model) and an
initial number `immune` per type, and takes a `depletion` rule.
`StartOfGenerationDepletion` thins the contacts of a type by its susceptible
share at the start of the generation, in one binomial draw capped at the
susceptibles left. This is the standard discrete-generation rule. The default,
`PerContactDepletion`, keeps the per-contact rule of the individual-based
simulation. Where the offspring law has a closed-form sum (`Poisson`, with or
without `multiplier`, `NegBin` and `Binomial`), the total of a generation is
drawn in one step, so the cost of a generation does not depend on the number of
cases. A `generation_length` in days can be recorded in the state. A multi-type
model with a finite `population_size` now needs `population_by_type`, because
its types no longer share one population.
