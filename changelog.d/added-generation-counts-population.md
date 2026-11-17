`simulate_by_generation` now tracks the susceptibles of each type, set from
the model's `population` and an initial number `immune` per type, and takes a `depletion` rule.
`StartOfGenerationDepletion` thins the contacts of a type by its susceptible
share at the start of the generation, in one binomial draw capped at the
susceptibles left. This is the standard discrete-generation rule. The default,
`PerContactDepletion`, decides contacts one at a time against each type's own
susceptible share. Where the offspring law has a closed-form sum (`Poisson`, with or
without `multiplier`, `NegBin` and `Binomial`), the total of a generation is
drawn in one step, so the cost of a generation does not depend on the number of
cases. A `generation_length` in days can be recorded in the state. A multi-type
model needs the size of each type's population, `Population(sizes)`.
