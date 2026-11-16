`simulate_by_generation` now tracks the susceptibles of each type, set from
the model's `population` and an initial number `immune` per type, and takes a `depletion` rule.
The default, `PerContactDepletion`, decides contacts one at a time against
each type's own susceptible share, drawn exactly without visiting every
contact. Where the offspring law is a mixed Poisson (`Poisson`, with or without
`multiplier`, and `NegBin`), a generation is one binomial draw per type, so its
cost does not depend on the number of cases; for any other law the cost grows
with the number of new infections alone. A `generation_length` in days can be recorded in the state. A multi-type
model needs the size of each type's population, `Population(sizes)`.
