`simulate_by_generation` now tracks the susceptibles of each type, set from
`population_size` (or `population_by_type` for a multi-type model) and an
initial number `immune` per type, and takes a `depletion` rule.
`StartOfGenerationDepletion` thins a type's contacts by its susceptible share
at the start of the generation in one binomial draw capped at the susceptibles
left, the standard discrete-generation rule, where the default
`PerContactDepletion` keeps the per-contact rule of the individual-based
simulation. Where the offspring law has a closed-form sum (`Poisson`, with or
without `multiplier`, `NegBin` and `Binomial`), a generation's total is drawn
in one go, so a generation costs the same whatever the number of cases.
With a `generation_length` in days, `interventions` act between generations:
a `GroupVaccination`, optionally inside `Scheduled` or `CapacityConstrained`
with its budget schedules, removes susceptibles of each type at the rate of
doses times efficacy, only the susceptible share of a type's doses reaching
susceptibles. A multi-type model with a finite `population_size` now needs
`population_by_type`, since its types no longer share one population.
