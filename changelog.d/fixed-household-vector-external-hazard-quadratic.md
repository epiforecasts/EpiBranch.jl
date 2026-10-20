`HouseholdProcess`'s `simulate_once` now prepares a vector or function
`external_hazard` once per simulation instead of once per household's race,
so a population of one-person households no longer costs time and memory
quadratic in its size. The equivalent rate or constant hazard was already
cheap; only the per-person vector or function form was affected.
