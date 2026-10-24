`simulate_once` in `HouseholdProcess` now prepares a vector or function
`external_hazard` once per simulation, rather than once per household in the
race loop. A population of one-person households no longer costs time and
memory quadratic in its size. The equivalent rate or constant hazard was
already cheap; only the per-person vector or function form was affected.
