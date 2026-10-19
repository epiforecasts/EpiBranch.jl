`HouseholdProcess` gains `household_members`, `external_hazard` and
`observation_end` accessors, alongside the existing `household_sizes`, so code
that works with a process from outside the package does not need to reach
into its fields.
