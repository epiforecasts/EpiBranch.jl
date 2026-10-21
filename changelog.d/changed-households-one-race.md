`HouseholdProcess` now simulates every household on one shared clock, whatever
the interventions, stopping rules or kernel. Household simulations run faster,
but a run with a fixed `rng` gives a different outbreak than before.
