`household_offspring` (in `EpiHouseholds`) now leaves each simulated case's
isolated or quarantined stretches out of its community-infectious
person-time, the way the pairwise likelihood already does. A removal that
lapses, such as a finite-duration `Isolation`, does not close the
infectious window; these stretches were previously counted as time spent
making community contacts, and R* under a finite isolation duration
overstated transmission between households.
