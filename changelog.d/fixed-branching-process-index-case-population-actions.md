Index cases in a `BranchingProcess` are now offered the actions of a
population-wide intervention at race start, as the fixed-population
models already do for their seeds. `MassVaccination` and other
interventions declaring `acts_on_population` previously reached the
contacts of a branching process but never its index cases.
