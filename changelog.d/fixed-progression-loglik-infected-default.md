`progression_loglik` now treats an individual with no `:infected` key as
infected, matching `is_infected`. A hand-built `Individual` holding only
progression event times — an example the docstring itself gives — used
to contribute zero with no error, since the `Individual(; ...)` keyword
constructor sets no `:infected` key.
