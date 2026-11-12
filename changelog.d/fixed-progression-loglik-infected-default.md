`progression_loglik` now treats an individual with no `:infected` key as
infected, matching `is_infected`. A hand-built `Individual` holding only
progression event times, as in the docstring example, used
to contribute zero with no error, because the `Individual(; ...)` keyword
constructor sets no `:infected` key.
