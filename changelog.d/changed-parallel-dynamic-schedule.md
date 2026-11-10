`simulate(model, n; parallel = true)` and `containment_probability(...; parallel = true)`
hand outbreaks to threads one at a time rather than splitting them evenly, so a
thread that draws several large outbreaks no longer leaves the others idle.
Results are unchanged for a given seeded `rng`.
