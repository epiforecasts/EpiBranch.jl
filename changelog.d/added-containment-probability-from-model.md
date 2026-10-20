`containment_probability(model, n; kwargs...)` simulates `n` outbreaks with the
keywords of `simulate(model, n)` and returns the share contained. It holds one
outbreak in memory at a time;
`containment_probability(simulate(model, n))` holds all `n`. `scenario_sweep` now uses it.
