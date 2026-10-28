`record_dose!` takes the simulation state in place of a random number
generator, drawing from `state.rng`, and a dose it records counts against any
`CapacityConstrained` budget under its dose label. Outside a run, pass
anything with an `rng` field, such as `(; rng = StableRNG(1))`.
