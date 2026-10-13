Simulating a `BranchingProcess` is about 1.6 times faster. Reading a time out
of `Individual.state` returned `Any` to the compiler, so the engine's hot path
boxed a float on every window it opened and every transition it settled; the
reads now assert the element type, as the accessors in `state_accessors.jl`
already did.
