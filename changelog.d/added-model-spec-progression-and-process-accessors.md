`ModelSpec` gains public `progression` and `process` accessors, alongside the
existing `interventions`, `attributes`, `observation` and `recorder`, so the
full set of layers in a spec can be read or extended from outside the package.
`ModelSpec(spec::ModelSpec; kwargs...)` rebuilds a spec with some layers
replaced, keeping the values already in `spec` for any keyword left out.
