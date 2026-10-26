`ModelSpec` gains public `progression` and `process` accessors, alongside the
existing `interventions`, `attributes`, `observation` and `recorder`, so a
spec's full set of layers can be read or extended from outside the package.
`ModelSpec(spec::ModelSpec; kwargs...)` rebuilds a spec with some layers
replaced, keeping `spec`'s own values for any keyword left out.
