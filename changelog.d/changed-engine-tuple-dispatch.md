The continuous-time race now takes the model's `progression` and `interventions`
as `Tuple`s once per run and walks them with the hook calls resolved when the
method is compiled, rather than dispatching at run time on a `Vector` of an
abstract element type for every settled case. A branching process with a
four-step progression runs about 30% faster, and one with isolation and contact
tracing about 15%. `state.transitions` and `ModelSpec.interventions` stay
`Vector`s, and no extension point changes. The hooks called for each proposed
contact still read the interventions from a `Vector`.
