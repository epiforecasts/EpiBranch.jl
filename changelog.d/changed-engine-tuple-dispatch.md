The continuous-time race now takes the `progression` and `interventions` of the
model as `Tuple`s once per run and walks them with the hook calls resolved when
the method is compiled. Before, it dispatched at run time on a `Vector` of an
abstract element type for every settled case. A branching process with a
four-step progression runs about 30% faster, and one with isolation and contact
tracing about 15%. `state.transitions` and `ModelSpec.interventions` stay
`Vector`s, and no extension point changes. The loops over proposed contacts
iterate the same `Tuple`, but `risk_applies` filters per route at run time
rather than at compile time.
