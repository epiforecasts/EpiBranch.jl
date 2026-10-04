**Breaking.** The `individual.state` keys an intervention keeps purely for its
own bookkeeping now start with an underscore, so that `linelist` can drop them
by prefix. Code that reads or writes one of these keys directly — a custom
intervention integrating with `Isolation` or `ContactTracing`, or analysis code
reading them off `state.individuals` — needs the new name, and sees the
default silently otherwise:

- `:isolated_by_isolation` → `:_isolated_by_isolation`
- `:isolation_unrecorded` → `:_isolation_unrecorded`
- `:isolation_time_before_isolation` → `:_isolation_time_before_isolation`
- `:isolation_unrecorded_before_isolation` → `:_isolation_unrecorded_before_isolation`
- `:traced_isolation_time` → `:_traced_isolation_time`
- `:ring_remaining` → `:_ring_remaining`
- `:ring_propagated` → `:_ring_propagated`

A key added from another package carries its tag inside the prefix,
`:_mypkg_budget`, since the underscored names above are reserved along with
the bare ones.
