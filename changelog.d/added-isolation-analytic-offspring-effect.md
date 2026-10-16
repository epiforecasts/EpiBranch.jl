`Isolation` now has a closed-form `analytic_offspring_effect`, so
`extinction_probability`, `probability_contain` and `reproduction_number`
work on a `ModelSpec` that holds it, without falling back to simulation. It
covers a fixed or distributed `onset_to_isolation_delay`, leaky
(`post_isolation_transmission`) and finite-`duration` isolation, and a
generation time linked to the incubation period, such as
`incubation_linked_generation_time`; a custom eligibility rule, or a
`duration` or `test_sensitivity` that depends on the individual, still has
none.

`GeneratingFunctionOffspring` gives any other intervention a way to declare
such a closed-form effect from its own generating function or probability
mass function, for `extinction_probability` and `probability_contain` to use.
