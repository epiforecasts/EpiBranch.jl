A `Conditional` parameter now works as a transition `delay` or `probability`
and in `transmission_traits`. Before, it failed with a `MethodError` inside
`simulate`. These now read their parameters through `sample_value`, like the
rest of the package.
