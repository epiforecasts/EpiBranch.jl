A `Conditional` parameter now works as a transition `delay` or `probability`
and in `transmission_traits`, which previously failed with a `MethodError`
inside `simulate`. These sites now read their parameters through
`sample_value`, like the rest of the package.
