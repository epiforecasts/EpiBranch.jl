A delay or probability can now also be a `Conditional(dist)`, wrapping a
function that returns a `Distribution` for the case rather than a value.
`EpiBranch.resolve(x, args...)` turns any of a number, a distribution or a
`Conditional` into the `Distribution` it stands for, without drawing from
it; `sample_value` now draws from that result. A plain `(rng, args...)`
function still works, through an adapter that only supports drawing, not
reading the density back.
