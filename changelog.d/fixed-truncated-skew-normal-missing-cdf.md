The truncated skew-normal distribution returned by
`incubation_linked_generation_time` defined only `rand` and `logpdf`.
Anything else that needed it raised a `MethodError`: a calendar derives
`cumhazard` from `cdf`, and `transmission_timing` in the offspring form
calls `truncated`. It now implements the full univariate distribution
interface, including a `quantile` found by Newton's method on `cdf`,
bracketed by bisection as a safeguard.
