The truncated skew-normal distribution returned by
`incubation_linked_generation_time` defined only `rand` and `logpdf`, so
anything else needing it raised a `MethodError`: a calendar, which derives
`cumhazard` from `cdf`, and the offspring form, where `transmission_timing`
calls `truncated`. It now implements the full univariate distribution
interface, including a `quantile` found by Newton's method on `cdf`,
bracketed by bisection as a safeguard.
