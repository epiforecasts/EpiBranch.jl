A `PairKernel` calendar built from `Steps` no longer returns `NaN` when the
support of a contact interval (`Uniform`, a `truncated` distribution, or a
`_TruncatedHazardKernel`) ends before a later calendar segment starts,
because computing the cumulative hazard of that segment subtracted one
infinite value from another. The cumulative hazard is `Inf` there instead,
and a draw settles at the top of the support where a zero-multiplier segment
holds it, matching a smooth calendar schedule.
