A `PairKernel` calendar built from `Steps` no longer returns `NaN` for a
contact interval whose support ends before a later calendar segment starts
(`Uniform`, a `truncated` distribution, or a `_TruncatedHazardKernel`), where
the segment's cumulative hazard subtracted one infinite value from another.
The cumulative hazard is `Inf` there instead, and a draw settles at the
support's top where a zero-multiplier segment holds it, matching a smooth
calendar schedule.
