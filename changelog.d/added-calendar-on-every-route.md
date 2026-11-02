A calendar schedule now scales transmission on every route, not only
`PairKernel`'s. `HomogeneousProcess` and `MixingProcess` take a `Steps`,
a `Seasonal`, or a function of time as `transmission_rate` or `calendar`,
and `Infectiousness` takes one as `calendar` for a branching process's
offspring windows; each route samples its contacts against it exactly,
as `PairKernel` already did. `loglikelihood(IncidenceCounts, ...)` and
`end_of_outbreak_probability` now throw an `ArgumentError` for a window
with a `calendar`, rather than reading its kernel as if the calendar
were not there.
