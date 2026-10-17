The incidence time-series likelihood (`loglikelihood(::IncidenceCounts, ...)`
and `incidence_moments`) now takes `reporting_probability` and
`reporting_delay`, for a series of reported cases counted by report date.
Each case is reported independently, and the moments stay exact. Called on a
model, it still refuses a `Reporting` step, whose delay runs from onset, and
points to the keywords instead.
