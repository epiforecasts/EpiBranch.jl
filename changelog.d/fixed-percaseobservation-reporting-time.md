`PerCaseObservation` now records a case's report time under
`:reporting_time`, the same key the `Reporting` transition sets, instead
of its own `:report_time`. A case that was not reported keeps
`:reporting_time` at `Inf`, matching `Reporting`'s convention, rather
than getting a report time regardless of detection.
`weekly_incidence(state; by = :reporting)` previously found no cases at
all for a simulation using `PerCaseObservation`; it now counts the
reported cases from either route.
