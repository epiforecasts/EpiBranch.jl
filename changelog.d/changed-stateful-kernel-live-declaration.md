`StatefulKernel` is now always the live pair kernel; the recorded kind
`record_kernel` returns for likelihood evaluation, or that you can build
directly from measured covariates, is the new `RecordedKernel`. A kernel
declares the host record its hazard depends on through
`EpiBranch.watched_records`, and a race redraws a route's pending contacts
only when that route's kernel reports one and it has moved, rather than
inferring this from whether any interventions are present.
`HouseholdProcess` likewise decides whether households race on one shared
clock through a `race_groups(model, kernel)` method dispatched on the
kernel, in place of a flag the engine used to infer the same way. The
per-host times an `InfectionLayer` subtype holds beyond its infectious
windows are now read through `EpiBranch.host_times`, a dispatched accessor
matching `EpiBranch.followup_end`.
