On the continuous-time models, `Scheduled` built with an `end_time`, or
from a predicate, no longer closes the infectious window at a wrapped
removal's own start. The per-contact risk already re-checks the schedule
at every proposal; a wrapped `Isolation` or `Quarantine` now hands the
case back on the schedule's own terms whatever its `duration` is,
including `Inf`. A schedule that cannot close again still keeps the
wrapped intervention's own answer, since nothing is left to hand the
case back with.
