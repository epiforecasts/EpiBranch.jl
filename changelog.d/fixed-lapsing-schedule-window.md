On the continuous-time models, `Scheduled` built with an `end_time`, or
from a predicate, no longer closes the infectious window at a wrapped
removal's own start. The per-contact risk already re-checks the schedule
at every proposal; a wrapped `Isolation` `duration` or `Quarantine`
`duration` now hands the case back on its own terms whatever the schedule
does later. Only a removal with no release of its own still closes the
window there. That's the conservative choice for a block the wrapper has
no way to speak for.
