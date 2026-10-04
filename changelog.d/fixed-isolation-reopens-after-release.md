On the continuous-time (Sellke) models, a finite `isolation_duration` or
`ContactTracing`'s `Quarantine` `duration` closed the infectious window for
good at the removal's own start, even when its release fell well before the
rest of the infectious period: a case isolated for a week part-way through a
month of infectiousness stayed out of transmission for the remaining three
weeks instead of resuming at the release, disagreeing with the generation
engine, which already let it go. `infectious_removal_time` for `Isolation` now
leaves the window open on the case's other removal states when its own
removal is due to lapse, and the release-aware per-contact `competing_risk`
blocks exactly the isolated interval instead, matching the generation engine.
A removal with no release (the default `Inf` duration) still closes the
window at its own start, as before. `ContactTracing`'s `Quarantine` keeps its
previous behaviour, having no per-contact risk of its own to fall back on;
compose `Isolation` alongside it for a quarantine that should let a case go
once released.
