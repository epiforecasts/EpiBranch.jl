On the continuous-time (Sellke) models, a finite `isolation_duration` or
`ContactTracing`'s `Quarantine` `duration` closed the infectious window for
good at the removal's own start, even when its release fell well before the
rest of the infectious period: a case isolated for a week part-way through a
month of infectiousness stayed out of transmission for the remaining three
weeks instead of resuming at the release, disagreeing with the generation
engine, which already let it go. `infectious_removal_time` for `Isolation` now
leaves the window open on the case's other removal states when its own
removal is due to lapse, and the release-aware per-contact `competing_risk`
blocks exactly the isolated stretches instead, matching the generation engine.
A removal with no release still closes the window at its own start, as before.
`ContactTracing`'s `Quarantine` behaves the same way, gaining a per-contact
risk of its own so a quarantine with a duration hands the case back once
released, and one without a duration now reduces onward transmission on the
generation engine as well.

Because the window stays open across a removal that lapses, a blocked contact
proposal asks the pair's kernel for a later one, which a kernel whose support
ends inside the window has unboundedly many of. The race now ends such a pair
where the block is certain to cover the rest of that support, and goes on
drawing where it lapses inside it, rather than refusing to sample. A truncated
or otherwise bounded generation interval therefore runs under a finite
duration, where it raised an error about the remaining integrated hazard. A
release is read only from a component declaring `EpiBranch.binding_release`,
which the built-in removals do and a `Scheduled` that can close does not, so a
certain block from anything else still raises rather than quietly ending a
pair.

The structured pairwise likelihood follows the same rule, taking every
isolated stretch out of each pair's exposure, so a finite duration can be
fitted on household and network data as well as simulated. A host quarantined,
released, and isolated again later keeps both stretches:
`EpiBranch.record_removal!` holds the history that one start and one release
cannot, and a removal written outside the package records its own stretches
and names the key with `EpiBranch.removal_gap_host_times`, which
`household_infections` and `network_infections` record in the layer's
`host_times`.
