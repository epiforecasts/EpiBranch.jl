`loglikelihood(data, model)` refused any model with leaky isolation
(`post_isolation_transmission > 0`), because the pairwise likelihood could
only exclude a removal's stretches from the exposure entirely, which rules
the infector out rather than reducing the hazard over the leak. A new trait,
`removal_block_probability`, gives the fraction of hazard a removal blocks
over its stretches (`1`, the default, for a complete removal); below `1`, the
likelihood scales the hazard over those stretches by the surviving fraction
instead of excluding it. `Isolation` declares its own fraction from
`post_isolation_transmission`, so it is now always compatible with the
likelihood.

`RingVaccination`'s `onward_efficacy` stays refused, since its per-contact
block on the vaccinated parent's own transmission, which can also wane over
time, needs its own treatment rather than this one.
