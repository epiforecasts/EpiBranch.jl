The mass-action docstrings and the homogeneous-mixing tutorial said
`max_cases`, `max_generations` and other stopping rules do not apply on
`HomogeneousProcess` and `MixingProcess`, and that `simulate` warns if you set
one; only the two keyword *defaults* are off, an explicit cap or a
`stopping_rules` entry is honoured, and the warning is for `RingVaccination`
and `ContactTracing` only, not `MassVaccination` or `GroupVaccination`. The
network and household tutorials said leaky isolation is not recorded in the
pairwise-likelihood data and that fitting it errors; the single-route
likelihood already records and scores it. The "what cannot be fitted yet"
box in the network tutorial named vaccination as unrepresented in the
routed likelihood, which a vaccinated `RoutedNetwork` already fits
correctly; the remaining gap is leaky isolation on a `RoutedNetwork`,
which the routed likelihood silently ignores rather than erroring on. The
`EpiNetwork` and `EpiHouseholds` READMEs are updated to match: a working
example in place of a non-existent keyword, a corrected case-sensitive link,
and the now-stale status sections dropped. The `NetworkProcess` docstring is
moved above the private `_as_adjacency` methods it had been binding to
instead of the struct. The reserved-keys table in the extending reference
gains the `:_ring_hop` and `:_exclusive_draw#N` rows it was missing.
