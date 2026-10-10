`HomogeneousProcess` runs on the same continuous-time race as the household and
network models, as mass action in which every pair meets at `β/N`, and the
separate Sellke pool is gone. The process and its final-size law are unchanged,
but a seeded run draws a different trajectory than before. Every case now has
the infector whose contact reached it first, and `on_infection_settled!` is
called on it as on the other continuous-time models.

Structured mixing in a closed population takes a pair rate,
`rate(infector_group, target_group)`, where the extending guide's recipe took a
`force(group, counts)`. A contact matrix gives the rate directly; a force that
is not linear in the number infectious has no pairwise form. Risks that read the
infector, such as a leaky isolation, are no longer refused under structured
mixing, because every contact comes from a named infector.

The race also runs faster on dense contact structures: a complete graph of 1,000
nodes takes about 9 ms a run, where it took about 130 ms.
