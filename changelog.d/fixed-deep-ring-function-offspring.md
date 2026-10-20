A ring of contact tracing deeper than one step now traces the contacts of a
ring member the same way however the offspring is written. Before, where the
offspring or generation time was a function, such as
`(rng, ind) -> rand(rng, NegBin(2.5, 0.5))` or
`incubation_linked_generation_time`, a ring member that was itself an eligible
case had its contacts traced only from its own trigger, so the deeper ring
added nothing there. Its contacts are now traced from when the ring reached it
as well, as they already were with `NegBin(2.5, 0.5)`, and the two give the
same results.
