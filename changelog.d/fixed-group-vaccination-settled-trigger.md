On the generation engine, `GroupVaccination` offered a group's doses only
when a case made a contact in it, so a case whose own trigger (such as lab
confirmation) arrived after its last contact, or an index case whose chain
died out before making one, was never offered a dose even though it was
itself eligible. Once a generation's cases are settled, the engine now also
offers every intervention declaring `settles_generation_batch`, as
`GroupVaccination` does, its actions for each case in the batch in turn, with
the clock at that case's own infection time, matching what a network or
household race already does after each case settles.
