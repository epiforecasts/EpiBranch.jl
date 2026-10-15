`Scheduled` isolation and contact tracing are now checked against the day
the isolation or trace itself falls on, the way a scheduled vaccination's
dose already is, rather than the case's or infector's infection time. A case
infected before a policy's `start_time` was previously never isolated, even
when its own isolation would have fallen inside the window; it now is.
`start_after`, including the `ReportedCases` and `Infections` triggers, is
checked at that later time too, so a trigger's reading of the population can
differ slightly from a plain case count where it did not before. This
changes which cases a `Scheduled` isolation or tracing policy reaches, and
with it the results of simulations using one.
