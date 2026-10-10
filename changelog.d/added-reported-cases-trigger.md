`Scheduled`'s `start_after_cases` counts every infection, including one
never reported or reported only after the date it opens on, which is not
what a policy triggered by surveillance needs. `Scheduled(iv; start_after =
ReportedCases(n))` opens once `n` cases are reported as of the simulation
clock instead. `start_after` takes any `AbstractTrigger`; `Infections(n)`
is the existing count, now also available as a trigger, and a trigger
written outside the package works without editing `Scheduled`.
