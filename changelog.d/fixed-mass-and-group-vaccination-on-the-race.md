`MassVaccination` reached nobody on the network, household, homogeneous or
mixing models, and `GroupVaccination` reached nobody on the homogeneous or
mixing models, because the continuous-time race only ever discovered an
intervention's actions for candidates arising from a model's named contacts.
Two dispatched traits now let an intervention opt out of that requirement:
`EpiBranch.acts_on_population(iv)` offers its actions once, before any case
settles, to the whole population, which `MassVaccination` declares, since its
rollout depends on nothing about who infected whom; `EpiBranch.needs_contacts(iv)`
says whether a `continuous_actions` intervention's candidates need the
model's named contacts at all, which `GroupVaccination` declares `false`,
since its candidates come from a case's group. Both default conservatively to
match every built-in intervention's previous behaviour except the two fixed
here.
