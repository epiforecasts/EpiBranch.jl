`external_hazard` on `HouseholdProcess`, `NetworkProcess` and `RoutedNetwork`
accepts a vector with one entry per person, or a function of a person's
number, in addition to the shared rate or distribution already accepted. This
lets the community hazard differ between people, households or clusters, as a
cluster-randomised or case-triggered intervention study needs, since the
indirect effect of such an intervention shows up as a difference in infection
from outside the household or network. Simulation and `pairwise_surv_loglik`
both read whichever form is given, so `simulate → loglikelihood` stays an
exact round trip for a model whose arms differ in community infection.
