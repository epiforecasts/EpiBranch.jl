`external_hazard` on `HouseholdProcess`, `NetworkProcess` and `RoutedNetwork`,
and in `pairwise_surv_loglik`, accepts a vector with one rate or distribution
per person, or a function of the person's number returning one. The community
hazard can then differ between people, households or trial arms. Both the
simulation and the likelihood read it, so `simulate → loglikelihood` stays an
exact round trip.
