`external_hazard` on `HouseholdProcess`, `NetworkProcess` and `RoutedNetwork`,
and in `pairwise_surv_loglik`, accepts a vector with one rate or distribution
per person, or a function of the person's number returning one, so the
community hazard can differ between people, households or trial arms. The
simulation and the likelihood both read it, so `simulate → loglikelihood`
stays an exact round trip.
