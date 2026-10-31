`MassActionHazard` is a density-dependent `external_hazard` for
`pairwise_surv_loglik`: the mass-action force of infection, read exactly off
an infection layer's own infectious and removal times, so it can be fitted
on its own or added to a household or network contact structure. Like without
an external hazard, it conditions on index cases rather than explaining them,
since nothing is infectious before the first one. `RoutedMassActionHazard` in
EpiNetwork confines it to one named route of a routed network.
