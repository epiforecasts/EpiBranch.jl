`compile_household_pairs` and `loglikelihood(::HouseholdInfections, ::HouseholdProcess)`
take `condition_on = :is_index` or `:earliest`. Without a community hazard,
`:earliest` conditions each household on whichever member currently has the
lowest infection time instead of the recruited index, resolved afresh on
every call — the recruited index need not be the first household member
infected, and augmenting infection times in inference can otherwise move an
earlier infection onto a non-index member and make the configuration
impossible.
