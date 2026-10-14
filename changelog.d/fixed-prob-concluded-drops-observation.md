`chain_size_distribution`'s `prob_concluded` mixture now evaluates against
the model's observed chain-size law rather than the bare offspring law with
no observation applied, matching every other route. Previously, a
`MinimumSize` or `PerCaseObservation` model with every cluster marked
concluded gave the likelihood of an unobserved model instead of matching the
closed form.
