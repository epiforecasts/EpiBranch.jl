`EpiBranch.interventions`, `EpiBranch.attributes`, `EpiBranch.observation`,
`EpiBranch.recorder`, `EpiBranch.progression`, `EpiBranch.process` and
`EpiBranch.n_types` now have docstrings, documented under "Writing a
transmission model" in the API reference, alongside `EpiBranch.transmission_time`
and `EpiBranch.kernel_projection`, the dose API
(`EpiBranch.dose_action`, `EpiBranch.record_dose!`, `EpiBranch.record_effect_draws!`,
`EpiBranch.store_draw!`, `EpiBranch.dose_value`) and `HouseholdFinalStatuses`,
which already had docstrings but were missing from the reference page.

The internal `_action_cache` is renamed to `EpiBranch.action_cache` and
declared `public`, since its docstring already addressed extension authors
rather than package internals.
