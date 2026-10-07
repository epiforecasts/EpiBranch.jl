A model with its own simulation loop — the continuous-time household and
network processes, and the fixed-size Sellke pool — now reaches the package
through public names instead of underscored internals. `EpiBranch.simulate_once`
is the run seam `simulate` dispatches to, with the `condition`/`max_attempts`
retry handled once in `simulate` rather than repeated in every model.
`EpiBranch.sellke_race!` and `EpiBranch.sellke_pool!` resolve a run's `max_time`
and reconcile its aggregate bookkeeping, so a model driving either construction
no longer touches `_max_time` or `_reconcile_sellke_bookkeeping!` directly.
`EpiBranch.infectious_from` derives a progression's infectious window, and
`EpiBranch.INTERVENTION_REMOVAL` is now declared `public`. A new
`MixingProcess` model runs structured mixing (age bands, spatial patches, …)
on the same Sellke pool as `HomogeneousProcess`, reachable entirely through
these public names.
