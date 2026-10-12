A custom `AbstractVaccination` can now record a dose, and compose the shared
per-dose draws its own effects need, without any name a package extender was
not meant to call:

- `EpiBranch.dose_action(v, ind, time)` is the `InterventionAction` the
  built-ins use to give a dose; a custom vaccination's own
  `intervention_actions` method returns one per candidate, so `Scheduled`
  and `CapacityConstrained` admit it exactly as they admit a
  `MassVaccination` dose.
- `EpiBranch.record_dose!(v, ind, time, rng)` records a dose directly,
  outside the action protocol, skipping an already-vaccinated individual and
  reconsidering a post-exposure abort against the dose just given — what
  `RingVaccination` and `GroupVaccination` already got through their own
  actions.
- `EpiBranch.record_effect_draws!` is now the public hook, with its no-op
  default, for drawing a per-dose effect only some vaccinations have (such
  as `RingVaccination`'s `post_exposure_efficacy`); `EpiBranch.store_draw!`
  and `EpiBranch.dose_value` are the helpers a method of it calls to store
  and read such a draw.
- `EpiBranch.vaccine_effect` is now declared public.
- `vaccine_efficacy(ind; dose_label = :default)` is a new exported accessor,
  next to `severity_efficacy`, reading the efficacy stored on an individual
  for a given dose label.
