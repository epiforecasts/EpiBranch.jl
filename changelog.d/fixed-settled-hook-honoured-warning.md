The continuous-time (Sellke) models no longer warn that an intervention's
`apply_post_transmission!` or `keep_active` method "will have no effect" when
the intervention also defines `on_infection_settled!`, the race's own
counterpart to `apply_post_transmission!` (as `trace_contacts!` already is to
`keep_active`). An intervention written for both engines, defining one hook
per engine, was previously reported as unhonoured on `HouseholdProcess` and
`NetworkProcess` even though its settled hook ran for every case. The warning
also now names the specific hook the race skips rather than declaring the
whole intervention inert, since its other hooks may still act.
