The generation engine calls the intervention hooks once per case, as the
continuous-time models do. For each case of the previous generation, in order
of infection time, it calls every intervention's `trace_contacts!` with the
contacts the case made, then every intervention's `intervention_actions`, then
`keep_active`. These hooks see the clock at the case's infection time and its
place in the case count, as `resolve_individual!` does. `ContactTracing`,
`RingVaccination`, `GroupVaccination` and `MassVaccination` no longer have
`apply_post_transmission!` methods, and an intervention written outside the
package with only `trace_contacts!` or `intervention_actions` now acts on the
generation engine too.

Tracing and vaccination on a branching process give statistically the same
results as before. Seeded runs differ, because the draws come in a different
order. Four things change:

- A `Scheduled` gate on tracing is evaluated at the infector's infection time.
  It used to read the latest infection time of the previous generation, and
  tracing under `start_time` now starts later than it did. The continuous-time models
  already did this.
- `CapacityConstrained` ranks and admits one case's candidates at a time, with
  the budget read at that case's infection time. It used to rank a whole
  generation's candidates together and read the budget at the latest
  infection time so far. That could admit more doses in a generation. The
  continuous-time models already admitted one case at a time. Because a later
  generation can hold cases infected before some of the previous one's, with
  `carry_over = false` a period's budget now counts only doses admitted within
  that period, and with `carry_over = true` every dose admitted so far still
  counts against the allowance.
- Every intervention's `trace_contacts!` for a case runs before any
  intervention's actions for it. `RingVaccination` listed before
  `ContactTracing` now doses the contacts traced for the same case.
- On a model whose `contacts_of` lets several cases reach the same node in one
  generation, each of those cases traces it, as on the continuous-time models.
  Only the case with the earliest exposure used to. `keep_active` is asked
  with one case's contacts, before they are decided infected or not.
