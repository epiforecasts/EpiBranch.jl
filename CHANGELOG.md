# Changelog

All notable changes to EpiBranch.jl are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `HomogeneousProcess`, a closed, homogeneously-mixing population of fixed size
  simulated by the Sellke threshold construction. Every infectious individual
  exerts the same force of infection on every susceptible, giving the exact
  stochastic SIR final-size law (`R0 = β·E[infectious period]`) and an infection
  time for every case.
- `NetworkProcess` (in `EpiNetwork`) can be fitted as well as simulated.
  `network_infections` reads the infection layer out of a simulation, and
  `loglikelihood(data, model)` scores it with the pairwise survival likelihood,
  whose generative model is the network's continuous-time race. Each node's
  possible infectors are its in-neighbours. Shared, covariate and per-edge
  kernels and a community hazard are supported. Each case's infectious window
  ends where the simulation ends it, including removal by the model's
  interventions such as isolation.
- The pairwise survival likelihood now lives in EpiBranch and works over any
  contact structure. `compile_contact_pairs` enumerates the (susceptible,
  possible infector) rows from a membership vector or an adjacency list into a
  `ContactPairsLayout`, and `pairwise_surv_loglik` evaluates it on any
  `InfectionLayer` subtype. `EpiHouseholds` now uses it through every form and
  keeps its API: `HouseholdPairsLayout` is another name for
  `ContactPairsLayout`. Evaluation is faster, most markedly with a community
  hazard.
- An infection layer can record the end of follow-up, `followup_end` (default
  `Inf`), on `HouseholdInfections`, `NetworkInfections` and a custom
  `InfectionLayer` alike, and `household_infections` and `network_infections`
  take it as a keyword. The pairwise survival likelihood ignores infections and
  exposure after it. An outbreak still going when the data end is then scored as
  observed so far, with a finite value and gradient, and a case still infectious
  at the end of follow-up can keep a removal time of `Inf`.
- `trigger_time(eligibility, infector, contact, state)` gives the trace's
  trigger time for the contact being traced, and `ContactTracing` calls it. A
  custom policy can define it to time the trace from the contact, and combinators
  check wrapped policies that read the contact against it. The
  three-argument form times the policies inside a combinator with their
  three-argument methods and checks them without a contact, so it cannot
  evaluate a combinator wrapping a policy whose `is_eligible` reads the contact.
- `RouteWindow` takes a `traceable` probability (default `1.0`): the chance that
  a case can name a contact made on that route, such as `1.0` for a household
  and something lower for casual community contact. On `RoutedNetwork` a contact
  is traced only if the case names it and the tracing intervention then traces
  it. A contact reachable on several routes is named with the highest of those
  routes' probabilities. Routes left at the default give the same outbreaks as
  before for the same seed.

- `RingVaccination`, `MassVaccination` and `GroupVaccination` gain
  `severity_efficacy`, the probability that a vaccinated individual's own
  disease course is milder once their immunity has developed — e.g. a lower
  chance of death — rather than blocked transmission. It does not gate transmission and so is not one
  of the risks `competing_risk` returns; a clinical transition's
  `probability` reads it, together with the new `immunity_time` accessor, so
  a dose whose immunity has not yet developed by the outcome it would affect
  confers no protection. Recorded per dose as `:severity_efficacy` and
  `:immunity_time`, alongside the existing `:vaccine_efficacy`.

- `RingVaccination` gains `post_exposure_efficacy`, the probability that a dose
  given to an already-exposed contact aborts that infection, which it can do
  whenever immunity arrives before the contact's symptom onset. An aborted
  infection keeps the transmissions made before immunity arrives, makes none
  after it, and has no symptom onset. Its clinical course ends at the abort, so
  no transition (hospitalisation, death or any other outcome) takes effect at or
  after that time, whatever it is timed from. The abort is recorded as
  `:infection_aborted_time`, and the infection still counts as a case. Unlike
  `efficacy`, `post_exposure_efficacy` acts under default tracing without
  quarantine. There a contact is traced once its infector has been isolated,
  which already blocks any later exposure, so `efficacy`, which needs immunity
  before the exposure, acts only with leaky isolation, tracing triggered by
  symptom onset, or rings deeper than one contact. `onward_efficacy` differs in
  reducing each later transmission of a vaccinated contact without ending its
  infection or disease; the two compose.
- `RingVaccination` can schedule a second dose. `dose_delay` sets the number of
  days from the trace to the dose, and `requires_dose` restricts the dose to
  contacts who have received the named earlier dose by then, so its `coverage`
  is the retention between doses. A dose listed before the dose it requires is
  rejected when the `ModelSpec` is built.
- `groups`, an attributes function that labels each individual with a group
  (a village, a health area, a household) under `:group` or an arbitrary key,
  and `GroupVaccination`, which vaccinates every member of a group once any
  case in it meets a [`TraceEligibility`](@ref) policy such as
  `OnLabConfirmation()` — the fallback an outbreak response reaches for when
  no ring can be built. Members are vaccinated at the triggering case's
  eligibility time plus `dose_delay`, whether created before or after the
  trigger, so doses scale with group size where `RingVaccination` doses scale
  with ring size. Listing a `RingVaccination` before a `GroupVaccination` with
  the same `dose_label` makes the group dose a pure fallback: a member the
  ring already reached is skipped.

### Changed

- The fixed-size population pool's mixing structure is now keyed on the
  individual's real attributes: a model names which attributes define mixing via
  `mixing_by` (a tuple of attribute keys, e.g. `(:age_band, :ses)`), and the pool
  buckets susceptibles by the actual values of those attributes. The between-group
  force of infection is a model-supplied `force(type, counts)`, where `type` is a
  susceptible's tuple of attribute values and `counts` maps each mixing type to
  its current infectious number. Structured mixing (age bands, sex, income
  strata, spatial patches) is then written on the extension surface without
  touching the pool primitive. The homogeneous case names no attributes
  (`mixing_by = ()`, one type).
- `NetworkProcess` (in `EpiNetwork`) is now a continuous-time contact-rate
  model. Transmission along each edge is a contact hazard racing the
  infector's recovery or isolation, drawn from a contact-interval kernel,
  replacing the earlier coin-flip-per-edge version. Shortening a case's
  infectious window — through recovery or isolation — now genuinely curtails
  onward spread.
- With a community hazard, the pairwise survival likelihood treats `obs_end` as
  the time community introductions stop, as the household and network
  simulations do. A host accrues community hazard until the earlier of its
  infection and `obs_end`, a host infected after `obs_end` adds no community
  hazard at its infection time, and a host that is never infected is exposed
  over each possible infector's whole infectious window. Household likelihood
  values with a community hazard change as a result. Where `obs_end` stood in
  for the end of follow-up, set `followup_end` instead.
- The continuous-time race behind `NetworkProcess`, `RoutedNetwork` and
  `HouseholdProcess` picks the next case to settle from a binary heap, so a race
  over `n` members with `E` contacts costs O(E log n) where it previously cost
  O(n²). A 100,000-node sparse network now simulates in about a second, down
  from about half a minute. Results for a given seed are unchanged.

### Fixed

- `household_infections` (in `EpiHouseholds`) ends each case's infectious window
  when the model's interventions remove it from transmission, such as by
  isolation or quarantine after tracing, as the simulation does. Fitting an
  outbreak simulated under isolation then recovers the kernel.
- The pairwise survival likelihood evaluated on a compiled layout counts a
  community infection at time 0, as the form without a layout does. An index
  case at time 0 contributes the community hazard at 0. When every hazard at an
  infection time is zero, both forms return `-Inf`.
- The pairwise survival likelihood returns `-Inf` for an infected host that is
  not conditioned on and that no possible infector or community hazard could
  have infected at its infection time, including a host with no possible
  infector at all. A sampler over latent infection times then rejects such
  configurations. The `-Inf` comes with a zero gradient, since the
  configuration is impossible throughout a neighbourhood of the parameters.
- Combined tracing eligibility policies now time the trace from the conditions
  that are met. Each condition, custom policies included, is checked with
  `is_eligible` against the contact being traced. With a custom `Over65` policy,
  `Over65() | OnSymptomOnset()` traces a younger case from onset even if it was
  quarantined earlier. `OnSymptomOnset() | OnLabConfirmation()` traces an
  asymptomatic, lab-confirmed case from its isolation. Previously it gave a
  `NaN` trace time, and quarantining the case's contacts had no effect. An
  `AnyOf` with no condition met, or an `AllOf` with any condition unmet,
  triggers at `Inf` (never). A negation that holds is met with no trigger time
  of its own, and a negated combinator is timed as its De Morgan form. An
  `AllOf` triggers at the latest time among its timed conditions, so
  `OnSymptomOnset() & !OnIsolation()` traces from onset; with no timed condition
  it has no time of its own. An `AnyOf` with a condition that has no time of its
  own has none either, so inside an `AllOf` it sets no time; otherwise it
  triggers at the earliest time among its met conditions. A policy with no time
  of its own traces from the earlier of the infector's isolation, the default
  for `TraceEveryone`, and any of its timed branches, so `OnSymptomOnset() |
  !OnIsolation()` traces a case that is never isolated from onset. Policies that
  are met and rewritten into each other by De Morgan's laws, double negation,
  commutativity, associativity or distributing `&` over `|` outside a negation
  get the same trigger time, as do `TraceNobody() | p` and a met `p`, unless
  `p`'s own trigger time is `NaN`, which a combinator turns into `Inf`. Other
  logically equal policies can differ, because a negation that holds has no time
  of its own and `TraceEveryone()` is timed at isolation: distributing inside a
  negation, absorption by a negation that holds, a condition joined with its
  negation as in `p & (q | !q)`, joining `TraceEveryone()` as in `p &
  TraceEveryone()`, and `!TraceNobody()`, which behaves as `TraceEveryone()`
  only at the top level. The `trigger_time` docstring gives an example of each.
- On the continuous-time models (`HomogeneousProcess`, and `NetworkProcess`,
  `RoutedNetwork` and `HouseholdProcess` in the companion packages), symptom
  onset from `clinical_presentation` is now measured from each case's own
  infection time. Previously it was measured from time 0, because each
  individual is created before it is infected, so onset-triggered isolation
  started too early and simulations overstated its effect.
- `RingVaccination` now gives each dose at the trace, using the `:trace_time`
  that `ContactTracing` records for every traced contact. It used to read the
  contact's isolation state, so under `quarantine_on_trace = false` doses came
  at symptom onset and traced contacts who were asymptomatic got no dose. With
  `:trace_time` recorded at every tracing depth, `linelist` now has a
  `date_trace` column at the default `depth = 1` as well.

## [0.1.0] - 2026-06-16

Initial release. EpiBranch brings together the branching-process cores of
[simulist](https://github.com/epiverse-trace/simulist),
[epichains](https://github.com/epiverse-trace/epichains) and
[ringbp](https://github.com/epiforecasts/ringbp), the analytical superspreading
methods of [superspreading](https://github.com/epiverse-trace/superspreading), and
the post-exposure prophylaxis model of [pepbp](https://github.com/sophiemeakin/pepbp),
in one Julia package with a shared simulation engine. It provides:

- One simulation engine across a branching-process core (`BranchingProcess`)
  and structure-driven companion packages: contact networks (`NetworkProcess`,
  in `EpiNetwork`) and continuous-time household transmission (`HouseholdProcess`,
  in `EpiHouseholds`). Offspring, generation time and clinical progression are
  set on the model.
- Interventions as population-level policies — isolation, contact tracing, and
  ring or mass vaccination — attached to a model and resolved through competing
  risks.
- Per-case attributes (age, type, susceptibility, clinical presentation) and an
  observation layer for under-reporting.
- Line-list and contact-tracing output as DataFrames (`linelist`, `contacts`)
  and transmission-chain statistics (`chain_statistics`).
- Analytical results where closed forms exist: extinction and epidemic
  probability, chain-size and chain-length laws, offspring-distribution helpers,
  end-of-outbreak probability, and superspreading summaries.
- A distribution interface for inference: `chain_size_distribution`,
  `chain_length_distribution` and `offspring_distribution` return `Distribution`s
  that sit directly on the right-hand side of a Turing `~`.
- A pairwise contact-interval likelihood for household data
  (`pairwise_surv_loglik`, `loglikelihood`), whose continuous-time Sellke
  simulator is its exact generative model.
- Containment probability and scenario sweeps for comparing interventions.

[0.1.0]: https://github.com/epiforecasts/EpiBranch.jl/releases/tag/v0.1.0
