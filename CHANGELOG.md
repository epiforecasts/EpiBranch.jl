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
