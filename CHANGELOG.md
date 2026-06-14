# Changelog

All notable changes to EpiBranch.jl are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First public release. EpiBranch brings the branching-process cores of the R
packages simulist, epichains and ringbp together in one Julia package with a
shared simulation engine.

### Added

- Branching-process and contact-network simulation through one engine
  (`BranchingProcess`, `NetworkProcess`). Offspring, generation time and
  clinical progression are set on the model.
- Interventions as population-level policies — isolation, contact tracing, and
  ring or mass vaccination — attached to a model and resolved by the engine
  through competing risks.
- Per-case attributes (age, type, susceptibility, clinical presentation) and an
  observation layer for under-reporting.
- Line-list and contact-tracing output as DataFrames (`linelist`, `contacts`)
  and transmission-chain statistics (`chain_statistics`).
- Analytical results where closed forms exist: extinction and epidemic
  probability, chain-size and chain-length laws, offspring-distribution
  helpers, end-of-outbreak probability, and superspreading summaries.
- Distribution interface for inference: `chain_size_distribution`,
  `chain_length_distribution` and `offspring_distribution` return
  `Distribution`s that sit directly on the right-hand side of a Turing `~`.
- Containment probability and scenario sweeps for comparing interventions.

### Changed

### Deprecated

### Removed

### Fixed

[Unreleased]: https://github.com/epiforecasts/EpiBranch.jl
