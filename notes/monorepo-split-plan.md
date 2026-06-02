# Monorepo-of-packages split plan

Reference layout: SpeedyWeather.jl (5 packages in one repo). End state per issue #104.

## Packages

| Package | Owns | Depends on |
|---|---|---|
| `EpiBranchCore` | Types, abstract types, hook generics, accessors, distribution helpers, attribute builders | (no internal deps) |
| `EpiBranchDynamics` | Engine (`simulate`, `simulate_batch`, `step!(::BranchingProcess)`, `make_contact!`), `BranchingProcess`, stopping rules | `EpiBranchCore` |
| `EpiInterventions` | `Isolation`, `ContactTracing`, vaccinations, `Scheduled`, intervention seam traits | `EpiBranchCore` |
| `EpiTransitions` | `Reporting`, `Hospitalisation`, `Death`, `Recovery` | `EpiBranchCore` |
| `EpiObservation` | `PerCaseObservation`, `Observed`, `ThinnedChainSize` | `EpiBranchCore` |
| `EpiOutput` | `linelist`, `contacts`, `chain_statistics`, summary helpers | `EpiBranchCore`, `EpiTransitions`, `EpiObservation` |
| `EpiAnalytics` | Chain-size distributions, likelihoods, fitting, EOO, `ClusterMixed`, superspreading | `EpiBranchCore`, `EpiBranchDynamics` (for `_sim_loglikelihood` running simulations), `EpiObservation` |
| `EpiBranch` (umbrella) | Re-exports everything; nothing else | All of the above |

**Why Core / Dynamics split**: keeps the protocol layer (types + hook generics + helpers) genuinely thin and separate from the simulator. Cleaner conceptual layering. Leaves room for an alternative engine (e.g. continuous-time SSA) to plug in against the same protocol as a swap for `EpiBranchDynamics`. The cost is one extra package's overhead (Project.toml, version line, CI workflow).

Users who want everything: `using EpiBranch`. Users who want only one slot-in: `using EpiAnalytics` etc.

## Cross-package extension contract

Every cross-package generic is declared in `EpiBranchCore`. Slot-in packages extend them by qualified definition: `function EpiBranchCore.foo(...)`. The Julia package system enforces that no slot-in can reach into another slot-in's internals.

Cross-package generics (declared in `EpiBranchCore` as `function f end` stubs):
- Engine seam: `simulate`, `simulate_batch`, `step!`, `make_contact!`, `draw_offspring`. Default methods in `EpiBranchDynamics`; observation models extend `simulate` for `Observed{...}` from `EpiObservation`.
- `TransmissionModel` interface: `population_size`, `latent_period`, `n_types`, `single_type_offspring`
- Intervention protocol: `initialise_individual!`, `resolve_individual!`, `apply_post_transmission!`, `competing_risk`, `is_active`, `intervention_time`, `reset!`, `required_fields`
- Transition protocol: `is_terminal`, `terminal_event`
- Analytics seam: `chain_size_distribution`

## Layout

```
EpiBranch.jl/
├── EpiBranchCore/
│   ├── Project.toml
│   ├── src/
│   └── test/
├── EpiBranchDynamics/
├── EpiInterventions/
├── EpiTransitions/
├── EpiObservation/
├── EpiOutput/
├── EpiAnalytics/
├── EpiBranch/            # umbrella
└── docs/
```

Each `Project.toml` uses `[sources]` with relative paths for in-tree development, e.g.:

```toml
[sources.EpiBranchCore]
path = "../EpiBranchCore"
```

## Per-PR overhead

- Bump `version = "..."` in each touched package's `Project.toml`. Convention from SpeedyWeather: append `+DEV` for minor, append `-DEV` to a semver-bumped version for breaking.
- CHANGELOG entry (we'd do this regardless).

## CI overhead

Per SpeedyWeather: one workflow file per package, ~1.7KB each, structurally near-identical. About 5–6 workflow files plus shared ones (TagBot, Documenter, enforce_changelog).

## Docs

Single docs site (`docs/`) builds against all packages. Documenter.jl handles cross-module docstrings via the `modules =` arg in `makedocs`. Already set up that way today.

## Phasing

1. **Now**: directory structure + empty Project.tomls + [sources] linkages. No file moves yet.
2. **Then**: move source files into their packages, one package at a time, tests passing at each step.
3. **Then**: per-package test/Project.toml, runtests.jl.
4. **Then**: per-package CI workflows.
5. **Docs**: figure out the multi-package Documenter setup (largely already in place since `modules =` is per-submodule today).

## Notes on naming

Going with `EpiBranchCore` for the protocol+engine package rather than `EpiBranchBase` because:
- "Base" collides with Julia's `Base` and reads awkwardly
- "Core" matches Julia community precedent (SciMLBase aside, most packages use `Core` or domain-specific names)

The umbrella keeps the name `EpiBranch` so `using EpiBranch` still works as users expect.

## Open questions

- Where does `compose` and the attribute builders (`clinical_presentation`, `demographics`, `transmission_traits`) live? Currently in `simulation.jl`. Probably stay in `EpiBranchCore` because they're engine-adjacent and the umbrella user expects them.
- `Scheduled` wrapper — its `apply_post_transmission!` reads `intervention_time`, declared on `EpiBranchCore.AbstractIntervention`. `Scheduled` lives in `EpiInterventions`. Should work.
- The `:reported` key is shared between `Reporting` transition (`EpiTransitions`) and `PerCaseObservation` (`EpiObservation`). They're separate packages but read/write the same key. The shared key is fine; the doc note saying "don't compose both" carries over.
