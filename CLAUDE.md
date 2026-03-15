# simulist.jl — Consolidated Branching Process Framework for Epidemiology

This is a monorepo that may host multiple Julia packages. You are building a framework that unifies the functionality of three R packages from the epiverse-trace ecosystem:

- **simulist** (https://github.com/epiverse-trace/simulist) — simulates line list and contact tracing data from branching process outbreaks
- **epichains** (https://github.com/epiverse-trace/epichains) — simulates and analyses transmission chain statistics using branching process models
- **ringbp** (https://github.com/epiforecasts/ringbp) — simulates outbreaks with non-pharmaceutical interventions (isolation, contact tracing, ring vaccination) to assess containment probability

## Context

These three R packages share a common core: stochastic branching process simulation of infectious disease outbreaks. In R they are separate CRAN packages; in Julia they should be a single package with a shared simulation engine and composable components. The user (Sebastian Funk, LSHTM) is one of the authors of ringbp and a key contributor to the epiverse ecosystem.

This is the **starting point** in a broader plan to build a Julia epidemiology ecosystem. It was chosen first because simulation-heavy code benefits most from Julia's performance, and the branching process core can be shared across use cases.

Related projects in `~/code/`:
- `EpiNow2.jl` — Rt estimation (long-term flagship, deprioritised)
- `outbreak-analytics-jl` — course materials and supporting packages
- `messy-line-lists` — typo challenge / data quality tools (may fold into this repo)

## Monorepo structure

This repo may contain multiple packages (e.g., BranchingProcess.jl, Interventions.jl, LineList.jl) or a single unified package — this is an open design decision. The prompt below covers the full scope; the package boundary question should be resolved early based on how cleanly the components separate.

## What to build

### Core simulation engine

Design a model-agnostic branching process simulation engine using Julia's multiple dispatch. The engine should:

1. **Step individuals through infection events**: each infected individual generates secondary cases drawn from an offspring distribution, with timing governed by delay distributions (generation time, incubation period, etc.)
2. **Support multiple transmission models** via dispatch: standard branching process, density-dependent susceptible adjustment (DSA), and future extensions — all sharing the same stepping interface
3. **Track individual-level state**: infection time, symptom onset time, reporting time, contact history, vaccination status, isolation status, and any other state the intervention layer needs
4. **Generate line list output**: a DataFrame with one row per case, columns for dates (infection, onset, hospitalisation, death/recovery), demographics, and contact tracing links — matching the output format of simulist

### Intervention layer

This is a critical design challenge. Interventions in ringbp operate at the **population policy level** (e.g., "isolate confirmed cases within 2 days", "trace and test contacts of confirmed cases", "vaccinate contacts in a ring"), not at the individual agent level. This means:

1. **Interventions are functions of global/population state**, not individual decision rules. They take the current epidemic state and modify transmission probabilities, delays, or individual statuses
2. **The simulation engine must expose a clear state interface** that interventions can read and modify: individual infection times, contact history, vaccination status, population-level summaries (cumulative cases, active cases, etc.)
3. **Interventions should be composable**: you should be able to layer isolation + contact tracing + ring vaccination and have them interact correctly
4. **Time-dependent policies**: interventions can switch on/off or change parameters over time (e.g., "start contact tracing on day 14")

### Agents.jl decision

Evaluate carefully whether to use Agents.jl or build a lighter-weight simulation loop:

- **Against Agents.jl**: the intervention layer sits above individual agent rules, which is not Agents.jl's natural model. Agents.jl adds overhead (scheduler, space, model object) that may not be needed for a branching process where individuals don't move or interact spatially
- **For Agents.jl**: mature ecosystem, visualisation tools, parameter scanning
- **Recommendation**: start with a minimal custom stepping function. If spatial structure or complex agent interactions become needed later, consider wrapping in Agents.jl then

### Inference / analytical solutions

Where analytical solutions exist (e.g., extinction probability from offspring distribution parameters, expected chain size), prefer them over simulation. Use simulation for quantities that don't have closed-form solutions (e.g., containment probability under complex intervention combinations).

The superspreading R package (https://github.com/epiverse-trace/superspreading) provides analytical functions for offspring distribution estimation and SSE probability — these should be incorporated as analytical methods alongside the simulation engine.

## Key design principles

1. **Multiple dispatch for transmission models**: define an abstract `TransmissionModel` type with concrete subtypes (`BranchingProcess`, `DensityDependent`, etc.). The `simulate` function dispatches on this type
2. **Distributions from Distributions.jl**: use standard `Distributions.jl` types for offspring distributions, delay distributions, etc. No custom distribution wrappers
3. **DataFrames output**: line lists and chain statistics returned as DataFrames, matching epidemiological conventions (one row per case or per contact pair)
4. **Composable interventions**: interventions are callable structs or functions with a standard signature. Stack them in a vector; the engine applies them in order each timestep
5. **Reproducibility**: explicit RNG threading for reproducible parallel simulations

## Research before coding

Before writing code, thoroughly investigate:

1. **ringbp repository** (https://github.com/epiforecasts/ringbp): read the README, all open issues (especially discussions about flexible simulation engines), closed issues, and the intervention interface. Also check linked repos: contact network extension and PEP (post-exposure prophylaxis) variant
2. **Ring vaccination Ebola code**: historical code that ringbp built on, for context on intervention modelling
3. **simulist internals**: understand how it generates line lists, handles age structure, time-varying CFR, and contact tracing data
4. **epichains**: understand its chain statistics (size, length) and how it relates to the offspring distribution
5. **Existing Julia solutions**: check what JuliaEpi, Epirecipes, and other Julia epi packages already provide — avoid duplicating existing work

## Package structure suggestion

```
OutbreakSim.jl/  (or EpiBranch.jl — name TBD)
├── src/
│   ├── OutbreakSim.jl          # module definition
│   ├── types.jl                # TransmissionModel, Individual, Intervention abstract types
│   ├── models/
│   │   ├── branching_process.jl  # standard BP
│   │   └── density_dependent.jl  # DSA
│   ├── simulation.jl           # core simulation loop
│   ├── interventions/
│   │   ├── isolation.jl
│   │   ├── contact_tracing.jl
│   │   ├── vaccination.jl      # ring vaccination, mass vaccination
│   │   └── compose.jl          # intervention stacking
│   ├── linelist.jl             # line list generation from simulation output
│   ├── chains.jl               # chain statistics (size, length, etc.)
│   ├── analytical.jl           # closed-form solutions (extinction prob, etc.)
│   ├── offspring.jl            # offspring distribution fitting (from superspreading)
│   └── summary.jl              # containment probability, summary statistics
├── test/
└── Project.toml
```

## Style and conventions

- Use British English in all documentation and comments ("modelling", "behaviour", etc.)
- Prefer explicit types over duck typing for the core simulation types
- Document all public functions with docstrings
- Write tests alongside implementation
- Keep the package self-contained — no dependency on EpiAware.jl (it will be redesigned)
- PrimaryCensored.jl is available and stable if needed for delay distribution censoring
