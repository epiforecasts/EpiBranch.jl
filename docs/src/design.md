# EpiBranch.jl — Design

## Core idea

The offspring draw is completely decoupled from timing and interventions. The competing risks framework on the transmission hazard is connected naturally to survival analysis (Kenah's pairwise survival analysis, dynamic survival analysis).

## Architecture

### Top-level diagram

The package separates into a process side, an observation side, a
data layer, and an inference dispatch surface. `Observed` combines
the first two; `loglikelihood(data, model)` is the single inference
verb.

```
PROCESS                                OBSERVATION
─────────                              ─────────
TransmissionModel (abstract)           ObservationModel (abstract)
└── BranchingProcess(offspring,        └── PerCaseObservation(
                     generation_time)        detection_prob, delay)
└── Observed{P, O}(process, observation) ← combines both
    (also <: TransmissionModel)

                       │
                       │ simulate(model)
                       ▼
                 SimulationState
                 (linelist, chain stats, ...)

                       │
DATA  ─────────────────┴─────────────►  INFERENCE
─────                                   ─────────
OffspringCounts                         loglikelihood(data, model)
ChainSizes
ChainLengths

ORTHOGONAL
─────────
AbstractIntervention (Isolation, ContactTracing, RingVaccination, Scheduled)
    — passed to simulate as a vector; modifies state during step!

EXTENSION POINTS (for users adding their own pieces)
─────────
- Custom transmission model: subtype TransmissionModel,
  define step!, single_type_offspring, chain_size_distribution
- Custom observation model: subtype ObservationModel,
  define loglikelihood(::DataType, ::Observed{<:Any, <:YourObs})
  and chain_size_distribution(::Observed{<:Any, <:YourObs}) if analytical
- Custom data type: define a struct + a loglikelihood method
- Custom intervention: subtype AbstractIntervention,
  define apply_post_transmission!, initialise_individual!, ...
```

The four design principles are documented separately
([Design principles](principles.md)) and are the basis on which this
shape is judged.

### Three separated stages

#### 1. Offspring draw

Pure branching process, no time, no interventions.

For a parent of type `j`, contacts are drawn from an offspring distribution. This is the classical branching process: a count from a distribution (single-type) or a vector of counts per type (multi-type). The mean (R) and overdispersion (k) come from the offspring distribution. If a contact matrix is provided, the mixing pattern across types is determined by it.

#### 2. Timing

Generation times are assigned independently to each contact.

Each contact is given a generation time sampled from a distribution (either fixed or derived from the parent's incubation period, as in the ringbp-style model). This generation time is the *potential* time of transmission — the infectiousness profile, or h(t) in survival analysis terms.

#### 3. Competing risks

Which contacts are actually infected?

Each contact is resolved independently:
- Was the parent isolated before this contact's generation time? (hazard truncation)
- Is the contact susceptible? (vaccination, prior immunity, population depletion)
- Is the parent still infectious? (infectiousness modifier)
- For leaky isolation: does residual transmission succeed?

Contacts that fail any of these checks are not infected but are still stored. They represent contacts that were made but did not result in transmission. Intervention effort (contacts traced, vaccines administered, tests used) is tracked this way.

### Why this separation matters

Because the offspring distribution is a pure probabilistic object, it can be analysed with standard tools: extinction probability from the dominant eigenvalue, chain size distributions, analytical likelihoods. None of this depends on timing or interventions.

At the intervention stage, the timing layer is modified, not the offspring layer. The hazard is truncated by isolation. The truncation point is shifted earlier by contact tracing. Susceptibility is modified by vaccination. All of these are competing risks on the same set of potential contacts.

The generation time CDF evaluated at the intervention time *is* the survival function. The same mathematical objects appear in simulation and in Kenah's pairwise likelihood framework. Intervention effectiveness can be estimated from observed generation times using the same quantities used in simulation.

All contacts are stored, not just successful infections. This gives the contacts table (with `infected` flag) and intervention effort tracking, without any additional bookkeeping.

## Extension model

New behaviour is added by defining a new type and a method, not by growing options on an existing struct. A user wanting a variant of an existing component should be able to write a small struct plus one or two methods, with no edits to the package source and no copy-pasting of existing function bodies.

When a field on a public struct is `Union{T, U}` whose union members trigger different runtime branches, or a `Symbol` that switches behaviour inside a function body, or a `Bool` that selects between policies, that is a signal the seam is in the wrong place — turn it into a dispatched-on type instead.

This applies on every extension axis:

- **Transmission models**: spatial structure, network structure, and immunity dynamics enter through new `TransmissionModel` subtypes (or composable wrappers around existing ones), not flags on `BranchingProcess`.
- **Interventions are orchestrators of smaller dispatched pieces**. An intervention struct (e.g. `ContactTracing`, `Isolation`) is a thin shell that wires together independently dispatched components — eligibility, fire-rate, delay, effect. Each component is itself a type with a defined method (e.g. `is_eligible(scope, parent, contact, state)`). Users extend behaviour by writing new component types. The intervention body itself contains no hardcoded policy branching.
- **Composition works at two levels**: between interventions (the stack passed to `simulate`) and within each intervention (its dispatched pieces).
- **Output, observation, and outcome rules** follow the same shape: mortality, hospitalisation, reporting probability, stopping conditions, line-list columns are typed objects with methods, not closed sets of fields.

The test of correctness for any component is: can a plausible new variant be added without editing the component's source? If not, the component is doing too much; lift the varying part into a dispatched-on trait.

### Naming convention for abstract types

The package follows the same role-vs-thing split that `Base` and `Distributions.jl` use:

- **Prefix with `Abstract`** when the type names *what's being abstracted* and the bare noun would read as a concrete value. The main-axis abstractions a user composes between fall here: `AbstractIntervention`, `AbstractVaccination`, `AbstractEffectMode`, `AbstractClinicalTransition`, `AbstractStoppingRule`. Same convention as `AbstractFloat`, `AbstractArray`, `AbstractDict`.
- **Drop the prefix** when the type already names a *role* or *concept*: `TransmissionModel`, `ObservationModel`, and the per-intervention seam traits (`IsolationEligibility`, `TraceEligibility`, `TraceRate`, `TraceDelay`, `TraceAction`). Same convention as `Number`, `Function`, `Distribution`.

Per-intervention seam traits should be qualified with the intervention's name (e.g. `TraceEligibility`, not `Eligibility`) so the type reads unambiguously in user code and doesn't collide with other interventions' versions of the same trait.

## Individual state

```
Individual (struct)
├── Core (used by the engine)
│   ├── id, parent_id, generation, chain_id    # transmission tree
│   ├── infection_time                          # timing
│   ├── susceptibility, infectiousness          # universal modifiers
│   └── secondary_case_ids                      # filled during simulation
│
└── state::Dict{Symbol, Any}                    # everything else
```

Only `susceptibility`, `infectiousness`, and `infection_time` are read by the engine. The dict carries everything else and is the deliberate extension hatch — interventions, attributes builders, transitions, and observation models each own a small set of keys. The engine never inspects them.

Why a dict instead of typed slots on `Individual`: a typed core would couple the struct to every intervention's state shape, and would break `RingVaccination` / `MassVaccination`'s dose-label namespacing (the keys are `:vaccinated_<label>`, so they cannot collapse to a single typed field). The dict keeps `Individual` independent of which interventions a user composes.

### Accessor convention

Read via a one-line wrapper that pins the result type and supplies the safe default — `onset_time(ind) = get(ind.state, :onset_time, NaN)::Float64`. New code should not call `get(ind.state, …)` directly; add an accessor in `src/state_accessors.jl` instead.

### Reserved keys

The keys below are reserved by the package. Custom interventions should pick names that do not collide.

| Key | Type | Default | Owner | When set |
|---|---|---|---|---|
| `:infected` | `Bool` | `true` | Engine | Competing-risks resolution |
| `:type` | `Int` | `1` | Engine (multi-type) | Contact creation |
| `:onset_time` | `Float64` | `NaN` | `clinical_presentation` | Init |
| `:asymptomatic` | `Bool` | `false` | `clinical_presentation` | Init |
| `:age` | `Real` | — | `demographics` | Init |
| `:sex` | `Symbol` | — | `demographics` | Init |
| `:risk_group` | `Symbol` | — | `demographics` | Init |
| `:isolated` | `Bool` | `false` | `Isolation` | `resolve_individual!` |
| `:isolation_time` | `Float64` | `Inf` | `Isolation` | `resolve_individual!` |
| `:test_positive` | `Bool` | `false` | `Isolation` | `resolve_individual!` |
| `:traced` | `Bool` | `false` | `ContactTracing` | `apply_post_transmission!` |
| `:quarantined` | `Bool` | `false` | `ContactTracing` | `apply_post_transmission!` |
| `:traced_isolation_time` | `Float64` | `Inf` | `ContactTracing` → `Isolation` | Internal handoff |
| `:vaccinated[_<label>]` | `Bool` | `false` | `AbstractVaccination` | Init / `apply_post_transmission!` |
| `:vaccination_time[_<label>]` | `Float64` | `Inf` | `AbstractVaccination` | `apply_post_transmission!` |
| `:vaccine_efficacy[_<label>]` | `Float64` | — | `AbstractVaccination` | `apply_post_transmission!` |
| `:reporting_time` | `Float64` | `Inf` | `Reporting` transition | `resolve_individual!` |
| `:admitted` | `Bool` | `false` | `Hospitalisation` transition | `resolve_individual!` |
| `:admission_time` | `Float64` | `Inf` | `Hospitalisation` transition | `resolve_individual!` |
| `:death_candidate_time` | `Float64` | `Inf` | `Outcome` transition | `resolve_individual!` |
| `:recovery_candidate_time` | `Float64` | `Inf` | `Outcome` transition | `resolve_individual!` |
| `:outcome` | `Symbol` | — | `Outcome` transition | `resolve_individual!` (terminal) |
| `:outcome_time` | `Float64` | — | `Outcome` transition | `resolve_individual!` (terminal) |
| `:reported` | `Bool` | `false` | `PerCaseObservation` *or* `Reporting` transition | Post-simulation projection / `resolve_individual!` |
| `:report_time` | `Float64` | — | `PerCaseObservation` | Post-simulation projection |
| `:cluster_theta` | `Float64` | — | `ClusterMixed` analytics | First simulation read |

The vaccination keys are namespaced by `dose_label` — the default label writes to plain `:vaccinated` / `:vaccination_time` / `:vaccine_efficacy`; any other label suffixes the key (so `dose_label = :boost` writes to `:vaccinated_boost` etc.). This lets multi-dose schedules compose without colliding.

`:reported` is shared between the `Reporting` clinical transition (which sets it from a probability gate) and `PerCaseObservation` (which sets it post-simulation from a detection-probability draw). Composing both in the same simulation is not supported — they will overwrite each other.

## Intervention interface

Four hooks, all optional. See the [Extending guide](@ref "Extending EpiBranch") for the full input/output contract, ordering guarantees, and a `BorderClosure` worked example.

- `initialise_individual!(intervention, individual, state)` — set up fields on a new contact
- `resolve_individual!(intervention, individual, state)` — determine intervention state before transmission (e.g. compute isolation time from onset + delay)
- `apply_post_transmission!(intervention, state, new_contacts)` — act on contacts after creation (e.g. contact tracing, ring vaccination). All contacts, infected and non-infected, are passed.
- `competing_risk(intervention, parent, contact, state)` — return the `Risk` (or `NTuple{N, Risk}`) this intervention contributes against the parent → contact transmission, or `nothing`.

Interventions are stacked in a vector and applied in order. Each intervention has its own fields on the individual and declares what fields it requires.

### Time-based scheduling

Time-based scheduling is provided by the `Scheduled` wrapper. Individual interventions do not carry their own `start_time` — they are wrapped with `Scheduled(iv; start_time = ...)` (and/or `end_time`, `start_after_cases`, or a custom predicate). `Scheduled` enforces start times at two levels:

- A population-level gate (`is_active`) skips the inner `resolve_individual!` and `apply_post_transmission!` until the condition is met.
- Once active, `Scheduled` performs individual-level reset: after each per-individual hook, if the individual's `intervention_time` falls before `start_time`, it calls `reset!` on the inner intervention.

Each schedulable intervention declares:

- `intervention_time(intervention, individual)` — the time at which the effect occurs (e.g. isolation time, trace time). Default: `-Inf` (always applies).
- `reset!(intervention, individual)` — reverts all state changes made by the intervention. Default: no-op.

The check is on *action time*, not infection time. An individual infected before the policy start can still be affected if their computed action time (e.g. `onset + delay`) falls after `start_time`. This is the correct competing-risk interpretation: the testing infrastructure must be available at the time the individual would be tested.

## Transmission modifiers

All interventions ultimately map onto two numbers:

- `susceptibility` (0--1): probability of being infected given exposure. Reduced by vaccination, prior immunity, or population depletion.
- `infectiousness` (0--1): modifier on onward transmission. Reduced by isolation (via hazard truncation), treatment, or asymptomatic status.

For a contact to be infected, it must survive the parent's infectiousness check AND the contact's susceptibility check AND the timing check (generation time vs isolation time).

## Multi-type branching processes

Multiple types (age groups, risk groups, spatial patches) are supported in the offspring draw. For a parent of type `j`, offspring counts per type are drawn from a joint distribution. The mixing pattern comes from a contact matrix; the count distribution comes from the offspring distribution family.

Each contact is allocated to a type, stored as `:type` in its state dict. Interventions and output are unchanged — they operate on individual-level state, not types.

## Composing analytical methods

The simulation side is extended by implementing the intervention protocol. The analytical side uses a different pattern. Each extension is a Julia type with a method on `chain_size_distribution` or `loglikelihood`, or both. There is no new abstract type beyond the existing `TransmissionModel`; multiple dispatch handles the rest.

### Two kinds of extension

Offspring specifications replace what a branching process draws per individual. An example is `ClusterMixed(build, mixing)`, which lets the offspring distribution's parameters vary from chain to chain. These are stored in `BranchingProcess.offspring` and participate via:

- `_draw_offspring(rng, offspring_spec, individual, state)` for simulation
- `chain_size_distribution(offspring_spec)` for analytics (returns a distribution)

Observation models capture how the latent process generates data. They subtype `ObservationModel` and combine with a process model via `Observed(process, observation)`. `PerCaseObservation(detection_prob, delay)` is the current example. They participate via:

- `chain_size_distribution(::Observed{<:Any, <:YourObservation})` returning a distribution that transforms the wrapped model's chain size distribution (e.g. `ThinnedChainSize(chain_size_distribution(m.process), ρ)`)
- `loglikelihood(data, ::Observed)`, which usually just routes through `chain_size_distribution`

### Why observation models return distributions

`chain_size_distribution(::Observed{<:Any, <:PerCaseObservation})` returns `ThinnedChainSize(base, ρ)`. The result is itself a `DiscreteUnivariateDistribution` with its own `logpdf`, so a second observation type (e.g. `CensoredAtSize`) can wrap a model already wrapped in another `Observed`, and the chain-size distribution nests through. Composition is then just nesting the distributions.

If instead the thinning were computed inside `loglikelihood(::Observed)`, adding a second observation type would require working out its joint likelihood with `PerCaseObservation` by hand, and every new pair would need the same treatment. Returning a distribution means each observation can be written once.

### Closed forms as dispatch optimisations

Some combinations of offspring and mixing have closed forms. Poisson offspring with a Gamma-distributed rate gives `PoissonGammaChainSize`, which is the `gborel` likelihood from `epichains`. These plug in by specialising `chain_size_distribution`:

```julia
chain_size_distribution(o::ClusterMixed) = ChainSizeMixture(o.build, o.mixing)
chain_size_distribution(o::ClusterMixed{PoissonFamily, <:Gamma}) = PoissonGammaChainSize(...)
```

The generic case uses adaptive Gauss-Kronrod quadrature via `ChainSizeMixture`. When a closed form applies, dispatch uses it without the user having to ask.

## Simulation, mutation, and automatic differentiation

The simulation engine uses in-place mutation: `step!` appends individuals, updates case counts, and modifies individual state via interventions. This is deliberate — branching process simulations grow an unbounded tree, and copying the full state at every generation would be prohibitively expensive.

### Analytical likelihoods

`loglikelihood(ChainSizes(data), Poisson(R))` and similar analytical methods are deterministic scalar functions of the parameters. They work with any AD backend (ForwardDiff, Enzyme, Mooncake, ReverseDiff) and with gradient-based samplers like NUTS.

### Simulation-based likelihoods

`loglikelihood(ChainSizes(data), model; interventions=...)` runs stochastic simulations internally. Because the simulation draws random numbers, the output is a noisy estimate of the true likelihood, not a smooth function of the parameters. Gradients of a single stochastic realisation are not useful estimates of the gradient of the expected likelihood. Gradient-based samplers (NUTS, HMC) should not be used here.

Use gradient-free samplers instead: Metropolis-Hastings (`MH()`), particle methods, or similar. The inference tutorial demonstrates this with `MH()`.

### Sim ↔ analytical consistency

An extension with both an analytical chain size distribution and a simulation path should have a regression test confirming they agree. The test suite has a helper at `test/testutils/sim_analytical_consistency.jl`. A new type plugs in by defining two methods:

- `generative_model(m)` strips observation wrappers so `simulate_batch` can run
- `observe_chain_sizes(m, true_sizes, rng)` transforms simulated true chain sizes into observed ones (defaults to the identity; observation models override it)

`sim_analytical_consistent(model; n_chains, sizes, rng)` then simulates, applies `observe_chain_sizes`, and compares the empirical PMF against `chain_size_distribution(model)`. The helper already covers bare offspring, `ClusterMixed`, and `Observed{<:Any, <:PerCaseObservation}`.

## Connection to survival analysis

The generation time distribution g(t) = h(t)/R is the normalised infectiousness profile. Its CDF, G(t), is the cumulative hazard. When isolation occurs at time t_iso, the probabilities are:

- P(transmission before isolation) = G(t_iso)
- P(transmission after isolation) = 1 - G(t_iso)

The result is right-censoring of the transmission process. The generation time distribution (derived from the hazard) is connected to the population growth rate through the Euler-Lotka equation R = 1/M_g(-r).

The same objects -- the generation time distribution and the censoring time -- appear in both simulation and Kenah's pairwise likelihood. Inference built on top of this framework would fit the same quantities we simulate from.

## References

- Kenah E, Lipsitch M, Robins JM (2008). Generation interval contraction and epidemic data analysis. *Mathematical Biosciences* 213(1):71–79. [doi:10.1016/j.mbs.2008.02.007](https://doi.org/10.1016/j.mbs.2008.02.007)
- Kenah E (2011). Contact intervals, survival analysis of epidemic data, and estimation of R0. *Biostatistics* 12(3):548–566. [doi:10.1093/biostatistics/kxq068](https://doi.org/10.1093/biostatistics/kxq068)
- KhudaBukhsh WR, Choi B, Kenah E, Rempala GA (2020). Survival dynamical systems: individual-level survival analysis from population-level epidemic models. *Interface Focus* 10(1):20190048. [doi:10.1098/rsfs.2019.0048](https://doi.org/10.1098/rsfs.2019.0048)
- Wallinga J, Lipsitch M (2007). How generation intervals shape the relationship between growth rates and reproductive numbers. *Proceedings of the Royal Society B* 274(1609):599–604. [doi:10.1098/rspb.2006.3754](https://doi.org/10.1098/rspb.2006.3754)
