# Extending EpiBranch

EpiBranch.jl is designed for user extension. This guide covers:

- Writing a custom intervention
- Adding custom individual attributes
- Using function-based offspring for custom transmission models

## Custom interventions

Every intervention is a struct that subtypes `AbstractIntervention`. The
engine calls four hooks on each intervention; you implement only the
ones your intervention needs (all default to no-ops).

### Hook contract

| Hook | Called | Receives | Must return |
|---|---|---|---|
| `initialise_individual!(iv, individual, state)` | Once when each individual is created | An `Individual` whose typed fields are set but whose `state` dict is empty | `nothing` (mutate `individual.state` in place) |
| `resolve_individual!(iv, individual, state)` | Once per active individual at the start of each generation, before offspring are drawn | The parent for the upcoming step | `nothing` (mutate `individual.state` in place) |
| `apply_post_transmission!(iv, state, new_contacts)` | Once per generation after all contacts for that generation have been created (across every active parent) | A `Vector{Individual}` of the new contacts | `nothing` (mutate any of the contacts' `state` in place) |
| `competing_risk(iv, parent, contact, state)` | Per `(parent, contact)` pair during infection resolution, after `apply_post_transmission!` has run | The parent and a single new contact | `nothing`, a single [`Risk`](@ref), or an `NTuple{N, Risk}` for interventions that gate transmission via more than one mechanism |

Ordering guarantees:

- `resolve_individual!` runs strictly before any `competing_risk` call for that generation, so a competing risk can read whatever `resolve_individual!` wrote on the parent.
- `apply_post_transmission!` runs strictly before any `competing_risk` call, so a competing risk can read whatever post-transmission hook wrote on the contact (e.g. `:vaccination_time`).
- Interventions are applied in the order they appear in `interventions = [...]`. For `apply_post_transmission!` and `competing_risk`, every intervention sees the state written by earlier interventions in the same generation.

A `Risk` applies to a contact when `event_time <= contact.infection_time`; in that case transmission is blocked with probability `block_probability`. Returning multiple risks (as a tuple) lets one intervention gate transmission through several mechanisms — `RingVaccination` returns both a susceptibility risk on the contact and an onward-infectiousness risk on the parent.

Tree-shaping changes — capping offspring per parent, gathering-size limits, anything that's really "this parent produces fewer contacts than its natural offspring distribution would say" — belong in the offspring distribution itself, not in the intervention protocol. See [Tree-shaping via the offspring distribution](#tree-shaping-via-the-offspring-distribution) below.

### What each hook looks like in practice

Short snippets from the built-in interventions, one per hook, to anchor the contract above. The full source lives in `src/interventions/`.

**`initialise_individual!`** — `ContactTracing` initialises the two flags it owns on every new individual so accessors elsewhere get a defined value:

```julia
function initialise_individual!(::ContactTracing, individual, state)
    individual.state[:traced] = false
    individual.state[:quarantined] = false
    return nothing
end
```

**`resolve_individual!`** — `Isolation` computes the isolation time for the upcoming generation's parent from the individual's onset time plus a sampled delay, and folds in any earlier trace-driven isolation time that `ContactTracing` may have written on a previous generation:

```julia
function resolve_individual!(iso::Isolation, individual, state)
    is_isolated(individual) && return nothing
    is_test_positive(individual) || return nothing

    iso_delay = rand(state.rng, iso.delay)
    iso_time = onset_time(individual) + iso_delay

    traced_time = get(individual.state, :traced_isolation_time, Inf)
    set_isolated!(individual, min(iso_time, traced_time))
    return nothing
end
```

**`apply_post_transmission!`** — `ContactTracing` walks the new contacts, looks up each contact's parent, and applies the configured trace action (`Quarantine` or `FlagOnly`) when the eligibility and rate traits both pass:

```julia
function apply_post_transmission!(ct::ContactTracing, state, new_contacts)
    rng = state.rng
    for ind in new_contacts
        ind.parent_id == 0 && continue
        parent = state.individuals[ind.parent_id]
        is_eligible(ct.eligibility, parent, ind, state) || continue
        traces(ct.trace_rate, parent, ind, state, rng) || continue
        trace_delay = draw_trace_delay(ct.delay, parent, ind, state, rng)
        trace_time = isolation_time(parent) + trace_delay
        apply_trace!(ct.action, ind, state, trace_time, rng)
    end
    return nothing
end
```

**`competing_risk`** — see the [`BorderClosure` minimal example](#minimal-example-a-custom-competing-risk) below for a complete worked custom intervention.

### Verifying your intervention

The engine never errors when a hook is missing — every hook has a no-op default. That is convenient for partial implementations but means that *forgotten* hooks fail silently. Quick checks:

- Run a tiny simulation (`SimOpts(max_cases = 50)`) with and without your intervention in the stack. If the outcome looks the same in both, your `competing_risk` or `apply_post_transmission!` is probably not being called for the cases you think.
- Override `required_fields` (see below) so the engine fails at simulation start when an upstream attributes function hasn't set a field your intervention needs.
- Inspect `state.individuals[1].state` after a small run to confirm your hook actually wrote the keys downstream code reads.

### Minimal example: a custom competing risk

A "border closure" intervention that blocks transmission between
contacts in different regions after a given date. Each individual
carries `:region` as a custom attribute; the intervention's
`competing_risk` reads both parent and contact regions and contributes
a blocking risk when they differ.

```@example extending
using EpiBranch
using Distributions
using StableRNGs

struct BorderClosure <: AbstractIntervention
    start_time::Float64
    leakage::Float64   # residual cross-border transmission probability
end

function EpiBranch.competing_risk(bc::BorderClosure, parent, contact, state)
    parent.state[:region] == contact.state[:region] && return nothing
    return Risk(event_time = bc.start_time,
        block_probability = 1.0 - bc.leakage)
end
```

`event_time = bc.start_time` means the risk only applies to contacts
whose transmission time is on or after the closure date; cross-border
transmissions before the closure are unaffected.

### Making the intervention schedulable

Time-based scheduling is provided uniformly by [`Scheduled`](@ref):
wrap any intervention with `Scheduled(iv; start_time = ...)` to delay
its activation. Individual interventions do not carry a `start_time`
field of their own.

For `Scheduled` to perform per-individual reset (the case where the
population gate has opened but a specific individual's sampled action
time would fall pre-policy), the intervention declares two methods:

- **`EpiBranch.intervention_time(intervention, individual)`** — the time
  at which this intervention's effect occurs for the individual (e.g.
  isolation time).
- **`EpiBranch.reset!(intervention, individual)`** — undo the
  intervention's effect on the individual.

Here is how `Isolation` implements these:

```julia
EpiBranch.intervention_time(::Isolation, ind::Individual) = isolation_time(ind)

function EpiBranch.reset!(::Isolation, ind::Individual)
    ind.state[:isolated] = false
    ind.state[:isolation_time] = Inf
    return nothing
end
```

Then a user schedules the intervention like:

```julia
# Activate border closure on day 10
Scheduled(BorderClosure(0.0, 0.05); start_time = 10.0)
```

### Requiring fields on individuals

If your intervention depends on fields set by an attributes function (e.g.
`:onset_time`), override `EpiBranch.required_fields` to get a clear error
at simulation start:

```julia
EpiBranch.required_fields(::MyIntervention) = [:onset_time, :asymptomatic]
```

### Composing with built-in interventions

Custom interventions compose naturally with the built-in ones. The
engine applies all interventions in order each generation. Building on
the `BorderClosure` above (interpreted here as everyone being in one
region so closure does nothing — illustrative only):

```@example extending
model = BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0))
clinical_with_region = compose(
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    (rng, ind) -> (ind.state[:region] = :only),
)
iso = Isolation(delay = Exponential(2.0))
bc = BorderClosure(10.0, 0.05)

rng = StableRNG(42)
results = simulate_batch(model, 200;
    interventions = [iso, bc],
    attributes = clinical_with_region,
    sim_opts = SimOpts(max_cases = 500), rng = rng,
)
println("Isolation + border closure: $(round(containment_probability(results), digits=3))")
```

## Tree-shaping via the offspring distribution

Some interventions don't filter individual transmissions — they change
*how many* contacts a parent makes. Gathering limits, event-size caps,
and superspreading-event surveillance all fit this pattern. They are
genuinely modifications to the offspring distribution, not per-contact
risks, and EpiBranch handles them by accepting a function-form
offspring distribution to [`BranchingProcess`](@ref).

A hard cap on offspring per parent:

```@example extending
capped_offspring(rng, ind) = min(rand(rng, NegBin(2.5, 0.16)), 5)
model_capped = BranchingProcess(capped_offspring, Exponential(5.0))
```

A state-aware cap that takes effect once the outbreak crosses 20
cases (mirroring what `Scheduled` does for risk-based interventions,
but for a tree-shape change):

```@example extending
function policy_offspring(rng, ind, state)
    n = rand(rng, NegBin(2.5, 0.16))
    return state.cumulative_cases >= 20 ? min(n, 5) : n
end
model_policy = BranchingProcess(policy_offspring, Exponential(5.0))
```

The function form supports either two or three arguments — `(rng, ind)`
when the offspring rule only needs the individual, `(rng, ind, state)`
when it also reads simulation state.

Time-varying R falls out of the same mechanism. If `R(t)` is a
function of (say) the parent's infection time, pass an offspring
distribution that reads `ind.infection_time`:

```@example extending
r_at_time(t) = max(1.0, 3.0 - 2.0 * t / 50.0)
time_varying = (rng, ind) -> rand(rng, Poisson(r_at_time(ind.infection_time)))
model_rt = BranchingProcess(time_varying, Exponential(5.0))
```

Use `ind.infection_time` when R varies with each parent's own
infection timing, or `state.max_infection_time` (via the
three-argument form) when R varies with the population-level outbreak
clock.

## Custom attributes functions

The `attributes` argument to `simulate` is a function `(rng, individual) -> nothing`
that sets fields on each individual when they are created (before any
intervention hooks run). The built-in constructors `clinical_presentation`,
`demographics`, and `transmission_traits` return such functions.

### Writing your own

For fields without a dedicated builder — anything in `ind.state` — write a
plain closure. Below, `:risk_group` is a custom state field, so it needs
the closure form; `susceptibility` is derived from it via
`transmission_traits`, which accepts a function:

```@example extending
risk_group = (rng, ind) -> (ind.state[:risk_group] = rand(rng) < 0.2 ? :high : :low)

attrs = compose(
    risk_group,
    transmission_traits(
        susceptibility = (rng, ind) -> ind.state[:risk_group] == :high ? 0.8 : 0.3,
    ),
)

rng = StableRNG(42)
state = simulate(model; attributes = attrs, sim_opts = SimOpts(max_cases = 100), rng = rng)
n_high = count(ind -> get(ind.state, :risk_group, :low) == :high, state.individuals)
println("High-risk individuals: $n_high / $(length(state.individuals))")
```

### Composing attributes functions

`compose` calls its arguments in order, so later builders or closures can
read fields set by earlier ones:

```@example extending
combined = compose(
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    demographics(age_distribution = Normal(40, 15)),
    risk_group,
    transmission_traits(
        susceptibility = (rng, ind) -> ind.state[:risk_group] == :high ? 0.8 : 0.3,
    ),
)

rng = StableRNG(42)
state = simulate(model;
    attributes = combined,
    sim_opts = SimOpts(max_cases = 100),
    rng = rng,
)
ind = state.individuals[1]
println("Individual 1: age=$(ind.state[:age]), sex=$(ind.state[:sex]), risk=$(ind.state[:risk_group])")
```

## Custom offspring distributions

`BranchingProcess` takes anything its sampling path can use. Two paths
are supported, with different trade-offs:

- A **`Distribution` subtype** (a struct `<: Distribution` from
  Distributions.jl). Full Distributions.jl interop — `rand`, `logpdf`,
  fitting, mixtures all work — and the analytical helpers
  (`extinction_probability`, `chain_size_distribution`, etc.) compose
  cleanly via [`single_type_offspring`](@ref).
- A **function** `(rng, individual)` or `(rng, individual, state)`
  returning the offspring count. Escape hatch for state-dependent
  rules that don't fit a single fixed Distribution (time-varying R,
  policy-dependent caps, etc.).

### `Distribution` subtype

For an offspring rule that's a proper probability distribution,
subtype `Distribution` and implement `Distributions.rand`. The
branching process picks it up via the standard constructor:

```julia
using Distributions
using Random: AbstractRNG

struct MyOffspring <: Distribution{Univariate, Discrete}
    # ... your parameters
end

function Distributions.rand(rng::AbstractRNG, d::MyOffspring)
    # ... return an integer offspring count
end

model = BranchingProcess(MyOffspring(...), Exponential(5.0))
```

The simulation loop calls `rand(rng, offspring)`, so any Distribution
subtype with a `rand` method works.

To enable the analytical helpers (`extinction_probability`,
`chain_size_distribution`, `proportion_transmission`), also specialise
`chain_size_distribution` for your type:

```julia
function EpiBranch.chain_size_distribution(d::MyOffspring)
    # return a Distribution over chain sizes — e.g. via numerical
    # iteration of the offspring PGF.
end
```

Simulation works without this specialisation; only the closed-form
analytics require it.

### Function-based offspring

For state-dependent rules that don't fit a single fixed Distribution,
pass a function. It receives `(rng, individual)` or `(rng, individual,
state)` and returns the number of offspring (an `Int` for single-type
models, or a `Vector{Int}` for multi-type).

### Example: context-dependent transmission

Here the reproduction number depends on a custom attribute:

```@example extending
function risk_offspring(rng, individual)
    base_R = individual.state[:risk_group] == :high ? 4.0 : 1.5
    return rand(rng, Poisson(base_R))
end

model_risk = BranchingProcess(risk_offspring, Exponential(5.0); n_types = 1)

rng = StableRNG(42)
results = simulate_batch(model_risk, 200;
    attributes = risk_group,
    sim_opts = SimOpts(max_cases = 500),
    rng = rng,
)
println("Risk-stratified model: $(round(containment_probability(results), digits=3))")
```

### Example: generation-dependent R

The offspring function can also read the individual's generation to model
waning transmission over the course of an outbreak:

```@example extending
function waning_offspring(rng, individual)
    R = 3.0 * exp(-0.1 * individual.generation)
    return rand(rng, Poisson(R))
end

model_waning = BranchingProcess(waning_offspring, Exponential(5.0); n_types = 1)

rng = StableRNG(42)
results = simulate_batch(model_waning, 200;
    sim_opts = SimOpts(max_cases = 500),
    rng = rng,
)
println("Waning-R model: $(round(containment_probability(results), digits=3))")
```

## Adding a transmission model

Most use cases stay inside `BranchingProcess` and customise via the
offspring distribution (function-based, `ClusterMixed`, multi-type).
But if you need a fundamentally different transmission process — a
density-dependent model, a network-structured one, a continuous-time
SEIR-like alternative — you can subtype `TransmissionModel` directly
and reuse the rest of the framework.

The contract is small. You implement what your model needs and reuse
defaults for the rest.

### What the framework expects

For **simulation**, define one method:

- `step!(model, state)` — advance one generation. The default
  `simulate(::TransmissionModel)` loop calls this until termination.
  It must return a `Vector{Individual}` of the new contacts for this
  generation. Build each contact with [`make_contact!`](@ref); do not
  push to `state.individuals` or set `:infected` yourself.

The signature has no `interventions` argument: a custom `step!` is
strictly the model's offspring-and-timing layer and cannot couple to
interventions. Produce every potential contact your model generates —
do not pre-filter by parent intervention state (`:isolated`,
`:vaccinated`, …). Whether each contact is actually infected is then
decided by the engine's competing-risks resolution after `step!`
returns; that is the only place intervention effects on transmission
outcomes apply.

The engine handles everything around `step!`: `resolve_individual!`
runs on each active parent before `step!`, so by the time your `step!`
is called every parent's intervention state for this generation is
already up to date. After `step!` returns, the engine runs
`initialise_individual!` and `apply_post_transmission!` on the new
contacts, resolves competing risks to set `:infected`, runs clinical
transitions, and updates the bookkeeping fields
(`cumulative_cases`, `current_generation`, `active_ids`, `extinct`,
`max_infection_time`).

For **analytical inference helpers** that route through the offspring
specification (`extinction_probability`, `epidemic_probability`,
`probability_contain`, `proportion_transmission`,
`chain_size_distribution`), define one method:

- [`single_type_offspring`](@ref)`(model)` returning the offspring
  distribution (or any object for which `chain_size_distribution` is
  defined). Specialise this and you get the analytical helpers for
  free.

For **likelihoods** on data types that don't go through the offspring
spec, define methods on `loglikelihood` directly.

For optional **state accessors**, override `population_size`,
`latent_period`, `n_types` if your model has values for them. The
defaults (`NoPopulation()`, `0.0`, `1`) are fine if not.

### Minimal sketch

A skeleton for a custom transmission model:

```julia
struct MyModel{O, G} <: TransmissionModel
    offspring::O
    generation_time::G
    # ... your model parameters
end

# Required for simulation: one generation step. Build contacts with
# `make_contact!` and return the vector; the engine handles
# `:infected`, post-transmission hooks, transitions, and bookkeeping.
function EpiBranch.step!(model::MyModel, state::SimulationState)
    new_contacts = Individual[]
    for idx in state.active_ids
        parent = state.individuals[idx]
        for _ in 1:rand(state.rng, model.offspring)
            t = parent.infection_time + rand(state.rng, model.generation_time)
            make_contact!(new_contacts, state, parent, t)
        end
    end
    return new_contacts
end

# Required for analytical helpers (optional but recommended).
EpiBranch.single_type_offspring(m::MyModel) = m.offspring

# Optional accessors, with defaults if unset.
EpiBranch.population_size(m::MyModel) = NoPopulation()
EpiBranch.latent_period(m::MyModel) = 0.0
EpiBranch.n_types(m::MyModel) = 1
```

If `single_type_offspring(m)` returns a NegBin or a `ClusterMixed` or
anything else with a `chain_size_distribution` method, the analytical
chain-size likelihood works automatically:

```julia
loglikelihood(ChainSizes(data), MyModel(NegBin(0.8, 0.5), ...))
```

### Composing with the observation side

Your model fits into `Observed` without any extra work. Process and
observation are combined through `Observed{P, O}` and the dispatch only
asks that `single_type_offspring` delegates correctly through any
wrappers — which it does for `Observed` automatically. So:

```julia
Observed(MyModel(...), PerCaseObservation(detection_prob = 0.7))
```

works the same way as it does for `BranchingProcess`.

## Adding an observation model

Observation models subtype `ObservationModel` and combine with a
process model via [`Observed`](@ref). `PerCaseObservation` (per-case
detection probability and reporting delay) is the reference. A new
observation model needs three pieces:

1. A struct holding the observation parameters, subtyping `ObservationModel`.
2. A transformed chain size distribution (a new `DiscreteUnivariateDistribution`) whose `logpdf` applies the observation process to a base distribution. This type can wrap any distribution, including one produced by another observation, so nesting works.
3. A `chain_size_distribution` method on `Observed{<:Any, <:YourObservation}` that pairs the wrapped distribution with the new transformed distribution.

### Minimal sketch

```julia
# 1. Observation model
struct CensoredAtSize <: ObservationModel
    cap::Int
end

# 2. Transformed chain size distribution
struct TruncatedChainSize{D} <: DiscreteUnivariateDistribution
    base::D
    cap::Int
end
Distributions.minimum(::TruncatedChainSize) = 1
Distributions.maximum(d::TruncatedChainSize) = d.cap
Distributions.insupport(d::TruncatedChainSize, n::Integer) = 1 <= n <= d.cap

function Distributions.logpdf(d::TruncatedChainSize, n::Integer)
    1 <= n <= d.cap || return -Inf
    Z = sum(pdf(d.base, m) for m in 1:d.cap)
    return logpdf(d.base, n) - log(Z)
end

# 3. Hook into the dispatch
function EpiBranch.chain_size_distribution(m::Observed{<:Any, CensoredAtSize})
    TruncatedChainSize(
        EpiBranch.chain_size_distribution(m.process), m.observation.cap)
end

function Distributions.loglikelihood(data::ChainSizes,
        m::Observed{<:Any, CensoredAtSize})
    d = EpiBranch.chain_size_distribution(m)
    sum(logpdf(d, n) for n in data.data)
end
```

Usage: `Observed(model, CensoredAtSize(10))`. Composition with another
observation type goes through nested `Observed`:
`Observed(Observed(model, PerCaseObservation(0.7)), CensoredAtSize(10))`.

### Sim ↔ analytical consistency test

The helper in `test/testutils/sim_analytical_consistency.jl` cross-checks
simulation against your new distribution once you define two methods:

```julia
# Transform simulated true sizes into observed ones
function observe_chain_sizes(m::Observed{<:Any, CensoredAtSize},
        true_sizes, rng::AbstractRNG)
    inner = observe_chain_sizes(m.process, true_sizes, rng)
    return filter(n -> n <= m.observation.cap, inner)
end
```

`generative_model(::Observed)` is already defined and strips the
observation. With your `observe_chain_sizes` method in place,
`sim_analytical_consistent(model; n_chains=5000, rng=StableRNG(1))`
returns empirical and analytical PMFs that should agree within
sampling error.

## Adding an offspring specification

Offspring specifications replace what `BranchingProcess` draws per
individual. `ClusterMixed(build, mixing)` (per-chain parameter
variation) is the reference. A new offspring type needs:

1. Simulation dispatch: `draw_offspring(rng, offspring, individual, state)` returning the number of offspring.
2. Analytical dispatch (optional but recommended): `chain_size_distribution(offspring)` returning the analytical PMF. Without it, the likelihood falls back to simulation.
3. A `BranchingProcess` constructor so the type can be stored in the `offspring` field.

See `src/analytical/cluster_mixed.jl` for the full pattern, including how `ClusterMixed` caches per-chain state on the index case and has descendants inherit it through `parent_id`.

## Adding per-observation metadata

[`ChainSizes`](@ref) already supports two per-observation fields: `seeds`
(multi-seed clusters) and `concluded` (right-censored ongoing clusters).
Both are decisions made by the analyst, not properties the framework
derives — and they show the pattern for any new field.

If your analysis needs different or richer per-cluster information, you
have two options.

### Stay in `ChainSizes` and pre-compute

If the new information resolves to a flag or a count that the existing
likelihood already handles, derive it upstream and pass it in. The Endo
7-day time-censoring rule is an example: it looks like time censoring
but is just a way to compute `concluded`.

```julia
using Dates
is_ongoing(latest_case, cutoff; window_days = 7) =
    cutoff - latest_case < Day(window_days)

data = ChainSizes(sizes;
    seeds = imports_per_cluster,
    concluded = .!is_ongoing.(last_case_dates, cutoff_date))
```

No new types or methods needed — the decision rule lives wherever it
belongs in the analysis.

### Define a new data type when the likelihood needs new information

If the likelihood itself needs to use new per-observation data (not just
collapse it into an existing flag), define a new struct and a
`loglikelihood` method.

```julia
struct MultiTypeChainSizes
    data::Vector{Int}
    type::Vector{Int}   # which strain/patch/group
end

# Different offspring distribution per type; pick by observation.
function Distributions.loglikelihood(data::MultiTypeChainSizes,
        offsprings::Vector{<:Distribution})
    total = 0.0
    for i in eachindex(data.data)
        d = chain_size_distribution(offsprings[data.type[i]])
        total += logpdf(d, data.data[i])
    end
    return total
end
```

The internal `EpiBranch._chain_size_logpdf(d, x, s)` is the reusable
piece — call it from your method if you need multi-seed support, and
your new data type inherits the same closed forms for `Borel`,
`GammaBorel`, `PoissonGammaChainSize` as the built-in `ChainSizes` uses.

## Summary of extension points

| Extension point | Mechanism | When called |
|---|---|---|
| Custom intervention | Struct `<: AbstractIntervention` + hook methods | Each generation |
| Time-dependent intervention | `Scheduled(iv; start_time = ...)` + `intervention_time`, `reset!` on `iv` | After each hook |
| Custom attributes | Function `(rng, ind) -> nothing` | Individual creation |
| Composed attributes | `compose(f1, f2, ...)` | Individual creation |
| Custom offspring (function) | Function `(rng, ind) -> Int` | Offspring draw |
| Multi-type offspring | Function `(rng, ind) -> Vector{Int}` | Offspring draw |
| Custom offspring (type) | Struct + `draw_offspring`, `chain_size_distribution` | Offspring draw + analytics |
| Custom transmission model | Struct `<: TransmissionModel` + `step!`, `single_type_offspring` | Simulation + analytics |
| Custom observation model | Struct `<: ObservationModel` + `chain_size_distribution(::Observed{...})` and/or `loglikelihood(::DataType, ::Observed{...})` | Analytics / inference |
| Per-observation metadata | Either pre-compute into existing `ChainSizes` fields, or define a new data type with a `loglikelihood` method that calls `_chain_size_logpdf` | Likelihood evaluation |
| Sim ↔ analytical test | `generative_model`, `observe_chain_sizes` | Regression test |
