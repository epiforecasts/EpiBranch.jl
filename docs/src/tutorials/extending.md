# Extending EpiBranch

EpiBranch.jl is designed for user extension. This guide covers:

- Writing a custom intervention
- Adding custom individual attributes
- Using function-based offspring for custom transmission models

## Custom interventions

Every intervention is a struct that subtypes `AbstractIntervention`. The
simulation engine calls three hooks on each intervention, in order:

1. **`initialise_individual!(intervention, individual, state)`** — called once
   when each individual is created. Use this to set up intervention-specific
   fields in the individual's `state` dict.
2. **`resolve_individual!(intervention, individual, state)`** — called once per
   active individual at the start of each generation, before offspring are
   drawn. Use this to determine the individual's intervention status (e.g.
   compute isolation time).
3. **`apply_post_transmission!(intervention, state, new_contacts)`** — called
   once per generation after all offspring have been created. Receives the full
   vector of new contacts. Use this for interventions that act on contacts
   (e.g. contact tracing, gathering limits).

All three default to no-ops, so you only need to implement the ones your
intervention uses.

### Minimal example: gathering limit

A simple intervention that caps the number of successful infections per
parent in each generation:

```@example extending
using EpiBranch
using Distributions
using StableRNGs

struct GatheringLimit <: AbstractIntervention
    max_contacts::Int
end

function EpiBranch.apply_post_transmission!(gl::GatheringLimit, state, new_contacts)
    parent_counts = Dict{Int, Int}()
    for c in new_contacts
        count = get(parent_counts, c.parent_id, 0) + 1
        parent_counts[c.parent_id] = count
        if count > gl.max_contacts
            c.state[:infected] = false
        end
    end
end

gl = GatheringLimit(5)
rng = StableRNG(42)
results = simulate_batch(
    BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0)), 200;
    interventions = [gl],
    sim_opts = SimOpts(max_cases = 500),
    rng = rng,
)
println("With gathering limit (max 5): $(round(containment_probability(results), digits=3))")
```

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
# Activate gathering limit on day 10
Scheduled(GatheringLimit(5); start_time = 10.0)
```

### Requiring fields on individuals

If your intervention depends on fields set by an attributes function (e.g.
`:onset_time`), override `EpiBranch.required_fields` to get a clear error
at simulation start:

```julia
EpiBranch.required_fields(::MyIntervention) = [:onset_time, :asymptomatic]
```

## Complete worked example: superspreading limit

Some interventions need all three hooks. Here we build a
`SuperspreadingLimit` that models targeted surveillance of superspreading
events: individuals who would generate more than a threshold number of
cases have their excess contacts blocked (e.g. via event cancellation),
but only after a policy start time.

```@example extending
"""
    SuperspreadingLimit(; threshold)

Block transmission beyond `threshold` contacts per individual.
Models targeted intervention against superspreading events (e.g.
large-gathering bans). Wrap with [`Scheduled`](@ref) for time-based
activation.
"""
Base.@kwdef struct SuperspreadingLimit <: AbstractIntervention
    threshold::Int
end

# --- Hook 1: initialise fields on each individual ---
function EpiBranch.initialise_individual!(ssl::SuperspreadingLimit, individual, state)
    individual.state[:sse_limited] = false
    individual.state[:sse_limit_time] = Inf
end

# --- Hook 2: mark individual before transmission ---
# (not needed here — we act on contacts, not parents)

# --- Hook 3: act on contacts after creation ---
function EpiBranch.apply_post_transmission!(ssl::SuperspreadingLimit, state, new_contacts)
    parent_counts = Dict{Int, Int}()
    for c in new_contacts
        count = get(parent_counts, c.parent_id, 0) + 1
        parent_counts[c.parent_id] = count
        if count > ssl.threshold && is_infected(c)
            c.state[:infected] = false
            c.state[:sse_limited] = true
            c.state[:sse_limit_time] = c.infection_time
        end
    end
end

# --- Scheduling support: declare action time + how to undo it ---
function EpiBranch.intervention_time(ssl::SuperspreadingLimit, ind::Individual)
    get(ind.state, :sse_limit_time, Inf)
end

function EpiBranch.reset!(ssl::SuperspreadingLimit, ind::Individual)
    if get(ind.state, :sse_limited, false)
        ind.state[:infected] = true
        ind.state[:sse_limited] = false
        ind.state[:sse_limit_time] = Inf
    end
    return nothing
end
```

Now test it — a highly overdispersed offspring distribution (low `k`) with
and without the superspreading limit:

```@example extending
model = BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0))

# No intervention
rng = StableRNG(42)
baseline = simulate_batch(model, 200;
    sim_opts = SimOpts(max_cases = 500), rng = rng,
)

# Superspreading limit from the start
ssl = SuperspreadingLimit(threshold = 5)
rng = StableRNG(42)
with_ssl = simulate_batch(model, 200;
    interventions = [ssl],
    sim_opts = SimOpts(max_cases = 500), rng = rng,
)

# Superspreading limit starting on day 10
ssl_delayed = Scheduled(SuperspreadingLimit(threshold = 5); start_time = 10.0)
rng = StableRNG(42)
with_ssl_delayed = simulate_batch(model, 200;
    interventions = [ssl_delayed],
    sim_opts = SimOpts(max_cases = 500), rng = rng,
)

println("Baseline:              $(round(containment_probability(baseline), digits=3))")
println("SSE limit (always):    $(round(containment_probability(with_ssl), digits=3))")
println("SSE limit (from day 10): $(round(containment_probability(with_ssl_delayed), digits=3))")
```

### Composing with built-in interventions

Custom interventions compose naturally with the built-in ones. The engine
applies all interventions in order each generation:

```@example extending
clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
iso = Isolation(delay = Exponential(2.0))
ssl = SuperspreadingLimit(threshold = 5)

rng = StableRNG(42)
results = simulate_batch(model, 200;
    interventions = [iso, ssl],
    attributes = clinical,
    sim_opts = SimOpts(max_cases = 500), rng = rng,
)
println("Isolation + SSE limit: $(round(containment_probability(results), digits=3))")
```

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

## Function-based offspring models

By default, `BranchingProcess` takes a `Distribution` for the offspring
distribution. For more complex transmission models, pass a **function**
instead. The function receives `(rng, individual)` and returns the number
of offspring (an `Int` for single-type models, or a `Vector{Int}` for
multi-type).

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

- `step!(model, state, interventions)` — advance one generation.
  The default `simulate(::TransmissionModel)` loop calls this until
  termination.

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

### Clinical transitions in your `step!`

When `step!` creates a new infected individual, call
[`EpiBranch.on_new_infection!`](@ref)`(model, state, individual)` after
the individual has been built and `:infected = true` has been set. The
default runs every clinical transition on `state.transitions` against
the new individual, so transitions added by the user via
`simulate(...; transitions = ...)` work out of the box without your
`step!` knowing anything about them. Override
`on_new_infection!(::YourModel, state, ind)` only if you want to
suppress or replace the default behaviour.

### Minimal sketch

A skeleton for a custom transmission model:

```julia
struct MyModel{O} <: TransmissionModel
    offspring::O
    # ... your model parameters
end

# Required for simulation: one generation step.
function EpiBranch.step!(model::MyModel, state::SimulationState, interventions)
    # ... build a new infected individual `ind`, then:
    EpiBranch.on_new_infection!(model, state, ind)   # runs transitions
    # ... push ind onto state.individuals, update counters, etc.
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
observation are wired through `Observed{P, O}` and the dispatch only
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

1. Simulation dispatch: `_draw_offspring(rng, offspring, individual, state)` returning the number of offspring.
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
| Custom offspring (type) | Struct + `_draw_offspring`, `chain_size_distribution` | Offspring draw + analytics |
| Custom transmission model | Struct `<: TransmissionModel` + `step!`, `single_type_offspring` | Simulation + analytics |
| Custom observation model | Struct `<: ObservationModel` + `chain_size_distribution(::Observed{...})` and/or `loglikelihood(::DataType, ::Observed{...})` | Analytics / inference |
| Per-observation metadata | Either pre-compute into existing `ChainSizes` fields, or define a new data type with a `loglikelihood` method that calls `_chain_size_logpdf` | Likelihood evaluation |
| Sim ↔ analytical test | `generative_model`, `observe_chain_sizes` | Regression test |
