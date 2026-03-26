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

### Supporting `start_time`

Interventions can be time-dependent: effects should only apply after a
policy start time. The framework handles this automatically if you implement
three methods:

- **`EpiBranch.start_time(intervention)`** — return the policy start time
  (default `0.0`, meaning always active).
- **`EpiBranch.intervention_time(intervention, individual)`** — return the
  time at which this intervention's effect occurs for the individual (e.g.
  isolation time).
- **`EpiBranch.reset!(intervention, individual)`** — undo the intervention's
  effect on the individual.

After each `resolve_individual!` and `apply_post_transmission!` call, the
framework checks: if `intervention_time(intervention, individual)` is
earlier than `start_time(intervention)`, it calls `reset!` to undo the
effect. Individual interventions never need to check `start_time`
themselves.

Here is how `Isolation` implements these:

```julia
EpiBranch.start_time(iso::Isolation) = iso.start_time
EpiBranch.intervention_time(::Isolation, ind::Individual) = isolation_time(ind)

function EpiBranch.reset!(::Isolation, ind::Individual)
    ind.state[:isolated] = false
    ind.state[:isolation_time] = Inf
    return nothing
end
```

Alternatively, wrap any intervention with [`Scheduled`](@ref) to add
time-based or case-count-based activation without modifying the intervention
itself:

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
    SuperspreadingLimit(; threshold, start_time=0.0)

Block transmission beyond `threshold` contacts per individual.
Models targeted intervention against superspreading events (e.g.
large-gathering bans). Only active after `start_time`.
"""
Base.@kwdef struct SuperspreadingLimit <: AbstractIntervention
    threshold::Int
    start_time::Float64 = 0.0
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

# --- start_time support ---
EpiBranch.start_time(ssl::SuperspreadingLimit) = ssl.start_time

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
ssl_delayed = SuperspreadingLimit(threshold = 5, start_time = 10.0)
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
`Disease`, and `demographics` return such functions.

### Writing your own

An attributes function receives the simulation RNG and the individual.
Set fields in the individual's `state` dict:

```@example extending
function my_attributes(rng, ind)
    # Assign a risk group
    ind.state[:risk_group] = rand(rng) < 0.2 ? :high : :low
    # High-risk individuals are more susceptible
    if ind.state[:risk_group] == :high
        ind.susceptibility = 1.5
    end
end

rng = StableRNG(42)
state = simulate(model; attributes = my_attributes, sim_opts = SimOpts(max_cases = 100), rng = rng)
n_high = count(ind -> get(ind.state, :risk_group, :low) == :high, state.individuals)
println("High-risk individuals: $n_high / $(length(state.individuals))")
```

### Composing attributes functions

Use `compose` to layer multiple attributes functions. They are called in
order, so later functions can read fields set by earlier ones:

```@example extending
clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
demog = demographics(age_distribution = Normal(40, 15))

# Combine clinical, demographics, and custom attributes
combined = compose(clinical, demog, my_attributes)

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
    attributes = my_attributes,
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

## Summary of extension points

| Extension point | Mechanism | When called |
|---|---|---|
| Custom intervention | Struct `<: AbstractIntervention` + hook methods | Each generation |
| Time-dependent intervention | `start_time`, `intervention_time`, `reset!` | After each hook |
| Custom attributes | Function `(rng, ind) -> nothing` | Individual creation |
| Composed attributes | `compose(f1, f2, ...)` | Individual creation |
| Custom offspring | Function `(rng, ind) -> Int` | Offspring draw |
| Multi-type offspring | Function `(rng, ind) -> Vector{Int}` | Offspring draw |
