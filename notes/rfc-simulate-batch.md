# RFC: replace `simulate_batch` with dispatch on `simulate`

Addresses #42. sbfnk's question: "Why does it exist? Can't we use multiple dispatch or a struct dispatch trigger?"

## Status quo

```julia
simulate(model; kwargs...) -> SimulationState
simulate_batch(model, n::Int; kwargs..., parallel=false) -> Vector{SimulationState}
```

`simulate_batch` is essentially `[simulate(model; kwargs...) for _ in 1:n]` plus thread support. Both functions take the same `model`, `interventions`, `transitions`, `attributes`, `sim_opts`, `rng` arguments. Maintaining two function names means callers have to know which one to use, and the duplication has to be kept in sync if the signature grows.

## Options

### A. Positional dispatch on `n`

```julia
simulate(model)            -> SimulationState
simulate(model, n::Int)    -> Vector{SimulationState}
```

Clean and matches Julia idiom (`rand(d)` vs `rand(d, n)`). The kwarg shape stays the same. Risk: `simulate(model, n)` reads as "simulate up to n cases"; the cap semantics are actually in `SimOpts(max_cases = ...)`, so a reader might misinterpret `n` as a stopping criterion. Mitigation: name the positional arg consistently and document.

### B. SimOpts field

```julia
simulate(model; sim_opts = SimOpts(n_replicates = 1)) -> SimulationState  # when n_replicates == 1
simulate(model; sim_opts = SimOpts(n_replicates = 500)) -> Vector{SimulationState}
```

Keeps a single function with one return type per call. Awkward because the return type depends on a config field — code that consumes the result has to either always handle a Vector, or branch on the option. Loses the dispatch story.

### C. Wrapper type

```julia
simulate(model)                   -> SimulationState
simulate(Replicates(model, n))    -> Vector{SimulationState}
```

Dispatch is on the wrapper, not a positional arg. Reads honestly. Cost: a new public type that exists only to control return shape.

## Recommendation

**Option A.** It's the Julia-idiomatic shape (`rand` precedent) and the documentation cost is small. `simulate(model, n)` becomes the batch entry point; `simulate_batch` becomes a deprecated alias that forwards for one release cycle, then goes away.

Parallel runs become a kwarg on the n-arg method:

```julia
simulate(model, 500; parallel = true, rng = ..., interventions = [...])
```

## Migration

1. Add `simulate(model::TransmissionModel, n::Int; ...)` calling the existing batch code path.
2. Convert `simulate_batch` to a thin `@deprecate simulate_batch(model, n; kwargs...) simulate(model, n; kwargs...)`.
3. Update tutorials (`docs/src/tutorials/getting-started.md`, `chains.md`, others using `simulate_batch`).
4. Remove `simulate_batch` after one minor release.

This RFC PR doesn't implement the change — it scopes it so we can agree the direction first. If you're happy with option A I'll do the migration in a follow-up.
