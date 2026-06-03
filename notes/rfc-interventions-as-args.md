# RFC: interventions as args vs as model fields

Addresses #35 — "Why are attributes and interventions arguments for `simulate` and not a model component?"

## Status quo

```julia
simulate(model;
    interventions = [Isolation(...), ContactTracing(...)],
    transitions = [Reporting(...)],
    attributes = clinical_presentation(...),
    sim_opts = ...,
    rng = ...,
)
```

The model knows about offspring and timing. Interventions, transitions, and attributes are passed as separate kwargs each call.

## What the current design buys

- **Orthogonality** (principles.md §3, design.md §Extension model): interventions are written as independent components that can be stacked. Keeping them out of the model means a model can be re-used under different intervention scenarios without re-construction.
- **One simulate call per scenario**: `simulate(model; interventions = scenario_A)` then `simulate(model; interventions = scenario_B)` makes scenario sweeps natural.
- **Cleaner type signatures**: `BranchingProcess{O, G, P, L}` doesn't carry type parameters for the intervention stack.

## What it costs

- **No single object representing "the model under scenario X"**. You can't pass it around, equality-check it, distribute it to workers, or feed it to a Distributions-style inference flow without dragging the kwargs.
- **`simulate` and `loglikelihood` have parallel kwarg lists** that must stay in sync. Easy to forget one on either side.
- **Turing models need `@addlogprob!`** because what you'd want to plug into `~` is "the model with these interventions and this observation", which has no name. See #41.

## The choice

Option A: keep as-is, document the orthogonality rationale in design.md.

Option B: introduce a `ModelSpec` that bundles model + interventions + transitions + attributes + observation. See the RFC in #41 for the full shape — this issue subsumes into that one.

Option C: hybrid — `simulate(model; ...)` stays for ergonomic scenario sweeps, *and* `ModelSpec(model; ...)` exists as the bundled form for inference / serialisation / equality.

## Recommendation

**Option C, sequenced via the #41 RFC.** The orthogonality argument is real for scenario work, so the kwarg path shouldn't disappear. But the absence of a bundled spec is a real cost for inference and reproducibility, so `ModelSpec` should exist alongside.

This RFC scopes the question for #35 specifically. The full implementation discussion belongs in #41; this PR just records the design tension and the recommended direction.

Relates to: #35, #39, #40, #41.
