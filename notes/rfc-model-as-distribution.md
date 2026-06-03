# RFC: model spec as a `Distribution`

Addresses #41 — "Make everything a distribution and then it will also just work with Turing vs needing the addlogprob."

This is the largest of the four "model spec boundary" issues (#35, #39, #40, #41). The other three are facets of the same architectural question; this RFC scopes the whole shape.

## What's wrong with the status quo

Today, a Turing model fitting an EpiBranch model looks like:

```julia
@model function my_model(data)
    R ~ LogNormal(0.0, 1.0)
    k ~ LogNormal(-1.0, 1.0)
    Turing.@addlogprob! loglikelihood(data, NegativeBinomial(k, k / (k + R)))
end
```

The `@addlogprob!` ceremony is required because `loglikelihood(data, model)` is just a function — it isn't a distribution Turing can plug into via `~`. A user who wants to fit `model` has to know:

1. To use `@addlogprob!` rather than `~`.
2. That `loglikelihood(data, model)` is the right entry point.
3. The Turing-side glue (`@model`, parameter priors, sampling call).

Items 2 and 3 are unavoidable; item 1 is friction the package could eliminate.

The deeper issue: a fully-specified EpiBranch "model" includes not just the offspring distribution and generation time, but the interventions stack, attributes function, and observation model. Today these are passed as separate keyword arguments to `simulate` / `loglikelihood`. There's no single object representing "the model under intervention scenario X observed by Y" that you can pass around, compare, fit, or distribute.

## Proposed shape

A `ModelSpec <: Distribution` (or `<: DiscreteMultivariateDistribution`, exact subtype TBD) that bundles every component needed to evaluate a likelihood:

```julia
spec = ModelSpec(
    process = BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5)),
    interventions = [Isolation(...), ContactTracing(...)],
    transitions = [Reporting(...)],
    observation = PerCaseObservation(detection_prob = 0.7),
    attributes = clinical_presentation(...),
)
```

`spec` would:

- Implement `Distributions.logpdf(spec, data)` so Turing's `data ~ spec` works without `@addlogprob!`.
- Implement `Random.rand(rng, spec)` to draw simulated outbreaks.
- Forward `simulate(spec)`, `simulate(spec, n)`, `chain_size_distribution(spec)`, etc. to the underlying components.

Inside the `@model`, the user writes:

```julia
@model function my_model(data)
    R ~ LogNormal(0.0, 1.0)
    k ~ LogNormal(-1.0, 1.0)
    spec = ModelSpec(process = BranchingProcess(NegBin(R, k), LogNormal(1.6, 0.5)),
                     interventions = [...])
    data ~ spec
end
```

No `@addlogprob!`. The dispatch path picks the analytical likelihood when one exists for the spec's components and falls back to simulation otherwise — same dispatch story as today's `loglikelihood(data, model)`, just routed through `logpdf`.

## What this also fixes

This change subsumes #35, #39, #40 as well:

- **#35** (why are interventions arguments?): they become fields on `ModelSpec`, so `simulate(spec)` and `loglikelihood(data, spec)` need no kwargs.
- **#39** (linelist is a converter, should it be a struct?): `linelist(spec)` could become `linelist(simulate(spec))` — a method on a simulated result, not on the spec. Or `Linelist <: AbstractOutput` if richer treatment is wanted. The spec/output boundary becomes cleaner.
- **#40** (why isn't observation part of linelist?): same answer — observation lives on the spec, line list reads the simulated state.

## What's hard

- **Distribution subtype**: `Distribution{Multivariate, Discrete}` is the closest fit but it carries baggage (`size`, `length`) that may not map cleanly onto "set of chains observed". Worth checking whether Turing can use a custom subtype of `Distribution` without those methods.
- **API churn**: every tutorial, every docstring, every test that calls `simulate(model; interventions=...)` would migrate to `simulate(spec)`. Mechanical but pervasive.
- **Backward compat**: do we deprecate `simulate(model; interventions=...)` immediately, or keep it as a thin wrapper around `simulate(ModelSpec(...; ...))`? The latter is gentler.
- **Stochastic-likelihood dispatch**: `logpdf` returning a noisy estimate (from simulation) breaks the "logpdf is a smooth function" expectation that gradient samplers rely on. Today's `loglikelihood` is honest about this through documentation; making it `logpdf` may mislead users into using NUTS where MH would be correct. Worth either a warning or a separate `logpdf_sim` channel.

## Recommendation

Do this, but as a deliberate redesign in a separate PR rather than folded into the bug-fix queue. Sequencing:

1. Land this RFC + agree the shape.
2. Implement `ModelSpec <: Distribution` alongside the existing API. Both work; the new one is the recommended path.
3. Migrate tutorials over.
4. Deprecate `simulate(model; interventions=...)` after one release cycle.

The four issues (#35, #39, #40, #41) all close together when (2) lands.

## Things to push back on

- The "everything is a distribution" framing may be overreach. `Distribution` semantics fit chain-size / chain-length data fine; line lists and contact tables are richer and not obviously distributions. The clean answer might be `ModelSpec` provides `logpdf` (for inference) and `rand` (for simulation) but is not itself a `Distribution` subtype — instead it implements the API contract Turing actually needs (just `logpdf` plus a few introspection methods).
- The user benefit is real but localised to the Turing-fit case. If most users fit via `fit(Poisson, ChainSizes(data))` (MLE) rather than NUTS, the `@addlogprob!` ceremony they're saving is for a minority workflow.

This RFC scopes the change for discussion. Implementation is deliberately out of scope for this PR.
