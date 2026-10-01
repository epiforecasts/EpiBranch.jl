# EpiBranch.jl

A Julia framework for branching-process models of infectious disease outbreaks,
consolidating what three R packages from the epiverse-trace ecosystem do onto
one shared engine with composable layers:

- **simulist** (https://github.com/epiverse-trace/simulist) — line list and
  contact tracing data from branching process outbreaks
- **epichains** (https://github.com/epiverse-trace/epichains) — transmission
  chain statistics
- **ringbp** (https://github.com/epiforecasts/ringbp) — isolation, contact
  tracing and ring vaccination, and the containment probability under them

Where a closed-form result exists (extinction probability, expected chain size,
offspring distribution fitting, as in
https://github.com/epiverse-trace/superspreading), prefer it to simulation.
Simulate what has no closed form, such as containment under a combination of
interventions.

## Design

The architecture lives in [`docs/src/design.md`](docs/src/design.md) — read it
first. The concrete contracts (hook signatures, the reserved keys table for
`Individual.state`, worked examples) are in
[`docs/src/tutorials/extending.md`](docs/src/tutorials/extending.md).

The rule that matters most, from design.md's "Extension by dispatch": new
behaviour is added by defining a new type and a method, not by growing options
on an existing struct. A `Union` field whose members trigger different
branches, a `Symbol` that switches behaviour inside a function, or a `Bool`
that selects a policy each mean a seam is in the wrong place.

Before adding a field, keyword or flag to an existing type, or a branch to an
engine loop, work through these:

1. **Which existing seam covers it?** `keep_active`, `competing_risk`,
   `transmission_risks`, `is_eligible`, `should_stop`, `_sample_value`,
   `trace_contacts!`, `intervention_actions`, `loglikelihood`, `Transition`
   and the attribute builders are all dispatched extension points. Prefer one
   of them to a new option.
2. **If none fits, add a dispatched trait with a default**, and document it in
   the extending guide, so the variant can be written from outside the
   package. A seam nobody outside can reach is not a seam (principle 4).
3. **Engine loops ask the composed layers; they never decide for them.** A
   race or stepping loop that names a concrete intervention type, reads a
   state key an intervention owns, or keeps its own table of what has already
   been done to whom has taken a policy decision into core.
4. **Say in the pull request which seam was used**, or which was considered
   and why it did not fit. A new `Bool`, `Symbol` or `Union` field on a core
   type, or a new verb alongside an existing one, needs that justification.

The test, as design.md puts it: can a plausible new variant be added without
editing the component's source? If not, the varying part belongs in a
dispatched-on trait.

## Conventions

1. **Distributions from Distributions.jl**: use standard `Distributions.jl`
   types for offspring distributions, delay distributions, etc. Do not wrap
   these in bespoke distribution types. The one sanctioned exception is the
   internal `_ChainSizeLaw`/`_ChainLengthLaw` wrappers in
   `src/likelihood_dists.jl`, which subtype `Distribution` solely so a model
   can sit on the right-hand side of Turing's `~`; they delegate to the
   existing `loglikelihood` methods and are not exported.
2. **DataFrames output**: line lists and chain statistics returned as
   DataFrames, matching epidemiological conventions (one row per case or per
   contact pair).
3. **Reproducibility**: explicit RNG threading for reproducible parallel
   simulations.

## Style

- Use British English in all documentation and comments ("modelling",
  "behaviour", etc.)
- Prefer explicit types over duck typing for the core simulation types
- Document all public functions with docstrings
- Write tests alongside implementation
- Keep the package self-contained — no dependency on EpiAware.jl (it will be
  redesigned)
- PrimaryCensored.jl is available and stable if needed for delay distribution
  censoring
