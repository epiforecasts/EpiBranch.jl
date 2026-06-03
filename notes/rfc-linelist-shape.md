# RFC: linelist as converter vs as model layer

Addresses #39 — "`linelist` is a converter for a data format, but in some sense it's a model? If the latter it feels like it should be a struct + a DataFrame method? Related: what's the rationale for why some elements are in model and some are in linelist?"

## Status quo

`linelist(state::SimulationState; reference_date = Date(2020, 1, 1))` is a pure converter from a finished simulation to a DataFrame. It reads `Individual.state` keys (`:onset_time`, `:isolated`, `:traced`, etc.), turns `*_time` floats into `Date`s using `reference_date`, and stacks one row per infected case.

The function lives in `src/output/linelist.jl`. It has no struct backing it — just a function on `SimulationState` returning `DataFrame`.

## The two questions in the issue

### Q1 — should there be a `Linelist` struct?

The status quo treats line lists as a *view* of the simulation state — a presentation layer. The function picks which `state` keys become columns and how to format them.

If `Linelist` were a struct, what would it carry?

- The `reference_date` (currently a kwarg)
- Custom column selectors / formatters
- The simulated state, or a derived snapshot of it

A `Linelist` struct would matter if a single simulation needs to be projected to different line-list shapes (different reference dates, column subsets, observation-driven gating). At the moment that's done by re-calling `linelist` with different kwargs, which is fine for the cases the package handles.

A small step that would help: `LinelistOptions` (or `LinelistSpec`) bundling the kwargs, so a complex line-list configuration can be named and reused. Doesn't require making the line list itself a struct.

### Q2 — why are some elements in the model and others in linelist?

The split is **process vs presentation**:

- **Model** owns dynamics that influence transmission: offspring, generation time, infectiousness, susceptibility.
- **Interventions / transitions** own per-individual state changes during the simulation: isolation time, trace flag, hospitalisation admission.
- **Observation models** (`PerCaseObservation`) own how the latent state generates observed data: detection probability, reporting delay.
- **Line list** is a presentation of the simulation's outcome. It reads from `Individual.state` after the simulation finishes. It does not own any dynamics.

The rationale is in design.md §"Top-level diagram" but only as a picture — the prose explanation doesn't single out the line list as the presentation layer.

## Recommendation

Two doc improvements rather than an API change:

1. **A short "What goes where" section in design.md** spelling out the process / observation / output split and how each step reads from the previous. This is what #39, #40, #35 are all circling.
2. **Mention `linelist` as the presentation layer explicitly** in the line list tutorial.

A `Linelist` struct (or `LinelistOptions`) is a smaller, lower-priority follow-up — only worth it if a real use case appears for re-projecting one simulation into multiple line-list shapes.

Larger redesign — bundling everything (model + interventions + observation + presentation) into a single object — lives in #41. This RFC just answers the line-list-shape question; the doc work it recommends will probably land alongside that bigger discussion.

Relates to: #39, #40, #41.
