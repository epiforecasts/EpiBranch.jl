# RFC: why isn't imperfect observation part of `linelist`?

Addresses #40 — "Why is the [imperfect-observation tutorial section](https://epiforecasts.io/EpiBranch.jl/dev/tutorials/chains#Imperfect-observation) not part of `linelist`? I assume because it's model-side pre-fit, but what is the rationale for why some things appear in different places? I think making it all part of model spec would be more logical."

## Status quo

- `PerCaseObservation(detection_prob, delay)` is an observation model: subtype of `ObservationModel`, composed with a process model via `Observed(process, obs)`.
- It contributes a likelihood (via `chain_size_distribution(::Observed{...})` returning `ThinnedChainSize`) and a post-simulation projection step that writes `:reported` and `:report_time` to `Individual.state`.
- `linelist(state)` reads `:reported` and `:report_time` if present and turns them into `reported` and `date_reporting` columns.

So observation **is** in the line list — but only as columns produced by `PerCaseObservation`'s post-simulation step, not as an argument to `linelist` itself. The user can't say `linelist(state, obs_model)` to apply a fresh observation model to an already-simulated state.

## The two interpretations of the issue

### Interpretation A (literal): make `linelist` aware of observation

`linelist(state, obs::ObservationModel; reference_date = ...)` would apply `obs` to `state` and project into the DataFrame. This is feasible — would call the same projection code that runs after `simulate(::Observed{...})` today.

Cost: another path to maintain (in-engine projection vs at-output-time projection) and a question about which takes precedence when both are configured.

Benefit: observation can be re-applied post-hoc without re-simulating. Useful for "what if I'd observed with detection probability 0.5 instead of 0.7" type questions on a fixed simulation.

### Interpretation B (deeper): collapse everything into a model spec

The issue's last sentence — "making it all part of model spec would be more logical" — points at the bigger question covered by #41. If `ModelSpec(process, interventions, observation, ...)` is the single object, then `simulate(spec)` produces a state, `linelist(state)` reads the columns, and you don't have to choose where to put observation.

In that framing, the answer to #40 is "observation is on the spec, line list reads from the simulated state — they're sequential stages, not competing locations."

## Recommendation

Interpretation B is the substantive change and lives in #41 — this issue closes when that one does.

Interpretation A (post-hoc observation re-projection on a fixed state) is a smaller, independent improvement. Worth doing if there's demand: a `project_observation!(state, obs)` function that writes `:reported` / `:report_time` could be lifted out of the current `simulate(::Observed{...})` body. Then both `simulate(::Observed{...})` and a separate `linelist(state, obs)` could call it.

For now: this RFC scopes the question and points at #41 for the architectural answer. If the post-hoc projection use case lands, I'll do that as a follow-up.

Relates to: #40, #41.
