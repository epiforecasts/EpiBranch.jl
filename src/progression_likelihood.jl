# ── Progression likelihood ────────────────────────────────────────────
#
# The natural-history counterpart to `pairwise_surv_loglik`: the
# log-likelihood of a case's clinical timeline (onset given infection,
# reporting given onset, and so on) under a `ModelSpec`'s `progression`,
# rather than the infection layer the pairwise likelihood evaluates. Together
# they give the log-density of the full augmented data for a simulated or
# hand-built outbreak; see the design notes in `docs/src/design.md`.

"""
    progression_loglik(spec::ModelSpec, individuals; followup_end = Inf) -> Float64

The log-likelihood of the cases' clinical timelines under the steps of the
model's `progression` (reporting, hospitalisation, death or recovery, each
measured from the time it starts from). Use it to estimate the delays in
those steps, such as a reporting delay, from a line list with event times.
An incubation period drawn by [`clinical_presentation`](@ref) is not
included unless onset is itself a step of the progression.

`individuals` is a vector of [`Individual`](@ref) or a [`SimulationState`](@ref)
from [`simulate`](@ref), or a hand-built equivalent holding the times each
progression step records. People who were exposed but never infected add
nothing. The result is the sum over cases and steps of
[`transition_loglik`](@ref EpiBranch.transition_loglik).

`followup_end` is the end of follow-up. A step that has not happened by then
(whether it was drawn to occur later, or not to occur at all) is treated as
censored: the probability that it had not happened by `followup_end`, in place
of a density. This matters when infection times are imputed, because a case
infected shortly before the cut-off may have no onset or report yet. Steps
that did happen enter with their plain density, without conditioning on
having happened before `followup_end`. Right truncation of the observed delays
is not yet handled, and a delay distribution fitted to a line list cut at a
date is biased towards short delays. A custom
[`transition_loglik`](@ref EpiBranch.transition_loglik) method needs a
`followup_end` keyword to be used with a finite `followup_end`.

A line list holds only the outcome that happened. With competing outcomes
such as [`Death`](@ref) and [`Recovery`](@ref), a case that has an `:outcome`
and `:outcome_time` but no candidate time for the other outcome is treated as
the other outcome not having happened by then: `f_D(t) S_R(t)` for a death at
`t`, and the reverse for a recovery.

It covers natural history only. Who infected whom and when is the transmission
part, evaluated by [`pairwise_surv_loglik`](@ref); add the two for the full
log-likelihood of an outbreak with known infection times. Neither includes an
observation model.

# Examples

```julia
using EpiBranch, Distributions, StableRNGs

progression = [
    Reporting(delay = LogNormal(1.0, 0.3)),
    Recovery(delay = LogNormal(2.0, 0.4)),
]
spec = ModelSpec(
    BranchingProcess(Poisson(2.0), Exponential(5.0));
    progression = progression,
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
)
state = simulate(spec; max_cases = 100, rng = StableRNG(1))
progression_loglik(spec, state)
```
"""
function progression_loglik(
        spec::ModelSpec, individuals::AbstractVector{<:Individual}; followup_end::Real = Inf
    )
    isnan(followup_end) && throw(ArgumentError("followup_end must not be NaN"))
    ll = 0.0
    for ind in individuals
        is_infected(ind) || continue
        for t in spec.progression
            # The default leaves custom `transition_loglik` methods without
            # the keyword working as before.
            ll += isinf(followup_end) ? transition_loglik(t, ind) :
                transition_loglik(t, ind; followup_end = followup_end)
        end
    end
    return ll
end

function progression_loglik(spec::ModelSpec, state::SimulationState; kwargs...)
    return progression_loglik(spec, state.individuals; kwargs...)
end
