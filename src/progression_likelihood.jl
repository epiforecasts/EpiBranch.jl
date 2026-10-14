# ── Progression likelihood ────────────────────────────────────────────
#
# The natural-history counterpart to `pairwise_surv_loglik`: the
# log-likelihood of a case's clinical timeline (onset given infection,
# reporting given onset, and so on) under a `ModelSpec`'s `progression`,
# rather than the infection layer the pairwise likelihood evaluates. Together
# they give the log-density of the full augmented data for a simulated or
# hand-built outbreak; see the design notes in `docs/src/design.md`.

"""
    progression_loglik(spec::ModelSpec, individuals) -> Float64

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
function progression_loglik(spec::ModelSpec, individuals::AbstractVector{<:Individual})
    ll = 0.0
    for ind in individuals
        get(ind.state, :infected, false) || continue
        for t in spec.progression
            ll += transition_loglik(t, ind)
        end
    end
    return ll
end

progression_loglik(spec::ModelSpec, state::SimulationState) = progression_loglik(spec, state.individuals)
