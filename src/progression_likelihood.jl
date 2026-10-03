# ── Progression likelihood ────────────────────────────────────────────
#
# The natural-history counterpart to `pairwise_surv_loglik`: the
# log-likelihood of a case's clinical timeline (onset given infection,
# reporting given onset, and so on) under a `ModelSpec`'s `progression`,
# rather than the infection layer the pairwise likelihood evaluates. Together
# they give the log density of the full augmented data of a simulated or
# hand-built outbreak;
# see the design notes in `docs/src/design.md`.

"""
    progression_loglik(spec::ModelSpec, individuals) -> Float64

The log-likelihood of `individuals`' clinical timelines under `spec`'s
`progression`: the sum, over every infected individual and every transition
in `progression`, of [`transition_loglik`](@ref EpiBranch.transition_loglik).

`individuals` is a vector of [`Individual`](@ref) (or a [`SimulationState`](@ref),
whose `individuals` field is read directly) holding the state each transition
in `progression` wrote in `resolve_individual!` — the augmented data a
simulation produces, or a hand-built equivalent for inference. An individual
never infected (`get(ind.state, :infected, false) == false`) contributes
nothing, matching the transitions engine, which never resolves one.

This is a natural-history term only: it excludes the infection layer, which
[`pairwise_surv_loglik`](@ref) evaluates, and any observation model. Added to
that pairwise likelihood, it gives the full log-likelihood of an outbreak's
augmented data — infection times, order and clinical timelines together —
under `spec`.

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
