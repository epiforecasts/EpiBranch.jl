# ── Timing ──────────────────────────────────────────────────────────
#
# Timing is a shared stage: both the offspring-driven engine and the
# structure-driven (network) engine assign each candidate transmission a
# generation time the same way, reading the model's `generation_time`. It
# is therefore a primitive here rather than any one model's concern.

"""
    get_generation_time(gt, individual)

Return the generation time distribution for a specific individual.

For a `Distribution`, everyone shares the same distribution. For a
`Function`, the engine calls it with the individual and uses the
`Distribution` it returns, so the generation time can read anything in
`individual.state`: the incubation period, or any per-individual
quantity an attributes function has stored. That is how the generation
time and symptom onset can come from one per-individual draw instead of
two independent ones. Use [`incubation_period`](@ref) to read the
incubation period inside such a function.
"""
get_generation_time(gt::Distribution, individual) = gt
get_generation_time(ngt::NoGenerationTime, individual) = ngt
get_generation_time(gt::Function, individual) = gt(individual)

"""Compute a contact's infection time from the parent's infection time
and the generation-time draw. Dispatches on the generation-time spec:
[`NoGenerationTime`](@ref) (no timing) returns the parent's time;
otherwise samples and adds. The generation time distribution should
already encode any biological constraint (e.g. a minimum latent
period); to enforce a lower bound use `truncated(gt_dist, lower, Inf)`
or a shifted distribution."""
_infection_time(::NoGenerationTime, parent, state) = parent.infection_time
function _infection_time(gt_dist::Distribution, parent, state)
    return parent.infection_time + rand(state.rng, gt_dist)
end
