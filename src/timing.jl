# ── Timing ──────────────────────────────────────────────────────────
#
# Timing is a shared stage: both the offspring-driven engine and the
# structure-driven (network) engine assign each candidate transmission a
# generation time the same way, reading the model's `generation_time`. It
# is therefore a primitive here rather than any one model's concern.

"""
    get_generation_time(gt, individual)

The generation time distribution for one case. A distribution is shared by
every case. A function of the individual, `(ind) -> distribution`, gives each
case its own, and can read anything recorded about the case, such as its
incubation period or other population characteristics. That way the
generation time and symptom onset of a case can be linked rather than drawn
independently. Use [`incubation_period`](@ref) to read the incubation period
inside such a function.
"""
get_generation_time(gt::Distribution, individual) = gt
get_generation_time(ngt::NoGenerationTime, individual) = ngt
get_generation_time(gt, individual) = gt(individual)

"""A contact's infection time: the infector's infection time plus a draw
from the generation time, or the infector's own time when there is no
generation time. Any biological constraint, such as a minimum latent period,
belongs in the generation time distribution itself, for example
`truncated(gt_dist, lower, Inf)` or a shifted distribution."""
transmission_time(::NoGenerationTime, parent, state) = parent.infection_time
function transmission_time(gt_dist::Distribution, parent, state)
    return parent.infection_time + rand(state.rng, gt_dist)
end
