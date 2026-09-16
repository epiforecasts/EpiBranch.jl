# ── Reproduction number ──────────────────────────────────────────────
#
# The verb that reads a threshold off an offspring specification. Each
# specification adds a method — the household-level law in `EpiHouseholds`
# returns R*, the mean number of households one infected household infects — so
# a model's threshold is asked for the same way however its offspring are
# specified, and the answer always comes from the same object the simulation
# draws from.

"""
    reproduction_number(offspring)

The reproduction number of a branching process, read off its offspring
specification: the mean number of secondary units one infected unit produces.
An outbreak grows with positive probability only if it exceeds 1.

What counts as a unit depends on the specification. For an offspring
distribution, or a single-type model carrying one, the unit is a case and the
number is the distribution's mean. For a household-level offspring law
(`EpiHouseholds.household_offspring`) the unit is a household and the number is
R*, the mean number of households infected by the members of one infected
household.
"""
function reproduction_number end

reproduction_number(d::DiscreteUnivariateDistribution) = mean(d)

function reproduction_number(model::Union{TransmissionModel, ModelSpec})
    return reproduction_number(single_type_offspring(model))
end
