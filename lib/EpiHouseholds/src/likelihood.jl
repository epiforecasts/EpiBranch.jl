# ── Household pairwise survival likelihood ───────────────────────────
#
# Household methods for EpiBranch's pairwise survival likelihood. The density,
# its compiled pair layout and the community hazard term work for any contact
# structure and live in EpiBranch. A household population supplies its partition
# as the contact structure: household-mates are each other's possible infectors.

# ── The household infection layer ────────────────────────────────────

"""
    HouseholdInfections(household_of, infection_time, infectious_time, removal_time, is_index;
                        obs_end = Inf, followup_end = Inf)

The [`InfectionLayer`](@ref) of a household outbreak. Its contact structure is
`household_of`, the household of each individual: household-mates are each
other's possible infectors. The per-individual vectors, `obs_end` and
`followup_end` are as described for `InfectionLayer`. Read one out of a
simulation with [`household_infections`](@ref), or augment it in inference.
"""
struct HouseholdInfections{T <: Real} <: InfectionLayer
    household_of::Vector{Int}
    infection_time::Vector{T}
    infectious_time::Vector{T}
    removal_time::Vector{T}
    is_index::Vector{Bool}
    obs_end::T
    followup_end::T
end

function HouseholdInfections(household_of, infection_time, infectious_time,
        removal_time, is_index; obs_end = Inf, followup_end = Inf)
    fields = _infection_layer_fields(length(household_of), infection_time,
        infectious_time, removal_time, is_index; obs_end, followup_end)
    return HouseholdInfections(collect(Int, household_of), fields...)
end

Base.length(d::HouseholdInfections) = length(d.household_of)

# Household-mates are each other's possible infectors.
EpiBranch.contact_structure(d::HouseholdInfections) = d.household_of

"""
    household_infections(state, model::ModelSpec; obs_end = model.process.obs_end,
                         followup_end = Inf) -> HouseholdInfections

Read the [`InfectionLayer`](@ref) out of a `state` simulated from `model`, with
each member's household as the contact structure. The infectious windows are
read as described for `InfectionLayer`, which makes the
`simulate → loglikelihood` round trip exact. A bare `HouseholdProcess` is
accepted too (its window opens at `:infection`, and it has no interventions).
"""
function household_infections(state::SimulationState,
        model::ModelSpec{<:HouseholdProcess}; obs_end = model.process.obs_end,
        followup_end = Inf)
    household_of = [ind.state[:household]::Int for ind in state.individuals]
    columns = _infection_layer_columns(state, model)
    return HouseholdInfections(household_of, columns...; obs_end, followup_end)
end

function household_infections(state::SimulationState, process::HouseholdProcess;
        kwargs...)
    return household_infections(state, ModelSpec(process); kwargs...)
end

"""
    loglikelihood(data::HouseholdInfections, model::HouseholdProcess) -> Float64

The contact-process log-density of `model`'s kernel given the infection layer
`data`: `pairwise_surv_loglik(model.kernel, data; external_hazard =
model.external_hazard)`.
"""
function Distributions.loglikelihood(data::HouseholdInfections, model::HouseholdProcess)
    pairwise_surv_loglik(model.kernel, data; external_hazard = model.external_hazard)
end

function Distributions.loglikelihood(data::HouseholdInfections,
        model::ModelSpec{<:HouseholdProcess})
    loglikelihood(data, model.process)
end

# ── Compiled pair layout ─────────────────────────────────────────────

"""
    HouseholdPairsLayout

The compiled pair layout for a household population. It is another name for
EpiBranch's [`ContactPairsLayout`](@ref), used when the layout is built from a
household partition. Each row is one ordered (susceptible, household-mate) pair
that the likelihood scores.

Build it with [`compile_household_pairs`](@ref).
"""
const HouseholdPairsLayout = ContactPairsLayout

"""
    compile_household_pairs(household_of, is_index, infected; external=false)
    compile_household_pairs(data::HouseholdInfections; external=false)

[`compile_contact_pairs`](@ref) on a household partition, where household-mates
are each other's possible infectors. The arguments and the layout are as
described there. Evaluate the result with
`pairwise_surv_loglik(kernel, data, layout; external_hazard)`.
"""
function compile_household_pairs(household_of::AbstractVector{<:Integer},
        is_index::AbstractVector{Bool},
        infected::AbstractVector{Bool};
        external::Bool = false)
    return compile_contact_pairs(household_of, is_index, infected; external)
end

function compile_household_pairs(d::HouseholdInfections; external::Bool = false)
    return compile_contact_pairs(d; external)
end
