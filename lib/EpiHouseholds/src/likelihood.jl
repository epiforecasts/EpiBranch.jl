# ── Household pairwise survival likelihood ───────────────────────────
#
# Household methods for EpiBranch's pairwise survival likelihood. The density,
# its compiled pair layout and the community hazard term work for any contact
# structure and live in EpiBranch. A household population supplies its partition
# as the contact structure: household-mates are each other's possible infectors.

# ── The household infection layer ────────────────────────────────────

"""
    HouseholdInfections(household_of, infection_time, infectious_time, removal_time, is_index;
                        obs_end = Inf, followup_end = Inf, host_times = (;))

The [`InfectionLayer`](@ref) of a household outbreak. Its contact structure is
`household_of`, the household of each individual: household-mates are each
other's possible infectors. The per-individual vectors, `obs_end`, `followup_end`
and `host_times` are as described for `InfectionLayer`. Read one out of a
simulation with [`household_infections`](@ref), or augment it in inference.
"""
struct HouseholdInfections{T <: Real, H <: NamedTuple} <: InfectionLayer
    household_of::Vector{Int}
    infection_time::Vector{T}
    infectious_time::Vector{T}
    removal_time::Vector{T}
    is_index::Vector{Bool}
    obs_end::T
    followup_end::T
    host_times::H
end

function HouseholdInfections(
        household_of, infection_time, infectious_time,
        removal_time, is_index; obs_end = Inf, followup_end = Inf, host_times = (;)
    )
    fields = _infection_layer_fields(
        length(household_of), infection_time,
        infectious_time, removal_time, is_index; obs_end, followup_end, host_times
    )
    return HouseholdInfections(collect(Int, household_of), fields...)
end

Base.length(d::HouseholdInfections) = length(d.household_of)

# Household-mates are each other's possible infectors.
EpiBranch.contact_structure(d::HouseholdInfections) = d.household_of

"""
    household_infections(state, model::ModelSpec; obs_end = model.process.obs_end,
                         followup_end = Inf, host_times = ()) -> HouseholdInfections

Read the [`InfectionLayer`](@ref) out of a `state` simulated from `model`, with
each member's household as the contact structure. The infectious windows are
read as described for `InfectionLayer`. Additional hazard modifications require
an effective kernel when evaluating; extraction records the windows only. A bare `HouseholdProcess` is
accepted too (its window opens at `:infection`, and it has no interventions).
`host_times` names further per-member times to record, such as `(:onset_time,)`,
read from each member's state (`missing` where a member has none) for a live
[`PairKernel`](@ref) to read.
"""
function household_infections(
        state::SimulationState,
        model::ModelSpec{<:HouseholdProcess}; obs_end = model.process.obs_end,
        followup_end = Inf, host_times = ()
    )
    household_of = [ind.state[:household]::Int for ind in state.individuals]
    columns = _infection_layer_columns(state, model)
    return HouseholdInfections(
        household_of, columns...; obs_end, followup_end,
        host_times = _host_time_columns(state, host_times)
    )
end

function household_infections(
        state::SimulationState, process::HouseholdProcess;
        kwargs...
    )
    return household_infections(state, ModelSpec(process); kwargs...)
end

"""
    ConditionOn

Supertype for the rule choosing which host in each household the likelihood
does not need to explain. [`RecruitedIndex`](@ref) and
[`EarliestInfected`](@ref) are the two supplied; a rule of your own needs a
[`condition_mask`](@ref EpiHouseholds.condition_mask) method and nothing else.
"""
abstract type ConditionOn end

"""
    RecruitedIndex()

Condition each household on its recruited index, `data.is_index`, as read.
"""
struct RecruitedIndex <: ConditionOn end

"""
    EarliestInfected()

Condition each household on whichever member has the lowest `infection_time`,
ties keeping the lowest host id. Resolved from `data` on every call, so the
host can change between augmented draws.
"""
struct EarliestInfected <: ConditionOn end

"""
    condition_mask(rule::ConditionOn, data::HouseholdInfections) -> AbstractVector{Bool}

The `is_index`-shaped mask `rule` conditions on: `true` for each household's
conditioned host, `false` elsewhere. One method per rule.
"""
condition_mask(::RecruitedIndex, data::HouseholdInfections) = data.is_index
function condition_mask(::EarliestInfected, data::HouseholdInfections)
    return _earliest_infected(
        data.household_of, data.infection_time, .!isnan.(data.infection_time)
    )
end

"""
    loglikelihood(data::HouseholdInfections, model::HouseholdProcess; condition_on = RecruitedIndex()) -> Float64

The contact-process log-density of `model`'s kernel given the infection layer
`data`, on the layout [`compile_household_pairs`](@ref) builds for
`condition_on` (see there): `pairwise_surv_loglik(model.kernel, data, layout;
external_hazard = model.external_hazard)`.
"""
function Distributions.loglikelihood(
        data::HouseholdInfections, model::HouseholdProcess;
        condition_on::ConditionOn = RecruitedIndex()
    )
    layout = compile_household_pairs(
        data; external = _ext_active(model.external_hazard), condition_on
    )
    return pairwise_surv_loglik(
        model.kernel, data, layout; external_hazard = model.external_hazard
    )
end

function Distributions.loglikelihood(
        data::HouseholdInfections,
        model::ModelSpec{<:HouseholdProcess};
        condition_on::ConditionOn = RecruitedIndex()
    )
    EpiBranch._validate_infection_likelihood(model)
    return loglikelihood(data, model.process; condition_on)
end

# ── Compiled pair layout ─────────────────────────────────────────────

"""
    HouseholdPairsLayout

The compiled pair layout for a household population. It is another name for
EpiBranch's [`ContactPairsLayout`](@ref), used when the layout is built from a
household partition. Each row is one ordered (susceptible, household-mate) pair
that the likelihood covers.

Build it with [`compile_household_pairs`](@ref).
"""
const HouseholdPairsLayout = ContactPairsLayout

"""
    compile_household_pairs(household_of, is_index, infected; external=false)
    compile_household_pairs(data::HouseholdInfections; external=false, condition_on=RecruitedIndex())

[`compile_contact_pairs`](@ref) on a household partition, where household-mates
are each other's possible infectors. The arguments and the layout are as
described there. Evaluate the result with
`pairwise_surv_loglik(kernel, data, layout; external_hazard)`.

`condition_on` is a [`ConditionOn`](@ref) rule choosing which host in each
household the likelihood does not need to explain, when there is no community
hazard (`external = false`; with one every host is explained and the rule has
no effect). [`RecruitedIndex`](@ref), the default, conditions on the recruited
index, `data.is_index`, fixed at read time. [`EarliestInfected`](@ref)
conditions on whichever household member has the lowest `infection_time` in
`data`, resolved afresh on every call. Use it when the recruited index need not
be the first household member infected, which is expected in real recruited
households and can otherwise turn an infection time augmented below the
recruited index's into an impossible (`-Inf`) configuration. Because the
conditioned host can change between calls, compile a fresh layout for
`EarliestInfected` on every evaluation rather than reusing one across augmented
draws.
"""
function compile_household_pairs(
        household_of::AbstractVector{<:Integer},
        is_index::AbstractVector{Bool},
        infected::AbstractVector{Bool};
        external::Bool = false
    )
    return compile_contact_pairs(household_of, is_index, infected; external)
end

function compile_household_pairs(
        d::HouseholdInfections; external::Bool = false,
        condition_on::ConditionOn = RecruitedIndex()
    )
    infected = .!isnan.(d.infection_time)
    return compile_contact_pairs(
        d.household_of, condition_mask(condition_on, d), infected; external
    )
end

# The earliest-infected member of each household named in `household_of`,
# among the hosts `infected` marks, as an `is_index`-shaped mask: `true` for
# each household's earliest case, `false` elsewhere (including households with
# no infected member). Ties keep the lowest host id.
function _earliest_infected(household_of, infection_time, infected::AbstractVector{Bool})
    n = length(household_of)
    earliest_time = Dict{Int, eltype(infection_time)}()
    earliest_host = Dict{Int, Int}()
    for i in 1:n
        infected[i] || continue
        h = household_of[i]
        t = infection_time[i]
        if !haskey(earliest_time, h) || t < earliest_time[h]
            earliest_time[h] = t
            earliest_host[h] = i
        end
    end
    is_earliest = falses(n)
    is_earliest[collect(values(earliest_host))] .= true
    return is_earliest
end
