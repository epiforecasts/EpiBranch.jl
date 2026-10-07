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

Household outbreak data for the pairwise likelihood: for each person, their
household, when they were infected, when their infectious period started and
ended, and whether they were an index case. Household members are each other's
possible infectors. `household_of` gives each person's household; the other
per-person vectors, `obs_end`, `followup_end` and `host_times` are as described
for [`InfectionLayer`](@ref). Read one from a simulation with
[`household_infections`](@ref), or build it from data, imputing unobserved
infection times in inference.
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

Collect who was infected when from a household outbreak simulated from
`model` (infection times, start and end of each infectious period, index
cases), in the form the pairwise likelihood needs. The infectious periods are
the ones the simulation used, as described for [`InfectionLayer`](@ref).
Besides the infectious periods, the event times listed below are recorded, and
`loglikelihood` applies interventions that change susceptibility, such as
vaccination, from them. Any other effect on transmission must be built into
the kernel passed to the likelihood. A bare `HouseholdProcess` is
accepted too (its infectious period starts at `:infection`, and it has no
interventions).

`host_times` names further per-person event times to record, such as
`(:onset_time,)`, read from each person's state (`missing` where a person has
none), for a [`PairKernel`](@ref) to use. The times the model's interventions
need (see [`susceptibility_host_times`](@ref
EpiBranch.susceptibility_host_times)), such as a vaccinee's
`:immunity_time`, are recorded as well.
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
        host_times = _layer_host_times(state, model, host_times)
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

How the household likelihood conditions on the index case: which member of
each household is taken as given rather than explained by transmission within
the household. [`RecruitedIndex`](@ref) and [`EarliestInfected`](@ref) are the
two supplied; a rule of your own needs only a [`condition_mask`](@ref
EpiHouseholds.condition_mask) method.
"""
abstract type ConditionOn end

"""
    RecruitedIndex()

Condition each household on its recruited index case, `data.is_index`, as
recorded.
"""
struct RecruitedIndex <: ConditionOn end

"""
    EarliestInfected()

Condition each household on whichever member has the earliest
`infection_time` (on a tie, the lowest person number). This is worked out from
`data` on every call, so the conditioned member can change as imputed
infection times change.
"""
struct EarliestInfected <: ConditionOn end

"""
    condition_mask(rule::ConditionOn, data::HouseholdInfections) -> AbstractVector{Bool}

Which person in each household `rule` conditions on: a vector shaped like
`data.is_index`, `true` for each household's conditioned member and `false`
elsewhere. One method per rule.
"""
condition_mask(::RecruitedIndex, data::HouseholdInfections) = data.is_index
function condition_mask(::EarliestInfected, data::HouseholdInfections)
    return _earliest_infected(
        data.household_of, data.infection_time, .!isnan.(data.infection_time)
    )
end

"""
    loglikelihood(data::HouseholdInfections, model::HouseholdProcess;
                  condition_on = RecruitedIndex(), susceptibility = nothing) -> Float64
    loglikelihood(data::HouseholdInfections, model::ModelSpec{<:HouseholdProcess};
                  condition_on = RecruitedIndex()) -> Float64

Log-likelihood of household outbreak data under `model`'s contact interval and
community hazard, conditioning on index cases as `condition_on` says (see
[`compile_household_pairs`](@ref)). It is
`pairwise_surv_loglik(model.kernel, data, layout; external_hazard = model.external_hazard, susceptibility)`.
For a `ModelSpec`, `susceptibility` is the model's interventions, so a
vaccination in the model is evaluated from the immunity times
[`household_infections`](@ref) recorded.
"""
function Distributions.loglikelihood(
        data::HouseholdInfections, model::HouseholdProcess;
        condition_on::ConditionOn = RecruitedIndex(), susceptibility = nothing
    )
    layout = compile_household_pairs(
        data; external = _ext_active(model.external_hazard), condition_on
    )
    return pairwise_surv_loglik(
        model.kernel, data, layout; external_hazard = model.external_hazard, susceptibility
    )
end

function Distributions.loglikelihood(
        data::HouseholdInfections,
        model::ModelSpec{<:HouseholdProcess};
        condition_on::ConditionOn = RecruitedIndex()
    )
    EpiBranch._validate_infection_likelihood(model)
    return loglikelihood(
        data, model.process; condition_on, susceptibility = model.interventions
    )
end

# ── Compiled pair layout ─────────────────────────────────────────────

"""
    HouseholdPairsLayout

The list of who could have infected whom in a household population, for the
pairwise likelihood: each row is one ordered (susceptible, household member)
pair. It is another name for EpiBranch's [`ContactPairsLayout`](@ref), built
from households. Build it with [`compile_household_pairs`](@ref).
"""
const HouseholdPairsLayout = ContactPairsLayout

"""
    compile_household_pairs(household_of, is_index, infected; external=false)
    compile_household_pairs(data::HouseholdInfections; external=false, condition_on=RecruitedIndex())

List once who could have infected whom in a household population, where
household members are each other's possible infectors; this is
[`compile_contact_pairs`](@ref) for households, with the arguments described
there. Evaluate the result with
`pairwise_surv_loglik(kernel, data, layout; external_hazard)`.

`condition_on` is a [`ConditionOn`](@ref) rule saying how to condition on the
index case: which member of each household the likelihood takes as given,
when there is no community hazard (`external = false`; with one, every
infection is explained and the rule has no effect). [`RecruitedIndex`](@ref),
the default, conditions on the recruited index case, `data.is_index`, fixed
when the data are read. [`EarliestInfected`](@ref) conditions on the household
member with the earliest `infection_time` in `data`, worked out on every call.
Use it when the recruited index case need not be the first member infected,
which is expected in real recruited households; otherwise an imputed infection
time earlier than the recruited index case's makes the likelihood impossible
(`-Inf`). Because the conditioned member can change between calls, build a
fresh layout for `EarliestInfected` on every evaluation rather than reusing
one across imputed infection times.
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
