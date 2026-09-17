# ── Household pairwise survival likelihood ───────────────────────────
#
# Household methods for EpiBranch's pairwise survival likelihood. The density,
# its compiled pair layout and the external hazard term work for any contact
# structure and live in EpiBranch. A household population supplies its partition
# as the contact structure, so household-mates are each other's possible
# infectors.

# ── The household infection layer ────────────────────────────────────

"""
    HouseholdInfections(household_of, infection_time, infectious_time, removal_time, is_index;
                        obs_end = Inf, followup_end = Inf)

The infection layer of a household outbreak: per individual, their household,
infection time (`NaN` if never infected), infectiousness onset and removal (the
infectious-window endpoints), and whether they were introduced from outside the
household. `obs_end` is when community introductions stop, and `followup_end`
when observation of the data ends: the likelihood ignores infections and
exposure after it, so an individual still infectious then can keep a removal
time of `Inf` (see [`InfectionLayer`](@ref)).

These are the latent quantities the contact process is a density over — read out
of a `simulate` round-trip with [`household_infections`](@ref), or augmented in
inference. Onsets, tests and other observables are *not* here: they are the
progression's outputs, conditioned separately.
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

# `obs_end` is when community introductions stop; spread within households goes
# on after it. Only used when the model has an external hazard, and may be left
# `Inf`.
function HouseholdInfections(household_of, infection_time, infectious_time,
        removal_time, is_index; obs_end = Inf, followup_end = Inf)
    T = promote_type(eltype(infection_time), eltype(infectious_time),
        eltype(removal_time), typeof(obs_end), typeof(followup_end), Float64)
    return HouseholdInfections{T}(collect(Int, household_of),
        Vector{T}(infection_time),
        Vector{T}(infectious_time),
        Vector{T}(removal_time),
        Vector{Bool}(is_index),
        T(obs_end),
        T(followup_end))
end

Base.length(d::HouseholdInfections) = length(d.household_of)

# Household-mates are each other's possible infectors.
EpiBranch.contact_structure(d::HouseholdInfections) = d.household_of

"""
    household_infections(state, model::ModelSpec; obs_end = model.process.obs_end,
                         followup_end = Inf) -> HouseholdInfections

Read the infection layer out of a simulated `state`: each member's household,
infection time, infectiousness onset (the infectious-window `from` state),
removal and index status. A member is removed at the earliest of its `until`
states and the time the model's interventions take it out of transmission, such
as by isolation or quarantine after tracing. The infectious window is read from
the same composed progression and interventions the simulation used, so the
`simulate → loglikelihood` round trip is exact. A bare `HouseholdProcess` is
accepted too (its window opens at `:infection`, and it has no interventions).
Pass `followup_end` to score the outbreak as if observation had stopped then.
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
`data` — sugar for `pairwise_surv_loglik(model.kernel, data; external_hazard =
model.external_hazard)`, and the exact `simulate → loglikelihood` round trip. The
observed onsets/tests are conditioned separately through the progression — there
is deliberately no `loglikelihood(onsets, model)`, since the latent infections
cannot be marginalised in closed form.
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
that the contact process scores.

Build it with [`compile_household_pairs`](@ref).
"""
const HouseholdPairsLayout = ContactPairsLayout

"""
    compile_household_pairs(household_of, is_index, infected; external=false)
    compile_household_pairs(data::HouseholdInfections; external=false)

Pre-compute the structural pair list for the inference fast path. `infected`
is the static at-risk mask — true iff the host appears in the posterior as
an infected case (its `infection_time` will be augmented). The single-arg
form reads the mask off `data` as `.!isnan.(data.infection_time)`.

With `external=true` each susceptible gets an additional row for the community
hazard term; otherwise index cases are conditioned on and contribute only as
infectors. This is [`compile_contact_pairs`](@ref) on the household partition.
Evaluate the result with
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
