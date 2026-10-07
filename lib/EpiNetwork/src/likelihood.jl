# ── Network pairwise survival likelihood ─────────────────────────────
#
# Network methods for EpiBranch's pairwise survival likelihood. The density, its
# compiled pair layout and the community hazard term work for any contact
# structure and live in EpiBranch. A network supplies its adjacency as the
# contact structure: a node's possible infectors are its in-neighbours. The
# Sellke race in `network_simulate.jl` is the exact generative model of this
# likelihood, which makes `simulate → loglikelihood` an exact round trip.

"""
    NetworkInfections(contacts, infection_time, infectious_time, removal_time, is_index;
                      obs_end = Inf, followup_end = Inf, host_times = (;))

Network outbreak data for the pairwise likelihood: for each person, when they
were infected, when their infectious period started and ended, and whether
they were infected from outside the network. `contacts[i]` lists the people
`i` can infect, as for [`NetworkProcess`](@ref), and a person's possible
infectors are those who list them. The per-person vectors, `obs_end`,
`followup_end` and `host_times` are as described for [`InfectionLayer`](@ref).
Read one from a simulation with [`network_infections`](@ref), or build it from
data, imputing unobserved infection times in inference.
"""
struct NetworkInfections{T <: Real, H <: NamedTuple} <: InfectionLayer
    contacts::Vector{Vector{Int}}
    infection_time::Vector{T}
    infectious_time::Vector{T}
    removal_time::Vector{T}
    is_index::Vector{Bool}
    obs_end::T
    followup_end::T
    host_times::H
end

function NetworkInfections(
        contacts::AbstractVector{<:AbstractVector{<:Integer}},
        infection_time, infectious_time, removal_time, is_index; obs_end = Inf,
        followup_end = Inf, host_times = (;)
    )
    adj = contacts isa Vector{Vector{Int}} ? contacts :
        Vector{Int}[Int.(nbrs) for nbrs in contacts]
    fields = _infection_layer_fields(
        length(adj), infection_time, infectious_time,
        removal_time, is_index; obs_end, followup_end, host_times
    )
    return NetworkInfections(adj, fields...)
end

Base.length(d::NetworkInfections) = length(d.contacts)

# A node's possible infectors are the nodes that list it as a contact.
EpiBranch.contact_structure(d::NetworkInfections) = d.contacts

"""
    network_infections(state, model::ModelSpec{<:NetworkProcess};
                       obs_end = model.process.obs_end, followup_end = Inf,
                       host_times = ()) -> NetworkInfections
    network_infections(state, process::NetworkProcess) -> NetworkInfections

Collect who was infected when from a network outbreak simulated from `model`
(infection times, start and end of each infectious period, index cases), in
the form the pairwise likelihood needs, with the model's network as the
contact structure. The infectious periods are the ones the simulation used, as
described for [`InfectionLayer`](@ref). Only the infectious periods are
recorded, so any other effect on transmission must be built into the kernel
passed to the likelihood. A bare `NetworkProcess` is accepted too (its
infectious period starts at `:infection`, and it has no interventions).

`host_times` names further per-person event times to record, such as
`(:onset_time,)`, read from each person's state (`missing` where a person has
none), for a [`PairKernel`](@ref) to use. The times the model's interventions
need (see [`susceptibility_host_times`](@ref
EpiBranch.susceptibility_host_times)), such as a vaccinee's
`:immunity_time`, are recorded as well.
"""
function network_infections(
        state::SimulationState,
        model::ModelSpec{<:NetworkProcess}; obs_end = model.process.obs_end,
        followup_end = Inf, host_times = ()
    )
    adjacency = model.process.adjacency
    n = length(adjacency)
    length(state.individuals) == n || throw(
        ArgumentError(
            "the state has $(length(state.individuals)) individuals but the network " *
                "has $n nodes"
        )
    )
    columns = _infection_layer_columns(state, model)
    return NetworkInfections(
        adjacency, columns...; obs_end, followup_end,
        host_times = _layer_host_times(state, model, host_times)
    )
end

function network_infections(state::SimulationState, process::NetworkProcess; kwargs...)
    return network_infections(state, ModelSpec(process); kwargs...)
end

"""
    loglikelihood(data::NetworkInfections, model::NetworkProcess;
                  susceptibility = nothing) -> Real
    loglikelihood(data::NetworkInfections, model::ModelSpec{<:NetworkProcess}) -> Real

Log-likelihood of network outbreak data under `model`'s contact interval and
community hazard:
`pairwise_surv_loglik(model.edge_kernel, data; external_hazard = model.external_hazard, susceptibility)`.
A per-contact kernel must be parallel to `data.contacts`. For a `ModelSpec`,
`susceptibility` is the model's interventions, so a vaccination in the model
is evaluated from the immunity times [`network_infections`](@ref) recorded.
"""
function Distributions.loglikelihood(
        data::NetworkInfections, model::NetworkProcess;
        susceptibility = nothing
    )
    return pairwise_surv_loglik(
        model.edge_kernel, data;
        external_hazard = model.external_hazard, susceptibility
    )
end

function Distributions.loglikelihood(
        data::NetworkInfections,
        model::ModelSpec{<:NetworkProcess}
    )
    EpiBranch._validate_infection_likelihood(model)
    return loglikelihood(data, model.process; susceptibility = model.interventions)
end
