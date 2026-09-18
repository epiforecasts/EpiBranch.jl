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
                      obs_end = Inf, followup_end = Inf)

The [`InfectionLayer`](@ref) of a network outbreak. Its contact structure is the
adjacency the outbreak spread over: `contacts[i]` lists the nodes `i` can
infect, as for [`NetworkProcess`](@ref), and a node's possible infectors are its
in-neighbours. The per-node vectors, `obs_end` and `followup_end` are as
described for `InfectionLayer`. Read one out of a simulation with
[`network_infections`](@ref), or augment it in inference.
"""
struct NetworkInfections{T <: Real} <: InfectionLayer
    contacts::Vector{Vector{Int}}
    infection_time::Vector{T}
    infectious_time::Vector{T}
    removal_time::Vector{T}
    is_index::Vector{Bool}
    obs_end::T
    followup_end::T
end

function NetworkInfections(contacts::AbstractVector{<:AbstractVector{<:Integer}},
        infection_time, infectious_time, removal_time, is_index; obs_end = Inf,
        followup_end = Inf)
    adj = contacts isa Vector{Vector{Int}} ? contacts :
          Vector{Int}[Int.(nbrs) for nbrs in contacts]
    fields = _infection_layer_fields(length(adj), infection_time, infectious_time,
        removal_time, is_index; obs_end, followup_end)
    return NetworkInfections(adj, fields...)
end

Base.length(d::NetworkInfections) = length(d.contacts)

# A node's possible infectors are the nodes that list it as a contact.
EpiBranch.contact_structure(d::NetworkInfections) = d.contacts

"""
    network_infections(state, model::ModelSpec{<:NetworkProcess};
                       obs_end = model.process.obs_end, followup_end = Inf) -> NetworkInfections
    network_infections(state, process::NetworkProcess) -> NetworkInfections

Read the [`InfectionLayer`](@ref) out of a `state` simulated from `model`, with
the model's adjacency as the contact structure. The infectious windows are read
as described for `InfectionLayer`. Additional hazard modifications require an
effective kernel when scoring; extraction records the windows only. A bare `NetworkProcess` is accepted too (its window opens at
`:infection`, and it has no interventions).
"""
function network_infections(state::SimulationState,
        model::ModelSpec{<:NetworkProcess}; obs_end = model.process.obs_end,
        followup_end = Inf)
    adjacency = model.process.adjacency
    n = length(adjacency)
    length(state.individuals) == n || throw(ArgumentError(
        "the state has $(length(state.individuals)) individuals but the network " *
        "has $n nodes"))
    columns = _infection_layer_columns(state, model)
    return NetworkInfections(adjacency, columns...; obs_end, followup_end)
end

function network_infections(state::SimulationState, process::NetworkProcess; kwargs...)
    return network_infections(state, ModelSpec(process); kwargs...)
end

"""
    loglikelihood(data::NetworkInfections, model::NetworkProcess) -> Real

The contact-process log-density of `model`'s kernel given the infection layer
`data`: `pairwise_surv_loglik(model.edge_kernel, data; external_hazard =
model.external_hazard)`. A per-edge kernel must be parallel to `data.contacts`.
"""
function Distributions.loglikelihood(data::NetworkInfections, model::NetworkProcess)
    return pairwise_surv_loglik(model.edge_kernel, data;
        external_hazard = model.external_hazard)
end

function Distributions.loglikelihood(data::NetworkInfections,
        model::ModelSpec{<:NetworkProcess})
    EpiBranch._validate_infection_likelihood(model)
    return loglikelihood(data, model.process)
end
