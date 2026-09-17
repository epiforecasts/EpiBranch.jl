# ── Network pairwise survival likelihood ─────────────────────────────
#
# Network methods for EpiBranch's pairwise survival likelihood. The density, its
# compiled pair layout and the external hazard term work for any contact
# structure and live in EpiBranch. A network supplies its adjacency as the
# contact structure, so a node's possible infectors are its in-neighbours. The
# Sellke race in `network_simulate.jl` is the exact generative model of this
# likelihood, so `simulate → loglikelihood` is an exact round trip.

"""
    NetworkInfections(contacts, infection_time, infectious_time, removal_time, is_index;
                      obs_end = Inf, followup_end = Inf)

The infection layer of a network outbreak: the adjacency it spread over
(`contacts[i]` lists the nodes `i` can infect, as for [`NetworkProcess`](@ref))
and, per node, its infection time (`NaN` if never infected), the opening and
closing of its infectious window, and
whether it was introduced from outside the network. `obs_end` is the time
community introductions stop, as for [`NetworkProcess`](@ref); spread along the
edges continues after it. It is only read when there is a community hazard.
`followup_end` is when observation of the data ends: the likelihood ignores
infections and exposure after it, so a node still infectious then can keep a
removal time of `Inf` (see [`InfectionLayer`](@ref)).

The contact process is a density over these latent quantities. Read them out of
a simulation with [`network_infections`](@ref), or augment them in inference.
Observables such as onsets are outputs of the progression and are conditioned
separately.
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

Read the infection layer out of a simulated `state`: the model's adjacency and,
per node, its infection time, infectiousness onset (the infectious-window `from`
state), removal and index status. A node is removed at the earliest of its
`until` states and the time the model's interventions take it out of
transmission, such as by isolation or quarantine after tracing. The window is
read from the same composed progression and interventions the simulation used,
so the `simulate → loglikelihood` round trip is exact. A bare `NetworkProcess`
is accepted too (its window opens at `:infection`, and it has no interventions).
Pass `followup_end` to score the outbreak as if observation had stopped then.
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
model.external_hazard)`, with each node's possible infectors its in-neighbours in
`data.contacts`. A per-edge kernel must be parallel to that adjacency. On an
infection layer read from a simulation of `model` the round trip is exact.
Observed onsets and tests are conditioned separately through the progression.
"""
function Distributions.loglikelihood(data::NetworkInfections, model::NetworkProcess)
    return pairwise_surv_loglik(model.edge_kernel, data;
        external_hazard = model.external_hazard)
end

function Distributions.loglikelihood(data::NetworkInfections,
        model::ModelSpec{<:NetworkProcess})
    return loglikelihood(data, model.process)
end
