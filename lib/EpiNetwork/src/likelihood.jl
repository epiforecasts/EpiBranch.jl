# ── Network pairwise survival likelihood ─────────────────────────────
#
# The network-facing side of EpiBranch's pairwise survival likelihood. The
# density, its compiled pair layout and the external hazard term are
# structure-agnostic and live in EpiBranch; a network supplies its adjacency as
# the contact structure, so a node's possible infectors are its in-neighbours.
# The Sellke race in `network_simulate.jl` is the exact generative model of this
# likelihood, so `simulate → loglikelihood` is an exact round trip.

"""
    NetworkInfections(contacts, infection_time, infectious_time, removal_time, is_index;
                      obs_end = Inf)

The infection layer of a network outbreak: the adjacency it spread over
(`contacts[i]` lists the nodes `i` can infect, as for [`NetworkProcess`](@ref))
and, per node, its infection time (`NaN` if never infected), the opening and
closing of its infectious window (`removal_time = Inf` when right-censored), and
whether it was introduced from outside the network. `obs_end` is the end of
follow-up over which a community hazard acts; it is only read when there is one.

These are the latent quantities the contact process is a density over, read out
of a simulation with [`network_infections`](@ref) or augmented in inference.
Observables such as onsets are the progression's outputs and are conditioned
separately.
"""
struct NetworkInfections{T <: Real} <: InfectionLayer
    contacts::Vector{Vector{Int}}
    infection_time::Vector{T}
    infectious_time::Vector{T}
    removal_time::Vector{T}
    is_index::Vector{Bool}
    obs_end::T
    function NetworkInfections{T}(contacts, infection_time, infectious_time,
            removal_time, is_index, obs_end) where {T <: Real}
        n = length(contacts)
        all(length(v) == n
        for v in (infection_time, infectious_time, removal_time, is_index)) ||
            throw(ArgumentError("contacts and the per-node vectors must cover the " *
                                "same nodes"))
        return new{T}(contacts, infection_time, infectious_time, removal_time,
            is_index, obs_end)
    end
end

function NetworkInfections(contacts::AbstractVector{<:AbstractVector{<:Integer}},
        infection_time, infectious_time, removal_time, is_index; obs_end = Inf)
    T = promote_type(eltype(infection_time), eltype(infectious_time),
        eltype(removal_time), typeof(obs_end), Float64)
    adj = contacts isa Vector{Vector{Int}} ? contacts :
          Vector{Int}[Int.(nbrs) for nbrs in contacts]
    return NetworkInfections{T}(adj,
        Vector{T}(infection_time),
        Vector{T}(infectious_time),
        Vector{T}(removal_time),
        Vector{Bool}(is_index),
        T(obs_end))
end

Base.length(d::NetworkInfections) = length(d.contacts)

# A node's possible infectors are the nodes that list it as a contact.
EpiBranch.contact_structure(d::NetworkInfections) = d.contacts

"""
    network_infections(state, model::ModelSpec{<:NetworkProcess}) -> NetworkInfections
    network_infections(state, process::NetworkProcess) -> NetworkInfections

Read the infection layer out of a simulated `state`: the model's adjacency and,
per node, its infection time, infectiousness onset (the infectious-window `from`
state) and removal (the earliest of its `until` states), and index status. The
window is read from the same composed progression the simulation used, so the
`simulate → loglikelihood` round trip is exact. A bare `NetworkProcess` is
accepted too (its window opens at `:infection`).
"""
function network_infections(state::SimulationState,
        model::ModelSpec{<:NetworkProcess}; obs_end = model.process.obs_end)
    process = model.process
    from = _resolve_infectious_from(process.from, model.progression)
    until = process.until
    n = length(process.adjacency)
    length(state.individuals) == n || throw(ArgumentError(
        "the state has $(length(state.individuals)) individuals but the network " *
        "has $n nodes"))
    infection = fill(NaN, n)
    infectious = fill(NaN, n)
    removal = fill(Inf, n)
    index = falses(n)
    for (k, ind) in enumerate(state.individuals)
        if get(ind.state, :infected, false)
            infection[k] = ind.infection_time
            infectious[k] = _window_open(ind, from)
            removal[k] = _window_close(ind, until)
            index[k] = get(ind.state, :index, false)
        end
    end
    return NetworkInfections(process.adjacency, infection, infectious, removal, index;
        obs_end)
end

function network_infections(state::SimulationState, process::NetworkProcess; kwargs...)
    return network_infections(state, ModelSpec(process); kwargs...)
end

"""
    loglikelihood(data::NetworkInfections, model::NetworkProcess) -> Real

The contact-process log-density of `model`'s kernel given the infection layer
`data`: `pairwise_surv_loglik(model.edge_kernel, data; external_hazard =
model.external_hazard)`, with each node's possible infectors its in-neighbours in
`data.contacts`. A per-edge kernel must be parallel to that adjacency. This is
the exact `simulate → loglikelihood` round trip; observed onsets and tests are
conditioned separately through the progression.
"""
function Distributions.loglikelihood(data::NetworkInfections, model::NetworkProcess)
    return pairwise_surv_loglik(model.edge_kernel, data;
        external_hazard = model.external_hazard)
end

function Distributions.loglikelihood(data::NetworkInfections,
        model::ModelSpec{<:NetworkProcess})
    return loglikelihood(data, model.process)
end
