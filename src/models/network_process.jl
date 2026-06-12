# ── NetworkProcess type and constructors ────────────────────────────

"""
Transmission over a fixed contact network.

`NetworkProcess` transmits along the edges of a fixed graph. Each node
is a person; an infectious node infects its neighbours, and each node
can be infected at most once. The run produces a network of
who-infected-whom.

The graph plays the role the offspring distribution plays in
[`BranchingProcess`](@ref). Timing, interventions, attributes, and
competing-risks resolution route through the same engine, so
`Isolation`, `ContactTracing`, `RingVaccination`,
`clinical_presentation`, and the rest apply directly.

# Fields

- `adjacency::Vector{Vector{Int}}` — `adjacency[i]` lists the nodes
  connected to node `i` (1-based).
- `edge_probability::Vector{Vector{Float64}}` — per-edge transmission
  probability, parallel to `adjacency`: `edge_probability[i][k]` is the
  probability that node `i`, when infectious, infects its `k`-th listed
  neighbour on contact. Node susceptibility and interventions apply on
  top of these through the usual competing risks.
- `generation_time` — `Distribution` or `Function`, same semantics as
  `BranchingProcess`.

# Construction

A scalar gives every edge the same probability; an adjacency matrix
carries a probability per edge (`A[i, j]` is the transmission
probability of the edge between `i` and `j`); or pass weights parallel
to the adjacency list.

# Examples

```julia
# Ring of 5 nodes, uniform per-edge probability
adjacency = [[2, 5], [1, 3], [2, 4], [3, 5], [4, 1]]
model = NetworkProcess(adjacency, 0.5, LogNormal(1.6, 0.5))

# Weighted edges via a matrix: A[i, j] is the per-edge probability
A = [0.0 0.5; 0.5 0.0]
model = NetworkProcess(A, LogNormal(1.6, 0.5))
```

# Population and attributes

The graph is the population: `population_size` equals the number of
nodes. (Passing a separate `population_size` warns and leaves the node
count in force.) Each node is built once with a stable identity, so its
attributes (age, type, susceptibility) are drawn once and belong to the
node for the whole run.

When several infectious neighbours reach a susceptible node in the same
generation, competing risks are evaluated for each incoming edge and
the node is infected if any edge transmits.

`clinical_presentation` records `:incubation_period` on each node, and
`:onset_time` is set to `infection_time + incubation_period` when the
node is infected.
"""
struct NetworkProcess{G} <: TransmissionModel
    adjacency::Vector{Vector{Int}}
    edge_probability::Vector{Vector{Float64}}
    generation_time::G
    forcings::Forcings

    # Single inner constructor: accepts any adjacency-list form with
    # per-edge probabilities parallel to it, normalises the neighbour
    # vectors, and defaults to no generation time. Defining it suppresses
    # the auto-generated outer constructors, avoiding ambiguity with the
    # scalar and matrix constructors.
    function NetworkProcess(adjacency::AbstractVector{<:AbstractVector{<:Integer}},
            edge_probability::AbstractVector{<:AbstractVector{<:Real}},
            generation_time::G = NoGenerationTime();
            population_size = nothing, forcing_kwargs...) where {G}
        length(adjacency) == length(edge_probability) || throw(ArgumentError(
            "adjacency and edge_probability must have the same number of nodes"))
        adj = Vector{Int}[Int.(nbrs) for nbrs in adjacency]
        ep = Vector{Float64}[Float64.(w) for w in edge_probability]
        for i in eachindex(adj)
            length(adj[i]) == length(ep[i]) || throw(ArgumentError(
                "node $i: adjacency and edge_probability have different lengths"))
            all(p -> 0.0 <= p <= 1.0, ep[i]) || throw(ArgumentError(
                "edge probabilities must be in [0, 1]"))
        end
        if population_size !== nothing
            @warn "`population_size` is ignored by NetworkProcess: the population " *
                  "is the network itself (got $(length(adjacency)) nodes). Build a " *
                  "larger graph to model a larger population." maxlog=1
        end
        new{G}(adj, ep, generation_time, make_forcings(; forcing_kwargs...))
    end
end

# The model carries its forcings; shared accessors read them from here.
forcings(m::NetworkProcess) = m.forcings

"""
    NetworkProcess(adjacency, p::Real, gt; population_size = nothing)

Give every edge the same transmission probability `p`.
"""
function NetworkProcess(adjacency::AbstractVector{<:AbstractVector{<:Integer}},
        p::Real, generation_time = NoGenerationTime();
        population_size = nothing, forcing_kwargs...)
    edge_probability = [fill(float(p), length(nbrs)) for nbrs in adjacency]
    NetworkProcess(adjacency, edge_probability, generation_time;
        population_size, forcing_kwargs...)
end

population_size(::NetworkProcess) = NoPopulation()
n_types(::NetworkProcess) = 1

# A network has no single per-case offspring law: each node's offspring
# is set by its neighbours and their edge probabilities, not a shared
# distribution. The analytical helpers that route through
# `single_type_offspring` (chain_size_distribution, offspring_distribution,
# extinction_probability, …) therefore can't apply; say so plainly rather
# than letting the generic "no offspring field" extension hint surface.
function single_type_offspring(::NetworkProcess)
    throw(ArgumentError(
        "NetworkProcess has no single offspring law: each node's offspring " *
        "is determined by its neighbours in the graph. Use " *
        "chain_length_distribution or simulate(); the chain-size and " *
        "offspring analytical paths need a closed-form offspring distribution."))
end

function Base.show(io::IO, m::NetworkProcess)
    n = length(m.adjacency)
    n_edges = sum(length, m.adjacency; init = 0) ÷ 2
    gt_str = m.generation_time isa NoGenerationTime ? "none" :
             m.generation_time isa Distribution ? string(typeof(m.generation_time)) :
             "Function"
    print(io,
        "NetworkProcess(nodes=$n, edges=$n_edges, generation_time=$(gt_str))")
end

"""
    NetworkProcess(A::AbstractMatrix, gt; population_size = nothing)

Build a `NetworkProcess` from a weighted adjacency matrix. A nonzero
`A[i, j]` means an edge between `i` and `j` whose value is the per-edge
transmission probability. The matrix is read as undirected: each edge
takes `A[i, j]` if nonzero, otherwise `A[j, i]`.
"""
function NetworkProcess(A::AbstractMatrix,
        gt::Union{Distribution, Function, NoGenerationTime} = NoGenerationTime();
        population_size = nothing, forcing_kwargs...)
    n = size(A, 1)
    size(A, 2) == n || throw(ArgumentError(
        "adjacency matrix must be square, got $(size(A))"))
    adjacency = [Int[] for _ in 1:n]
    edge_probability = [Float64[] for _ in 1:n]
    for i in 1:n, j in (i + 1):n

        w = !iszero(A[i, j]) ? A[i, j] : A[j, i]
        if !iszero(w)
            push!(adjacency[i], j)
            push!(edge_probability[i], float(w))
            push!(adjacency[j], i)
            push!(edge_probability[j], float(w))
        end
    end
    NetworkProcess(adjacency, edge_probability, gt; population_size, forcing_kwargs...)
end

# ── Node bookkeeping ─────────────────────────────────────────────────

network_node(ind::Individual) = get(ind.state, :network_node, 0)

# The generation loop for NetworkProcess lives in
# `models/network_simulate.jl` (a dedicated `simulate` method), because
# pre-instantiated nodes need engine helpers defined after this file.
