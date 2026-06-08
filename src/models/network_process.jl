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
- `transmission_probability` — per-edge probability that an infectious
  node infects a susceptible neighbour on contact. A `Real`, or a
  function `(rng, parent, neighbour_node) -> Real`. Node susceptibility
  and interventions apply on top of this through the usual competing
  risks.
- `generation_time` — `Distribution` or `Function`, same semantics as
  `BranchingProcess`.

# Examples

```julia
# Ring of 5 nodes
adjacency = [[2, 5], [1, 3], [2, 4], [3, 5], [4, 1]]
model = NetworkProcess(adjacency, 0.5, LogNormal(1.6, 0.5))
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
struct NetworkProcess{G, T} <: TransmissionModel
    adjacency::Vector{Vector{Int}}
    transmission_probability::T
    generation_time::G

    # Single inner constructor: accepts any adjacency-list form,
    # normalises neighbour vectors to `Vector{Int}`, and defaults to no
    # generation time. Defining it suppresses the auto-generated outer
    # constructors, avoiding ambiguity with the matrix constructor.
    function NetworkProcess(adjacency::AbstractVector{<:AbstractVector{<:Integer}},
            transmission_probability::T,
            generation_time::G = NoGenerationTime();
            population_size = nothing) where {G, T}
        if population_size !== nothing
            @warn "`population_size` is ignored by NetworkProcess: the population " *
                  "is the network itself (got $(length(adjacency)) nodes). Build a " *
                  "larger graph to model a larger population." maxlog=1
        end
        new{G, T}(Vector{Int}[Int.(nbrs) for nbrs in adjacency],
            transmission_probability, generation_time)
    end
end

population_size(::NetworkProcess) = NoPopulation()
n_types(::NetworkProcess) = 1

function Base.show(io::IO, m::NetworkProcess)
    n = length(m.adjacency)
    n_edges = sum(length, m.adjacency; init = 0) ÷ 2
    p_str = m.transmission_probability isa Real ?
            string(m.transmission_probability) : "Function"
    gt_str = m.generation_time isa NoGenerationTime ? "none" :
             m.generation_time isa Distribution ? string(typeof(m.generation_time)) :
             "Function"
    print(io,
        "NetworkProcess(nodes=$n, edges=$n_edges, p=$(p_str), generation_time=$(gt_str))")
end

"""
    NetworkProcess(A::AbstractMatrix, transmission_probability, gt)

Build a `NetworkProcess` from an adjacency matrix. `A[i, j]` nonzero
means an edge between `i` and `j`. The matrix is read as undirected
(an edge is added whenever either `A[i, j]` or `A[j, i]` is nonzero).
"""
function NetworkProcess(A::AbstractMatrix, transmission_probability,
        gt::Union{Distribution, Function, NoGenerationTime} = NoGenerationTime();
        population_size = nothing)
    n = size(A, 1)
    size(A, 2) == n || throw(ArgumentError(
        "adjacency matrix must be square, got $(size(A))"))
    adjacency = [Int[] for _ in 1:n]
    for i in 1:n, j in (i + 1):n

        if !iszero(A[i, j]) || !iszero(A[j, i])
            push!(adjacency[i], j)
            push!(adjacency[j], i)
        end
    end
    NetworkProcess(adjacency, transmission_probability, gt; population_size)
end

# ── Node bookkeeping ─────────────────────────────────────────────────

network_node(ind::Individual) = get(ind.state, :network_node, 0)

"""Edge transmission probability for a parent → neighbour-node contact.
Supports a constant or a `(rng, parent, neighbour_node)` function."""
_edge_probability(p::Real, rng, parent, node) = float(p)
_edge_probability(p::Function, rng, parent, node) = float(p(rng, parent, node))

# The generation loop for NetworkProcess lives in
# `models/network_simulate.jl` (a dedicated `simulate` method), because
# pre-instantiated nodes need engine helpers defined after this file.
