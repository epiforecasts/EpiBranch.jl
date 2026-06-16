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
struct NetworkProcess{G, A, O} <: TransmissionModel
    adjacency::Vector{Vector{Int}}
    edge_probability::Vector{Vector{Float64}}
    generation_time::G
    interventions::Vector{AbstractIntervention}
    attributes::A
    observation::O

    # Single inner constructor: accepts any adjacency-list form with
    # per-edge probabilities parallel to it, normalises the neighbour
    # vectors, and defaults to no generation time. Defining it suppresses
    # the auto-generated outer constructors, avoiding ambiguity with the
    # scalar and matrix constructors.
    function NetworkProcess(adjacency::AbstractVector{<:AbstractVector{<:Integer}},
            edge_probability::AbstractVector{<:AbstractVector{<:Real}},
            generation_time::G = NoGenerationTime();
            population_size = nothing,
            interventions = AbstractIntervention[],
            attributes::A = NoAttributes(),
            observation::O = NoObservation()) where {G, A, O}
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
        new{G, A, O}(adj, ep, generation_time,
            _intervention_list(interventions), attributes, observation)
    end
end

# Normalise the interventions argument to a vector, on the public surface only
# (rather than reaching into EpiBranch's internal `_intervention_vector`).
_intervention_list(iv::AbstractVector) = convert(Vector{AbstractIntervention}, iv)
_intervention_list(iv) = AbstractIntervention[iv]

# The model carries its interventions, attributes and observation; the
# shared accessors read them from here.
interventions(m::NetworkProcess) = m.interventions
attributes(m::NetworkProcess) = m.attributes
observation(m::NetworkProcess) = m.observation

"""
    NetworkProcess(adjacency, p::Real, gt; population_size = nothing)

Give every edge the same transmission probability `p`.
"""
function NetworkProcess(adjacency::AbstractVector{<:AbstractVector{<:Integer}},
        p::Real, generation_time = NoGenerationTime();
        population_size = nothing,
        interventions = AbstractIntervention[], attributes = NoAttributes(),
        observation::ObservationModel = NoObservation())
    edge_probability = [fill(float(p), length(nbrs)) for nbrs in adjacency]
    NetworkProcess(adjacency, edge_probability, generation_time;
        population_size, interventions, attributes, observation)
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
        population_size = nothing,
        interventions = AbstractIntervention[], attributes = NoAttributes(),
        observation::ObservationModel = NoObservation())
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
    NetworkProcess(adjacency, edge_probability, gt;
        population_size, interventions, attributes, observation)
end

# ── Node bookkeeping ─────────────────────────────────────────────────

# The key is package-prefixed (`:epinetwork_node`) per the downstream-package
# naming convention, so it can't collide with built-in or other-package keys.
network_node(ind::Individual) = get(ind.state, :epinetwork_node, 0)

# ── Edge probability as a competing risk ─────────────────────────────
#
# The per-edge transmission probability belongs on the competing-risks surface,
# not as a filter in `contacts_of`: that way every graph neighbour is produced
# as a contact (visible to interventions for tracing/vaccination) and the
# probability decides infection alongside susceptibility, infectiousness and
# interventions. The model contributes it through `transmission_risks`.

struct EdgeTransmission{M <: NetworkProcess}
    model::M
end

transmission_risks(m::NetworkProcess) = (EdgeTransmission(m),)

# Block transmission across the (parent → contact) edge with probability
# `1 - edge_probability`; `p == 1` adds no risk, `p == 0` blocks entirely.
function competing_risk(e::EdgeTransmission, parent, contact, state)
    p = _edge_probability(e.model, parent.id, contact.id)
    p < 1.0 ? Risk(block_probability = 1.0 - p) : nothing
end

# The probability of the edge from node `from_id` to node `to_id` (node ids are
# the individuals' ids, since the population is pre-instantiated). A missing edge
# yields 0.0 — it never transmits — but `contacts_of` only offers real edges.
function _edge_probability(m::NetworkProcess, from_id::Int, to_id::Int)
    nbrs = m.adjacency[from_id]
    k = findfirst(==(to_id), nbrs)
    k === nothing ? 0.0 : m.edge_probability[from_id][k]
end

# The generation loop for NetworkProcess lives in
# `models/network_simulate.jl` (a dedicated `simulate` method), because
# pre-instantiated nodes need engine helpers defined after this file.
