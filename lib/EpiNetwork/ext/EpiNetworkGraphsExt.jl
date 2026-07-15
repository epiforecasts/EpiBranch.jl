module EpiNetworkGraphsExt

# Ingest a Graphs.jl graph as the contact network for a `NetworkProcess`. The
# core model already takes an adjacency list, so this reads each vertex's
# neighbours into that form and delegates to the existing constructor — the
# graph is the population, one node per vertex. For a directed graph the
# out-neighbours are the nodes a case can infect, the transmission-relevant
# direction; an undirected graph gives every neighbour. Loading Graphs.jl is
# what turns this on, so the dependency stays optional for adjacency-list users.

using EpiNetwork
using Graphs: AbstractGraph, vertices, neighbors

"""
    NetworkProcess(g::Graphs.AbstractGraph, kernel; kwargs...)

Build a `NetworkProcess` over a Graphs.jl graph. Each vertex becomes a node and
each vertex's `neighbors` become its contacts; every edge shares `kernel` (or a
callable/per-edge kernel, as in the adjacency-list constructor). Any generator
that returns a graph works, e.g. `watts_strogatz`, `barabasi_albert` or
`stochastic_block_model`. (`euclidean_graph` returns a `(graph, distances)`
tuple, so pass its first element.) Available once Graphs.jl is loaded.
"""
function EpiNetwork.NetworkProcess(g::AbstractGraph, kernel; kwargs...)
    adjacency = [collect(neighbors(g, v)) for v in vertices(g)]
    return NetworkProcess(adjacency, kernel; kwargs...)
end

end # module
