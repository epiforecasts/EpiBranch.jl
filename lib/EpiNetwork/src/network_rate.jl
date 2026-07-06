# ── NetworkRateProcess ───────────────────────────────────────────────
#
# A network whose transmission is driven by contact *rates* rather than a
# coin flip per edge. Each graph edge carries a contact-interval kernel
# (Kenah 2011): the time from an infector becoming infectious to its
# infectious contact with a graph neighbour is drawn from that kernel, and
# transmission happens only while the infector is inside its infectious
# window — the window `progression` opens at `from` and closes at the
# earliest of the `until` states. This is `HouseholdProcess` with the graph
# neighbours playing the role of the household clique: the same
# continuous-time (Sellke) race, run once over every node with each node's
# graph neighbours as its contacts.
#
# Because transmission is a hazard racing removal, closing a node's
# infectious window early (recovery, isolation) genuinely curtails onward
# transmission — the thing the coin-flip `NetworkProcess` cannot express.

"""
    NetworkRateProcess(adjacency, kernel; infectious_period = nothing,
                       latent_period = nothing, progression = [], from = nothing,
                       until = (:recovered, :died, :isolated), external_hazard = 0.0,
                       interventions = [], attributes = NoAttributes(),
                       observation = NoObservation())

Rate-based transmission over a fixed contact network. `adjacency[i]` lists
the graph neighbours of node `i` (1-based); the graph is the population.
`kernel` is the **contact interval** — the one required input — a continuous
`Distributions.jl` distribution shared by every edge, a callable
`(infector, susceptible) -> Distribution` for covariate models, or a
per-edge vector of distributions parallel to `adjacency`
(`kernel[i][k]` for node `i`'s `k`-th listed neighbour). The kernel times
each infectious contact from the infector's `from` state.

The infectious timeline is a flexible `progression` of EpiBranch
`Transition`s, exactly as for `HouseholdProcess`: a latent period is
`Transition(:infectious, from = :infection, delay = …)`, an infectious
period a removal transition, and onset, testing and the rest are further
transitions the line list reads. `from` is the state the kernel times from
(`:infectious` when the timeline has a latent period, `:infection`
otherwise); `until` names the removal states that close the infectious
window. Adding a `Transition(:isolated, …)` (or an intervention that writes
`:isolated_time`) closes the window early and so cuts onward transmission.

`infectious_period` and `latent_period` are sugar for the two common
transitions, each a scalar (constant) or a `Distribution` (per-case).
`external_hazard` is the community force of infection — a scalar for a
constant hazard or a calendar-time distribution for a time-varying one.

# Example

```julia
using EpiNetwork, Distributions
adjacency = [[2, 5], [1, 3], [2, 4], [3, 5], [4, 1]]   # ring of 5
model = NetworkRateProcess(adjacency, Exponential(2.0); infectious_period = 6.0)
```
"""
struct NetworkRateProcess{K, E, A, O <: ObservationModel} <: TransmissionModel
    adjacency::Vector{Vector{Int}}   # adjacency[i] = graph neighbours of node i
    edge_kernel::K                   # contact interval: shared, callable, or per-edge
    from::Symbol                     # state the kernel times contacts from
    until::Tuple                     # removal states that close the infectious window
    progression::Vector{AbstractClinicalTransition}  # natural-history timeline
    external_hazard::E               # community force of infection (0 = none)
    interventions::Vector{AbstractIntervention}
    attributes::A
    observation::O
end

function NetworkRateProcess(adjacency::AbstractVector{<:AbstractVector{<:Integer}},
        kernel;
        infectious_period = nothing,
        latent_period = nothing,
        progression = AbstractClinicalTransition[],
        from = nothing,
        until = (:recovered, :died, :isolated),
        external_hazard = 0.0,
        interventions = AbstractIntervention[],
        attributes = NoAttributes(),
        observation::ObservationModel = NoObservation())
    adj = Vector{Int}[Int.(nbrs) for nbrs in adjacency]
    edge_kernel = _validate_kernel(kernel, adj)
    _valid_external(external_hazard) ||
        throw(ArgumentError("external_hazard must be a non-negative number or a continuous distribution"))

    prog = convert(Vector{AbstractClinicalTransition}, progression)
    # Desugar the common timeline shorthands when no explicit progression given:
    # a latent period to :infectious, an infectious period to a recovery removal.
    if isempty(prog)
        latent_period === nothing ||
            push!(prog, Transition(:infectious; from = :infection, delay = _as_delay(latent_period)))
        anchor = latent_period === nothing ? :infection : :infectious
        infectious_period === nothing ||
            push!(prog,
                Transition(:recovered; from = anchor, delay = _as_delay(infectious_period),
                    terminal = true))
    end

    # The kernel times contacts from `from`: default to :infectious when the
    # timeline produces it (a latent period), otherwise from infection.
    kfrom = from === nothing ?
            (any(t -> _writes_state(t, :infectious), prog) ? :infectious : :infection) :
            from

    return NetworkRateProcess(adj, edge_kernel, kfrom, Tuple(until),
        prog, _normalise_external(external_hazard), _intervention_list(interventions),
        attributes, observation)
end

"""
    NetworkRateProcess(A::AbstractMatrix, kernel; kwargs...)

Build a `NetworkRateProcess` from an adjacency matrix: a nonzero `A[i, j]`
means an (undirected) edge between `i` and `j`. Every edge shares `kernel`
(the matrix marks graph structure only, not per-edge rates).
"""
function NetworkRateProcess(A::AbstractMatrix, kernel; kwargs...)
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
    return NetworkRateProcess(adjacency, kernel; kwargs...)
end

# The interventions/attributes/observation the model carries.
interventions(m::NetworkRateProcess) = m.interventions
attributes(m::NetworkRateProcess) = m.attributes
observation(m::NetworkRateProcess) = m.observation

# The population is the graph; there is no separate finite susceptible pool.
population_size(::NetworkRateProcess) = NoPopulation()

function Base.show(io::IO, m::NetworkRateProcess)
    n = length(m.adjacency)
    n_edges = sum(length, m.adjacency; init = 0) ÷ 2
    kstr = m.edge_kernel isa Distribution ? nameof(typeof(m.edge_kernel)) :
           m.edge_kernel isa AbstractVector ? "per-edge" : "Function"
    print(io, "NetworkRateProcess(nodes=$n, edges=$n_edges, kernel=$kstr, ",
        "from=:$(m.from), $(length(m.progression)) transitions",
        _ext_active(m.external_hazard) ? ", external_hazard=$(m.external_hazard))" : ")")
end

# ── Kernel handling ──────────────────────────────────────────────────
#
# The contact-interval kernel is stored in whatever form the constructor was
# given — a shared distribution, a callable `(infector, susceptible) ->
# Distribution`, or a per-edge vector parallel to the adjacency list — and
# resolved per contact by `_edge_kernel(model, infector, position)`, where
# `position` is the index of the neighbour within `adjacency[infector]`.

# A shared distribution is used as-is; a per-edge vector is validated to line
# up with the adjacency list; anything else is taken to be a callable.
_validate_kernel(k::ContinuousUnivariateDistribution, adj) = k
function _validate_kernel(k::AbstractVector{<:AbstractVector}, adj)
    length(k) == length(adj) || throw(ArgumentError(
        "per-edge kernel and adjacency must have the same number of nodes"))
    for i in eachindex(adj)
        length(k[i]) == length(adj[i]) || throw(ArgumentError(
            "node $i: per-edge kernel and adjacency have different lengths"))
    end
    return [collect(row) for row in k]
end
_validate_kernel(k, adj) = k   # callable (infector, susceptible) -> Distribution

# The contact-interval distribution for the `pos`-th neighbour of node `i`.
function _edge_kernel(m::NetworkRateProcess, i::Int, pos::Int)
    _resolve_kernel(m.edge_kernel, m, i, pos)
end
_resolve_kernel(k::ContinuousUnivariateDistribution, m, i, pos) = k
_resolve_kernel(k::AbstractVector, m, i, pos) = k[i][pos]
_resolve_kernel(k, m, i, pos) = k(i, m.adjacency[i][pos])

# ── Timeline / external-hazard helpers ───────────────────────────────
#
# Mirror the HouseholdProcess helpers: this model shares the same
# structure-driven timeline sugar and community-introduction machinery.

# A `Transition` delay is a `Distribution` or `(rng, ind) -> Real`; a scalar
# period becomes a constant-delay closure so the sugar accepts both.
_as_delay(d::Distribution) = d
_as_delay(x::Real) = (rng, ind) -> float(x)

# Does a clinical transition write the named state?
_writes_state(t, s::Symbol) = hasproperty(t, :state) && getfield(t, :state) === s

# The external community source: a non-negative scalar (constant hazard) or any
# continuous distribution on the non-negative reals (a calendar-time hazard).
_valid_external(α::Real) = α >= 0
_valid_external(d::ContinuousUnivariateDistribution) = minimum(d) >= 0
_valid_external(_) = false
_normalise_external(α::Real) = Float64(α)
_normalise_external(d::ContinuousUnivariateDistribution) = d

_ext_active(α::Real) = α > 0
_ext_active(::ContinuousUnivariateDistribution) = true

# A community introduction time under the external hazard: the constant case is
# its Exponential survival time, a distribution is sampled directly.
_ext_draw(rng, α::Real) = rand(rng, Exponential(1 / α))
_ext_draw(rng, d::ContinuousUnivariateDistribution) = rand(rng, d)
