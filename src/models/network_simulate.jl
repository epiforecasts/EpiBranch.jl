# ── Network simulation ──────────────────────────────────────────────
#
# A fixed set of nodes with a given adjacency. Infectious nodes transmit
# along their edges to susceptible neighbours; a node reached by several
# neighbours is infected if any incoming edge transmits. This is a
# structure-driven model: it plugs into the shared `simulate` loop by
# defining `initialise_state` (pre-instantiate the pool) and
# `_advance_generation!` (build the per-generation exposures), then reuses
# the engine's competition machinery (`_set_provisional_sources!`,
# `_resolve_exposures!`) to resolve who is infected.

function initialise_state(model::NetworkProcess, sim_opts::SimOpts,
        interventions, transitions, attributes, rng::AbstractRNG)
    n = length(model.adjacency)
    individuals = Individual[]
    state = SimulationState(
        individuals, Int[], 0, rng, 0, false,
        population_size(model), 0.0, attributes,
        convert(Vector{AbstractClinicalTransition}, transitions))

    # Pre-instantiate every node with fixed attributes.
    for i in 1:n
        ind = _create_individual(state, 0, i, i, 0.0)
        ind.state[:network_node] = i
        ind.state[:infected] = false
        for intervention in interventions
            initialise_individual!(intervention, ind, state)
        end
        push!(individuals, ind)
    end

    # Seed infections on random nodes.
    n_seed = min(sim_opts.n_initial, n)
    seed_ids = randperm(rng, n)[1:n_seed]
    for id in seed_ids
        ind = individuals[id]
        ind.state[:infected] = true
        _set_onset_from_incubation!(ind)
    end
    if n_seed >= 1
        _validate_required_fields(individuals[seed_ids[1]], interventions)
        _validate_required_fields(individuals[seed_ids[1]], transitions)
    end
    for id in seed_ids
        _resolve_transitions!(state, individuals[id])
    end

    state.cumulative_cases = n_seed
    state.active_ids = seed_ids
    state.extinct = n_seed == 0
    return state
end

"""
    _advance_generation!(model::NetworkProcess, state, interventions)

One generation over the fixed contact network. Infectious nodes expose
their susceptible neighbours along edges that pass the per-edge
transmission draw; the shared competition machinery then resolves which
exposed nodes are infected and fixes each one's source and timing.
"""
function _advance_generation!(model::NetworkProcess, state::SimulationState, interventions)
    rng = state.rng

    # Resolve intervention state on the currently-infectious nodes.
    for idx in state.active_ids
        for intervention in interventions
            resolve_individual!(intervention, state.individuals[idx], state)
        end
    end

    # Gather exposures: each susceptible neighbour collects the
    # (infectious source, infection time) of every edge that passes the
    # per-edge transmission-probability draw.
    exposures = Exposures()
    for idx in state.active_ids
        parent = state.individuals[idx]
        gt_dist = get_generation_time(model.generation_time, parent)
        nbrs = model.adjacency[idx]
        probs = model.edge_probability[idx]
        for k in eachindex(nbrs)
            nb = nbrs[k]
            target = state.individuals[nb]
            is_infected(target) && continue
            p = probs[k]
            (p >= 1.0 || rand(rng) < p) || continue
            inf_time = _infection_time(gt_dist, parent, state)
            push!(get!(exposures, nb, Tuple{Int, Float64}[]), (idx, inf_time))
        end
    end

    if isempty(exposures)
        state.extinct = true
        state.active_ids = Int[]
        state.current_generation += 1
        return nothing
    end

    # Earliest exposing edge becomes each node's provisional source, so
    # contact-level interventions can act on the exposed susceptibles.
    exposed = _set_provisional_sources!(state, exposures)
    for intervention in interventions
        apply_post_transmission!(intervention, state, exposed)
    end

    # Resolve infection per node and record the new cases.
    newly_infected = _resolve_exposures!(state, exposures, interventions)

    state.cumulative_cases += length(newly_infected)
    state.current_generation += 1
    state.active_ids = newly_infected
    isempty(newly_infected) && (state.extinct = true)
    for nb in newly_infected
        _resolve_transitions!(state, state.individuals[nb])
    end
    return nothing
end
