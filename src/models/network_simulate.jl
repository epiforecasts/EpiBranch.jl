# ── Network model: setup and exposure collection ────────────────────
#
# NetworkProcess runs through the same generic `simulate`/
# `_advance_generation!` engine as BranchingProcess. It supplies just two
# model-specific pieces:
#
#   * `initialise_state` — the population is the graph, so every node is
#     pre-instantiated once with a stable identity and fixed attributes,
#     and `sim_opts.n_initial` seed nodes are infected at random.
#   * `collect_exposures` — each generation, infectious nodes transmit
#     along their edges; exposures are gathered by target node, so a
#     susceptible reached by several infectious neighbours in one
#     generation is resolved once (the loop case the tree never hits).
#
# Everything else — intervention hooks, competing-risks resolution,
# clinical transitions, bookkeeping — is the shared engine.

function initialise_state(model::NetworkProcess, sim_opts::SimOpts,
        interventions, transitions, attributes, rng::AbstractRNG)
    n = length(model.adjacency)
    individuals = Individual[]
    state = SimulationState(
        individuals, Int[], 0, rng, 0, false,
        population_size(model), 0.0, attributes,
        convert(Vector{AbstractClinicalTransition}, transitions))

    # Pre-instantiate every node with fixed attributes and intervention
    # state. (The tree case mints contacts lazily; the network's nodes
    # exist from the start.)
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

    state.cumulative_cases = n_seed
    state.active_ids = seed_ids
    state.extinct = n_seed == 0
    return state
end

# A network's contacts can be shared across parents in a generation
# (the loop case), so it gathers exposures by target rather than using the
# tree-default `collect_exposures`.
function collect_exposures(model::NetworkProcess, state::SimulationState)
    gather_by_target(model, state)
end

"""
    contacts_of(model::NetworkProcess, parent, state)

Transmission over a fixed graph: `parent`'s contacts this generation are
its existing graph neighbours. Each edge fires with its per-edge
probability; already-infected neighbours are skipped, and surviving edges
are returned as `(node, infection_time)` pairs at a generation-time-shifted
time. No contacts are created — the nodes already exist — so the engine
recognises them as pre-existing by id. A susceptible reached by several
infectious neighbours in one generation is deduplicated downstream by
[`gather_by_target`](@ref).
"""
function contacts_of(model::NetworkProcess, parent::Individual,
        state::SimulationState)
    rng = state.rng
    idx = parent.id
    gt_dist = get_generation_time(model.generation_time, parent)
    nbrs = model.adjacency[idx]
    probs = model.edge_probability[idx]
    result = Tuple{Individual, Float64}[]
    for k in eachindex(nbrs)
        target = state.individuals[nbrs[k]]
        is_infected(target) && continue
        p = probs[k]
        (p >= 1.0 || rand(rng) < p) || continue
        push!(result, (target, _infection_time(gt_dist, parent, state)))
    end
    return result
end
