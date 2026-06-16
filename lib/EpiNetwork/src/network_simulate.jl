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
    state = new_state(model, transitions, attributes, rng)

    # The population is the graph: pre-instantiate every node with a stable
    # identity, fixed attributes and intervention state. (The tree case
    # mints contacts lazily; the network's nodes exist from the start.)
    add_individuals!(state, n, interventions;
        setup = (ind, i) -> (ind.state[:epinetwork_node] = i))

    # Seed infections on random nodes.
    n_seed = min(sim_opts.n_initial, n)
    seed!(state, randperm(rng, n)[1:n_seed], interventions, transitions)
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

Transmission over a fixed graph: `parent`'s contacts this generation are its
still-susceptible graph neighbours, each returned as a `(node, infection_time)`
candidate at a generation-time-shifted time. Every neighbour is produced (so
`apply_post_transmission!` — contact tracing, ring vaccination — sees the whole
graph contact, not only the ones that transmit); the per-edge probability is a
competing risk via [`transmission_risks`](@ref EpiBranch.transmission_risks), not
a filter here. No contacts are created — the nodes already exist — so the engine
recognises them by id, and a susceptible reached by several infectious
neighbours in one generation is deduplicated downstream by
[`gather_by_target`](@ref).
"""
function contacts_of(model::NetworkProcess, parent::Individual,
        state::SimulationState)
    gt_dist = get_generation_time(model.generation_time, parent)
    result = Tuple{Individual, Float64}[]
    for nb in model.adjacency[parent.id]
        target = state.individuals[nb]
        is_infected(target) && continue
        push!(result, (target, transmission_time(gt_dist, parent, state)))
    end
    return result
end
