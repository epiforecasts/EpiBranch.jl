# ── Network simulation loop ─────────────────────────────────────────
#
# Instantiates the full set of network nodes once, each with a stable
# identity and attributes drawn at setup. Every generation, infectious
# nodes transmit along their edges to susceptible neighbours; a node
# reached by several neighbours is infected if any incoming edge
# transmits, and stays a single node.
#
# Infection resolution reuses the engine's competing-risks resolver
# (`_decide_infected`), the intervention hooks, and the clinical
# transitions; transmission references the existing nodes.

function _init_network_state(model::NetworkProcess, sim_opts::SimOpts,
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
    simulate(model::NetworkProcess; kwargs...)

Run a single outbreak over a fixed contact network. Each node carries a
stable identity; transmission flows along edges and each node is
infected at most once. Interventions, attributes, and clinical
transitions apply as for [`BranchingProcess`](@ref).

Keyword arguments match the [`BranchingProcess`](@ref) method
(`interventions`, `transitions`, `attributes`, `sim_opts`, `rng`).
`sim_opts.n_initial` seed nodes are chosen at random.
"""
function simulate(model::NetworkProcess;
        interventions::Vector{<:AbstractIntervention} = AbstractIntervention[],
        transitions::Vector{<:AbstractClinicalTransition} = AbstractClinicalTransition[],
        attributes::Union{Function, NoAttributes} = NoAttributes(),
        sim_opts::SimOpts = SimOpts(),
        rng::AbstractRNG = Random.default_rng())
    state = _init_network_state(
        model, sim_opts, interventions, transitions, attributes, rng)

    while !should_terminate(state, sim_opts)
        # 1. Resolve intervention state on the currently-infectious nodes.
        for idx in state.active_ids
            for intervention in interventions
                resolve_individual!(intervention, state.individuals[idx], state)
            end
        end

        # 2. Gather exposures: each susceptible neighbour collects the
        #    (infectious parent, infection time) of every edge that
        #    passes the per-edge transmission-probability draw.
        exposures = Dict{Int, Vector{Tuple{Int, Float64}}}()
        for idx in state.active_ids
            parent = state.individuals[idx]
            gt_dist = get_generation_time(model.generation_time, parent)
            for nb in model.adjacency[idx]
                target = state.individuals[nb]
                is_infected(target) && continue
                p = _edge_probability(model.transmission_probability, rng, parent, nb)
                (p >= 1.0 || rand(rng) < p) || continue
                inf_time = _infection_time(gt_dist, parent, state)
                push!(get!(exposures, nb, Tuple{Int, Float64}[]), (idx, inf_time))
            end
        end

        if isempty(exposures)
            state.extinct = true
            state.active_ids = Int[]
            state.current_generation += 1
            continue
        end

        # 3. Earliest exposing edge becomes each node's provisional
        #    parent so contact-level interventions (tracing, ring
        #    vaccination) can act on the exposed susceptibles.
        exposed = Individual[]
        for (nb, edges) in exposures
            sort!(edges, by = last)
            target = state.individuals[nb]
            target.parent_id = edges[1][1]
            target.infection_time = edges[1][2]
            push!(exposed, target)
        end
        for intervention in interventions
            apply_post_transmission!(intervention, state, exposed)
        end

        # 4. Resolve infection per node: infected if ANY incoming edge
        #    transmits under competing risks. Edges are tried earliest
        #    first, so the first success gives the infection time.
        newly_infected = Int[]
        for (nb, edges) in exposures
            target = state.individuals[nb]
            infected = false
            for (pid, t) in edges
                target.parent_id = pid
                target.infection_time = t
                if _decide_infected(state, target, interventions, 0)
                    infected = true
                    break
                end
            end
            if infected
                parent = state.individuals[target.parent_id]
                target.state[:infected] = true
                target.generation = parent.generation + 1
                target.chain_id = parent.chain_id
                _set_onset_from_incubation!(target)
                push!(parent.secondary_case_ids, nb)
                push!(newly_infected, nb)
                target.infection_time > state.max_infection_time &&
                    (state.max_infection_time = target.infection_time)
            else
                target.parent_id = 0
                target.infection_time = 0.0
            end
        end

        # 5. Bookkeeping and clinical transitions on the new cases.
        state.cumulative_cases += length(newly_infected)
        state.current_generation += 1
        state.active_ids = newly_infected
        isempty(newly_infected) && (state.extinct = true)
        for nb in newly_infected
            _resolve_transitions!(state, state.individuals[nb])
        end
    end

    return state
end
