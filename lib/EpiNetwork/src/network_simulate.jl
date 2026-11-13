# ── Continuous-time (Sellke) simulation over the graph ───────────────
#
# A rate-based network outbreak is simulated by the same Sellke/Dijkstra
# race as HouseholdProcess, but run once over *all* nodes with each node's
# graph neighbours as its contacts (rather than once per household clique).
# Nodes are processed in increasing infection-time order; popping the
# earliest pending infection makes it final. Every other concern is the
# shared engine: the population is built with `new_state` and, by default,
# `lazily_populate!`, each infected node's natural history is stamped through
# the primitive's `resolve_transitions!`, and the result is an EpiBranch
# `SimulationState` that `linelist` renders.

"""
    simulate_once(model::NetworkProcess, sim_opts; interventions, attributes,
                  progression, observation, recorder, rng)

Simulate a network outbreak exactly in continuous time, with the natural
history and interventions supplied by the caller (a bare process, or a
`ModelSpec`). The start of the infectious period (`from`) is taken from the
`progression`. Returns an EpiBranch `SimulationState`; `linelist(state)` turns
it into a table.

With no external hazard, `sim_opts.n_initial` different people are infected at
day 0 and the outbreak spreads along the network. With an `external_hazard` (a
positive rate or a distribution of the time of infection from outside),
community introductions happen over the first `model.obs_end` days, which must
be finite (otherwise everyone would eventually be infected from outside).
Chosen `initial_cases` are infected at day 0 regardless, and any external
hazard still acts on everyone else from day 0.
"""
function simulate_once(
        model::NetworkProcess, sim_opts::SimOpts;
        interventions, attributes, progression, observation, recorder, rng
    )
    from = something(model.from, infectious_from(progression))
    Tobs = model.obs_end
    n_initial = sim_opts.n_initial

    state = new_state(model, progression, attributes, rng)
    n = length(model.adjacency)
    EpiBranch._populate!(state, n, interventions, model.eager)

    EpiBranch._check_finite_obs_end(model.external_hazard, Tobs, "the whole network")
    EpiBranch.sellke_race!(
        state, collect(1:n), rng, sim_opts;
        from = from, until = model.until, interventions = interventions,
        risks = EpiBranch.transmission_risks(model),
        watches = (EpiBranch.watched_records(model.edge_kernel),),
        projections = (EpiBranch.kernel_projection(model.edge_kernel),), recorder = recorder,
        seed! = (best, members, r) -> _seed_network!(
            best, members, state, model.external_hazard, n_initial, Tobs, r;
            initial_cases = sim_opts.initial_cases
        ),
        introduction = _ext_active(model.external_hazard) ?
            (EpiBranch._ext_prepare(model.external_hazard), Tobs) : nothing,
        targets = (inf, st) -> (
            (nb, _edge_kernel(model, inf, k, st, from))
                for (k, nb) in enumerate(model.adjacency[inf])
                if !is_infected(st.individuals[nb])
        ),
        # Tracing reaches every neighbour, infected or not, so it reads the
        # adjacency directly rather than the susceptible-only `targets`.
        contacts = (inf, st) -> model.adjacency[inf]
    )

    # Apply the observation model (under-reporting, report delays), as core
    # `simulate` does. A no-op for the default `NoObservation`.
    apply_observation!(observation, state, rng)
    return state
end

# Seed the candidate table over all nodes: `n_initial` distinct random nodes at
# time 0 unless chosen `initial_cases` or an external hazard seed it.
function _seed_network!(
        best, members, state, extsrc, n_initial, Tobs, rng;
        initial_cases = nothing
    )
    EpiBranch._seed_candidates!(
        best, members, state, extsrc, Tobs, rng, initial_cases
    ) do b, m, r
        for id in randperm(r, length(m))[1:min(n_initial, length(m))]
            b[id] = 0.0
        end
    end
    return nothing
end

function EpiBranch._validate_initial_cases(model::NetworkProcess, opts::SimOpts)
    return EpiBranch._validate_initial_case_ids(opts, length(model.adjacency))
end
