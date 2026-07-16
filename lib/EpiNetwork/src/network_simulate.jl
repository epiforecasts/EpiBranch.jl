# ── Continuous-time (Sellke) simulation over the graph ───────────────
#
# A rate-based network outbreak is simulated by the same Sellke/Dijkstra
# race as HouseholdProcess, but run once over *all* nodes with each node's
# graph neighbours as its contacts (rather than once per household clique).
# Nodes are processed in increasing infection-time order; popping the
# earliest pending infection makes it final. Every other concern is the
# shared engine: the population is built with `new_state`/`add_individuals!`,
# each infected node's natural history is stamped through the primitive's
# `resolve_transitions!`, and the result is an EpiBranch `SimulationState`
# that `linelist` renders.

"""
    _simulate(model::NetworkProcess, sim_opts; interventions, attributes,
              progression, observation, rng, condition, max_attempts)

Simulate `model` by the Sellke construction in continuous time, with the
modelling layers supplied by the caller (a bare process, or a `ModelSpec`). The
infectious window's `from` state is derived from the composed `progression`.
Returns an EpiBranch `SimulationState`; `linelist(state)` renders it.

With no external hazard, `sim_opts.n_initial` distinct nodes are seeded at time
0 and the outbreak spreads along the edges. With an `external_hazard` — a
positive scalar or a calendar-time distribution — community introductions emerge
over `[0, model.obs_end]`, so a finite `obs_end` is required (an unbounded window
would seed every node).
"""
function _simulate(model::NetworkProcess, sim_opts::SimOpts;
        interventions, attributes, progression, observation, rng, condition,
        max_attempts)
    condition !== nothing && return _retry_for_condition(
        () -> _simulate(model, sim_opts; interventions, attributes, progression,
            observation, rng, condition = nothing, max_attempts),
        condition, max_attempts)

    from = _resolve_infectious_from(model.from, progression)
    Tobs = model.obs_end
    n_initial = sim_opts.n_initial

    state = new_state(model, progression, attributes, rng)
    n = length(model.adjacency)
    add_individuals!(state, n, interventions; setup = (ind, i) -> nothing)

    # A community hazard over an unbounded window would introduce every node,
    # swamping the graph. Require a finite observation window instead.
    _ext_active(model.external_hazard) && !isfinite(Tobs) &&
        throw(ArgumentError(
            "an external hazard needs a finite `obs_end` (an unbounded window seeds " *
            "the whole network); build the process with e.g. `obs_end = 30.0`"))
    EpiBranch._sellke_race!(state, collect(1:n), rng;
        from = from, until = model.until, interventions = interventions,
        seed! = (best, members, r) -> _seed_network!(
            best, members, model.external_hazard, n_initial, Tobs, r),
        targets = (inf, st) -> ((nb, _edge_kernel(model, inf, k))
        for (k, nb) in enumerate(model.adjacency[inf])
        if !is_infected(st.individuals[nb])))

    _reconcile_sellke_bookkeeping!(state)
    # Apply the observation model (under-reporting, report delays), as core
    # `simulate` does. A no-op for the default `NoObservation`.
    apply_observation!(observation, state, rng)
    return state
end

# Seed the candidate table over all nodes: community introductions under the
# external hazard (each node drawn, kept if it lands within `[0, Tobs]`), or
# `n_initial` distinct random nodes at time 0 when there is no external source.
function _seed_network!(best, members, extsrc, n_initial, Tobs, rng)
    m = length(members)
    if _ext_active(extsrc)
        for k in 1:m
            t = _ext_draw(rng, extsrc)
            t <= Tobs && (best[k] = t)
        end
    else
        for id in randperm(rng, m)[1:min(n_initial, m)]
            best[id] = 0.0
        end
    end
    return nothing
end
