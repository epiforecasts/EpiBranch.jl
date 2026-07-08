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
    simulate(model::NetworkProcess; rng = default_rng(), n_initial = 1,
             obs_end = Inf) -> SimulationState

Simulate `model` by the Sellke construction in continuous time. Returns an
EpiBranch `SimulationState`; `linelist(state)` turns it into the
one-row-per-case DataFrame, carrying the infection, infectiousness, onset
and removal times the model's `progression` stamps on each case.

With no external hazard, `n_initial` distinct nodes are seeded at time 0 and
the outbreak spreads along the edges. With an `external_hazard` — a positive
scalar or a calendar-time distribution — community introductions emerge over
`[0, obs_end]`, so a finite `obs_end` is required (an unbounded window would
seed every node).
"""
function simulate(model::NetworkProcess; rng::AbstractRNG = default_rng(),
        n_initial::Integer = 1, obs_end::Real = Inf)
    state = new_state(model, model.progression, attributes(model), rng)
    n = length(model.adjacency)
    add_individuals!(state, n, interventions(model); setup = (ind, i) -> nothing)

    Tobs = Float64(obs_end)
    # A community hazard over an unbounded window would introduce every node,
    # swamping the graph. Require a finite observation window instead.
    _ext_active(model.external_hazard) && !isfinite(Tobs) &&
        throw(ArgumentError(
            "an external hazard needs a finite `obs_end` (an unbounded window seeds " *
            "the whole network); pass e.g. `obs_end = 30.0`"))
    EpiBranch._sellke_race!(state, collect(1:n), rng;
        from = model.from, until = model.until,
        seed! = (best, members, r) -> _seed_network!(
            best, members, model.external_hazard, n_initial, Tobs, r),
        targets = (inf, st) -> ((nb, _edge_kernel(model, inf, k))
        for (k, nb) in enumerate(model.adjacency[inf])
        if !is_infected(st.individuals[nb])))

    # The race writes per-individual state directly, so reconcile the aggregate
    # bookkeeping the engine would otherwise maintain, keeping the returned
    # state consistent with core `simulate` for downstream consumers.
    state.cumulative_cases = count(ind -> get(ind.state, :infected, false), state.individuals)
    state.max_infection_time = maximum(
        (ind.infection_time
        for ind in state.individuals if get(ind.state, :infected, false));
        init = 0.0)

    # Apply the model's observation model (under-reporting, report delays), as
    # core `simulate` does. A no-op for the default `NoObservation`.
    apply_observation!(observation(model), state, rng)
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
