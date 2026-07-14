# ── Continuous-time (Sellke) simulation ──────────────────────────────
#
# A household outbreak is a finite, depleting clique, so it is simulated by the
# Sellke/Dijkstra construction in continuous time — the exact generative model
# of the pairwise likelihood — rather than the generation-based engine. Members
# are processed in increasing infection-time order; popping the earliest pending
# infection makes it final, because any not-yet-processed infector has a later
# infection time, becomes infectious no earlier, and so reaches it later.
#
# Every other concern is the shared engine: the population is built with the
# public `new_state`/`add_individuals!`, each infected member's natural history
# (latent period, onset, recovery, …) is stamped through `resolve_transitions!`,
# and the result is an EpiBranch `SimulationState` that `linelist` renders.

"""
    _simulate(model::HouseholdProcess, sim_opts; interventions, attributes,
              progression, observation, rng, condition, max_attempts)

Simulate `model` by the Sellke construction in continuous time — the exact
generative model of the pairwise likelihood — with the modelling layers supplied
by the caller (a bare process, or a `ModelSpec`). The infectious window's `from`
state is derived from the composed `progression`. Returns an EpiBranch
`SimulationState`; `linelist(state)` renders the one-row-per-case DataFrame.

With no external hazard each household is seeded with one index at time 0 and
spreads only within the household. With an `external_hazard` — a positive scalar
or a calendar-time distribution — community introductions emerge over
`[0, model.obs_end]`; give the process a finite `obs_end` in that case.
"""
function _simulate(model::HouseholdProcess, sim_opts::SimOpts;
        interventions, attributes, progression, observation, rng, condition,
        max_attempts)
    if condition !== nothing
        for _ in 1:max_attempts
            state = _simulate(model, sim_opts; interventions, attributes, progression,
                observation, rng, condition = nothing, max_attempts)
            state.cumulative_cases in condition && return state
        end
        throw(ErrorException(
            "No simulation produced an outbreak of size $condition within $max_attempts attempts"))
    end

    from = _resolve_infectious_from(model.from, progression)
    Tobs = model.obs_end

    state = new_state(model, progression, attributes, rng)
    add_individuals!(state, length(model.household_of), interventions;
        setup = (ind, i) -> (ind.state[:household] = model.household_of[i]))

    for mem in model.members
        EpiBranch._sellke_race!(state, mem, rng;
            from = from, until = model.until,
            seed! = (best, members, r) -> _seed_clique!(
                best, members, model.external_hazard, Tobs, r),
            targets = (inf, st) -> ((oid, _pairkernel(model.kernel, inf, oid))
            for oid in mem if oid != inf))
    end

    # The clique loop writes per-individual state directly, so reconcile the
    # aggregate bookkeeping the engine would otherwise maintain, keeping the
    # returned state consistent with core `simulate` for downstream consumers.
    state.cumulative_cases = count(ind -> get(ind.state, :infected, false), state.individuals)
    state.max_infection_time = maximum(
        (ind.infection_time
        for ind in state.individuals if get(ind.state, :infected, false));
        init = 0.0)

    # Apply the observation model (under-reporting, report delays), as core
    # `simulate` does. A no-op for the default `NoObservation`.
    apply_observation!(observation, state, rng)
    return state
end

# Seed one household's candidate table: community introductions under the
# external hazard (each member drawn, kept if it lands within `[0, Tobs]`), or a
# single seeded index at time 0 when there is no external source.
function _seed_clique!(best, members, extsrc, Tobs, rng)
    m = length(members)
    if _ext_active(extsrc)
        for k in 1:m
            t = _ext_draw(rng, extsrc)
            t <= Tobs && (best[k] = t)
        end
    else
        best[rand(rng, 1:m)] = 0.0
    end
    return nothing
end

# Resolve the contact-interval distribution for an ordered (infector,
# susceptible) pair: a shared distribution, or a callable for covariate models.
_pairkernel(k::ContinuousUnivariateDistribution, i, j) = k
_pairkernel(k, i, j) = k(i, j)

# A community introduction time under the external hazard: the constant case is
# its Exponential survival time, a distribution is sampled directly.
_ext_draw(rng, α::Real) = rand(rng, Exponential(1 / α))
_ext_draw(rng, d::ContinuousUnivariateDistribution) = rand(rng, d)
