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
    simulate(model::HouseholdProcess; rng = default_rng(), obs_end = Inf) -> SimulationState

Simulate `model` by the Sellke construction in continuous time — the exact
generative model of the pairwise likelihood. Returns an EpiBranch
`SimulationState`; `linelist(state)` turns it into the one-row-per-case
DataFrame, carrying the infection, infectiousness, onset and recovery times the
model's `progression` stamps on each case.

With no external hazard each household is seeded with one index at time 0 and
spreads only within the household. With an `external_hazard` — a positive scalar
or a calendar-time distribution — community introductions emerge over
`[0, obs_end]`; pass a finite `obs_end` in that case.
"""
function simulate(model::HouseholdProcess; rng::AbstractRNG = default_rng(),
        obs_end::Real = Inf)
    state = new_state(model, model.progression, attributes(model), rng)
    add_individuals!(state, length(model.household_of), interventions(model);
        setup = (ind, i) -> (ind.state[:household] = model.household_of[i]))

    Tobs = Float64(obs_end)
    for mem in model.members
        _simulate_clique!(state, model, mem, Tobs)
    end

    # The clique loop writes per-individual state directly, so reconcile the
    # aggregate bookkeeping the engine would otherwise maintain, keeping the
    # returned state consistent with core `simulate` for downstream consumers.
    state.cumulative_cases = count(ind -> get(ind.state, :infected, false), state.individuals)
    state.max_infection_time = maximum(
        (ind.infection_time
        for ind in state.individuals if get(ind.state, :infected, false));
        init = 0.0)
    return state
end

# The Sellke construction on one household. `best[k]` is member `k`'s earliest
# candidate infection time (local index), `src[k]` the global id of the infector
# that offered it (`0` for an exogenous introduction).
function _simulate_clique!(state, model::HouseholdProcess, mem, Tobs::Float64)
    rng = state.rng
    m = length(mem)
    extsrc = model.external_hazard
    best = fill(Inf, m)
    src = zeros(Int, m)
    processed = falses(m)

    # Exogenous candidates: community introductions under the external hazard,
    # or a single seeded index per household when there is no external source.
    if _ext_active(extsrc)
        for k in 1:m
            t = _ext_draw(rng, extsrc)
            t <= Tobs && (best[k] = t)
        end
    else
        best[rand(rng, 1:m)] = 0.0
    end

    while true
        # The unprocessed member with the earliest candidate is final.
        j = 0
        bt = Inf
        for k in 1:m
            if !processed[k] && best[k] < bt
                bt = best[k]
                j = k
            end
        end
        j == 0 && break
        processed[j] = true

        ind = state.individuals[mem[j]]
        ind.infection_time = best[j]
        ind.state[:infected] = true
        ind.state[:index] = src[j] == 0
        if src[j] != 0
            infector = state.individuals[src[j]]
            ind.parent_id = infector.id
            ind.generation = infector.generation + 1
            ind.chain_id = infector.chain_id
        end
        # Stamp this case's natural history (:infectious_time, :outcome_time, …).
        resolve_transitions!(state, ind)

        open_t = _window_open(ind, model.from)   # infectiousness onset
        isfinite(open_t) || continue             # never became infectious
        close_t = _window_close(ind, model.until)  # earliest removal

        # Expose every still-susceptible household-mate: a contact interval from
        # the kernel, timed from onset and accepted within the infectious window.
        for k in 1:m
            (k == j || processed[k]) && continue
            dt = rand(rng, _pairkernel(model.kernel, mem[j], mem[k]))
            cand = open_t + dt
            (cand <= close_t && cand < best[k]) || continue
            best[k] = cand
            src[k] = mem[j]
        end
    end
    return nothing
end

# When the infector becomes infectious: the `from` state's time (the infection
# time itself when the kernel times from :infection, otherwise a state key).
function _window_open(ind, from::Symbol)
    from === :infection ? ind.infection_time :
    (get(ind.state, Symbol(from, :_time), Inf)::Float64)
end

# When the infectious window closes: the earliest of the `until` removal states'
# times, or Inf when none has been reached (infectious until the clique is spent).
function _window_close(ind, until::Tuple)
    isempty(until) && return Inf
    return minimum(get(ind.state, Symbol(s, :_time), Inf)::Float64 for s in until)
end

# Resolve the contact-interval distribution for an ordered (infector,
# susceptible) pair: a shared distribution, or a callable for covariate models.
_pairkernel(k::ContinuousUnivariateDistribution, i, j) = k
_pairkernel(k, i, j) = k(i, j)

# A community introduction time under the external hazard: the constant case is
# its Exponential survival time, a distribution is sampled directly.
_ext_draw(rng, α::Real) = rand(rng, Exponential(1 / α))
_ext_draw(rng, d::ContinuousUnivariateDistribution) = rand(rng, d)
