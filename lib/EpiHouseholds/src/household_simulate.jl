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
    condition !== nothing && return _retry_for_condition(
        () -> _simulate(model, sim_opts; interventions, attributes, progression,
            observation, rng, condition = nothing, max_attempts),
        condition, max_attempts)

    length(model.members) > 1 && foreach(_validate_household_capacity, interventions)
    from = _resolve_infectious_from(model.from, progression)
    Tobs = model.obs_end

    # A community hazard over an unbounded window would introduce every member;
    # require a finite observation window instead (as the network simulator does).
    _ext_active(model.external_hazard) && !isfinite(Tobs) &&
        throw(ArgumentError(
            "an external hazard needs a finite `obs_end` (an unbounded window seeds " *
            "every household member); build the process with e.g. `obs_end = 30.0`"))

    state = new_state(model, progression, attributes, rng)
    add_individuals!(state, length(model.household_of), interventions;
        setup = (ind, i) -> (ind.state[:household] = model.household_of[i]))

    # Reuse the population lookup across household races.
    initial_cases = sim_opts.initial_cases === nothing ? nothing :
                    Set(sim_opts.initial_cases)
    for mem in model.members
        EpiBranch._sellke_race!(state, mem, rng;
            from = from, until = model.until, interventions = interventions,
            max_time = EpiBranch._max_time(sim_opts),
            risks = EpiBranch.transmission_risks(model),
            seed! = (best, members, r) -> _seed_clique!(
                best, members, state, model.external_hazard, Tobs, r;
                initial_cases = initial_cases),
            introduction = _ext_active(model.external_hazard) ?
                           (EpiBranch._ext_survival(model.external_hazard), Tobs) : nothing,
            targets = (inf, st) -> ((oid, _pairkernel(model.kernel, inf, oid, st, from))
            for oid in mem if oid != inf),
            # A case's contacts are its household-mates, traced whether or not
            # transmission reached them.
            contacts = (inf, st) -> (oid for oid in mem if oid != inf))
    end

    _reconcile_sellke_bookkeeping!(state)
    # Apply the observation model (under-reporting, report delays), as core
    # `simulate` does. A no-op for the default `NoObservation`.
    apply_observation!(observation, state, rng)
    return state
end

# Seed one household's candidate table: community introductions under the
# external hazard (each member drawn, kept if it lands within `[0, Tobs]`), or a
# single seeded index at time 0 when there is no external source.
function _seed_clique!(best, members, state, extsrc, Tobs, rng;
        initial_cases = nothing)
    initial_cases === nothing ||
        return EpiBranch._seed_initial_cases!(best, members, initial_cases)
    m = length(members)
    if _ext_active(extsrc)
        for k in 1:m
            t = _ext_draw(rng, extsrc, state.individuals[members[k]].susceptibility)
            t <= Tobs && (best[k] = t)
        end
    else
        best[rand(rng, 1:m)] = 0.0
    end
    return nothing
end

# Resolve the contact-interval distribution for an ordered (infector,
# susceptible) pair: a shared distribution, or a callable for covariate models.
function _pairkernel(k, i, j, state, from)
    EpiBranch.pair_kernel(k, i, j, state.individuals[i].infection_time,
        EpiBranch._window_open(state.individuals[i], from))
end

function EpiBranch._validate_initial_cases(model::HouseholdProcess, opts::SimOpts)
    EpiBranch._validate_initial_case_ids(opts, length(model.household_of), model.external_hazard)
end

# Separate household races revisit earlier times. Periodic shared budgets need
# a single chronological race; lifetime budgets remain valid across races.
_validate_household_capacity(::EpiBranch.AbstractIntervention) = nothing
function _validate_household_capacity(iv::EpiBranch.InterventionWrapper)
    _validate_household_capacity(iv.intervention)
end
function _validate_household_capacity(iv::CapacityConstrained)
    isfinite(iv.period) && throw(ArgumentError(
        "finite-period capacity budgets require chronological admission across households; " *
        "use period = Inf for a shared lifetime budget, or simulate one household"))
    _validate_household_capacity(iv.intervention)
end
