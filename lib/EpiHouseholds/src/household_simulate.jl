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
              progression, observation, recorder, rng, condition, max_attempts)

Simulate a household outbreak exactly in continuous time; this is the model
the pairwise likelihood describes. The natural history and interventions come
from the caller (a bare process, or a `ModelSpec`), and the start of the
infectious period (`from`) is taken from the `progression`. Returns an EpiBranch
`SimulationState`; `linelist(state)` turns it into a table with one row per
case.

With no external hazard each household starts with one index case at day 0
and spreads only within the household. With an `external_hazard` (a positive
rate or a distribution of the time of infection from outside), community
introductions happen over the first `model.obs_end` days, which must be
finite. Chosen `initial_cases` are infected at day 0 regardless, and any
external hazard still acts on everyone else from day 0.
"""
function _simulate(
        model::HouseholdProcess, sim_opts::SimOpts;
        interventions, attributes, progression, observation, recorder, rng,
        condition, max_attempts
    )
    condition !== nothing && return _retry_for_condition(
        () -> _simulate(
            model, sim_opts; interventions, attributes, progression,
            observation, recorder, rng, condition = nothing, max_attempts
        ),
        condition, max_attempts
    )

    from = _resolve_infectious_from(model.from, progression)
    Tobs = model.obs_end

    # A community hazard over an unbounded window would introduce every member;
    # require a finite observation window instead (as the network simulator does).
    _ext_active(model.external_hazard) && !isfinite(Tobs) &&
        throw(
        ArgumentError(
            "an external hazard needs a finite `obs_end` (an unbounded window seeds " *
                "every household member); build the process with e.g. `obs_end = 30.0`"
        )
    )

    state = new_state(model, progression, attributes, rng)
    add_individuals!(
        state, length(model.household_of), interventions;
        setup = (ind, i) -> (ind.state[:household] = model.household_of[i])
    )

    # Reuse the population lookup across household races. The population is
    # extinct only when every household's race ran to its own extinction,
    # rather than being cut off at `max_time` with candidates still pending.
    initial_cases = sim_opts.initial_cases === nothing ? nothing :
        Set(sim_opts.initial_cases)
    # The keys the kernel's hazards depend on, for `_sellke_race!` to redraw a
    # pending contact when one moves.
    watched = EpiBranch.watched_records(model.kernel)
    # Only a policy that can read population-wide state — cases in other
    # households, a kernel reading host records an intervention can move, a
    # capacity budget shared across households — needs every household on one
    # clock. `reads_population_state` is how an intervention declares that;
    # `race_groups` is how the process itself partitions races for a kernel,
    # which `HouseholdProcess` narrows to one race per household unless the
    # kernel's watched records say otherwise.
    races = any(EpiBranch.reads_population_state, interventions) ?
        (collect(eachindex(model.household_of)),) : race_groups(model, model.kernel)
    extinct = true
    for mem in races
        extinct &= EpiBranch._sellke_race!(
            state, mem, rng;
            from = from, until = model.until, interventions = interventions,
            max_time = EpiBranch._max_time(sim_opts),
            risks = EpiBranch.transmission_risks(model),
            watches = (watched,), recorder = recorder,
            seed! = (best, members, r) -> _seed_household_race!(
                best, members, model, state, Tobs, r, initial_cases
            ),
            introduction = _ext_active(model.external_hazard) ?
                (EpiBranch._ext_survival(model.external_hazard), Tobs) : nothing,
            targets = (inf, st) -> (
                (oid, _pairkernel(model.kernel, inf, oid, st, from))
                    for oid in model.members[model.household_of[inf]] if oid != inf
            ),
            # A case's contacts are its household-mates, traced whether or not
            # transmission reached them.
            contacts = (inf, st) -> (
                oid
                    for oid in model.members[model.household_of[inf]] if oid != inf
            )
        )
    end

    _reconcile_sellke_bookkeeping!(state, extinct)
    # Apply the observation model (under-reporting, report delays), as core
    # `simulate` does. A no-op for the default `NoObservation`.
    apply_observation!(observation, state, rng)
    return state
end

# Seed one household's candidate table: chosen `initial_cases` at time 0, each
# remaining member drawn from the external hazard and kept if it lands within
# `[0, Tobs]`, or a single seeded index at time 0 when there is neither.
function _seed_clique!(
        best, members, state, extsrc, Tobs, rng;
        initial_cases = nothing
    )
    if initial_cases !== nothing
        EpiBranch._seed_initial_cases!(best, members, initial_cases)
        _ext_active(extsrc) &&
            _seed_external!(best, members, state, extsrc, Tobs, rng, initial_cases)
        return nothing
    end
    m = length(members)
    if _ext_active(extsrc)
        _seed_external!(best, members, state, extsrc, Tobs, rng)
    else
        best[rand(rng, 1:m)] = 0.0
    end
    return nothing
end

# Draw an external-hazard introduction time for each member not in `skip`
# (e.g. already-seeded chosen cases), keeping those landing within `[0, Tobs]`.
function _seed_external!(best, members, state, extsrc, Tobs, rng, skip = ())
    for k in eachindex(members)
        members[k] in skip && continue
        t = _ext_draw(rng, extsrc, state.individuals[members[k]].susceptibility)
        t <= Tobs && (best[k] = t)
    end
    return nothing
end

# Resolve the contact-interval distribution for an ordered (infector,
# susceptible) pair: a shared distribution, or a callable for covariate models.
function _pairkernel(k, i, j, state, from)
    return EpiBranch.pair_kernel(
        k, i, j, state.individuals[i].infection_time,
        EpiBranch._window_open(state.individuals[i], from), state
    )
end

function EpiBranch._validate_initial_cases(model::HouseholdProcess, opts::SimOpts)
    return EpiBranch._validate_initial_case_ids(opts, length(model.household_of))
end

# One race per household by default, since each household's outbreak is
# otherwise independent; a kernel whose watched records name any key needs
# every household sharing one clock instead, so a key moving on one
# household's host redraws a pending contact in another that reads it.
function race_groups(model::HouseholdProcess, kernel)
    return isempty(EpiBranch.watched_records(kernel)) ?
        model.members : (collect(eachindex(model.household_of)),)
end

# A race's `members` can span more than one household (sharing a clock, per
# `race_groups`), so each household within it is seeded as its own clique
# rather than drawing one index case across the merged group. `best` and
# `members` are grouped by household, using position within the race rather
# than global id, before `_seed_clique!` sees them.
function _seed_household_race!(best, members, model, state, Tobs, rng, initial_cases)
    households = Dict{Int, Vector{Int}}()
    for k in eachindex(members)
        push!(get!(() -> Int[], households, model.household_of[members[k]]), k)
    end
    for idxs in values(households)
        _seed_clique!(
            view(best, idxs), view(members, idxs), state, model.external_hazard, Tobs,
            rng; initial_cases
        )
    end
    return nothing
end
