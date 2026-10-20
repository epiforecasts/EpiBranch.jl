# ── Continuous-time (Sellke) simulation ──────────────────────────────
#
# A household outbreak is a finite, depleting clique, so it is simulated by the
# Sellke/Dijkstra construction in continuous time — the exact generative model
# of the pairwise likelihood. Members
# are processed in increasing infection-time order; popping the earliest pending
# infection makes it final, because any not-yet-processed infector has a later
# infection time, becomes infectious no earlier, and so reaches it later.
#
# Every other concern is the shared engine: the population is built with the
# public `new_state`/`add_individuals!`, each infected member's natural history
# (latent period, onset, recovery, …) is stamped through `resolve_transitions!`,
# and the result is an EpiBranch `SimulationState` that `linelist` renders.

"""
    simulate_once(model::HouseholdProcess, sim_opts; interventions, attributes,
                  progression, observation, recorder, rng)

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
function simulate_once(
        model::HouseholdProcess, sim_opts::SimOpts;
        interventions, attributes, progression, observation, recorder, rng
    )
    from = something(model.from, infectious_from(progression))
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

    initial_cases = sim_opts.initial_cases === nothing ? nothing :
        Set(sim_opts.initial_cases)
    # The keys the kernel's hazards depend on, for the race to redraw a pending
    # contact when one moves, and the projection it reads them through, so the
    # race can check that declaration against what the projection actually
    # reads.
    watched = EpiBranch.watched_records(model.kernel)
    projection = EpiBranch.kernel_projection(model.kernel)
    # Every household runs on one clock, so an intervention, stopping rule or
    # kernel that reads the whole population sees cases in time order.
    EpiBranch.sellke_race!(
        state, collect(eachindex(model.household_of)), rng, sim_opts;
        from = from, until = model.until, interventions = interventions,
        risks = EpiBranch.transmission_risks(model),
        watches = (watched,), projections = (projection,), recorder = recorder,
        seed! = (best, members, r) -> _seed_household_race!(
            best, members, model, state, Tobs, r, initial_cases
        ),
        introduction = _ext_active(model.external_hazard) ?
            (EpiBranch._ext_prepare(model.external_hazard), Tobs) : nothing,
        targets = (inf, st) -> (
            (oid, _pairkernel(model.kernel, inf, oid, st, from))
                for oid in model.members[model.household_of[inf]] if oid != inf
        ),
        # A case's contacts are its household-mates, traced whether or not
        # transmission reached them.
        contacts = (inf, st) -> (
            oid for oid in model.members[model.household_of[inf]] if oid != inf
        )
    )

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
        t = _ext_draw(rng, extsrc, members[k], state.individuals[members[k]].susceptibility)
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

# The race's `members` are every person in id order, so a household's ids are
# also its positions in `best`. Each household is seeded as its own clique, in
# household order, rather than drawing one index case across the population.
function _seed_household_race!(best, members, model, state, Tobs, rng, initial_cases)
    for hh in model.members
        _seed_clique!(
            view(best, hh), view(members, hh), state, model.external_hazard, Tobs,
            rng; initial_cases
        )
    end
    return nothing
end
