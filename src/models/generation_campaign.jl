# ── Vaccination campaigns in generation counts ───────────────────────
#
# A count has no individuals to offer a dose to. A campaign acts on the
# susceptibles of each type between generations instead. Its doses for a generation's
# interval come from the budget of a `CapacityConstrained` (read, for a
# schedule, at the start of each period) and the window of a `Scheduled`
# around it, in either order. The types share one budget, as in every model.

# A vaccination campaign `simulate_by_generation` can apply: the efficacy of
# its dose, the days it may give doses between, and the budget limiting how
# many.
struct _Campaign{B}
    efficacy::Float64
    window::Tuple{Float64, Float64}
    budget::B
end

# Walks the wrappers around a `GroupVaccination`, intersecting the windows of
# any `Scheduled` and finding the one `CapacityConstrained`. Anything else, or
# a `Scheduled` whose condition is not a calendar window, cannot be applied to
# a count and is named in the error.
_campaign(iv::AbstractIntervention, window, budget) =
    _refuse_campaign("simulate_by_generation", nameof(typeof(iv)))
function _campaign(s::Scheduled, window, budget)
    w = _clock_window(s, "simulate_by_generation")
    return _campaign(
        s.intervention, (max(window[1], w.start), min(window[2], w.stop)), budget
    )
end
function _campaign(cc::CapacityConstrained, window, budget)
    budget === nothing || _refuse_campaign(
        "simulate_by_generation", "a campaign with more than one CapacityConstrained"
    )
    cc.on_refusal isa DropRefused || _refuse_campaign(
        "simulate_by_generation", "`on_refusal = $(nameof(typeof(cc.on_refusal)))`",
        " (each person is offered a dose once)"
    )
    return _campaign(cc.intervention, window, cc)
end
function _campaign(gv::GroupVaccination, window, budget)
    budget === nothing && throw(
        ArgumentError(
            "simulate_by_generation needs GroupVaccination wrapped in CapacityConstrained: " *
                "without a dose budget there is no number of doses to give"
        )
    )
    gv.group_key === :type || throw(
        ArgumentError(
            "simulate_by_generation applies GroupVaccination to the types, so it needs " *
                "group_key = :type, not :$(gv.group_key)"
        )
    )
    eff = efficacy(gv)
    eff isa Real || throw(
        ArgumentError(
            "simulate_by_generation needs a numeric efficacy for GroupVaccination, " *
                "not $(typeof(eff))"
        )
    )
    for (name, value, none) in (
            (:coverage, gv.coverage, 1), (:dose_delay, gv.dose_delay, 0),
            (:delay_to_immunity, delay_to_immunity(gv), 0), (:waning, waning(gv), nothing),
        )
        value === none || (value isa Real && none isa Real && value == none) || throw(
            ArgumentError(
                "simulate_by_generation cannot apply GroupVaccination with $name = " *
                    "$value: a dose protects at once, with probability efficacy"
            )
        )
    end
    return _Campaign(Float64(eff), window, budget)
end

# The campaigns a spec holds, or an error naming every intervention that
# cannot be applied, and the missing `generation_length` or population.
function _generation_campaigns(spec::ModelSpec, generation_length, finite_population::Bool)
    isempty(progression(spec)) && attributes(spec) isa NoAttributes &&
        observation(spec) isa NoObservation && recorder(spec) isa NoContactRecorder ||
        throw(
        ArgumentError(
            "simulate_by_generation reads only the interventions of a ModelSpec, not its " *
                "progression, attributes, observation or recorder"
        )
    )
    ivs = interventions(spec)
    isempty(ivs) && return _Campaign[]
    names = join((nameof(typeof(iv)) for iv in ivs), ", ")
    generation_length === nothing && throw(
        ArgumentError(
            "simulate_by_generation needs a generation_length to date the doses of the " *
                "interventions $names"
        )
    )
    finite_population || throw(
        ArgumentError(
            "simulate_by_generation needs a finite population for the " *
                "interventions $names to act on"
        )
    )
    population = EpiBranch.population(process(spec))
    return [_by_type(_campaign(iv, (0.0, Inf), nothing), population) for iv in ivs]
end

# A budget per group becomes one budget for each type, in the order of the
# types, checked against the population's groups. A budget shared by the types
# stays as it is.
_by_type(c::_Campaign, population) = _by_type(c, c.budget.budget_per_period, population)
_by_type(c::_Campaign, ::Any, population) = c
function _by_type(c::_Campaign, pg::PerGroup, population)
    _group_budgets(pg, population)
    budgets = [_for_resource(c.budget, name) for name in _group_names(population)]
    return _Campaign(c.efficacy, c.window, budgets)
end

# The doses a campaign gives to each type between `t0` and `t1`. With
# `carry_over` (or a single budget, `period = Inf`) unused doses of earlier
# periods are given as soon as the window opens, so the doses given by a time
# are the cumulative budget of the periods begun by then, none before the
# window opens. Without it, each period's own budget is given in the period.
# Only whole doses are given.
function _interval_doses(c::_Campaign, t0::Real, t1::Real, cc = c.budget)
    s, e = c.window
    a = max(t0, s)
    b = min(t1, e)
    b > a || return 0
    if cc.carry_over || isinf(cc.period)
        before = a <= s ? 0.0 : _doses_by(cc, a)
        return max(0, floor(Int, _doses_by(cc, b)) - floor(Int, before))
    end
    first_period = _periods_begun(cc, a)
    last_period = _periods_begun(cc, b) - 1
    total = sum(
        (_period_budget(cc.budget_per_period, k * cc.period) for k in first_period:last_period);
        init = 0.0
    )
    return floor(Int, total)
end

# The number of periods that start before `t`, with a little slack so a time
# that is a whole number of periods is not pushed into the next one.
_periods_begun(cc::CapacityConstrained, t) = max(0, ceil(Int, t / cc.period - 1.0e-9))

function _doses_by(cc::CapacityConstrained, t)
    isinf(cc.period) && return t > 0 ? _period_budget(cc.budget_per_period, 0.0) : 0.0
    return sum(
        (_period_budget(cc.budget_per_period, k * cc.period) for k in 0:(_periods_begun(cc, t) - 1));
        init = 0.0
    )
end

# The doses of `[t0, t1)` given to each type. A shared budget is a pool of doses
# for everyone: each dose goes to a type in proportion to its population, as it
# would to a person picked at random. A budget per group gives each type its own.
function _doses_by_type(rng::AbstractRNG, c::_Campaign, t0, t1, sizes)
    return _doses_by_type(rng, c, c.budget, t0, t1, sizes)
end
function _doses_by_type(rng::AbstractRNG, c::_Campaign, budgets::AbstractVector, t0, t1, sizes)
    return [_interval_doses(c, t0, t1, b) for b in budgets]
end
function _doses_by_type(rng::AbstractRNG, c::_Campaign, ::CapacityConstrained, t0, t1, sizes)
    doses = _interval_doses(c, t0, t1)
    doses == 0 && return zeros(Int, length(sizes))
    length(sizes) == 1 && return [doses]
    return rand(rng, Multinomial(doses, Float64.(sizes) ./ sum(sizes)))
end

# Give a campaign's doses for `[t0, t1)` to the susceptibles. A dose reaches a
# susceptible of a type with probability equal to the share of susceptibles in
# the population of the type at the start of the interval, and protects with
# probability `efficacy`. With no doses nothing is drawn.
function _vaccinate!(rng::AbstractRNG, c::_Campaign, t0, t1, susceptibles, sizes)
    doses = _doses_by_type(rng, c, t0, t1, sizes)
    for i in eachindex(susceptibles)
        doses[i] == 0 && continue
        (susceptibles[i] <= 0 || sizes[i] <= 0) && continue
        reached = min(rand(rng, Binomial(doses[i], susceptibles[i] / sizes[i])), susceptibles[i])
        susceptibles[i] -= rand(rng, Binomial(reached, c.efficacy))
    end
    return susceptibles
end
