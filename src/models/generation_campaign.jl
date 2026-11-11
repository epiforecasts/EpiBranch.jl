# ── Vaccination campaigns in generation counts ───────────────────────
#
# A count has no individuals to offer a dose to. A campaign acts on the
# susceptibles of each type between generations instead. Its doses for a generation's
# interval come from the budget of a `CapacityConstrained` (read, for a
# schedule, at the start of each period) and the window of a `Scheduled`
# around it, in either order. As in the individual-based engine once
# `resource_key` splits `GroupVaccination` by type, every type has the whole
# budget to itself.

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
_campaign(iv::AbstractIntervention, window, budget) = throw(
    ArgumentError(
        "simulate_by_generation cannot apply $(nameof(typeof(iv))); it applies " *
            "GroupVaccination under CapacityConstrained and Scheduled"
    )
)
function _campaign(s::Scheduled, window, budget)
    s.window === nothing && throw(
        ArgumentError(
            "simulate_by_generation can only apply a Scheduled with a start_time and " *
                "end_time: a case count or other condition has no clock between generations"
        )
    )
    return _campaign(
        s.intervention, (max(window[1], s.window.start), min(window[2], s.window.stop)), budget
    )
end
function _campaign(cc::CapacityConstrained, window, budget)
    budget === nothing || throw(
        ArgumentError("simulate_by_generation needs a single CapacityConstrained per campaign")
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
            "simulate_by_generation needs a finite population_size for the " *
                "interventions $names to act on"
        )
    )
    return [_campaign(iv, (0.0, Inf), nothing) for iv in ivs]
end

# The doses a campaign gives to each type between `t0` and `t1`. With
# `carry_over` (or a single budget, `period = Inf`) unused doses of earlier
# periods are given as soon as the window opens, so the doses given by a time
# are the cumulative budget of the periods begun by then, none before the
# window opens. Without it, each period's own budget is given in the period.
# Only whole doses are given.
function _interval_doses(c::_Campaign, t0::Real, t1::Real)
    s, e = c.window
    a = max(t0, s)
    b = min(t1, e)
    b > a || return 0
    cc = c.budget
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

# Give a campaign's doses for `[t0, t1)` to the susceptibles. A dose reaches a
# susceptible of a type with probability equal to the share of susceptibles in
# the population of the type at the start of the interval, and protects with
# probability `efficacy`. With no doses nothing is drawn.
function _vaccinate!(rng::AbstractRNG, c::_Campaign, t0, t1, susceptibles, sizes)
    doses = _interval_doses(c, t0, t1)
    doses == 0 && return susceptibles
    for i in eachindex(susceptibles)
        (susceptibles[i] <= 0 || sizes[i] <= 0) && continue
        reached = min(rand(rng, Binomial(doses, susceptibles[i] / sizes[i])), susceptibles[i])
        susceptibles[i] -= rand(rng, Binomial(reached, c.efficacy))
    end
    return susceptibles
end
