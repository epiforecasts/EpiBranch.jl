# ── Vaccination campaigns in generation counts ───────────────────────
#
# A campaign gives its doses to the contacts of each generation, in the order
# they are made. Its doses for a generation's interval come from the budget of a
# `CapacityConstrained` and the window of a `Scheduled` around it.

# A vaccination campaign `simulate_by_generation` can apply: the efficacy of
# its dose, the days it may give doses between, and the budget limiting how
# many.
struct _Campaign{B}
    efficacy::Float64
    window::Tuple{Float64, Float64}
    budget::B
end

# Reads the wrappers around an intervention (the same ones the mixing
# approximations read), then its effect on a group (`count_dosing`). Anything
# else, or a `Scheduled` with a condition other than a calendar window, cannot be
# applied to a count and is named in the error.
function _campaign(iv::AbstractIntervention, window, budget)
    engine = "simulate_by_generation"
    leaf, window, budget = _unwrap_campaign(iv, window, budget, engine)
    dosing = count_dosing(leaf)
    dosing === nothing && _refuse_campaign(engine, nameof(typeof(leaf)))
    budget === nothing && throw(
        ArgumentError(
            "simulate_by_generation needs GroupVaccination wrapped in CapacityConstrained: " *
                "without a dose budget there is no number of doses to give"
        )
    )
    dosing.group_key === :type || throw(
        ArgumentError(
            "simulate_by_generation applies GroupVaccination to the types, so it needs " *
                "group_key = :type, not :$(dosing.group_key)"
        )
    )
    dosing.efficacy isa Real || throw(
        ArgumentError(
            "simulate_by_generation needs a numeric efficacy for GroupVaccination, " *
                "not $(typeof(dosing.efficacy))"
        )
    )
    for (name, value, none) in (
            (:coverage, dosing.coverage, 1), (:dose_delay, dosing.dose_delay, 0),
            (:delay_to_immunity, dosing.delay_to_immunity, 0), (:waning, dosing.waning, nothing),
        )
        value === none || (value isa Real && none isa Real && value == none) || throw(
            ArgumentError(
                "simulate_by_generation cannot apply GroupVaccination with $name = " *
                    "$value: a dose protects at once, with probability efficacy"
            )
        )
    end
    return _Campaign(Float64(dosing.efficacy), window, budget)
end

# The campaigns a spec holds, or an error naming every intervention that
# cannot be applied, and the missing `generation_length` or population.
function _check_spec_inputs(spec::ModelSpec)
    isempty(progression(spec)) && attributes(spec) isa NoAttributes &&
        observation(spec) isa NoObservation && recorder(spec) isa NoContactRecorder ||
        throw(
        ArgumentError(
            "simulate_by_generation reads only the interventions of a ModelSpec, not its " *
                "progression, attributes, observation or recorder"
        )
    )
    return nothing
end

function _generation_campaigns(spec::ModelSpec, generation_length, finite_population::Bool)
    _check_spec_inputs(spec)
    ivs = interventions(spec)
    isempty(ivs) && return _Campaign[]
    generation_length === nothing && throw(
        ArgumentError(
            "simulate_by_generation needs a generation_length to date the doses of the " *
                "interventions $(_intervention_names(ivs))"
        )
    )
    finite_population || throw(
        ArgumentError(
            "simulate_by_generation needs a finite population for the " *
                "interventions $(_intervention_names(ivs)) to act on"
        )
    )
    population = EpiBranch.population(process(spec))
    return [_by_type(_campaign(iv, (0.0, Inf), nothing), population) for iv in ivs]
end

_intervention_names(ivs) = join((nameof(typeof(iv)) for iv in ivs), ", ")

# The inputs the campaigns of a spec were built from. A later call with the
# same spec reuses the campaigns. A filter calls a continuation for every
# particle at every step, and building the campaigns costs far more than
# simulating a generation.
struct _CampaignCache{V}
    process::Any
    population::Any
    interventions::Vector{AbstractIntervention}
    generation_length::Float64
    campaigns::V
end

# Keyed by the identity of the intervention vector of the spec. The vector
# lives as long as the spec and is shared with any copy of it. `WeakKeyDict`
# locks itself. Two threads that miss together both build, and either result
# is right.
const _CAMPAIGN_CACHE = WeakKeyDict{Vector{AbstractIntervention}, _CampaignCache}()

function _cache_valid(c::_CampaignCache, spec::ModelSpec, generation_length::Float64)
    ivs = interventions(spec)
    return c.generation_length == generation_length &&
        c.process === process(spec) &&
        c.population === EpiBranch.population(process(spec)) &&
        length(c.interventions) == length(ivs) &&
        all(i -> c.interventions[i] === ivs[i], eachindex(ivs))
end

# As `_generation_campaigns`, built once per spec and reused. The checks of the
# spec run on every call. A spec that fails them is never cached, so the errors
# are unchanged. The cache is rebuilt if the interventions or process of the
# spec, or the `generation_length`, differ from what it was built from.
function _cached_generation_campaigns(
        spec::ModelSpec, generation_length, finite_population::Bool
    )
    _check_spec_inputs(spec)
    ivs = interventions(spec)
    (isempty(ivs) || generation_length === nothing || !finite_population) &&
        return _generation_campaigns(spec, generation_length, finite_population)
    cached = get(_CAMPAIGN_CACHE, ivs, nothing)
    if cached !== nothing && _cache_valid(cached, spec, Float64(generation_length))
        return cached.campaigns
    end
    campaigns = _generation_campaigns(spec, generation_length, finite_population)
    _CAMPAIGN_CACHE[ivs] = _CampaignCache(
        process(spec), EpiBranch.population(process(spec)), copy(ivs),
        Float64(generation_length), campaigns
    )
    return campaigns
end

# A budget per group becomes one budget for each type, in the order of the
# types, checked against the groups of the population. A budget shared by the types
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

# The doses each campaign has for generation `gen`, with its efficacy: an Int
# for a shared budget, a vector for a budget per type. Campaigns with no doses
# are left out. `taken` holds the `used` entries those doses draw on.
function _generation_doses(campaigns, used, generation_length, gen)
    doses = Tuple{Union{Int, Vector{Int}}, Float64}[]
    taken = Vector{Int}[]
    isempty(campaigns) && return doses, taken
    t0, t1 = _generation_interval(generation_length, gen)
    for (k, c) in enumerate(campaigns)
        d = _doses_of(c, c.budget, t0, t1, used[k])
        (d isa Int ? d > 0 : any(>(0), d)) || continue
        push!(doses, (d, c.efficacy))
        push!(taken, used[k])
    end
    return doses, taken
end
_doses_of(c::_Campaign, cc::CapacityConstrained, t0, t1, used) =
    _doses_left(c, cc, t0, t1, only(used))
_doses_of(c::_Campaign, budgets::AbstractVector, t0, t1, used) =
    [_doses_left(c, b, t0, t1, u) for (b, u) in zip(budgets, used)]

# With `carry_over` (or a single budget) the doses not yet given, up to the end
# of the interval; without it, only those of the interval.
function _doses_left(c::_Campaign, cc::CapacityConstrained, t0, t1, used)
    (cc.carry_over || isinf(cc.period)) || return _interval_doses(c, t0, t1, cc)
    s, e = c.window
    b = min(t1, e)
    return b > s ? max(0, floor(Int, _doses_by(cc, b)) - used) : 0
end

# `_generation_doses` for a single type, written into `doses`; returns whether
# any campaign has a dose.
function _scalar_doses!(doses::Vector{Int}, campaigns, used, generation_length, gen)
    isempty(campaigns) && return false
    t0, t1 = _generation_interval(generation_length, gen)
    any_doses = false
    for (k, c) in enumerate(campaigns)
        cc = c.budget isa AbstractVector ? only(c.budget) : c.budget
        doses[k] = _doses_left(c, cc, t0, t1, only(used[k]))
        any_doses |= doses[k] > 0
    end
    return any_doses
end

# `_deplete_dosed!` for a single type, returning the contacts kept and the
# susceptibles left.
function _deplete_dosed(
        rng, ::PerContactDepletion, potential::Real, susceptibles::Int, size::Int,
        campaigns, doses::Vector{Int}, used
    )
    contacts = potential isa Int ? potential : rand(rng, Poisson(potential))
    kept = 0
    done = 0
    while done < contacts
        level = contacts
        for d in doses
            done < d < level && (level = d)
        end
        pass = 1.0
        for (k, c) in enumerate(campaigns)
            doses[k] >= level && (pass *= 1 - c.efficacy)
        end
        remaining = level - done
        while remaining > 0 && susceptibles > 0 && pass > 0
            remaining -= rand(rng, Geometric(pass * susceptibles / size)) + 1
            remaining < 0 && break
            kept += 1
            susceptibles -= 1
        end
        done = level
    end
    for k in eachindex(doses)
        used[k][1] += min(doses[k], contacts)
    end
    return kept, susceptibles
end

# How many of each type's `contacts` every campaign doses. A shared budget
# doses the first contacts of all the types, split between them
# hypergeometrically; a budget per type doses the first of its own. Campaigns
# are served in the same order of contacts.
function _dosed_contacts(rng::AbstractRNG, doses, contacts::Vector{Int})
    dosed = Vector{Vector{Int}}(undef, length(doses))
    given = zeros(Int, length(contacts))
    for k in sortperm([d isa Int ? d : 0 for (d, _) in doses])
        d = doses[k][1]
        if d isa Int
            _extend_dosed!(rng, given, contacts, d)
            dosed[k] = copy(given)
        else
            dosed[k] = min.(d, contacts)
        end
    end
    return dosed
end

# Raise `given` to `total` dosed contacts (or all of them if fewer), drawing the
# extra ones without replacement from those not yet dosed.
function _extend_dosed!(rng::AbstractRNG, given, contacts, total)
    extra = min(total, sum(contacts)) - sum(given)
    extra <= 0 && return given
    rest = contacts .- given
    pool = sum(rest)
    for i in eachindex(rest)
        pool -= rest[i]
        x = if i == lastindex(rest) || pool == 0
            extra
        elseif rest[i] == 0
            0
        else
            rand(rng, Hypergeometric(rest[i], pool, extra))
        end
        given[i] += x
        extra -= x
    end
    return given
end

# As `_deplete!` when some contacts are dosed: a dosed contact is also blocked
# with the efficacy of each dose it has. The doses given are added to `taken`.
function _deplete_dosed!(
        rng, ::PerContactDepletion, potential::Vector, susceptibles, sizes, doses, taken
    )
    contacts = potential isa Vector{Int} ? potential : [rand(rng, Poisson(p)) for p in potential]
    dosed = _dosed_contacts(rng, doses, contacts)
    for k in eachindex(dosed)
        taken[k] .+= length(taken[k]) == 1 ? sum(dosed[k]) : dosed[k]
    end
    kept = zeros(Int, length(contacts))
    for i in eachindex(contacts)
        cuts = sort!(unique([d[i] for d in dosed if d[i] > 0]))
        previous = 0
        for cut in [cuts; contacts[i]]
            cut > previous || continue
            pass = prod((1 - doses[k][2] for k in eachindex(dosed) if dosed[k][i] >= cut); init = 1.0)
            remaining = cut - previous
            previous = cut
            while remaining > 0 && susceptibles[i] > 0 && pass > 0
                remaining -= rand(rng, Geometric(pass * susceptibles[i] / sizes[i])) + 1
                remaining < 0 && break
                kept[i] += 1
                susceptibles[i] -= 1
            end
        end
    end
    return kept
end
