# ── Interventions on generation counts ───────────────────────────────
#
# `simulate_by_generation` holds no individuals, so an intervention cannot act
# on one. An intervention defines `count_removals` to act on the counts
# instead: given what is left of each type, it returns how many susceptibles of
# each type it removes over the days of one generation. The wrappers pass a
# budget of doses down to the vaccination they wrap, as `CapacityConstrained`
# does on individuals by admitting or refusing each one, and `Scheduled`
# decides whether the generation is inside its window.

# What an intervention can see of the generation it acts before: the days it
# covers, and the susceptibles and population of each type. `susceptibles` is
# a copy owned by the context, lowered as people are removed. `ledgers` holds
# what an intervention keeps between generations, under a key giving its place
# in the list of interventions, so identical interventions keep separate
# records.
struct GenerationContext{R <: AbstractRNG}
    rng::R
    start_time::Float64
    end_time::Float64
    generation::Int
    cumulative_cases::Int
    susceptibles::Vector{Int}
    type_sizes::Vector{Int}
    ledgers::Dict{Tuple, Any}
end

# Apply each intervention in turn, each seeing the susceptibles the one before
# left.
function _apply_interventions!(interventions, ctx::GenerationContext)
    for (i, iv) in enumerate(interventions)
        result = count_removals(iv, ctx, Inf, (i,))
        removed = result.removed
        ctx.susceptibles .-= removed
    end
    return nothing
end

"""
    EpiBranch.count_removals(intervention, ctx, budget, key) -> (removed, doses)

For extension authors: what `intervention` does to the counts of
[`simulate_by_generation`](@ref) over one generation. `ctx` holds the days the
generation covers (`ctx.start_time`, `ctx.end_time`), the susceptibles of each
type (`ctx.susceptibles`) and the population of each type (`ctx.type_sizes`),
with the run's `ctx.rng`. `budget` is the number of doses a wrapping
[`CapacityConstrained`](@ref) allows (`Inf` if none does). `key` is a tuple
giving the place of this intervention in the list, under which `ctx.ledgers`
can store what it must remember between generations.

Returns the number of susceptibles of each type removed, a `Vector{Int}`, and
the number of doses used, which a wrapping `CapacityConstrained` charges to its
budget. An intervention without a method raises an `ArgumentError` when it is
given to `simulate_by_generation`.
"""
function count_removals(iv::AbstractIntervention, ctx::GenerationContext, budget, key)
    return _no_count_method(iv)
end

function _no_count_method(iv)
    throw(
        ArgumentError(
            "simulate_by_generation cannot apply $(nameof(typeof(iv))) to counts; it " *
                "supports GroupVaccination, optionally inside Scheduled and " *
                "CapacityConstrained. Define EpiBranch.count_removals for another intervention."
        )
    )
end

# Raise the error for an intervention the counts cannot take when
# `simulate_by_generation` is called, rather than in the generation that first
# reaches it, which a run that goes extinct early never would. An intervention
# with a `count_removals` method of its own is accepted, and may add a method
# of this to check its own settings.
function _validate_counts(iv::AbstractIntervention)
    _has_own_method(count_removals, typeof(iv), AbstractIntervention) ||
        _no_count_method(iv)
    return nothing
end
_validate_counts(w::InterventionWrapper) = _validate_counts(w.intervention)

# ── Scheduled ────────────────────────────────────────────────────────

# The condition of `Scheduled` reads `max_infection_time`, `cumulative_cases`
# and `current_generation` off the state. A count has all three at the first
# day of the generation.
function _validate_counts(s::Scheduled)
    isempty(s.triggers) || throw(
        ArgumentError(
            "a Scheduled with `start_after` needs individuals to read, which " *
                "simulate_by_generation does not hold; use start_time, end_time or " *
                "start_after_cases"
        )
    )
    return _validate_counts(s.intervention)
end

function count_removals(s::Scheduled, ctx::GenerationContext, budget, key)
    view = (
        max_infection_time = ctx.start_time, cumulative_cases = ctx.cumulative_cases,
        current_generation = ctx.generation,
    )
    s.condition(view) || return (removed = zero(ctx.susceptibles), doses = 0.0)
    return count_removals(s.intervention, ctx, budget, (key..., :scheduled))
end

# ── CapacityConstrained ──────────────────────────────────────────────

# The doses its budget makes available over the generation, less any it has
# already given if unused capacity is carried over: those of every period
# started before the generation's end, or, without `carry_over`, only of the
# periods that start within it. An infinite period is a single budget, read
# at day 0.
function _generation_budget(cc::CapacityConstrained, ctx::GenerationContext, given)
    if isinf(cc.period)
        return floor(_period_budget(cc.budget_per_period, 0.0) - given)
    end
    first_period = cc.carry_over ? 0 : ceil(Int, ctx.start_time / cc.period)
    last_period = ceil(Int, ctx.end_time / cc.period) - 1
    total = sum(
        (_period_budget(cc.budget_per_period, k * cc.period) for k in first_period:last_period);
        init = 0.0
    )
    return floor(total - (cc.carry_over ? given : 0.0))
end

function count_removals(cc::CapacityConstrained, ctx::GenerationContext, budget, key)
    ledger = get!(() -> Ref(0.0), ctx.ledgers, (key..., :capacity))
    available = max(min(budget, _generation_budget(cc, ctx, ledger[])), 0.0)
    result = count_removals(cc.intervention, ctx, available, (key..., :capacity_inner))
    ledger[] += result.doses
    return result
end

# ── GroupVaccination ─────────────────────────────────────────────────

# What a vaccination remembers per type between generations: how many of its
# members have been given a dose, and how many of those doses reached a
# susceptible and failed to protect it. Those members stay susceptible and
# cannot be dosed again.
mutable struct _VaccinationLedger
    dosed::Vector{Int}
    unprotected::Vector{Int}
end

function _validate_counts(gv::GroupVaccination)
    efficacy(gv) isa Real || throw(
        ArgumentError(
            "simulate_by_generation needs a numeric efficacy for GroupVaccination, " *
                "not $(typeof(efficacy(gv)))"
        )
    )
    gv.delay_to_immunity isa Real && gv.delay_to_immunity == 0 || throw(
        ArgumentError(
            "simulate_by_generation does not support a delay_to_immunity other than 0"
        )
    )
    gv.coverage isa Real || throw(
        ArgumentError(
            "simulate_by_generation needs a numeric coverage for GroupVaccination, " *
                "not $(typeof(gv.coverage))"
        )
    )
    return nothing
end

# Split `total` doses between types in proportion to `weights`, as whole
# doses that add up to `total` (rounding the running share, so that no type
# is favoured by where the remainder falls).
function _split_doses(total::Int, weights::Vector{Int})
    w = sum(weights)
    w == 0 && return zero(weights)
    shares = round.(Int, cumsum(weights) .* (total / w))
    return diff([0; shares])
end

function count_removals(gv::GroupVaccination, ctx::GenerationContext, budget, key)
    nt = length(ctx.susceptibles)
    ledger = get!(() -> _VaccinationLedger(zeros(Int, nt), zeros(Int, nt)), ctx.ledgers, key)
    # Those the campaign can still reach: the covered share of the
    # population of each type, less those already dosed.
    reachable = max.(round.(Int, gv.coverage .* ctx.type_sizes) .- ledger.dosed, 0)
    total = isfinite(budget) ? min(Int(budget), sum(reachable)) : sum(reachable)
    doses = min.(_split_doses(total, reachable), reachable)
    removed = zeros(Int, nt)
    for i in 1:nt
        doses[i] == 0 && continue
        # A dose goes to a member who has not had one; the susceptibles among
        # them are those the campaign has not already failed to protect.
        undosed = ctx.type_sizes[i] - ledger.dosed[i]
        eligible = ctx.susceptibles[i] - ledger.unprotected[i]
        hit = min(rand(ctx.rng, Binomial(doses[i], clamp(eligible / undosed, 0.0, 1.0))), eligible)
        removed[i] = rand(ctx.rng, Binomial(hit, efficacy(gv)))
        ledger.unprotected[i] += hit - removed[i]
        ledger.dosed[i] += doses[i]
    end
    return (removed = removed, doses = Float64(sum(doses)))
end
