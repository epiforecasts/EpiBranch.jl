# ── Vaccinating the population on the individual-based race ──────────
#
# An intervention that doses the population, not the people the race creates,
# opts in through `population_ledger`. At run start the race asks each
# intervention once for a ledger and keeps the ledgers in `state.scratch`. It
# thins each contact against the susceptibles of its group, less the share
# that the ledgers report through `protected_share` at the time of exposure.

const _MASS_LEDGERS_KEY = :_mass_ledgers

"""
    population_ledger(intervention, population, window = (0.0, Inf), budget = nothing)

The ledger through which `intervention` doses the population, or `nothing`.
The race calls this once at run start for each intervention. An intervention
that answers with a ledger leaves the interventions of the race. The race then
asks its ledger for the share of each group protected
([`protected_share`](@ref EpiBranch.protected_share)) whenever a contact is
thinned against the susceptibles of a population. By default the answer is
`nothing` and the intervention acts on the people the race creates.

A budgeted [`MassVaccination`](@ref) answers with a ledger because its doses
must come from the susceptibles that infection has not already depleted. A
campaign defined outside the package can do the same by returning its own
ledger with a `protected_share` method. The wrappers around `intervention`
gather `window` and `budget`: [`Scheduled`](@ref) narrows `window` to its
calendar window and [`CapacityConstrained`](@ref) supplies `budget`. A wrapper
type defined outside the package forwards them.
"""
population_ledger(::Any, population, window = (0.0, Inf), budget = nothing) = nothing
function population_ledger(
        w::InterventionWrapper, population, window = (0.0, Inf), budget = nothing
    )
    return population_ledger(w.intervention, population, window, budget)
end
function population_ledger(s::Scheduled, population, window = (0.0, Inf), budget = nothing)
    s.window isa ClockWindow || return nothing
    narrowed = (max(window[1], s.window.start), min(window[2], s.window.stop))
    return population_ledger(s.intervention, population, narrowed, budget)
end
function population_ledger(
        cc::CapacityConstrained, population, window = (0.0, Inf), budget = nothing
    )
    limited = !(cc.budget_per_period isa Real && isinf(cc.budget_per_period))
    (limited && cc.on_refusal isa DropRefused && budget === nothing) || return nothing
    return population_ledger(cc.intervention, population, window, cc)
end
function population_ledger(mv::MassVaccination, population, window = (0.0, Inf), budget = nothing)
    budget === nothing && return nothing
    population isa NoPopulation && return nothing
    c = _by_type(
        _campaign(mv, window, budget; engine = "a model of individuals"), population
    )
    return _MassLedger(c, population)
end

"""
    protected_share(ledger, state, group, t) -> Real

The share of the people of `group` (1 for a population with one group, the
type for a grouped one) that `ledger` has protected at time `t`, as a number
from 0 to 1. The race subtracts it from the share of the group still
susceptible. The answer must be a step function of `t`: a query at `t` reads
the value at `t`, whatever times were asked before.
"""
function protected_share end

# State of a campaign between exposures: the doses given so far (`used`), the
# next budget period to give (`next`) and, for each period given, the time
# from which its doses count and the number protected in each group after it.
# A query at `t` reads the last entry dated before `t`.
mutable struct _MassLedger{C <: _Campaign}
    campaign::C
    used::Vector{Int}
    next::Int
    times::Vector{Float64}
    history::Vector{Vector{Int}}
    sizes::Vector{Int}
end

_group_sizes(p::SinglePopulation) = [p.size]
_group_sizes(p::GroupedPopulation) = p.sizes

function _MassLedger(c::_Campaign, population)
    sizes = _group_sizes(population)
    used = zeros(Int, c.budget isa AbstractVector ? length(c.budget) : 1)
    return _MassLedger(c, used, 0, [-Inf], [zeros(Int, length(sizes))], sizes)
end

# Moves the interventions that answer `population_ledger` out of `interventions`
# into ledgers in `state.scratch`.
function _take_mass_campaigns!(state, interventions)
    state.population isa NoPopulation && return interventions
    ledgers = [population_ledger(iv, state.population) for iv in interventions]
    all(isnothing, ledgers) && return interventions
    state.scratch[_MASS_LEDGERS_KEY] = filter(!isnothing, ledgers)
    return [iv for (iv, l) in zip(interventions, ledgers) if l === nothing]
end

function _without_mass_campaigns(state, ivs)
    haskey(state.scratch, _MASS_LEDGERS_KEY) || return ivs
    return filter(iv -> population_ledger(iv, state.population) === nothing, ivs)
end

# The period of the budget, shared by the budgets per type of a campaign.
_ledger_period(c::_Campaign) = (c.budget isa AbstractVector ? first(c.budget) : c.budget).period

# Gives, period by period, the doses due before `t` over the susceptibles left
# in each group, and dates each period's doses at its start (or the window's).
function _advance!(l::_MassLedger, state, t)
    c = l.campaign
    s, e = c.window
    period = _ledger_period(c)
    if isinf(period)
        l.next == 0 && t > s && _give_period!(l, state, 0.0, Inf, s)
        return nothing
    end
    while l.next * period < min(t, e)
        k = l.next
        _give_period!(l, state, k * period, (k + 1) * period, max(k * period, s))
    end
    return nothing
end

function _give_period!(l::_MassLedger, state, t0, t1, dated)
    c = l.campaign
    l.next += 1
    d = _doses_of(c, c.budget, t0, t1, l.used)
    by_type = d isa Int ? _split_by_size(state.rng, d, l.sizes) : d
    cases = _cases_by_group(state.population, state)
    protected = copy(l.history[end])
    for g in eachindex(by_type)
        susceptibles = max(l.sizes[g] - cases[g] - protected[g], 0)
        protected[g] += protected_by_doses(
            c.dosing, state.rng, by_type[g], c.efficacy, susceptibles, l.sizes[g]
        )
    end
    l.used .+= length(l.used) == 1 ? sum(by_type) : by_type
    push!(l.times, dated)
    push!(l.history, protected)
    return nothing
end

_cases_by_group(::SinglePopulation, state) = [state.cumulative_cases]
_cases_by_group(::GroupedPopulation, state) = state.group_cases

function protected_share(l::_MassLedger, state, group, t)
    _advance!(l, state, t)
    i = searchsortedlast(l.times, t; lt = <=)
    return l.history[i][group] / l.sizes[group]
end

_ledger_group(::SinglePopulation, ind) = 1
_ledger_group(::GroupedPopulation, ind) = individual_type(ind)

# `share` less the share protected at `t` in the group of `contact`.
_after_vaccination(share, ::Nothing, state, contact, t) = share
function _after_vaccination(share, ledgers::Vector, state, contact, t)
    g = _ledger_group(state.population, contact)
    for l in ledgers
        share -= protected_share(l, state, g, t)
    end
    return max(share, 0.0)
end
