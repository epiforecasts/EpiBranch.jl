# ── Vaccinating the population on the individual-based race ──────────
#
# A `MassVaccination` under a `CapacityConstrained` gives its doses to the
# population rather than to the people the race creates. The race holds one
# `_MassLedger` per campaign and thins every contact against the susceptibles
# of its group once those the doses protected have been taken out, as
# `simulate_by_generation` does with the counts.

const _MASS_LEDGERS_KEY = :_mass_ledgers

# Whether `iv` is a `MassVaccination` with a limited dose budget, through the
# wrappers a count of doses can read.
_mass_budgeted(::Any, budgeted::Bool = false) = false
_mass_budgeted(::MassVaccination, budgeted::Bool = false) = budgeted
function _mass_budgeted(s::Scheduled, budgeted::Bool = false)
    return s.window isa ClockWindow && _mass_budgeted(s.intervention, budgeted)
end
function _mass_budgeted(cc::CapacityConstrained, budgeted::Bool = false)
    limited = !(cc.budget_per_period isa Real && isinf(cc.budget_per_period))
    return limited && cc.on_refusal isa DropRefused && _mass_budgeted(cc.intervention, true)
end

# The campaign as it stands between two exposures: the doses given so far
# (`used`), the time they were given up to (`clock`) and the people each group
# they protected (`protected`).
mutable struct _MassLedger{C <: _Campaign}
    campaign::C
    used::Vector{Int}
    clock::Float64
    protected::Vector{Int}
    sizes::Vector{Int}
end

_group_sizes(p::SinglePopulation) = [p.size]
_group_sizes(p::GroupedPopulation) = p.sizes

function _mass_ledger(iv, population)
    c = _by_type(
        _campaign(iv, (0.0, Inf), nothing; engine = "a model of individuals"), population
    )
    sizes = _group_sizes(population)
    used = [zeros(Int, c.budget isa AbstractVector ? length(c.budget) : 1)]
    return _MassLedger(c, only(used), 0.0, zeros(Int, length(sizes)), sizes)
end

# Take the campaigns with a dose budget out of `interventions`, which the race
# then does not see, and hold them as ledgers in `state.scratch`. With no
# population there are no people to give the doses to, and each person the
# race creates is dosed in turn.
function _take_mass_campaigns!(state, interventions)
    state.population isa NoPopulation && return interventions
    any(_mass_budgeted, interventions) || return interventions
    state.scratch[_MASS_LEDGERS_KEY] = [
        _mass_ledger(iv, state.population) for iv in interventions if _mass_budgeted(iv)
    ]
    return filter(!_mass_budgeted, interventions)
end

# A continued run holds its ledgers already, and so does not see the campaigns
# they stand for.
_without_mass_campaigns(state, ivs) =
    haskey(state.scratch, _MASS_LEDGERS_KEY) ? filter(!_mass_budgeted, ivs) : ivs

# The doses given up to `t`, spread over the susceptibles left in each group.
function _advance!(l::_MassLedger, state, t)
    t > l.clock || return nothing
    c = l.campaign
    d = _doses_of(c, c.budget, l.clock, t, l.used)
    l.clock = t
    by_type = d isa Int ? _split_by_size(state.rng, d, l.sizes) : d
    cases = _cases_by_group(state.population, state)
    for g in eachindex(by_type)
        susceptibles = max(l.sizes[g] - cases[g] - l.protected[g], 0)
        l.protected[g] += protected_by_doses(
            c.dosing, state.rng, by_type[g], c.efficacy, susceptibles, l.sizes[g]
        )
    end
    l.used .+= length(l.used) == 1 ? sum(by_type) : by_type
    return nothing
end

_cases_by_group(::SinglePopulation, state) = [state.cumulative_cases]
_cases_by_group(::GroupedPopulation, state) = state.group_cases

_ledger_group(::SinglePopulation, ind) = 1
_ledger_group(::GroupedPopulation, ind) = individual_type(ind)

# The susceptible share of `contact`'s group at `t` with the protected taken
# out, from `share` with infection alone.
_after_vaccination(share, ::Nothing, state, contact, t) = share
function _after_vaccination(share, ledgers::Vector, state, contact, t)
    g = _ledger_group(state.population, contact)
    for l in ledgers
        _advance!(l, state, t)
        share -= l.protected[g] / l.sizes[g]
    end
    return max(share, 0.0)
end
