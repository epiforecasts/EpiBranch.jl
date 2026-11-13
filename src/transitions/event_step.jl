# Event steps record whether an event happened and when, as a flag key and a
# time key in the state of the individual. `Reporting`, `Hospitalisation` and
# `Transition` differ only in those keys and share one body.
const _EventStep = Union{Transition, Reporting, Hospitalisation}

_event_keys(::Reporting) = (:reported, :reporting_time)
_event_keys(::Hospitalisation) = (:admitted, :admission_time)
_event_keys(t::Transition) = (t.state, t.time_key)

# Only `Transition` holds a resolved `from` key; the others resolve it on demand.
_event_from_key(t::Transition) = t.from_key
_event_from_key(::Union{Reporting, Hospitalisation}) = nothing

# These steps, `Recovery` and `Death` only write their own event state; they
# never touch susceptibility or infectiousness.
offspring_law_unchanged(::Union{_EventStep, Recovery, Death}) = true

function initialise_individual!(x::_EventStep, individual, state)
    flag_key, time_key = _event_keys(x)
    individual.state[flag_key] = false
    individual.state[time_key] = Inf
    return nothing
end

function censor_after_outcome!(x::_EventStep, individual)
    is_terminal(x) && return nothing
    flag_key, time_key = _event_keys(x)
    _censor_event!(individual, flag_key, time_key, x.from, _event_from_key(x))
    return nothing
end

function transition_loglik(x::_EventStep, individual::Individual; followup_end = Inf)
    anchor = _state_time(individual, x.from, _event_from_key(x))
    isfinite(anchor) || return 0.0
    flag_key, time_key = _event_keys(x)
    flag = individual.state[flag_key]::Bool
    time = flag ? individual.state[time_key] : Inf
    # The outcome never cuts off a terminal step, because that step defines it.
    cap = is_terminal(x) || _timed_from_outcome(individual, x.from, _event_from_key(x)) ?
        Inf : outcome_time(individual)
    cap = min(cap, followup_end)
    occurred = flag && time <= cap
    ll = transition_term(
        x.probability, x.delay, individual, anchor, occurred;
        censor = min(infection_aborted_time(individual), cap)
    )
    occurred || return ll
    return ll + _delay_loglik(x.delay, time - anchor)
end
