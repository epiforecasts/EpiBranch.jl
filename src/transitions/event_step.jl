# Event steps that record whether an event happened and when, as a flag key and
# a time key in the individual's state. `Reporting`, `Hospitalisation` and
# `Transition` differ only in those keys, so one body serves all three.
const _EventStep = Union{Transition, Reporting, Hospitalisation}

_event_keys(::Reporting) = (:reported, :reporting_time)
_event_keys(::Hospitalisation) = (:admitted, :admission_time)
_event_keys(t::Transition) = (t.state, t.time_key)

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
    _censor_event!(individual, flag_key, time_key, x.from)
    return nothing
end

function transition_loglik(x::_EventStep, individual::Individual; followup_end = Inf)
    anchor = _state_time(individual, x.from)
    isfinite(anchor) || return 0.0
    flag_key, time_key = _event_keys(x)
    flag = individual.state[flag_key]::Bool
    time = flag ? individual.state[time_key] : Inf
    # A terminal step defines the outcome, so the outcome does not cut it off.
    cap = is_terminal(x) || _timed_from_outcome(individual, x.from) ?
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
