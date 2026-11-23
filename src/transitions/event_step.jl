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

function transition_loglik(
        x::_EventStep, individual::Individual; followup_end = Inf, truncated = false
    )
    flag_key, time_key = _event_keys(x)
    return first(
        _event_loglik(
            x, individual, flag_key, time_key, _event_from_key(x);
            followup_end, truncated
        )
    )
end

# Returns the event log-likelihood and whether the event occurred.
function _event_loglik(
        x, individual::Individual, flag_key, time_key, from_key;
        followup_end, truncated
    )
    anchor = _state_time(individual, x.from, from_key)
    isfinite(anchor) || return (0.0, false)
    flag = individual.state[flag_key]::Bool
    time = flag ? individual.state[time_key] : Inf
    # The outcome never cuts off a terminal step, because that step defines it.
    cap = is_terminal(x) || _timed_from_outcome(individual, x.from, from_key) ?
        Inf : outcome_time(individual)
    cap = min(cap, followup_end)
    occurred = flag && time <= cap
    ll = transition_term(
        x.probability, x.delay, individual, anchor, occurred;
        censor = min(infection_aborted_time(individual), cap)
    )
    occurred || return (ll, false)
    # Cases are in the data because the step happened by `followup_end`, so the
    # density is conditioned on that: the gate cancels, leaving the delay
    # density over the probability the delay is at most `followup_end - anchor`.
    # A terminal step competes with other outcomes, so it is not corrected.
    if truncated && !is_terminal(x)
        delay = _resolved_delay(x.delay, individual)
        return (
            _delay_loglik(delay, time - anchor) -
                _delay_logcdf(delay, followup_end - anchor),
            true,
        )
    end
    return (ll + _delay_loglik(x.delay, time - anchor, individual), true)
end
