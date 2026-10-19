"""
    Reporting(; delay, probability = 1.0, from = :onset_time)

Case reporting: each case is reported with probability `probability`, `delay`
days after symptom onset (or after `from`, if given). Each case records
`:reported` (`true`/`false`) and `:reporting_time` (`Inf` if never reported).

- `delay`: a distribution, a fixed number of days, or a function of the random
  number generator and the individual, `(rng, ind) -> ...`.
- `probability`: the chance a case is reported, a number or a function
  `(rng, ind) -> ...` (for example a different detection probability per risk
  group).
- `from`: the time the delay is measured from. Symptom onset by default;
  another recorded time such as `:test_time` (`:test` works too) or
  `:admission_time`; or a function of the individual, such as
  `ind -> ind.infection_time`.

Cases that never reached `from` are not reported. With the default `from`,
this includes asymptomatic cases from [`clinical_presentation`](@ref), whose
onset time is `NaN`. The time measured from must be one that the model's
`attributes` set or that a step in its `progression` records; the simulation
checks this when it starts, and stops with an error if neither does. A report
drawn for a time after the case has died or recovered is reset to "not
reported".

`simulate`, [`progression_loglik`](@ref), the chain-size likelihood and
[`end_of_outbreak_probability`](@ref) all read this step's reporting
probability. It replaces the deprecated [`PerCaseObservation`](@ref).

# Examples

```julia
# report 80% of symptomatic cases, on average 3 days after onset
Reporting(delay = Gamma(2.0, 1.5), probability = 0.8)
```
"""
Base.@kwdef struct Reporting{D, P, F} <: AbstractClinicalTransition
    delay::D
    probability::P = 1.0
    from::F = :onset_time
end

# Only writes `:reported`/`:reporting_time`; never touches susceptibility or
# infectiousness.
offspring_law_unchanged(::Reporting) = true

required_fields(r::Reporting) = _from_required(r.from)

function initialise_individual!(::Reporting, individual, state)
    individual.state[:reported] = false
    individual.state[:reporting_time] = Inf
    return nothing
end

function resolve_individual!(r::Reporting, individual, state)
    anchor = _state_time(individual, r.from)
    time = transition_time(
        state.rng, individual, anchor, r.delay;
        probability = r.probability
    )
    time === nothing && return nothing
    individual.state[:reported] = true
    individual.state[:reporting_time] = time
    _log_report!(state, individual, time)
    return nothing
end

function censor_after_outcome!(::Reporting, individual)
    _censor_event!(individual, :reported, :reporting_time)
    return nothing
end

function transition_loglik(r::Reporting, individual::Individual)
    anchor = _state_time(individual, r.from)
    _anchor_ok(anchor) || return 0.0
    flag = individual.state[:reported]::Bool
    time = flag ? individual.state[:reporting_time] : Inf
    occurred = flag && time <= outcome_time(individual)
    ll = transition_term(
        r.probability, r.delay, individual, anchor, occurred;
        censor = min(infection_aborted_time(individual), outcome_time(individual))
    )
    occurred || return ll
    return ll + _delay_loglik(r.delay, time - anchor)
end

# The reports `Reporting` has drawn in a run, so that `ReportedCases` can count
# those due without walking the population. The first count builds the log from
# the population, and `Reporting` adds each report it draws from then on. A
# report is drawn before the case's outcome can censor it and before an abort can
# undo it, so each one waits in `pending` until the next count reads the state it
# was written to and keeps it in `times`, sorted, only if it stood. Each pending
# report records the host's id, how many episodes it had archived when the
# report was drawn (which finds that state again after a reinfection has
# archived it) and the report time.
struct _ReportLog{T <: Real}
    pending::Vector{Tuple{Int, Int, T}}
    times::Vector{T}
end

function _log_report!(state::SimulationState{T}, individual, time) where {T}
    # Most runs keep nothing in `scratch`, and a run without a log has no
    # count to keep it for.
    isempty(state.scratch) && return nothing
    log = get(state.scratch, (:report_log,), nothing)
    log === nothing && return nothing
    push!(
        (log::_ReportLog{T}).pending,
        (individual.id, length(individual.episodes), convert(T, time))
    )
    return nothing
end
_log_report!(state, individual, time) = nothing

_report_time(st) =
    get(st, :reported, false) ? get(st, :reporting_time, Inf) : Inf

# Every report the population holds, over each host's archived episodes and its
# live state. A reinfection archives the episode as a copy of the live state and
# only clears the live `:reported` later, when the new infection's transitions
# are initialised (`close_episode!` in `_resolve!`, then
# `resolve_transitions!`). In between, the newest episode and the live state
# are one report, so the live one is kept only when its time differs.
function _report_log(state::SimulationState{T}) where {T}
    times = T[]
    for ind in state.individuals
        for episode in ind.episodes
            t = _report_time(episode.state)
            isfinite(t) && push!(times, t)
        end
        t = _report_time(ind.state)
        isfinite(t) || continue
        isempty(ind.episodes) || t != _report_time(last(ind.episodes).state) ||
            continue
        push!(times, t)
    end
    return _ReportLog{T}(Tuple{Int, Int, T}[], sort!(times))
end

function _confirm_reports!(log::_ReportLog, individuals)
    for (id, k, time) in log.pending
        ind = individuals[id]
        st = length(ind.episodes) == k ? ind.state : ind.episodes[k + 1].state
        isequal(_report_time(st), time) || continue
        insert!(log.times, searchsortedlast(log.times, time) + 1, time)
    end
    empty!(log.pending)
    return log
end
