# ── Scoring several signals against one simulated path ───────────────
# A `SignalData`'s series have to be scored against the same simulated
# path: separate `~` statements would each simulate their own outbreak, so
# the model and the observation have to be evaluated together rather than
# left to compose on their own. `loglikelihood(::SignalData,
# ::SimulationState)` is the primitive this comes down to; the model methods
# below simulate once and call it.

"""
    loglikelihood(data::SignalData, state::SimulationState)

Score `data` against one simulated path `state`: for every step, every
series' signal reads a value off `state` over that step's interval, and
`logpdf(series.density(value), series.data[step])` is added.
"""
function loglikelihood(data::SignalData, state::SimulationState)
    total = 0.0
    steps = data.steps
    for series in data.series
        for i in eachindex(series.data)
            interval = (steps[i], steps[i + 1])
            value = series.signal(state, interval)
            total += logpdf(series.density(value), series.data[i])
        end
    end
    return total
end

"""
    loglikelihood(data::SignalData, model::TransmissionModel; rng = Random.default_rng(), kwargs...)
    loglikelihood(data::SignalData, spec::ModelSpec; rng = Random.default_rng(), kwargs...)

Simulate `model` (or `spec`) once and score `data` against that single
simulated path (see `loglikelihood(data::SignalData, state::SimulationState)`).
`kwargs` are passed on to [`simulate`](@ref) (`max_cases`, `max_time`, and so
on).
"""
function loglikelihood(
        data::SignalData, model::TransmissionModel;
        rng::AbstractRNG = Random.default_rng(), kwargs...
    )
    return loglikelihood(data, simulate(model; rng, kwargs...))
end

function loglikelihood(
        data::SignalData, spec::ModelSpec;
        rng::AbstractRNG = Random.default_rng(), kwargs...
    )
    return loglikelihood(data, simulate(spec; rng, kwargs...))
end
