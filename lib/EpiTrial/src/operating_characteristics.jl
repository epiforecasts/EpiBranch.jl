"""
    trial_estimates(trial, estimators; n_sim = 1000, rng = Random.default_rng(),
        level = 0.95) -> DataFrame

Simulate `n_sim` replicate trials and apply each estimator in `estimators` to
each. Returns one row per replicate and estimator, with columns `replicate`,
`estimator`, `events`, `ve`, `lower`, `upper` and `p_value`. Every estimator is
applied to the same simulated trials, so differences between them are not
Monte Carlo noise from separate runs.
"""
function trial_estimates(trial::Trial, estimators;
        n_sim::Integer = 1000, rng::AbstractRNG = default_rng(), level = 0.95)
    rows = NamedTuple[]
    for r in 1:n_sim
        data = simulate(trial; rng)
        events = count(data.event)
        for est in estimators
            push!(rows,
                (replicate = r, estimator = est, events = events,
                    estimate(est, data; level)...))
        end
    end
    return DataFrame(rows)
end

"""
    operating_characteristics(estimates; null_ve = 0.0, target = nothing) -> DataFrame
    operating_characteristics(trial, estimators; null_ve = 0.0, target = nothing,
        n_sim = 1000, rng = Random.default_rng(), level = 0.95) -> DataFrame

Summarise replicate trials, one row per estimator:

- `power`: proportion of replicates whose lower confidence bound exceeds
  `null_ve` (with `null_ve = 0.0` and a vaccine with no effect this is the
  type I error, one-sided at `(1 - level) / 2`);
- `mean_ve`, `sd_ve`: mean and standard deviation of the point estimates;
- `mean_events`: mean number of endpoint events per trial;
- `failed`: proportion of replicates with no estimate (`NaN`), which count
  against power;
- with `target` given, `bias` (mean estimate minus target) and `coverage`
  (proportion of intervals containing the target). `target` is a number, or a
  function of the estimator for estimators whose targets differ, such as
  `est -> expected_ve(est, LeakyMode(), 0.6, 1.0)`.

The first form summarises the output of [`trial_estimates`](@ref); the second
simulates the replicates first.
"""
function operating_characteristics(estimates::DataFrame; null_ve = 0.0, target = nothing)
    return combine(groupby(estimates, :estimator; sort = false)) do g
        ok = .!isnan.(g.ve)
        summary = (n_sim = nrow(g),
            power = count(g.lower[ok] .> null_ve) / nrow(g),
            mean_ve = mean(g.ve[ok]), sd_ve = std(g.ve[ok]),
            mean_events = mean(g.events), failed = 1 - count(ok) / nrow(g))
        target === nothing && return summary
        t = _target(target, first(g.estimator))
        return (; summary..., bias = mean(g.ve[ok]) - t,
            coverage = mean(g.lower[ok] .<= t .<= g.upper[ok]))
    end
end

function operating_characteristics(trial::Trial, estimators;
        null_ve = 0.0, target = nothing, kwargs...)
    return operating_characteristics(trial_estimates(trial, estimators; kwargs...);
        null_ve, target)
end

_target(t::Real, est) = t
_target(f, est) = f(est)
