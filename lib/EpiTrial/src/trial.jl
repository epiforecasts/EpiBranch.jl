"""
Base type for a trial endpoint: which event ends a participant's follow-up as
a case. A subtype implements `event_time(endpoint, ind)`, the time of the
endpoint event for an individual, or `Inf` if it never occurs.
"""
abstract type AbstractEndpoint end

"""
    Infection()

Endpoint at infection, as ascertained by perfect testing (for example
frequent serology or PCR).
"""
struct Infection <: AbstractEndpoint end

event_time(::Infection, ind) = is_infected(ind) ? ind.infection_time : Inf

"""
    Trial(spec; follow_up, endpoint = Infection())

A trial run within the outbreak that `spec` generates. Everyone in the
simulated population who is uninfected at enrolment (time 0) is enrolled and
followed until their endpoint event or until `follow_up`, whichever is first.
`spec` must carry an arm on every individual (see
[`IndividualRandomisation`](@ref)) and, for a trial of a vaccine, a
[`TrialVaccination`](@ref).

Enrolling the whole simulated population suits the continuous-time models,
where every individual exists from the start (`HomogeneousProcess`,
households, networks). On a branching process only individuals who were
exposed exist, so there is no denominator of unexposed participants.

[`simulate`](@ref EpiBranch.simulate) on a `Trial` returns its
[`trial_data`](@ref).
"""
struct Trial{S <: ModelSpec, E <: AbstractEndpoint}
    spec::S
    follow_up::Float64
    endpoint::E
    function Trial(spec::S, follow_up::Real, endpoint::E) where {S, E}
        follow_up > 0 || throw(ArgumentError("follow_up must be positive"))
        return new{S, E}(spec, Float64(follow_up), endpoint)
    end
end

Trial(spec::ModelSpec; follow_up, endpoint = Infection()) = Trial(spec, follow_up, endpoint)

"""
    trial_data(trial, state) -> DataFrame

One row per enrolled participant of the outbreak `state` (from `simulate` on
`trial.spec`), with columns

- `id`: the individual's id in `state`;
- `arm`: `:vaccine` or `:control`;
- `exit_time`: time of the endpoint event or of censoring at `follow_up`;
- `event`: whether the participant reached the endpoint during follow-up.

Everyone infected at or before enrolment (time 0) is excluded.
"""
function trial_data(trial::Trial, state)
    enrolled = [ind
                for ind in state.individuals
                if !(is_infected(ind) && ind.infection_time <= 0)]
    t_event = [event_time(trial.endpoint, ind) for ind in enrolled]
    return DataFrame(
        id = [ind.id for ind in enrolled],
        arm = [arm(ind) for ind in enrolled],
        exit_time = min.(t_event, trial.follow_up),
        event = t_event .<= trial.follow_up)
end

"""
    simulate(trial::Trial; rng = Random.default_rng(), sim_kwargs...) -> DataFrame

Simulate one outbreak from `trial.spec` and return the [`trial_data`](@ref)
of the trial run within it. Other keywords (`n_initial`, `initial_cases`, …)
are passed to `simulate` on `trial.spec`, for example to seed the outbreak with
several cases so it rarely dies out before the trial accrues events. A
`max_time` or `MaxTime` stopping rule must cover the follow-up, since nobody is
infected once the simulation stops.
"""
function EpiBranch.simulate(trial::Trial; rng::AbstractRNG = default_rng(), sim_kwargs...)
    # The run's end as the engine computes it, from `max_time` and any
    # `MaxTime` among the stopping rules.
    opts = SimOpts(; max_time = get(sim_kwargs, :max_time, nothing),
        stopping_rules = get(sim_kwargs, :stopping_rules, nothing))
    _max_time(opts) >= trial.follow_up || throw(ArgumentError(
        "the simulation stops at time $(_max_time(opts)), before the end of " *
        "follow-up ($(trial.follow_up))"))
    return trial_data(trial, simulate(trial.spec; rng, sim_kwargs...))
end
