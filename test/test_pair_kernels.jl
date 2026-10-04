using ForwardDiff
using DifferentiationInterface: DifferentiationInterface
using ADTypes: AutoMooncake
import Mooncake

struct ContextRule{T}
    slope::T
end
function (rule::ContextRule)(context::PairContext)
    return Exponential(exp(rule.slope * context.infector_infection_time))
end

struct ContextInfections{T} <: InfectionLayer
    infection_time::Vector{T}
    infectious_time::Vector{T}
    removal_time::Vector{T}
    is_index::Vector{Bool}
    obs_end::Float64
end
EpiBranch.contact_structure(::ContextInfections) = [[2], [1]]
function context_data(t)
    return ContextInfections(
        [t, oftype(t, 4)], [t + 1, oftype(t, 5)],
        [t + 6, oftype(t, 10)], [true, false], 0.0
    )
end

@testset "Contextual pair kernels" begin
    distribution = Exponential(2.0)
    @test EpiBranch.pair_kernel(distribution, 1, 2, 3.0) === distribution
    @test mean(EpiBranch.pair_kernel((i, j) -> Exponential(i + j), 1, 2, NaN)) == 3.0
    rule = PairKernel(ContextRule(0.2))
    @test mean(EpiBranch.pair_kernel(rule, 1, 2, 3.0)) ≈ exp(0.6)
    @test isbitstype(typeof(PairContext(1, 2, 3.0)))
    rows = PairwiseSurvivalData([2], [0.0], [1.0], [true])
    @test_throws ArgumentError pairwise_surv_loglik(rule, rows)

    covariates = [0.5, 1.5]
    data = context_data(2.0)
    layout = compile_contact_pairs(data)
    kernel(a, b) = PairKernel(
        context -> Exponential(
            exp(
                a + b * context.infector_infection_time +
                    0.1 * covariates[context.susceptible]
            )
        )
    )
    expected(a, b, t) = -(a + b * t + 0.15) - (3 - t) * exp(-(a + b * t + 0.15))
    for t in (1.8, 2.0, 2.2)
        current = context_data(t)
        @test pairwise_surv_loglik(kernel(0.3, 0.2), current, layout) ≈
            expected(0.3, 0.2, t)
        @test pairwise_surv_loglik(kernel(0.3, 0.2), current) ≈ expected(0.3, 0.2, t)
    end
    f(x) = pairwise_surv_loglik(kernel(x[1], x[2]), context_data(x[3]), layout)
    reference(x) = expected(x[1], x[2], x[3])
    x = [0.3, 0.2, 2.0]
    @test ForwardDiff.gradient(f, x) ≈ ForwardDiff.gradient(reference, x)
    @test DifferentiationInterface.gradient(f, AutoMooncake(), x) ≈
        ForwardDiff.gradient(reference, x)
end

struct CalendarInfections{T} <: InfectionLayer
    infection_time::Vector{T}
    infectious_time::Vector{T}
    removal_time::Vector{T}
    is_index::Vector{Bool}
    obs_end::Float64
end
EpiBranch.contact_structure(::CalendarInfections) = [[2], [1]]
function calendar_data(t, opening)
    return CalendarInfections(
        [t, oftype(t, 4)],
        [opening, oftype(opening, 5)], [oftype(t, 6), oftype(t, 7)], [true, false], 0.0
    )
end

@testset "Calendar-time pair kernels" begin
    before, after, date = 0.4, 0.1, 3.0
    kernel = PairKernel(ctx -> Exponential(1.0); calendar = Steps([date], [before, after]))
    integrated(t) = before * min(t, date) + after * max(t - date, 0.0)
    for opening in (1.0, 3.5)
        interval = EpiBranch.pair_kernel(kernel, 1, 2, 0.0, opening)
        for d in (opening + 0.2, opening + 2.1)
            @test EpiBranch.cumhazard(interval, d - opening) ≈
                integrated(d) - integrated(opening)
            @test exp(EpiBranch.loghazard(interval, d - opening)) ≈
                (d < date ? before : after)
        end
    end
    interval = EpiBranch.pair_kernel(kernel, 1, 2, 0.0, 1.0)
    rng = StableRNG(233)
    draws = [rand(rng, interval) for _ in 1:10000]
    @test count(<=(4.0), draws) / length(draws) ≈
        1 - exp(-EpiBranch.cumhazard(interval, 4.0)) atol = 0.02
    @test_throws ArgumentError EpiBranch.pair_kernel(kernel, 1, 2, 0.0)
    @test_throws ArgumentError Steps([1.0, 0.5], [1.0, 1.0, 1.0])
    @test_throws ArgumentError Steps([1.0], [1.0, -0.5])
    @test_throws ArgumentError Steps([1.0], [1.0])
    rows = PairwiseSurvivalData([2], [0.0], [1.0], [true])
    @test_throws ArgumentError pairwise_surv_loglik(kernel, rows)

    data = calendar_data(1.0, 2.0)
    layout = compile_contact_pairs(data)
    @test pairwise_surv_loglik(kernel, data, layout) ≈ log(0.1) - 0.5
    @test pairwise_surv_loglik(kernel, calendar_data(1.0, 3.5), layout) ≈ log(0.1) - 0.05
    @test pairwise_surv_loglik(kernel, calendar_data(1.0, Inf), layout) == -Inf
    @test pairwise_surv_loglik(kernel, calendar_data(1.0, NaN), layout) == -Inf

    # A single step's rate and its breakpoint are both differentiable, since the
    # cumulative hazard splits into a scaled difference of the profile's own.
    f(x) = pairwise_surv_loglik(
        PairKernel(ctx -> Exponential(1.0); calendar = Steps([x[3]], [x[1], x[2]])),
        data, layout
    )
    reference(x) = begin
        integ(t) = x[1] * min(t, x[3]) + x[2] * max(t - x[3], 0.0)
        log(x[2]) - (integ(4.0) - integ(2.0))
    end
    x = [before, after, date]
    @test f(x) ≈ reference(x)
    @test ForwardDiff.gradient(f, x) ≈ ForwardDiff.gradient(reference, x)
    @test DifferentiationInterface.gradient(f, AutoMooncake(), x) ≈
        ForwardDiff.gradient(reference, x)

    # A pair's own calendar, returned from the callback, evaluates the same as the
    # kernel-level schedule.
    contextual = PairKernel(c -> (profile = Exponential(1.0), calendar = Steps([date], [before, after])))
    @test pairwise_surv_loglik(contextual, data, layout) ≈
        pairwise_surv_loglik(kernel, data, layout)
    # A named tuple without a calendar keeps the kernel's own schedule.
    profile_only = PairKernel(
        c -> (profile = Exponential(1.0),);
        calendar = Steps([date], [before, after])
    )
    @test pairwise_surv_loglik(profile_only, data, layout) ==
        pairwise_surv_loglik(kernel, data, layout)
    @test EpiBranch.pair_kernel(PairKernel(c -> (profile = Exponential(2.0),)), 1, 2, 0.0) ==
        Exponential(2.0)
end

include("testutils/pair_kernels.jl")

# A multiplier that is zero until calendar day 3 and then rises linearly, so
# quadrature meets exact zeros and a kink.
struct RampSchedule end
EpiBranch.calendar_multiplier(::RampSchedule, t) = max(zero(t), t - 3)
EpiBranch.calendar_shape(::RampSchedule) = EpiBranch.SmoothCalendar()

# A multiplier whose integral over all time is finite.
struct DecayingSchedule end
EpiBranch.calendar_multiplier(::DecayingSchedule, t) = exp(-t)
EpiBranch.calendar_shape(::DecayingSchedule) = EpiBranch.SmoothCalendar()

struct BreaklessSchedule end
EpiBranch.calendar_multiplier(::BreaklessSchedule, t) = 1.0

@testset "Smooth calendar schedules" begin
    seasonal = SeasonalSchedule(0.5, 0.8, 10.0)
    @test EpiBranch.calendar_shape(seasonal) === EpiBranch.SmoothCalendar()
    @test EpiBranch.calendar_shape(Steps([1.0], [1.0, 2.0])) ===
        EpiBranch.PiecewiseConstantCalendar()
    @test EpiBranch.calendar_multiplier(Steps([1.0], [1.0, 2.0]), 1.0) == 2.0
    @test EpiBranch.next_calendar_break(Steps([1.0, 2.0], [1.0, 2.0, 3.0]), 1.0) == 2.0
    @test EpiBranch.next_calendar_break(Steps([1.0], [1.0, 2.0]), 1.0) == Inf
    # A piecewise-constant schedule must say where it breaks.
    breakless = PairKernel(ctx -> Exponential(1.0); calendar = BreaklessSchedule())
    @test_throws MethodError EpiBranch.cumhazard(
        EpiBranch.pair_kernel(breakless, 1, 2, 0.0, 1.0), 1.0
    )

    kernel = PairKernel(ctx -> Exponential(2.0); calendar = seasonal)
    profile_rate = 0.5
    integrated(o, τ) = profile_rate *
        (seasonal_integral(seasonal, o + τ) - seasonal_integral(seasonal, o))
    for opening in (0.0, 2.0, 7.5)
        interval = EpiBranch.pair_kernel(kernel, 1, 2, 0.0, opening)
        @test EpiBranch.cumhazard(interval, 0.0) == 0
        for τ in (0.3, 4.0, 17.0)
            @test EpiBranch.cumhazard(interval, τ) ≈ integrated(opening, τ)
            @test exp(EpiBranch.loghazard(interval, τ)) ≈
                profile_rate * EpiBranch.calendar_multiplier(seasonal, opening + τ)
        end
        for lp in (-0.01, -1.0, -4.0)
            @test logccdf(interval, invlogccdf(interval, lp)) ≈ lp
        end
    end
    interval = EpiBranch.pair_kernel(kernel, 1, 2, 0.0, 2.0)
    @test EpiBranch.cumhazard(interval, Inf) == Inf
    @test invlogccdf(interval, -Inf) == Inf
    @test invlogccdf(interval, 0.0) == 0
    @test_throws ArgumentError EpiBranch.cumhazard(interval, -1.0)
    @test_throws ArgumentError invlogccdf(interval, 0.5)
    @test minimum(interval) == 0
    @test Distributions.partype(interval) == Float64

    # Simulated contact intervals follow the integrated hazard: the largest gap
    # between their empirical and exact distribution functions is within a
    # Kolmogorov–Smirnov bound.
    rng = StableRNG(233)
    n = 4000
    draws = sort([rand(rng, interval) for _ in 1:n])
    exact = [1 - exp(-integrated(2.0, t)) for t in draws]
    @test maximum(abs.(exact .- (1:n) ./ n)) < 1.63 / sqrt(n)

    # The ramp is zero before its kink, so nothing happens there.
    ramp = EpiBranch.pair_kernel(
        PairKernel(ctx -> Exponential(1.0); calendar = RampSchedule()), 1, 2, 0.0, 1.0
    )
    @test EpiBranch.cumhazard(ramp, 2.0) == 0
    @test EpiBranch.cumhazard(ramp, 4.0) ≈ 2.0
    @test invlogccdf(ramp, -2.0) ≈ 4.0
    @test exp(EpiBranch.loghazard(ramp, 1.0)) == 0

    # A survival that never falls below the target has no contact time.
    decaying = EpiBranch.pair_kernel(
        PairKernel(ctx -> Exponential(1.0); calendar = DecayingSchedule()), 1, 2, 0.0, 0.0
    )
    @test EpiBranch.cumhazard(decaying, 30.0) ≈ 1 - exp(-30.0)
    @test invlogccdf(decaying, -0.5) ≈ log(2)
    @test invlogccdf(decaying, -2.0) == Inf

    # A bounded profile has no survival past its support, and every contact
    # falls inside it.
    bounded = EpiBranch.pair_kernel(
        PairKernel(ctx -> Uniform(0.0, 3.0); calendar = seasonal), 1, 2, 0.0, 2.0
    )
    @test EpiBranch.cumhazard(bounded, 3.0) == Inf
    flat = EpiBranch.pair_kernel(
        PairKernel(ctx -> Uniform(0.0, 3.0); calendar = SeasonalSchedule(1.0, 0.0, 10.0)),
        1, 2, 0.0, 2.0
    )
    @test EpiBranch.cumhazard(flat, 1.0) ≈ -logccdf(Uniform(0.0, 3.0), 1.0)
    @test 0 < invlogccdf(bounded, -50.0) <= 3.0
    @test 0 < invlogccdf(bounded, -0.5) < 3.0

    # The schedule's parameters are differentiable through the quadrature.
    data = calendar_data(1.0, 2.0)
    layout = compile_contact_pairs(data)
    f(x) = pairwise_surv_loglik(
        PairKernel(ctx -> Exponential(1.0); calendar = SeasonalSchedule(x[1], x[2], 10.0)),
        data, layout
    )
    reference(x) = begin
        s = SeasonalSchedule(x[1], x[2], 10.0)
        log(EpiBranch.calendar_multiplier(s, 4.0)) -
            (seasonal_integral(s, 4.0) - seasonal_integral(s, 2.0))
    end
    x = [0.5, 0.8]
    @test f(x) ≈ reference(x)
    @test ForwardDiff.gradient(f, x) ≈ ForwardDiff.gradient(reference, x)

    # A drawn interval is differentiable in the schedule's parameters too:
    # with Λ(t) = target, dt/dθ = -(∂Λ/∂θ) / λ(t).
    opening = 2.0
    target = 0.7
    draw(x) = invlogccdf(
        EpiBranch.pair_kernel(
            PairKernel(ctx -> Exponential(1.0); calendar = SeasonalSchedule(x[1], x[2], 10.0)),
            1, 2, 0.0, opening
        ),
        -target
    )
    t = draw(x)
    s = SeasonalSchedule(x[1], x[2], 10.0)
    @test seasonal_integral(s, opening + t) - seasonal_integral(s, opening) ≈ target
    implicit = -ForwardDiff.gradient(
        y -> begin
            sy = SeasonalSchedule(y[1], y[2], 10.0)
            seasonal_integral(sy, opening + t) - seasonal_integral(sy, opening)
        end, x
    ) ./ EpiBranch.calendar_multiplier(s, opening + t)
    @test ForwardDiff.gradient(draw, x) ≈ implicit rtol = 1.0e-6

    # Near a trough of the schedule the hazard is tiny. The draw must still
    # solve its own cumulative hazard, so the step cannot move it by more than
    # the bisection's last interval.
    deep(y) = PairKernel(
        ctx -> Exponential(1.0); calendar = SeasonalSchedule(y[1], y[2], 10.0)
    )
    y = [1.0, 1.0 - 1.0e-9]
    trough = seasonal_integral(SeasonalSchedule(y[1], y[2], 10.0), 7.5) -
        seasonal_integral(SeasonalSchedule(y[1], y[2], 10.0), 0.0)
    near = EpiBranch.pair_kernel(deep(y), 1, 2, 0.0, 0.0)
    t_near = invlogccdf(near, -trough)
    @test EpiBranch.cumhazard(near, t_near) ≈ trough rtol = 1.0e-8
    drawn(y) = invlogccdf(EpiBranch.pair_kernel(deep(y), 1, 2, 0.0, 0.0), -trough)
    @test drawn(y) == t_near
    @test all(isfinite, ForwardDiff.gradient(drawn, y))
end

struct StateKernelInfections{T} <: InfectionLayer
    infection_time::Vector{T}
    infectious_time::Vector{T}
    removal_time::Vector{T}
    is_index::Vector{Bool}
    obs_end::Float64
end
EpiBranch.contact_structure(::StateKernelInfections) = [[2], [1]]

@testset "Recorded pair state" begin
    data = StateKernelInfections([0.0, 3.0], [1.0, 4.0], [5.0, 6.0], [true, false], 0.0)
    layout = compile_contact_pairs(data)
    callback(c, a, b) = Exponential(
        exp(
            a.log_scale + b.log_scale +
                c.infector_infection_time
        )
    )
    records = [(log_scale = 0.1,), (log_scale = 0.3,)]
    kernel = PairKernel(callback; state = records)
    @test mean(EpiBranch.pair_kernel(kernel, 1, 2, 0.0)) ≈ exp(0.4)
    @test pairwise_surv_loglik(kernel, data, layout) ≈ -0.4 - 2exp(-0.4)
    live = PairKernel(callback; state = ind -> (log_scale = ind.state[:log_scale]::Float64,), watches = (:log_scale,))
    @test_throws ArgumentError pairwise_surv_loglik(live, data, layout)
    @test_throws ArgumentError pairwise_surv_loglik(
        kernel,
        PairwiseSurvivalData([2], [0.0], [1.0], [true])
    )
    f(x) = pairwise_surv_loglik(
        PairKernel(
            callback; state = [
                (log_scale = x[1],),
                (log_scale = x[2],),
            ]
        ), data, layout
    )
    reference(x) = -sum(x) - 2exp(-sum(x))
    x = [0.1, 0.3]
    @test ForwardDiff.gradient(f, x) ≈ ForwardDiff.gradient(reference, x)
    @test DifferentiationInterface.gradient(f, AutoMooncake(), x) ≈
        ForwardDiff.gradient(reference, x)

    state = EpiBranch.new_state(BranchingProcess(Poisson(0.0)), [], NoAttributes(), StableRNG(233))
    EpiBranch.add_individuals!(state, 2, [])
    for (ind, record) in zip(state.individuals, records)
        ind.state[:log_scale] = record.log_scale
        ind.state[:history] = [1.0]
    end
    saved = record_kernel(live, state)
    @test saved.state == records
    @test pairwise_surv_loglik(saved, data, layout) ≈ reference(x)
    @test record_kernel(Exponential(), state) == Exponential()
    history = record_kernel(PairKernel(callback; state = ind -> ind.state[:history], watches = (:history,)), state)
    push!(state.individuals[1].state[:history], 2.0)
    @test history.state[1] == [1.0]
    @test !EpiBranch._live_kernel(saved)
    # Extracted records cannot move, so the recorded kernel watches nothing and
    # a race given it keeps the ordinary path.
    @test EpiBranch.watched_records(saved) == ()
    @test EpiBranch.watched_records(history) == ()

    # A calendar schedule is carried through recording unchanged, and a kernel
    # with one is live exactly when its host state is.
    calendar_live = PairKernel(
        (c, a, b) -> Exponential(2.0);
        state = ind -> (tag = 0.0,), calendar = Steps([5.0], [1.0, 0.5]), watches = ()
    )
    @test EpiBranch._live_kernel(calendar_live)
    recorded_calendar = record_kernel(calendar_live, state)
    @test recorded_calendar.calendar === calendar_live.calendar
    @test !EpiBranch._live_kernel(recorded_calendar)
end

@testset "Differentiable recorded event dates" begin
    data = StateKernelInfections([0.0, 3.0], [1.0, 4.0], [5.0, 6.0], [true, false], 0.0)
    layout = compile_contact_pairs(data)
    f = function (x)
        records = [(date = x[3],), (date = x[3],)]
        callback = function (c, a, b)
            (profile = Exponential(1.0), calendar = Steps([b.date], [x[1], x[2]]))
        end
        pairwise_surv_loglik(PairKernel(callback; state = records), data, layout)
    end
    reference(x) = log(x[2]) - x[1] * (x[3] - 1) - x[2] * (3 - x[3])
    x = [0.4, 0.1, 2.0]
    @test f(x) ≈ reference(x)
    @test ForwardDiff.gradient(f, x) ≈ ForwardDiff.gradient(reference, x)
    @test DifferentiationInterface.gradient(f, AutoMooncake(), x) ≈
        ForwardDiff.gradient(reference, x)
end

# Exercise the shared primitive without depending on a companion package.
function stateful_test_race(
        kernel, initial_times; interventions = (), introduction = nothing,
        seed = 233
    )
    rng = StableRNG(seed)
    progression = [Transition(:recovered; delay = 5.0, terminal = true)]
    state = EpiBranch.new_state(
        BranchingProcess(Poisson(0.0)), progression,
        NoAttributes(), rng
    )
    n = length(initial_times)
    EpiBranch.add_individuals!(state, n, interventions)
    targets = (i, st) -> (
        (
            j,
            EpiBranch.pair_kernel(
                kernel, i, j,
                st.individuals[i].infection_time, st.individuals[i].infection_time, st
            ),
        )
            for j in 1:n if j != i && !is_infected(st.individuals[j])
    )
    EpiBranch._sellke_race!(
        state, collect(1:n), rng;
        seed! = (best, members, r) -> copyto!(best, initial_times),
        targets, from = :infection, until = (:recovered,), interventions,
        introduction, watches = (EpiBranch.watched_records(kernel),)
    )
    return state
end

@testset "An unchanging live kernel leaves the race stream alone" begin
    seeds = [0.0; fill(Inf, 11)]
    for d in (Exponential(1.5), Weibull(2.0, 2.0), Gamma(3.0, 0.7))
        live = PairKernel((c, a, b) -> d; state = ind -> (tag = get(ind.state, :tag, 0.0)::Float64,), watches = (:tag,))
        @test isequal(
            [i.infection_time for i in stateful_test_race(d, seeds).individuals],
            [i.infection_time for i in stateful_test_race(live, seeds).individuals]
        )
    end
end

@testset "A mutable history is seen to change" begin
    state = EpiBranch.new_state(
        BranchingProcess(Poisson(0.0)), [], NoAttributes(),
        StableRNG(233)
    )
    EpiBranch.add_individuals!(state, 3, [])
    for ind in state.individuals
        ind.state[:history] = Float64[]
    end
    members = [1, 2, 3]
    watched_keys = [:history]
    key_routes = [[1]]            # the one route reads `:history`
    snapshot = Any[
        EpiBranch._remember(EpiBranch._watched_value(state.individuals[i], key))
            for key in watched_keys, i in members
    ]
    # Case 1 has an open opening that reaches member 2; member 3 is out of reach.
    openings = [
        EpiBranch._RouteOpening(0, 0, 0.0, Inf),
        EpiBranch._RouteOpening(1, 1, 0.0, 5.0),
    ]
    watch = EpiBranch._LiveWatch(3, 1)
    EpiBranch._watch_opening!(watch, 1, true)
    EpiBranch._watch_target!(watch, 2, 2)
    processed = [true, false, false]
    changed!(case, now = 1.0) = EpiBranch._records_changed!(
        snapshot, watched_keys, key_routes, state,
        members, case, now, watch, openings, processed
    )
    @test !changed!(1)
    # An intervention appending in place must not compare equal to its own
    # remembered record, which is why the race keeps a copy rather than an alias.
    push!(state.individuals[2].state[:history], 1.0)
    @test changed!(1)
    @test !changed!(1)
    # No pending or future draw reads a member out of reach, and the settled
    # case's own record is only brought up to date.
    push!(state.individuals[3].state[:history], 1.0)
    @test !changed!(1)
    push!(state.individuals[1].state[:history], 1.0)
    @test !changed!(1)
    @test snapshot[1, 1] == [1.0]
    # Member 2 is compared once however many open openings reach it, and leaves
    # the watch once they have all closed.
    push!(openings, EpiBranch._RouteOpening(1, 1, 0.0, 8.0))
    EpiBranch._watch_opening!(watch, 1, true)
    EpiBranch._watch_target!(watch, 3, 2)
    @test sort(watch.tracked) == [1, 2]
    @test !changed!(1, 6.0)
    @test watch.open == [3]
    @test !changed!(1, 9.0)
    @test isempty(watch.tracked)
    push!(state.individuals[2].state[:history], 2.0)
    @test !changed!(1, 9.0)
end

@testset "A layer host reads recorded times like an individual" begin
    columns = (
        onset_time = [2.0], trace_time = Union{Missing, Float64}[missing],
        isolation_time = [NaN],
    )
    host = EpiBranch._LayerHostState(columns, 1)
    @test host[:onset_time] == 2.0
    @test get(host, :onset_time, Inf) == 2.0
    # A missing entry is a time this host does not have.
    @test get(host, :trace_time, Inf) == Inf
    @test_throws KeyError host[:trace_time]
    @test haskey(host, :onset_time) && !haskey(host, :trace_time)
    # A NaN entry is a recorded value, as for an individual.
    @test isnan(host[:isolation_time]) && isnan(get(host, :isolation_time, Inf))
    @test haskey(host, :isolation_time)
    # A key the layer never recorded is an error, whatever the default.
    @test_throws ArgumentError get(host, :vaccination_time, Inf)
    # A column holding `missing` reads with a concrete type.
    read_trace(h) = get(h, :trace_time, Inf)
    @test @inferred(read_trace(host)) == Inf
end

@testset "Compaction keeps each member's live proposals" begin
    P = EpiBranch._Pending{Float64}
    # Member 1 has settled; member 2 lists proposals 3 then 1, with 2 unlinked by
    # a redraw; member 3 lists proposal 4.
    proposals = [
        P(2, 0, 1.5, true), P(3, 0, 0.5, true), P(4, 1, 2.5, false),
        P(5, 0, 3.0, true), P(6, 0, 0.7, true),
    ]
    head = [5, 3, 4]
    best = [0.7, 1.5, 3.0]
    represents = [5, 1, 4]
    pending = Tuple{Float64, Int, Int}[(0.5, 2, 2), (1.5, 2, 1), (3.0, 3, 4), (0.7, 1, 5)]
    EpiBranch._compact_proposals!(
        pending, proposals, head, best, represents,
        [true, false, false]
    )
    @test length(proposals) == 3
    @test head[1] == 0 && represents[1] == 0
    chain = Float64[]
    q = head[2]
    while q != 0
        push!(chain, proposals[q].time)
        q = proposals[q].chain
    end
    @test chain == [2.5, 1.5]
    @test proposals[represents[2]].time == 1.5 && proposals[represents[2]].queued
    @test proposals[represents[3]].time == 3.0
    @test count(p -> p.queued, proposals) == 2
    @test sort(pending) == [(1.5, 2, represents[2]), (3.0, 3, represents[3])]
end

@testset "Shared race refreshes live pair kernels" begin
    ties = PairKernel((c, a, b) -> Dirac(1.0); state = tick_state, watches = (:tick,))
    state = stateful_test_race(ties, [0.0, Inf, Inf]; interventions = [TickEveryCase()])
    @test [i.infection_time for i in state.individuals] == [0.0, 1.0, 1.0]

    project(ind) = (date = get(ind.state, :policy_time, Inf)::Float64,)
    callback = function (c, a, b)
        (c.infector, c.susceptible) == (1, 2) && return Dirac(1.0)
        (c.infector, c.susceptible) == (1, 3) &&
            return state_policy_law(0.1, 1.0, b.date)
        return Dirac(20.0)
    end
    kernel = PairKernel(callback; state = project, watches = (:policy_time,))
    changed = stateful_test_race(
        kernel, [0.0, Inf, Inf];
        interventions = [RecordKernelPolicy()]
    )
    @test changed.individuals[2].infection_time == 1.0
    @test changed.individuals[3].state[:policy_time] == 1.5
    recorded = record_kernel(kernel, changed)
    @test logccdf(EpiBranch.pair_kernel(kernel, 1, 3, 0.0, 0.0, changed), 2.0) ≈
        logccdf(EpiBranch.pair_kernel(recorded, 1, 3, 0.0), 2.0)
    @test mean(EpiBranch.pair_kernel(Exponential(2.0), 1, 3, 0.0, 0.5, changed)) == 2.0
    @test logccdf(EpiBranch.pair_kernel(recorded, 1, 3, 0.0, 0.0, changed), 2.0) ≈
        logccdf(EpiBranch.pair_kernel(recorded, 1, 3, 0.0), 2.0)
    fixed = PairKernel((c, a, b) -> Exponential(1.0); state = [nothing, nothing])
    replay = stateful_test_race(fixed, [0.0, Inf])
    ordinary = stateful_test_race(Exponential(1.0), [0.0, Inf])
    @test isequal(
        [i.infection_time for i in replay.individuals],
        [i.infection_time for i in ordinary.individuals]
    )

    # Retried introductions must remain later than the admission boundary even
    # when another introduction settles and refreshes the remaining queue.
    inactive = PairKernel((c, a, b) -> Dirac(20.0); state = tick_state, watches = (:tick,))
    introduced = stateful_test_race(
        inactive, [0.1, 0.2, 0.3];
        interventions = [WaitForKernelDay(), TickEveryCase()],
        introduction = (Exponential(0.2), 3.0)
    )
    cases = filter(is_infected, introduced.individuals)
    @test length(cases) == 3
    @test all(i -> 1.0 <= i.infection_time <= 3.0, cases)
end

@testset "Moved records redraw pending contacts in a shared race" begin
    # Host 2 comes before host 3, so when host 3 settles at t = 1 every contact
    # to host 2 at t = 1 has been resolved; the redraw must not offer it again.
    two_atoms = DiscreteNonParametric([1.0, 2.0], [0.5, 0.5])
    atoms = PairKernel(
        (c, a, b) -> c.susceptible == 2 ? two_atoms : Dirac(1.0);
        state = tick_state, watches = (:tick,)
    )
    at_one = count(1:2000) do seed
        state = stateful_test_race(
            atoms, [0.0, Inf, Inf];
            interventions = [TickEveryCase()], seed
        )
        state.individuals[2].infection_time == 1.0
    end / 2000
    @test isapprox(at_one, 0.5; atol = 0.04)

    # Every case moves every record, so a large race redraws, and compacts its
    # proposals, many times over; the hazards never change, so it spreads as an
    # ordinary kernel does.
    seeds = [0.0; fill(Inf, 29)]
    law = Exponential(4.0)
    live = PairKernel((c, a, b) -> law; state = tick_state, watches = (:tick,))
    early(state) = count(ind -> ind.infection_time < 1.0, state.individuals)
    redrawn = mean(
        early(
            stateful_test_race(
                live, seeds;
                interventions = [TickEveryCase()], seed
            )
        ) for seed in 1:1000
    )
    ordinary = mean(early(stateful_test_race(law, seeds; seed)) for seed in 1:1000)
    @test isapprox(redrawn, ordinary; atol = 0.6)
end

@testset "Live kernels outside a race" begin
    live = PairKernel((c, a, b) -> Exponential(1.0); state = tick_state, watches = (:tick,))
    @test EpiBranch.watched_records(live) == (:tick,)
    @test EpiBranch.watched_records(Exponential(1.0)) == ()
    @test EpiBranch.watched_records([Exponential(1.0), live]) == (:tick,)
    @test_throws ArgumentError EpiBranch.pair_kernel(live, 1, 2, 0.0)

    # A projection has to declare what it reads, and a kernel with nothing to
    # read has nothing to declare.
    @test_throws "must declare every" PairKernel((c, a, b) -> Exponential(1.0); state = tick_state)
    @test_throws "no records to watch" PairKernel(c -> Exponential(1.0); watches = (:tick,))
    @test_throws "no records to watch" PairKernel(
        (c, a, b) -> Exponential(1.0); state = [nothing, nothing], watches = (:tick,)
    )
    @test_throws "`Symbol`s" PairKernel(
        (c, a, b) -> Exponential(1.0); state = tick_state, watches = ("tick",)
    )
    # A race needs one declaration per route, in route order.
    @test_throws ArgumentError EpiBranch._route_watch_keys(((:tick,),), 2)

    # Every opening takes a slot whether or not its route is watched, so an
    # opening's position in `openings` is its position in the watch.
    watch = EpiBranch._LiveWatch(3, 2)
    EpiBranch._watch_opening!(watch, 1, false)
    EpiBranch._watch_opening!(watch, 1, true)
    @test length(watch.reach) == 3            # the seeds' slot and these two
    @test watch.opened_by[1] == [3]

    state = EpiBranch.new_state(
        BranchingProcess(Poisson(0.0)), [], NoAttributes(),
        StableRNG(1)
    )
    EpiBranch.add_individuals!(state, 3, [])
    state.individuals[1].state[:onset_time] = 2.0
    state.individuals[2].state[:onset_time] = NaN
    columns = EpiBranch._host_time_columns(state, (:onset_time, :policy_time))
    @test isequal(columns.onset_time, [2.0, NaN, missing])
    @test all(ismissing, columns.policy_time)
end
