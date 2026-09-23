using ForwardDiff
using DifferentiationInterface: DifferentiationInterface
using ADTypes: AutoMooncake
import Mooncake

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
    callback(c, a, b) = Exponential(exp(a.log_scale + b.log_scale +
                                        c.infector_infection_time))
    records = [(log_scale = 0.1,), (log_scale = 0.3,)]
    kernel = StatefulKernel(records, callback)
    @test mean(EpiBranch.pair_kernel(kernel, 1, 2, 0.0)) ≈ exp(0.4)
    @test pairwise_surv_loglik(kernel, data, layout) ≈ -0.4 - 2exp(-0.4)
    live = StatefulKernel(ind -> (log_scale = ind.state[:log_scale]::Float64,), callback)
    @test_throws ArgumentError pairwise_surv_loglik(live, data, layout)
    @test_throws ArgumentError pairwise_surv_loglik(kernel,
        PairwiseSurvivalData([2], [0.0], [1.0], [true]))
    f(x) = pairwise_surv_loglik(
        StatefulKernel([(log_scale = x[1],),
                (log_scale = x[2],)], callback), data, layout)
    reference(x) = -sum(x) - 2exp(-sum(x))
    x = [0.1, 0.3]
    @test ForwardDiff.gradient(f, x) ≈ ForwardDiff.gradient(reference, x)
    @test DifferentiationInterface.gradient(f, AutoMooncake(), x) ≈
          ForwardDiff.gradient(reference, x)
    @test pairwise_surv_loglik(CalendarKernel(kernel), data, layout) ≈
          pairwise_surv_loglik(kernel, data, layout)

    state = EpiBranch.new_state(BranchingProcess(Poisson(0.0)), [], NoAttributes(), StableRNG(233))
    EpiBranch.add_individuals!(state, 2, [])
    for (ind, record) in zip(state.individuals, records)
        ind.state[:log_scale] = record.log_scale
        ind.state[:history] = [1.0]
    end
    saved = record_kernel(live, state)
    @test saved.state == records
    @test pairwise_surv_loglik(saved, data, layout) ≈ reference(x)
    @test record_kernel(CalendarKernel(live), state).kernel.state == records
    @test record_kernel(Exponential(), state) == Exponential()
    history = record_kernel(StatefulKernel(ind -> ind.state[:history], callback), state)
    push!(state.individuals[1].state[:history], 2.0)
    @test history.state[1] == [1.0]
    @test !EpiBranch._live_kernel(saved)
    @test EpiBranch._live_kernel(CalendarKernel(live))
end

@testset "Differentiable recorded event dates" begin
    data = StateKernelInfections([0.0, 3.0], [1.0, 4.0], [5.0, 6.0], [true, false], 0.0)
    layout = compile_contact_pairs(data)
    f = function (x)
        records = [(date = x[3],), (date = x[3],)]
        callback = function (c, a, b)
            survival = exp(-x[1] * b.date)
            MixtureModel(
                [truncated(Exponential(inv(x[1])); upper = b.date),
                    b.date + Exponential(inv(x[2]))],
                [1 - survival, survival])
        end
        pairwise_surv_loglik(CalendarKernel(StatefulKernel(records, callback)), data, layout)
    end
    reference(x) = log(x[2]) - x[1] * (x[3] - 1) - x[2] * (3 - x[3])
    x = [0.4, 0.1, 2.0]
    @test f(x) ≈ reference(x)
    @test ForwardDiff.gradient(f, x) ≈ ForwardDiff.gradient(reference, x)
    @test DifferentiationInterface.gradient(f, AutoMooncake(), x) ≈
          ForwardDiff.gradient(reference, x)
end

include("testutils/stateful_kernels.jl")

# Exercise the shared primitive without depending on a companion package.
function stateful_test_race(kernel, initial_times; interventions = (), introduction = nothing)
    rng = StableRNG(233)
    progression = [Transition(:recovered; delay = 5.0, terminal = true)]
    state = EpiBranch.new_state(BranchingProcess(Poisson(0.0)), progression,
        NoAttributes(), rng)
    n = length(initial_times)
    EpiBranch.add_individuals!(state, n, interventions)
    targets = (i, st) -> ((j,
                              EpiBranch.pair_kernel(kernel, i, j,
                                  st.individuals[i].infection_time, st.individuals[i].infection_time, st))
    for j in 1:n if j != i && !is_infected(st.individuals[j]))
    EpiBranch._sellke_race!(state, collect(1:n), rng;
        seed! = (best, members, r) -> copyto!(best, initial_times),
        targets, from = :infection, until = (:recovered,), interventions,
        introduction, refresh_projection = EpiBranch._kernel_projection(kernel))
    return state
end

@testset "An unchanging live kernel leaves the race stream alone" begin
    seeds = [0.0; fill(Inf, 11)]
    for d in (Exponential(1.5), Weibull(2.0, 2.0), Gamma(3.0, 0.7))
        live = StatefulKernel(ind -> (tag = get(ind.state, :tag, 0.0)::Float64,),
            (c, a, b) -> d)
        @test isequal([i.infection_time for i in stateful_test_race(d, seeds).individuals],
            [i.infection_time for i in stateful_test_race(live, seeds).individuals])
    end
end

@testset "A mutable history is seen to change" begin
    state = EpiBranch.new_state(BranchingProcess(Poisson(0.0)), [], NoAttributes(),
        StableRNG(233))
    EpiBranch.add_individuals!(state, 2, [])
    for ind in state.individuals
        ind.state[:history] = Float64[]
    end
    project = ind -> ind.state[:history]
    members = [1, 2]
    records = [deepcopy(project(state.individuals[i])) for i in members]
    @test !EpiBranch._records_changed!(records, project, state, members)
    # An intervention appending in place must not compare equal to its own
    # remembered record, which is why the race keeps a copy rather than an alias.
    push!(state.individuals[2].state[:history], 1.0)
    @test EpiBranch._records_changed!(records, project, state, members)
    @test !EpiBranch._records_changed!(records, project, state, members)
end

@testset "Shared race refreshes live pair kernels" begin
    ties = StatefulKernel(tick_state, (c, a, b) -> Dirac(1.0))
    state = stateful_test_race(ties, [0.0, Inf, Inf]; interventions = [TickEveryCase()])
    @test [i.infection_time for i in state.individuals] == [0.0, 1.0, 1.0]

    project(ind) = (date = get(ind.state, :policy_time, Inf)::Float64,)
    callback = function (c, a, b)
        (c.infector, c.susceptible) == (1, 2) && return Dirac(1.0)
        (c.infector, c.susceptible) == (1, 3) &&
            return state_policy_law(0.1, 1.0, b.date)
        return Dirac(20.0)
    end
    kernel = StatefulKernel(project, callback)
    changed = stateful_test_race(kernel, [0.0, Inf, Inf];
        interventions = [RecordKernelPolicy()])
    @test changed.individuals[2].infection_time == 1.0
    @test changed.individuals[3].state[:policy_time] == 1.5
    recorded = record_kernel(kernel, changed)
    @test logccdf(EpiBranch.pair_kernel(kernel, 1, 3, 0.0, 0.0, changed), 2.0) ≈
          logccdf(EpiBranch.pair_kernel(recorded, 1, 3, 0.0), 2.0)
    calendar = CalendarKernel(StatefulKernel(project,
        (c, a, b) -> Exponential(2.0)))
    @test mean(EpiBranch.pair_kernel(calendar, 1, 3, 0.0, 0.5, changed)) ≈ 2.0
    @test mean(EpiBranch.pair_kernel(Exponential(2.0), 1, 3, 0.0, 0.5, changed)) == 2.0
    @test logccdf(EpiBranch.pair_kernel(recorded, 1, 3, 0.0, 0.0, changed), 2.0) ≈
          logccdf(EpiBranch.pair_kernel(recorded, 1, 3, 0.0), 2.0)
    fixed = StatefulKernel([nothing, nothing], (c, a, b) -> Exponential(1.0))
    replay = stateful_test_race(fixed, [0.0, Inf])
    ordinary = stateful_test_race(Exponential(1.0), [0.0, Inf])
    @test isequal([i.infection_time for i in replay.individuals],
        [i.infection_time for i in ordinary.individuals])

    # Retried introductions must remain later than the admission boundary even
    # when another introduction settles and refreshes the remaining queue.
    inactive = StatefulKernel(tick_state, (c, a, b) -> Dirac(20.0))
    introduced = stateful_test_race(inactive, [0.1, 0.2, 0.3];
        interventions = [WaitForKernelDay(), TickEveryCase()],
        introduction = (Exponential(0.2), 3.0))
    cases = filter(is_infected, introduced.individuals)
    @test length(cases) == 3
    @test all(i -> 1.0 <= i.infection_time <= 3.0, cases)
end
