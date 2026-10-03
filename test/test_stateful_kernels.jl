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
    callback(c, a, b) = Exponential(
        exp(
            a.log_scale + b.log_scale +
                c.infector_infection_time
        )
    )
    records = [(log_scale = 0.1,), (log_scale = 0.3,)]
    kernel = RecordedKernel(records, callback)
    @test mean(EpiBranch.pair_kernel(kernel, 1, 2, 0.0)) ≈ exp(0.4)
    @test pairwise_surv_loglik(kernel, data, layout) ≈ -0.4 - 2exp(-0.4)
    live = StatefulKernel(ind -> (log_scale = ind.state[:log_scale]::Float64,), callback)
    @test_throws ArgumentError pairwise_surv_loglik(live, data, layout)
    @test_throws ArgumentError pairwise_surv_loglik(
        kernel,
        PairwiseSurvivalData([2], [0.0], [1.0], [true])
    )
    f(x) = pairwise_surv_loglik(
        RecordedKernel(
            [
                (log_scale = x[1],),
                (log_scale = x[2],),
            ], callback
        ), data, layout
    )
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
                [
                    truncated(Exponential(inv(x[1])); upper = b.date),
                    b.date + Exponential(inv(x[2])),
                ],
                [1 - survival, survival]
            )
        end
        pairwise_surv_loglik(CalendarKernel(RecordedKernel(records, callback)), data, layout)
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
        introduction, kernel
    )
    return state
end

@testset "An unchanging live kernel leaves the race stream alone" begin
    seeds = [0.0; fill(Inf, 11)]
    for d in (Exponential(1.5), Weibull(2.0, 2.0), Gamma(3.0, 0.7))
        live = StatefulKernel(
            ind -> (tag = get(ind.state, :tag, 0.0)::Float64,),
            (c, a, b) -> d
        )
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
    project = ind -> ind.state[:history]
    members = [1, 2, 3]
    records = [deepcopy(project(state.individuals[i])) for i in members]
    # Case 1 has an open opening that reaches member 2; member 3 is out of reach.
    openings = [
        EpiBranch._RouteOpening(0, 0, 0.0, Inf),
        EpiBranch._RouteOpening(1, 1, 0.0, 5.0),
    ]
    watch = EpiBranch._LiveWatch(3)
    EpiBranch._watch_opening!(watch, 1)
    EpiBranch._watch_target!(watch, 2, 2)
    processed = [true, false, false]
    changed!(case, now = 1.0) = EpiBranch._records_changed!(
        records, project, state,
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
    @test records[1] == [1.0]
    # Member 2 is compared once however many open openings reach it, and leaves
    # the watch once they have all closed.
    push!(openings, EpiBranch._RouteOpening(1, 1, 0.0, 8.0))
    EpiBranch._watch_opening!(watch, 1)
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
    changed = stateful_test_race(
        kernel, [0.0, Inf, Inf];
        interventions = [RecordKernelPolicy()]
    )
    @test changed.individuals[2].infection_time == 1.0
    @test changed.individuals[3].state[:policy_time] == 1.5
    recorded = record_kernel(kernel, changed)
    @test logccdf(EpiBranch.pair_kernel(kernel, 1, 3, 0.0, 0.0, changed), 2.0) ≈
        logccdf(EpiBranch.pair_kernel(recorded, 1, 3, 0.0), 2.0)
    calendar = CalendarKernel(
        StatefulKernel(
            project,
            (c, a, b) -> Exponential(2.0)
        )
    )
    @test mean(EpiBranch.pair_kernel(calendar, 1, 3, 0.0, 0.5, changed)) ≈ 2.0
    @test mean(EpiBranch.pair_kernel(Exponential(2.0), 1, 3, 0.0, 0.5, changed)) == 2.0
    @test logccdf(EpiBranch.pair_kernel(recorded, 1, 3, 0.0, 0.0, changed), 2.0) ≈
        logccdf(EpiBranch.pair_kernel(recorded, 1, 3, 0.0), 2.0)
    fixed = RecordedKernel([nothing, nothing], (c, a, b) -> Exponential(1.0))
    replay = stateful_test_race(fixed, [0.0, Inf])
    ordinary = stateful_test_race(Exponential(1.0), [0.0, Inf])
    @test isequal(
        [i.infection_time for i in replay.individuals],
        [i.infection_time for i in ordinary.individuals]
    )

    # Retried introductions must remain later than the admission boundary even
    # when another introduction settles and refreshes the remaining queue.
    inactive = StatefulKernel(tick_state, (c, a, b) -> Dirac(20.0))
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
    atoms = StatefulKernel(
        tick_state,
        (c, a, b) -> c.susceptible == 2 ? two_atoms : Dirac(1.0)
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
    live = StatefulKernel(tick_state, (c, a, b) -> law)
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
    live = StatefulKernel(tick_state, (c, a, b) -> Exponential(1.0))
    # Watching is a property of the kernel, not of whatever interventions happen
    # to be in play: it no longer depends on them at all.
    @test EpiBranch.watched_records(live) === (tick_state,)
    @test isempty(EpiBranch.watched_records(Exponential(1.0)))
    @test_throws MethodError EpiBranch.pair_kernel(live, 1, 2, 0.0)

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
