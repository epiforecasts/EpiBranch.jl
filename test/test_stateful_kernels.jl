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
