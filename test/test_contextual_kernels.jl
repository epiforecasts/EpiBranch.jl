using ForwardDiff
using DifferentiationInterface: DifferentiationInterface
using ADTypes: AutoMooncake
import Mooncake

struct ContextInfections{T} <: InfectionLayer
    infection_time::Vector{T}
    infectious_time::Vector{T}
    removal_time::Vector{T}
    is_index::Vector{Bool}
    obs_end::Float64
end
EpiBranch.contact_structure(::ContextInfections) = [[2], [1]]
function context_data(t)
    ContextInfections([t, oftype(t, 4)], [t + 1, oftype(t, 5)],
        [t + 6, oftype(t, 10)], [true, false], 0.0)
end

struct ContextRule{T}
    slope::T
end
function (rule::ContextRule)(context::PairContext)
    Exponential(exp(rule.slope * context.infector_infection_time))
end

@testset "Contextual pair kernels" begin
    distribution = Exponential(2.0)
    @test EpiBranch.pair_kernel(distribution, 1, 2, 3.0) === distribution
    @test mean(EpiBranch.pair_kernel((i, j) -> Exponential(i + j), 1, 2, NaN)) == 3.0
    rule = ContextualKernel(ContextRule(0.2))
    @test mean(EpiBranch.pair_kernel(rule, 1, 2, 3.0)) ≈ exp(0.6)
    @test isbitstype(typeof(PairContext(1, 2, 3.0)))
    rows = PairwiseSurvivalData([2], [0.0], [1.0], [true])
    @test_throws ArgumentError pairwise_surv_loglik(rule, rows)

    covariates = [0.5, 1.5]
    data = context_data(2.0)
    layout = compile_contact_pairs(data)
    kernel(a, b) = ContextualKernel(context -> Exponential(
        exp(a + b * context.infector_infection_time +
            0.1 * covariates[context.susceptible])))
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
