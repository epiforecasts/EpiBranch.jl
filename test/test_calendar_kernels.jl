using ForwardDiff
using DifferentiationInterface: DifferentiationInterface
using ADTypes: AutoMooncake
import Mooncake

struct CalendarInfections{T} <: InfectionLayer
    infection_time::Vector{T}
    infectious_time::Vector{T}
    removal_time::Vector{T}
    is_index::Vector{Bool}
    obs_end::Float64
end
EpiBranch.contact_structure(::CalendarInfections) = [[2], [1]]
function calendar_data(t, opening)
    CalendarInfections([t, oftype(t, 4)],
        [opening, oftype(opening, 5)], [oftype(t, 6), oftype(t, 7)], [true, false], 0.0)
end

# A constant rate before a policy date and another constant rate afterwards.
function policy_calendar(before, after, date)
    survival = exp(-before * date)
    MixtureModel(
        [truncated(Exponential(inv(before)); upper = date),
            date + Exponential(inv(after))],
        [1 - survival, survival])
end

@testset "Calendar-time pair kernels" begin
    calendar = policy_calendar(0.4, 0.1, 3.0)
    kernel = CalendarKernel(calendar)
    for opening in (1.0, 3.5)
        interval = EpiBranch.pair_kernel(kernel, 1, 2, 0.0, opening)
        for date in (opening + 0.2, opening + 2.1)
            integrated(t) = 0.4 * min(t, 3.0) + 0.1 * max(t - 3.0, 0.0)
            @test EpiBranch.cumhazard(interval, date - opening) ≈
                  integrated(date) - integrated(opening)
            @test exp(EpiBranch.loghazard(interval, date - opening)) ≈
                  (date < 3 ? 0.4 : 0.1)
        end
    end
    interval = EpiBranch.pair_kernel(kernel, 1, 2, 0.0, 1.0)
    draws = rand(StableRNG(233), interval, 10000)
    @test count(<=(4.0), draws) / length(draws) ≈ 1 - exp(-1) atol = 0.02
    @test_throws ArgumentError EpiBranch.pair_kernel(kernel, 1, 2, 0.0)
    @test_throws ArgumentError EpiBranch.pair_kernel(CalendarKernel(Uniform(0.0, 1.0)),
        1, 2, 0.0, 2.0)
    rows = PairwiseSurvivalData([2], [0.0], [1.0], [true])
    @test_throws ArgumentError pairwise_surv_loglik(kernel, rows)

    data = calendar_data(1.0, 2.0)
    layout = compile_contact_pairs(data)
    @test pairwise_surv_loglik(kernel, data, layout) ≈ log(0.1) - 0.5
    @test pairwise_surv_loglik(kernel, calendar_data(1.0, 3.5), layout) ≈ log(0.1) - 0.05
    @test pairwise_surv_loglik(kernel, calendar_data(1.0, Inf), layout) == -Inf
    @test pairwise_surv_loglik(kernel, calendar_data(1.0, NaN), layout) == -Inf

    # A smooth calendar law permits an analytical check of both AD modes.
    f(x) = pairwise_surv_loglik(CalendarKernel(Weibull(2.0, x[1])),
        calendar_data(x[2], x[3]), layout)
    reference(x) = log(8 / x[1]^2) - (16 - x[3]^2) / x[1]^2
    x = [3.0, 1.0, 2.0]
    @test f(x) ≈ reference(x)
    @test ForwardDiff.gradient(f, x) ≈ ForwardDiff.gradient(reference, x)
    @test DifferentiationInterface.gradient(f, AutoMooncake(), x) ≈
          ForwardDiff.gradient(reference, x)

    contextual = CalendarKernel(ContextualKernel(c -> Weibull(2.0, 2 +
                                                                   c.infector_infection_time)))
    @test pairwise_surv_loglik(contextual, data, layout) ≈ reference(x)
    per_edge = CalendarKernel([[Weibull(2.0, 3.0)], [Weibull(2.0, 3.0)]])
    @test pairwise_surv_loglik(per_edge, data, layout) ≈ reference(x)
end
