include(joinpath(@__DIR__, "..", "..", "..", "test", "testutils", "calendar_kernels.jl"))
test_calendar_simulation(k -> HouseholdProcess([3], k), household_infections)

@testset "calendar_time keyword matches an explicit calendar kernel" begin
    calendar = Weibull(2.0, 3.0)
    progression = [
        Transition(:infectious; delay = Uniform(0.4, 0.8)),
        Transition(:recovered; from = :infectious, delay = 4.0, terminal = true),
    ]
    keyword = ModelSpec(HouseholdProcess([3, 2], calendar; calendar_time = true); progression)
    wrapped = ModelSpec(HouseholdProcess([3, 2], CalendarKernel(calendar)); progression)
    @test keyword.process.kernel isa CalendarKernel
    a = simulate(keyword; rng = StableRNG(234))
    b = simulate(wrapped; rng = StableRNG(234))
    @test isequal(
        [i.infection_time for i in a.individuals],
        [i.infection_time for i in b.individuals]
    )
    data = household_infections(a, keyword)
    @test loglikelihood(data, keyword) == loglikelihood(data, wrapped)
end
