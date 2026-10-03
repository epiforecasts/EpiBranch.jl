include(joinpath(@__DIR__, "..", "..", "..", "test", "testutils", "pair_kernels.jl"))
test_contextual_simulation((k; kwargs...) -> HouseholdProcess([3], k; kwargs...), household_infections)
test_calendar_simulation((k; kwargs...) -> HouseholdProcess([3], k; kwargs...), household_infections)
test_stateful_simulation((k; kwargs...) -> HouseholdProcess([3], k; kwargs...), household_infections)

struct RecordKernelClock <: AbstractIntervention end
function EpiBranch.resolve_individual!(::RecordKernelClock, ind, state)
    clock = get!(state.individuals[1].state, :kernel_clock, Float64[])
    push!(clock, ind.infection_time)
    return nothing
end

@testset "Live kernels share one household clock" begin
    kernel = PairKernel(
        (c, a, b) -> Exponential(1.0);
        state = ind -> (tick = get(ind.state, :tick, 0)::Int,), watches = (:tick,)
    )
    model = ModelSpec(
        HouseholdProcess([2, 2], kernel);
        progression = [Transition(:recovered; delay = 10.0, terminal = true)],
        interventions = [RecordKernelClock()]
    )
    state = simulate(model; initial_cases = [1, 3], rng = StableRNG(233))
    @test issorted(state.individuals[1].state[:kernel_clock])
    @test [i.id for i in state.individuals if get(i.state, :index, false)] == [1, 3]
    @test state.cumulative_cases == 4
    @test state.individuals[2].parent_id == 1
    @test state.individuals[4].parent_id == 3
    # Default household seeding still chooses one index in each partition.
    default = simulate(model; rng = StableRNG(233))
    @test count(i -> get(i.state, :index, false), default.individuals[1:2]) == 1
    @test count(i -> get(i.state, :index, false), default.individuals[3:4]) == 1
end

@testset "A kernel watching no record keeps separate household races" begin
    # A projection reading no state key has nothing that can move, so neither
    # redrawing nor a shared clock is needed and the run must match an ordinary
    # kernel exactly.
    scales = fill(2.0, 7)
    by_id(ind) = (scale = scales[ind.id],)
    progression = [Transition(:recovered; delay = 4.0, terminal = true)]
    kernels = (
        Exponential(2.0),
        PairKernel((c, a, b) -> Exponential(a.scale); state = by_id, watches = ()),
    )
    for seed in 1:25
        runs = map(kernels) do kernel
            process = HouseholdProcess(
                [2, 3, 2], kernel; external_hazard = 0.1,
                obs_end = 10.0
            )
            state = simulate(ModelSpec(process; progression); rng = StableRNG(seed))
            [i.infection_time for i in state.individuals]
        end
        @test isequal(runs...)
    end
end

@testset "A watched record puts every household on one clock" begin
    # Declaring a record is what asks for the shared clock, whether or not an
    # intervention is the thing that moves it: an attribute builder or a
    # transition can move one too. One clock seeds and races the households
    # together, so the stream differs run by run while the outbreak stays the
    # same distribution.
    project(ind) = (tag = get(ind.state, :tag, 0.0)::Float64,)
    progression = [Transition(:recovered; delay = 4.0, terminal = true)]
    household_run(watches, seed) = simulate(
        ModelSpec(
            HouseholdProcess(
                [2, 3, 2],
                PairKernel(
                    (c, a, b) -> Exponential(2.0); state = project, watches = watches
                ); external_hazard = 0.1, obs_end = 10.0
            );
            progression = progression
        );
        rng = StableRNG(seed)
    )
    times(state) = [i.infection_time for i in state.individuals]
    @test any(
        !isequal(times(household_run((:tag,), seed)), times(household_run((), seed)))
            for seed in 1:5
    )
    means = map(((:tag,), ())) do watches
        mean(household_run(watches, seed).cumulative_cases for seed in 1:400)
    end
    @test isapprox(means[1], means[2]; atol = 0.25)
end
