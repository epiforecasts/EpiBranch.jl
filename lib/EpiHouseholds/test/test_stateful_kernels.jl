include(joinpath(@__DIR__, "..", "..", "..", "test", "testutils", "stateful_kernels.jl"))
test_stateful_simulation((k; kwargs...) -> HouseholdProcess([3], k; kwargs...), household_infections)

struct RecordKernelClock <: AbstractIntervention end
function EpiBranch.resolve_individual!(::RecordKernelClock, ind, state)
    clock = get!(state.individuals[1].state, :kernel_clock, Float64[])
    push!(clock, ind.infection_time)
    return nothing
end

@testset "Live kernels share one household clock" begin
    kernel = StatefulKernel(_ -> nothing, (c, a, b) -> Exponential(1.0))
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

@testset "race_groups dispatches on the kernel's own declaration, not on interventions" begin
    # Whether a kernel can see other households is a property it declares
    # itself (EpiBranch.watched_records), not something the engine infers from
    # whether any interventions happen to be in play. A fixed kernel always
    # races each household on its own; a live kernel always shares one clock,
    # even here, where nothing ever writes the record it reads.
    project(ind) = (tag = get(ind.state, :tag, 0.0)::Float64,)
    fixed = Exponential(2.0)
    live = StatefulKernel(project, (c, a, b) -> Exponential(2.0))
    process = HouseholdProcess([2, 3, 2], fixed)
    @test EpiHouseholds.race_groups(process, fixed) == process.members
    @test EpiHouseholds.race_groups(process, live) ==
        (collect(eachindex(process.household_of)),)
end
