using EpiBranch: initialise_state
using Distributions
using DataFrames
using Statistics
using StableRNGs

@testset "initialise_state from observed cases" begin
    model = BranchingProcess(Poisson(1.5), Gamma(2.5, 2.0))
    cases = DataFrame(
        id = ["a", "b", "c", "d"], parent_id = [missing, "a", "a", "b"],
        infection_time = [0.0, 3.1, 4.6, 7.2],
    )

    @testset "the table becomes the state at `now`" begin
        state = initialise_state(model, cases; rng = StableRNG(1))
        @test state isa SimulationState
        @test state.cumulative_cases == 4
        @test [ind.parent_id for ind in state.individuals[1:4]] == [0, 1, 1, 2]
        @test [ind.generation for ind in state.individuals[1:4]] == [0, 1, 1, 2]
        @test [ind.infection_time for ind in state.individuals[1:4]] == cases.infection_time
        @test state.individuals[1].secondary_case_ids[1:2] == [2, 3]
        @test state.max_infection_time == 7.2
        @test !state.extinct
    end

    @testset "it is continued by simulate" begin
        state = initialise_state(model, cases; rng = StableRNG(2))
        simulate(model, state; max_time = 60.0)
        @test state.cumulative_cases >= 4
        @test [ind.infection_time for ind in state.individuals[1:4]] == cases.infection_time
        later = [ind for ind in state.individuals[5:end] if is_infected(ind)]
        @test all(ind.infection_time > 7.2 for ind in later)
        @test all(ind.parent_id in 1:state.cumulative_cases for ind in later)
    end

    @testset "known offspring are all a case has made by `now`" begin
        # An old case with no listed offspring has made none by `now`. With
        # nearly all of the generation-time distribution behind it, it makes
        # no more.
        one_case = DataFrame(infection_time = [0.0])
        for seed in 1:20
            state = initialise_state(model, one_case; now = 200.0, rng = StableRNG(seed))
            simulate(model, state; max_time = 400.0)
            @test state.cumulative_cases == 1
        end
    end

    @testset "later contacts follow the offspring law given those seen" begin
        # Exponential generation time and Poisson offspring: with none seen by
        # `now`, the later contacts are Poisson with mean R * P(T > now).
        R, mean_gt, now = 0.8, 2.0, 1.0
        m = BranchingProcess(Poisson(R), Exponential(mean_gt))
        one_case = DataFrame(infection_time = [0.0])
        rng = StableRNG(3)
        n = 4000
        children = map(1:n) do _
            state = initialise_state(m, one_case; now, rng)
            simulate(m, state; max_time = 1000.0, rng)
            count(
                i -> is_infected(state.individuals[i]) &&
                    state.individuals[i].parent_id == 1, 2:length(state.individuals)
            )
        end
        expected = R * exp(-now / mean_gt)
        @test isapprox(mean(children), expected; atol = 4 * sqrt(expected / n))
    end

    @testset "an isolated case infects no one after isolation" begin
        iso = DataFrame(
            infection_time = [0.0, 1.0], isolation_time = [0.5, missing],
            parent_id = [missing, 1]
        )
        m = BranchingProcess(Poisson(5.0), Gamma(2.5, 2.0))
        for seed in 1:10
            state = initialise_state(m, iso; now = 2.0, rng = StableRNG(seed))
            simulate(m, state; max_time = 30.0)
            kids = [
                i for i in 3:length(state.individuals)
                    if is_infected(state.individuals[i]) && state.individuals[i].parent_id == 1
            ]
            @test isempty(kids)
        end
    end

    @testset "a ModelSpec carries its layers, and onset is used as observed" begin
        spec = ModelSpec(
            model;
            attributes = clinical_presentation(incubation_period = LogNormal(1.6, 0.5)),
            interventions = [Isolation(onset_to_isolation_delay = 1.0, duration = Inf)],
        )
        with_onset = DataFrame(
            infection_time = [0.0, 2.0], parent_id = [missing, 1],
            onset_time = [4.0, missing],
        )
        state = initialise_state(spec, with_onset; rng = StableRNG(4))
        @test onset_time(state.individuals[1]) == 4.0
        simulate(spec, state; max_time = 30.0)
        @test state.cumulative_cases >= 2
    end

    @testset "invalid tables are refused" begin
        @test_throws ArgumentError initialise_state(model, DataFrame(x = [1.0]))
        @test_throws ArgumentError initialise_state(
            model, DataFrame(infection_time = [0.0], parent_id = [7])
        )
        @test_throws ArgumentError initialise_state(
            model, DataFrame(infection_time = [0.0, 1.0], parent_id = [2, 1])
        )
        @test_throws ArgumentError initialise_state(
            model, DataFrame(infection_time = [0.0, 5.0]); now = 2.0
        )
        @test_throws ArgumentError initialise_state(
            model, DataFrame(infection_time = [0.0], other = [1])
        )
        two_windows = BranchingProcess(
            (
                Infectiousness(Poisson(1.0); kernel = Gamma(2, 1)),
                Infectiousness(Poisson(1.0); kernel = Gamma(2, 1), name = :b),
            )
        )
        @test_throws ArgumentError initialise_state(
            two_windows, DataFrame(infection_time = [0.0])
        )
    end
end
