using DataFrames
using Dates

@testset "Line list output" begin
    @testset "linelist basic output" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = simulate(model;
            sim_opts=SimOpts(max_cases=50, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng)

        rng2 = StableRNG(99)
        df = linelist(state; rng=rng2)

        @test df isa DataFrame
        @test nrow(df) == state.cumulative_cases
        @test ncol(df) == 11
        @test "id" in names(df)
        @test "case_type" in names(df)
        @test "sex" in names(df)
        @test "age" in names(df)
        @test "date_infection" in names(df)
        @test "date_onset" in names(df)
        @test "outcome" in names(df)
    end

    @testset "linelist with delays" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = simulate(model;
            sim_opts=SimOpts(max_cases=30, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng)

        rng2 = StableRNG(99)
        df = linelist(state;
            delay_opts=DelayOpts(
                onset_to_reporting=Exponential(3.0),
                onset_to_admission=Exponential(5.0),
                onset_to_outcome=Exponential(10.0),
            ),
            rng=rng2)

        # Some rows should have non-missing reporting dates
        @test any(!ismissing, df.date_reporting)
        # All onset dates should be present (incubation period was set)
        @test all(!ismissing, df.date_onset)
        # Onset should be on or after infection
        for row in eachrow(df)
            if !ismissing(row.date_onset)
                @test row.date_onset >= row.date_infection
            end
        end
    end

    @testset "linelist demographics" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = simulate(model;
            sim_opts=SimOpts(max_cases=100, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng)

        rng2 = StableRNG(99)
        df = linelist(state;
            demographic_opts=DemographicOpts(
                age_distribution=Normal(40, 15),
                age_range=(0, 90),
                prob_female=0.6,
            ),
            rng=rng2)

        @test all(0 .<= df.age .<= 90)
        @test all(s -> s in ("female", "male"), df.sex)
    end

    @testset "linelist with age-specific CFR" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = simulate(model;
            sim_opts=SimOpts(max_cases=200, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng)

        age_cfr = Dict((0, 50) => 0.01, (51, 90) => 0.5)
        rng2 = StableRNG(99)
        df = linelist(state;
            outcome_opts=OutcomeOpts(age_specific_cfr=age_cfr),
            rng=rng2)

        @test any(df.outcome .== "died")
        @test any(df.outcome .== "recovered")
    end

    @testset "linelist empty state" begin
        state = SimulationState(Individual[], Int[], 0, StableRNG(1), 0, true, nothing, 0.0, 1.0, 1.0, 0.0, nothing)
        df = linelist(state)
        @test nrow(df) == 0
        @test ncol(df) == 11
    end

    @testset "contacts output" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        state = simulate(model;
            sim_opts=SimOpts(max_cases=50),
            rng=rng)

        df = contacts(state)
        @test df isa DataFrame
        @test "from" in names(df)
        @test "to" in names(df)
        @test "date_infection" in names(df)

        # Number of contact pairs should equal total secondary cases
        n_secondary = state.cumulative_cases - count(i -> i.parent_id == 0, state.individuals)
        @test nrow(df) == n_secondary
    end

    @testset "index cases labelled correctly" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = simulate(model;
            sim_opts=SimOpts(max_cases=20, n_initial=3, incubation_period=LogNormal(1.5, 0.5)),
            rng=rng)

        rng2 = StableRNG(99)
        df = linelist(state; rng=rng2)

        n_index = count(df.case_type .== "index")
        @test n_index == 3
    end
end
