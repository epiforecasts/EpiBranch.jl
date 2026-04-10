using DataFrames
using Dates

@testset "Line list output" begin
    clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

    @testset "linelist basic output" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = simulate(model; attributes = clinical, sim_opts = SimOpts(max_cases = 50), rng = rng)

        df = linelist(state; rng = StableRNG(99))

        @test df isa DataFrame
        @test nrow(df) == state.cumulative_cases
        @test "id" in names(df)
        @test "case_type" in names(df)
        @test "date_infection" in names(df)
        @test "date_onset" in names(df)
    end

    @testset "linelist without clinical presentation has no onset" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = simulate(model; sim_opts = SimOpts(max_cases = 50), rng = rng)

        df = linelist(state)
        @test "id" in names(df)
        @test !("date_onset" in names(df))
        @test !("age" in names(df))
    end

    @testset "linelist with delays" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = simulate(model; attributes = clinical, sim_opts = SimOpts(max_cases = 30), rng = rng)

        df = linelist(state;
            delays = DelayOpts(
                onset_to_reporting = Exponential(3.0),
                onset_to_admission = Exponential(5.0),
                onset_to_outcome = Exponential(10.0)
            ),
            outcomes = OutcomeOpts(),
            rng = StableRNG(99))

        @test "date_reporting" in names(df)
        @test "date_admission" in names(df)
        @test "outcome" in names(df)
        @test all(!ismissing, df.date_onset)
        for row in eachrow(df)
            if !ismissing(row.date_onset)
                @test row.date_onset >= row.date_infection
            end
        end
    end

    @testset "linelist post-hoc demographics" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = simulate(model; attributes = clinical, sim_opts = SimOpts(max_cases = 100), rng = rng)

        df = linelist(state;
            demographics = DemographicOpts(
                age_distribution = Normal(40, 15),
                age_range = (0, 90),
                prob_female = 0.6
            ),
            rng = StableRNG(99))

        @test "age" in names(df)
        @test "sex" in names(df)
        @test all(0 .<= df.age .<= 90)
        @test all(s -> s in ("female", "male"), df.sex)
    end

    @testset "linelist with demographics from attributes" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        init_fn = compose(
            clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
            demographics(age_distribution = Normal(40, 15))
        )
        state = simulate(model; attributes = init_fn, sim_opts = SimOpts(max_cases = 50), rng = rng)

        df = linelist(state)
        @test "age" in names(df)
        @test "sex" in names(df)
        # Demographics should NOT be overwritten by post-hoc
        @test !ismissing(df.age[1])
    end

    @testset "linelist with age-specific CFR" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        state = simulate(model;
            condition = 50:500, attributes = clinical, sim_opts = SimOpts(max_cases = 500), rng = rng)

        age_cfr = Dict((0, 50) => 0.01, (51, 90) => 0.5)
        df = linelist(state;
            demographics = DemographicOpts(age_range = (0, 90)),
            outcomes = OutcomeOpts(age_specific_cfr = age_cfr),
            rng = StableRNG(99))

        @test any(df.outcome .== "died")
        @test any(df.outcome .== "recovered")
    end

    @testset "linelist empty state" begin
        # Use simulate with impossible conditions to get an empty-ish state
        model = BranchingProcess(Poisson(0.0), Exponential(1.0))
        state = simulate(model; sim_opts = SimOpts(max_cases = 1), rng = StableRNG(1))
        empty_state = SimulationState(Individual[], Int[], 0, StableRNG(1), 0, true,
            nothing, 0.0, 0.0, nothing)
        df = linelist(empty_state)
        @test nrow(df) == 0
    end

    @testset "linelist includes intervention state" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        iso = Isolation(delay = Exponential(1.0))
        state = simulate(model;
            interventions = [iso], attributes = clinical,
            sim_opts = SimOpts(max_cases = 50), rng = rng)

        df = linelist(state)
        @test "isolated" in names(df)
    end

    @testset "contacts output" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        state = simulate(model; sim_opts = SimOpts(max_cases = 50), rng = rng)

        df = contacts(state)
        @test df isa DataFrame
        @test "from" in names(df)
        @test "to" in names(df)
        @test "was_case" in names(df)
    end

    @testset "index cases labelled correctly" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = simulate(
            model; attributes = clinical, sim_opts = SimOpts(max_cases = 20, n_initial = 3), rng = rng)

        df = linelist(state; rng = StableRNG(99))

        n_index = count(df.case_type .== "index")
        @test n_index == 3
    end
end
