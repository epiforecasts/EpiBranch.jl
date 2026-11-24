using EpiBranch, Distributions, StableRNGs, DataFrames, Dates, Test

@testset "incidence" begin
    model = ModelSpec(
        BranchingProcess(NegBin(1.0, 0.5), Exponential(5.0));
        attributes = clinical_presentation(incubation_period = LogNormal(1.6, 0.5))
    )
    states = simulate(model, 20; rng = StableRNG(1), max_time = 60.0)
    st = states[1]

    @testset "weekly matches weekly_incidence on non-empty weeks" begin
        w = weekly_incidence(st)
        d = incidence(st)
        @test names(d) == ["date", "cases"]
        @test sum(d.cases) == sum(w.cases)
        @test filter(:cases => >(0), d).date == w.week
        @test filter(:cases => >(0), d).cases == w.cases
        @test all(dayofweek.(d.date) .== 1)
        @test d.date == collect(first(d.date):Week(1):last(d.date))
    end

    @testset "by takes a TransitionCount or a function of the case" begin
        inf = weekly_incidence(st; by = TransitionCount(:infection))
        @test sum(inf.cases) == st.cumulative_cases
        by_fn = weekly_incidence(st; by = ind -> ind.infection_time)
        @test by_fn == inf
        onset = weekly_incidence(st; by = TransitionCount(:onset))
        @test onset.week[1] >= inf.week[1]
        @test incidence(st; by = TransitionCount(:infection)).cases ==
            incidence(st; by = ind -> ind.infection_time).cases
        @test_throws TypeError weekly_incidence(st; by = :infection)
        @test_throws TypeError end_of_outbreak_inputs(st, 20.0; by = :onset)
        @test end_of_outbreak_inputs(st, 20.0; by = TransitionCount(:infection)).τs ==
            [20.0 - ind.infection_time for ind in individuals(st) if ind.infection_time <= 20.0]
    end

    @testset "daily, other intervals and range" begin
        d1 = incidence(st; interval = 1)
        @test sum(d1.cases) == sum(incidence(st).cases)
        @test diff(d1.date) == fill(Day(1), nrow(d1) - 1)
        d3 = incidence(st; interval = 3)
        @test sum(d3.cases) == sum(d1.cases)
        @test diff(d3.date) == fill(Day(3), nrow(d3) - 1)
        r = incidence(st; interval = 1, range = (Date(2020, 1, 1), Date(2020, 1, 10)))
        @test r.date == collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 10))
        @test_throws ArgumentError incidence(st; interval = 0)
        @test_throws ArgumentError incidence(st; range = (Date(2020, 2, 1), Date(2020, 1, 1)))
    end

    @testset "cumulative and by generation" begin
        d = incidence(st; cumulative = true)
        @test issorted(d.cases)
        @test d.cases[end] == sum(incidence(st).cases)
        g = incidence(st; by_generation = true)
        @test names(g) == ["date", "generation", "cases"]
        @test sum(g.cases) == sum(incidence(st).cases)
        n_gen = maximum(i.generation for i in individuals(st)) + 1
        @test nrow(g) == nrow(incidence(st)) * n_gen
        gc = incidence(st; by_generation = true, cumulative = true)
        @test all(issorted(x.cases) for x in groupby(gc, :generation))
    end

    @testset "several outbreaks share rows" begin
        df = incidence(states; interval = 1)
        @test names(df) == ["sim_id", "date", "cases"]
        @test sort(unique(df.sim_id)) == 1:20
        @test length(unique(nrow(g) for g in groupby(df, :sim_id))) == 1
        @test sum(df.cases) == sum(sum(incidence(s; interval = 1).cases) for s in states)
        @test any(==(0), df.cases)
    end

    @testset "by names the state that places a case" begin
        by_infection = incidence(st; interval = 1, by = TransitionCount(:infection))
        @test sum(by_infection.cases) == count(is_infected, individuals(st))
        @test incidence(st; interval = 1, by = TransitionCount(:infection)) ==
            incidence(st; interval = 1, by = TransitionCount(ind -> ind.infection_time))
    end

    @testset "no cases" begin
        @test nrow(incidence(st; by = TransitionCount(:no_such_time))) == 0
        z = incidence(st; by = TransitionCount(:no_such_time), range = (Date(2020, 1, 1), Date(2020, 1, 14)), interval = 7)
        @test z.cases == [0, 0, 0]
    end
end
