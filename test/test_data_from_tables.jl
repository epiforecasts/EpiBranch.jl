using EpiBranch, Distributions, DataFrames, Dates, StableRNGs, Test

@testset "data types from tables and simulations" begin
    model = BranchingProcess(Poisson(0.8), Exponential(5.0))
    st = simulate(model; rng = StableRNG(1))
    ll = linelist(st)

    @testset "OffspringCounts" begin
        oc = OffspringCounts(ll)
        @test length(oc.data) == nrow(ll)
        @test sum(oc.data) == count(!=(0), ll.parent_id)
        @test OffspringCounts(st).data == oc.data
        df = DataFrame(who = [1, 2, 3, 4], by = [0, 1, 1, missing])
        @test sort(OffspringCounts(df; id = :who, parent_id = :by).data) == [0, 0, 0, 2]
    end

    @testset "ChainSizes" begin
        @test sort(ChainSizes(DataFrame(chain_id = [1, 1, 2, 3, 3, 3])).data) == [1, 2, 3]
        sts = simulate(BranchingProcess(Poisson(0.8)), 10; rng = StableRNG(2))
        cs = ChainSizes(sts)
        @test length(cs.data) == 10
        @test sum(cs.data) == sum(length(linelist(s).id) for s in sts)
        @test ChainSizes(sts[1]).data == [nrow(linelist(sts[1]))]
        @test_throws ArgumentError ChainSizes(typeof(st)[])
    end

    @testset "IncidenceCounts and weekly_incidence" begin
        df = DataFrame(
            date_onset = Date(2024, 3, 1) .+
                Day.([0, 1, 3, 4, 4, 8, 9, 9, 10, 12, 12])
        )
        ic = IncidenceCounts(df; step = 3)
        @test ic.seeds == 2
        @test ic.data == [3, 1, 3, 2]
        @test_throws ArgumentError IncidenceCounts(df; step = 100)
        wk = weekly_incidence(df)
        @test wk.cases == [2, 6, 3]
        @test wk.week == [Date(2024, 2, 26), Date(2024, 3, 4), Date(2024, 3, 11)]
        @test weekly_incidence(
            DataFrame(date_onset = [Date(2024, 3, 1), missing])
        ).cases == [1]
        sim = simulate(model; rng = StableRNG(3), max_cases = 200)
        ics = IncidenceCounts(sim; step = 7)
        @test ics.seeds + sum(ics.data) == nrow(linelist(sim))
    end

    @testset "end_of_outbreak_inputs" begin
        df = DataFrame(
            id = [1, 2, 3, 4], parent_id = [0, 1, 1, 2],
            date_onset = [Date(2024, 3, 1), Date(2024, 3, 4), Date(2024, 3, 6), Date(2024, 3, 20)]
        )
        inp = end_of_outbreak_inputs(df, Date(2024, 3, 10))
        @test inp.τs == [9.0, 6.0, 4.0]
        @test inp.secondary_cases == [2, 0, 0]
        p = end_of_outbreak_probability(
            NegBin(0.8, 0.5), Gamma(2.0, 3.0), inp.τs, inp.secondary_cases
        )
        @test 0 < p < 1
        sinp = end_of_outbreak_inputs(st, 20.0)
        @test length(sinp.τs) == length(sinp.secondary_cases)
        @test all(>=(0), sinp.τs)
    end
end
