@testset "Infectiousness windows" begin
    @testset "explicit single window holds offspring + kernel" begin
        m = BranchingProcess(Infectiousness(NegBin(2.5, 0.16); kernel = LogNormal(1.6, 0.5)))
        @test length(m.infectiousness) == 1
        @test m.infectiousness[1].from == :infection
        @test single_type_offspring(m) isa NegativeBinomial
        @test m.infectiousness[1].kernel isa LogNormal
    end

    @testset "from-state offsets the generation interval" begin
        latent = Transition(:infectious, from = :infection, delay = (rng, ind) -> 4.0)
        m = BranchingProcess(
            Infectiousness((rng, ind) -> 3;
                from = :infectious, kernel = Exponential(0.5));
            progression = [latent])
        s = simulate(m;
            n_initial = 30, max_generations = 1, rng = StableRNG(1))
        kids = filter(i -> i.parent_id != 0 && is_infected(i), s.individuals)
        gi = [k.infection_time - s.individuals[k.parent_id].infection_time for k in kids]
        @test minimum(gi) > 4.0   # latent 4 + a positive contact interval
    end

    @testset "a window whose from-state is never reached produces no contacts" begin
        # `from = :died` but no death transition, so `:died_time` is never
        # set and the window never opens: the outbreak cannot spread. The
        # model warns about the unreachable from-state at construction.
        m = @test_logs (:warn,) match_mode=:any BranchingProcess(
            Infectiousness((rng, ind) -> 5; from = :died, kernel = Exponential(1.0)))
        s = simulate(m; n_initial = 10, max_generations = 3,
            rng = StableRNG(2))
        @test s.cumulative_cases == 10
    end

    @testset "two windows each contribute offspring" begin
        w1 = Infectiousness((rng, ind) -> 2; from = :infection)  # default kernel: no timing
        w2 = Infectiousness((rng, ind) -> 3; from = :infection)
        m = BranchingProcess(w1, w2)
        s = simulate(m; n_initial = 1, max_generations = 1,
            rng = StableRNG(3))
        kids = filter(i -> i.parent_id != 0, s.individuals)
        @test length(kids) == 5   # 2 + 3 from the single index
    end

    @testset "until censors contacts after the infector is removed" begin
        off = (rng, ind) -> 10
        kernel = Exponential(1.0)
        win = Infectiousness(off; from = :infection, until = (:recovered,), kernel = kernel)
        opts = (; n_initial = 5, max_generations = 1)

        # Recovery far in the future: no contact is censored.
        rec_late = Transition(:recovered, from = :infection, delay = (rng, ind) -> 100.0)
        s_late = simulate(ModelSpec(BranchingProcess(win); progression = [rec_late]);
            opts..., rng = StableRNG(1))
        kids_late = count(i -> i.parent_id != 0 && is_infected(i), s_late.individuals)
        @test kids_late == 50   # 5 index × 10, nothing removed early

        # Near-instant recovery: almost every contact lands after removal.
        rec_early = Transition(:recovered, from = :infection, delay = (rng, ind) -> 0.001)
        s_early = simulate(ModelSpec(BranchingProcess(win); progression = [rec_early]);
            opts..., rng = StableRNG(1))
        kids_early = count(i -> i.parent_id != 0 && is_infected(i), s_early.individuals)
        @test kids_early < 5
        @test kids_early < kids_late
    end

    @testset "analytics: single window closed-form, multi-window directs to simulation" begin
        single = BranchingProcess(NegBin(2.0, 0.5), LogNormal(1.6, 0.5))
        @test single_type_offspring(single) isa NegativeBinomial
        @test extinction_probability(single) isa Real

        multi = BranchingProcess(
            Infectiousness(NegBin(1.5, 0.5); from = :infection),
            Infectiousness(NegBin(0.8, 0.5); from = :infection))
        @test_throws ArgumentError single_type_offspring(multi)
        @test_throws ArgumentError extinction_probability(multi)
    end
end
