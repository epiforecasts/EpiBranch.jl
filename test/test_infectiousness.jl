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
        m = ModelSpec(
            BranchingProcess(
                Infectiousness(
                    (rng, ind) -> 3;
                    from = :infectious, kernel = Exponential(0.5)
                )
            );
            progression = [latent]
        )
        s = simulate(
            m;
            n_initial = 30, max_generations = 1, rng = StableRNG(1)
        )
        kids = filter(i -> i.parent_id != 0 && is_infected(i), s.individuals)
        gi = [k.infection_time - s.individuals[k.parent_id].infection_time for k in kids]
        @test minimum(gi) > 4.0   # latent 4 + a positive contact interval
    end

    @testset "a deep ring reaching a case before its window opens leaves its offspring" begin
        # Every case has three contacts, so tracing that does not isolate
        # leaves the size of each generation fixed at any ring depth.
        latent = Transition(:infectious, from = :infection, delay = (rng, ind) -> 4.0)
        m = BranchingProcess(
            Infectiousness((rng, ind) -> 3; from = :infectious, kernel = Exponential(0.5))
        )
        for depth in (1, 2)
            ct = ContactTracing(
                TraceEveryone(), 1.0, Exponential(1000.0), FlagOnly(); depth = depth
            )
            s = simulate(
                ModelSpec(m; progression = [latent], interventions = [ct]);
                n_initial = 5, max_generations = 2, rng = StableRNG(1)
            )
            @test s.cumulative_cases == 5 + 15 + 45
        end
    end

    @testset "a window whose from-state is never reached produces no contacts" begin
        # `from = :died` but no death transition, so `:died_time` is never
        # set and the window never opens: the outbreak cannot spread.
        # Composing the process with a progression warns about the
        # unreachable from-state.
        m = @test_logs (:warn,) match_mode = :any ModelSpec(
            BranchingProcess(
                Infectiousness((rng, ind) -> 5; from = :died, kernel = Exponential(1.0))
            )
        )
        s = simulate(
            m; n_initial = 10, max_generations = 3,
            rng = StableRNG(2)
        )
        @test s.cumulative_cases == 10
    end

    @testset "two windows each contribute offspring" begin
        w1 = Infectiousness((rng, ind) -> 2; from = :infection)  # default kernel: no timing
        w2 = Infectiousness((rng, ind) -> 3; from = :infection)
        m = BranchingProcess(w1, w2)
        s = simulate(
            m; n_initial = 1, max_generations = 1,
            rng = StableRNG(3)
        )
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
        s_late = simulate(
            ModelSpec(BranchingProcess(win); progression = [rec_late]);
            opts..., rng = StableRNG(1)
        )
        kids_late = count(i -> i.parent_id != 0 && is_infected(i), s_late.individuals)
        @test kids_late == 50   # 5 index × 10, nothing removed early

        # Near-instant recovery: almost every contact lands after removal.
        rec_early = Transition(:recovered, from = :infection, delay = (rng, ind) -> 0.001)
        s_early = simulate(
            ModelSpec(BranchingProcess(win); progression = [rec_early]);
            opts..., rng = StableRNG(1)
        )
        kids_early = count(i -> i.parent_id != 0 && is_infected(i), s_early.individuals)
        @test kids_early < 5
        @test kids_early < kids_late
    end

    @testset "chain-size likelihood routes a window with until to simulation" begin
        # Transmission stops at infection (recovery is immediate), so every
        # cluster is a singleton; the bare offspring law `until` leaves
        # unscored would give log(pdf(Poisson(0.8), 1)) ≈ -0.8 instead.
        w = Infectiousness(Poisson(0.8); until = (:recovered,), kernel = Exponential(1.0))
        spec = ModelSpec(
            BranchingProcess(w);
            progression = [
                Transition(:recovered, from = :infection, delay = 0.0, terminal = true),
            ]
        )
        @test !EpiBranch.offspring_law_unchanged(spec.process)
        ll = loglikelihood(ChainSizes([1]), spec; rng = StableRNG(1), n_sim = 500)
        @test isapprox(ll, 0.0; atol = 1.0e-9)
    end

    @testset "analytics: single window closed-form, multi-window directs to simulation" begin
        single = BranchingProcess(NegBin(2.0, 0.5), LogNormal(1.6, 0.5))
        @test single_type_offspring(single) isa NegativeBinomial
        @test extinction_probability(single) isa Real

        multi = BranchingProcess(
            Infectiousness(NegBin(1.5, 0.5); from = :infection),
            Infectiousness(NegBin(0.8, 0.5); from = :infection)
        )
        @test_throws ArgumentError single_type_offspring(multi)
        @test_throws ArgumentError extinction_probability(multi)
    end
end

@testset "A deep ring leaves a case's settled state to its offspring" begin
    # Severe cases make three contacts and others one. Tracing that changes
    # nothing must leave every case with the count its own severity implies,
    # at any ring depth.
    progression = [
        Transition(:severe; from = :infection, delay = 0.0, probability = 0.5),
        Transition(:recovered; from = :infection, delay = 20.0),
    ]
    severe(ind) = isfinite(get(ind.state, :severe_time, Inf))
    m = BranchingProcess(
        Infectiousness((rng, ind) -> severe(ind) ? 3 : 1; kernel = Exponential(2.0))
    )
    for depth in (1, 2)
        ct = ContactTracing(
            TraceEveryone(), 1.0, Exponential(1000.0), FlagOnly(); depth = depth
        )
        wrong = 0
        for seed in 1:20
            s = simulate(
                ModelSpec(m; progression, interventions = [ct]);
                max_generations = 4, max_cases = nothing, rng = StableRNG(seed)
            )
            for ind in s.individuals
                is_infected(ind) && ind.generation < 4 || continue
                drawn = count(
                    c -> !isnan(s.individuals[c].infection_time), ind.secondary_case_ids
                )
                drawn == (severe(ind) ? 3 : 1) || (wrong += 1)
            end
        end
        @test wrong == 0
    end
end

@testset "A deep ring traces the offspring of an infected member that cannot seed" begin
    # An asymptomatic ring member is not eligible under `OnSymptomOnset`, so
    # its own contacts are reached only by the ring growing through it, with
    # the offspring a distribution, a function, or one window of each.
    attributes = clinical_presentation(
        incubation_period = Exponential(1.0), prob_asymptomatic = 0.5
    )
    ct = ContactTracing(
        OnSymptomOnset(), 1.0, Exponential(1000.0), FlagOnly(); depth = 2
    )
    windows = (
        (Infectiousness(Dirac(2); kernel = Exponential(2.0)),),
        (Infectiousness((rng, ind) -> 2; kernel = Exponential(2.0)),),
        (
            Infectiousness(Dirac(2); kernel = Exponential(2.0)),
            Infectiousness((rng, ind) -> 2; kernel = Exponential(2.0)),
        ),
    )
    for w in windows
        m = BranchingProcess(w...)
        reached = 0
        traced = 0
        for seed in 1:30
            s = simulate(
                ModelSpec(m; interventions = [ct], attributes);
                max_generations = 4, max_cases = nothing, rng = StableRNG(seed)
            )
            for ind in s.individuals
                is_infected(ind) && ind.parent_id != 0 || continue
                parent = s.individuals[ind.parent_id]
                parent.parent_id == 0 && continue
                grandparent = s.individuals[parent.parent_id]
                is_traced(parent) && is_asymptomatic(parent) &&
                    !is_asymptomatic(grandparent) || continue
                reached += 1
                is_traced(ind) && (traced += 1)
            end
        end
        @test reached > 0
        @test traced == reached
    end
end

@testset "A deep ring traces a member's contacts from the earlier of ring and trigger" begin
    # Whether the offspring is a distribution, whose contacts are drawn as soon
    # as the ring reaches a member, or a function, whose contacts are drawn
    # only once the member settles, each contact of a traced member is traced
    # from whichever came first: the ring reaching the member, or the member's
    # own trigger as an eligible case.
    attributes = clinical_presentation(
        incubation_period = Exponential(2.0), prob_asymptomatic = 0.0
    )
    iso = Isolation(onset_to_isolation_delay = Dirac(4.0), duration = Inf)
    ct = ContactTracing(
        probability = 1.0, isolation_to_trace_delay = Dirac(0.5),
        action = FlagOnly(), depth = 2
    )
    for offspring in (Dirac(2), (rng, ind) -> 2)
        m = ModelSpec(
            BranchingProcess(offspring, Exponential(3.0));
            interventions = [iso, ct], attributes
        )
        checked = 0
        wrong = 0
        for seed in 1:10
            s = simulate(m; max_generations = 4, max_cases = nothing, rng = StableRNG(seed))
            for c in s.individuals
                c.parent_id == 0 && continue
                b = s.individuals[c.parent_id]
                b.parent_id != 0 && is_traced(b) && is_infected(b) || continue
                haskey(c.state, :trace_time) || continue
                trigger = EpiBranch.trigger_time(ct.eligibility, b, c, s)
                expected = min(b.state[:trace_time], trigger) + 0.5
                checked += 1
                c.state[:trace_time] ≈ expected || (wrong += 1)
            end
        end
        @test checked > 0
        @test wrong == 0
    end
end
