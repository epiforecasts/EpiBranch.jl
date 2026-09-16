@testset "HomogeneousProcess (Sellke fixed pool)" begin
    @testset "deterministic final size (major outbreaks)" begin
        # With β = 2 and mean infectious period 1, R0 = β·E[T] = 2; the
        # deterministic attack rate solves z = 1 - exp(-R0·z), z ≈ 0.7968.
        # Conditioning on major outbreaks, the mean should match.
        N = 3000
        m = ModelSpec(HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = [Transition(:recovered; from = :infection,
                delay = Exponential(1.0), terminal = true)])
        finals = [simulate(m; rng = StableRNG(s), n_initial = 5).cumulative_cases
                  for s in 1:40]
        z = 0.7968
        major = filter(x -> x > 0.3 * N, finals)
        @test length(major) > 20                 # most seeds take off at R0 = 2
        @test all(x -> x <= N, finals)            # final size never exceeds the pool
        @test isapprox(mean(major) / N, z; atol = 0.03)
    end

    @testset "sub-critical outbreaks stay small" begin
        # β = 0.5 with mean infectious period 1 gives R0 = 0.5 < 1.
        N = 2000
        m = ModelSpec(HomogeneousProcess(; transmission_rate = 0.5, population_size = N);
            progression = [Transition(:recovered; from = :infection, delay = 1.0,
                terminal = true)])
        finals = [simulate(m; rng = StableRNG(s), n_initial = 1).cumulative_cases
                  for s in 1:100]
        @test mean(finals) < 0.1 * N
    end

    @testset "saturation infects the whole pool" begin
        N = 200
        m = ModelSpec(HomogeneousProcess(; transmission_rate = 100.0, population_size = N);
            progression = [Transition(:recovered; from = :infection, delay = 1.0,
                terminal = true)])
        finals = [simulate(m; rng = StableRNG(s), n_initial = 1).cumulative_cases
                  for s in 1:20]
        @test mean(finals) > 0.98 * N
        @test all(x -> x <= N, finals)
    end

    @testset "depletion is real (final size ≤ N)" begin
        N = 300
        m = ModelSpec(HomogeneousProcess(; transmission_rate = 5.0, population_size = N);
            progression = [Transition(:recovered; from = :infection, delay = 1.0,
                terminal = true)])
        for s in 1:20
            @test simulate(m; rng = StableRNG(s), n_initial = 2).cumulative_cases <= N
        end
    end

    @testset "transmission_traits susceptibility and infectiousness scale the pool" begin
        # The pool's force is a built-in engine seam (`_sellke_pool!`), not a
        # model-specific one, so `transmission_traits` should bear on it exactly
        # as it does on the generation-based engine's default risk sources.
        N = 2000
        build(attrs) = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = [Transition(:recovered; from = :infection,
                delay = Exponential(1.0), terminal = true)],
            attributes = attrs)
        finals(attrs) = [simulate(build(attrs); rng = StableRNG(s), n_initial = 5).cumulative_cases
                          for s in 1:20]
        baseline = mean(finals(NoAttributes()))
        @test mean(finals(transmission_traits(susceptibility = 0.3))) < baseline
        @test mean(finals(transmission_traits(infectiousness = 0.3))) < baseline

        # A trait of exactly zero blocks spread past the index cases entirely:
        # no susceptible ever crosses an infinite threshold, and an all-zero
        # infectiousness pool exerts no force at all.
        none_susceptible = simulate(build(transmission_traits(susceptibility = 0.0));
            rng = StableRNG(1), n_initial = 5)
        @test none_susceptible.cumulative_cases == 5
        none_infectious = simulate(build(transmission_traits(infectiousness = 0.0));
            rng = StableRNG(1), n_initial = 5)
        @test none_infectious.cumulative_cases == 5
    end

    @testset "isolation shortens the outbreak" begin
        N = 1000
        base = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = [Transition(:recovered; from = :infection,
                delay = Exponential(1.0), terminal = true)])
        # An early isolation transition closes the infectious window (`:isolated`
        # is in the default `until`), curtailing spread.
        iso = ModelSpec(HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = [
                Transition(:recovered; from = :infection,
                    delay = Exponential(1.0), terminal = true),
                Transition(:isolated; from = :infection, delay = (rng, ind) -> 0.1)
            ])
        base_mean = mean(simulate(base; rng = StableRNG(s), n_initial = 3).cumulative_cases
        for s in 1:30)
        iso_mean = mean(simulate(iso; rng = StableRNG(s), n_initial = 3).cumulative_cases
        for s in 1:30)
        @test iso_mean < base_mean
    end

    @testset "Isolation intervention shortens the outbreak" begin
        # The Isolation *intervention* (not a Transition) must act on the pool:
        # its resolve_individual! runs in the Sellke loop and its isolation time
        # closes the infectious window. Onset comes from a progression transition
        # so it is anchored on the real infection time.
        N = 1000
        prog = [
            Transition(:onset; from = :infection, delay = 0.1),
            Transition(:recovered; from = :infection,
                delay = Exponential(1.0), terminal = true)
        ]
        base = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = prog)
        iso = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = prog,
            interventions = [Isolation(onset_to_isolation_delay = Exponential(0.1))])

        base_mean = mean(simulate(base; rng = StableRNG(s), n_initial = 3).cumulative_cases
        for s in 1:20)
        iso_mean = mean(simulate(iso; rng = StableRNG(s), n_initial = 3).cumulative_cases
        for s in 1:20)
        # Fast isolation pushes R below 1: the outbreak is curtailed.
        @test iso_mean < 0.1 * base_mean

        # A case's isolation time is actually written (the hook ran).
        state = simulate(iso; rng = StableRNG(1), n_initial = 3)
        @test any(isfinite(EpiBranch.isolation_time(ind)) for ind in state.individuals)
    end

    @testset "Unhonoured intervention warns rather than silently ignoring" begin
        prog = [Transition(:recovered; from = :infection,
            delay = Exponential(1.0), terminal = true)]
        ct = ModelSpec(
            HomogeneousProcess(; transmission_rate = 1.5, population_size = 200);
            progression = prog,
            interventions = [ContactTracing(probability = 0.5,
                isolation_to_trace_delay = Exponential(1.0))])
        @test_logs (:warn, r"does not honour"i) match_mode=:any simulate(
            ct; rng = StableRNG(1), n_initial = 2)

        # Leaky isolation (residual transmission) also can't be expressed as a
        # window close, so it warns too — only perfect isolation is honoured.
        leaky = ModelSpec(
            HomogeneousProcess(; transmission_rate = 1.5, population_size = 200);
            progression = prog,
            interventions = [Isolation(onset_to_isolation_delay = Exponential(0.5),
                post_isolation_transmission = 0.5)])
        @test_logs (:warn, r"does not honour"i) match_mode=:any simulate(
            leaky; rng = StableRNG(1), n_initial = 2)

        # The warning holds: the process's default `until` lists `:isolated`,
        # and isolating a case must not close its window through that state.
        pool = HomogeneousProcess(; transmission_rate = 2.0, population_size = 500)
        onsets = clinical_presentation(incubation_period = LogNormal(-1.0, 0.3),
            prob_asymptomatic = 0.0)
        nearly_leaky = [Isolation(onset_to_isolation_delay = Exponential(0.5),
            post_isolation_transmission = 0.9)]
        mean_size(ivs) = sum(
            simulate(
                ModelSpec(pool; progression = prog, interventions = ivs,
                    attributes = onsets);
                rng = StableRNG(s), n_initial = 5).cumulative_cases
        for s in 1:20) / 20
        base_mean = mean_size(AbstractIntervention[])
        leaky_mean = @test_logs (:warn, r"does not honour"i) match_mode=:any mean_size(nearly_leaky)
        @test leaky_mean > 0.8 * base_mean
    end

    @testset "Scheduled interventions gate on the running clock on the pool" begin
        # The loop exposes each case's infection time as the running clock, so a
        # Scheduled(Isolation; start_time) isolates only cases infected at/after
        # start_time — curtailing when it activates, and gating by time.
        N = 1000
        prog = [
            Transition(:onset; from = :infection, delay = 0.1),
            Transition(:recovered; from = :infection,
                delay = Exponential(1.0), terminal = true)
        ]
        mk(start_time) = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = prog,
            interventions = [Scheduled(
                Isolation(onset_to_isolation_delay = Exponential(0.1)); start_time)])
        mean_cc(m) = mean(simulate(m; rng = StableRNG(s), n_initial = 3).cumulative_cases
        for s in 1:15)

        base = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = prog)
        # Active from t = 0: curtails hard. Start far past the outbreak: barely.
        @test mean_cc(mk(0.0)) < 0.1 * mean_cc(base)
        @test mean_cc(mk(1000.0)) > 0.5 * mean_cc(base)

        # Mid start_time gates on the case's own infection time.
        st = simulate(mk(3.0); rng = StableRNG(1), n_initial = 3)
        @test !any(ind -> ind.infection_time < 3.0 && is_isolated(ind), st.individuals)
        @test any(ind -> ind.infection_time >= 3.0 && is_isolated(ind), st.individuals)

        # A Scheduled wrapping an unhonoured intervention still warns.
        sched_ct = ModelSpec(
            HomogeneousProcess(; transmission_rate = 1.5, population_size = 200);
            progression = prog,
            interventions = [Scheduled(
                ContactTracing(probability = 0.5,
                    isolation_to_trace_delay = Exponential(1.0)); start_time = 5.0)])
        @test_logs (:warn, r"does not honour"i) match_mode=:any simulate(
            sched_ct; rng = StableRNG(1), n_initial = 2)
    end

    @testset "removal before infectious onset never infects" begin
        # A latent period opens the window at :infectious, but isolation fires
        # first (close_t <= open_t). Such a case is never infectious: it must be
        # skipped rather than pop a close event against an id never made infectious.
        N = 100
        progression = [
            Transition(:infectious; from = :infection, delay = (rng, ind) -> 5.0),
            Transition(:recovered; from = :infectious,
                delay = (rng, ind) -> 1.0, terminal = true),
            Transition(:isolated; from = :infection, delay = (rng, ind) -> 0.1)
        ]
        # The infectious window is derived to open at :infectious (a latent period
        # produces it).
        @test EpiBranch._resolve_infectious_from(nothing, progression) === :infectious
        m = ModelSpec(HomogeneousProcess(; transmission_rate = 5.0, population_size = N);
            progression = progression)
        # No index case reaches :infectious, so no secondary transmission occurs
        # and the run completes with only the seeds infected.
        state = simulate(m; rng = StableRNG(1), n_initial = 5)
        @test state.cumulative_cases == 5
    end

    @testset "onset is measured from each case's own infection time" begin
        # Pool members are created, and their incubation periods drawn, before
        # the pool sets their infection times. Isolation depends on onset, so
        # onset must be counted from the time each case was infected.
        clinical = clinical_presentation(incubation_period = LogNormal(1.0, 0.3))
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0),
            test_sensitivity = 1.0)
        m = ModelSpec(HomogeneousProcess(; transmission_rate = 1.0, population_size = 500);
            progression = [Transition(:recovered; from = :infection, delay = 10.0,
                terminal = true)],
            attributes = clinical, interventions = [iso])
        state = simulate(m; n_initial = 1, rng = StableRNG(3))
        secondary = [ind
                     for ind in state.individuals
                     if is_infected(ind) && ind.parent_id != 0]
        @test !isempty(secondary)
        @test all(onset_time(ind) >= ind.infection_time for ind in secondary)
        @test all(onset_time(ind) - ind.infection_time ≈ ind.state[:incubation_period]
        for ind in secondary)
        @test all(isolation_time(ind) >= onset_time(ind) for ind in secondary)
    end

    @testset "line list and timing" begin
        progression = [
            Transition(:infectious; from = :infection, delay = Exponential(1.0)),
            Transition(:recovered; from = :infectious, delay = Exponential(2.0),
                terminal = true)
        ]
        @test EpiBranch._resolve_infectious_from(nothing, progression) === :infectious
        m = ModelSpec(HomogeneousProcess(; transmission_rate = 2.0, population_size = 500);
            progression = progression)
        state = simulate(m; rng = StableRNG(1), n_initial = 5)
        ll = linelist(state)
        @test size(ll, 1) == state.cumulative_cases
        @test EpiBranch._timetype(state) === Float64
        @test :date_infectious in propertynames(ll)
        @test :date_recovered in propertynames(ll)
    end

    @testset "transmission_rate is the transmission parameter" begin
        # β is the per-infective rate, stored and used directly (no R0 map).
        p = HomogeneousProcess(; transmission_rate = 2.0, population_size = 100)
        @test p.transmission_rate == 2.0
        # `transmission_rate` is required.
        @test_throws UndefKeywordError HomogeneousProcess(; population_size = 10)
        # β must be finite and non-negative; negative, infinite and NaN rates
        # are rejected before they reach the Sellke hazard.
        @test_throws ArgumentError HomogeneousProcess(; transmission_rate = -1.0,
            population_size = 10)
        @test_throws ArgumentError HomogeneousProcess(; transmission_rate = Inf,
            population_size = 10)
        @test_throws ArgumentError HomogeneousProcess(; transmission_rate = NaN,
            population_size = 10)
        # β = 0 (no transmission) is a valid degenerate model.
        @test HomogeneousProcess(; transmission_rate = 0.0,
            population_size = 10).transmission_rate == 0.0
        # A non-positive population is rejected at construction.
        @test_throws ArgumentError HomogeneousProcess(; transmission_rate = 2.0,
            population_size = 0)
    end

    # A small helper that runs the structured pool directly: tag a real
    # attribute (`:age_band`) on a fixed population of size N via `band_of`, name
    # it as the mixing attribute with `mixing_by = (:age_band,)`, supply a force
    # keyed on the band value, and return the band each infected case fell in.
    function _run_pool(N, band_of, force; n_initial = 5, rng)
        process = HomogeneousProcess(; transmission_rate = 1.0, population_size = N)
        prog = [Transition(:recovered; from = :infection, delay = Exponential(1.0),
            terminal = true)]
        state = EpiBranch.new_state(process, prog, NoAttributes(), rng)
        EpiBranch.add_individuals!(state, N, AbstractIntervention[];
            setup = (ind, i) -> (ind.state[:age_band] = band_of(ind)))
        EpiBranch._sellke_pool!(state, collect(1:N), rng;
            mixing_by = (:age_band,), force = force,
            n_initial = n_initial, from = :infection,
            until = (:recovered, :died, :isolated))
        return [ind.state[:age_band]
                for ind in state.individuals
                if get(ind.state, :infected, false)]
    end

    @testset "two-band uniform matrix reduces to one pool" begin
        # Two bands with uniform contact behave as a single pool of size N: the
        # force felt is β/N·(total infectious) regardless of band, so the
        # major-outbreak attack rate matches the homogeneous law (z ≈ 0.7968 at
        # R0 = 2, since β = 2 and mean infectious period = 1).
        N = 3000
        β = 2.0
        band_of = ind -> (ind.id <= N ÷ 2 ? 1 : 2)
        force = (type, counts) -> β / N * sum(values(counts))
        finals = [length(_run_pool(N, band_of, force; rng = StableRNG(s)))
                  for s in 1:40]
        major = filter(x -> x > 0.3 * N, finals)
        @test length(major) > 20
        @test all(x -> x <= N, finals)
        @test isapprox(mean(major) / N, 0.7968; atol = 0.03)
    end

    @testset "asymmetric mixing orders attack rates" begin
        # A 2×2 contact matrix keyed by band, where band 1 mixes far more than
        # band 2. With equal band sizes, the force on a band-`b` susceptible is
        # (1/half)·Σ_h M[b,h]·counts[(h,)]. The high-contact band should suffer a
        # strictly higher attack rate than the low-contact band, over replicates.
        N = 2000
        half = N ÷ 2
        M = [3.0 0.5; 0.5 0.5]         # band 1 mixes much more than band 2
        band_of = ind -> (ind.id <= half ? 1 : 2)
        force = (type, counts) -> begin
            b = type[1]
            sum(M[b, h] * get(counts, (h,), 0) for h in 1:2) / half
        end
        ar1 = Float64[]
        ar2 = Float64[]
        for s in 1:40
            bands = _run_pool(N, band_of, force; n_initial = 10,
                rng = StableRNG(s))
            n1 = count(==(1), bands)
            n2 = count(==(2), bands)
            # Keep major outbreaks only, so the ordering is about who is hit hardest.
            (n1 + n2) > 0.3 * N || continue
            push!(ar1, n1 / half)
            push!(ar2, n2 / half)
        end
        @test length(ar1) > 20
        @test mean(ar1) > mean(ar2)
    end

    @testset "positive force with empty infectious pool is index-labelled" begin
        # A custom force with a count-independent positive hazard (external
        # importation) keeps firing infections even when no one is infectious. The
        # first infection draws its source from an empty pool: without a guard that
        # throws; with the guard it falls back to the index-case label 0.
        N = 50
        process = HomogeneousProcess(; transmission_rate = 1.0, population_size = N)
        prog = [Transition(:recovered; from = :infection, delay = Exponential(1.0),
            terminal = true)]
        rng = StableRNG(1)
        state = EpiBranch.new_state(process, prog, NoAttributes(), rng)
        EpiBranch.add_individuals!(state, N, AbstractIntervention[])
        EpiBranch._sellke_pool!(state, collect(1:N), rng; mixing_by = (),
            force = (type, counts) -> 0.5, n_initial = 0,
            from = :infection, until = (:recovered, :died, :isolated))
        @test count(ind -> get(ind.state, :infected, false), state.individuals) > 0
    end

    @testset "conditioned simulation and show" begin
        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        spec = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 500);
            progression = prog)
        # `condition` retries until the final size falls in the range
        state = simulate(spec; condition = 100:500, n_initial = 5, rng = StableRNG(1))
        @test state.cumulative_cases in 100:500

        # show renders the β form
        @test occursin("β=",
            repr(HomogeneousProcess(; transmission_rate = 2.0, population_size = 10)))
    end

    @testset "termination controls warn on the fixed pool" begin
        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        spec = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 200);
            progression = prog)
        # A set termination control has no effect on the extinction-run pool, so
        # `simulate` warns rather than silently ignoring it.
        @test_logs (:warn, r"ignores termination controls") simulate(
            spec; n_initial = 3, max_cases = 50, rng = StableRNG(1))
        # No termination keyword set → no warning.
        @test_logs simulate(spec; n_initial = 3, rng = StableRNG(1))
        # The trait itself: the pool ignores the controls, the generation engine
        # honours them.
        @test !EpiBranch._honours_termination_controls(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 10))
        @test EpiBranch._honours_termination_controls(
            BranchingProcess(Poisson(1.5), Exponential(2.0)))
    end
end
