@testset "HomogeneousProcess (Sellke fixed pool)" begin
    @testset "deterministic final size (major outbreaks)" begin
        # For R0 = 2 the deterministic attack rate solves z = 1 - exp(-R0·z),
        # z ≈ 0.7968. Conditioning on major outbreaks, the mean should match.
        N = 3000
        m = ModelSpec(HomogeneousProcess(; R0 = 2.0, population_size = N);
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
        N = 2000
        m = ModelSpec(HomogeneousProcess(; R0 = 0.5, population_size = N);
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

    @testset "line list and timing" begin
        progression = [
            Transition(:infectious; from = :infection, delay = Exponential(1.0)),
            Transition(:recovered; from = :infectious, delay = Exponential(2.0),
                terminal = true)
        ]
        @test EpiBranch._resolve_infectious_from(nothing, progression) === :infectious
        m = ModelSpec(HomogeneousProcess(; R0 = 2.0, population_size = 500);
            progression = progression)
        state = simulate(m; rng = StableRNG(1), n_initial = 5)
        ll = linelist(state)
        @test size(ll, 1) == state.cumulative_cases
        @test EpiBranch._timetype(state) === Float64
        @test :date_infectious in propertynames(ll)
        @test :date_recovered in propertynames(ll)
    end

    @testset "R0 versus transmission_rate" begin
        # β is resolved at simulate time as R0 / mean infectious period; with mean
        # 1.0 an R0 of 2 gives the same β as transmission_rate = 2.
        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        p_r0 = HomogeneousProcess(; R0 = 2.0, population_size = 100)
        p_beta = HomogeneousProcess(; transmission_rate = 2.0, population_size = 100)
        @test EpiBranch._resolve_transmission_rate(p_r0, prog, :infection) == 2.0
        @test EpiBranch._resolve_transmission_rate(p_beta, prog, :infection) == 2.0
        # A distribution mean is used when the infectious period is a Distribution.
        prog_exp = [Transition(:recovered; from = :infection, delay = Exponential(2.0),
            terminal = true)]
        p_dist = HomogeneousProcess(; R0 = 3.0, population_size = 100)
        @test EpiBranch._resolve_transmission_rate(p_dist, prog_exp, :infection) ≈ 3.0 / 2.0

        # Exactly one of transmission_rate / R0 is required.
        @test_throws ArgumentError HomogeneousProcess(; population_size = 10)
        @test_throws ArgumentError HomogeneousProcess(; R0 = 2.0,
            transmission_rate = 2.0, population_size = 10)
        # R0 with no infectious period in the progression cannot resolve β.
        @test_throws ArgumentError simulate(
            ModelSpec(HomogeneousProcess(; R0 = 2.0, population_size = 10));
            rng = StableRNG(1))
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
end
