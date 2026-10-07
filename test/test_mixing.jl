@testset "MixingProcess (structured Sellke pool)" begin
    @testset "construction and show" begin
        force = (group, counts) -> 0.0
        m = MixingProcess(; population_size = 100, force)
        @test m isa MixingProcess
        @test m.mixing_by == ()
        @test m.from === nothing
        @test m.until == (:recovered, :died, :isolated)
        @test EpiBranch.population_size(m) == 100

        @test_throws ArgumentError MixingProcess(; population_size = 0, force)

        shown = repr(MixingProcess(; population_size = 10, mixing_by = (:age_band,), force))
        @test occursin("MixingProcess", shown)
        @test occursin("age_band", shown)
    end

    @testset "mixing_by = () reproduces HomogeneousProcess exactly" begin
        # The homogeneous pool is the one-group special case of the structured
        # pool, called with the same `force`, population and seed: the two must
        # agree case for case, not just in distribution.
        N = 500
        β = 2.0
        prog = [Transition(:recovered; from = :infection, delay = Exponential(1.0), terminal = true)]
        force = (group, counts) -> β / N * sum(values(counts))

        homog = simulate(
            ModelSpec(
                HomogeneousProcess(; transmission_rate = β, population_size = N);
                progression = prog
            );
            n_initial = 5, rng = StableRNG(1)
        )
        mixed = simulate(
            ModelSpec(MixingProcess(; population_size = N, force); progression = prog);
            n_initial = 5, rng = StableRNG(1)
        )
        @test mixed.cumulative_cases == homog.cumulative_cases
        @test mixed.extinct == homog.extinct
        @test isequal(
            [i.infection_time for i in mixed.individuals],
            [i.infection_time for i in homog.individuals]
        )
    end

    @testset "asymmetric mixing orders attack rates" begin
        # Band 1 mixes far more than band 2, so it suffers the higher attack
        # rate; the pool engine's own statistics are tested in test_homogeneous.jl,
        # this only checks MixingProcess wires attributes and force through to it.
        N = 2000
        M = [3.0 0.5; 0.5 0.5]
        band = (rng, ind) -> (ind.state[:age_band] = ind.id <= N ÷ 2 ? 1 : 2)
        force = (group, counts) -> begin
            b = group[1]
            sum(M[b, h] * get(counts, (h,), 0) / (N ÷ 2) for h in 1:2)
        end
        prog = [Transition(:recovered; from = :infection, delay = Exponential(1.0), terminal = true)]
        state = simulate(
            ModelSpec(
                MixingProcess(; population_size = N, mixing_by = (:age_band,), force);
                progression = prog, attributes = band
            );
            n_initial = 10, rng = StableRNG(1)
        )
        band1 = [ind for ind in state.individuals if ind.state[:age_band] == 1]
        band2 = [ind for ind in state.individuals if ind.state[:age_band] == 2]
        @test count(is_infected, band1) / length(band1) >
            count(is_infected, band2) / length(band2)
    end

    @testset "from override and derivation from progression" begin
        force = (group, counts) -> 0.0
        # With no latent transition the window opens at :infection.
        sir = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        @test EpiBranch.infectious_from(sir) === :infection
        # A latent transition anchors the window at :infectious.
        seir = [
            Transition(:infectious; from = :infection, delay = 1.0),
            Transition(:recovered; from = :infectious, delay = 1.0, terminal = true),
        ]
        @test EpiBranch.infectious_from(seir) === :infectious
        # An explicit `from` overrides the derivation.
        m = MixingProcess(; population_size = 10, force, from = :infection)
        @test m.from === :infection
    end

    @testset "termination controls warn on the structured pool" begin
        force = (group, counts) -> 2.0 / 200 * sum(values(counts))
        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        spec = ModelSpec(MixingProcess(; population_size = 200, force); progression = prog)
        @test_logs (:warn, r"ignores the other termination controls") simulate(
            spec; n_initial = 3, max_cases = 50, rng = StableRNG(1)
        )
        @test_logs simulate(spec; n_initial = 3, rng = StableRNG(1))
        @test !EpiBranch._honours_termination_controls(
            MixingProcess(; population_size = 10, force)
        )
    end

    @testset "n_initial is validated" begin
        force = (group, counts) -> 0.0
        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        spec = ModelSpec(MixingProcess(; population_size = 10, force); progression = prog)
        @test_throws ArgumentError simulate(spec; n_initial = 0, rng = StableRNG(1))
        @test_throws ArgumentError simulate(spec; n_initial = 11, rng = StableRNG(1))
    end

    @testset "condition retries through the public simulate_once seam" begin
        force = (group, counts) -> 2.0 / 500 * sum(values(counts))
        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        spec = ModelSpec(MixingProcess(; population_size = 500, force); progression = prog)
        state = simulate(spec; condition = 100:500, n_initial = 5, rng = StableRNG(1))
        @test state.cumulative_cases in 100:500
    end
end
