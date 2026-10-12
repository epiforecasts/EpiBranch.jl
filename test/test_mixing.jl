using Distributions
using StableRNGs

@testset "MixingProcess (mass action between mixing groups)" begin
    @testset "construction and show" begin
        rate = (from, to) -> 0.0
        m = MixingProcess(; population_size = 100, rate)
        @test m isa MixingProcess
        @test m.mixing_by == ()
        @test m.from === nothing
        @test m.until == (:recovered, :died, :isolated)
        @test EpiBranch.population_size(m) == 100

        @test_throws ArgumentError MixingProcess(; population_size = 0, rate)

        shown = repr(MixingProcess(; population_size = 10, mixing_by = (:age_band,), rate))
        @test occursin("MixingProcess", shown)
        @test occursin("age_band", shown)
    end

    @testset "mixing_by = () reproduces HomogeneousProcess exactly" begin
        # The homogeneous model is the one-group special case of structured
        # mixing, called with the same pair rate, population and seed: the two
        # must agree case for case, not just in distribution.
        N = 500
        β = 2.0
        prog = [Transition(:recovered; from = :infection, delay = Exponential(1.0), terminal = true)]
        rate = (from, to) -> β / N

        homog = simulate(
            ModelSpec(
                HomogeneousProcess(; transmission_rate = β, population_size = N);
                progression = prog
            );
            n_initial = 5, rng = StableRNG(1)
        )
        mixed = simulate(
            ModelSpec(MixingProcess(; population_size = N, rate); progression = prog);
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
        # rate; the race engine's own statistics are tested in
        # test_homogeneous.jl, this only checks MixingProcess wires attributes
        # and rate through to it.
        N = 2000
        M = [3.0 0.5; 0.5 0.5]
        band = (rng, ind) -> (ind.state[:age_band] = ind.id <= N ÷ 2 ? 1 : 2)
        rate = (from, to) -> M[to[1], from[1]] / (N ÷ 2)
        prog = [Transition(:recovered; from = :infection, delay = Exponential(1.0), terminal = true)]
        state = simulate(
            ModelSpec(
                MixingProcess(; population_size = N, mixing_by = (:age_band,), rate);
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
        rate = (from, to) -> 0.0
        # With no latent transition the window opens at :infection.
        sir = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        @test EpiBranch.infectious_from(sir) === :infection
        # A latent transition moves the window's open state to :infectious.
        seir = [
            Transition(:infectious; from = :infection, delay = 1.0),
            Transition(:recovered; from = :infectious, delay = 1.0, terminal = true),
        ]
        @test EpiBranch.infectious_from(seir) === :infectious
        # An explicit `from` overrides the derivation.
        m = MixingProcess(; population_size = 10, rate, from = :infection)
        @test m.from === :infection
    end

    @testset "termination controls warn on the structured population" begin
        rate = (from, to) -> 2.0 / 200
        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        spec = ModelSpec(MixingProcess(; population_size = 200, rate); progression = prog)
        @test_logs (:warn, r"ignores the other termination controls") simulate(
            spec; n_initial = 3, max_cases = 50, rng = StableRNG(1)
        )
        @test_logs simulate(spec; n_initial = 3, rng = StableRNG(1))
        @test !EpiBranch._honours_termination_controls(
            MixingProcess(; population_size = 10, rate)
        )
    end

    @testset "n_initial is validated" begin
        rate = (from, to) -> 0.0
        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        spec = ModelSpec(MixingProcess(; population_size = 10, rate); progression = prog)
        @test_throws ArgumentError simulate(spec; n_initial = 0, rng = StableRNG(1))
        @test_throws ArgumentError simulate(spec; n_initial = 11, rng = StableRNG(1))
    end

    @testset "condition retries through the public simulate_once seam" begin
        rate = (from, to) -> 2.0 / 500
        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        spec = ModelSpec(MixingProcess(; population_size = 500, rate); progression = prog)
        state = simulate(spec; condition = 100:500, n_initial = 5, rng = StableRNG(1))
        @test state.cumulative_cases in 100:500
    end
end
