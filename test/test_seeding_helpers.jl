using Test
using Distributions
using StableRNGs
using EpiBranch: _seed_candidates!, _seed_external!, _check_obs_end,
    _check_finite_obs_end, _populate!

# Stand-in for a simulation state: every member is already realised with unit
# susceptibility.
struct SeedState end
EpiBranch._realised!(::SeedState, id::Int) = (; susceptibility = 1.0)

@testset "Seeding helpers" begin
    members = collect(1:6)
    state = SeedState()
    place_at_first!(best, members, rng) = (best[1] = 0.0; nothing)

    @testset "_check_obs_end" begin
        @test _check_obs_end(30) === 30.0
        @test _check_obs_end(Inf) == Inf
        @test_throws ArgumentError _check_obs_end(-1)
        @test_throws ArgumentError _check_obs_end(NaN)
    end

    @testset "_check_finite_obs_end" begin
        @test _check_finite_obs_end(0.1, 30.0, "everyone") === nothing
        @test _check_finite_obs_end(0.0, Inf, "everyone") === nothing
        @test_throws ArgumentError _check_finite_obs_end(0.1, Inf, "everyone")
    end

    @testset "_seed_external!" begin
        best = fill(Inf, 6)
        _seed_external!(best, members, state, Exponential(1.0), 1.0e6, StableRNG(1))
        @test all(isfinite, best)

        # Skipped members and draws beyond the window stay unseeded.
        best = fill(Inf, 6)
        _seed_external!(
            best, members, state, Exponential(1.0), 1.0e6, StableRNG(1), (2, 3)
        )
        @test isinf(best[2]) && isinf(best[3])
        @test isfinite(best[1])

        best = fill(Inf, 6)
        _seed_external!(best, members, state, Exponential(1.0), -1.0, StableRNG(1))
        @test all(isinf, best)
    end

    @testset "_seed_candidates!" begin
        # Chosen cases at time 0 and no hazard: nothing else is seeded.
        best = fill(Inf, 6)
        _seed_candidates!(
            place_at_first!, best, members, state, 0.0, 10.0, StableRNG(1), [2, 4]
        )
        @test best[2] == 0 && best[4] == 0
        @test count(isfinite, best) == 2

        # A set of chosen cases and a hazard: the rest are drawn.
        best = fill(Inf, 6)
        _seed_candidates!(
            place_at_first!, best, members, state, Exponential(1.0), 1.0e6,
            StableRNG(1), Set([2])
        )
        @test best[2] == 0
        @test all(isfinite, best)

        # No chosen cases: the hazard alone seeds the table.
        best = fill(Inf, 6)
        _seed_candidates!(
            place_at_first!, best, members, state, Exponential(1.0), 1.0e6,
            StableRNG(1), nothing
        )
        @test all(isfinite, best)

        # Neither: the structure places the index cases itself.
        best = fill(Inf, 6)
        _seed_candidates!(
            place_at_first!, best, members, state, 0.0, 10.0, StableRNG(1), nothing
        )
        @test best[1] == 0.0
        @test count(isfinite, best) == 1
    end
end
