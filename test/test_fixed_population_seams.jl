using Test
using EpiBranch
using Distributions
using StableRNGs

# A fixed-population structure written outside the package, naming its window
# fields differently from the built-in processes.
struct _RingProcess <: EpiBranch.FixedPopulationProcess
    n::Int
    opens::Union{Symbol, Nothing}
    closes::Tuple
end
EpiBranch.process_window(p::_RingProcess) = (p.opens, p.closes)

struct _SeededRing <: EpiBranch.FixedPopulationProcess
    n::Int
end
EpiBranch.supports_initial_cases(::_SeededRing) = true
function EpiBranch.validate_initial_cases(m::_SeededRing, opts::SimOpts)
    invoke(
        EpiBranch.validate_initial_cases, Tuple{TransmissionModel, SimOpts}, m, opts
    )
    return EpiBranch.validate_initial_case_ids(opts, m.n)
end

struct _UndeclaredIntervention <: AbstractIntervention end

@testset "Seams for a fixed-population structure" begin
    @testset "names are public" begin
        if VERSION >= v"1.11"
            for name in (
                    :FixedPopulationProcess, :process_window, :supports_initial_cases,
                    :validate_initial_cases, :validate_initial_case_ids,
                    :seed_initial_cases!, :honours_termination_controls,
                    :validate_process_windows, :warn_uncovered_terminal_states,
                    :infection_layer_columns, :layer_host_times,
                    :validate_infection_likelihood, :validate_removal_gap_keys,
                    :REMOVAL_STRETCHES_KEY, :LEAKY_REMOVAL_STRETCHES_KEY,
                )
                @test Base.ispublic(EpiBranch, name)
            end
        else
            @test_skip true
        end
    end

    @testset "defaults of the abstract type" begin
        ring = _RingProcess(10, nothing, (:recovered,))
        @test !EpiBranch.honours_termination_controls(ring)
        @test EpiBranch.honours_termination_controls(
            BranchingProcess(Poisson(1.0), Exponential(1.0))
        )
        for T in (HomogeneousProcess, MixingProcess)
            @test T <: EpiBranch.FixedPopulationProcess
            @test !EpiBranch.honours_termination_controls(
                T === HomogeneousProcess ?
                    HomogeneousProcess(; transmission_rate = 1.0, population_size = 10) :
                    MixingProcess(; population_size = 10, rate = (a, b) -> 0.1)
            )
        end
    end

    @testset "initial_cases" begin
        opts = SimOpts(1, [2, 4], nothing, AbstractStoppingRule[Extinction()])
        none = SimOpts(1, nothing, nothing, AbstractStoppingRule[Extinction()])
        ring = _RingProcess(10, nothing, (:recovered,))
        @test !EpiBranch.supports_initial_cases(ring)
        @test_throws ArgumentError EpiBranch.validate_initial_cases(ring, opts)
        @test EpiBranch.validate_initial_cases(ring, none) === nothing

        seeded = _SeededRing(10)
        @test EpiBranch.supports_initial_cases(seeded)
        @test EpiBranch.validate_initial_cases(seeded, opts) === nothing
        @test_throws ArgumentError EpiBranch.validate_initial_cases(
            _SeededRing(3), opts
        )

        best = fill(Inf, 3)
        EpiBranch.seed_initial_cases!(best, [3, 4, 5], [4, 9])
        @test best == [Inf, 0, Inf]
    end

    @testset "window read through process_window" begin
        progression = [
            Transition(:recovered; from = :infection, rate = 1.0, terminal = true),
            Transition(:died; from = :infection, rate = 0.1, terminal = true),
        ]
        ring = _RingProcess(10, nothing, (:recovered,))
        @test_logs (:warn, r"died") EpiBranch.validate_process_windows(ring, progression)
        covered = _RingProcess(10, nothing, (:recovered, :died))
        @test_logs EpiBranch.validate_process_windows(covered, progression)
    end

    @testset "infection layer checks" begin
        progression = [
            Transition(:recovered; from = :infection, rate = 1.0, terminal = true),
        ]
        spec = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 50);
            progression
        )
        state = simulate(spec; rng = StableRNG(1), n_initial = 2)
        cols = EpiBranch.infection_layer_columns(state, spec)
        @test keys(cols) == (:infection_time, :infectious_time, :removal_time, :is_index)
        @test length(cols.is_index) == length(state.individuals)
        @test count(cols.is_index) == 2
        @test EpiBranch.validate_infection_likelihood(spec) === nothing
        @test EpiBranch.layer_host_times(state, spec, ()) isa NamedTuple

        bad = @test_logs (:warn, r"no effect") match_mode = :any ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 50);
            progression, interventions = [_UndeclaredIntervention()]
        )
        @test_throws ArgumentError EpiBranch.validate_infection_likelihood(bad)
    end
end
