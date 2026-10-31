using Random
using Distributions
using StableRNGs

# Build a minimal `SimulationState` by hand, with individuals carrying just
# the state keys a test needs, rather than running a full simulation.
function _toy_state(individuals::Vector{<:Individual})
    return SimulationState(
        individuals, Int[], 0, Xoshiro(1), length(individuals), true,
        NoPopulation(), 100.0, NoAttributes(), EpiBranch.AbstractClinicalTransition[]
    )
end

function _case(; onset = NaN, outcome = Inf, infected = true)
    state = Dict{Symbol, Any}(:onset_time => onset, :outcome_time => outcome, :infected => infected)
    return Individual(; id = 1, state)
end

@testset "Signals" begin
    @testset "TransitionCount" begin
        ind1 = _case(; onset = 0.5, outcome = 5.0)   # onset on day 1
        ind2 = _case(; onset = 1.5, outcome = Inf)   # onset on day 2
        ind3 = _case(; onset = NaN)                  # never onsets (asymptomatic)
        ind4 = _case(; onset = 0.5, infected = false) # a contact, not a case
        state = _toy_state([ind1, ind2, ind3, ind4])

        sig = TransitionCount(:onset)
        @test sig(state, (0.0, 1.0)) == 1
        @test sig(state, (1.0, 2.0)) == 1
        @test sig(state, (2.0, 3.0)) == 0

        # `:infection` reads the field directly, not a state key.
        @test TransitionCount(:infection)(state, (-1.0, 1.0)) == 3  # every case, infection_time = 0.0
    end

    @testset "Prevalence" begin
        ind1 = _case(; onset = 0.5, outcome = 5.0)   # leaves by day 5
        ind2 = _case(; onset = 1.5, outcome = Inf)   # never leaves
        ind3 = _case(; onset = NaN)                  # never enters
        state = _toy_state([ind1, ind2, ind3])

        sig = Prevalence(:onset)   # defaults `leave = :outcome`
        @test sig(state, (0.0, 1.0)) == 1   # only ind1 has onset by day 1
        @test sig(state, (1.0, 2.0)) == 2   # ind1 (still in, outcome day 5) and ind2
        @test sig(state, (5.0, 6.0)) == 1   # ind1 has left by day 6

        # a custom `leave`, such as a released isolation
        iso_dict = Dict{Symbol, Any}(
            :onset_time => NaN, :outcome_time => Inf, :infected => true,
            :isolation_time => 2.0, :isolation_release_time => 4.0
        )
        iso_case = Individual(; id = 1, state = iso_dict)
        iso_state = _toy_state([iso_case])
        isolated = Prevalence(:isolation_time, :isolation_release_time)
        @test isolated(iso_state, (0.0, 1.0)) == 0  # not isolated yet
        @test isolated(iso_state, (2.0, 3.0)) == 1  # isolated by day 2, still in at day 3
        @test isolated(iso_state, (4.0, 5.0)) == 0  # released by day 4
    end

    @testset "SignalSeries and SignalData validate their inputs" begin
        sig = TransitionCount(:onset)
        @test_throws ArgumentError SignalSeries(sig, v -> Poisson(v + 0.1), Float64[])

        series = SignalSeries(sig, v -> Poisson(v + 0.1), [1.0, 2.0])
        @test_throws ArgumentError SignalData([0.0, 1.0, 3.0, 2.0], [series])  # not sorted
        @test_throws ArgumentError SignalData([0.0, 1.0], [series])           # too few steps for the data
        @test_throws ArgumentError SignalData(
            [0.0, 1.0, 2.0], SignalSeries[]
        )                                                                     # no series
    end

    @testset "loglikelihood(::SignalData, ::SimulationState)" begin
        ind1 = _case(; onset = 0.5, outcome = 5.0)
        ind2 = _case(; onset = 1.5, outcome = Inf)
        state = _toy_state([ind1, ind2])

        series = SignalSeries(TransitionCount(:onset), v -> Poisson(v + 0.1), [1.0, 1.0])
        data = SignalData([0.0, 1.0, 2.0], [series])

        expected = logpdf(Poisson(1.1), 1.0) + logpdf(Poisson(1.1), 1.0)
        @test loglikelihood(data, state) ≈ expected

        # Several series are added together.
        series2 = SignalSeries(Prevalence(:onset), v -> Poisson(v + 0.1), [1.0, 2.0])
        data2 = SignalData([0.0, 1.0, 2.0], [series, series2])
        @test loglikelihood(data2, state) ≈ loglikelihood(data, state) +
            logpdf(Poisson(1.1), 1.0) + logpdf(Poisson(2.1), 2.0)
    end

    @testset "loglikelihood(::SignalData, ::TransmissionModel) simulates once and scores it" begin
        process = BranchingProcess(Poisson(1.2), Exponential(2.0))
        series = SignalSeries(
            TransitionCount(:infection), v -> Poisson(v + 0.1), fill(1.0, 10)
        )
        data = SignalData(0.0:1.0:10.0, [series])

        ll_direct = loglikelihood(data, simulate(process; max_cases = 30, rng = StableRNG(7)))
        ll_model = loglikelihood(data, process; max_cases = 30, rng = StableRNG(7))
        @test ll_model ≈ ll_direct

        spec = ModelSpec(process)
        ll_spec = loglikelihood(data, spec; max_cases = 30, rng = StableRNG(7))
        @test ll_spec ≈ ll_direct
    end
end
