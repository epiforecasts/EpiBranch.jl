# final_status_loglik: the likelihood of who ended up infected, with no
# infection times, under a community hazard and a within-household kernel.
#
# A household of two is small enough to write every final-set probability by
# hand (Ball 1986's construction collapses to three terms), so the exact
# formula is the main check; a household of three cross-checks the same
# function against simulated frequencies.

@testset "final_status_loglik" begin
    @testset "one person: pure community exposure" begin
        α, T = 0.2, 3.0
        process = HouseholdProcess([1], Exponential(1.0); external_hazard = α, obs_end = T)
        escape = exp(-α * T)
        @test final_status_loglik(process, [true]; window = 2.0) ≈ log(1 - escape)
        @test final_status_loglik(process, [false]; window = 2.0) ≈ log(escape)
        @test final_status_loglik(process, [missing]; window = 2.0) ≈ 0.0
    end

    @testset "two people: matches Ball's construction by hand" begin
        α, T, β, L = 0.15, 4.0, 0.6, 2.0
        process = HouseholdProcess(
            [2], Exponential(1 / β); external_hazard = α, obs_end = T
        )
        π_esc = exp(-α * T)       # escapes the community
        q = exp(-β * L)           # escapes a single household-mate's contact

        p_none = π_esc^2
        p_one = π_esc * (1 - π_esc) * q      # one specific member only
        p_both = 1 - p_none - 2 * p_one

        @test final_status_loglik(process, [false, false]; window = L) ≈ log(p_none)
        @test final_status_loglik(process, [true, false]; window = L) ≈ log(p_one)
        @test final_status_loglik(process, [false, true]; window = L) ≈ log(p_one)
        @test final_status_loglik(process, [true, true]; window = L) ≈ log(p_both)

        # marginalising a missing member over its two outcomes recovers the
        # probability of what is actually known
        @test final_status_loglik(process, [true, missing]; window = L) ≈
            log(p_one + p_both)
        @test final_status_loglik(process, [missing, missing]; window = L) ≈ 0.0
    end

    @testset "complete patterns of a household sum to one" begin
        α, T, β, L = 0.1, 5.0, 0.4, 3.0
        process = HouseholdProcess(
            [3], Exponential(1 / β); external_hazard = α, obs_end = T
        )
        total = sum(
            exp(final_status_loglik(process, collect(pattern); window = L))
                for pattern in Iterators.product(fill((true, false), 3)...)
        )
        @test total ≈ 1.0
    end

    @testset "matches simulated frequencies" begin
        α, T, β, L = 0.08, 6.0, 0.5, 2.0
        n = 40_000
        sizes = fill(3, n)
        spec = ModelSpec(
            HouseholdProcess(sizes, Exponential(1 / β); external_hazard = α, obs_end = T);
            progression = [Transition(:recovered; from = :infection, delay = L, terminal = true)]
        )
        state = simulate(spec; rng = StableRNG(11))
        infected = reshape(is_infected.(state.individuals), 3, n)

        process = HouseholdProcess(
            [3], Exponential(1 / β); external_hazard = α, obs_end = T
        )
        for pattern in ([true, false, false], [true, true, false], [true, true, true])
            observed = count(k -> infected[:, k] == pattern, 1:n) / n
            expected = exp(final_status_loglik(process, pattern; window = L))
            se = sqrt(expected * (1 - expected) / n)
            @test abs(observed - expected) < max(5 * se, 0.002)
        end
    end

    @testset "susceptibility and infectiousness shift the probability of spread" begin
        α, T, β, L = 0.1, 4.0, 0.5, 2.0
        process = HouseholdProcess(
            [2], Exponential(1 / β); external_hazard = α, obs_end = T
        )
        base = final_status_loglik(process, [true, true]; window = L)
        less_susceptible = final_status_loglik(
            process, [true, true]; window = L, susceptibility = [1.0, 0.2]
        )
        less_infectious = final_status_loglik(
            process, [true, true]; window = L, infectiousness = [0.2, 1.0]
        )
        @test less_susceptible < base
        @test less_infectious < base

        # susceptibility 0 also thins community exposure to nothing, so a
        # fully protected member can never be infected and drops out of the
        # household entirely: the other member's own escape is all that is left
        @test final_status_loglik(
            process, [false, false]; window = L, susceptibility = [1.0, 0.0]
        ) ≈ -α * T
    end

    @testset "many households sum independently" begin
        α, T, β, L = 0.1, 5.0, 0.4, 2.0
        one = HouseholdProcess([2], Exponential(1 / β); external_hazard = α, obs_end = T)
        two = HouseholdProcess([2, 3], Exponential(1 / β); external_hazard = α, obs_end = T)
        a = final_status_loglik(one, [true, false]; window = L)
        b = final_status_loglik(
            HouseholdProcess([3], Exponential(1 / β); external_hazard = α, obs_end = T),
            [true, true, missing]; window = L
        )
        @test final_status_loglik(
            two, [true, false, true, true, missing]; window = L
        ) ≈ a + b
    end

    @testset "reads the window from a ModelSpec's progression" begin
        α, T, β, L = 0.1, 5.0, 0.4, 2.0
        process = HouseholdProcess(
            [3], Exponential(1 / β); external_hazard = α, obs_end = T
        )
        m = ModelSpec(
            process;
            progression = [Transition(:recovered; from = :infection, delay = L, terminal = true)]
        )
        statuses = [true, false, missing]
        @test final_status_loglik(m, statuses) ≈
            final_status_loglik(process, statuses; window = L)

        with_iv = ModelSpec(
            process;
            progression = [Transition(:recovered; from = :infection, delay = L, terminal = true)],
            interventions = [Isolation(onset_to_isolation_delay = Exponential(1.0), duration = 1.0)]
        )
        @test_throws ArgumentError final_status_loglik(with_iv, statuses)
    end

    @testset "invalid arguments" begin
        process = HouseholdProcess([2], Exponential(1.0); external_hazard = 0.1, obs_end = 5.0)
        @test_throws ArgumentError final_status_loglik(process, [true]; window = 2.0)
        @test_throws ArgumentError final_status_loglik(
            process, [true, false]; window = 2.0, susceptibility = [1.0]
        )

        no_hazard = HouseholdProcess([2], Exponential(1.0))
        @test_throws ArgumentError final_status_loglik(
            no_hazard, [true, false]; window = 2.0
        )

        unbounded = HouseholdProcess(
            [2], Exponential(1.0); external_hazard = 0.1, obs_end = Inf
        )
        @test_throws ArgumentError final_status_loglik(
            unbounded, [true, false]; window = 2.0
        )

        pk = HouseholdProcess(
            [2], PairKernel(ctx -> Exponential(1.0)); external_hazard = 0.1, obs_end = 5.0
        )
        @test_throws ArgumentError final_status_loglik(
            pk, [true, false]; window = 2.0
        )
    end
end
