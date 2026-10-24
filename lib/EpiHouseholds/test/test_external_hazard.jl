@testset "External hazard that differs between people" begin
    sir = [Transition(:recovered; from = :infection, delay = 5.0, terminal = true)]
    sizes = fill(4, 10)
    arm = [fill(0.01, 20); fill(0.005, 20)]

    @testset "a vector or a function of the person is accepted" begin
        @test HouseholdProcess(
            sizes, Exponential(3.0); external_hazard = arm, obs_end = 60.0
        ) isa HouseholdProcess
        @test HouseholdProcess(
            sizes, Exponential(3.0); external_hazard = j -> arm[j], obs_end = 60.0
        ) isa HouseholdProcess
        @test HouseholdProcess(
            sizes, Exponential(3.0);
            external_hazard = [fill(Exponential(100.0), 20); fill(0.005, 20)],
            obs_end = 60.0
        ) isa HouseholdProcess
        @test_throws DimensionMismatch HouseholdProcess(
            sizes, Exponential(3.0); external_hazard = arm[1:39], obs_end = 60.0
        )
        @test_throws ArgumentError HouseholdProcess(
            sizes, Exponential(3.0); external_hazard = [arm[1:39]; -1.0], obs_end = 60.0
        )
    end

    @testset "the same rate for everyone matches the shared rate" begin
        shared = ModelSpec(
            HouseholdProcess(sizes, Exponential(3.0); external_hazard = 0.02, obs_end = 30.0);
            progression = sir
        )
        for source in (fill(0.02, 40), j -> 0.02)
            per_person = ModelSpec(
                HouseholdProcess(
                    sizes, Exponential(3.0); external_hazard = source, obs_end = 30.0
                );
                progression = sir
            )
            a = simulate(shared; rng = StableRNG(3))
            b = simulate(per_person; rng = StableRNG(3))
            @test [i.infection_time for i in a.individuals] ==
                [i.infection_time for i in b.individuals]
            data = household_infections(a, shared)
            @test loglikelihood(data, per_person) ≈ loglikelihood(data, shared)
        end
    end

    @testset "each person's term is their own rate's" begin
        α = (0.03, 0.01)
        model = ModelSpec(
            HouseholdProcess(
                fill(4, 400), Exponential(3.0);
                external_hazard = j -> j <= 800 ? α[1] : α[2], obs_end = 30.0
            );
            progression = sir
        )
        data = household_infections(simulate(model; rng = StableRNG(11)), model)
        layout = compile_household_pairs(data; external = true)
        per_arm = [
            pairwise_surv_loglik_by_component(
                Exponential(3.0), data, layout; external_hazard = a
            ) for a in α
        ]
        mixed = pairwise_surv_loglik_by_component(
            Exponential(3.0), data, layout; external_hazard = model.process.external_hazard
        )
        @test mixed[1:200] ≈ per_arm[1][1:200]
        @test mixed[201:400] ≈ per_arm[2][201:400]
        @test loglikelihood(data, model) ≈ sum(mixed)
    end

    @testset "a person with no external hazard is never infected from outside" begin
        # Only the first household meets the community; the rest have no index
        # case and stay uninfected.
        model = ModelSpec(
            HouseholdProcess(
                sizes, Exponential(3.0);
                external_hazard = [fill(0.5, 4); zeros(36)], obs_end = 30.0
            );
            progression = sir
        )
        state = simulate(model; rng = StableRNG(5))
        @test any(is_infected, state.individuals[1:4])
        @test !any(is_infected, state.individuals[5:end])
        data = household_infections(state, model)
        @test isfinite(loglikelihood(data, model))
    end

    @testset "round trip recovers the difference between arms" begin
        # A cluster-randomised design: half the households have their community
        # hazard reduced by a rate ratio, which the likelihood recovers along
        # with the baseline rate and the within-household kernel.
        n = 3000
        true_θ = [log(3.0), log(0.02), log(0.5)]
        in_arm = [h > n ÷ 2 for h in 1:n for _ in 1:4]
        hazard(θ) = exp(θ[2]) .* ifelse.(in_arm, exp(θ[3]), 1.0)
        model = ModelSpec(
            HouseholdProcess(
                fill(4, n), Exponential(3.0);
                external_hazard = hazard(true_θ), obs_end = 30.0
            );
            progression = sir
        )
        data = household_infections(simulate(model; rng = StableRNG(17)), model)
        layout = compile_household_pairs(data; external = true)
        g(θ) = pairwise_surv_loglik(
            Exponential(exp(θ[1])), data, layout; external_hazard = hazard(θ)
        )
        θhat = copy(true_θ)
        for _ in 1:20
            θhat -= ForwardDiff.hessian(g, θhat) \ ForwardDiff.gradient(g, θhat)
        end
        cov = inv(-ForwardDiff.hessian(g, θhat))
        se = [sqrt(cov[k, k]) for k in eachindex(θhat)]
        @test all(abs.(θhat .- true_θ) .< 3 .* se)
    end

    @testset "a vector hazard scales linearly, not quadratically, with the population" begin
        # Preparing the vector hazard once per simulation, rather than once
        # per household, keeps the cost linear in population size rather
        # than quadratic: a 4x larger population should cost ~4x, not ~16x.
        function bytes_for(n)
            model = ModelSpec(
                HouseholdProcess(
                    fill(1, n), Exponential(1.0);
                    external_hazard = fill(0.001, n), obs_end = 100.0
                );
                progression = sir
            )
            simulate(model; rng = StableRNG(1))
            return @allocated simulate(model; rng = StableRNG(2))
        end
        small = bytes_for(1000)
        large = bytes_for(4000)
        @test large < 8 * small
    end
end

@testset "External hazard following a calendar schedule" begin
    sir = [Transition(:recovered; from = :infection, delay = 5.0, terminal = true)]
    # single-member households: with no household-mate to infect, every
    # infection is a community one, which isolates the calendar term
    sizes = fill(1, 200)

    @testset "a schedule with one value everywhere matches the shared rate" begin
        α = 0.02
        shared = ModelSpec(
            HouseholdProcess(sizes, Exponential(3.0); external_hazard = α, obs_end = 30.0);
            progression = sir
        )
        scheduled = ModelSpec(
            HouseholdProcess(
                sizes, Exponential(3.0);
                external_hazard = Steps([10.0], [α, α]), obs_end = 30.0
            );
            progression = sir
        )
        data = household_infections(simulate(shared; rng = StableRNG(7)), shared)
        @test loglikelihood(data, scheduled) ≈ loglikelihood(data, shared)
    end

    @testset "a changing rate is read off a weekly calendar" begin
        # a community hazard of 0.002 per day for the first week and 0.0005
        # afterwards
        model = ModelSpec(
            HouseholdProcess(
                sizes, Exponential(1.0);
                external_hazard = Steps([7.0], [0.002, 0.0005]), obs_end = 30.0
            );
            progression = sir
        )
        state = simulate(model; rng = StableRNG(1))
        @test any(is_infected, state.individuals)
        data = household_infections(state, model)
        @test isfinite(loglikelihood(data, model))
    end
end
