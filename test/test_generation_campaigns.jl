using Distributions
using StableRNGs
using Statistics

# A calendar time at which a group's eligibility triggers, so the individual-based
# engine dates a dose without a delay.
struct _EligibleAt <: EpiBranch.TraceEligibility
    time::Float64
end
EpiBranch.trigger_time(e::_EligibleAt, infector, state) = e.time

@testset "Vaccination campaigns in generation counts" begin
    matrix = [1.2 0.3; 0.3 1.2]
    discrete(; N = 200_000) = BranchingProcess(
        matrix, R -> Poisson(R), EpiBranch.NoGenerationTime(); population_size = N
    )
    group(; efficacy = 0.5, kwargs...) = GroupVaccination(;
        efficacy, group_key = :type, kwargs...
    )
    run_counts(spec; seed = 1, kwargs...) = simulate_by_generation(
        spec; n_initial = [10, 2], max_generations = 7,
        population_by_type = [40_000, 160_000], depletion = StartOfGenerationDepletion(),
        generation_length = 14.0, rng = StableRNG(seed), kwargs...
    )

    @testset "doses of a generation's interval" begin
        doses(iv, t0, t1) = EpiBranch._interval_doses(
            EpiBranch._campaign(iv, (0.0, Inf), nothing), t0, t1
        )
        schedule = Steps([140.0, 154.0], [0.0, 2_000.0, 0.0])
        steps = CapacityConstrained(group(); budget_per_period = schedule, period = 1.0)
        @test doses(steps, 126.0, 140.0) == 0
        @test doses(steps, 140.0, 154.0) == 14 * 2_000
        @test doses(steps, 154.0, 168.0) == 0

        # Unused budget carries over into the first interval the window opens in
        # unless `carry_over = false`.
        daily = CapacityConstrained(group(); budget_per_period = 10.0, period = 1.0)
        opened = Scheduled(daily; start_time = 20.0, end_time = 40.0)
        @test doses(opened, 14.0, 28.0) == 28 * 10
        @test doses(opened, 28.0, 42.0) == 12 * 10
        @test doses(Scheduled(daily; start_time = 20.0), 42.0, 56.0) == 14 * 10
        wrapped = CapacityConstrained(
            Scheduled(group(); start_time = 20.0, end_time = 40.0);
            budget_per_period = 10.0, period = 1.0
        )
        @test doses(wrapped, 14.0, 28.0) == 28 * 10
        own = CapacityConstrained(
            group(); budget_per_period = 10.0, period = 1.0, carry_over = false
        )
        @test doses(Scheduled(own; start_time = 20.0), 14.0, 28.0) == 8 * 10

        # One budget for the whole outbreak is given in the first interval.
        single = CapacityConstrained(group(); budget_per_period = 500.0)
        @test doses(single, 0.0, 14.0) == 500
        @test doses(single, 14.0, 28.0) == 0
    end

    @testset "a dose reaches the susceptibles it finds, and protects with efficacy" begin
        sizes = [10_000, 10_000]
        removed(budget) = begin
            campaign = EpiBranch._campaign(
                CapacityConstrained(group(; efficacy = 0.6); budget_per_period = budget),
                (0.0, Inf), nothing
            )
            [
                begin
                    s = [8_000, 2_000]
                    EpiBranch._vaccinate!(StableRNG(seed), campaign, 0.0, 14.0, s, sizes)
                    [8_000, 2_000] .- s
                end for seed in 1:400
            ]
        end
        # With a budget of its own each type has 1,000 doses, a share 0.8 or
        # 0.2 of which reach susceptibles, and 0.6 of those protect.
        own = removed(PerGroup(1 => 1_000.0, 2 => 1_000.0))
        @test isapprox(mean(first, own), 1_000 * 0.8 * 0.6; rtol = 0.03)
        @test isapprox(mean(last, own), 1_000 * 0.2 * 0.6; rtol = 0.05)

        # A shared budget of 1,000 is shared out by population, so equal types
        # have 500 doses each; a type given none of its own budget gets none.
        shared = removed(1_000.0)
        @test isapprox(mean(first, shared), 500 * 0.8 * 0.6; rtol = 0.03)
        @test isapprox(mean(last, shared), 500 * 0.2 * 0.6; rtol = 0.05)
        none = removed(PerGroup(1 => 0.0, 2 => 1_000.0))
        @test all(r -> r[1] == 0, none)
        @test mean(last, none) > 0
    end

    @testset "a campaign with no doses changes nothing" begin
        plain = run_counts(discrete())
        for campaign in (
                CapacityConstrained(group(); budget_per_period = 0.0, period = 1.0),
                CapacityConstrained(
                    group(); budget_per_period = Steps([500.0], [0.0, 0.0]), period = 1.0
                ),
                Scheduled(
                    CapacityConstrained(group(); budget_per_period = 100.0, period = 1.0);
                    start_time = 1_000.0
                ),
            )
            state = run_counts(ModelSpec(discrete(); interventions = [campaign]))
            @test state.counts == plain.counts
            @test state.susceptibles == plain.susceptibles
            @test state.cumulative_cases == plain.cumulative_cases
        end
        @test run_counts(ModelSpec(discrete())).counts == plain.counts
    end

    @testset "a campaign inside a window lowers the cases" begin
        campaign = CapacityConstrained(group(); budget_per_period = 1.0e9, period = 1.0)
        spec = ModelSpec(
            discrete();
            interventions = [Scheduled(campaign; start_time = 30.0, end_time = 40.0)]
        )
        plain = [run_counts(discrete(); seed = s).cumulative_cases for s in 1:60]
        vaccinated = [run_counts(spec; seed = s).cumulative_cases for s in 1:60]
        @test mean(vaccinated) < 0.5 * mean(plain)
    end

    @testset "continuing a state applies the campaign from its generation" begin
        campaign = CapacityConstrained(group(); budget_per_period = 1.0e9, period = 1.0)
        spec = ModelSpec(
            discrete(); interventions = [Scheduled(campaign; start_time = 30.0, end_time = 40.0)]
        )
        whole = run_counts(spec)
        first_part = simulate_by_generation(
            spec; n_initial = [10, 2], max_generations = 3,
            population_by_type = [40_000, 160_000], depletion = StartOfGenerationDepletion(),
            generation_length = 14.0, rng = StableRNG(1)
        )
        continued = simulate(spec, first_part; max_generations = 7)
        @test continued.counts == whole.counts
        @test continued.susceptibles == whole.susceptibles
    end

    @testset "interventions it cannot apply are refused, by name" begin
        spec(iv; model = discrete()) = ModelSpec(model; interventions = [iv])
        capacity = CapacityConstrained(group(); budget_per_period = 10.0, period = 1.0)
        err(s; kwargs...) = try
            run_counts(s; kwargs...)
            nothing
        catch e
            sprint(showerror, e)
        end

        @test occursin("generation_length", err(spec(capacity); generation_length = nothing))
        @test occursin("CapacityConstrained", err(spec(group())))
        @test occursin(
            "RingVaccination",
            err(spec(CapacityConstrained(RingVaccination(efficacy = 0.5); budget_per_period = 1.0)))
        )
        @test occursin(
            "Isolation",
            err(spec(Isolation(onset_to_isolation_delay = 1.0, duration = 7.0)))
        )
        @test occursin(
            "group_key = :type", err(
                spec(
                    CapacityConstrained(GroupVaccination(efficacy = 0.5); budget_per_period = 10.0)
                )
            )
        )
        @test occursin(
            "coverage", err(
                spec(
                    CapacityConstrained(group(; coverage = 0.5); budget_per_period = 10.0)
                )
            )
        )
        @test occursin(
            "dose_delay", err(
                spec(
                    CapacityConstrained(group(; dose_delay = 2.0); budget_per_period = 10.0)
                )
            )
        )
        @test occursin("start_time", err(spec(Scheduled(capacity; start_after_cases = 5))))
        @test occursin("finite population", err(spec(capacity; model = BranchingProcess(matrix, R -> Poisson(R), EpiBranch.NoGenerationTime())); population_by_type = nothing))
        @test occursin(
            "BranchingProcess",
            err(ModelSpec(HomogeneousProcess(transmission_rate = 0.1, population_size = 100); interventions = [capacity]))
        )
    end

    @testset "agrees with individual-based simulation without depletion" begin
        # Individual-based generations a fixed 14 days apart. Doses are given
        # at a day in the campaign window to everyone, so the campaign is the
        # same one each engine applies: half of each type's contacts from day
        # 42 on are protected. Both start from the same index cases.
        individual = BranchingProcess(matrix, R -> Poisson(R), Dirac(14.0))
        campaign = Scheduled(
            CapacityConstrained(
                group(; eligibility = _EligibleAt(30.0)); budget_per_period = 1.0e9,
                period = 1.0
            );
            start_time = 30.0, end_time = 40.0
        )
        generations = 7
        reps = 100
        function by_individuals(spec, seed)
            state = simulate(
                spec; index_types = [fill(1, 10); fill(2, 2)], max_generations = generations,
                rng = StableRNG(seed), max_cases = 10^6
            )
            return count(i -> is_infected(i) && i.generation <= generations, state.individuals)
        end
        by_counts(spec, seed) = run_counts(spec; seed).cumulative_cases
        for ivs in ([], [campaign])
            a = [by_individuals(ModelSpec(individual; interventions = ivs), s) for s in 1:reps]
            b = [by_counts(ModelSpec(discrete(); interventions = ivs), 10^6 + s) for s in 1:reps]
            se = sqrt(var(a) / reps + var(b) / reps)
            @test isapprox(mean(a), mean(b); atol = 4se)
        end
    end

    @testset "a PerGroup budget is read per type, and must name every type" begin
        doses = Steps([140.0, 154.0], [0.0, 2_000.0, 0.0])
        campaign(budget) = CapacityConstrained(group(); budget_per_period = budget, period = 1.0)
        per_type = EpiBranch._campaign(
            campaign(PerGroup(1 => doses, 2 => 500.0)), (0.0, Inf), nothing
        )
        @test EpiBranch._interval_doses(per_type, 140.0, 154.0, 1) == 14 * 2_000
        @test EpiBranch._interval_doses(per_type, 140.0, 154.0, 2) == 14 * 500
        @test EpiBranch._interval_doses(per_type, 126.0, 140.0, 1) == 0

        err = try
            run_counts(ModelSpec(discrete(); interventions = [campaign(PerGroup(1 => 10.0))]))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("group 2", err.msg)
        @test_throws ArgumentError run_counts(
            ModelSpec(
                discrete(); interventions = [campaign(PerGroup(1 => 10.0, 2 => 1.0; key = :community))]
            )
        )
    end
end
