using Distributions
using StableRNGs
using Statistics
using Random: shuffle

# A calendar time at which a group's eligibility triggers, so the individual-based
# engine dates a dose without a delay.
struct _EligibleAt <: EpiBranch.TraceEligibility
    time::Float64
end
EpiBranch.trigger_time(e::_EligibleAt, infector, state) = e.time

@testset "Vaccination campaigns in generation counts" begin
    matrix = [1.2 0.3; 0.3 1.2]
    discrete(; sizes = [40_000, 160_000]) = BranchingProcess(
        matrix, R -> Poisson(R), EpiBranch.NoGenerationTime(); population = Population(sizes)
    )
    group(; efficacy = 0.5, kwargs...) = GroupVaccination(;
        efficacy, group_key = :type, kwargs...
    )
    run_counts(spec; seed = 1, kwargs...) = simulate_by_generation(
        spec; n_initial = [10, 2], max_generations = 7,
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
        # unless `unused_budget = ExpireUnused()`.
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
            group(); budget_per_period = 10.0, period = 1.0, unused_budget = ExpireUnused()
        )
        @test doses(Scheduled(own; start_time = 20.0), 14.0, 28.0) == 8 * 10

        # One budget for the whole outbreak is given in the first interval.
        single = CapacityConstrained(group(); budget_per_period = 500.0)
        @test doses(single, 0.0, 14.0) == 500
        @test doses(single, 14.0, 28.0) == 0
    end

    @testset "doses go to the contacts made, while the budget lasts" begin
        # Visit every contact in the order made, as the individual-based
        # simulation does. A shared budget doses the first contacts of all the
        # types, a budget per type the first of its own.
        function by_visiting(rng, contacts, s, sizes, doses)
            order = shuffle(rng, reduce(vcat, [fill(i, n) for (i, n) in enumerate(contacts)]))
            s = copy(s)
            seen = zeros(Int, length(contacts))
            kept = zeros(Int, length(contacts))
            for (j, i) in enumerate(order)
                seen[i] += 1
                pass = prod(
                    (1 - e for (d, e) in doses if (d isa Int ? j : seen[i]) <= (d isa Int ? d : d[i]));
                    init = 1.0
                )
                if rand(rng) < pass * s[i] / sizes[i]
                    kept[i] += 1
                    s[i] -= 1
                end
            end
            return kept
        end
        by_counting(rng, potential, s, sizes, doses) = EpiBranch._deplete_dosed!(
            rng, PerContactDepletion(), potential, copy(s), sizes, doses,
            [zeros(Int, d isa Int ? 1 : length(d)) for (d, _) in doses]
        )
        function agree(contacts, s, sizes, doses; reps = 3_000)
            visited = [by_visiting(StableRNG(i), contacts, s, sizes, doses) for i in 1:reps]
            counted = [by_counting(StableRNG(10^6 + i), contacts, s, sizes, doses) for i in 1:reps]
            return all(eachindex(contacts)) do t
                a, b = getindex.(visited, t), getindex.(counted, t)
                isapprox(mean(a), mean(b); atol = 4 * sqrt(var(a) / reps + var(b) / reps))
            end
        end
        contacts, s, sizes = [300, 500], [400, 600], [1_000, 1_000]

        # Budgets below, around and above the number of contacts.
        for budget in (50, 250, 800, 5_000)
            @test agree(contacts, s, sizes, [(budget, 0.8)])
        end
        @test agree(contacts, s, sizes, [([100, 300], 0.5)])
        @test agree(contacts, s, sizes, [(150, 0.5), (400, 0.3), (50, 0.9)])
        @test agree(contacts, s, sizes, [(150, 0.5), ([200, 100], 0.3)])

        # A full efficacy blocks every dosed contact.
        kept = by_counting(StableRNG(1), [300, 500], s, sizes, [(800, 1.0)])
        @test kept == [0, 0]

        # A Poisson mean gives Poisson contacts, as the visiting reference draws.
        reps = 3_000
        counted = [
            sum(by_counting(StableRNG(10^6 + i), [300.0, 500.0], s, sizes, [(250, 0.8)]))
                for i in 1:reps
        ]
        visited = [
            begin
                rng = StableRNG(i)
                n = [rand(rng, Poisson(300)), rand(rng, Poisson(500))]
                sum(by_visiting(rng, n, s, sizes, [(250, 0.8)]))
            end for i in 1:reps
        ]
        @test isapprox(
            mean(counted), mean(visited);
            atol = 4 * sqrt(var(counted) / reps + var(visited) / reps)
        )
    end

    @testset "a single type does the same without allocating" begin
        function by_visiting(rng, n, s, size, doses)
            kept = 0
            for j in 1:n
                pass = prod((1 - e for (d, e) in doses if j <= d); init = 1.0)
                if rand(rng) < pass * s / size
                    kept += 1
                    s -= 1
                end
            end
            return kept
        end
        function by_counting(rng, n, s, size, doses)
            kept, left, taken = EpiBranch._deplete_dosed(
                rng, PerContactDepletion(), n, s, size, Tuple(last.(doses)),
                Tuple(first.(doses))
            )
            @test left == s - kept
            @test taken == Tuple(min.(first.(doses), n))
            return kept
        end
        reps = 4_000
        for doses in ([(60, 0.8)], [(300, 0.8)], [(900, 0.8)], [(50, 0.5), (200, 0.3), (400, 0.9)])
            visited = [by_visiting(StableRNG(i), 400, 500, 1_000, doses) for i in 1:reps]
            counted = [by_counting(StableRNG(10^6 + i), 400, 500, 1_000, doses) for i in 1:reps]
            @test isapprox(
                mean(visited), mean(counted);
                atol = 4 * sqrt(var(visited) / reps + var(counted) / reps)
            )
        end
    end

    @testset "a shared budget splits between types by the contacts they make" begin
        contacts = [300, 700]
        split = [
            only(EpiBranch._dosed_contacts(StableRNG(i), [(200, 0.5)], contacts))
                for i in 1:2_000
        ]
        @test all(==(200), sum.(split))
        @test isapprox(mean(first, split), 200 * 0.3; rtol = 0.03)
        # More doses than contacts dose them all, and a larger budget doses
        # every contact a smaller one does, whatever the order given.
        @test only(EpiBranch._dosed_contacts(StableRNG(1), [(5_000, 0.5)], contacts)) ==
            contacts
        nested = EpiBranch._dosed_contacts(StableRNG(1), [(400, 0.5), (100, 0.5)], contacts)
        @test all(nested[2] .<= nested[1])
        @test sum(nested[1]) == 400 && sum(nested[2]) == 100
    end

    @testset "doses a generation does not use stay for the next" begin
        sizes = [40_000, 160_000]
        # A single budget of 20 doses is used up by the first contacts.
        once = CapacityConstrained(group(); budget_per_period = 20.0)
        state = run_counts(ModelSpec(discrete(); interventions = [once]))
        @test state.doses_used == [[20]]
        # With more doses than contacts in a generation, the rest are used later.
        late = CapacityConstrained(group(); budget_per_period = 1_000.0, period = 14.0)
        state = run_counts(ModelSpec(discrete(); interventions = [late]))
        @test only(only(state.doses_used)) > 0
        whole = run_counts(ModelSpec(discrete(); interventions = [late]))
        first_part = simulate_by_generation(
            ModelSpec(discrete(); interventions = [late]); n_initial = [10, 2],
            max_generations = 3, generation_length = 14.0, rng = StableRNG(1)
        )
        continued = simulate(
            ModelSpec(discrete(); interventions = [late]), first_part; max_generations = 7
        )
        @test continued.doses_used == whole.doses_used
        # Without `CarryOver` the doses of a period are lost if they are not used.
        lost = CapacityConstrained(
            group(); budget_per_period = 1.0e6, period = 14.0, unused_budget = ExpireUnused()
        )
        pool = EpiBranch._campaign(lost, (0.0, Inf), nothing)
        @test EpiBranch._doses_left(pool, pool.budget, 14.0, 28.0, 10^6) == 10^6
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
        @test mean(vaccinated) < 0.9 * mean(plain)
    end

    @testset "continuing a state applies the campaign from its generation" begin
        campaign = CapacityConstrained(group(); budget_per_period = 1.0e9, period = 1.0)
        spec = ModelSpec(
            discrete(); interventions = [Scheduled(campaign; start_time = 30.0, end_time = 40.0)]
        )
        whole = run_counts(spec)
        first_part = simulate_by_generation(
            spec; n_initial = [10, 2], max_generations = 3,
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
            err(spec(Isolation(delay = 1.0, duration = 7.0)))
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
        @test occursin("finite population", err(spec(capacity; model = BranchingProcess(matrix, R -> Poisson(R), EpiBranch.NoGenerationTime()))))
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
            return count(i -> is_infected(i) && i.generation <= generations, individuals(state))
        end
        by_counts(spec, seed) = run_counts(spec; seed).cumulative_cases
        for ivs in ([], [campaign])
            a = [by_individuals(ModelSpec(individual; interventions = ivs), s) for s in 1:reps]
            b = [by_counts(ModelSpec(discrete(); interventions = ivs), 10^6 + s) for s in 1:reps]
            se = sqrt(var(a) / reps + var(b) / reps)
            @test isapprox(mean(a), mean(b); atol = 4se)
        end
    end

    @testset "GroupVaccination budgets are shared by the types" begin
        gv = CapacityConstrained(group(); budget_per_period = 1.0)
        a = Individual(id = 1, state = Dict{Symbol, Any}(:type => 1))
        @test EpiBranch._resource_of(gv, a, nothing) === nothing
    end

    @testset "a prepared spec starts states as the spec does" begin
        single = BranchingProcess(
            Poisson(2.0); population = Population(50_000; immune = 20_000),
            resolution = ByGeneration(generation_length = 14.0)
        )
        campaign = CapacityConstrained(
            MassVaccination(efficacy = 1.0, eligibility_time = 0.0);
            budget_per_period = 100.0, period = 14.0
        )
        for interventions in ([], [campaign], [campaign, campaign])
            spec = ModelSpec(single; interventions)
            prepared = prepare_generations(spec, simulate(spec; max_generations = 1, rng = StableRNG(1)))
            a = @inferred simulate(prepared; n_initial = 50, max_generations = 3, rng = StableRNG(2))
            b = simulate(spec; n_initial = 50, max_generations = 3, rng = StableRNG(2))
            for field in (:counts, :cumulative_cases, :susceptibles, :generation, :doses_used, :protected)
                @test getproperty(a, field) == getproperty(b, field)
            end
            @test_throws ArgumentError simulate(prepared; n_initial = 50_001, rng = StableRNG(2))
        end
        spec = ModelSpec(discrete())
        prepared = prepare_generations(spec, run_counts(spec))
        a = simulate(prepared; n_initial = [10, 2], max_generations = 3, rng = StableRNG(3))
        b = simulate_by_generation(spec; n_initial = [10, 2], max_generations = 3, rng = StableRNG(3))
        @test a.counts == b.counts
        @test a.susceptibles == b.susceptibles
        @test a.doses_used == b.doses_used
    end

    @testset "a prepared spec continues a state as the spec does" begin
        budget = CapacityConstrained(group(); budget_per_period = 500.0, period = 14.0)
        spec = ModelSpec(discrete(); interventions = [budget])
        state = run_counts(spec)
        prepared = prepare_generations(spec, state)
        @test prepared isa GenerationCampaigns
        n = state.generation + 2
        a = simulate(prepared, state; max_generations = n, rng = StableRNG(2))
        b = simulate(spec, state; max_generations = n, rng = StableRNG(2))
        @test a.counts == b.counts
        @test a.susceptibles == b.susceptibles

        # Continuing a state that already holds one budget per campaign
        # allocates nothing and leaves its doses unchanged.
        full = GenerationCounts(
            300, 300, 50_000, 30_000, 50_000, 5, 14.0, PerContactDepletion(),
            StableRNG(1), (10^6,), 0
        )
        spent = prepare_generations(
            ModelSpec(
                BranchingProcess(Poisson(2.0); population = Population(50_000));
                interventions = [
                    CapacityConstrained(
                        MassVaccination(efficacy = 1.0, eligibility_time = 0.0);
                        budget_per_period = 5.0, period = Inf
                    ),
                ]
            ), full
        )
        out = simulate(spent, full; max_generations = 7, rng = StableRNG(2))
        @test out.doses_used == full.doses_used
        alloc_rng(prepared, s, n, rng) = @allocated simulate(prepared, s; max_generations = n, rng)
        rng = StableRNG(3)
        alloc_rng(spent, full, 7, rng)
        @test alloc_rng(spent, full, 7, rng) == 0
        @test full.doses_used == (10^6,)

        # A single type counts its doses as integers. A continuation
        # that takes doses from a population or from contacts then allocates
        # nothing once the state holds the doses of its campaigns.
        alloc(prepared, s, n) = @allocated simulate(prepared, s; max_generations = n)
        single = BranchingProcess(Poisson(2.0); population = Population(50_000))
        fresh = GenerationCounts(
            300, 300, 50_000, 30_000, 50_000, 0, 14.0, PerContactDepletion(),
            StableRNG(1), (), 0
        )
        for vaccine in (
                MassVaccination(efficacy = 0.9, eligibility_time = 0.0),
                GroupVaccination(efficacy = 0.5, group_key = :type),
            )
            campaign = CapacityConstrained(vaccine; budget_per_period = 500.0, period = 14.0)
            prepared_single = prepare_generations(
                ModelSpec(single; interventions = [campaign]), fresh
            )
            started = simulate(prepared_single, fresh; max_generations = 2)
            @test only(started.doses_used) > 0
            alloc(prepared_single, started, 4)
            @test alloc(prepared_single, started, 4) == 0
        end

        # A fresh state built by `simulate` holds one zero per campaign. Its
        # first continuation then keeps the type of the state and allocates nothing.
        for interventions in (
                [
                    CapacityConstrained(
                        MassVaccination(efficacy = 0.9, eligibility_time = 0.0);
                        budget_per_period = 500.0, period = 14.0
                    ),
                ],
                [
                    CapacityConstrained(
                        GroupVaccination(efficacy = 0.5, group_key = :type);
                        budget_per_period = 500.0, period = 14.0
                    ),
                ],
                [],
            )
            spec_single = ModelSpec(single; interventions)
            prepared_fresh = prepare_generations(spec_single, fresh)
            zeroed = EpiBranch._zero_doses(fresh, prepared_fresh.campaigns)
            @test zeroed.doses_used == ntuple(_ -> 0, length(interventions))
            @test typeof(simulate(prepared_fresh, zeroed; max_generations = 2)) ==
                typeof(zeroed)
            alloc(prepared_fresh, zeroed, 4)
            @test alloc(prepared_fresh, zeroed, 4) == 0
        end

        # The checks of the spec still apply.
        @test_throws ArgumentError prepare_generations(
            ModelSpec(
                discrete(); interventions = [budget],
                attributes = clinical_presentation(incubation_period = LogNormal(1.6, 0.5))
            ), state
        )
    end
end

@testset "multivariate hypergeometric split" begin
    using Statistics
    rng = StableRNG(7)
    # Matches the exact pmf of the univariate case, including a skewed one.
    for (k, m, n) in [(5, 7, 4), (3, 40, 30), (40, 3, 30), (6, 6, 12)]
        d = Hypergeometric(k, m, n)
        draws = [EpiBranch._rand_hypergeometric(rng, k, m, n) for _ in 1:100_000]
        for x in support(d)
            @test isapprox(count(==(x), draws) / 1.0e5, pdf(d, x); atol = 0.01)
        end
    end
    # Means and the total are right at sizes where `Hypergeometric` takes seconds.
    sizes = [3 * 10^8, 5 * 10^8, 2 * 10^8]
    total = 4 * 10^8
    split = [EpiBranch._rand_multivariate_hypergeometric(rng, sizes, total) for _ in 1:200]
    @test all(==(total), sum.(split))
    @test all(all(split[j] .<= sizes) for j in eachindex(split))
    expected = total .* sizes ./ sum(sizes)
    sd = sqrt.(expected .* (1 .- sizes ./ sum(sizes)) .* (sum(sizes) - total) / (sum(sizes) - 1))
    @test all(abs.(mean(split) .- expected) .< 4 .* sd ./ sqrt(200))
    # Products of the counts exceed `typemax(Int)` here.
    big = [4 * 10^10, 6 * 10^10]
    bsplit = [EpiBranch._rand_multivariate_hypergeometric(rng, big, 5 * 10^10) for _ in 1:20]
    @test all(==(5 * 10^10), sum.(bsplit))
    @test all(abs.(mean(bsplit) .- [2.0e10, 3.0e10]) .< 1.0e6)
    @test EpiBranch._rand_multivariate_hypergeometric(rng, [0, 4, 0], 3) == [0, 3, 0]
end
