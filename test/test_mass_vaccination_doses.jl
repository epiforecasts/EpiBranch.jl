using Distributions
using StableRNGs
using Statistics

# A rule that protects every dose it gives, to show a dosing rule needs only
# `protected_by_doses`.
struct _NeverWastes <: AbstractDosing end
function EpiBranch.protected_by_doses(::_NeverWastes, rng, doses, efficacy, susceptibles, vaccinated, size)
    return min(doses, susceptibles)
end

@testset "MassVaccination with a dose budget" begin
    mass(; kwargs...) = MassVaccination(; efficacy = 0.9, eligibility_time = 0.0, kwargs...)
    budget(iv, doses; kwargs...) = CapacityConstrained(iv; budget_per_period = doses, kwargs...)

    @testset "the dosing rule is a field, random by default" begin
        @test mass().dosing isa RandomDosing
        @test mass(dosing = _NeverWastes()).dosing isa _NeverWastes
        @test occursin("dosing = ", sprint(show, mass()))
    end

    @testset "random dosing protects as many as the doses reach" begin
        protected(doses, e, s, n; seed = 1) = EpiBranch.protected_by_doses(
            RandomDosing(), StableRNG(seed), doses, e, s, 0, n
        )
        @test protected(0, 0.9, 500, 1000) == 0
        @test protected(100, 0.0, 500, 1000) == 0
        @test protected(100, 0.9, 0, 1000) == 0
        # Doses beyond the susceptibles cannot protect more than there are.
        @test protected(10^6, 1.0, 500, 1000) == 500
        # Each dose reaches a susceptible with probability s / n and protects it
        # with probability `efficacy`, so s susceptibles fall to s exp(-e d / n).
        reps = 400
        draws = [protected(2_000, 0.8, 800, 1_000; seed = i) for i in 1:reps]
        expected = 800 * (1 - exp(-0.8 * 2_000 / 1_000))
        @test abs(mean(draws) - expected) < 4 * std(draws) / sqrt(reps)
    end

    @testset "dosing rules that follow the susceptibles or favour the vaccinated" begin
        protected(rule, doses, e, s, v, n; seed = 1) = EpiBranch.protected_by_doses(
            rule, StableRNG(seed), doses, e, s, v, n
        )
        reps = 400
        mean_of(rule, args...) = mean(protected(rule, args...; seed = i) for i in 1:reps)
        # Doses go to susceptibles only: each of s susceptibles escapes d doses
        # with probability (1 - e / s)^d.
        s, d, e = 400, 600, 0.8
        expected = s * (1 - (1 - e / s)^d)
        draws = [protected(SusceptibleDosing(), d, e, s, 0, 1_000; seed = i) for i in 1:reps]
        @test abs(mean(draws) - expected) < 4 * std(draws) / sqrt(reps)
        @test protected(SusceptibleDosing(), 10^6, 1.0, 500, 0, 1_000) == 500
        # A weight of 1 is random dosing, whatever the vaccinated count.
        same = [
            protected(FavourVaccinated(1.0), 2_000, 0.8, 800, 100, 1_000; seed = i)
                for i in 1:reps
        ]
        random = [
            protected(RandomDosing(), 2_000, 0.8, 800, 100, 1_000; seed = i)
                for i in 1:reps
        ]
        @test same == random
        # A weight above 1 leaves fewer doses for the unprotected. The pool is
        # the unprotected plus the weighted protected.
        draws = [
            protected(FavourVaccinated(3.0), 2_000, 0.8, 600, 300, 1_000; seed = i)
                for i in 1:reps
        ]
        expected = 600 * (1 - (1 - 0.8 / (700 + 3 * 300))^2_000)
        @test abs(mean(draws) - expected) < 4 * std(draws) / sqrt(reps)
        @test protected(FavourVaccinated(3.0), 2_000, 0.8, 600, 300, 1_000) <
            protected(FavourVaccinated(1.0), 2_000, 0.8, 600, 300, 1_000)
        @test_throws ArgumentError FavourVaccinated(-1.0)
    end

    @testset "the protected accumulate in the counts and feed the next batch" begin
        susceptibles = [1_000]
        protected = [0]
        used = [[0]]
        c = EpiBranch._campaign(
            budget(mass(dosing = FavourVaccinated(0.0)), 5_000.0), (0.0, Inf), nothing
        )
        EpiBranch._vaccinate_population!(
            StableRNG(1), [c], used, 14.0, 0, susceptibles, protected, [1_000]
        )
        @test protected[1] > 0
        @test susceptibles[1] == 1_000 - protected[1]
    end

    @testset "simulate_by_generation takes the protected out of the susceptibles" begin
        model = BranchingProcess(Poisson(4.0); population = Population(100_000))
        campaign = budget(
            mass(), Steps([14.0, 28.0], [0.0, 50_000.0, 0.0]); period = 14.0, unused_budget = ExpireUnused()
        )
        spec = ModelSpec(model; interventions = [campaign])
        start(i) = EpiBranch.GenerationCounts(
            100, 100, 100_000, 90_000, 100_000, 0, 14.0, PerContactDepletion(),
            StableRNG(i), Vector{Int}[], [0]
        )
        # The 50,000 doses are given when the second generation starts. Each
        # reaches a susceptible with probability s / n and protects it with
        # probability 0.9, so a share 1 - exp(-0.9 * 50,000 / 100,000) of those
        # then susceptible is protected.
        reps = 100
        shares = map(1:reps) do i
            before = simulate(spec, start(i); max_generations = 1)
            after = simulate(spec, before; max_generations = 2)
            infected = after.counts
            @test after.doses_used == [[50_000]]
            (before.susceptibles - after.susceptibles - infected) / before.susceptibles
        end
        @test abs(mean(shares) - (1 - exp(-0.45))) < 4 * std(shares) / sqrt(reps)
    end

    @testset "a budget per type doses one type only" begin
        campaign = budget(
            mass(), PerGroup(2 => 40_000.0, 1 => 0.0)
        )
        model = BranchingProcess(
            [0.0 0.0; 0.0 0.0], R -> Poisson(R), EpiBranch.NoGenerationTime();
            population = Population([50_000, 50_000])
        )
        c = EpiBranch._by_type(
            EpiBranch._campaign(campaign, (0.0, Inf), nothing), EpiBranch.population(model)
        )
        susceptibles = [50_000, 50_000]
        used = [[0, 0]]
        EpiBranch._vaccinate_population!(
            StableRNG(1), [c], used, 14.0, 0, susceptibles, [0, 0], [50_000, 50_000]
        )
        @test susceptibles[1] == 50_000
        @test susceptibles[2] < 50_000 - 10_000
        @test used == [[0, 40_000]]
    end

    @testset "a shared budget is spread over the types by size" begin
        campaign = budget(mass(dosing = _NeverWastes()), 10_000.0)
        c = EpiBranch._campaign(campaign, (0.0, Inf), nothing)
        reps = 100
        firsts = map(1:reps) do i
            susceptibles = [20_000, 80_000]
            used = [[0]]
            EpiBranch._vaccinate_population!(
                StableRNG(i), [c], used, 14.0, 0, susceptibles, [0, 0], [20_000, 80_000]
            )
            @test sum(susceptibles) == 100_000 - 10_000
            @test used == [[10_000]]
            20_000 - susceptibles[1]
        end
        @test abs(mean(firsts) - 2_000) < 4 * std(firsts) / sqrt(reps)
    end

    @testset "a campaign the counts cannot apply is refused" begin
        model = BranchingProcess(Poisson(2.0); population = Population(1_000))
        run(iv) = simulate_by_generation(
            ModelSpec(model; interventions = [iv]); max_generations = 2,
            generation_length = 14.0
        )
        @test_throws ArgumentError run(mass())
        @test_throws ArgumentError run(budget(mass(coverage = 0.5), 100.0))
        @test_throws ArgumentError run(budget(mass(delay_to_immunity = 7.0), 100.0))
        @test_throws ArgumentError run(budget(mass(efficacy = (rng, ind) -> 0.9), 100.0))
        @test run(budget(mass(), 100.0)) isa EpiBranch.GenerationCounts
    end

    @testset "individual-based runs agree with the counts" begin
        n = 2_000
        campaign = budget(mass(), 1_000.0)
        timed = BranchingProcess(Poisson(2.0), Gamma(2.0, 2.0); population = Population(n))
        untimed = BranchingProcess(Poisson(2.0); population = Population(n))
        reps = 150
        rng = StableRNG(7)
        individual = [
            simulate(
                ModelSpec(timed; interventions = [campaign]); n_initial = 10,
                max_generations = 60, max_cases = 10^6, rng
            ).cumulative_cases for _ in 1:reps
        ]
        counts = [
            simulate_by_generation(
                ModelSpec(untimed; interventions = [campaign]); n_initial = 10,
                max_generations = 60, generation_length = 7.0, rng
            ).cumulative_cases for _ in 1:reps
        ]
        se = sqrt(var(individual) / reps + var(counts) / reps)
        @test abs(mean(individual) - mean(counts)) < 4 * se
        # Both are well below the unvaccinated outbreak, which infects most of the
        # population.
        @test mean(individual) < 0.7 * n
    end

    @testset "a window that opens late agrees with the counts" begin
        # The counts date doses by generation interval and the individual-based
        # run by the clock. The two agree to a few per cent, which is looser
        # than the noise.
        n = 2_000
        campaign = Scheduled(
            budget(mass(), 200.0; period = 4.0, unused_budget = ExpireUnused());
            start_time = 12.0, end_time = 40.0
        )
        timed = BranchingProcess(Poisson(2.0), Gamma(2.0, 2.0); population = Population(n))
        untimed = BranchingProcess(Poisson(2.0); population = Population(n))
        reps = 150
        rng = StableRNG(11)
        individual = [
            simulate(
                ModelSpec(timed; interventions = [campaign]); n_initial = 10,
                max_generations = 60, max_cases = 10^6, rng
            ).cumulative_cases for _ in 1:reps
        ]
        counts = [
            simulate_by_generation(
                ModelSpec(untimed; interventions = [campaign]); n_initial = 10,
                max_generations = 60, generation_length = 4.0, rng
            ).cumulative_cases for _ in 1:reps
        ]
        @test abs(mean(individual) - mean(counts)) < 0.1 * mean(counts)
    end

    @testset "population_ledger is opt-in and read through wrappers" begin
        pop = Population(1_000)
        @test EpiBranch.population_ledger(mass(), pop) === nothing
        @test EpiBranch.population_ledger(budget(mass(), Inf), pop) === nothing
        @test EpiBranch.population_ledger(budget(mass(), 5.0), pop) !== nothing
        @test EpiBranch.population_ledger(
            Scheduled(budget(mass(), 5.0); start_time = 3.0), pop
        ) !== nothing
        @test EpiBranch.population_ledger(
            budget(Scheduled(mass(); start_time = 3.0), 5.0), pop
        ) !== nothing
    end

    @testset "a ledger answers by the time asked, in any order" begin
        model = BranchingProcess(Poisson(2.0), Gamma(2.0, 2.0); population = Population(1_000))
        campaign = Scheduled(
            budget(mass(), 300.0; period = 5.0, unused_budget = ExpireUnused());
            start_time = 10.0, end_time = 30.0
        )
        state = simulate(
            ModelSpec(model; interventions = [campaign]); n_initial = 1,
            max_generations = 1, rng = StableRNG(3)
        )
        ledger = EpiBranch.population_ledger(campaign, model.population)
        @test ledger !== nothing
        share(t) = EpiBranch.protected_share(ledger, state, 1, t)
        late = share(50.0)
        @test late > 0
        @test share(5.0) == 0
        @test share(10.0) == 0
        @test 0 < share(12.0) < late
        @test share(50.0) == late
    end

    @testset "a period's doses leave out the cases infected after it" begin
        model = BranchingProcess(Poisson(3.0), Gamma(2.0, 2.0); population = Population(100))
        campaign = budget(mass(dosing = _NeverWastes()), 90.0)
        state = simulate(model; n_initial = 1, max_generations = 4, rng = StableRNG(5))
        @test state.cumulative_cases > 20
        ledger = EpiBranch.population_ledger(campaign, model.population)
        # Only the first case is infected when the doses are given.
        @test EpiBranch.protected_share(ledger, state, 1, 1.0e3) * 100 == 90
    end

    @testset "without a population or a limit each person is dosed in turn" begin
        open_model = BranchingProcess(Poisson(2.0), Gamma(2.0, 2.0))
        state = simulate(
            ModelSpec(open_model; interventions = [budget(mass(), 5.0)]);
            n_initial = 10, max_generations = 3, rng = StableRNG(1)
        )
        @test count(is_vaccinated, individuals(state)) == 5
        closed = BranchingProcess(Poisson(2.0), Gamma(2.0, 2.0); population = Population(500))
        state = simulate(
            ModelSpec(closed; interventions = [budget(mass(), Inf)]);
            n_initial = 10, max_generations = 3, rng = StableRNG(1)
        )
        @test !haskey(state.scratch, :_mass_ledgers)
        @test any(is_vaccinated, individuals(state))
    end

    @testset "an individual-based run continues without redosing" begin
        campaign = budget(mass(), 1_000.0)
        model = BranchingProcess(Poisson(2.0), Gamma(2.0, 2.0); population = Population(2_000))
        spec = ModelSpec(model; interventions = [campaign])
        state = simulate(spec; n_initial = 10, max_time = 8.0, max_cases = 10^6, rng = StableRNG(2))
        @test state.continuation !== nothing
        state = simulate(spec, state; max_time = 40.0)
        @test !any(ind -> get(ind.state, :vaccinated, false), individuals(state))
        @test only(state.scratch[:_mass_ledgers]).used == [1_000]
        @test only(state.scratch[:_mass_ledgers]).history[end][1] <= 1_000
    end
end

# A model written against `contacts_of` that creates its own contacts and has
# a population.
struct _MintedPopulation <: EpiBranch.TransmissionModel
    k::Int
    size::Int
end
EpiBranch.population_size(m::_MintedPopulation) = Population(m.size)
function EpiBranch.contacts_of(m::_MintedPopulation, parent, state::EpiBranch.SimulationState)
    return [
        (
            t = parent.infection_time + rand(state.rng);
            (EpiBranch.make_contact!(state, parent, t), t)
        )
            for _ in 1:(m.k)
    ]
end

@testset "A budgeted MassVaccination protects people on a contacts_of model" begin
    model = _MintedPopulation(3, 400)
    campaign = CapacityConstrained(
        MassVaccination(; efficacy = 1.0, eligibility_time = 0.0); budget_per_period = 380.0
    )
    run(ivs, seed) = simulate(
        ModelSpec(model; interventions = ivs); n_initial = 2,
        max_generations = 5, rng = StableRNG(seed)
    )
    cases(state) = count(is_infected, individuals(state))
    open_sizes = [cases(run([], i)) for i in 1:30]
    vaccinated = [run([campaign], i) for i in 1:30]
    @test all(s -> haskey(s.scratch, :_mass_ledgers), vaccinated)
    @test mean(cases.(vaccinated)) < 0.5 * mean(open_sizes)
end
