@testset "Branching process" begin
    @testset "Subcritical outbreak goes extinct" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(0.5), Exponential(5.0))
        state = simulate(model; rng = rng)
        @test state.extinct
        @test state.cumulative_cases < 100
    end

    @testset "Supercritical outbreak grows" begin
        rng = StableRNG(123)
        model = BranchingProcess(Poisson(3.0), Exponential(5.0))
        state = simulate(model; max_cases = 500, rng = rng)
        @test state.cumulative_cases >= 500
    end

    @testset "Multiple index cases" begin
        rng = StableRNG(99)
        model = BranchingProcess(Poisson(0.5), Exponential(5.0))
        state = simulate(model; n_initial = 5, rng = rng)
        index_chains = [ind.chain_id for ind in state.individuals if ind.parent_id == 0]
        @test length(unique(index_chains)) == 5
    end

    @testset "Generation tracking" begin
        rng = StableRNG(77)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        state = simulate(model; max_generations = 3, rng = rng)
        max_gen = maximum(ind.generation for ind in state.individuals)
        @test max_gen <= 3
    end

    @testset "Infection times increase across generations" begin
        rng = StableRNG(55)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        state = simulate(model; max_cases = 100, rng = rng)

        for ind in filter(is_infected, state.individuals)
            if ind.parent_id > 0
                parent_idx = findfirst(i -> i.id == ind.parent_id, state.individuals)
                parent = state.individuals[parent_idx]
                @test ind.infection_time > parent.infection_time
            end
        end
    end

    @testset "Incubation period sets onset times" begin
        rng = StableRNG(33)
        init_fn = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
        state = simulate(
            ModelSpec(BranchingProcess(Poisson(1.5), Exponential(5.0)); attributes = init_fn);
            max_cases = 50, rng = rng
        )

        for ind in filter(is_infected, state.individuals)
            @test !isnan(onset_time(ind))
            @test onset_time(ind) >= ind.infection_time
        end
    end

    @testset "incubation_period accessor" begin
        ind = Individual(id = 1, infection_time = 2.0)
        @test isnan(incubation_period(ind))   # onset not set yet
        ind.state[:onset_time] = 5.5
        @test incubation_period(ind) ≈ 3.5
    end

    @testset "Generation time can depend on individual state" begin
        # An attributes function stores a per-individual latent and sets
        # onset from it; the generation_time function reads the same
        # latent, so onset and generation time come from one draw rather
        # than two independent ones.
        rng = StableRNG(7)
        init_fn = function (rng, ind)
            scale = 3.0 + rand(rng)
            ind.state[:gt_scale] = scale
            ind.state[:onset_time] = ind.infection_time + scale
        end
        gt_fn = ind -> Exponential(ind.state[:gt_scale])
        state = simulate(
            ModelSpec(BranchingProcess(Poisson(1.2), gt_fn); attributes = init_fn);
            max_cases = 50, rng = rng
        )

        # Require secondary transmission so the generation_time function
        # is actually exercised, not just the index case.
        secondary = filter(
            ind -> is_infected(ind) && ind.parent_id > 0,
            state.individuals
        )
        @test !isempty(secondary)
        for ind in secondary
            @test incubation_period(ind) ≈ ind.state[:gt_scale]
        end
    end

    @testset "Truncated generation time enforces minimum" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(3.0), truncated(Exponential(5.0), 3.0, Inf))
        state = simulate(
            model;
            condition = 20:500, max_cases = 500, rng = rng
        )

        for ind in filter(is_infected, state.individuals)
            if ind.parent_id > 0
                parent_idx = findfirst(i -> i.id == ind.parent_id, state.individuals)
                parent = state.individuals[parent_idx]
                @test ind.infection_time - parent.infection_time >= 3.0 - 1.0e-10
            end
        end
    end
end

struct CallableContactInterval end
(::CallableContactInterval)(ind) = Exponential(2.0)
struct CallableOffspring end
(::CallableOffspring)(rng, ind) = rand(rng, Poisson(0.6))
struct CallableStateOffspring end
(::CallableStateOffspring)(rng, ind) = error("The state-aware method should be used")
(::CallableStateOffspring)(rng, ind, state) = state.cumulative_cases < 8 ? 1 : 0
struct CallableOffspringFamily end
(::CallableOffspringFamily)(R) = Poisson(R)

@testset "Callable model inputs" begin
    signature(state) = [
        (i.parent_id, i.infection_time, i.generation)
            for i in state.individuals
    ]
    compare(a, b) = @test signature(simulate(a; rng = StableRNG(718))) ==
        signature(simulate(b; rng = StableRNG(718)))
    compare(
        BranchingProcess(CallableOffspring(), CallableContactInterval()),
        BranchingProcess(
            (rng, ind) -> rand(rng, Poisson(0.6)),
            ind -> Exponential(2.0)
        )
    )
    compare(
        BranchingProcess(Poisson(0.6), CallableContactInterval()),
        BranchingProcess(Poisson(0.6), Exponential(2.0))
    )
    compare(
        BranchingProcess(CallableStateOffspring(), CallableContactInterval()),
        BranchingProcess(
            (rng, ind, state) -> state.cumulative_cases < 8 ? 1 : 0,
            Exponential(2.0)
        )
    )
    M = [0.3 0.1; 0.2 0.4]
    compare(
        BranchingProcess(M, CallableOffspringFamily(), CallableContactInterval()),
        BranchingProcess(M, R -> Poisson(R), Exponential(2.0))
    )
    mixed = ClusterMixed(Poisson, Gamma(2.0, 0.1))
    compare(
        BranchingProcess(mixed, CallableContactInterval()),
        BranchingProcess(mixed, Exponential(2.0))
    )
    @test EpiBranch.single_type_offspring(
        BranchingProcess(
            mixed,
            CallableContactInterval()
        )
    ) === mixed
    windows = (
        Infectiousness(Poisson(0.2); kernel = CallableContactInterval()),
        Infectiousness(Poisson(0.3); kernel = CallableContactInterval()),
    )
    @test BranchingProcess(windows...).infectiousness == windows
end

struct ExternalOffspringRule end
EpiBranch.draw_offspring(rng, ::ExternalOffspringRule, ind, state) = 0

@testset "External offspring dispatch" begin
    model = BranchingProcess(Infectiousness(ExternalOffspringRule()))
    @test simulate(model; rng = StableRNG(42)).cumulative_cases == 1
end

# Infection times of one outbreak up to `horizon`, drawn by recursing down the
# tree from an index case at time 0. Written to share nothing with the
# simulation, as a check on the timing of incidence, which has no closed form.
function _reference_infection_times(rng, offspring, generation_time, horizon)
    times = [0.0]
    to_visit = [0.0]
    while !isempty(to_visit)
        t = pop!(to_visit)
        for _ in 1:rand(rng, offspring)
            child = t + rand(rng, generation_time)
            child <= horizon || continue
            push!(times, child)
            push!(to_visit, child)
        end
    end
    return times
end

@testset "Incidence over time matches an independent tree sampler" begin
    offspring, generation_time, horizon = Poisson(1.2), Gamma(2.5, 2.0), 30.0
    n = 4000
    simulated = simulate(
        BranchingProcess(offspring, generation_time), n;
        max_time = horizon, max_cases = nothing, max_generations = nothing,
        rng = StableRNG(7)
    )
    rng = StableRNG(8)
    reference = [
        _reference_infection_times(rng, offspring, generation_time, horizon)
            for _ in 1:n
    ]
    sim_times = [
        [ind.infection_time for ind in s.individuals if is_infected(ind)]
            for s in simulated
    ]
    for day in (5.0, 10.0, 20.0, 30.0)
        a = [count(<=(day), ts) for ts in sim_times]
        b = [count(<=(day), ts) for ts in reference]
        se = sqrt(var(a) / n + var(b) / n)
        @test abs(mean(a) - mean(b)) < 4 * se
    end
end

@testset "A run cut short lists no exposure after it ends" begin
    # A depth-2 ring draws the contacts of members it reaches before they
    # settle, and the run can stop before they do.
    spec = ModelSpec(
        BranchingProcess(Poisson(4.0), Gamma(4.0, 1.0));
        interventions = [
            Isolation(onset_to_isolation_delay = Dirac(6.0), duration = Inf),
            ContactTracing(
                probability = 0.5, isolation_to_trace_delay = Dirac(0.1),
                action = Quarantine(duration = Inf), depth = 2
            ),
        ],
        attributes = clinical_presentation(incubation_period = Exponential(0.5))
    )
    for (opts, cutoff) in (((; max_time = 8.0), s -> 8.0), ((; max_cases = 15), s -> s.max_infection_time))
        listed_late = 0
        from_undecided = 0
        for seed in 1:30
            s = simulate(spec; rng = StableRNG(seed), opts...)
            # A run that dies out decides every contact it drew.
            stop_t = s.extinct ? Inf : cutoff(s)
            for ind in s.individuals
                ind.parent_id == 0 && continue
                isnan(ind.infection_time) && continue
                ind.infection_time > stop_t && (listed_late += 1)
                isnan(s.individuals[ind.parent_id].infection_time) && (from_undecided += 1)
            end
        end
        @test listed_late == 0
        @test from_undecided == 0
    end
end

@testset "A contact timed before its infector's infection is dropped" begin
    # Transmission timed from onset can fall before the infection itself,
    # where the infector could not yet transmit.
    spec = ModelSpec(
        BranchingProcess(
            Infectiousness(Poisson(2.0); from = :onset, kernel = Normal(0.0, 3.0))
        );
        attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
    )
    states = simulate(
        spec, 200; max_cases = 2000, max_time = nothing, max_generations = nothing,
        rng = StableRNG(3)
    )
    # No one is infected, or even exposed, before their infector's infection.
    early = 0
    for s in states, ind in s.individuals
        ind.parent_id == 0 && continue
        isnan(ind.infection_time) && continue
        ind.infection_time < s.individuals[ind.parent_id].infection_time && (early += 1)
    end
    @test early == 0
    @test any(s -> s.cumulative_cases > 1, states)

    # Dropped before tracing or a ring can reach it: a run that dies out
    # leaves no one without an exposure time, traced or dosed.
    ring_spec = ModelSpec(
        BranchingProcess(
            Infectiousness(Poisson(0.9); from = :onset, kernel = Normal(-2.0, 2.0))
        );
        interventions = [
            Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf),
            ContactTracing(
                probability = 1.0, isolation_to_trace_delay = Exponential(0.5),
                action = Quarantine(duration = Inf)
            ),
            RingVaccination(efficacy = 0.9),
        ],
        attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
    )
    phantom = 0
    for seed in 1:100
        s = simulate(
            ring_spec; max_cases = nothing, max_time = nothing, max_generations = nothing,
            rng = StableRNG(seed)
        )
        s.extinct || continue
        phantom += count(ind -> isnan(ind.infection_time), s.individuals)
    end
    @test phantom == 0
end

@testset "A recorder lists each branching-process contact once" begin
    spec = ModelSpec(
        BranchingProcess(Poisson(3.0), Exponential(2.0));
        attributes = transmission_traits(susceptibility = 0.5),
        recorder = ExposedContactRecorder()
    )
    s = simulate(spec; max_cases = 50, n_initial = 5, rng = StableRNG(4))
    rows = contacts(s)
    @test nrow(rows) == nrow(unique(rows[:, [:from, :to]]))
    @test any(!, rows.infected)
end
