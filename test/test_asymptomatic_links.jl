# The models of the asymptomatic-cases tutorial: each link between symptoms,
# infectiousness and a within-host trajectory is stated by the user through an
# existing seam, with no built-in assumption.

using Test, EpiBranch, Distributions, StableRNGs, Statistics

function _link_summary(model; n = 300)
    states = simulate(model, n; max_generations = 3, max_cases = 5000, rng = StableRNG(1))
    asymptomatic, secondary, before_onset = Bool[], Int[], Float64[]
    for state in states
        by_id = Dict(ind.id => ind for ind in state.individuals)
        for ind in state.individuals
            EpiBranch.is_infected(ind) && ind.generation < 2 || continue
            infected = [
                by_id[id] for id in ind.secondary_case_ids
                    if EpiBranch.is_infected(by_id[id])
            ]
            push!(asymptomatic, is_asymptomatic(ind))
            push!(secondary, length(infected))
            if !is_asymptomatic(ind) && !isempty(infected)
                push!(before_onset, mean(c.infection_time < onset_time(ind) for c in infected))
            end
        end
    end
    return (
        asymptomatic = mean(asymptomatic),
        R_asymptomatic = mean(secondary[asymptomatic]),
        R_symptomatic = mean(secondary[.!asymptomatic]),
        before_onset = mean(before_onset),
    )
end

function _draw_trajectory(rng)
    peak_time = rand(rng, Gamma(6.0, 0.7))
    peak = rand(rng, LogNormal(0.0, 0.6))
    clearance = rand(rng, Gamma(8.0, 0.1))
    load = t -> t < peak_time ? peak * t / peak_time : peak * exp(-clearance * (t - peak_time))
    return (; load, peak_time, peak)
end

function _both_follow(rng, ind)
    trajectory = _draw_trajectory(rng)
    ind.state[:trajectory] = trajectory.load
    symptomatic = trajectory.peak > 0.8
    onset = trajectory.peak_time * 0.8 / trajectory.peak
    ind.state[:asymptomatic] = !symptomatic
    ind.state[:incubation_period] = onset
    return nothing
end

@testset "Links between symptoms, infectiousness and trajectory" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.3
    )
    plain = BranchingProcess(Poisson(3.0), Exponential(5.0))
    less_shedding = [
        clinical,
        transmission_traits(infectiousness = (rng, ind) -> ind.state[:asymptomatic] ? 0.5 : 1.0),
    ]

    @testset "a proportion asymptomatic links nothing else" begin
        s = _link_summary(ModelSpec(plain; attributes = clinical))
        @test s.asymptomatic ≈ 0.3 atol = 0.05
        @test s.R_asymptomatic ≈ s.R_symptomatic atol = 0.4
    end

    @testset "less shedding (biology) and more contacts (behaviour) are separate" begin
        biology = _link_summary(ModelSpec(plain; attributes = less_shedding))
        @test biology.R_asymptomatic / biology.R_symptomatic ≈ 0.5 atol = 0.12

        contacts = (rng, ind) -> rand(rng, Poisson(ind.state[:asymptomatic] ? 4.5 : 3.0))
        behaviour = _link_summary(
            ModelSpec(BranchingProcess(contacts, Exponential(5.0)); attributes = clinical)
        )
        @test behaviour.R_asymptomatic / behaviour.R_symptomatic ≈ 1.5 atol = 0.2

        both = _link_summary(
            ModelSpec(BranchingProcess(contacts, Exponential(5.0)); attributes = less_shedding)
        )
        @test both.R_asymptomatic / both.R_symptomatic ≈ 0.75 atol = 0.15
    end

    @testset "a shorter infectious period" begin
        progression = [
            Transition(
                :recovered, from = :infection, terminal = true,
                delay = (rng, ind) -> ind.state[:asymptomatic] ? 3.0 : 10.0
            ),
        ]
        window = Infectiousness(Poisson(3.0); kernel = Exponential(5.0), until = (:recovered,))
        s = _link_summary(ModelSpec(BranchingProcess(window); attributes = clinical, progression))
        @test s.R_asymptomatic < 0.7 * s.R_symptomatic
    end

    @testset "onset linked to infectiousness, asymptomatic timing chosen" begin
        linked = incubation_linked_generation_time(presymptomatic_fraction = 0.3)
        s = _link_summary(ModelSpec(BranchingProcess(Poisson(3.0), linked); attributes = clinical))
        @test s.before_onset ≈ 0.3 atol = 0.06
        own = ind -> is_asymptomatic(ind) ? Exponential(5.0) : linked(ind)
        s = _link_summary(ModelSpec(BranchingProcess(Poisson(3.0), own); attributes = clinical))
        @test s.before_onset ≈ 0.3 atol = 0.06
    end

    @testset "asymptomatic cases with their own incubation period" begin
        own = clinical_presentation(
            incubation_period = (rng, ind) -> rand(
                rng, ind.state[:asymptomatic] ? LogNormal(1.2, 0.5) : LogNormal(1.5, 0.5)
            ),
            prob_asymptomatic = 0.3
        )
        linked = incubation_linked_generation_time(presymptomatic_fraction = 0.3)
        model = ModelSpec(BranchingProcess(Poisson(3.0), linked); attributes = own)
        states = simulate(model, 100; max_generations = 2, max_cases = 2000, rng = StableRNG(3))
        cases = [ind for s in states for ind in s.individuals if EpiBranch.is_infected(ind)]
        asymptomatic = [incubation_period(ind) for ind in cases if is_asymptomatic(ind)]
        symptomatic = [incubation_period(ind) for ind in cases if !is_asymptomatic(ind)]
        @test !any(isnan, asymptomatic)
        @test mean(log, asymptomatic) < mean(log, symptomatic)
        @test all(ind -> isnan(onset_time(ind)), filter(is_asymptomatic, cases))
    end

    @testset "trajectory drives onset and infectiousness" begin
        kernel = ind -> HazardKernel(
            t -> 0.4 * EpiBranch.trajectory_load(ind, t); upper = 60.0
        )
        window = Infectiousness(Poisson(6.0); kernel)

        both = _link_summary(ModelSpec(BranchingProcess(window); attributes = _both_follow))
        @test both.R_asymptomatic < 0.8 * both.R_symptomatic

        # Only infectiousness follows it: symptoms stay independent of load.
        store_only = within_host((rng, ind) -> _draw_trajectory(rng).load)
        infectiousness = _link_summary(
            ModelSpec(BranchingProcess(window); attributes = [clinical, store_only])
        )
        @test infectiousness.R_asymptomatic ≈ infectiousness.R_symptomatic atol = 0.7

        # Only onset follows it: transmission uses a plain generation time.
        onset = _link_summary(ModelSpec(plain; attributes = _both_follow))
        @test onset.R_asymptomatic ≈ onset.R_symptomatic atol = 0.5
        @test 0.0 < onset.asymptomatic < 1.0
    end

    @testset "who is tested and classified follows the trajectory, not only symptoms" begin
        kernel = ind -> HazardKernel(
            t -> 0.4 * EpiBranch.trajectory_load(ind, t); upper = 60.0
        )
        assay = Assay(ct = load -> 40.0 - 3.3 * load, detect = 0.5)
        model = ModelSpec(
            BranchingProcess(Infectiousness(Poisson(6.0); kernel));
            attributes = _both_follow,
            progression = [Testing(assay = assay, from = :infection, delay = Uniform(3.0, 8.0))],
            case_definition = ClinicalCaseDefinition()
        )
        states = simulate(model, 40; max_generations = 3, max_cases = 2000, rng = StableRNG(2))
        cases = [ind for s in states for ind in s.individuals if EpiBranch.is_infected(ind)]
        asymptomatic = filter(is_asymptomatic, cases)
        @test any(ind -> ind.state[:tested], asymptomatic)
        @test any(ind -> isequal(get(ind.state, :case_classification, missing), :confirmed), asymptomatic)
    end
end
