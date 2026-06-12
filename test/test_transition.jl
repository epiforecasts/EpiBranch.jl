@testset "Generic Transition" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.0)
    bp(progression; attributes = NoAttributes()) = BranchingProcess(
        Poisson(1.5), Exponential(5.0);
        progression = progression, attributes = attributes)

    @testset "writes :state and :state_time, anchored on :infection" begin
        latent = Transition(:infectious, from = :infection, delay = (rng, ind) -> 2.0)
        state = simulate(bp([latent]; attributes = clinical); max_cases = 30, rng = StableRNG(1))
        for ind in state.individuals
            @test ind.state[:infectious] == true
            @test ind.state[:infectious_time] ≈ ind.infection_time + 2.0
        end
    end

    @testset "chains: a state measured from an earlier transition's state" begin
        severe = Transition(:severe, from = :onset, delay = (rng, ind) -> 1.0)
        state = simulate(bp([severe]; attributes = clinical); max_cases = 30, rng = StableRNG(2))
        for ind in state.individuals
            @test ind.state[:severe_time] ≈ onset_time(ind) + 1.0
        end
    end

    @testset "probability gate" begin
        never = Transition(:flagged, delay = (rng, ind) -> 1.0, probability = 0.0)
        always = Transition(:marked, delay = (rng, ind) -> 1.0, probability = 1.0)
        state = simulate(bp([never, always]; attributes = clinical); max_cases = 30, rng = StableRNG(3))
        for ind in state.individuals
            @test ind.state[:flagged] == false
            @test ind.state[:flagged_time] == Inf
            @test ind.state[:marked] == true
        end
    end

    @testset "skips when the from-state was never reached" begin
        # Nobody becomes severe (probability 0), so death measured from
        # :severe never happens for anyone.
        severe = Transition(:severe, from = :onset, delay = (rng, ind) -> 1.0,
            probability = 0.0)
        died = Transition(:died, from = :severe, delay = (rng, ind) -> 1.0,
            terminal = true)
        state = simulate(bp([severe, died]; attributes = clinical); max_cases = 30, rng = StableRNG(4))
        for ind in state.individuals
            @test ind.state[:died] == false
            @test ind.state[:died_time] == Inf
            @test !haskey(ind.state, :outcome)
        end
    end

    @testset "terminal arbitration picks the earliest" begin
        died = Transition(:died, from = :onset, delay = (rng, ind) -> 2.0,
            probability = 1.0, terminal = true)
        recovered = Transition(:recovered, from = :onset, delay = (rng, ind) -> 5.0,
            terminal = true)
        state = simulate(bp([died, recovered]; attributes = clinical); max_cases = 30, rng = StableRNG(5))
        for ind in state.individuals
            @test ind.state[:outcome] == :died
            @test ind.state[:outcome_time] ≈ onset_time(ind) + 2.0
        end
    end

    @testset "asymptomatic case skips an onset-anchored transition" begin
        all_asymp = clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 1.0)
        sev = Transition(:severe, from = :onset, delay = (rng, ind) -> 1.0)
        state = simulate(bp([sev]; attributes = all_asymp); max_cases = 30, rng = StableRNG(6))
        for ind in state.individuals
            @test ind.state[:severe] == false
        end
    end
end
