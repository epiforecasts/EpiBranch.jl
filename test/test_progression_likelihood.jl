# A transition with no `transition_loglik` method of its own, as one
# written before `progression_loglik` existed.
struct _UntrackedTransition <: EpiBranch.AbstractClinicalTransition end

@testset "Progression likelihood" begin
    clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

    @testset "matches a hand-computed sum over the progression" begin
        reporting_delay = LogNormal(1.0, 0.3)
        recovery_delay = Exponential(3.0)
        reporting = Reporting(delay = reporting_delay, probability = 0.7)
        recovery = Recovery(delay = recovery_delay)
        spec = ModelSpec(
            BranchingProcess(Poisson(1.5), Exponential(5.0));
            progression = [reporting, recovery], attributes = clinical
        )
        state = simulate(spec; max_cases = 200, rng = StableRNG(1))

        expected = 0.0
        for ind in state.individuals
            get(ind.state, :infected, false) || continue
            onset = onset_time(ind)
            isfinite(onset) || continue
            expected += if ind.state[:reported]
                log(0.7) + logpdf(reporting_delay, ind.state[:reporting_time] - onset)
            else
                log(0.3)
            end
            expected += logpdf(recovery_delay, ind.state[:recovery_candidate_time] - onset)
        end
        @test expected < 0  # sanity: the hand-computed sum is a genuine log-likelihood
        @test progression_loglik(spec, state) ≈ expected
        @test progression_loglik(spec, state.individuals) ≈ expected
    end

    @testset "never-infected individuals contribute nothing" begin
        attrs = [clinical, transmission_traits(susceptibility = 0.4)]
        spec = ModelSpec(
            BranchingProcess(Poisson(3.0), Exponential(5.0));
            progression = [Recovery(delay = Exponential(2.0))], attributes = attrs
        )
        state = simulate(spec; max_cases = 300, rng = StableRNG(2))
        @test any(!get(ind.state, :infected, false) for ind in state.individuals)
        infected_only = filter(ind -> get(ind.state, :infected, false), state.individuals)
        @test progression_loglik(spec, state) ≈ progression_loglik(spec, infected_only)
    end

    @testset "an unreached anchor (asymptomatic case) contributes zero" begin
        all_asymp = clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 1.0
        )
        spec = ModelSpec(
            BranchingProcess(Poisson(1.5), Exponential(5.0));
            progression = [Reporting(delay = LogNormal(1.0, 0.3))], attributes = all_asymp
        )
        state = simulate(spec; max_cases = 50, rng = StableRNG(3))
        @test all(!isfinite(onset_time(ind)) for ind in state.individuals)
        @test progression_loglik(spec, state) == 0.0
    end

    @testset "an outcome the model rules out has zero density" begin
        reporting = Reporting(delay = LogNormal(1.0, 0.3), probability = 0.0)
        spec = ModelSpec(
            BranchingProcess(Poisson(0.0));
            progression = [reporting], attributes = clinical
        )
        ind = Individual(id = 1, infection_time = 0.0)
        ind.state[:infected] = true
        ind.state[:onset_time] = 1.0
        ind.state[:reported] = true          # impossible under probability = 0.0
        ind.state[:reporting_time] = 2.0
        @test progression_loglik(spec, [ind]) == -Inf
    end

    @testset "a Function delay cannot be evaluated" begin
        latent = Transition(:infectious, from = :infection, delay = (rng, ind) -> 2.0)
        spec = ModelSpec(
            BranchingProcess(Poisson(0.0));
            progression = [latent], attributes = clinical
        )
        state = simulate(spec; max_cases = 5, rng = StableRNG(4))
        @test_throws ArgumentError progression_loglik(spec, state)
    end

    @testset "a death contributes its gate and, when it happened, its delay" begin
        death = Death(delay = Exponential(2.0), probability = 0.25)
        spec = ModelSpec(
            BranchingProcess(Poisson(0.0));
            progression = [death], attributes = clinical
        )
        function loglik(candidate_time)
            ind = Individual(id = 1, infection_time = 0.0)
            ind.state[:infected] = true
            ind.state[:onset_time] = 1.0
            ind.state[:death_candidate_time] = candidate_time
            return progression_loglik(spec, [ind])
        end

        # Died at 4.0, three days after the onset the delay is measured from.
        @test loglik(4.0) ≈ log(0.25) + logpdf(Exponential(2.0), 3.0)
        # Survived: the gate alone, with no delay to evaluate.
        @test loglik(Inf) ≈ log(1 - 0.25)
    end

    @testset "a hospitalisation contributes its gate and, when it happened, its delay" begin
        admission_delay = LogNormal(1.0, 0.4)
        hospitalisation = Hospitalisation(delay = admission_delay, probability = 0.2)
        spec = ModelSpec(
            BranchingProcess(Poisson(0.0));
            progression = [hospitalisation], attributes = clinical
        )
        function loglik(admitted, admission_time)
            ind = Individual(id = 1, infection_time = 0.0)
            ind.state[:infected] = true
            ind.state[:onset_time] = 1.0
            ind.state[:admitted] = admitted
            ind.state[:admission_time] = admission_time
            return progression_loglik(spec, [ind])
        end

        # Admitted at 3.5, two and a half days after onset.
        @test loglik(true, 3.5) ≈ log(0.2) + logpdf(admission_delay, 2.5)
        # Not admitted: the gate alone, with no delay to evaluate.
        @test loglik(false, Inf) ≈ log(1 - 0.2)
    end

    @testset "a fixed numeric delay has zero log-density at its value and rules out any other" begin
        reporting = Reporting(delay = 2.0, probability = 0.6)
        spec = ModelSpec(
            BranchingProcess(Poisson(0.0));
            progression = [reporting], attributes = clinical
        )
        function loglik(onset, reporting_time)
            ind = Individual(id = 1, infection_time = 0.0)
            ind.state[:infected] = true
            ind.state[:onset_time] = onset
            ind.state[:reported] = true
            ind.state[:reporting_time] = reporting_time
            return progression_loglik(spec, [ind])
        end

        # Only the gate contributes when the delay matches exactly.
        @test loglik(1.0, 3.0) ≈ log(0.6)
        # Floating-point subtraction leaves 3.3 - 1.3 just off 2.0; still a match.
        @test 3.3 - 1.3 != 2.0
        @test loglik(1.3, 3.3) ≈ log(0.6)
        @test loglik(1.0, 3.5) == -Inf
    end

    @testset "a shared-draw probability gate cannot be evaluated on a hand-built individual" begin
        death_p, _ = exclusive_probabilities([0.64, 0.36])
        death = Death(delay = Exponential(2.0), probability = death_p)
        spec = ModelSpec(
            BranchingProcess(Poisson(0.0));
            progression = [death], attributes = clinical
        )
        ind = Individual(id = 1, infection_time = 0.0)
        ind.state[:infected] = true
        ind.state[:onset_time] = 1.0
        ind.state[:death_candidate_time] = Inf
        n_keys = length(ind.state)
        @test_throws ArgumentError progression_loglik(spec, [ind])
        # deterministic, and does not cache a spurious draw on the individual
        @test_throws ArgumentError progression_loglik(spec, [ind])
        @test length(ind.state) == n_keys
    end

    @testset "a transition without its own method needs one to be evaluated" begin
        spec = ModelSpec(
            BranchingProcess(Poisson(0.0));
            progression = [_UntrackedTransition()], attributes = clinical
        )
        state = simulate(spec; max_cases = 5, rng = StableRNG(5))
        @test_throws ArgumentError progression_loglik(spec, state)
    end
end
