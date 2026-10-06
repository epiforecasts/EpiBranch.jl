using ForwardDiff

# A custom transition that gates itself, in the shape the extending guide
# documents: its own keys, its gate under `probability`.
struct _GatedVisit{P} <: EpiBranch.AbstractClinicalTransition
    delay::Float64
    probability::P
end
function EpiBranch.initialise_individual!(::_GatedVisit, ind, state)
    ind.state[:visited] = false
    ind.state[:visit_time] = Inf
    return nothing
end
function EpiBranch.resolve_individual!(t::_GatedVisit, ind, state)
    time = EpiBranch.transition_time(
        state.rng, ind, ind.infection_time, t.delay; probability = t.probability
    )
    time === nothing && return nothing
    ind.state[:visited] = true
    ind.state[:visit_time] = time
    return nothing
end

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
        @test_throws "has no density to evaluate" progression_loglik(spec, state)
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

    @testset "a shared-draw group keeps the bucket its draw selected" begin
        # The bucket's width is the probability of selecting it, which a run's
        # 0 or 1 hides: without it the case-fatality ratio would reach the
        # likelihood only through the delays.
        death_p, recovered_p = exclusive_probabilities([0.64, 0.36])
        death = Death(delay = Exponential(2.0), probability = death_p)
        recovery = Transition(
            :recovered, from = :onset, delay = Exponential(3.0),
            probability = recovered_p, terminal = true
        )
        spec = ModelSpec(
            BranchingProcess(Poisson(0.0));
            progression = [death, recovery], attributes = clinical
        )
        died = Individual(id = 1, infection_time = 0.0)
        died.state[:infected] = true
        died.state[:onset_time] = 1.0
        died.state[:death_candidate_time] = 3.0
        died.state[:recovered] = false
        n_keys = length(died.state)
        @test progression_loglik(spec, [died]) ≈
            log(0.64) + logpdf(Exponential(2.0), 2.0)
        # The siblings it passed over say nothing more, and a hand-built
        # individual needs no cached draw to be evaluated.
        @test length(died.state) == n_keys

        recovered = Individual(id = 2, infection_time = 0.0)
        recovered.state[:infected] = true
        recovered.state[:onset_time] = 1.0
        recovered.state[:death_candidate_time] = Inf
        recovered.state[:recovered] = true
        recovered.state[:recovered_time] = 4.0
        @test progression_loglik(spec, [recovered]) ≈
            log(0.36) + logpdf(Exponential(3.0), 3.0)
    end

    @testset "a group still partitions a case an abort cut short" begin
        # The undo restores what a transition recorded and leaves the uniform
        # its group shares, so the sibling after it reads the same draw rather
        # than a fresh one. Without that, the second sibling occurs on three
        # quarters of these cases instead of its own half, and the likelihood
        # has no draw to read.
        first_p, second_p = exclusive_probabilities([0.5, 0.5])
        progression = EpiBranch.AbstractClinicalTransition[
            Transition(:first_step, from = :infection, delay = 5.0, probability = first_p),
            Transition(:second_step, from = :infection, delay = 0.1, probability = second_p),
        ]
        spec = ModelSpec(BranchingProcess(Poisson(0.0)); progression = progression)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(0.0)), progression, NoAttributes(), StableRNG(7)
        )
        second = 0
        n = 2000
        for _ in 1:n
            ind = Individual(id = 1, infection_time = 0.0)
            ind.state[:infected] = true
            EpiBranch.abort_infection!(ind, 1.0)
            EpiBranch.resolve_transitions!(state, ind)
            second += ind.state[:second_step]::Bool
            # The first step always lands past the abort, so it is undone; the
            # likelihood still reads the group's draw.
            @test isfinite(progression_loglik(spec, [ind]))
        end
        @test isapprox(second / n, 0.5; atol = 0.04)
    end

    @testset "a custom transition's group partitions an aborted case too" begin
        # The undo asks the transition for the records that belong to its
        # group, so a custom transition holding its gate under `probability`,
        # as every documented one does, keeps the draw exactly as a built-in
        # does. Wiring this to the built-in types alone left the split broken
        # at three quarters for anyone else.
        first_p, second_p = exclusive_probabilities([0.5, 0.5])
        progression = AbstractClinicalTransition[
            _GatedVisit(5.0, first_p),
            Transition(:second_step, from = :infection, delay = 0.1, probability = second_p),
        ]
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(0.0)), progression, NoAttributes(), StableRNG(7)
        )
        second = 0
        n = 2000
        for _ in 1:n
            ind = Individual(id = 1, infection_time = 0.0)
            ind.state[:infected] = true
            EpiBranch.abort_infection!(ind, 1.0)
            EpiBranch.resolve_transitions!(state, ind)
            second += ind.state[:second_step]::Bool
        end
        @test isapprox(second / n, 0.5; atol = 0.04)
    end

    @testset "a shared-draw group below 1 holds its shortfall once" begin
        # A group whose probabilities leave room is the one case that needs the
        # draw itself: nothing in the individual's own flags says whether the
        # draw selected a sibling or fell past them all.
        first_p, second_p = exclusive_probabilities([0.3, 0.3])
        spec = ModelSpec(
            BranchingProcess(Poisson(0.0));
            progression = [
                Transition(
                    :hospitalised, from = :onset, delay = Exponential(2.0),
                    probability = first_p
                ),
                Transition(
                    :recovered, from = :onset, delay = Exponential(3.0),
                    probability = second_p, terminal = true
                ),
            ],
            attributes = clinical
        )
        neither = Individual(id = 1, infection_time = 0.0)
        neither.state[:infected] = true
        neither.state[:onset_time] = 1.0
        neither.state[:hospitalised] = false
        neither.state[:recovered] = false
        @test_throws "needs the draw" progression_loglik(spec, [neither])
        neither.state[second_p.key] = 0.9
        @test progression_loglik(spec, [neither]) ≈ log1p(-0.6)
        # A draw that did select a sibling leaves the shortfall out of it.
        selected = Individual(id = 2, infection_time = 0.0)
        selected.state[:infected] = true
        selected.state[:onset_time] = 1.0
        selected.state[:hospitalised] = true
        selected.state[:hospitalised_time] = 3.0
        selected.state[:recovered] = false
        selected.state[second_p.key] = 0.1
        @test progression_loglik(spec, [selected]) ≈
            log(0.3) + logpdf(Exponential(2.0), 2.0)
    end

    @testset "a transition an abort undid is censored at the abort" begin
        # `resolve_transitions!` undoes a transition that would take effect at
        # or after the abort, so what the individual records is that it would
        # have happened no earlier than then. Read as a failed gate instead, an
        # unconditional transition would take the whole outbreak to `-Inf`.
        # An abort leaves the onset `NaN`, so both transitions are anchored
        # before it: a latent period from infection, and a second step from
        # the state that one writes.
        latent = Transition(:infectious, from = :infection, delay = Exponential(2.0))
        treated = Transition(
            :treated, from = :infectious, delay = Exponential(1.0),
            probability = 0.6
        )
        spec = ModelSpec(
            BranchingProcess(Poisson(0.0));
            progression = [latent, treated], attributes = clinical
        )
        ind = Individual(id = 1, infection_time = 0.0)
        ind.state[:infected] = true
        ind.state[:onset_time] = NaN
        ind.state[:infectious] = false
        ind.state[:infectious_time] = Inf
        ind.state[:treated] = false
        ind.state[:infection_aborted_time] = 1.5
        # The latent step is censored at the abort; the second step's own
        # anchor was never reached, so it contributes nothing.
        @test progression_loglik(spec, [ind]) ≈ logccdf(Exponential(2.0), 1.5)
        # With the latent step standing, it contributes as it ever did and the
        # second step is censored from its own anchor.
        ind.state[:infectious] = true
        ind.state[:infectious_time] = 1.2
        @test progression_loglik(spec, [ind]) ≈
            logpdf(Exponential(2.0), 1.2) +
            log1p(-0.6 * cdf(Exponential(1.0), 0.3))
    end

    @testset "an aborted case differentiates" begin
        # `abort_infection!` stores the abort in the individual's own number
        # type, so a dual pool must reach a finite value and gradient rather
        # than a type error.
        function censored(scale)
            ind = Individual(id = 1, infection_time = zero(scale))
            ind.state[:infected] = true
            ind.state[:infectious] = false
            ind.state[:infectious_time] = oftype(scale, Inf)
            EpiBranch.abort_infection!(ind, 1.5 * scale)
            spec = ModelSpec(
                BranchingProcess(Poisson(0.0));
                progression = [
                    Transition(:infectious, from = :infection, delay = Exponential(2.0)),
                ]
            )
            return progression_loglik(spec, [ind])
        end
        @test censored(1.0) ≈ logccdf(Exponential(2.0), 1.5)
        @test ForwardDiff.derivative(censored, 1.0) ≈ -0.75
    end

    @testset "a shared-draw group censored at an abort keeps its bucket" begin
        # The censored term has to read the draw rather than the gate's own 0
        # or 1, or the case-fatality ratio drops out for every aborted case.
        death_p, recovered_p = exclusive_probabilities([0.64, 0.36])
        spec = ModelSpec(
            BranchingProcess(Poisson(0.0));
            progression = [
                Death(
                    from = ind -> ind.infection_time, delay = Exponential(2.0),
                    probability = death_p
                ),
                Transition(
                    :recovered, from = :infection, delay = Exponential(3.0),
                    probability = recovered_p, terminal = true
                ),
            ]
        )
        ind = Individual(id = 1, infection_time = 0.0)
        ind.state[:infected] = true
        ind.state[:death_candidate_time] = Inf
        ind.state[:recovered] = false
        ind.state[:infection_aborted_time] = 1.0
        # The draw selected death, whose candidate the abort then undid.
        ind.state[death_p.key] = 0.3
        @test progression_loglik(spec, [ind]) ≈
            log(0.64) + logccdf(Exponential(2.0), 1.0)
        # The draw selected recovery instead, so death says nothing.
        ind.state[death_p.key] = 0.9
        @test progression_loglik(spec, [ind]) ≈
            log(0.36) + logccdf(Exponential(3.0), 1.0)
    end

    @testset "a group summing to 1 up to rounding owns no shortfall" begin
        # `sum([0.7, 0.2, 0.1])` is a hair below 1, which must not leave the
        # last bucket demanding a draw a hand-built individual cannot have.
        gates = exclusive_probabilities([0.7, 0.2, 0.1])
        spec = ModelSpec(
            BranchingProcess(Poisson(0.0));
            progression = [
                Transition(
                    Symbol(:step, i), from = :infection, delay = Exponential(1.0),
                    probability = g
                ) for (i, g) in enumerate(gates)
            ]
        )
        ind = Individual(id = 1, infection_time = 0.0)
        ind.state[:infected] = true
        for i in eachindex(gates)
            ind.state[Symbol(:step, i)] = false
        end
        # The first bucket occurred, so the expected value is its own width and
        # its delay density — a value a skipped anchor could not give.
        ind.state[:step1] = true
        ind.state[:step1_time] = 2.0
        @test progression_loglik(spec, [ind]) ≈
            log(0.7) + logpdf(Exponential(1.0), 2.0)
    end

    @testset "an aborted run has a finite likelihood" begin
        spec = ModelSpec(
            BranchingProcess(Poisson(2.0), Exponential(5.0));
            interventions = [
                Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf),
                ContactTracing(
                    probability = 1.0, isolation_to_trace_delay = Exponential(0.5),
                    action = FlagOnly()
                ),
                RingVaccination(efficacy = 0.0, post_exposure_efficacy = 1.0),
            ],
            progression = [
                Transition(:infectious, from = :infection, delay = LogNormal(1.0, 0.3)),
                Reporting(delay = LogNormal(1.0, 0.3), probability = 0.7),
            ],
            attributes = clinical
        )
        state = simulate(spec; n_initial = 3, max_cases = 300, rng = StableRNG(1))
        aborted = count(
            ind -> isfinite(get(ind.state, :infection_aborted_time, Inf)),
            state.individuals
        )
        @test aborted > 0                     # otherwise the test is vacuous
        @test isfinite(progression_loglik(spec, state))
    end

    @testset "a transition without its own method needs one to be evaluated" begin
        spec = ModelSpec(
            BranchingProcess(Poisson(0.0));
            progression = [_UntrackedTransition()], attributes = clinical
        )
        state = simulate(spec; max_cases = 5, rng = StableRNG(5))
        @test_throws "needs a method for" progression_loglik(spec, state)
    end
end
