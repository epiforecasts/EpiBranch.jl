# Custom IsolationEligibility used by the user-extension test below.
struct OnlyOlder <: EpiBranch.IsolationEligibility
    age_threshold::Int
end
function EpiBranch.is_eligible_for_isolation(e::OnlyOlder, ind, state)
    return !is_asymptomatic(ind) && get(ind.state, :age, 0) >= e.age_threshold
end
EpiBranch._required_for_eligibility(::OnlyOlder) = [:onset_time, :asymptomatic, :age]

# A ward shut until day 5, blocking with certainty while shut and reopening
# after. Its `Risk` holds plain numbers and looks identical to a standing block,
# which is why `binding_release` has to be declared rather than inferred: this
# one leaves it at the conservative default.
struct ReopeningWard <: EpiBranch.AbstractIntervention end
function EpiBranch.competing_risk(::ReopeningWard, parent, contact, state)
    state.max_infection_time < 5.0 || return nothing
    return Risk(event_time = 0.0, block_probability = 1.0, release_time = Inf)
end
EpiBranch.risk_applies(::ReopeningWard, route) = true

@testset "Isolation trait seams" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5),
        prob_asymptomatic = 0.0
    )

    @testset "Default keyword constructor reproduces previous behaviour" begin
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf)
        @test iso.eligibility isa SymptomaticOnly
        @test iso.test_sensitivity == 1.0
        @test iso.post_isolation_transmission == 0.0
    end

    @testset "AllCases eligibility isolates asymptomatic individuals too" begin
        # Make the population partially asymptomatic, then with AllCases
        # eligibility (and full sensitivity) every case should get
        # :test_positive = true.
        clin_mixed = clinical_presentation(
            incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.5
        )
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0), eligibility = AllCases(), duration = Inf)
        rng = StableRNG(42)
        state = simulate(
            ModelSpec(
                BranchingProcess(Poisson(2.0), Exponential(5.0));
                interventions = [iso], attributes = clin_mixed
            );
            max_cases = 100,
            rng = rng
        )
        # AllCases + sensitivity = 1.0 means every individual tests
        # positive, including asymptomatic ones.
        @test all(get(ind.state, :test_positive, false) for ind in state.individuals)
    end

    @testset "test_sensitivity accepts a function" begin
        # Age-conditional sensitivity: 0+ → 0%, 50+ → 100%.
        attrs = [clinical, demographics(age_distribution = Uniform(0, 90))]
        iso = Isolation(
            onset_to_isolation_delay = Exponential(0.1),
            test_sensitivity = (rng, ind) -> ind.state[:age] >= 50 ? 1.0 : 0.0, duration = Inf
        )
        rng = StableRNG(13)
        state = simulate(
            ModelSpec(
                BranchingProcess(Poisson(2.0), Exponential(5.0));
                interventions = [iso], attributes = attrs
            );
            max_cases = 100,
            rng = rng
        )
        for ind in state.individuals
            expected = !is_asymptomatic(ind) && ind.state[:age] >= 50
            @test get(ind.state, :test_positive, false) == expected
        end
    end

    @testset "onset_to_isolation_delay accepts a function" begin
        # Age-conditional delay stands in for a delay that depends on
        # per-individual state recorded by another intervention, e.g. a
        # group's own event time.
        attrs = [clinical, demographics(age_distribution = Uniform(0, 90))]
        iso = Isolation(
            onset_to_isolation_delay = (rng, ind) -> ind.state[:age] >= 50 ? 0.1 : 5.0, duration = Inf
        )
        rng = StableRNG(21)
        state = simulate(
            ModelSpec(
                BranchingProcess(Poisson(2.0), Exponential(5.0));
                interventions = [iso], attributes = attrs
            );
            max_cases = 100,
            rng = rng
        )
        # Isolation time is only set once `resolve_individual!` has run on an
        # individual, which does not happen for every case before the run
        # stops at `max_cases`; restrict the check to those it did reach.
        checked = 0
        for ind in state.individuals
            is_test_positive(ind) || continue
            isfinite(isolation_time(ind)) || continue
            checked += 1
            expected_delay = ind.state[:age] >= 50 ? 0.1 : 5.0
            @test isolation_time(ind) - onset_time(ind) ≈ expected_delay
        end
        @test checked > 0
    end

    @testset "required_fields dispatches on eligibility" begin
        # Default SymptomaticOnly requires :asymptomatic.
        @test :asymptomatic in EpiBranch.required_fields(
            Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf)
        )
        # AllCases doesn't.
        @test :asymptomatic ∉ EpiBranch.required_fields(
            Isolation(onset_to_isolation_delay = Exponential(1.0), eligibility = AllCases(), duration = Inf)
        )
        # Custom eligibility declares its own required fields.
        @test :age in EpiBranch.required_fields(
            Isolation(onset_to_isolation_delay = Exponential(1.0), eligibility = OnlyOlder(50), duration = Inf)
        )
    end

    @testset "Custom IsolationEligibility integrates end-to-end" begin
        attrs = [clinical, demographics(age_distribution = Uniform(0, 90))]
        iso = Isolation(onset_to_isolation_delay = Exponential(0.1), eligibility = OnlyOlder(50), duration = Inf)
        rng = StableRNG(17)
        state = simulate(
            ModelSpec(
                BranchingProcess(Poisson(2.0), Exponential(5.0));
                interventions = [iso], attributes = attrs
            );
            max_cases = 100,
            rng = rng
        )
        for ind in state.individuals
            ind.state[:test_positive] && @test ind.state[:age] >= 50
        end
    end

    @testset "A certain block past a bounded kernel's reach ends the pair" begin
        # The window now runs to recovery, so a blocked proposal asks the kernel
        # for a later contact. Under a kernel whose support ends inside the
        # window the remaining integrated hazard is infinite, which the race
        # refuses to sample against. A block certain to cover the rest of that
        # support answers every contact that could still happen, so the pair
        # ends; one that lapses inside it leaves contacts it does not block, and
        # the redraws terminate on one of them. Before the fix both threw.
        prog = [Transition(:recovered; from = :infection, delay = 30.0, terminal = true)]
        attrs = clinical_presentation(incubation_period = Dirac(3.0))
        kernel = Uniform(0.0, 10.0)
        function race(duration)
            iso = Isolation(
                onset_to_isolation_delay = Dirac(1.0), duration = duration
            )
            rng = StableRNG(3)
            state = EpiBranch.new_state(
                BranchingProcess(Dirac(1), kernel), prog, attrs, rng
            )
            EpiBranch.add_individuals!(state, 2, [iso])
            EpiBranch._sellke_race!(
                state, [1, 2], rng; from = :infection, until = (:recovered,),
                interventions = [iso],
                targets = (inf, st) -> inf == 1 && !is_infected(st.individuals[2]) ?
                    ((2, kernel),) : (),
                seed! = (best, members, r) -> (best[1] = 0.0)
            )
            return state
        end

        # Isolated on day 4, released on day 11, past the kernel's support: no
        # contact after day 4 can transmit, so the pair ends rather than erroring.
        covered = race(Dirac(7.0))
        @test isolation_release_time(covered.individuals[1]) == 11.0
        @test !is_infected(covered.individuals[2])

        # Released on day 7 instead, inside the support: the same seed infects
        # the contact after the release.
        lapses = race(Dirac(3.0))
        @test isolation_release_time(lapses.individuals[1]) == 7.0
        @test is_infected(lapses.individuals[2])
        @test lapses.individuals[2].infection_time > 7.0
    end

    @testset "Only a binding release can end a pair, and wrappers forward it" begin
        # The race reads a release only from a component that says its releases
        # bind. A recorded stretch is append-only, so the removals declare it,
        # and a schedule that can close declares it away again. Without the
        # declaration a certain block raises rather than silently dropping
        # contacts that could still transmit.
        iso = Isolation(
            onset_to_isolation_delay = Dirac(1.0), duration = Dirac(7.0)
        )
        @test EpiBranch.binding_release(iso)
        @test EpiBranch.binding_release(Scheduled(iso; start_time = 0.0))
        @test EpiBranch.binding_release(Scheduled(iso; start_after_cases = 2))
        @test EpiBranch.binding_release(
            CapacityConstrained(iso; budget_per_period = 1.0e6)
        )
        @test !EpiBranch.binding_release(
            Scheduled(iso; start_time = 0.0, end_time = 10.0)
        )
        @test !EpiBranch.binding_release(ReopeningWard())

        prog = [Transition(:recovered; from = :infection, delay = 30.0, terminal = true)]
        attrs = clinical_presentation(incubation_period = Dirac(3.0))
        kernel = Uniform(0.0, 10.0)
        function race(interventions)
            rng = StableRNG(3)
            state = EpiBranch.new_state(
                BranchingProcess(Dirac(1), kernel), prog, attrs, rng
            )
            EpiBranch.add_individuals!(state, 2, interventions)
            EpiBranch._sellke_race!(
                state, [1, 2], rng; from = :infection, until = (:recovered,),
                interventions = interventions,
                targets = (inf, st) -> inf == 1 && !is_infected(st.individuals[2]) ?
                    ((2, kernel),) : (),
                seed! = (best, members, r) -> (best[1] = 0.0)
            )
            return state
        end

        # A start-only schedule keeps the window open and relies on the
        # per-contact block, so its release has to reach the race: before the
        # fix this threw, the gate having read whether the wrapper could lapse
        # rather than whether its releases bind.
        wrapped = race([Scheduled(iso; start_time = 0.0)])
        @test isolation_release_time(wrapped.individuals[1]) == 11.0
        @test !is_infected(wrapped.individuals[2])

        # The ward's block looks certain at this proposal and lifts at day 5, so
        # ending the pair would lose the contacts after it. The race says so
        # instead of guessing.
        @test_throws ArgumentError race([ReopeningWard()])
    end

    @testset "A leaky isolation records no stretch and keeps its window open" begin
        # Leaky isolation reduces each contact's hazard and removes nobody, so
        # it has no stretch for a likelihood to take out and no window to
        # close. A wrapper that cannot read stretches must not narrow the
        # window for it either, which would turn the reduction into a
        # permanent, perfect removal.
        leaky = Isolation(
            onset_to_isolation_delay = Dirac(1.0), duration = Dirac(7.0),
            post_isolation_transmission = 0.5
        )
        perfect = Isolation(
            onset_to_isolation_delay = Dirac(1.0), duration = Dirac(7.0)
        )
        permanent = Isolation(
            onset_to_isolation_delay = Dirac(1.0), duration = Inf
        )
        case = Individual(id = 1)
        set_isolated!(case, 8.0; release_time = 15.0)

        @test isempty(EpiBranch.removal_gap_host_times(leaky))
        @test EpiBranch.removal_gap_host_times(perfect) ==
            (EpiBranch.REMOVAL_STRETCHES_KEY,)
        for wrap in (
                identity,
                iv -> Scheduled(iv; start_time = 0.0),
                iv -> Scheduled(iv; start_time = 0.0, end_time = 10.0),
                iv -> CapacityConstrained(iv; budget_per_period = 1.0e6),
            )
            @test EpiBranch.infectious_removal_time(wrap(leaky), case) == Inf
        end

        # The same lapsing wrapper leaves a perfect isolation's window open: the
        # recorded stretch has its own release, which the per-contact risk
        # re-checks the schedule against at every proposal regardless.
        # Narrowing here would turn a removal due to lapse into one that
        # never does (see issue #410).
        lapsing = Scheduled(perfect; start_time = 0.0, end_time = 10.0)
        @test EpiBranch.infectious_removal_time(lapsing, case) == Inf

        # A removal with no release of its own leaves nothing for either side
        # to hand back, so the same lapsing wrapper still narrows its window —
        # the conservative answer for a block it genuinely cannot speak for.
        permanent_case = Individual(id = 2)
        set_isolated!(permanent_case, 8.0; release_time = Inf)
        lapsing_permanent = Scheduled(permanent; start_time = 0.0, end_time = 10.0)
        @test EpiBranch.infectious_removal_time(lapsing_permanent, permanent_case) == 8.0

        # A host no removal reached has no permanent removal to narrow to.
        @test EpiBranch.permanent_removal_time(Individual(id = 3)) == Inf
        @test EpiBranch.infectious_removal_time(lapsing, Individual(id = 4)) == Inf
    end

    @testset "A lapsed isolation lets the case go again, on every engine" begin
        # Isolated on day 4 for 7 days (released day 11), recovering on day 30,
        # meeting its one contact at a fixed interval of 15 days — after the
        # release, so the contact should go through on every engine. Before the
        # fix, a continuous-time window closed for good at the isolation time,
        # missing this contact entirely.
        prog = [Transition(:recovered; from = :infection, delay = 30.0, terminal = true)]
        attrs = clinical_presentation(incubation_period = Dirac(3.0))
        iso = Isolation(
            onset_to_isolation_delay = Dirac(1.0), duration = Dirac(7.0)
        )

        # The continuous-time race, which `HouseholdProcess` and `NetworkProcess`
        # share.
        rng = StableRNG(1)
        state = EpiBranch.new_state(
            BranchingProcess(Dirac(1), Dirac(15.0)), prog, attrs, rng
        )
        EpiBranch.add_individuals!(state, 2, [iso])
        EpiBranch._sellke_race!(
            state, [1, 2], rng; from = :infection, until = (:recovered,),
            interventions = [iso],
            targets = (inf, st) -> inf == 1 && !is_infected(st.individuals[2]) ?
                ((2, Dirac(15.0)),) : (),
            seed! = (best, members, r) -> (best[1] = 0.0)
        )
        @test isolation_time(state.individuals[1]) == 4.0
        @test isolation_release_time(state.individuals[1]) == 11.0
        @test is_infected(state.individuals[2])

        # The other half of the same seam: a contact due inside [4, 11) is
        # blocked, which is what the per-contact risk has to do now that the
        # window no longer closes at the isolation. Without it the release
        # would hand the case back and block nobody at all.
        inside_rng = StableRNG(1)
        inside = EpiBranch.new_state(
            BranchingProcess(Dirac(1), Dirac(7.0)), prog, attrs, inside_rng
        )
        EpiBranch.add_individuals!(inside, 2, [iso])
        EpiBranch._sellke_race!(
            inside, [1, 2], inside_rng; from = :infection, until = (:recovered,),
            interventions = [iso],
            targets = (inf, st) -> inf == 1 && !is_infected(st.individuals[2]) ?
                ((2, Dirac(7.0)),) : (),
            seed! = (best, members, r) -> (best[1] = 0.0)
        )
        @test isolation_time(inside.individuals[1]) == 4.0
        @test isolation_release_time(inside.individuals[1]) == 11.0
        @test !is_infected(inside.individuals[2])

        # The generation engine, which already answers this correctly, for the
        # same isolation and the same fixed contact time.
        gen_state = simulate(
            ModelSpec(
                BranchingProcess(Dirac(1), Dirac(15.0));
                progression = prog, attributes = attrs, interventions = [iso]
            );
            max_cases = 2, rng = StableRNG(1)
        )
        secondary = [ind for ind in gen_state.individuals if ind.parent_id != 0]
        @test length(secondary) == 1
        @test is_infected(only(secondary))
    end

    @testset "A schedule that never lapses still hands the case back" begin
        # Wrapping the same isolation in a schedule whose `end_time` is never
        # reached, or a predicate that never turns false, must not change the
        # answer. The per-contact risk re-checks the schedule at every
        # proposal regardless, and the release still hands the case back.
        # Before the fix, a wrapper that could lapse closed the window at the
        # isolation's own start whatever its own condition actually did,
        # missing this contact entirely.
        prog = [Transition(:recovered; from = :infection, delay = 30.0, terminal = true)]
        attrs = clinical_presentation(incubation_period = Dirac(3.0))
        iso = Isolation(
            onset_to_isolation_delay = Dirac(1.0), duration = Dirac(7.0)
        )

        for wrapped in (
                Scheduled(iso; start_time = 0.0, end_time = 1.0e6),
                Scheduled(iso, state -> true),
            )
            rng = StableRNG(1)
            state = EpiBranch.new_state(
                BranchingProcess(Dirac(1), Dirac(15.0)), prog, attrs, rng
            )
            EpiBranch.add_individuals!(state, 2, [wrapped])
            EpiBranch._sellke_race!(
                state, [1, 2], rng; from = :infection, until = (:recovered,),
                interventions = [wrapped],
                targets = (inf, st) -> inf == 1 && !is_infected(st.individuals[2]) ?
                    ((2, Dirac(15.0)),) : (),
                seed! = (best, members, r) -> (best[1] = 0.0)
            )
            @test isolation_release_time(state.individuals[1]) == 11.0
            @test is_infected(state.individuals[2])
        end
    end
end
