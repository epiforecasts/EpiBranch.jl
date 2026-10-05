# Custom IsolationEligibility used by the user-extension test below.
struct OnlyOlder <: EpiBranch.IsolationEligibility
    age_threshold::Int
end
function EpiBranch.is_eligible_for_isolation(e::OnlyOlder, ind, state)
    return !is_asymptomatic(ind) && get(ind.state, :age, 0) >= e.age_threshold
end
EpiBranch._required_for_eligibility(::OnlyOlder) = [:onset_time, :asymptomatic, :age]

@testset "Isolation trait seams" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5),
        prob_asymptomatic = 0.0
    )

    @testset "Default keyword constructor reproduces previous behaviour" begin
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
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
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0), eligibility = AllCases())
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
            test_sensitivity = (rng, ind) -> ind.state[:age] >= 50 ? 1.0 : 0.0
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
            onset_to_isolation_delay = (rng, ind) -> ind.state[:age] >= 50 ? 0.1 : 5.0
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
            Isolation(onset_to_isolation_delay = Exponential(1.0))
        )
        # AllCases doesn't.
        @test :asymptomatic ∉ EpiBranch.required_fields(
            Isolation(onset_to_isolation_delay = Exponential(1.0), eligibility = AllCases())
        )
        # Custom eligibility declares its own required fields.
        @test :age in EpiBranch.required_fields(
            Isolation(onset_to_isolation_delay = Exponential(1.0), eligibility = OnlyOlder(50))
        )
    end

    @testset "Custom IsolationEligibility integrates end-to-end" begin
        attrs = [clinical, demographics(age_distribution = Uniform(0, 90))]
        iso = Isolation(onset_to_isolation_delay = Exponential(0.1), eligibility = OnlyOlder(50))
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

    @testset "A leaky isolation records no stretch and keeps its window open" begin
        # Leaky isolation reduces each contact's hazard and removes nobody, so
        # it has no stretch for a likelihood to take out and no window to
        # close. A wrapper that cannot read stretches must not narrow the
        # window for it either, which would turn the reduction into a
        # permanent, perfect removal.
        leaky = Isolation(
            onset_to_isolation_delay = Dirac(1.0), isolation_duration = Dirac(7.0),
            post_isolation_transmission = 0.5
        )
        perfect = Isolation(
            onset_to_isolation_delay = Dirac(1.0), isolation_duration = Dirac(7.0)
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

        # The same lapsing wrapper does narrow a perfect isolation's window,
        # which is the conservative answer for a block it can withdraw
        # part-way through a stretch it recorded.
        lapsing = Scheduled(perfect; start_time = 0.0, end_time = 10.0)
        @test EpiBranch.infectious_removal_time(lapsing, case) == 8.0

        # A host no removal reached has no first removal to narrow to.
        @test EpiBranch.first_removal_time(Individual(id = 2)) == Inf
        @test EpiBranch.infectious_removal_time(lapsing, Individual(id = 3)) == Inf
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
            onset_to_isolation_delay = Dirac(1.0), isolation_duration = Dirac(7.0)
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
end
