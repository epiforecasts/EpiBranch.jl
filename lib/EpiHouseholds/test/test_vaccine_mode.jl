# A household clique lets a susceptible be exposed by the same infector more
# than once, unlike a branching process, where every contact is a unique
# exposure. `AllOrNothingMode` and `LeakyMode` should then diverge.
@testset "AllOrNothingMode under repeated household exposure" begin
    clinical = clinical_presentation(incubation_period = Dirac(0.0))
    # Trace and vaccinate as soon as the index case shows symptoms, without
    # quarantining anyone or otherwise shortening the household window, so
    # a vaccinated member goes on meeting its infector for the rest of the
    # infectious period — a fast kernel makes that many repeated exposures.
    ct = ContactTracing(OnSymptomOnset(), 1.0, Dirac(0.0), FlagOnly())
    progression = [Transition(:recovered; delay = 15.0, terminal = true)]

    @testset "A fully-effective dose leaves no vaccinated member infected" begin
        rv = RingVaccination(efficacy = 1.0, mode = AllOrNothingMode())
        process = HouseholdProcess(fill(6, 100), Exponential(0.05))
        model = ModelSpec(process; progression, attributes = clinical,
            interventions = [ct, rv])
        state = simulate(model; rng = StableRNG(11))
        vaccinated = [ind for ind in state.individuals if is_vaccinated(ind)]
        @test length(vaccinated) > 100
        # Every responder's immunity is in force before it can be reached
        # again, so none of them is ever infected after it — here, at
        # efficacy 1.0, not at all.
        @test !any(is_infected, vaccinated)
    end

    @testset "Repeated exposure erodes leaky protection but not all-or-nothing" begin
        function vaccinated_attack_rate(mode, seed)
            rv = RingVaccination(efficacy = 0.5, mode = mode)
            process = HouseholdProcess(fill(6, 100), Exponential(0.05))
            model = ModelSpec(process; progression, attributes = clinical,
                interventions = [ct, rv])
            state = simulate(model; rng = StableRNG(seed))
            vaccinated = [ind for ind in state.individuals if is_vaccinated(ind)]
            return count(is_infected, vaccinated), length(vaccinated)
        end
        leaky_infected, leaky_n = vaccinated_attack_rate(LeakyMode(), 21)
        aon_infected, aon_n = vaccinated_attack_rate(AllOrNothingMode(), 22)
        @test leaky_n > 100 && aon_n > 100
        # Leaky blocks each exposure independently at strength 0.5, so with
        # many repeated exposures the attack rate among the vaccinated tends
        # to 1; all-or-nothing protects a responder for good, so it tends to
        # 1 - efficacy = 0.5. The two should differ by a wide margin.
        @test leaky_infected / leaky_n > aon_infected / aon_n + 0.3
    end
end
