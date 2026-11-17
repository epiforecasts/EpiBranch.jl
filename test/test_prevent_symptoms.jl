# Preventing symptoms through the public API, from an intervention the
# package does not ship.

# Protects every infection from the day it starts: the shape of an antiviral.
struct _Antiviral <: AbstractIntervention end
function EpiBranch.on_infection_settled!(::_Antiviral, ind, state, rng)
    EpiBranch.prevent_symptoms!(ind, ind.infection_time)
    return nothing
end

@testset "prevent_symptoms!" begin
    function exposed(; infection_time = 1.0, incubation = 6.0)
        ind = Individual(id = 1, infection_time = infection_time)
        ind.state[:incubation_period] = incubation
        EpiBranch._set_onset_from_incubation!(ind)
        return ind
    end

    @testset "protection by onset removes the onset" begin
        ind = exposed()
        @test EpiBranch.symptoms_prevented_time(ind) == Inf
        @test !EpiBranch.is_asymptomatic(ind)
        EpiBranch.prevent_symptoms!(ind, 5.0)
        @test EpiBranch.symptoms_prevented_time(ind) == 5.0
        @test EpiBranch.is_asymptomatic(ind)
        @test isnan(ind.state[:onset_time])
    end

    @testset "protection after onset changes nothing" begin
        ind = exposed()
        EpiBranch.prevent_symptoms!(ind, 8.0)
        @test !EpiBranch.is_asymptomatic(ind)
        @test ind.state[:onset_time] == 7.0
    end

    @testset "the earliest time counts" begin
        ind = exposed()
        EpiBranch.prevent_symptoms!(ind, 5.0)
        EpiBranch.prevent_symptoms!(ind, 9.0)
        @test EpiBranch.symptoms_prevented_time(ind) == 5.0
        EpiBranch.prevent_symptoms!(ind, 3.0)
        @test EpiBranch.symptoms_prevented_time(ind) == 3.0
    end

    @testset "survives a vaccine dose being recorded and withdrawn" begin
        ind = exposed()
        EpiBranch.prevent_symptoms!(ind, 5.0)
        ind.state[:symptom_protection_time] = 2.0
        EpiBranch._refresh_symptom_protection!(ind)
        @test EpiBranch.symptoms_prevented_time(ind) == 2.0
        delete!(ind.state, :symptom_protection_time)
        EpiBranch._refresh_symptom_protection!(ind)
        @test EpiBranch.symptoms_prevented_time(ind) == 5.0
        @test EpiBranch.is_asymptomatic(ind)
    end

    @testset "in a simulation" begin
        spec = ModelSpec(
            BranchingProcess(Poisson(2.0), Exponential(5.0));
            interventions = [_Antiviral()],
            attributes = clinical_presentation(
                incubation_period = Dirac(5.0), prob_asymptomatic = 0.0
            )
        )
        sim = simulate(spec; rng = StableRNG(1), max_cases = 200)
        @test length(sim.individuals) > 1
        @test all(EpiBranch.is_asymptomatic, sim.individuals)
        @test all(i -> isnan(i.state[:onset_time]), sim.individuals)
    end
end
