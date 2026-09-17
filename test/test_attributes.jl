@testset "Attributes builders" begin
    @testset "transmission_traits" begin
        @testset "constants" begin
            attrs = transmission_traits(susceptibility = 0.3, infectiousness = 0.7)
            ind = Individual(id = 1)
            attrs(StableRNG(1), ind)
            @test ind.susceptibility == 0.3
            @test ind.infectiousness == 0.7
        end

        @testset "default leaves traits at 1.0" begin
            attrs = transmission_traits()
            ind = Individual(id = 1)
            attrs(StableRNG(1), ind)
            @test ind.susceptibility == 1.0
            @test ind.infectiousness == 1.0
        end

        @testset "Real argument coerced to Float64" begin
            attrs = transmission_traits(susceptibility = 1//2)
            ind = Individual(id = 1)
            attrs(StableRNG(1), ind)
            @test ind.susceptibility === 0.5
        end

        @testset "Distribution sampled per individual" begin
            attrs = transmission_traits(susceptibility = Beta(2, 5))
            rng = StableRNG(42)
            samples = map(1:200) do _
                ind = Individual(id = 1)
                attrs(rng, ind)
                ind.susceptibility
            end
            @test all(0 .<= samples .<= 1)
            @test length(unique(samples)) > 100  # genuinely sampled, not constant
        end

        @testset "Function dispatch sees ind state set by earlier builders" begin
            attrs = [
                demographics(age_distribution = Uniform(0, 90)),
                transmission_traits(
                    susceptibility = (rng, ind) -> ind.state[:age] >= 65 ? 0.8 : 0.2,
                )
            ]
            rng = StableRNG(7)
            for _ in 1:200
                ind = Individual(id = 1)
                EpiBranch._apply_attributes!(attrs, rng, ind)
                expected = ind.state[:age] >= 65 ? 0.8 : 0.2
                @test ind.susceptibility == expected
            end
        end

        @testset "Independent fields" begin
            # only susceptibility specified — infectiousness stays at default
            attrs = transmission_traits(susceptibility = 0.4)
            ind = Individual(id = 1)
            attrs(StableRNG(1), ind)
            @test ind.susceptibility == 0.4
            @test ind.infectiousness == 1.0
        end

        @testset "Integrates with simulate" begin
            attrs = transmission_traits(susceptibility = 0.5, infectiousness = 0.8)
            state = simulate(
                ModelSpec(BranchingProcess(Poisson(1.5), Exponential(5.0)); attributes = attrs);
                max_cases = 30, rng = StableRNG(1))
            @test all(ind.susceptibility == 0.5 for ind in state.individuals)
            @test all(ind.infectiousness == 0.8 for ind in state.individuals)
        end
    end

    @testset "clinical_presentation prob_asymptomatic accepts distribution and function" begin
        @testset "scalar (default behaviour)" begin
            attrs = clinical_presentation(
                incubation_period = LogNormal(1.5, 0.5),
                prob_asymptomatic = 0.0)
            ind = Individual(id = 1)
            attrs(StableRNG(1), ind)
            @test ind.state[:asymptomatic] == false
            @test !isnan(ind.state[:onset_time])
        end

        @testset "function: age-conditional asymptomatic fraction" begin
            attrs = [
                demographics(age_distribution = Uniform(0, 90)),
                clinical_presentation(
                    incubation_period = LogNormal(1.5, 0.5),
                    prob_asymptomatic = (rng, ind) -> ind.state[:age] < 18 ? 1.0 : 0.0
                )
            ]
            rng = StableRNG(7)
            for _ in 1:200
                ind = Individual(id = 1)
                EpiBranch._apply_attributes!(attrs, rng, ind)
                expected = ind.state[:age] < 18
                @test ind.state[:asymptomatic] == expected
            end
        end

        @testset "distribution: per-individual probability" begin
            # Beta(2, 8) has mean 0.2 — most individuals draw a low
            # probability of being asymptomatic.
            attrs = clinical_presentation(
                incubation_period = LogNormal(1.5, 0.5),
                prob_asymptomatic = Beta(2, 8)
            )
            rng = StableRNG(11)
            asymp_count = 0
            for _ in 1:500
                ind = Individual(id = 1)
                attrs(rng, ind)
                ind.state[:asymptomatic] && (asymp_count += 1)
            end
            # Expected ~100/500 = 0.2. Allow generous bounds.
            @test 50 <= asymp_count <= 150
        end
    end

    @testset "vaccine_acceptance draws once per ring" begin
        clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

        @testset "contacts of the same case share one value; other cases differ" begin
            model = BranchingProcess(Poisson(1.0), Exponential(5.0))
            attrs = vaccine_acceptance(propensity = Beta(2, 2))
            rng = StableRNG(1)
            state = EpiBranch.new_state(
                model, EpiBranch.AbstractClinicalTransition[], attrs, rng)
            parent1, parent2 = EpiBranch.add_individuals!(state, 2, [])
            ring1 = [make_contact!(state, parent1, 1.0) for _ in 1:20]
            ring2 = [make_contact!(state, parent2, 1.0) for _ in 1:20]
            @test length(unique(c.state[:vaccine_acceptance] for c in ring1)) == 1
            @test length(unique(c.state[:vaccine_acceptance] for c in ring2)) == 1
            @test ring1[1].state[:vaccine_acceptance] != ring2[1].state[:vaccine_acceptance]
        end

        @testset "key is customisable" begin
            model = BranchingProcess(Poisson(1.0), Exponential(5.0))
            attrs = vaccine_acceptance(propensity = Beta(2, 2), key = :acceptance)
            rng = StableRNG(1)
            state = EpiBranch.new_state(
                model, EpiBranch.AbstractClinicalTransition[], attrs, rng)
            parent = only(EpiBranch.add_individuals!(state, 1, []))
            contact = make_contact!(state, parent, 1.0)
            @test haskey(contact.state, :acceptance)
            @test !haskey(contact.state, :vaccine_acceptance)
        end

        @testset "applied without simulation state errors" begin
            attrs = vaccine_acceptance(propensity = 0.5)
            ind = Individual(id = 1)
            @test_throws ArgumentError EpiBranch._apply_attributes!(attrs, StableRNG(1), ind)
        end

        @testset "0/1 propensity gives all-or-nothing rings at the independent-case mean" begin
            iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
            ct = ContactTracing(
                probability = 1.0, isolation_to_trace_delay = Exponential(0.5))
            p = 0.4
            attrs = [clinical,
                vaccine_acceptance(
                    propensity = (rng, ind) -> Float64(rand(rng, Bernoulli(p))))]
            rv = RingVaccination(efficacy = 0.9,
                coverage = (rng, ind) -> ind.state[:vaccine_acceptance])
            state = simulate(
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                    interventions = [iso, ct, rv], attributes = attrs);
                max_cases = 300, rng = StableRNG(3))

            rings = Dict{Int, Vector{Bool}}()
            for ind in state.individuals
                is_traced(ind) || continue
                push!(get!(rings, ind.parent_id, Bool[]), is_vaccinated(ind))
            end
            @test !isempty(rings)
            @test all(length(unique(v)) == 1 for v in values(rings))

            traced = filter(is_traced, state.individuals)
            rate = count(is_vaccinated, traced) / length(traced)
            @test p - 0.15 <= rate <= p + 0.15
        end

        @testset "clustered coverage inflates the variance of per-ring coverage at the same mean" begin
            # Many synthetic rings, each contact's coverage decided by the
            # package's own `_covers`, comparing a shared Beta-distributed
            # propensity per ring (drawn via `vaccine_acceptance`) against
            # independent per-contact draws at the same mean coverage.
            k = 20        # contacts per ring
            n_rings = 500
            propensity = Beta(2, 2)  # mean 0.5
            model = BranchingProcess(Poisson(1.0), Exponential(5.0))
            attrs = vaccine_acceptance(propensity = propensity)
            coverage = (rng, ind) -> ind.state[:vaccine_acceptance]
            rng = StableRNG(4)

            clustered_means = map(1:n_rings) do _
                state = EpiBranch.new_state(
                    model, EpiBranch.AbstractClinicalTransition[], attrs, rng)
                parent = only(EpiBranch.add_individuals!(state, 1, []))
                ring = [make_contact!(state, parent, 1.0) for _ in 1:k]
                mean(EpiBranch._covers(coverage, c, rng) for c in ring)
            end
            independent_means = map(1:n_rings) do _
                mean(EpiBranch._covers(mean(propensity), nothing, rng) for _ in 1:k)
            end

            @test isapprox(mean(clustered_means), mean(independent_means); atol = 0.05)
            @test var(clustered_means) > var(independent_means)
        end
    end

    @testset "groups" begin
        @testset "assigns a group in range" begin
            attrs = groups(4)
            rng = StableRNG(1)
            labels = map(1:200) do _
                ind = Individual(id = 1)
                attrs(rng, ind)
                ind.state[:group]
            end
            @test all(l -> 1 <= l <= 4, labels)
            @test length(unique(labels)) == 4  # all groups eventually drawn
        end

        @testset "custom key" begin
            attrs = groups(3; key = :household)
            ind = Individual(id = 1)
            attrs(StableRNG(1), ind)
            @test haskey(ind.state, :household)
            @test !haskey(ind.state, :group)
        end

        @testset "rejects non-positive n_groups" begin
            @test_throws ArgumentError groups(0)
        end

        @testset "single group assigns everyone to it" begin
            attrs = groups(1)
            ind = Individual(id = 1)
            attrs(StableRNG(1), ind)
            @test ind.state[:group] == 1
        end
    end
end
