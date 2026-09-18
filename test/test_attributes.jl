struct _AttributeTagger end
(::_AttributeTagger)(rng, ind) = (ind.state[:tag] = ind.id)

@testset "Attributes builders" begin
    @testset "Callable structs compose with attribute builders" begin
        for attributes in (
            (groups(2), _AttributeTagger()),
            [groups(2), _AttributeTagger()]
        )
            state = simulate(
                ModelSpec(BranchingProcess(Poisson(0.0), Exponential(5.0)); attributes);
                n_initial = 3, rng = StableRNG(1))
            @test length(state.individuals) == 3
            @test all(ind.state[:tag] == ind.id for ind in state.individuals)
            @test all(haskey(ind.state, :group) for ind in state.individuals)
        end
    end

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

    @testset "group_attribute shares reporting probabilities per run" begin
        attrs = [groups(2; key = :household),
            group_attribute(:reporting_probability; value = Beta(6, 4),
                group_key = :household)]
        model = ModelSpec(BranchingProcess(Poisson(0.0), Exponential(5.0));
            attributes = attrs,
            observation = PerCaseObservation(
                detection_prob = (rng, ind) -> ind.state[:reporting_probability],
                from = :infection_time))
        serial = simulate(model, 10; n_initial = 30, rng = StableRNG(42))
        parallel = simulate(model, 10; n_initial = 30, rng = StableRNG(42),
            parallel = true)
        probabilities(states) = [[ind.state[:reporting_probability]
                                  for ind in run.individuals] for run in states]
        @test probabilities(parallel) == probabilities(simulate(model, 10;
            n_initial = 30, rng = StableRNG(42), parallel = true))
        @test length(unique(first.(probabilities(serial)))) == 10
        @test length(unique(first.(probabilities(parallel)))) == 10
        for run in vcat(serial, parallel)
            per_household = [unique([ind.state[:reporting_probability]
                                     for ind in run.individuals
                                     if ind.state[:household] == h]) for h in 1:2]
            @test all(length(v) == 1 for v in per_household)
            @test only(per_household[1]) != only(per_household[2])
        end
        @test isempty(attrs[2].cache)

        # The callback reads the first member, not each subsequent member.
        callback_attrs = [groups(1),
            group_attribute(:first_member; value = (rng, ind) -> ind.id)]
        callback_state = simulate(
            ModelSpec(BranchingProcess(Poisson(0.0), Exponential(5.0)); attributes = callback_attrs);
            n_initial = 3, rng = StableRNG(1))
        @test all(ind.state[:first_member] == 1 for ind in callback_state.individuals)
        @test_throws ArgumentError simulate(ModelSpec(
            BranchingProcess(Poisson(0.0), Exponential(5.0));
            attributes = group_attribute(:shared; value = 0.5)))
    end

    @testset "vaccine_acceptance draws once per group" begin
        clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
        process = BranchingProcess(Poisson(1.0), Exponential(5.0))
        read_acceptance = (rng, ind) -> ind.state[:vaccine_acceptance]

        @testset "members of one group share a value; other groups differ" begin
            attrs = [groups(2), vaccine_acceptance(propensity = Beta(2, 2))]
            state = EpiBranch.new_state(
                process, EpiBranch.AbstractClinicalTransition[], attrs, StableRNG(1))
            people = EpiBranch.add_individuals!(state, 60, [])
            by_group = Dict{Int, Set{Float64}}()
            for ind in people
                push!(get!(by_group, ind.state[:group], Set{Float64}()),
                    ind.state[:vaccine_acceptance])
            end
            @test length(by_group) == 2
            @test all(length(v) == 1 for v in values(by_group))
            @test length(union(values(by_group)...)) == 2
        end

        @testset "the shared value outlives the generation it was drawn in" begin
            attrs = [groups(1), vaccine_acceptance(propensity = Beta(2, 2))]
            state = EpiBranch.new_state(
                process, EpiBranch.AbstractClinicalTransition[], attrs, StableRNG(1))
            case = only(EpiBranch.add_individuals!(state, 1, []))
            child = make_contact!(state, case, 1.0)
            grandchild = make_contact!(state, child, 2.0)
            @test child.state[:vaccine_acceptance] == case.state[:vaccine_acceptance]
            @test grandchild.state[:vaccine_acceptance] == case.state[:vaccine_acceptance]
        end

        @testset "separate runs draw their own values" begin
            attrs = [groups(1), vaccine_acceptance(propensity = Beta(2, 2))]
            drawn = map(1:20) do seed
                state = EpiBranch.new_state(process,
                    EpiBranch.AbstractClinicalTransition[], attrs, StableRNG(seed))
                only(EpiBranch.add_individuals!(state, 1, [])).state[:vaccine_acceptance]
            end
            @test length(unique(drawn)) == 20
        end

        @testset "keys are customisable" begin
            attrs = [groups(2; key = :village),
                vaccine_acceptance(propensity = Beta(2, 2),
                    group_key = :village, key = :acceptance)]
            state = EpiBranch.new_state(
                process, EpiBranch.AbstractClinicalTransition[], attrs, StableRNG(1))
            ind = only(EpiBranch.add_individuals!(state, 1, []))
            @test haskey(ind.state, :acceptance)
            @test !haskey(ind.state, :vaccine_acceptance)
        end

        @testset "a missing group key is an error naming the key" begin
            attrs = vaccine_acceptance(propensity = 0.5, group_key = :village)
            ind = Individual(id = 1)
            err = try
                EpiBranch._apply_attributes!(attrs, StableRNG(1), ind)
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin(":village", err.msg)
        end

        @testset "clusters GroupVaccination coverage inside the group" begin
            gv = GroupVaccination(efficacy = 0.9, coverage = read_acceptance)
            # Villages 1 and 2 accept and decline as blocks; which is which
            # follows from the group label, so the test does not depend on
            # the order the propensities are drawn in.
            attrs = [groups(2),
                vaccine_acceptance(
                    propensity = (rng, ind) -> ind.state[:group] == 1 ? 1.0 : 0.0)]
            state = EpiBranch.new_state(process,
                EpiBranch.AbstractClinicalTransition[], attrs, StableRNG(1))
            people = EpiBranch.add_individuals!(state, 40, [])
            for ind in people
                ind.state[:test_positive] = true
                set_isolated!(ind, 2.0)
            end

            EpiBranch.apply_post_transmission!(gv, state, people)

            accepting = filter(ind -> ind.state[:group] == 1, people)
            declining = filter(ind -> ind.state[:group] == 2, people)
            @test !isempty(accepting) && !isempty(declining)
            @test all(is_vaccinated, accepting)
            @test !any(is_vaccinated, declining)
        end

        @testset "0/1 propensity gives all-or-nothing groups at the independent mean" begin
            iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
            ct = ContactTracing(
                probability = 1.0, isolation_to_trace_delay = Exponential(0.5))
            p = 0.4
            attrs = [clinical, groups(10),
                vaccine_acceptance(
                    propensity = (rng, ind) -> Float64(rand(rng, Bernoulli(p))))]
            rv = RingVaccination(efficacy = 0.9, coverage = read_acceptance)
            states = simulate(
                ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
                    interventions = [iso, ct, rv], attributes = attrs), 20;
                max_cases = 300, rng = StableRNG(3))

            traced = [ind for state in states
                      for ind in state.individuals
                      if is_traced(ind)]
            @test !isempty(traced)
            for state in states
                by_group = Dict{Int, Vector{Bool}}()
                for ind in filter(is_traced, state.individuals)
                    push!(get!(by_group, ind.state[:group], Bool[]), is_vaccinated(ind))
                end
                @test all(length(unique(v)) == 1 for v in values(by_group))
            end
            rate = count(is_vaccinated, traced) / length(traced)
            @test p - 0.1 <= rate <= p + 0.1
        end

        @testset "clustering inflates the variance of per-group coverage at the same mean" begin
            # Coverage is decided by the package's own `_covers`, comparing a
            # Beta propensity shared within a village against independent
            # per-individual draws at the same mean coverage.
            n_villages = 200
            per_village = 20
            propensity = Beta(2, 2)  # mean 0.5
            attrs = [groups(n_villages), vaccine_acceptance(propensity = propensity)]
            rng = StableRNG(4)
            state = EpiBranch.new_state(
                process, EpiBranch.AbstractClinicalTransition[], attrs, rng)
            people = EpiBranch.add_individuals!(
                state, n_villages * per_village, [])

            covered = Dict{Int, Vector{Bool}}()
            for ind in people
                push!(get!(covered, ind.state[:group], Bool[]),
                    EpiBranch._covers(read_acceptance, ind, rng))
            end
            clustered_means = [mean(v) for v in values(covered) if length(v) >= 5]
            independent_means = [mean(EpiBranch._covers(mean(propensity), nothing, rng)
                                 for _ in 1:per_village)
                                 for _ in 1:n_villages]

            @test isapprox(mean(clustered_means), mean(independent_means); atol = 0.05)
            @test var(clustered_means) > 2 * var(independent_means)
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
