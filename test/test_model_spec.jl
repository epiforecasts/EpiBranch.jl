@testset "ModelSpec" begin
    # A ModelSpec composes the modelling layers (progression, interventions,
    # attributes, observation) around a pure transmission process. The process
    # carries none of them; the spec is the single place they are attached.

    @testset "wraps a process faithfully" begin
        bp = BranchingProcess(Poisson(1.5), Exponential(5.0))
        s1 = simulate(bp; max_cases = 500, rng = StableRNG(1))
        s2 = simulate(ModelSpec(bp); max_cases = 500, rng = StableRNG(1))
        @test s2.cumulative_cases == s1.cumulative_cases
    end

    @testset "layers on the spec are applied" begin
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        attr = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
        bp = BranchingProcess(Poisson(2.0), Exponential(5.0))

        spec = ModelSpec(bp; interventions = [iso], attributes = attr)
        @test EpiBranch.interventions(spec) == [iso]
        @test EpiBranch.attributes(spec) === attr

        # The isolation layer takes effect: some cases are isolated.
        state = simulate(spec; max_cases = 500, rng = StableRNG(7))
        @test count(is_isolated, state.individuals) > 0
    end

    @testset "keywords set the layers; default to none" begin
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        bp = BranchingProcess(Poisson(2.0), Exponential(5.0))
        # A bare process carries no layers, so the spec defaults to none.
        @test isempty(EpiBranch.interventions(ModelSpec(bp)))
        # ...and a keyword attaches one.
        @test length(EpiBranch.interventions(ModelSpec(bp; interventions = [iso]))) == 1
    end

    @testset "first-class across the analytical and distribution API" begin
        bp = BranchingProcess(NegativeBinomial(2.5, 0.5), Exponential(5.0))
        spec = ModelSpec(bp)
        # structural accessors delegate to the wrapped process
        @test EpiBranch.n_types(spec) == 1
        @test EpiBranch.population_size(spec) == EpiBranch.population_size(bp)
        @test single_type_offspring(spec) == single_type_offspring(bp)
        # offspring-based analytical helpers accept a spec
        @test extinction_probability(spec) == extinction_probability(bp)
        @test epidemic_probability(spec) == epidemic_probability(bp)
        @test probability_contain(spec) == probability_contain(bp)
        @test proportion_transmission(spec) == proportion_transmission(bp)
        @test proportion_cluster_size(spec) == proportion_cluster_size(bp)
        # end-of-outbreak needs a single-window, no-observation model
        bpg = BranchingProcess(NegativeBinomial(2.5, 0.5), LogNormal(1.6, 0.5))
        @test end_of_outbreak_probability(ModelSpec(bpg), 10.0) ==
              end_of_outbreak_probability(bpg, 10.0)
        # distribution entry points
        @test offspring_distribution(spec) == offspring_distribution(bp)
        @test pdf(chain_size_distribution(spec), 2) ≈ pdf(chain_size_distribution(bp), 2)
        subc = BranchingProcess(Poisson(0.6))
        @test logpdf(chain_length_distribution(ModelSpec(subc)), [0, 1, 0]) ≈
              logpdf(chain_length_distribution(subc), [0, 1, 0])
        # loglikelihood through a spec: analytical chain sizes and chain lengths
        @test loglikelihood(ChainSizes([1, 2, 1, 3]), spec) ≈
              loglikelihood(ChainSizes([1, 2, 1, 3]), bp)
        @test loglikelihood(ChainLengths([0, 1, 0]), ModelSpec(subc)) ≈
              loglikelihood(ChainLengths([0, 1, 0]), subc)
        # batch simulation through a spec
        @test length(simulate(spec, 20; max_cases = 100, rng = StableRNG(3))) == 20
    end
end
