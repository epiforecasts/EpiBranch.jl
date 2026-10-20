@testset "Likelihood distributions" begin
    @testset "offspring_distribution returns the bare offspring" begin
        bp = BranchingProcess(Poisson(1.0))
        d = offspring_distribution(bp)
        @test d isa Poisson
        data = [0, 1, 2, 0, 3, 1, 0]
        @test loglikelihood(d, data) ≈ loglikelihood(OffspringCounts(data), Poisson(1.0))
    end

    @testset "chain_size_distribution — bare model gives the analytical form" begin
        bp = BranchingProcess(Poisson(0.5))
        d = chain_size_distribution(bp)
        @test d isa Borel
        data = [1, 2, 1, 3, 1, 5, 2]
        @test sum(logpdf(d, n) for n in data) ≈
            loglikelihood(ChainSizes(data), Poisson(0.5))

        bp_nb = BranchingProcess(NegBin(0.5, 0.5))
        d_nb = chain_size_distribution(bp_nb)
        @test d_nb isa GammaBorel
    end

    @testset "chain_size_distribution — kwargs trigger the wrapper" begin
        bp = BranchingProcess(Poisson(0.5))
        data = [3, 5, 10, 2]
        seeds = [1, 2, 1, 1]

        d = chain_size_distribution(bp; seeds = seeds)
        @test logpdf(d, data) ≈
            loglikelihood(ChainSizes(data; seeds = seeds), Poisson(0.5))

        pc = [1.0, 0.5, 0.0, 0.7]
        d_pc = chain_size_distribution(bp; seeds = seeds, prob_concluded = pc)
        @test logpdf(d_pc, data) ≈
            loglikelihood(
            ChainSizes(data; seeds = seeds), Poisson(0.5);
            prob_concluded = pc
        )

        # With interventions, prob_concluded routes through the
        # simulation-based likelihood.
        clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf)
        d_iv = chain_size_distribution(
            ModelSpec(bp; interventions = [iso], attributes = clinical);
            seeds = seeds, prob_concluded = pc, n_sim = 500, rng = StableRNG(1)
        )
        @test isfinite(logpdf(d_iv, data))
    end

    @testset "chain_size_distribution — multi-type model falls back without n_sim" begin
        bp_mt = BranchingProcess(
            [0.4 0.2; 0.1 0.3], R -> Poisson(R), Exponential(5.0)
        )
        data = [1, 2, 4]
        @test chain_size_distribution(bp_mt) isa Distribution
        ll_direct = loglikelihood(ChainSizes(data), bp_mt; rng = StableRNG(1))
        ll_via_distribution = logpdf(
            chain_size_distribution(bp_mt; n_sim = 10_000, rng = StableRNG(1)), data
        )
        @test ll_direct ≈ ll_via_distribution

        # A `ModelSpec` wrapping a multi-type process, with no interventions,
        # takes the same fallback via `has_single_type_law(::ModelSpec)`.
        @test chain_size_distribution(ModelSpec(bp_mt)) isa Distribution
    end

    @testset "chain_size_distribution — Binomial offspring takes the analytical path" begin
        # The hitting-time theorem gives Binomial offspring a
        # chain_size_distribution too, so the bare model now takes the
        # analytical fast path rather than needing simulation.
        bp = BranchingProcess(Binomial(5, 0.3))
        data = [1, 2, 3]
        d = chain_size_distribution(bp)
        @test d isa EpiBranch.HittingTimeChainSize
        @test sum(logpdf(d, n) for n in data) ≈ loglikelihood(ChainSizes(data), Binomial(5, 0.3))

        # Explicit kwargs still force the simulation-based wrapper.
        @test chain_size_distribution(bp; n_sim = 500, rng = StableRNG(1)) isa
            Distribution
        @test isfinite(
            loglikelihood(ChainSizes(data), bp; n_sim = 500, rng = StableRNG(1))
        )
    end

    @testset "chain_size_distribution — bare-pgf GeneratingFunctionOffspring falls back" begin
        # Regression: a GeneratingFunctionOffspring has a chain_size_distribution
        # method, so has_single_type_law's hasmethod check passes, but calling
        # it on a bare-pgf instance (no stored pmf) throws ArgumentError.
        # The analytical fast path must catch that and fall through to the
        # simulation-based wrapper, rather than let it escape the primary
        # `~` entry point as an uncaught error.
        R, k = 2.5, 0.16
        off = GeneratingFunctionOffspring(s -> EpiBranch._pgf(NegBin(R, k), s), R)
        bp = BranchingProcess(off, Exponential(5.0))
        @test chain_size_distribution(bp) isa Distribution
    end

    @testset "chain_length_distribution" begin
        bp = BranchingProcess(Poisson(0.5))
        d = chain_length_distribution(bp)
        data = [0, 1, 2, 0, 3, 1]
        @test logpdf(d, data) ≈ loglikelihood(ChainLengths(data), Poisson(0.5))
    end

    @testset "chain_size_distribution — per-case observation" begin
        data = [1, 2, 1, 3, 1]
        obs = ModelSpec(
            BranchingProcess(Poisson(0.5)); observation = PerCaseObservation(;
                detection_prob = 0.7
            )
        )
        d = chain_size_distribution(obs; n_sim = 0)
        # n_sim=0 forces the wrapper path; logpdf should still match the
        # analytical loglikelihood with no kwargs since there are no
        # interventions.
        @test logpdf(d, data) ≈ loglikelihood(ChainSizes(data), obs)
    end

    @testset "prob_concluded keeps the model's observation" begin
        # With every cluster marked concluded, the mixture must reduce to
        # the plain closed form under the observation, not to the bare
        # offspring law with no observation applied.
        bp = BranchingProcess(NegBin(0.6, 0.5))
        data = [2, 3, 7]
        pc = fill(1.0, length(data))
        specs = (
            ModelSpec(bp; observation = MinimumSize(2)),
            ModelSpec(bp; observation = PerCaseObservation(; detection_prob = 0.5)),
            ModelSpec(bp; progression = [Reporting(delay = Dirac(0.0), probability = 0.5)]),
        )
        for spec in specs
            closed_form = sum(logpdf(chain_size_distribution(spec), n) for n in data)
            d_pc = chain_size_distribution(spec; prob_concluded = pc)
            @test logpdf(d_pc, data) ≈ closed_form
        end
    end

    @testset "chain_size_distribution — callable Reporting probability falls back to simulation" begin
        # A callable `Reporting` probability has no scalar for the analytical
        # fast path to read, so it must fall through to the simulation-based
        # wrapper instead of throwing, as `loglikelihood` already does.
        model = ModelSpec(
            BranchingProcess(NegBin(1.5, 0.5));
            progression = [
                Reporting(
                    delay = Dirac(0.0), probability = (rng, ind) -> 0.5,
                    from = ind -> ind.infection_time
                ),
            ]
        )
        @test chain_size_distribution(model) isa Distribution
    end

    @testset "rand draws chain sizes from the underlying model" begin
        bp = BranchingProcess(Poisson(0.5), Exponential(5.0))
        d = chain_size_distribution(bp)
        samples = rand(StableRNG(7), d, 50)
        @test length(samples) == 50
        @test all(s -> s >= 1, samples)
    end

    @testset "rand sums multi-seed clusters rather than dropping chains" begin
        bp = BranchingProcess(Poisson(0.5), Exponential(5.0))
        # Three seeds per cluster: each draw is the total infected across
        # all three seed chains, so every cluster size is at least 3.
        d = chain_size_distribution(bp; n_initial = 3)
        samples = rand(StableRNG(7), d, 50)
        @test length(samples) == 50
        @test all(s -> s >= 3, samples)

        # Chain length aggregates with `maximum`, so deeper seed chains
        # are not lost.
        dl = chain_length_distribution(bp; n_initial = 3)
        lengths = rand(StableRNG(7), dl, 50)
        @test length(lengths) == 50
        @test all(l -> l >= 0, lengths)
    end

    @testset "rand draws reported cluster sizes rather than total infected" begin
        # Offspring mean 0.5 gives a full outbreak size that averages
        # 1 / (1 - 0.5) = 2; with only 10% of cases reported, the observed
        # cluster sizes `logpdf` scores are thinned well below that, and
        # never drop to 0 (an unreported cluster is redrawn rather than kept).
        spec = ModelSpec(
            BranchingProcess(Poisson(0.5));
            progression = [Reporting(probability = 0.1, delay = 0.0, from = :infection)]
        )
        d = chain_size_distribution(spec; seeds = ones(Int, 500))
        samples = rand(StableRNG(1), d, 500)
        @test length(samples) == 500
        @test all(s -> s >= 1, samples)
        @test mean(samples) < 1.5
    end

    @testset "rand raises rather than redrawing forever when no cluster is ever observed" begin
        # Every case goes unreported, so `vals` is always empty and the
        # loop would otherwise redraw forever; it must raise instead.
        # `seeds` routes this through the simulation-based wrapper rather
        # than the analytical fast path.
        spec = ModelSpec(
            BranchingProcess(Poisson(0.5));
            progression = [Reporting(probability = 0.0, delay = 0.0, from = :infection)]
        )
        d = chain_size_distribution(spec; seeds = [1])
        @test_throws ErrorException rand(StableRNG(1), d, 1)
    end

    @testset "rand bounds total attempts by max_attempts when a condition is set" begin
        # A `condition` that can never hold, on a model where every cluster
        # also goes unreported, exercises both the redraw-for-observed-size
        # loop and the condition check at once. The error must report the
        # caller's own `max_attempts` rather than a fixed, larger cap, so
        # that a restrictive condition cannot multiply the two into far more
        # attempts than the caller asked for.
        spec = ModelSpec(
            BranchingProcess(Poisson(0.5));
            progression = [Reporting(probability = 0.0, delay = 0.0, from = :infection)]
        )
        d = chain_size_distribution(
            spec; seeds = [1], condition = 0:0, max_attempts = 5
        )
        err = try
            rand(StableRNG(1), d, 1)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("5 attempts", err.msg)
    end

    @testset "rand drops a kwarg simulate does not accept" begin
        # `n_sim` is a documented likelihood keyword, accepted by the
        # constructor for `logpdf`, but `simulate` itself has no such
        # keyword; `rand` must not forward it.
        bp = BranchingProcess(Poisson(0.8))
        @test length(rand(StableRNG(1), chain_size_distribution(bp; n_sim = 5), 2)) == 2
        @test length(rand(StableRNG(1), chain_length_distribution(bp; n_sim = 5), 2)) == 2
    end

    @testset "logpdf drops a kwarg only rand's simulation accepts" begin
        # `condition`, `max_attempts` and `initial_cases` steer `rand`'s
        # single-outbreak draw; `logpdf`'s simulation is batched and
        # data-driven and has no such keywords. It must not forward them —
        # a kwarg that makes `rand` work must not make `logpdf` throw.
        bp = BranchingProcess(Poisson(0.5))
        d = chain_size_distribution(
            bp; condition = 1:10, max_attempts = 5, initial_cases = [1, 2]
        )
        @test logpdf(d, [2, 3]) isa Real

        dl = chain_length_distribution(bp; initial_cases = [1, 2], parallel = false)
        @test logpdf(dl, [0, 1]) isa Real
    end

    @testset "a kwarg neither rand nor logpdf accepts is refused at construction" begin
        # A typo such as `n_sims` for `n_sim` is not in either callee's
        # accepted-keyword list, so neither side's filtering would catch it;
        # construction must refuse it up front instead.
        bp = BranchingProcess(Poisson(0.5))
        @test_throws ArgumentError chain_size_distribution(bp; n_sims = 50)
        @test_throws ArgumentError chain_length_distribution(bp; n_sims = 50)
    end

    @testset "coverage matrix: every cell is a Distribution or a clear ArgumentError" begin
        # Locks the model × data-type matrix so a gap can't silently
        # reopen: each entry point either returns something usable with
        # Turing's `~` (a `Distribution`) or refuses with a user-facing
        # `ArgumentError`. No bare `MethodError`s, no developer-only hints.
        bp = BranchingProcess(Poisson(0.5))
        bp_obs = ModelSpec(
            BranchingProcess(Poisson(0.5));
            observation = PerCaseObservation(detection_prob = 0.6)
        )
        bp_cm = BranchingProcess(ClusterMixed(Poisson, Gamma(2.0, 0.4)))
        bp_mt = BranchingProcess(
            [1.0 0.5; 0.5 1.0],
            R -> NegBin(R, 0.16), LogNormal(1.6, 0.5)
        )

        @testset "chain_size_distribution" begin
            @test chain_size_distribution(bp) isa Distribution
            @test chain_size_distribution(bp_obs) isa Distribution
            @test chain_size_distribution(bp_cm) isa Distribution
            # A multi-type model has no closed-form chain-size law; this
            # falls back to the simulation-based wrapper rather than
            # erroring, as `loglikelihood(ChainSizes, ...)` already does.
            @test chain_size_distribution(bp_mt) isa Distribution
        end

        @testset "chain_length_distribution" begin
            @test chain_length_distribution(bp) isa Distribution
            # per-case detection has no well-defined chain length; the
            # refusal surfaces when the law is evaluated, not at construction.
            dl = chain_length_distribution(bp_obs)
            @test_throws ArgumentError logpdf(dl, [0, 1, 2])
        end

        @testset "offspring_distribution" begin
            @test offspring_distribution(bp) isa Distribution
            @test_throws ArgumentError offspring_distribution(bp_cm)
            @test_throws ArgumentError offspring_distribution(bp_mt)
        end
    end
end
