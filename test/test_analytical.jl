@testset "Analytical" begin
    @testset "Extinction probability" begin
        @testset "Subcritical always extinct" begin
            @test extinction_probability(0.5, 0.1) == 1.0
            @test extinction_probability(0.99, 1.0) == 1.0
            @test extinction_probability(1.0, 0.5) == 1.0
        end

        @testset "Supercritical extinction < 1" begin
            q = extinction_probability(2.0, 0.5)
            @test 0.0 < q < 1.0
        end

        @testset "Known value: R=2, k=1 (Geometric)" begin
            # For Geometric offspring (k=1, mean=R), extinction prob = 1/R
            q = extinction_probability(2.0, 1.0)
            @test q ≈ 0.5 atol=1e-8
        end

        @testset "Higher R → lower extinction" begin
            q1 = extinction_probability(1.5, 0.5)
            q2 = extinction_probability(3.0, 0.5)
            @test q1 > q2
        end

        @testset "Higher k → lower extinction (for same R)" begin
            # More dispersion (lower k) → higher extinction probability
            q_low_k = extinction_probability(2.0, 0.1)
            q_high_k = extinction_probability(2.0, 10.0)
            @test q_low_k > q_high_k
        end

        @testset "Poisson offspring" begin
            q = extinction_probability(Poisson(0.5))
            @test q == 1.0

            q = extinction_probability(Poisson(2.0))
            @test 0.0 < q < 1.0

            # Known: for Poisson(λ), extinction prob is the solution to q = exp(λ(q-1))
            # For λ=2: q ≈ 0.2032
            @test q ≈ 0.2032 atol=0.001
        end

        @testset "NegativeBinomial distribution method" begin
            d = NegBin(2.5, 0.16)
            q = extinction_probability(d)
            q2 = extinction_probability(2.5, 0.16)
            @test q ≈ q2
        end

        @testset "Delegates through PartiallyObserved" begin
            # Regression: extinction_probability goes via
            # _single_type_offspring, which must delegate through
            # observation wrappers. Before the fix it threw FieldError
            # because PartiallyObserved has no `offspring` field.
            bp = BranchingProcess(NegBin(0.5, 0.5))
            @test extinction_probability(PartiallyObserved(bp, 0.7)) ==
                  extinction_probability(bp)
        end
    end

    @testset "Epidemic probability" begin
        @test epidemic_probability(0.5, 0.5) == 0.0
        @test epidemic_probability(2.0, 1.0) ≈ 0.5 atol=1e-8
    end

    @testset "Borel distribution" begin
        @testset "Construction" begin
            d = Borel(0.5)
            @test d.μ == 0.5
            @test_throws ArgumentError Borel(-0.1)
            # Supercritical allowed: PMF valid pointwise, used inside quadrature
            d_super = Borel(1.5)
            @test d_super.μ == 1.5
            @test isfinite(logpdf(d_super, 2))
        end

        @testset "PMF sums to ≈ 1" begin
            d = Borel(0.5)
            total = sum(pdf(d, n) for n in 1:100)
            @test total ≈ 1.0 atol=1e-6
        end

        @testset "Mean" begin
            d = Borel(0.5)
            @test mean(d) ≈ 2.0 atol=1e-10  # 1/(1-μ)
        end

        @testset "Borel(1.0) has infinite mean" begin
            d = Borel(1.0)
            @test mean(d) == Inf
        end

        @testset "rand throws on supercritical" begin
            # Regression: previously silently returned 10_000 with a
            # warning, giving a nominally valid sample for a distribution
            # whose total mass is < 1.
            @test_throws ArgumentError rand(Borel(1.5))
            @test_throws ArgumentError rand(Borel(1.0))
        end
    end

    @testset "GammaBorel distribution" begin
        @testset "Construction" begin
            d = GammaBorel(0.5, 0.8)
            @test d.k == 0.5
            @test d.R == 0.8
            @test_throws ArgumentError GammaBorel(-1.0, 0.5)
            @test_throws ArgumentError GammaBorel(0.5, -0.5)
        end

        @testset "PMF sums to ≈ 1 for subcritical" begin
            d = GammaBorel(0.5, 0.8)
            total = sum(pdf(d, n) for n in 1:500)
            @test total ≈ 1.0 atol=1e-2
        end

        @testset "Supercritical allowed in logpdf, rejected in rand" begin
            # Regression: supercritical GammaBorel is needed for
            # quadrature integrands that cross R=1, so logpdf stays
            # defined, but rand would silently truncate at chain size
            # 10_000.
            d_super = GammaBorel(0.5, 1.5)
            @test isfinite(logpdf(d_super, 3))
            @test_throws ArgumentError rand(d_super)
        end
    end

    @testset "PoissonGammaChainSize" begin
        @testset "rand throws when Gamma mean ≥ 1" begin
            # Regression: supercritical mean R puts too much Gamma mass
            # above 1, so rand would silently truncate.
            d_super = PoissonGammaChainSize(0.5, 1.5)
            @test isfinite(logpdf(d_super, 2))
            @test_throws ArgumentError rand(d_super)
        end
    end

    @testset "Chain size distribution" begin
        @testset "Poisson offspring → Borel" begin
            d = chain_size_distribution(Poisson(0.5))
            @test d isa Borel
            @test d.μ == 0.5
        end

        @testset "NegBin offspring → GammaBorel" begin
            d = chain_size_distribution(NegBin(0.8, 0.5))
            @test d isa GammaBorel
        end
    end

    @testset "Chain size likelihood" begin
        @testset "Basic evaluation" begin
            ll = loglikelihood(ChainSizes([1, 1, 2, 1, 3]), Poisson(0.5))
            @test isfinite(ll)
            @test ll < 0.0
        end

        @testset "Higher λ gives different likelihood" begin
            data = ChainSizes([1, 1, 1, 2, 1])
            ll1 = loglikelihood(data, Poisson(0.3))
            ll2 = loglikelihood(data, Poisson(0.8))
            @test ll1 != ll2
        end

        @testset "NegBin offspring" begin
            ll = loglikelihood(ChainSizes([1, 2, 1, 3, 1]), NegBin(0.8, 0.5))
            @test isfinite(ll)
        end

        @testset "With partial observation" begin
            base = BranchingProcess(Poisson(0.5))
            ll = loglikelihood(ChainSizes([1, 1, 2]), PartiallyObserved(base, 0.8))
            @test isfinite(ll)

            # Full observation (detection_prob=1) should agree with bare offspring
            ll_full = loglikelihood(ChainSizes([1, 1, 2]), PartiallyObserved(base, 1.0))
            ll_bare = loglikelihood(ChainSizes([1, 1, 2]), Poisson(0.5))
            @test ll_full ≈ ll_bare atol=1e-6

            # Stricter detection should lower the likelihood for these small sizes
            ll_strict = loglikelihood(ChainSizes([1, 1, 2]), PartiallyObserved(base, 0.3))
            @test isfinite(ll_strict)

            # State-space form gives the same answer as the wrapper.
            for p in (0.3, 0.8, 1.0)
                ll_wrapper = loglikelihood(ChainSizes([1, 1, 2]),
                    PartiallyObserved(base, p))
                ll_ss = loglikelihood(ChainSizes([1, 1, 2]),
                    Surveilled(base, PerCaseObservation(p, Dirac(0.0))))
                @test ll_wrapper ≈ ll_ss atol=1e-12
            end
            # Reporting delay is irrelevant for closed-outbreak chain
            # sizes — same answer regardless of delay distribution.
            ll_with_delay = loglikelihood(ChainSizes([1, 1, 2]),
                Surveilled(base, PerCaseObservation(0.7, LogNormal(1.0, 0.5))))
            ll_no_delay = loglikelihood(ChainSizes([1, 1, 2]),
                Surveilled(base, PerCaseObservation(0.7, Dirac(0.0))))
            @test ll_with_delay ≈ ll_no_delay atol=1e-12
        end

        @testset "Cluster-level heterogeneity" begin
            data = ChainSizes([1, 1, 2, 3, 1])

            # Narrow mixing should approximately match a fixed offspring
            o_narrow = ClusterMixed(R -> NegBin(R, 0.5),
                truncated(Normal(0.8, 0.001); lower = 0.0, upper = 0.999))
            ll_narrow = loglikelihood(data, o_narrow)
            ll_fixed = loglikelihood(data, NegBin(0.8, 0.5))
            @test ll_narrow ≈ ll_fixed atol=0.05

            # Wider mixing gives a finite but different likelihood
            o_wide = ClusterMixed(R -> NegBin(R, 0.5), Gamma(2.0, 0.3))
            ll_wide = loglikelihood(data, o_wide)
            @test isfinite(ll_wide)
            @test ll_wide != ll_fixed

            # Poisson + Gamma: dispatch picks closed-form PoissonGammaChainSize
            k, R = 0.5, 0.8
            cm = ClusterMixed(Poisson, Gamma(k, R / k))
            @test chain_size_distribution(cm) isa PoissonGammaChainSize
            data_pg = ChainSizes([1, 1, 2, 3, 1, 4])
            ll_cm = loglikelihood(data_pg, cm)
            d_closed = PoissonGammaChainSize(k, R)
            ll_closed = sum(logpdf(d_closed, n) for n in data_pg.data)
            @test ll_cm ≈ ll_closed atol=1e-8

            # Quadrature fallback for non-closed-form combinations agrees with
            # the closed form when the combination reduces to Poisson + Gamma
            cm_quad = ClusterMixed(λ -> Poisson(λ), Gamma(k, R / k))
            ll_quad = loglikelihood(data_pg, cm_quad)
            @test ll_quad ≈ ll_closed atol=0.05
        end

        @testset "Multi-seed and right-censored chain sizes" begin
            # Default-metadata path matches the direct distribution PMF.
            d = GammaBorel(0.5, 0.8)
            data = ChainSizes([1, 1, 2, 3, 1])
            @test loglikelihood(data, NegBin(0.8, 0.5)) ≈
                  sum(logpdf(d, n) for n in data.data)

            # Multi-seed s=2 agrees with the single-seed self-convolution
            # (the PMF from two independent index cases is the convolution
            # of two single-seed PMFs).
            function brute_s2(d, x)
                s = 0.0
                for m in 1:(x - 1)
                    s += pdf(d, m) * pdf(d, x - m)
                end
                return s
            end
            for x in 2:8
                formula = exp(EpiBranch._chain_size_logpdf(d, x, 2))
                @test formula≈brute_s2(d, x) atol=1e-10
            end

            # Right-censored (ongoing) likelihood P(X ≥ x) is monotone:
            # a higher observed lower bound means a smaller tail mass.
            ll_ongoing_3 = loglikelihood(
                ChainSizes([3]; concluded = [false]), NegBin(0.8, 0.5))
            ll_ongoing_5 = loglikelihood(
                ChainSizes([5]; concluded = [false]), NegBin(0.8, 0.5))
            @test ll_ongoing_3 > ll_ongoing_5
            # P(X ≥ 1) = 1 for any chain that exists at all.
            ll_tail_1 = loglikelihood(
                ChainSizes([1]; concluded = [false]), NegBin(0.8, 0.5))
            @test ll_tail_1 ≈ 0.0 atol=1e-10

            # Mixed data likelihood is the sum of concluded and
            # ongoing per-observation contributions.
            mixed = ChainSizes([3, 5, 10, 2];
                seeds = [1, 2, 1, 1],
                concluded = [true, true, false, true])
            ll_mixed = loglikelihood(mixed, NegBin(0.8, 0.5))
            ll_parts = logpdf(d, 3) +
                       EpiBranch._chain_size_logpdf(d, 5, 2) +
                       loglikelihood(
                           ChainSizes([10]; concluded = [false]), NegBin(0.8, 0.5)) +
                       logpdf(d, 2)
            @test ll_mixed≈ll_parts atol=1e-10

            # Poisson (Borel) and PoissonGammaChainSize also support the
            # multi-seed formula via _chain_size_logpdf.
            @test isfinite(loglikelihood(
                ChainSizes([3, 5]; seeds = [2, 2]), Poisson(0.6)))
            @test isfinite(loglikelihood(
                ChainSizes([3, 5]; seeds = [2, 2]),
                ClusterMixed(Poisson, Gamma(0.5, 0.8 / 0.5))))

            # Constructor validation: seeds must be ≥ 1 and ≤ observed size.
            @test_throws ArgumentError ChainSizes([3]; seeds = [0])
            @test_throws ArgumentError ChainSizes([2]; seeds = [3])
            @test_throws ArgumentError ChainSizes([3]; seeds = [1, 1])
        end

        @testset "Real-time cluster-size likelihood" begin
            R, k = 0.6, 0.3
            gt = Gamma(2.0, 2.5)
            model = BranchingProcess(NegBin(R, k), gt)

            # End-of-outbreak probability monotone in τ: longer silence
            # → higher confidence the outbreak is extinct.
            π_short = end_of_outbreak_probability(R, k, gt, Dirac(0.0); tau = 1.0)
            π_med = end_of_outbreak_probability(R, k, gt, Dirac(0.0); tau = 14.0)
            π_long = end_of_outbreak_probability(R, k, gt, Dirac(0.0); tau = 200.0)
            @test 0 <= π_short <= π_med <= π_long
            @test π_long ≈ 1.0 atol=1e-6

            # Reporting delay reduces π for the same τ (recent reports
            # could still be in the pipeline).
            π_no_delay = end_of_outbreak_probability(R, k, gt, Dirac(0.0); tau = 5.0)
            π_with_delay = end_of_outbreak_probability(
                R, k, gt, LogNormal(1.5, 0.5); tau = 5.0)
            @test π_with_delay < π_no_delay

            # τ → ∞ should match the all-concluded ChainSizes
            # likelihood (cluster has clearly extinguished).
            #
            # τ = 0 does NOT match all-ongoing: at zero silence the
            # mixture weight on "concluded" is π(0) = exp(-R · S(0))
            # = exp(-R), the probability that the just-observed case
            # has no further offspring. So `ll_zero_tau` lies strictly
            # between `ll_concluded` and `ll_ongoing`, and increasing τ
            # moves the mixture monotonically towards concluded.
            sizes = [3, 5, 10, 2]
            seeds = [1, 2, 1, 1]
            ll_concluded = loglikelihood(
                ChainSizes(sizes; seeds = seeds), NegBin(R, k))
            ll_ongoing = loglikelihood(
                ChainSizes(sizes;
                    seeds = seeds, concluded = falses(length(sizes))),
                NegBin(R, k))

            ll_long_tau = loglikelihood(
                RealTimeChainSizes(sizes, fill(500.0, 4); seeds = seeds),
                model)
            ll_zero_tau = loglikelihood(
                RealTimeChainSizes(sizes, zeros(4); seeds = seeds),
                model)
            ll_med_tau = loglikelihood(
                RealTimeChainSizes(sizes, fill(7.0, 4); seeds = seeds),
                model)
            @test ll_long_tau≈ll_concluded atol=1e-3
            @test min(ll_concluded, ll_ongoing) <= ll_zero_tau <=
                  max(ll_concluded, ll_ongoing)
            # Monotonic interpolation as τ grows: more silence pulls
            # the likelihood towards the concluded value.
            @test abs(ll_long_tau - ll_concluded) <
                  abs(ll_med_tau - ll_concluded) <
                  abs(ll_zero_tau - ll_concluded)

            # Reported wrapper composes a delay around the model. With
            # Dirac(0.0) it should match the bare-model likelihood; with
            # a real delay it pushes the mixture away from extinction
            # (consistent with the per-τ π_with_delay < π_no_delay test
            # above).
            rt_data = RealTimeChainSizes(sizes, fill(7.0, 4); seeds = seeds)
            ll_no_delay = loglikelihood(rt_data, Reported(model, Dirac(0.0)))
            @test ll_no_delay ≈ ll_med_tau atol=1e-10
            ll_with_delay = loglikelihood(rt_data,
                Reported(model, LogNormal(1.5, 0.5)))
            @test ll_with_delay != ll_med_tau

            # State-space form gives the same answer as the Reported wrapper.
            for delay in (Dirac(0.0), LogNormal(1.0, 0.4), Gamma(2.0, 1.5))
                ll_wrap = loglikelihood(rt_data, Reported(model, delay))
                ll_ss = loglikelihood(rt_data,
                    Surveilled(model, PerCaseObservation(1.0, delay)))
                @test ll_wrap ≈ ll_ss atol=1e-12
            end
            # Under-reporting not yet supported via the state-space form.
            @test_throws ArgumentError loglikelihood(rt_data,
                Surveilled(model, PerCaseObservation(0.7, Dirac(0.0))))

            # Constructor validation.
            @test_throws ArgumentError RealTimeChainSizes([3], [-1.0])
            @test_throws ArgumentError RealTimeChainSizes([3], [1.0]; seeds = [0])
            @test_throws ArgumentError RealTimeChainSizes([3, 5], [1.0])
        end

        @testset "Composition: PartiallyObserved(BranchingProcess(ClusterMixed))" begin
            k, R = 0.5, 0.8
            data = ChainSizes([1, 1, 2, 3, 1, 4])
            cm = ClusterMixed(Poisson, Gamma(k, R / k))
            bp = BranchingProcess(cm)

            # With detection_prob = 1 should equal the bare likelihood
            ll_full = loglikelihood(data, PartiallyObserved(bp, 1.0))
            ll_bare = loglikelihood(data, cm)
            @test ll_full ≈ ll_bare atol=1e-6

            # Partial observation reduces the likelihood for small observed sizes
            ll_partial = loglikelihood(data, PartiallyObserved(bp, 0.7))
            @test isfinite(ll_partial)
            @test ll_partial < ll_full

            # Composition also works over the quadrature fallback
            cm_nb = ClusterMixed(R -> NegBin(R, 0.5), Gamma(2.0, 0.3))
            bp_nb = BranchingProcess(cm_nb)
            ll_nb_partial = loglikelihood(data, PartiallyObserved(bp_nb, 0.7))
            @test isfinite(ll_nb_partial)
        end

        @testset "Composition via pipe and stacking" begin
            data = ChainSizes([1, 1, 2, 3, 1, 4])
            model = BranchingProcess(NegBin(0.5, 0.5))

            # Stacking compounds detection probabilities
            ll_nested = loglikelihood(data,
                PartiallyObserved(PartiallyObserved(model, 0.5), 0.5))
            ll_single = loglikelihood(data, PartiallyObserved(model, 0.25))
            @test ll_nested ≈ ll_single

            # Julia pipe
            ll_pipe = loglikelihood(data, model |> PartiallyObserved(0.25))
            @test ll_pipe ≈ ll_single
            ll_stacked_pipe = loglikelihood(data,
                model |> PartiallyObserved(0.5) |> PartiallyObserved(0.5))
            @test ll_stacked_pipe ≈ ll_single
        end

        @testset "Sim ↔ analytical consistency" begin
            # Bare NegBin offspring: simulation should match GammaBorel PMF
            bp = BranchingProcess(NegBin(0.6, 0.5), Exponential(5.0))
            emp, ana = sim_analytical_consistent(bp;
                n_chains = 5000, rng = StableRNG(1))
            for (e, a) in zip(emp, ana)
                @test e≈a atol=0.02
            end

            # ClusterMixed(Poisson, Gamma): simulation should match
            # closed-form PoissonGammaChainSize via dispatch
            k, R = 0.5, 0.6
            cm = ClusterMixed(Poisson, Gamma(k, R / k))
            bp_cm = BranchingProcess(cm, Exponential(5.0))
            emp_cm,
            ana_cm = sim_analytical_consistent(bp_cm;
                n_chains = 5000, rng = StableRNG(42))
            for (e, a) in zip(emp_cm, ana_cm)
                @test e≈a atol=0.02
            end

            # PartiallyObserved(bp, p): simulate bare, thin per case, compare
            # against ThinnedChainSize PMF. This is how any new observation
            # wrapper should be tested — define observe_chain_sizes and
            # the helper does the rest.
            emp_po,
            ana_po = sim_analytical_consistent(
                bp |> PartiallyObserved(0.6);
                n_chains = 5000, rng = StableRNG(7))
            for (e, a) in zip(emp_po, ana_po)
                @test e≈a atol=0.02
            end

            # Verify the per-chain θ invariant: all individuals in a
            # chain share :cluster_theta, and chains with distinct
            # chain_id get independent θ draws (including the
            # n_initial > 1 case where multiple index cases exist).
            model = BranchingProcess(cm, Exponential(5.0))
            states = simulate_batch(model, 500;
                sim_opts = SimOpts(max_cases = 200), rng = StableRNG(3))
            all_consistent = true
            for s in states
                by_chain = Dict{Int, Float64}()
                for ind in s.individuals
                    haskey(ind.state, :cluster_theta) || continue
                    θ = ind.state[:cluster_theta]
                    if haskey(by_chain, ind.chain_id) &&
                       by_chain[ind.chain_id] != θ
                        all_consistent = false
                    else
                        by_chain[ind.chain_id] = θ
                    end
                end
            end
            @test all_consistent

            # With n_initial > 1, each index case starts its own chain
            # and samples an independent θ. Collect θs across runs and
            # check they are not all identical (which would indicate
            # incorrect inheritance).
            states_multi = simulate_batch(model, 200;
                sim_opts = SimOpts(max_cases = 50, n_initial = 3),
                rng = StableRNG(5))
            thetas = Float64[]
            for s in states_multi
                seen_chains = Set{Int}()
                for ind in s.individuals
                    ind.chain_id in seen_chains && continue
                    haskey(ind.state, :cluster_theta) || continue
                    push!(thetas, ind.state[:cluster_theta])
                    push!(seen_chains, ind.chain_id)
                end
            end
            @test length(thetas) > 1
            @test length(unique(thetas)) > 1
        end
    end

    @testset "Chain length likelihood" begin
        @testset "Poisson offspring" begin
            ll = loglikelihood(ChainLengths([0, 1, 0, 2, 1]), Poisson(0.5))
            @test isfinite(ll)
        end

        @testset "NegBin offspring" begin
            ll = loglikelihood(ChainLengths([0, 1, 0, 2, 1]), NegBin(0.8, 0.5))
            @test isfinite(ll)
        end

        @testset "Supercritical throws" begin
            @test_throws ArgumentError loglikelihood(ChainLengths([1, 2]), Poisson(1.5))
            @test_throws ArgumentError loglikelihood(ChainLengths([1, 2]), NegBin(1.5, 0.5))
        end

        @testset "PartiallyObserved rejects ChainLengths" begin
            # Per-case detection doesn't cleanly transform chain length.
            # Raise a clear error rather than routing through an undefined
            # simulate path.
            bp = BranchingProcess(Poisson(0.5), Exponential(5.0))
            @test_throws ArgumentError loglikelihood(
                ChainLengths([0, 1]), PartiallyObserved(bp, 0.7))
        end
    end

    @testset "Simulation-based chain size likelihood" begin
        @testset "Basic evaluation" begin
            model = BranchingProcess(Poisson(0.5), Exponential(5.0))
            ll = loglikelihood(ChainSizes([1, 1, 2, 1, 3]), model; n_sim = 5000, rng = StableRNG(42))
            @test isfinite(ll)
            @test ll < 0.0
        end

        @testset "Consistent with analytical for Poisson" begin
            data = ChainSizes([1, 1, 1, 2, 1])
            ll_analytical = loglikelihood(data, Poisson(0.5))
            model = BranchingProcess(Poisson(0.5), Exponential(5.0))
            ll_simulated = loglikelihood(data, model; n_sim = 10_000, rng = StableRNG(42))
            @test abs(ll_analytical - ll_simulated) < 1.0
        end

        @testset "Fast path catches ClusterMixed analytical" begin
            # BranchingProcess(ClusterMixed(...)) should route to the
            # analytical closed form via the generic kwarg method, not
            # fall through to simulation.
            k, R = 0.5, 0.6
            cm = ClusterMixed(Poisson, Gamma(k, R / k))
            model = BranchingProcess(cm, Exponential(5.0))
            data = ChainSizes([1, 1, 2, 3])
            ll_fast = loglikelihood(data, model; n_sim = 100, rng = StableRNG(1))
            ll_direct = loglikelihood(data, cm)
            @test ll_fast ≈ ll_direct atol=1e-8
        end

        @testset "With interventions" begin
            model = BranchingProcess(Poisson(2.0), Exponential(5.0))
            iso = Isolation(delay = Exponential(1.0))
            ll = loglikelihood(ChainSizes([1, 1, 2, 1]), model;
                interventions = [iso],
                attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
                sim_opts = SimOpts(max_cases = 500),
                n_sim = 500, rng = StableRNG(42))
            @test isfinite(ll)
        end

        @testset "PartiallyObserved sim path with interventions" begin
            # Regression: previously crashed because the generic sim
            # fallback accessed model.offspring, which PartiallyObserved
            # does not have. The merged kwarg method now simulates the
            # wrapped model, thins chain sizes per case, and compares.
            model = BranchingProcess(Poisson(2.0), Exponential(5.0))
            iso = Isolation(delay = Exponential(1.0))
            po = PartiallyObserved(model, 0.7)
            ll = loglikelihood(ChainSizes([1, 1, 2, 1]), po;
                interventions = [iso],
                attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
                sim_opts = SimOpts(max_cases = 500),
                n_sim = 500, rng = StableRNG(42))
            @test isfinite(ll)
        end
    end

    @testset "Simulation-based chain length likelihood" begin
        @testset "Basic evaluation" begin
            model = BranchingProcess(Poisson(0.5), Exponential(5.0))
            ll = loglikelihood(ChainLengths([0, 1, 0, 2, 1]), model; n_sim = 5000, rng = StableRNG(42))
            @test isfinite(ll)
            @test ll < 0.0
        end
    end

    @testset "Proportion transmission (superspreading)" begin
        @testset "Basic 80/20" begin
            # With high R and low k, top 20% should cause most transmission
            prop = proportion_transmission(2.5, 0.16; prop_cases = 0.2)
            @test 0.0 < prop < 1.0
            @test prop > 0.5  # high overdispersion → top 20% cause > 50%
        end

        @testset "No overdispersion → more equal" begin
            # High k means little overdispersion
            prop_low_k = proportion_transmission(2.0, 0.1; prop_cases = 0.2)
            prop_high_k = proportion_transmission(2.0, 100.0; prop_cases = 0.2)
            # Lower k → more unequal → top 20% cause more
            @test prop_low_k > prop_high_k
        end

        @testset "Argument validation" begin
            @test_throws ArgumentError proportion_transmission(-1.0, 0.5)
            @test_throws ArgumentError proportion_transmission(2.0, -0.5)
            @test_throws ArgumentError proportion_transmission(2.0, 0.5; prop_cases = 0.0)
            @test_throws ArgumentError proportion_transmission(2.0, 0.5; prop_cases = 1.0)
        end

        @testset "Known approximate values" begin
            # For k → ∞ (Poisson limit), distribution is nearly uniform
            # Top 20% should cause ≈ 20% of transmission
            prop = proportion_transmission(2.0, 1000.0; prop_cases = 0.2)
            @test prop ≈ 0.2 atol=0.05
        end
    end

    @testset "Proportion cluster size" begin
        @testset "High overdispersion concentrates cases" begin
            # k=0.1 → most cases from large clusters
            prop = proportion_cluster_size(2.0, 0.1; cluster_size = 5)
            @test 0.0 < prop < 1.0
            @test prop > 0.5
        end

        @testset "Low overdispersion spreads cases" begin
            prop_low_k = proportion_cluster_size(2.0, 0.1; cluster_size = 5)
            prop_high_k = proportion_cluster_size(2.0, 10.0; cluster_size = 5)
            @test prop_low_k > prop_high_k
        end

        @testset "cluster_size=1 captures everything" begin
            prop = proportion_cluster_size(2.0, 0.5; cluster_size = 1)
            @test prop ≈ 1.0 atol=1e-6
        end
    end

    @testset "Containment probability" begin
        @testset "Subcritical always contained" begin
            @test probability_contain(0.5, 0.5) == 1.0
        end

        @testset "Supercritical partially contained" begin
            p = probability_contain(2.0, 0.5)
            @test 0.0 < p < 1.0
        end

        @testset "Matches extinction_probability for defaults" begin
            q = extinction_probability(2.0, 0.5)
            p = probability_contain(2.0, 0.5)
            @test p ≈ q atol=1e-8
        end

        @testset "Multiple introductions reduce containment" begin
            p1 = probability_contain(2.0, 0.5; n_initial = 1)
            p5 = probability_contain(2.0, 0.5; n_initial = 5)
            @test p5 < p1
            @test p5 ≈ p1^5 atol=1e-8
        end

        @testset "Population control increases containment" begin
            p_none = probability_contain(2.0, 0.5; pop_control = 0.0)
            p_half = probability_contain(2.0, 0.5; pop_control = 0.5)
            @test p_half > p_none
        end

        @testset "Individual control increases containment" begin
            p_none = probability_contain(2.0, 0.5; ind_control = 0.0)
            p_half = probability_contain(2.0, 0.5; ind_control = 0.5)
            @test p_half > p_none
        end

        @testset "Full control → certain containment" begin
            @test probability_contain(5.0, 0.1; pop_control = 0.9) ≈ 1.0 atol=1e-6
        end
    end

    @testset "Network R" begin
        @testset "Homogeneous contacts" begin
            result = network_R(10.0, 0.0, 1.0, 0.1)
            @test result.R ≈ 1.0
            @test result.R_net ≈ 1.0  # no variance → no adjustment
        end

        @testset "Heterogeneous contacts amplify R" begin
            result = network_R(10.0, 20.0, 1.0, 0.1)
            @test result.R_net > result.R
        end

        @testset "Zero contacts" begin
            result = network_R(0.0, 0.0, 1.0, 0.5)
            @test result.R == 0.0
            @test result.R_net == 0.0
        end
    end

    @testset "Offspring likelihood" begin
        @testset "Basic evaluation" begin
            ll = loglikelihood(OffspringCounts([0, 1, 2, 0, 3]), Poisson(1.2))
            @test isfinite(ll)
            @test ll < 0.0
        end
    end

    @testset "fit" begin
        @testset "Poisson from chain sizes" begin
            rng = StableRNG(42)
            model = BranchingProcess(Poisson(0.5), Exponential(5.0))
            states = simulate_batch(model, 500; rng = rng)
            sizes = Int[]
            for s in states
                cs = chain_statistics(s)
                append!(sizes, cs.size)
            end
            d = fit(Poisson, ChainSizes(sizes))
            @test d isa Poisson
            @test mean(d) ≈ 0.5 atol=0.2
        end

        @testset "NegBin from chain sizes" begin
            rng = StableRNG(42)
            true_R, true_k = 0.6, 0.5
            model = BranchingProcess(NegBin(true_R, true_k), Exponential(5.0))
            states = simulate_batch(model, 500; rng = rng)
            sizes = Int[]
            for s in states
                cs = chain_statistics(s)
                append!(sizes, cs.size)
            end
            d = fit(NegativeBinomial, ChainSizes(sizes))
            @test d isa NegativeBinomial
            @test mean(d) ≈ true_R atol=0.3
        end

        @testset "NegBin from chain lengths" begin
            rng = StableRNG(42)
            true_R, true_k = 0.6, 0.5
            model = BranchingProcess(NegBin(true_R, true_k), Exponential(5.0))
            states = simulate_batch(model, 500; rng = rng)
            lengths = Int[]
            for s in states
                cs = chain_statistics(s)
                append!(lengths, cs.length)
            end
            d = fit(NegativeBinomial, ChainLengths(lengths))
            @test d isa NegativeBinomial
            @test mean(d) ≈ true_R atol=0.3
        end

        @testset "MLE maximises chain-size likelihood" begin
            rng = StableRNG(42)
            model = BranchingProcess(Poisson(0.4), Exponential(5.0))
            states = simulate_batch(model, 300; rng = rng)
            sizes = Int[]
            for s in states
                cs = chain_statistics(s)
                append!(sizes, cs.size)
            end
            data = ChainSizes(sizes)
            d_mle = fit(Poisson, data)
            ll_mle = loglikelihood(data, d_mle)
            ll_other = loglikelihood(data, Poisson(0.9))
            @test ll_mle >= ll_other
        end
    end

    @testset "Data wrapper validation" begin
        @test_throws ArgumentError OffspringCounts(Int[])
        @test_throws ArgumentError OffspringCounts([-1, 0, 1])
        @test_throws ArgumentError ChainSizes(Int[])
        @test_throws ArgumentError ChainSizes([0, 1, 2])
        @test_throws ArgumentError ChainLengths(Int[])
        @test_throws ArgumentError ChainLengths([-1, 0])
    end
end
