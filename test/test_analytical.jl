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

        @testset "Delegates through Observed" begin
            # Regression: extinction_probability goes via
            # single_type_offspring, which must delegate through
            # observation wrappers. Before the fix it threw FieldError
            # because the wrapper has no `offspring` field.
            bp = BranchingProcess(NegBin(0.5, 0.5))
            obs_model = ModelSpec(BranchingProcess(NegBin(0.5, 0.5));
                observation = PerCaseObservation(0.7, Dirac(0.0)))
            @test extinction_probability(obs_model) ==
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

        @testset "Mean depends only on R" begin
            # Expected total progeny is 1/(1-R) for any subcritical
            # offspring family, infinite once R ≥ 1.
            @test mean(GammaBorel(0.5, 0.8)) ≈ 1 / (1 - 0.8)
            @test mean(GammaBorel(0.5, 1.5)) == Inf
        end
    end

    @testset "PoissonGammaChainSize" begin
        @testset "rand throws for all parameters" begin
            # The law is defective for every R: the Gamma rate always
            # places mass above 1, so finite-chain sampling is ill-defined
            # (consistent with the infinite mean). logpdf stays defined.
            @test_throws ArgumentError rand(PoissonGammaChainSize(0.5, 0.3))
            d_super = PoissonGammaChainSize(0.5, 1.5)
            @test isfinite(logpdf(d_super, 2))
            @test_throws ArgumentError rand(d_super)
        end

        @testset "Mean is infinite for all parameters" begin
            # The Gamma rate always places density at/above 1, where the
            # conditional chain size 1/(1-λ) diverges.
            @test mean(PoissonGammaChainSize(0.5, 0.8)) == Inf
            @test mean(PoissonGammaChainSize(2.0, 0.3)) == Inf
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
            mk(p) = ModelSpec(BranchingProcess(Poisson(0.5));
                observation = PerCaseObservation(p, Dirac(0.0)))

            ll = loglikelihood(ChainSizes([1, 1, 2]), mk(0.8))
            @test isfinite(ll)

            # ρ = 1 should agree with the bare offspring likelihood.
            ll_full = loglikelihood(ChainSizes([1, 1, 2]), mk(1.0))
            ll_bare = loglikelihood(ChainSizes([1, 1, 2]), Poisson(0.5))
            @test ll_full ≈ ll_bare atol=1e-6

            # Stricter detection should lower the likelihood for small sizes.
            ll_strict = loglikelihood(ChainSizes([1, 1, 2]), mk(0.3))
            @test isfinite(ll_strict)

            # Reporting delay is irrelevant for closed-outbreak chain
            # sizes — same answer regardless of delay distribution.
            ll_with_delay = loglikelihood(ChainSizes([1, 1, 2]),
                ModelSpec(BranchingProcess(Poisson(0.5));
                    observation = PerCaseObservation(0.7, LogNormal(1.0, 0.5))))
            ll_no_delay = loglikelihood(ChainSizes([1, 1, 2]),
                ModelSpec(BranchingProcess(Poisson(0.5));
                    observation = PerCaseObservation(0.7, Dirac(0.0))))
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

        @testset "Real-time mixture likelihood and end_of_outbreak_probability" begin
            # Right-tail helper: P(X ≥ s) = 1 ⇒ logright_tail = 0.
            d = GammaBorel(0.5, 0.8)
            @test EpiBranch._chain_size_right_tail_logprob(d, 1, 1) == 0.0
            @test EpiBranch._chain_size_right_tail_logprob(d, 2, 2) == 0.0

            # log P(X = x) + log P(X ≥ x + 1) sums consistently with the
            # PMF: P(X = x) + P(X ≥ x + 1) = P(X ≥ x).
            for x in 2:8
                pmf = exp(EpiBranch._chain_size_logpdf(d, x, 1))
                tail_below = exp(EpiBranch._chain_size_right_tail_logprob(d, x, 1))
                tail_above = exp(EpiBranch._chain_size_right_tail_logprob(d, x + 1, 1))
                @test tail_below≈pmf + tail_above atol=1e-9
            end

            # prob_concluded == ones reproduces the concluded-only likelihood.
            data = ChainSizes([1, 1, 2, 3])
            @test loglikelihood(data, NegBin(0.8, 0.5)) ≈
                  loglikelihood(data, NegBin(0.8, 0.5);
                prob_concluded = [1.0, 1.0, 1.0, 1.0])

            # prob_concluded == zeros ⇒ all clusters ongoing (right-tail only).
            ll_ongoing = loglikelihood(data, NegBin(0.8, 0.5);
                prob_concluded = [0.0, 0.0, 0.0, 0.0])
            d = GammaBorel(0.5, 0.8)
            ll_expected = sum(
                EpiBranch._chain_size_right_tail_logprob(d, x, 1)
            for x in data.data)
            @test ll_ongoing ≈ ll_expected

            # Mixture: ll bounded between the all-concluded and all-ongoing ends.
            ll_mid = loglikelihood(data, NegBin(0.8, 0.5);
                prob_concluded = [0.5, 0.5, 0.5, 0.5])
            ll_concluded = loglikelihood(data, NegBin(0.8, 0.5))
            @test min(ll_concluded, ll_ongoing) <= ll_mid <=
                  max(ll_concluded, ll_ongoing)

            # Likelihood-time validation of `prob_concluded`.
            @test_throws ArgumentError loglikelihood(
                data, NegBin(0.8, 0.5); prob_concluded = [0.5, 0.5])     # wrong length
            @test_throws ArgumentError loglikelihood(
                data, NegBin(0.8, 0.5); prob_concluded = [0.5, 0.5, 0.5, 1.1])  # out of [0, 1]
            @test_throws ArgumentError loglikelihood(
                data, NegBin(0.8, 0.5); prob_concluded = [-0.1, 0.5, 0.5, 0.5])

            # end_of_outbreak_probability: G(0) at τ=0, → 1 as τ → ∞, monotone.
            GT = Gamma(2.78, 1.8)
            R, k = 2.5, 0.1
            @test end_of_outbreak_probability(R, k, GT, 0.0) ≈ (k / (k + R))^k
            @test end_of_outbreak_probability(R, k, GT, Inf) == 1.0
            taus = 0.0:1.0:30.0
            πs = [end_of_outbreak_probability(R, k, GT, τ) for τ in taus]
            @test issorted(πs)
            @test all(0 .<= πs .<= 1)

            # Poisson overload: π(τ=0) = exp(-R); → 1 as τ → ∞.
            @test end_of_outbreak_probability(Poisson(R), GT, 0.0) ≈ exp(-R)
            @test end_of_outbreak_probability(Poisson(R), GT, Inf) == 1.0

            # NegBin offspring overload agrees with the (R, k) form.
            @test end_of_outbreak_probability(NegBin(R, k), GT, 5.0) ≈
                  end_of_outbreak_probability(R, k, GT, 5.0)

            # BranchingProcess overload agrees with the offspring/gt one.
            bp = BranchingProcess(NegBin(R, k), GT)
            @test end_of_outbreak_probability(bp, 5.0) ≈
                  end_of_outbreak_probability(R, k, GT, 5.0)

            # The Observed{..., PerCaseObservation} wrapper is refused
            # explicitly: under-reporting (ρ < 1) needs the Volterra
            # recursion which is not implemented.
            om = ModelSpec(BranchingProcess(NegBin(R, k), GT);
                observation = PerCaseObservation(0.5, Dirac(0.0)))
            @test_throws ArgumentError end_of_outbreak_probability(om, 5.0)

            # Vector-of-τ overload returns same elements.
            πs_vec = end_of_outbreak_probability(R, k, GT, collect(taus))
            @test πs_vec ≈ πs

            # End-to-end: end_of_outbreak_probability feeds the prob_concluded kwarg
            # and the mixture likelihood evaluates without error.
            sizes = [1, 2, 100, 1766]
            seeds = [1, 1, 3, 17]
            taus_data = [0.0, 7.0, 0.0, 0.0]
            π_vals = [end_of_outbreak_probability(R, k, GT, τ) for τ in taus_data]
            data_endo = ChainSizes(sizes; seeds = seeds)
            ll = loglikelihood(data_endo, NegBin(R, k); prob_concluded = π_vals)
            @test isfinite(ll)
        end

        @testset "Per-case observation: simulation decoration" begin
            R, k = 0.6, 0.3
            gt = Gamma(2.0, 2.5)

            # simulate(::Observed) decorates each individual with
            # :reported and :report_time so downstream code can filter
            # to observed reports.
            using StableRNGs
            sim_state = simulate(
                ModelSpec(BranchingProcess(NegBin(R, k), gt);
                    observation = PerCaseObservation(0.6, LogNormal(1.0, 0.4)));
                rng = StableRNG(42))
            @test all(haskey(ind.state, :reported) for ind in sim_state.individuals)
            @test all(haskey(ind.state, :report_time) for ind in sim_state.individuals)
            @test all(ind.state[:report_time] >= ind.infection_time
            for ind in sim_state.individuals)
        end

        @testset "Composition: BranchingProcess(ClusterMixed) with observation" begin
            k, R = 0.5, 0.8
            data = ChainSizes([1, 1, 2, 3, 1, 4])
            cm = ClusterMixed(Poisson, Gamma(k, R / k))
            mk(p) = ModelSpec(BranchingProcess(cm);
                observation = PerCaseObservation(p, Dirac(0.0)))

            # ρ = 1 should equal the bare likelihood.
            ll_full = loglikelihood(data, mk(1.0))
            ll_bare = loglikelihood(data, cm)
            @test ll_full ≈ ll_bare atol=1e-6

            # ρ < 1 reduces the likelihood for small observed sizes.
            ll_partial = loglikelihood(data, mk(0.7))
            @test isfinite(ll_partial)
            @test ll_partial < ll_full

            # Composition also works over the quadrature fallback.
            cm_nb = ClusterMixed(R -> NegBin(R, 0.5), Gamma(2.0, 0.3))
            ll_nb_partial = loglikelihood(data,
                ModelSpec(BranchingProcess(cm_nb);
                    observation = PerCaseObservation(0.7, Dirac(0.0))))
            @test isfinite(ll_nb_partial)
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

            # BranchingProcess with a PerCaseObservation(p, ...): simulate
            # bare, thin per case, compare against ThinnedChainSize PMF.
            # This is how any new observation model should be tested —
            # define observe_chain_sizes and the helper does the rest.
            emp_po,
            ana_po = sim_analytical_consistent(
                ModelSpec(BranchingProcess(NegBin(0.6, 0.5), Exponential(5.0));
                    observation = PerCaseObservation(0.6, Dirac(0.0)));
                n_chains = 5000, rng = StableRNG(7))
            for (e, a) in zip(emp_po, ana_po)
                @test e≈a atol=0.02
            end

            # Verify the per-chain θ invariant: all individuals in a
            # chain share :cluster_theta, and chains with distinct
            # chain_id get independent θ draws (including the
            # n_initial > 1 case where multiple index cases exist).
            model = BranchingProcess(cm, Exponential(5.0))
            states = simulate(model, 500;
                max_cases = 200, rng = StableRNG(3))
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
            states_multi = simulate(model, 200;
                max_cases = 50, n_initial = 3,
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

        @testset "Observed + PerCaseObservation rejects ChainLengths" begin
            # Per-case detection doesn't cleanly transform chain length.
            # Raise a clear error rather than routing through an undefined
            # simulate path.
            @test_throws ArgumentError loglikelihood(
                ChainLengths([0, 1]),
                ModelSpec(BranchingProcess(Poisson(0.5), Exponential(5.0));
                    observation = PerCaseObservation(0.7, Dirac(0.0))))
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
            iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
            ll = loglikelihood(ChainSizes([1, 1, 2, 1]),
                ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
                    interventions = [iso],
                    attributes = clinical_presentation(
                        incubation_period = LogNormal(1.5, 0.5)));
                max_cases = 500,
                n_sim = 500,
                rng = StableRNG(42))
            @test isfinite(ll)
        end

        @testset "Observed sim path with interventions" begin
            # Regression: previously crashed because the generic sim
            # fallback accessed model.offspring, which the Observed
            # wrapper does not have. The intervention path now simulates
            # the wrapped process, thins chain sizes per case, and compares.
            iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
            ll = loglikelihood(ChainSizes([1, 1, 2, 1]),
                ModelSpec(BranchingProcess(Poisson(2.0), Exponential(5.0));
                    observation = PerCaseObservation(0.7, Dirac(0.0)),
                    interventions = [iso],
                    attributes = clinical_presentation(
                        incubation_period = LogNormal(1.5, 0.5)));
                max_cases = 500,
                n_sim = 500,
                rng = StableRNG(42))
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
            result = heterogeneous_contact_R(10.0, 0.0, 1.0, 0.1)
            @test result.R ≈ 1.0
            @test result.R_net ≈ 1.0  # no variance → no adjustment
        end

        @testset "Heterogeneous contacts amplify R" begin
            result = heterogeneous_contact_R(10.0, 20.0, 1.0, 0.1)
            @test result.R_net > result.R
        end

        @testset "Zero contacts" begin
            result = heterogeneous_contact_R(0.0, 0.0, 1.0, 0.5)
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

    @testset "Data wrapper validation" begin
        @test_throws ArgumentError OffspringCounts(Int[])
        @test_throws ArgumentError OffspringCounts([-1, 0, 1])
        @test_throws ArgumentError ChainSizes(Int[])
        @test_throws ArgumentError ChainSizes([0, 1, 2])
        @test_throws ArgumentError ChainLengths(Int[])
        @test_throws ArgumentError ChainLengths([-1, 0])
    end
end
