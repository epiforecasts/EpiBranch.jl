# Records the name of every route `risk_applies` is asked about, so a test
# can check which named windows a model's routes reach it as.
struct _SeenRoutes <: EpiBranch.AbstractIntervention
    seen::Vector{Symbol}
end
function EpiBranch.risk_applies(iv::_SeenRoutes, route)
    route === nothing || push!(iv.seen, route.name)
    return true
end
# Acts only through `risk_applies`, which `ModelSpec`'s check does not scan.
EpiBranch.has_effect(::_SeenRoutes) = true

@testset "Infectiousness windows" begin
    @testset "explicit single window holds offspring + kernel" begin
        m = BranchingProcess(Infectiousness(NegBin(2.5, 0.16); kernel = LogNormal(1.6, 0.5)))
        @test length(m.infectiousness) == 1
        @test m.infectiousness[1].from == :infection
        @test single_type_offspring(m) isa NegativeBinomial
        @test m.infectiousness[1].kernel isa LogNormal
        @test m.infectiousness[1].name === :transmission
    end

    @testset "a calendar scales the window's kernel over calendar time" begin
        # One contact attempt per case, timed from an exponential hazard of 1
        # per day; a calendar that drops the rate to 0 from day `cutoff` leaves
        # a secondary case only when the attempt falls before `cutoff`, with
        # probability `1 - exp(-cutoff)`.
        cutoff = 1.5
        off = Infectiousness(
            Dirac(1); kernel = Exponential(1.0), calendar = Steps([cutoff], [1.0, 0.0])
        )
        @test off.calendar isa Steps
        m = BranchingProcess(off)
        n = 4000
        has_case = [
            simulate(m; n_initial = 1, max_generations = 1, rng = StableRNG(s)).cumulative_cases > 1
                for s in 1:n
        ]
        @test mean(has_case) ≈ 1 - exp(-cutoff) atol = 0.02

        # With no calendar at all, every attempt eventually causes an infection.
        uncut = BranchingProcess(Infectiousness(Dirac(1); kernel = Exponential(1.0)))
        always = [
            simulate(uncut; n_initial = 1, max_generations = 1, rng = StableRNG(s)).cumulative_cases
                for s in 1:200
        ]
        @test all(==(2), always)
    end

    @testset "a named window reaches risk_applies by that name" begin
        # `risk_applies` could not previously tell the two windows apart: both
        # reached the race as a `RouteWindow` named `:transmission`. Naming
        # them lets an intervention select one and leave the other alone.
        household = Infectiousness(Poisson(1.0); kernel = Uniform(0, 5), name = :household)
        community = Infectiousness(Poisson(2.0); kernel = Uniform(0, 5), name = :community)
        model = BranchingProcess(household, community)
        seen = _SeenRoutes(Symbol[])
        simulate(
            ModelSpec(model; interventions = [seen]);
            max_generations = 1, rng = StableRNG(1)
        )
        @test Set(seen.seen) == Set([:household, :community])
    end

    @testset "from-state offsets the generation interval" begin
        latent = Transition(:infectious, from = :infection, delay = (rng, ind) -> 4.0)
        m = ModelSpec(
            BranchingProcess(
                Infectiousness(
                    (rng, ind) -> 3;
                    from = :infectious, kernel = Exponential(0.5)
                )
            );
            progression = [latent]
        )
        s = simulate(
            m;
            n_initial = 30, max_generations = 1, rng = StableRNG(1)
        )
        kids = filter(i -> i.parent_id != 0 && is_infected(i), s.individuals)
        gi = [k.infection_time - s.individuals[k.parent_id].infection_time for k in kids]
        @test minimum(gi) > 4.0   # latent 4 + a positive contact interval
    end

    @testset "a deep ring reaching a case before its window opens leaves its offspring" begin
        # Every case has three contacts, so tracing that does not isolate
        # leaves the size of each generation fixed at any ring depth.
        latent = Transition(:infectious, from = :infection, delay = (rng, ind) -> 4.0)
        m = BranchingProcess(
            Infectiousness((rng, ind) -> 3; from = :infectious, kernel = Exponential(0.5))
        )
        for depth in (1, 2)
            ct = ContactTracing(
                TraceEveryone(), 1.0, Exponential(1000.0), FlagOnly(); depth = depth
            )
            s = simulate(
                ModelSpec(m; progression = [latent], interventions = [ct]);
                n_initial = 5, max_generations = 2, rng = StableRNG(1)
            )
            @test s.cumulative_cases == 5 + 15 + 45
        end
    end

    @testset "a window whose from-state is never reached produces no contacts" begin
        # `from = :died` but no death transition, so `:died_time` is never
        # set and the window never opens: the outbreak cannot spread.
        # Composing the process with a progression warns about the
        # unreachable from-state.
        m = @test_logs (:warn,) match_mode = :any ModelSpec(
            BranchingProcess(
                Infectiousness((rng, ind) -> 5; from = :died, kernel = Exponential(1.0))
            )
        )
        s = simulate(
            m; n_initial = 10, max_generations = 3,
            rng = StableRNG(2)
        )
        @test s.cumulative_cases == 10
    end

    @testset "two windows each contribute offspring" begin
        w1 = Infectiousness((rng, ind) -> 2; from = :infection)  # default kernel: no timing
        w2 = Infectiousness((rng, ind) -> 3; from = :infection)
        m = BranchingProcess(w1, w2)
        s = simulate(
            m; n_initial = 1, max_generations = 1,
            rng = StableRNG(3)
        )
        kids = filter(i -> i.parent_id != 0, s.individuals)
        @test length(kids) == 5   # 2 + 3 from the single index
    end

    @testset "until censors contacts after the infector is removed" begin
        off = (rng, ind) -> 10
        kernel = Exponential(1.0)
        win = Infectiousness(off; from = :infection, until = (:recovered,), kernel = kernel)
        opts = (; n_initial = 5, max_generations = 1)

        # Recovery far in the future: no contact is censored.
        rec_late = Transition(:recovered, from = :infection, delay = (rng, ind) -> 100.0)
        s_late = simulate(
            ModelSpec(BranchingProcess(win); progression = [rec_late]);
            opts..., rng = StableRNG(1)
        )
        kids_late = count(i -> i.parent_id != 0 && is_infected(i), s_late.individuals)
        @test kids_late == 50   # 5 index × 10, nothing removed early

        # Near-instant recovery: almost every contact lands after removal.
        rec_early = Transition(:recovered, from = :infection, delay = (rng, ind) -> 0.001)
        s_early = simulate(
            ModelSpec(BranchingProcess(win); progression = [rec_early]);
            opts..., rng = StableRNG(1)
        )
        kids_early = count(i -> i.parent_id != 0 && is_infected(i), s_early.individuals)
        @test kids_early < 5
        @test kids_early < kids_late
    end

    @testset "chain-size likelihood routes a window with until to simulation" begin
        # Transmission stops at infection (recovery is immediate), so every
        # cluster is a singleton; the bare offspring law `until` leaves
        # unscored would give log(pdf(Poisson(0.8), 1)) ≈ -0.8 instead.
        w = Infectiousness(Poisson(0.8); until = (:recovered,), kernel = Exponential(1.0))
        spec = ModelSpec(
            BranchingProcess(w);
            progression = [
                Transition(:recovered, from = :infection, delay = 0.0, terminal = true),
            ]
        )
        @test !EpiBranch.offspring_law_unchanged(spec.process)
        ll = loglikelihood(ChainSizes([1]), spec; rng = StableRNG(1), n_sim = 500)
        @test isapprox(ll, 0.0; atol = 1.0e-9)
    end

    @testset "an offspring function can return OffspringTimes instead of a count" begin
        # The kernel draws from an exponential with mean 1000, so a contact
        # timed from it would almost never land at exactly 1.0 or 4.5.
        off = (rng, ind) -> OffspringTimes([1.0, 1.0, 4.5])
        m = BranchingProcess(Infectiousness(off; kernel = Exponential(1000.0)))
        s = simulate(m; n_initial = 1, max_generations = 1, rng = StableRNG(1))
        kids = filter(i -> i.parent_id != 0 && is_infected(i), s.individuals)
        @test sort([k.infection_time for k in kids]) == [1.0, 1.0, 4.5]
    end

    @testset "OffspringTimes before the case's own infection drops that contact" begin
        # `from = :infectious` opens 4 days after infection; an offset of
        # -5.0 then falls before the case's own infection, so it is dropped
        # exactly as a kernel draw landing there would be.
        latent = Transition(:infectious, from = :infection, delay = (rng, ind) -> 4.0)
        off = (rng, ind) -> OffspringTimes([-5.0, 1.0])
        m = ModelSpec(
            BranchingProcess(Infectiousness(off; from = :infectious));
            progression = [latent]
        )
        s = simulate(m; n_initial = 1, max_generations = 1, rng = StableRNG(1))
        kids = filter(i -> i.parent_id != 0 && is_infected(i), s.individuals)
        @test length(kids) == 1
        @test only(kids).infection_time ≈ 5.0
    end

    @testset "OffspringTimes raises in a multi-type model, which it cannot label" begin
        off = (rng, ind) -> OffspringTimes([1.0, 4.5])
        m = BranchingProcess(Infectiousness(off); n_types = 2)
        @test_throws ArgumentError simulate(
            m; n_initial = 1, max_generations = 1, rng = StableRNG(1)
        )
    end

    @testset "analytics: single window closed-form, multi-window directs to simulation" begin
        single = BranchingProcess(NegBin(2.0, 0.5), LogNormal(1.6, 0.5))
        @test single_type_offspring(single) isa NegativeBinomial
        @test extinction_probability(single) isa Real

        multi = BranchingProcess(
            Infectiousness(NegBin(1.5, 0.5); from = :infection),
            Infectiousness(NegBin(0.8, 0.5); from = :infection)
        )
        @test_throws ArgumentError single_type_offspring(multi)
        @test_throws ArgumentError extinction_probability(multi)
    end
end

@testset "HazardKernel" begin
    @testset "a constant hazard matches the corresponding exponential" begin
        rate = 0.4
        k = HazardKernel(t -> rate)
        e = Exponential(1 / rate)
        for tau in (0.1, 1.0, 5.0)
            @test EpiBranch.cumhazard(k, tau) ≈ EpiBranch.cumhazard(e, tau)
        end
        draws = [rand(StableRNG(seed), k) for seed in 1:2000]
        @test isapprox(sum(draws) / length(draws), mean(e); rtol = 0.1)
    end

    @testset "a hazard built from a named distribution reproduces its cumulative hazard" begin
        w = Weibull(1.5, 3.0)
        k = HazardKernel(t -> EpiBranch.hazard(w, t))
        for tau in (0.1, 1.0, 3.0, 8.0)
            @test EpiBranch.cumhazard(k, tau) ≈ EpiBranch.cumhazard(w, tau) rtol = 1.0e-6
        end
    end

    @testset "hazard and loghazard read back the stored hazard function" begin
        k = HazardKernel(t -> 2.0 + t)
        for tau in (0.1, 1.0, 3.0)
            @test EpiBranch.hazard(k, tau) == 2.0 + tau
            @test EpiBranch.loghazard(k, tau) ≈ log(2.0 + tau)
        end
    end

    @testset "hazard and loghazard are zero outside (onset, upper)" begin
        k = HazardKernel(t -> 2.0; onset = 15.0, upper = 30.0)
        @test EpiBranch.hazard(k, 3.0) == 0.0
        @test EpiBranch.loghazard(k, 3.0) == -Inf
        @test EpiBranch.hazard(k, 30.0) == 0.0
        @test EpiBranch.hazard(k, 16.0) == 2.0
        # A calendar-scaled window integrates the kernel's own hazard rather
        # than its `cumhazard`, so the onset has to hold there as well.
        kernel = HazardKernel(t -> 2.0; onset = 15.0)
        window = Infectiousness(
            Poisson(3.0); kernel,
            calendar = Seasonal(amplitude = 0.5, peak_day = 0.0, period = 365.0)
        )
        s = simulate(
            ModelSpec(BranchingProcess(window));
            n_initial = 50, max_generations = 1, rng = StableRNG(1)
        )
        kids = filter(i -> i.parent_id != 0 && is_infected(i), s.individuals)
        @test length(kids) > 0
        @test all(kids) do kid
            kid.infection_time - s.individuals[kid.parent_id].infection_time >= 15.0
        end
    end

    @testset "upper bounds the integration, and a draw beyond it never happens" begin
        k = HazardKernel(t -> 2.0; upper = 1.0)
        @test maximum(k) == Inf
        @test minimum(k) == 0.0
        draws = [rand(StableRNG(seed), k) for seed in 1:500]
        @test all(d -> d <= 1.0 || d == Inf, draws)
        # The total hazard is 2, so about exp(-2) of the draws exhaust it.
        @test isapprox(count(isinf, draws) / length(draws), exp(-2.0); atol = 0.05)
        @test invlogccdf(k, -Inf) == Inf
        @test logccdf(k, 1.0) ≈ -2.0
        @test logccdf(k, 5.0) ≈ -2.0
        @test EpiBranch.cumhazard(k, 5.0) ≈ 2.0
        @test_throws ArgumentError invlogccdf(k, Inf)
        @test_throws ArgumentError invlogccdf(k, NaN)
    end

    @testset "a finite total hazard under a smooth calendar drops the draws it cannot reach" begin
        # The calendar-scaled path inverts the combined hazard itself, so a
        # draw beyond the total must come back as `Inf` here too, whether or
        # not the user bounded the integration, rather than growing the
        # bracket until the calendar's phase overflows.
        cal = Seasonal(amplitude = 0.5, peak_day = 0.0, period = 365.0)
        count_kids(upper) = begin
            kernel = ind -> HazardKernel(t -> 0.1 * exp(-t); upper)
            w = Infectiousness(Poisson(10.0); kernel, calendar = cal)
            s = simulate(
                ModelSpec(BranchingProcess(w));
                n_initial = 200, max_generations = 1, rng = StableRNG(3)
            )
            count(i -> i.parent_id != 0 && is_infected(i), s.individuals)
        end
        # The multiplier is about 1.5 over the few days the hazard lasts.
        expected = 10 * (1 - exp(-0.15))
        @test isapprox(count_kids(50.0) / 200, expected; rtol = 0.15)
        @test isapprox(count_kids(Inf) / 200, expected; rtol = 0.15)
        # The race's scaled draw takes the kernel's `Inf` as the answer rather
        # than bisecting a survival that never reaches the target.
        k = HazardKernel(t -> 0.1 * exp(-t); upper = 50.0)
        @test EpiBranch._time_at_log_survival(k, -0.5) == Inf
        @test EpiBranch._time_at_log_survival(k, -0.05) ≈ invlogccdf(k, -0.05)
        # An unbounded kernel under the calendar is trusted too, rather than
        # bisected again to overflow: the search itself is the whole cost.
        calls = Ref(0)
        counted = HazardKernel(t -> (calls[] += 1; 0.1 * exp(-t)))
        ck = EpiBranch._CalendarScaledKernel(counted, cal, 0.0)
        calls[] = 0
        direct = invlogccdf(ck, -0.5)
        search = calls[]
        calls[] = 0
        @test EpiBranch._time_at_log_survival(ck, -0.5) == Inf == direct
        @test calls[] <= search + 10
    end

    @testset "a bound on the integration leaves the realised R unchanged" begin
        # A hazard that integrates to 0.1 infects about a tenth of the
        # contacts it is offered, whether or not the user wrote a bound.
        count_kids(upper) = begin
            k = HazardKernel(t -> 0.1 * exp(-t); upper)
            m = ModelSpec(BranchingProcess(Infectiousness(Poisson(10.0); kernel = k)))
            s = simulate(m; n_initial = 200, max_generations = 1, rng = StableRNG(3))
            count(i -> i.parent_id != 0 && is_infected(i), s.individuals)
        end
        bounded, unbounded = count_kids(50.0), count_kids(Inf)
        @test isapprox(bounded / 200, 10 * (1 - exp(-0.1)); rtol = 0.15)
        @test bounded == unbounded
    end

    @testset "a hazard that has cleared is still summed over a long range" begin
        # One quadrature pass over [0, 1000] has no node in [0, 2].
        k = HazardKernel(t -> t < 2.0 ? 1.0 : 0.0; upper = 1000.0)
        for tau in (50.0, 600.0, 1000.0, Inf)
            @test EpiBranch.cumhazard(k, tau) ≈ 2.0
        end
        @test logccdf(k, Inf) ≈ -2.0
        @test EpiBranch.cumhazard(HazardKernel(t -> t < 2.0 ? 1.0 : 0.0), 1000.0) ≈ 2.0
        ck = EpiBranch._CalendarScaledKernel(k, t -> 1.0, 0.0)
        @test EpiBranch.cumhazard(ck, 1000.0) ≈ 2.0
        @test isfinite(truncated(k, 0.0, 900.0).logF_hi)
        # The reads stay inferred, because the pairwise likelihood calls them
        # once per pair.
        @test only(Base.return_types(EpiBranch.cumhazard, (typeof(k), Float64))) == Float64
        # A decaying hazard is not refined over the tail it has left: a long
        # range costs about the evaluations a short one does.
        calls = Ref(0)
        decaying = HazardKernel(t -> (calls[] += 1; 0.1 * exp(-t)))
        calls[] = 0
        EpiBranch.cumhazard(decaying, 50.0)
        short = calls[]
        calls[] = 0
        EpiBranch.cumhazard(decaying, 1000.0)
        @test calls[] <= 2 * short
        # Nor is a steep rise after a faint run-up refined without end.
        calls[] = 0
        rising = HazardKernel(t -> (calls[] += 1; 1 / (1 + exp(-2 * (t - 50.0)))))
        @test EpiBranch.cumhazard(rising, 100.0) ≈ 50.0 rtol = 1.0e-6
        @test calls[] < 2000
    end

    @testset "the total hazard over an unbounded horizon is infinite where it diverges" begin
        @test logccdf(HazardKernel(t -> 2.0), Inf) == -Inf
        @test EpiBranch.cumhazard(HazardKernel(t -> 0.1 * exp(-t)), Inf) ≈ 0.1
    end

    @testset "an onset or upper holding a parameter differentiates" begin
        d_onset = ForwardDiff.derivative(1.0) do θ
            EpiBranch.cumhazard(HazardKernel(t -> 0.5; onset = θ), 5.0)
        end
        @test d_onset ≈ ForwardDiff.derivative(θ -> EpiBranch.cumhazard(Exponential(2.0) + θ, 5.0), 1.0)
        d_total = ForwardDiff.derivative(1.0) do θ
            EpiBranch.cumhazard(HazardKernel(t -> 2.0 * exp(-t); onset = θ), Inf)
        end
        @test d_total ≈ -2 * exp(-1)
        d_upper = ForwardDiff.derivative(3.0) do θ
            EpiBranch.cumhazard(HazardKernel(t -> 2.0; upper = θ), 5.0)
        end
        @test d_upper ≈ 2.0
        cal = Seasonal(amplitude = 0.5, peak_day = 0.0, period = 365.0)
        d_upper_cal = ForwardDiff.derivative(3.0) do θ
            EpiBranch.cumhazard(EpiBranch._CalendarScaledKernel(HazardKernel(t -> 2.0; upper = θ), cal, 0.0), Inf)
        end
        @test d_upper_cal ≈ 2 * EpiBranch.calendar_multiplier(cal, 3.0)
        # Drawing through a dual bound runs, the bound's derivative aside.
        @test isfinite(ForwardDiff.derivative(θ -> quantile(HazardKernel(t -> 2.0; onset = θ), 0.5), 1.0))
        # The truncation the `offspring` form builds holds a dual too.
        d_rate = ForwardDiff.derivative(2.0) do θ
            quantile(truncated(HazardKernel(t -> θ), 0.0, 3.0), 0.5)
        end
        # Against the closed form of the exponential truncated at 3.
        @test d_rate ≈ ForwardDiff.derivative(θ -> -log1p(-0.5 * (1 - exp(-3θ))) / θ, 2.0)
        @test isfinite(ForwardDiff.derivative(θ -> quantile(truncated(HazardKernel(t -> 2.0; onset = θ), 0.0, 3.0), 0.5), 1.0))
        @test isfinite(ForwardDiff.derivative(θ -> quantile(truncated(HazardKernel(t -> 2.0), 0.0, θ), 0.5), 3.0))
    end

    @testset "rejects a non-positive upper" begin
        @test_throws ArgumentError HazardKernel(t -> 1.0; upper = 0.0)
    end

    @testset "cdf and logcdf match the corresponding exponential" begin
        rate = 0.4
        k = HazardKernel(t -> rate)
        e = Exponential(1 / rate)
        for tau in (0.1, 1.0, 5.0)
            @test cdf(k, tau) ≈ cdf(e, tau)
            @test logcdf(k, tau) ≈ logcdf(e, tau)
        end
    end

    @testset "logpdf and pdf match the corresponding exponential" begin
        rate = 0.4
        k = HazardKernel(t -> rate)
        e = Exponential(1 / rate)
        for tau in (0.1, 1.0, 5.0)
            @test logpdf(k, tau) ≈ logpdf(e, tau)
            @test pdf(k, tau) ≈ pdf(e, tau)
        end
        @test logpdf(k, -1.0) == -Inf
        # A delay drawn from a kernel can be scored like any other.
        delay = HazardKernel(t -> 0.5)
        @test EpiBranch._delay_loglik(delay, 2.0) ≈ logpdf(Exponential(2.0), 2.0)
    end

    @testset "a truncated HazardKernel's cdf, logcdf and logccdf match truncating by hand" begin
        rate = 0.4
        hi = 2.0
        tk = truncated(HazardKernel(t -> rate), 0.0, hi)
        te = truncated(Exponential(1 / rate), 0.0, hi)
        for tau in (0.1, 1.0, 1.9)
            @test cdf(tk, tau) ≈ cdf(te, tau)
            @test logcdf(tk, tau) ≈ logcdf(te, tau)
            @test logccdf(tk, tau) ≈ logccdf(te, tau)
        end
        @test cdf(tk, hi) == 1.0
        @test logcdf(tk, hi) == 0.0
    end

    @testset "a truncated kernel is a Distribution, so every seam reaches it" begin
        tk = truncated(HazardKernel(t -> 2.0), 0.0, 5.0)
        @test tk isa Distribution
        te = truncated(Exponential(0.5), 0.0, 5.0)
        for tau in (0.1, 1.0, 4.9)
            @test logpdf(tk, tau) ≈ logpdf(te, tau)
            @test EpiBranch.loghazard(tk, tau) ≈ logpdf(te, tau) - logccdf(te, tau)
            @test EpiBranch.cumhazard(tk, tau) ≈ -logccdf(te, tau)
        end
        @test logpdf(tk, 5.0) == -Inf
        # Between the base kernel's `upper` and the cap the hazard is zero.
        tu = truncated(HazardKernel(t -> 2.0; upper = 3.0), 0.0, 5.0)
        @test EpiBranch.loghazard(tu, 4.0) == -Inf
        @test EpiBranch.hazard(tu, 4.0) == 0.0
        # Under a calendar the survival has ended at the base `upper` too.
        cu = EpiBranch._CalendarScaledKernel(tu, Seasonal(amplitude = 0.5, peak_day = 0.0, period = 365.0), 0.0)
        @test EpiBranch.cumhazard(cu, 4.0) == Inf
        @test isfinite(EpiBranch.cumhazard(cu, 2.0))
        interval(s, kid) = kid.infection_time - s.individuals[kid.parent_id].infection_time
        # As a bare window kernel, through `get_generation_time`.
        m = ModelSpec(BranchingProcess(Infectiousness(Poisson(3.0); kernel = tk)))
        s = simulate(m; n_initial = 50, max_generations = 1, rng = StableRNG(1))
        kids = filter(i -> i.parent_id != 0 && is_infected(i), s.individuals)
        @test length(kids) > 0
        @test all(kid -> interval(s, kid) <= 5.0, kids)
        # Per case under a calendar, through the calendar-scaled wrapper.
        cal = Seasonal(amplitude = 0.5, peak_day = 0.0, period = 365.0)
        w = Infectiousness(
            Poisson(3.0); kernel = ind -> truncated(HazardKernel(t -> 2.0), 0.0, 5.0),
            calendar = cal
        )
        s2 = simulate(
            ModelSpec(BranchingProcess(w)); n_initial = 50, max_generations = 1,
            rng = StableRNG(1)
        )
        kids2 = filter(i -> i.parent_id != 0 && is_infected(i), s2.individuals)
        @test length(kids2) > 0
        @test all(kid -> interval(s2, kid) <= 5.0, kids2)
        # A late onset under the calendar: contacts start at the onset and
        # are not pushed past the short stretch just after it.
        late = Infectiousness(
            Poisson(3.0);
            kernel = ind -> truncated(HazardKernel(t -> 2.0; onset = 15.0), 0.0, 20.0),
            calendar = t -> 1.0
        )
        s3 = simulate(
            ModelSpec(BranchingProcess(late)); n_initial = 200, max_generations = 1,
            rng = StableRNG(1)
        )
        intervals = [interval(s3, kid) for kid in s3.individuals if kid.parent_id != 0 && is_infected(kid)]
        @test minimum(intervals) >= 15.0
        @test minimum(intervals) < 15.01
        # Conditional on falling before 20, exp(-2 * 0.06) of the mass is
        # gone by 15.06, so about 11% of intervals sit in [15, 15.06].
        @test isapprox(count(<=(15.06), intervals) / length(intervals), 0.113; atol = 0.03)
    end

    @testset "works as a bare contact_interval, not just a per-case kernel" begin
        # Both implied_R and the `offspring` form of transmission_timing call
        # `cdf`/`truncated` on `contact_interval` directly.
        rate = 0.4
        k = HazardKernel(t -> rate)
        e = Exponential(1 / rate)
        offspring = NegBin(3.0, 0.5)
        infectious_period = Gamma(5.0, 2.0)
        @test implied_R(offspring, k, infectious_period) ≈
            implied_R(offspring, e, infectious_period)

        timing = transmission_timing(
            k; offspring, latent_period = LogNormal(1.0, 0.3), infectious_period
        )
        model = ModelSpec(BranchingProcess(timing.window); progression = timing.progression)
        s = simulate(model; n_initial = 200, max_generations = 1, rng = StableRNG(4))
        kids = filter(i -> i.parent_id != 0 && is_infected(i), s.individuals)
        @test length(kids) > 0
    end

    @testset "a bare kernel (not a per-case function) still drives a simulation" begin
        # The `contacts` form leaves `contact_interval` as the window's own
        # kernel, rather than closing over it per case. This is the only
        # path that exercises a `HazardKernel` sitting there directly.
        k = HazardKernel(t -> 0.4)
        timing = transmission_timing(k; contacts = NegBin(3.0, 0.5))
        model = ModelSpec(BranchingProcess(timing.window))
        s = simulate(model; n_initial = 200, max_generations = 1, rng = StableRNG(1))
        kids = filter(i -> i.parent_id != 0 && is_infected(i), s.individuals)
        @test length(kids) > 0
    end

    @testset "truncation to a short infectious period still draws a contact time" begin
        # `truncated` picks `quantile` over `invlogccdf` once the truncation
        # probability isn't tiny; a short infectious period relative to the
        # kernel's rate forces that here.
        k = HazardKernel(t -> 2.0)
        timing = transmission_timing(
            k;
            offspring = NegBin(3.0, 0.5),
            latent_period = LogNormal(1.0, 0.3),
            infectious_period = Gamma(2.0, 0.3)
        )
        model = ModelSpec(BranchingProcess(timing.window); progression = timing.progression)
        s = simulate(model; n_initial = 200, max_generations = 1, rng = StableRNG(1))
        kids = filter(i -> i.parent_id != 0 && is_infected(i), s.individuals)
        @test length(kids) > 0
    end

    @testset "onset shifts where the hazard switches on" begin
        switch_on = 15.0
        k = HazardKernel(t -> 2.0; onset = switch_on)
        m = quantile(k, 0.5)
        @test isfinite(m)
        @test m ≈ switch_on + quantile(Exponential(0.5), 0.5)
    end

    @testset "a wrong onset raises an error" begin
        # A hazard still flat just past the declared onset looks just like one
        # that has permanently converged; given the wrong onset, construction
        # cannot tell them apart, so it rejects the hazard outright.
        switch_on = 15.0
        @test_throws ArgumentError HazardKernel(t -> t < switch_on ? 0.0 : 2.0; onset = 5.0)
    end

    @testset "a window that closes before the onset draws no contact" begin
        k = HazardKernel(t -> 2.0; onset = 10.0)
        tk = truncated(k, 0.0, 5.0)
        @test rand(StableRNG(1), tk) == Inf
        @test cdf(tk, 3.0) == 0.0
        @test cdf(tk, 5.0) == 0.0
        # The keyword spelling reaches the same truncation.
        @test rand(StableRNG(1), truncated(k; upper = 5.0)) == Inf
        @test truncated(k; lower = 0.0) === k
        @test_throws ArgumentError truncated(k; lower = 1.0)
        # The `offspring` form builds exactly this truncation for a case whose
        # infectious period ends before its kernel switches on; it must then
        # infect nobody rather than place every contact at the opening.
        timing = transmission_timing(
            ind -> HazardKernel(t -> 2.0; onset = ind.state[:switch_on]);
            offspring = Poisson(3.0), infectious_period = Gamma(2.0, 0.5)
        )
        attributes = (rng, ind) -> ind.state[:switch_on] = 10.0
        model = ModelSpec(
            BranchingProcess(timing.window); progression = timing.progression, attributes
        )
        s = simulate(model; n_initial = 100, max_generations = 1, rng = StableRNG(1))
        @test count(i -> i.parent_id != 0 && is_infected(i), s.individuals) == 0
    end

    @testset "a per-case trajectory times contacts through a hazard built from it" begin
        # Each case's own switch-on time, drawn as an ordinary attribute, is
        # given to the kernel as its onset: the window's own kernel reads it
        # back off the individual, so no contact can be timed before it.
        attributes = (rng, ind) -> ind.state[:switch_on] = rand(rng, Uniform(1.0, 6.0))
        kernel = ind -> HazardKernel(t -> 2.0; onset = ind.state[:switch_on])
        m = ModelSpec(
            BranchingProcess(Infectiousness(NegBin(3.0, 0.5); kernel));
            attributes
        )
        s = simulate(m; n_initial = 50, max_generations = 1, rng = StableRNG(1))
        kids = filter(i -> i.parent_id != 0 && is_infected(i), s.individuals)
        @test length(kids) > 0
        @test all(kids) do k
            parent = s.individuals[k.parent_id]
            k.infection_time - parent.infection_time >= parent.state[:switch_on]
        end
    end
end

@testset "transmission_timing" begin
    @testset "a plain generation time matches BranchingProcess(offspring, gt)" begin
        gt = LogNormal(1.6, 0.5)
        off = NegBin(2.0, 0.3)
        timing = transmission_timing(gt; offspring = off)
        @test isempty(timing.progression)
        @test timing.window.from == :infection
        @test timing.window.until == ()
        @test timing.window.kernel === gt
        @test timing.window.offspring === off

        s1 = simulate(BranchingProcess(off, gt); n_initial = 20, max_generations = 3, rng = StableRNG(1))
        s2 = simulate(BranchingProcess(timing.window); n_initial = 20, max_generations = 3, rng = StableRNG(1))
        @test s1.cumulative_cases == s2.cumulative_cases
    end

    @testset "needs exactly one of offspring or contacts" begin
        gt = Exponential(4.0)
        @test_throws ArgumentError transmission_timing(gt)
        @test_throws ArgumentError transmission_timing(
            gt; offspring = NegBin(2.0, 0.3), contacts = NegBin(2.0, 0.3)
        )
    end

    @testset "a latent and infectious period builds the matching progression and window" begin
        timing = transmission_timing(
            Exponential(4.0); contacts = NegBin(5.0, 0.3),
            latent_period = LogNormal(1.0, 0.3), infectious_period = Gamma(5.0, 2.0)
        )
        @test length(timing.progression) == 2
        @test timing.progression[1].state == :infectious
        @test timing.progression[1].from == :infection
        @test timing.progression[2].state == :recovered
        @test timing.progression[2].from == :infectious
        @test timing.progression[2].terminal
        @test timing.window.from == :infectious
        @test timing.window.until == (:recovered,)
    end

    @testset "an infectious period with no latent period opens the window at infection" begin
        timing = transmission_timing(
            Exponential(4.0); contacts = NegBin(5.0, 0.3), infectious_period = Gamma(5.0, 2.0)
        )
        @test length(timing.progression) == 1
        @test timing.progression[1].state == :recovered
        @test timing.progression[1].from == :infection
        @test timing.window.from == :infection
        @test timing.window.until == (:recovered,)
    end

    @testset "contacts are thinned by recovery, close to implied_R" begin
        latent = LogNormal(1.0, 0.3)
        infectious = Gamma(5.0, 2.0)
        ci = Exponential(4.0)
        contacts = NegBin(5.0, 0.3)
        timing = transmission_timing(
            ci; contacts, latent_period = latent, infectious_period = infectious
        )
        model = ModelSpec(BranchingProcess(timing.window); progression = timing.progression)
        n = 3000
        s = simulate(model; n_initial = n, max_generations = 1, max_cases = nothing, rng = StableRNG(2))
        kids = count(i -> i.parent_id != 0 && is_infected(i), s.individuals)
        @test isapprox(kids / n, implied_R(contacts, ci, infectious); rtol = 0.1)
        @test implied_R(contacts, ci, infectious) < mean(contacts)
    end

    @testset "offspring keeps R exact, timed before recovery" begin
        latent = LogNormal(1.0, 0.3)
        infectious = Gamma(5.0, 2.0)
        ci = Exponential(4.0)
        offspring = NegBin(5.0, 0.3)
        timing = transmission_timing(
            ci; offspring, latent_period = latent, infectious_period = infectious
        )
        model = ModelSpec(BranchingProcess(timing.window); progression = timing.progression)
        n = 3000
        s = simulate(model; n_initial = n, max_generations = 1, max_cases = nothing, rng = StableRNG(3))
        kids = filter(i -> i.parent_id != 0 && is_infected(i), s.individuals)
        @test isapprox(length(kids) / n, mean(offspring); rtol = 0.1)
        @test all(kids) do k
            parent = s.individuals[k.parent_id]
            k.infection_time <= parent.state[:recovered_time]
        end
    end
end

@testset "implied_R" begin
    @testset "matches the contact-interval cdf for a fixed infectious period" begin
        contacts = NegBin(3.0, 0.5)
        ci = Exponential(2.0)
        @test implied_R(contacts, ci, 5.0) ≈ mean(contacts) * cdf(ci, 5.0)
    end

    @testset "integrates over a distributed infectious period" begin
        contacts = NegBin(3.0, 0.5)
        ci = Exponential(2.0)
        infectious = Gamma(5.0, 2.0)
        # A fixed infectious period at the mean of the distribution is a rough
        # but directionally sound cross-check for the quadrature.
        @test implied_R(contacts, ci, infectious) ≈ implied_R(contacts, ci, mean(infectious)) atol = 0.5
    end
end

@testset "A deep ring leaves a case's settled state to its offspring" begin
    # Severe cases make three contacts and others one. Tracing that changes
    # nothing must leave every case with the count its own severity implies,
    # at any ring depth.
    progression = [
        Transition(:severe; from = :infection, delay = 0.0, probability = 0.5),
        Transition(:recovered; from = :infection, delay = 20.0),
    ]
    severe(ind) = isfinite(get(ind.state, :severe_time, Inf))
    m = BranchingProcess(
        Infectiousness((rng, ind) -> severe(ind) ? 3 : 1; kernel = Exponential(2.0))
    )
    for depth in (1, 2)
        ct = ContactTracing(
            TraceEveryone(), 1.0, Exponential(1000.0), FlagOnly(); depth = depth
        )
        wrong = 0
        for seed in 1:20
            s = simulate(
                ModelSpec(m; progression, interventions = [ct]);
                max_generations = 4, max_cases = nothing, rng = StableRNG(seed)
            )
            for ind in s.individuals
                is_infected(ind) && ind.generation < 4 || continue
                drawn = count(
                    c -> !isnan(s.individuals[c].infection_time), ind.secondary_case_ids
                )
                drawn == (severe(ind) ? 3 : 1) || (wrong += 1)
            end
        end
        @test wrong == 0
    end
end

@testset "A deep ring traces the offspring of an infected member that cannot seed" begin
    # An asymptomatic ring member is not eligible under `OnSymptomOnset`, so
    # its own contacts are reached only by the ring growing through it, with
    # the offspring a distribution, a function, or one window of each.
    attributes = clinical_presentation(
        incubation_period = Exponential(1.0), prob_asymptomatic = 0.5
    )
    ct = ContactTracing(
        OnSymptomOnset(), 1.0, Exponential(1000.0), FlagOnly(); depth = 2
    )
    windows = (
        (Infectiousness(Dirac(2); kernel = Exponential(2.0)),),
        (Infectiousness((rng, ind) -> 2; kernel = Exponential(2.0)),),
        (
            Infectiousness(Dirac(2); kernel = Exponential(2.0)),
            Infectiousness((rng, ind) -> 2; kernel = Exponential(2.0)),
        ),
    )
    for w in windows
        m = BranchingProcess(w...)
        reached = 0
        traced = 0
        for seed in 1:30
            s = simulate(
                ModelSpec(m; interventions = [ct], attributes);
                max_generations = 4, max_cases = nothing, rng = StableRNG(seed)
            )
            for ind in s.individuals
                is_infected(ind) && ind.parent_id != 0 || continue
                parent = s.individuals[ind.parent_id]
                parent.parent_id == 0 && continue
                grandparent = s.individuals[parent.parent_id]
                is_traced(parent) && is_asymptomatic(parent) &&
                    !is_asymptomatic(grandparent) || continue
                reached += 1
                is_traced(ind) && (traced += 1)
            end
        end
        @test reached > 0
        @test traced == reached
    end
end

@testset "A deep ring traces a member's contacts from the earlier of ring and trigger" begin
    # Whether the offspring is a distribution, whose contacts are drawn as soon
    # as the ring reaches a member, or a function, whose contacts are drawn
    # only once the member settles, each contact of a traced member is traced
    # from whichever came first: the ring reaching the member, or the member's
    # own trigger as an eligible case.
    attributes = clinical_presentation(
        incubation_period = Exponential(2.0), prob_asymptomatic = 0.0
    )
    iso = Isolation(onset_to_isolation_delay = Dirac(4.0), duration = Inf)
    ct = ContactTracing(
        probability = 1.0, isolation_to_trace_delay = Dirac(0.5),
        action = FlagOnly(), depth = 2
    )
    for offspring in (Dirac(2), (rng, ind) -> 2)
        m = ModelSpec(
            BranchingProcess(offspring, Exponential(3.0));
            interventions = [iso, ct], attributes
        )
        checked = 0
        wrong = 0
        for seed in 1:10
            s = simulate(m; max_generations = 4, max_cases = nothing, rng = StableRNG(seed))
            for c in s.individuals
                c.parent_id == 0 && continue
                b = s.individuals[c.parent_id]
                b.parent_id != 0 && is_traced(b) && is_infected(b) || continue
                haskey(c.state, :trace_time) || continue
                trigger = EpiBranch.trigger_time(ct.eligibility, b, c, s)
                expected = min(b.state[:trace_time], trigger) + 0.5
                checked += 1
                c.state[:trace_time] ≈ expected || (wrong += 1)
            end
        end
        @test checked > 0
        @test wrong == 0
    end
end
