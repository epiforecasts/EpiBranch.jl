using Test
using EpiHouseholds
using EpiBranch
using Distributions
using StableRNGs
using ForwardDiff
using DifferentiationInterface
import Mooncake
using ADTypes: AutoMooncake

# The within-household natural history, composed onto the process with a
# ModelSpec. A recovery removal (SIR) unless a latent step is prepended (SEIR).
_sir(ip) = [Transition(:recovered; from = :infection, delay = ip, terminal = true)]

# Sample variance, for the standard error of a mean.
_var(v) = (m = sum(v) / length(v); sum((x - m)^2 for x in v) / (length(v) - 1))

# A per-contact risk written from outside the package, blocking every
# transmission it is asked about.
struct BlockEverything <: EpiBranch.AbstractIntervention end
function EpiBranch.competing_risk(::BlockEverything, parent, contact, state)
    Risk(block_probability = 1.0)
end

@testset "EpiHouseholds.jl" begin
    @testset "construction" begin
        m = HouseholdProcess([3, 4, 2], Exponential(3.0))
        @test m isa HouseholdProcess
        @test household_sizes(m) == [3, 4, 2]
        @test length(m.household_of) == 9
        @test m.members[1] == [1, 2, 3]
        @test m.members[3] == [8, 9]
        @test m.from === nothing              # window start derived from the progression
        @test m.external_hazard == 0.0

        # with no latent transition the window opens at :infection
        @test EpiBranch._resolve_infectious_from(m.from, _sir(6.0)) === :infection

        # a latent transition anchors the window at :infectious
        seir = [Transition(:infectious; from = :infection, delay = LogNormal(1.0, 0.4)),
            Transition(:recovered; from = :infectious, delay = Gamma(6, 1),
                terminal = true)]
        @test EpiBranch._resolve_infectious_from(nothing, seir) === :infectious

        # a scalar and a distribution external hazard are both accepted
        @test HouseholdProcess([2], Exponential(1.0); external_hazard = 0.05) isa
              HouseholdProcess
        @test HouseholdProcess(
            [2], Exponential(1.0); external_hazard = Exponential(20.0)) isa
              HouseholdProcess

        @test_throws ArgumentError HouseholdProcess([0, 2], Exponential(1.0))
        @test_throws ArgumentError HouseholdProcess(
            [2], Exponential(1.0); external_hazard = -1.0)
    end

    @testset "simulation seeds one index per household and spreads within it" begin
        m = ModelSpec(HouseholdProcess(fill(4, 100), Weibull(1.3, 2.0));
            progression = _sir(6.0))
        state = simulate(m; rng = StableRNG(1))
        df = linelist(state)
        @test count(df.index) == 100              # one index per household
        @test 100 <= size(df, 1) <= 400           # indexes plus within-household spread
        @test issubset((:date_infection, :household, :index), propertynames(df))
    end

    @testset "the timeline is stamped through the progression (line-list columns)" begin
        # a latent period and an infectious period give onset and recovery dates
        m = ModelSpec(HouseholdProcess(fill(4, 50), Exponential(2.0));
            progression = [
                Transition(:infectious; from = :infection, delay = LogNormal(1.0, 0.3)),
                Transition(:recovered; from = :infectious, delay = Gamma(6, 1),
                    terminal = true)])
        df = linelist(simulate(m; rng = StableRNG(2)))
        @test :date_infectious in propertynames(df)   # :infectious_time → date_infectious
        @test :date_recovered in propertynames(df)     # the removal transition
    end

    @testset "a fixed seed reproduces a pinned outbreak" begin
        # The race draws from the RNG stream in settling order, so this test pins
        # the order in which each household's members settle.
        st = simulate(
            ModelSpec(HouseholdProcess([3, 4], Exponential(2.0));
                progression = _sir(4.0));
            rng = StableRNG(7))
        @test [ind.infection_time for ind in st.individuals] ≈
              [1.1701609052240476, 0.0, 1.2332841100671945, 0.0, 0.3297017722468899,
            1.4196539602204872, 1.0835818339525922]
        @test [ind.parent_id for ind in st.individuals] == [2, 0, 2, 0, 4, 7, 5]
    end

    @testset "no within-household spread when the kernel is far out of the period" begin
        # contact intervals almost never fall within a tiny infectious period,
        # so only the index cases are infected.
        m = ModelSpec(HouseholdProcess(fill(5, 200), Exponential(50.0));
            progression = _sir(0.001))
        df = linelist(simulate(m; rng = StableRNG(3)))
        @test 200 <= size(df, 1) <= 205
    end

    @testset "final size grows with transmissibility" begin
        sizes = fill(5, 300)
        low = ModelSpec(HouseholdProcess(sizes, Exponential(20.0)); progression = _sir(6.0))
        high = ModelSpec(HouseholdProcess(sizes, Exponential(2.0)); progression = _sir(6.0))
        n_low = size(linelist(simulate(low; rng = StableRNG(4))), 1)
        n_high = size(linelist(simulate(high; rng = StableRNG(4))), 1)
        @test n_high > n_low
    end

    @testset "Isolation intervention reduces household spread" begin
        # The Isolation *intervention* runs on the continuous-time household
        # path: its resolve_individual! fires in the Sellke race and its
        # isolation time closes the infectious window, cutting secondary cases.
        # Onset comes from a progression transition, anchored on infection time.
        sizes = fill(6, 300)
        prog = [
            Transition(:onset; from = :infection, delay = 0.3),
            Transition(:recovered; from = :infection,
                delay = Exponential(6.0), terminal = true)]
        base = ModelSpec(HouseholdProcess(sizes, Exponential(1.0)); progression = prog)
        iso = ModelSpec(HouseholdProcess(sizes, Exponential(1.0)); progression = prog,
            interventions = [Isolation(onset_to_isolation_delay = Exponential(0.2))])

        base_cases = sum(simulate(base; rng = StableRNG(s)).cumulative_cases for s in 1:10)
        iso_cases = sum(simulate(iso; rng = StableRNG(s)).cumulative_cases for s in 1:10)
        @test iso_cases < base_cases
    end

    @testset "per-individual susceptibility and infectiousness apply" begin
        # The race puts each contact it proposes to the composed competing risks,
        # and a blocked contact leaves the pair meeting, so a susceptibility of s
        # scales each pair's hazard by s. Households of six with a mean contact
        # interval of six days and a two-day infectious period are far enough
        # from saturation for that to show in the outbreak size.
        sizes = fill(6, 200)
        build(attrs) = ModelSpec(HouseholdProcess(sizes, Exponential(6.0));
            progression = _sir(2.0), attributes = attrs)
        meansize(attrs) = sum(simulate(build(attrs);
                                  rng = StableRNG(s)).cumulative_cases for s in 1:10) / 10

        full = meansize(transmission_traits(susceptibility = 1.0))
        half = meansize(transmission_traits(susceptibility = 0.5))
        @test half < full
        @test meansize(transmission_traits(susceptibility = 0.2)) < half

        # Susceptibility 0 blocks every contact, however many the pair makes:
        # only the one index case per household is ever infected.
        blocked = simulate(build(transmission_traits(susceptibility = 0.0));
            rng = StableRNG(1))
        @test blocked.cumulative_cases == length(sizes)

        # Infectiousness acts on the other side of the same pair.
        @test meansize(transmission_traits(infectiousness = 0.5)) < full
        silent = simulate(build(transmission_traits(infectiousness = 0.0));
            rng = StableRNG(1))
        @test silent.cumulative_cases == length(sizes)
    end

    @testset "a blocked contact thins the hazard, as in the pool" begin
        # A clique whose pairs meet at rate 1 is the same process as a fixed-size
        # pool of the same size at β = N, because a susceptible there feels
        # β/N = 1 per infective. A per-contact block thins the hazard on both, so
        # the two agree with a susceptibility as well as without one.
        prog = [Transition(:recovered; from = :infection, delay = 2.0, terminal = true)]
        function race_sizes(size, sus)
            spec = ModelSpec(HouseholdProcess(fill(size, 400), Exponential(1.0));
                progression = prog,
                attributes = transmission_traits(susceptibility = sus))
            sizes = Float64[]
            for s in 1:5
                st = simulate(spec; rng = StableRNG(s))
                counts = zeros(Int, 400)
                for ind in st.individuals
                    is_infected(ind) && (counts[(ind.id - 1) ÷ size + 1] += 1)
                end
                append!(sizes, counts)
            end
            return sizes
        end
        function pool_sizes(size, sus)
            spec = ModelSpec(
                HomogeneousProcess(; transmission_rate = float(size),
                    population_size = size);
                progression = prog,
                attributes = transmission_traits(susceptibility = sus))
            return [Float64(simulate(spec; rng = StableRNG(s), n_initial = 1).cumulative_cases)
                    for s in 1:2000]
        end
        # A two-person clique at susceptibility 0.5: the secondary case is
        # infected with probability 1 - exp(-0.5 * 2), where blocking the
        # transmission probability instead would give half of 1 - exp(-2).
        pair = race_sizes(2, 0.5)
        @test isapprox(sum(pair) / length(pair) - 1, 1 - exp(-1.0); atol = 0.03)
        @test sum(pair) / length(pair) - 1 > 0.5 * (1 - exp(-2.0)) + 0.05

        for (size, sus) in ((2, 1.0), (2, 0.5), (5, 0.5))
            race = race_sizes(size, sus)
            pool = pool_sizes(size, sus)
            # Three standard errors of the difference, and never fewer than 0.05
            # cases, so a real disagreement of the kind a probability-thinning
            # race showed (4.07 against 4.82 for households of five) fails.
            se = sqrt(_var(race) / length(race) + _var(pool) / length(pool))
            @test abs(sum(race) / length(race) - sum(pool) / length(pool)) <
                  max(3 * se, 0.05)
        end
    end

    @testset "onset is measured from each case's own infection time" begin
        # Members are created, and their incubation periods drawn, before the
        # race sets their infection times. Isolation depends on onset, so onset
        # must be counted from the time each case was infected.
        clinical = clinical_presentation(incubation_period = LogNormal(1.0, 0.3))
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0),
            test_sensitivity = 1.0)
        m = ModelSpec(HouseholdProcess(fill(6, 50), Exponential(3.0));
            progression = _sir(10.0), attributes = clinical, interventions = [iso])
        state = simulate(m; rng = StableRNG(3))
        secondary = [ind
                     for ind in state.individuals
                     if is_infected(ind) && ind.parent_id != 0]
        @test !isempty(secondary)
        @test all(onset_time(ind) >= ind.infection_time for ind in secondary)
        @test all(onset_time(ind) - ind.infection_time ≈ ind.state[:incubation_period]
        for ind in secondary)
        @test all(isolation_time(ind) >= onset_time(ind) for ind in secondary)
    end

    @testset "FlagOnly tracing isolates traced members through the traced pathway" begin
        # With no test-positive cases, only the traced pathway can isolate. A
        # member is traced before the race has settled its onset, so this
        # checks that the trace is still recorded and turned into isolation.
        clinical = clinical_presentation(incubation_period = LogNormal(1.0, 0.3))
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0),
            test_sensitivity = 0.0)
        flag = ContactTracing(OnSymptomOnset(), 1.0, Exponential(0.5), FlagOnly())
        m = ModelSpec(HouseholdProcess(fill(6, 50), Exponential(3.0));
            progression = _sir(10.0), attributes = clinical, interventions = [iso, flag])
        isolated = [ind
                    for s in 1:20
                    for ind in simulate(m; rng = StableRNG(s)).individuals
                    if is_infected(ind) && isfinite(isolation_time(ind))]
        @test !isempty(isolated)
        @test all(is_traced, isolated)
        @test all(isolation_time(ind) >= onset_time(ind) for ind in isolated)
    end

    @testset "external force of infection introduces community cases" begin
        m = ModelSpec(
            HouseholdProcess(fill(4, 300), Exponential(3.0);
                external_hazard = 0.05, obs_end = 30.0);
            progression = _sir(6.0))
        df = linelist(simulate(m; rng = StableRNG(5)))
        @test count(df.index) >= 1                 # community introductions happened
        @test size(df, 1) > count(df.index)        # plus within-household spread
        @test count(df.index) != length(m.process.members) # not the one-index fallback

        # An introduction is put to the risks like any other contact: a member
        # with no susceptibility is never introduced from the community, and
        # neither is anyone while a risk blocks every transmission.
        build(attrs, ivs) = ModelSpec(
            HouseholdProcess(fill(4, 300), Exponential(3.0);
                external_hazard = 0.05, obs_end = 30.0);
            progression = _sir(6.0), attributes = attrs, interventions = ivs)
        @test !any(is_infected,
            simulate(
                build(transmission_traits(susceptibility = 0.0),
                    AbstractIntervention[]);
                rng = StableRNG(5)).individuals)
        @test !any(is_infected,
            simulate(build(EpiBranch.NoAttributes(), [BlockEverything()]);
                rng = StableRNG(5)).individuals)
    end

    @testset "simulate → loglikelihood round trip recovers the kernel" begin
        # the Sellke construction is the generative model the pairwise likelihood
        # assumes, so the simulated infection layer recovers the kernel scale.
        true_scale = 4.0
        L = 6.0
        m = ModelSpec(HouseholdProcess(fill(4, 1500), Exponential(true_scale));
            progression = _sir(L))
        state = simulate(m; rng = StableRNG(20260615))
        data = household_infections(state, m)
        @test count(data.is_index) == 1500            # one index per household

        layout = compile_household_pairs(data)
        ll(s) = pairwise_surv_loglik(Exponential(s), data, layout)
        @test ll(true_scale) > ll(true_scale / 2)
        @test ll(true_scale) > ll(true_scale * 2)
        grid = 2.0:0.5:6.0
        @test abs(grid[argmax([ll(s) for s in grid])] - true_scale) <= 1.0

        # the dispatched loglikelihood routes through pairwise_surv_loglik
        @test loglikelihood(data, m) ≈ ll(true_scale)
    end

    @testset "isolation ends the infectious window in the infection layer" begin
        # the race closes a case's window when it is isolated and the data must
        # too; otherwise the likelihood sees cases infectious after isolation and
        # overestimates the kernel scale
        clinical = clinical_presentation(incubation_period = LogNormal(1.0, 0.3),
            prob_asymptomatic = 0.0)
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0),
            test_sensitivity = 1.0)
        m = ModelSpec(HouseholdProcess(fill(4, 1500), Exponential(4.0));
            progression = _sir(8.0), interventions = [iso], attributes = clinical)
        state = simulate(m; rng = StableRNG(201))
        data = household_infections(state, m)

        infected = findall(!isnan, data.infection_time)
        expected = [min(data.infection_time[i] + 8.0,
                        EpiBranch.isolation_time(state.individuals[i])) for i in infected]
        @test data.removal_time[infected] == expected
        @test count(data.removal_time[infected] .< data.infection_time[infected] .+ 8.0) >
              length(infected) / 2

        layout = compile_household_pairs(data)
        f(θ) = pairwise_surv_loglik(Exponential(exp(θ)), data, layout)
        d2(z) = ForwardDiff.derivative(y -> ForwardDiff.derivative(f, y), z)
        θhat = log(4.0)
        for _ in 1:20
            θhat -= ForwardDiff.derivative(f, θhat) / d2(θhat)
        end
        se = 1 / sqrt(-d2(θhat))
        @test abs(θhat - log(4.0)) < 3 * se
    end

    @testset "external (community) term: round trip recovers the kernel" begin
        # with a community hazard, indexes emerge from it and the within-household
        # kernel scale is still recovered through the same likelihood.
        true_scale = 3.0
        L = 6.0
        Tobs = 30.0
        m = ModelSpec(
            HouseholdProcess(fill(4, 1500), Exponential(true_scale);
                external_hazard = 0.05, obs_end = Tobs);
            progression = _sir(L))
        state = simulate(m; rng = StableRNG(7))
        data = household_infections(state, m)
        @test count(data.is_index) >= 1
        @test isfinite(loglikelihood(data, m))          # dispatched form with external

        layout = compile_household_pairs(data; external = true)
        ll(s) = pairwise_surv_loglik(Exponential(s), data, layout; external_hazard = 0.05)
        grid = 1.5:0.5:5.0
        @test abs(grid[argmax([ll(s) for s in grid])] - true_scale) <= 1.5
    end

    @testset "community introductions stop at obs_end and household spread goes on" begin
        # with a short obs_end most infections come later, within households; the
        # likelihood must give them no community hazard and keep uninfected
        # members exposed over their household-mates' whole windows
        Tobs = 2.0
        m = ModelSpec(
            HouseholdProcess(fill(6, 1500), Exponential(10.0);
                external_hazard = 0.1, obs_end = Tobs);
            progression = _sir(12.0))
        data = household_infections(simulate(m; rng = StableRNG(71)), m)
        inf = filter(!isnan, data.infection_time)
        @test count(>(Tobs), inf) > length(inf) / 2

        layout = compile_household_pairs(data; external = true)
        @test loglikelihood(data, m) ≈
              pairwise_surv_loglik(Exponential(10.0), data, layout;
            external_hazard = 0.1)
        g(θ) = pairwise_surv_loglik(Exponential(exp(θ[1])), data, layout;
            external_hazard = exp(θ[2]))
        θ = [log(10.0), log(0.1)]
        θhat = copy(θ)
        for _ in 1:20
            θhat -= ForwardDiff.hessian(g, θhat) \ ForwardDiff.gradient(g, θhat)
        end
        Σ = inv(-ForwardDiff.hessian(g, θhat))
        se = sqrt.([Σ[1, 1], Σ[2, 2]])
        @test all(abs.(θhat - θ) .< 3 .* se)
    end

    @testset "an outbreak still going at the end of follow-up" begin
        m = ModelSpec(
            HouseholdProcess(fill(6, 1500), Exponential(10.0);
                external_hazard = 0.1, obs_end = 2.0);
            progression = _sir(12.0))
        state = simulate(m; rng = StableRNG(71))
        tf = 6.0
        full = household_infections(state, m)
        late = .!(full.infection_time .<= tf)
        nan_late(x) = [l ? NaN : v for (v, l) in zip(x, late)]
        ongoing = HouseholdInfections(full.household_of, nan_late(full.infection_time),
            nan_late(full.infectious_time),
            [l ? NaN : (r > tf ? Inf : r) for (r, l) in zip(full.removal_time, late)],
            full.is_index .& .!late; obs_end = 2.0, followup_end = tf)
        @test any(isinf, ongoing.removal_time)
        @test count(!isnan, ongoing.infection_time) < count(!isnan, full.infection_time)

        read = household_infections(state, m; followup_end = tf)
        @test read.followup_end == tf
        k = Exponential(10.0)
        v = loglikelihood(ongoing, m)
        @test isfinite(v)
        @test v ≈ loglikelihood(read, m)
        layout = compile_household_pairs(ongoing; external = true)
        g(θ) = pairwise_surv_loglik(Exponential(exp(θ[1])), ongoing, layout;
            external_hazard = exp(θ[2]))
        θ = [log(10.0), log(0.1)]
        @test g(θ) ≈ v
        grad = ForwardDiff.gradient(g, θ)
        h = 1e-4
        fd = [(g(θ .+ h .* e) - g(θ .- h .* e)) / 2h for e in ([1.0, 0.0], [0.0, 1.0])]
        @test grad ≈ fd rtol = 1e-5
    end

    @testset "inference-friendly likelihood: kernel varies over a fixed infection layer" begin
        # the form a household @model evaluates each iteration: the kernel carries
        # the fitted parameter, the infection layer is the augmented latent state.
        m = ModelSpec(HouseholdProcess(fill(4, 400), Exponential(3.0));
            progression = _sir(6.0))
        data = household_infections(simulate(m; rng = StableRNG(8)), m)

        f(logβ) = pairwise_surv_loglik(Exponential(1 / exp(logβ)), data)
        @test f(log(1 / 3)) ≈ loglikelihood(data, m)            # matches the model-dispatch sugar
        g = ForwardDiff.derivative(f, log(1 / 3))               # gradient for HMC
        @test isfinite(g)
        grid = 2.0:0.5:6.0
        @test abs(grid[argmax([f(log(1 / s)) for s in grid])] - 3.0) <= 1.5

        # composes with a progression observation term (onset = infection + incubation)
        incubation = LogNormal(1.0, 0.3)
        obs = findall(!isnan, data.infection_time)
        onset = copy(data.infection_time)
        onset[obs] .+= rand(StableRNG(9), incubation, length(obs))
        joint(logβ) = f(logβ) + sum(logpdf(incubation, onset[i] - data.infection_time[i])
        for i in obs)
        @test isfinite(joint(log(1 / 3)))
    end

    @testset "the model's observation is applied in simulate" begin
        # an observation model composed onto the process reports cases through the
        # shared observation protocol, like core simulate.
        obs = PerCaseObservation(; detection_prob = 0.5, delay = Exponential(2.0))
        m = ModelSpec(HouseholdProcess(fill(4, 200), Exponential(3.0));
            progression = _sir(6.0), observation = obs)
        df = linelist(simulate(m; rng = StableRNG(11)))
        @test :reported in propertynames(df)            # observation ran
        @test 0 < count(df.reported) < size(df, 1)       # ~half detected, not all
    end

    @testset "a reused layout matches a freshly compiled one (shared kernel)" begin
        # the two-argument form is the three-argument one with a layout compiled
        # on the spot, and this is not an independent check of the density. That
        # cross-check against hand-built counting-process rows lives in the root
        # suite, in "evaluation matches hand-built counting-process rows". This
        # test checks that evaluating a layout leaves it unchanged: one object
        # reused across a grid of kernel scales keeps agreeing with a fresh one,
        # which inference relies on.
        m = ModelSpec(HouseholdProcess(fill(4, 500), Exponential(3.0));
            progression = _sir(6.0))
        data = household_infections(simulate(m; rng = StableRNG(101)), m)
        layout = compile_household_pairs(data)

        @test layout isa HouseholdPairsLayout
        @test !layout.external
        @test length(layout) == length(layout.sus)
        @test length(layout) > 0                          # there is real spread to score

        for s in 1.5:0.5:6.0
            @test pairwise_surv_loglik(Exponential(s), data, layout) ≈
                  pairwise_surv_loglik(Exponential(s), data)
        end

        # the single-argument constructor derives the at-risk mask from the data,
        # and must land on the same layout as passing that mask explicitly
        layout1 = compile_household_pairs(data.household_of, data.is_index,
            .!isnan.(data.infection_time))
        @test length(layout1) == length(layout)
        @test pairwise_surv_loglik(Exponential(3.0), data, layout1) ≈
              pairwise_surv_loglik(Exponential(3.0), data, layout)
    end

    @testset "a reused layout matches a freshly compiled one (community hazard)" begin
        # with a community term every susceptible also carries an external row;
        # the layout must be built with external=true, and reusing it across
        # kernel scales must keep agreeing with a layout compiled per call.
        Tobs = 30.0
        m = ModelSpec(
            HouseholdProcess(fill(4, 500), Exponential(3.0);
                external_hazard = 0.05, obs_end = Tobs);
            progression = _sir(6.0))
        state = simulate(m; rng = StableRNG(102))
        data = household_infections(state, m)
        layout = compile_household_pairs(data; external = true)

        @test layout.external
        for s in 1.5:0.5:5.0
            @test pairwise_surv_loglik(Exponential(s), data, layout;
                external_hazard = 0.05) ≈
                  pairwise_surv_loglik(Exponential(s), data; external_hazard = 0.05)
        end

        # a distribution-valued community hazard routes the same way
        @test pairwise_surv_loglik(Exponential(3.0), data, layout;
            external_hazard = Exponential(20.0)) ≈
              pairwise_surv_loglik(Exponential(3.0), data;
            external_hazard = Exponential(20.0))
    end

    @testset "compiled pair layout: covariate (per-pair) kernel" begin
        # a two-argument (infector, susceptible) -> Distribution kernel is
        # resolved per row on both paths, and a reused layout agrees with one
        # compiled per call.
        m = ModelSpec(HouseholdProcess(fill(4, 300), Exponential(3.0));
            progression = _sir(6.0))
        data = household_infections(simulate(m; rng = StableRNG(103)), m)
        layout = compile_household_pairs(data)

        # a mild dependence on the pair ids exercises the routing, not the physics
        kern(i, j) = Exponential(3.0 + 0.01 * (i + j))
        @test pairwise_surv_loglik(kern, data, layout) ≈
              pairwise_surv_loglik(kern, data)
    end

    @testset "covariate kernel: simulate → likelihood round trip recovers both scales" begin
        # Host id sets the role: in each household of four, the first two members
        # are adults and the last two children. An adult infects a household-mate
        # faster than a child does, so the kernel depends on the infector's role.
        # The simulator and both likelihood forms call
        # `kernel(infector, susceptible)`. Fitting checks that all three agree on
        # that order, and the final sizes below check which order the simulator
        # uses. With `by_infector = false` the kernel reads the role from the
        # susceptible, which is what a reversed order would fit. The flag keeps a
        # single closure type, so both fits share compiled code.
        is_adult(i) = (i - 1) % 4 < 2
        function kernel(adult_scale, child_scale; by_infector = true)
            return (infector, susceptible) -> Exponential(
                is_adult(by_infector ? infector : susceptible) ?
                adult_scale : child_scale)
        end
        truth = [3.0, 12.0]
        m = ModelSpec(HouseholdProcess(fill(4, 600), kernel(truth...));
            progression = _sir(6.0))
        data = household_infections(simulate(m; rng = StableRNG(230)), m)
        layout = compile_household_pairs(data)

        # The simulator applies the infector's role: households whose index case is
        # an adult, the faster infector, end with more cases than those whose
        # index case is a child. A simulator reading the susceptible's role would
        # reverse this.
        final_size(adult_index) = mean(
            count(i -> !isnan(data.infection_time[i]), mem)
        for mem in (findall(==(h), data.household_of) for h in 1:600)
        if is_adult(only(filter(i -> data.is_index[i], mem))) == adult_index)
        @test final_size(true) > final_size(false) + 0.3

        ll(θ; by_infector = true) = pairwise_surv_loglik(
            kernel(exp.(θ)...; by_infector), data, layout)
        # a reused layout and one compiled per call agree for the covariate
        # kernel, and `loglikelihood` on the model passes its own kernel the
        # same way
        for θ in (log.(truth), log.([2.0, 5.0]), log.([6.0, 3.0]))
            @test ll(θ) ≈ pairwise_surv_loglik(kernel(exp.(θ)...), data)
        end
        @test loglikelihood(data, m) ≈ ll(log.(truth))

        # Newton maximisation on the log scales, from a start that does not
        # distinguish the roles
        function newton(f, θ)
            for _ in 1:8
                θ = θ - ForwardDiff.hessian(f, θ) \ ForwardDiff.gradient(f, θ)
            end
            return θ
        end
        θ̂ = newton(ll, log.([4.0, 4.0]))
        @test all(abs.(ForwardDiff.gradient(ll, θ̂)) .< 1e-6)

        # With about 2,000 cases the observed information gives standard errors of
        # roughly 0.03 (adult) and 0.06 (child) on the log scales. Both estimates
        # must lie within three standard errors of the truth.
        info = -ForwardDiff.hessian(ll, θ̂)
        se = sqrt.([inv(info)[k, k] for k in 1:2])
        @test all(se .< 0.1)
        @test all(abs.(θ̂ .- log.(truth)) .< 3 .* se)

        # the data distinguish the roles: a kernel keyed on the susceptible's role
        # fits far worse even at its own optimum, so the recovery above depends on
        # the id order
        swapped(θ) = ll(θ; by_infector = false)
        @test ll(θ̂) > swapped(newton(swapped, log.([4.0, 4.0]))) + 10
    end

    @testset "a reused layout propagates AD duals like a freshly compiled one" begin
        # the fast path exists to be differentiated in the kernel parameters
        # (its whole reason for being reused across gradient evaluations). The
        # fitted parameter rides the kernel, not the data, so the layout must
        # carry the AD duals through both the cumulative-hazard pass and the
        # per-susceptible log-sum-exp, and must still do so after being
        # evaluated. Checked in all three kernel modes.
        m = ModelSpec(HouseholdProcess(fill(4, 300), Exponential(3.0));
            progression = _sir(6.0))
        data = household_infections(simulate(m; rng = StableRNG(104)), m)
        layout = compile_household_pairs(data)

        f_fast(logβ) = pairwise_surv_loglik(Exponential(1 / exp(logβ)), data, layout)
        f_dyn(logβ) = pairwise_surv_loglik(Exponential(1 / exp(logβ)), data)
        @test f_fast(log(1 / 3)) ≈ f_dyn(log(1 / 3))

        g_fast = ForwardDiff.derivative(f_fast, log(1 / 3))
        g_dyn = ForwardDiff.derivative(f_dyn, log(1 / 3))
        @test isfinite(g_fast)
        @test g_fast ≈ g_dyn

        # covariate (per-pair) kernel: the parameter still rides the kernel, now
        # resolved per pair — the accumulator must promote to hold it.
        kern(β) = (i, j) -> Exponential(1 / (exp(β) * (1 + 0.001 * (i + j))))
        h_fast(β) = pairwise_surv_loglik(kern(β), data, layout)
        h_dyn(β) = pairwise_surv_loglik(kern(β), data)
        @test ForwardDiff.derivative(h_fast, log(1 / 3)) ≈
              ForwardDiff.derivative(h_dyn, log(1 / 3))

        # external mode: differentiate the within-household kernel while a fixed
        # community hazard also contributes rows.
        Tobs = 30.0
        me = ModelSpec(
            HouseholdProcess(fill(4, 300), Exponential(3.0);
                external_hazard = 0.05, obs_end = Tobs);
            progression = _sir(6.0))
        de = household_infections(simulate(me; rng = StableRNG(106)), me)
        le = compile_household_pairs(de; external = true)
        e_fast(logβ) = pairwise_surv_loglik(Exponential(1 / exp(logβ)), de, le;
            external_hazard = 0.05)
        e_dyn(logβ) = pairwise_surv_loglik(Exponential(1 / exp(logβ)), de;
            external_hazard = 0.05)
        @test e_fast(log(1 / 3)) ≈ e_dyn(log(1 / 3))
        @test ForwardDiff.derivative(e_fast, log(1 / 3)) ≈
              ForwardDiff.derivative(e_dyn, log(1 / 3))
    end

    @testset "compiled pair layout: hand-built multi-infector household" begin
        # one household, member 3 infected late with two eligible infectors
        # (members 1 and 2) — exercises the per-susceptible log-sum-exp over more
        # than one event row, plus a single-infector susceptible and a conditioned
        # index case.
        data = HouseholdInfections([1, 1, 1], [0.0, 0.5, 2.0], [0.0, 0.5, 2.0],
            [Inf, Inf, Inf], [true, false, false])
        layout = compile_household_pairs(data)

        # the layout enumerates every ordered (susceptible, infector) structural
        # pair among the non-index members: 2←{1,3} and 3←{1,2}, i.e. four rows —
        # even pairs whose timing later contributes nothing are kept, and pruned
        # on the fly at evaluation. The index (member 1) is conditioned on.
        @test length(layout) == 4
        @test 3 in layout.sus_unique && 2 in layout.sus_unique
        @test !(1 in layout.sus_unique)

        # and the reused layout keeps agreeing with one compiled per call
        for s in 1.0:1.0:5.0
            @test pairwise_surv_loglik(Exponential(s), data, layout) ≈
                  pairwise_surv_loglik(Exponential(s), data)
        end
    end

    @testset "compiled pair layout: edge cases" begin
        # empty population → empty layout, zero log-likelihood, consistent length
        empty = HouseholdInfections(Int[], Float64[], Float64[], Float64[], Bool[])
        elayout = compile_household_pairs(empty)
        @test length(elayout) == 0
        @test pairwise_surv_loglik(Exponential(3.0), empty, elayout) == 0.0
        @test compile_household_pairs(Int[], Bool[], Bool[]) isa HouseholdPairsLayout

        # a household where the sole housemate escapes: the index recovers at
        # t=3 and member 2 is never infected. There is still one structural row
        # (member 2 at risk from the index), whose only contribution is the
        # escaped cumulative hazard, which is finite and the same however the
        # layout was obtained.
        lone = HouseholdInfections([1, 1], [0.0, NaN], [0.0, NaN], [3.0, Inf],
            [true, false])
        llayout = compile_household_pairs(lone)
        @test length(llayout) == 1
        ll_lone = pairwise_surv_loglik(Exponential(3.0), lone, llayout)
        @test isfinite(ll_lone)
        @test ll_lone < 0                                 # pure escaped hazard
        @test ll_lone ≈ pairwise_surv_loglik(Exponential(3.0), lone)

        # calling the external-built layout without an external hazard (and vice
        # versa) is a mismatch and must raise
        m = ModelSpec(HouseholdProcess(fill(4, 50), Exponential(3.0));
            progression = _sir(6.0))
        data = household_infections(simulate(m; rng = StableRNG(105)), m)
        ext_layout = compile_household_pairs(data; external = true)
        int_layout = compile_household_pairs(data; external = false)
        @test_throws ArgumentError pairwise_surv_loglik(Exponential(3.0), data,
            ext_layout)                                   # no external_hazard given
        @test_throws ArgumentError pairwise_surv_loglik(Exponential(3.0), data,
            int_layout; external_hazard = 0.05)

        # mismatched input lengths are rejected at compile time
        @test_throws ArgumentError compile_household_pairs([1, 1], [true],
            [true, false])
        @test_throws ArgumentError HouseholdInfections([1, 1], [0.0], [0.0], [1.0],
            [true])
    end

    @testset "simulated index cases at time 0 scored with a community hazard" begin
        # index cases simulated at 0 without a community hazard, scored with one
        m = ModelSpec(HouseholdProcess(fill(4, 300), Exponential(3.0));
            progression = _sir(5.0))
        sim = household_infections(simulate(m; rng = StableRNG(1)), m)
        d = HouseholdInfections(sim.household_of, sim.infection_time,
            sim.infectious_time, sim.removal_time, sim.is_index; obs_end = 20.0)
        @test count(==(0.0), filter(!isnan, d.infection_time)) == 300
        ld = compile_household_pairs(d; external = true)
        for α in (0.001, 0.01, 0.1, 1.0)
            @test pairwise_surv_loglik(Exponential(3.0), d, ld; external_hazard = α) ≈
                  pairwise_surv_loglik(Exponential(3.0), d; external_hazard = α)
        end
    end

    @testset "simulated infection layers never have zero density" begin
        clinical = clinical_presentation(incubation_period = LogNormal(1.0, 0.3),
            prob_asymptomatic = 0.0)
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0),
            test_sensitivity = 1.0)
        latent = [Transition(:infectious; from = :infection, delay = LogNormal(0.3, 0.3)),
            Transition(:recovered; from = :infectious, delay = 5.0, terminal = true)]
        for seed in 1:4, (ext, Tobs) in ((0.0, Inf), (0.03, 10.0)),
            interventions in ([], [iso])
            m = ModelSpec(
                HouseholdProcess(rand(StableRNG(seed), 1:6, 200), Weibull(1.5, 4.0);
                    external_hazard = ext, obs_end = Tobs);
                progression = latent, interventions, attributes = clinical)
            d = household_infections(simulate(m; n_initial = 2, rng = StableRNG(seed)), m)
            layout = compile_household_pairs(d; external = ext > 0)
            @test isfinite(loglikelihood(d, m))
            @test isfinite(pairwise_surv_loglik(Weibull(1.5, 4.0), d, layout;
                external_hazard = ext))
        end
    end

    @testset "compiled pair layout: inference workflow (compile once, reuse)" begin
        # the documented workflow: with the household structure fixed, the layout
        # is compiled once and reused across every gradient evaluation of the fit.
        # Recovering the kernel scale by Newton MLE with the same layout at every
        # step must land on the same optimum as compiling a layout per call, and
        # near the truth.
        true_scale = 4.0
        m = ModelSpec(HouseholdProcess(fill(4, 800), Exponential(true_scale));
            progression = _sir(6.0))
        data = household_infections(simulate(m; rng = StableRNG(202)), m)
        layout = compile_household_pairs(data)

        # 1-D Newton on logβ (β = 1/scale), reusing `layout` at every evaluation
        function mle(f; x0 = log(1 / true_scale), steps = 12)
            x = x0
            for _ in 1:steps
                g = ForwardDiff.derivative(f, x)
                h = ForwardDiff.derivative(z -> ForwardDiff.derivative(f, z), x)
                x -= g / h
            end
            return x
        end
        f_fast(logβ) = pairwise_surv_loglik(Exponential(1 / exp(logβ)), data, layout)
        f_dyn(logβ) = pairwise_surv_loglik(Exponential(1 / exp(logβ)), data)

        x_fast = mle(f_fast)
        x_dyn = mle(f_dyn)
        @test x_fast ≈ x_dyn                              # same optimum either way
        @test isapprox(exp(-x_fast), true_scale; rtol = 0.2)  # recovers the scale

        # a scan reusing the one layout object matches a per-call scan pointwise
        grid = 2.0:0.5:6.0
        @test [f_fast(log(1 / s)) for s in grid] ≈ [f_dyn(log(1 / s)) for s in grid]
    end

    @testset "compiled pair layout: Mooncake reverse-mode gradient" begin
        # the fast path targets reverse-mode AD (its zero-allocation reduction is
        # what makes a Mooncake tape cheap). Check the reverse-mode gradient of the
        # compiled path matches ForwardDiff — shared kernel and external mode.
        backend = AutoMooncake(; config = nothing)

        m = ModelSpec(HouseholdProcess(fill(4, 300), Exponential(3.0));
            progression = _sir(6.0))
        data = household_infections(simulate(m; rng = StableRNG(104)), m)
        layout = compile_household_pairs(data)
        f_fast(θ) = pairwise_surv_loglik(Exponential(1 / exp(θ[1])), data, layout)
        x = [log(1 / 3)]
        @test DifferentiationInterface.gradient(f_fast, backend, x) ≈
              ForwardDiff.gradient(f_fast, x)

        Tobs = 30.0
        me = ModelSpec(
            HouseholdProcess(fill(4, 300), Exponential(3.0);
                external_hazard = 0.05, obs_end = Tobs);
            progression = _sir(6.0))
        de = household_infections(simulate(me; rng = StableRNG(106)), me)
        le = compile_household_pairs(de; external = true)
        e_fast(θ) = pairwise_surv_loglik(Exponential(1 / exp(θ[1])), de, le;
            external_hazard = 0.05)
        xe = [log(1 / 3)]
        @test DifferentiationInterface.gradient(e_fast, backend, xe) ≈
              ForwardDiff.gradient(e_fast, xe)
    end

    @testset "contact tracing" begin
        # A case's contacts are its household-mates, so tracing reaches them and
        # quarantine closes their own infectious window.
        clinical = clinical_presentation(incubation_period = LogNormal(1.0, 0.3),
            prob_asymptomatic = 0.0)
        iso = Isolation(onset_to_isolation_delay = Exponential(2.0),
            test_sensitivity = 1.0)
        ct = ContactTracing(probability = 1.0,
            isolation_to_trace_delay = Exponential(0.5))

        build(ivs) = ModelSpec(HouseholdProcess(fill(6, 200), Weibull(1.5, 6.0));
            progression = _sir(7.0), interventions = ivs, attributes = clinical)
        meansize(ivs) = sum(simulate(build(ivs);
                                rng = StableRNG(s)).cumulative_cases for s in 1:15) / 15

        @test meansize([iso, ct]) < meansize([iso])

        st = simulate(build([iso, ct]); rng = StableRNG(2))
        traced = filter(is_traced, st.individuals)
        @test !isempty(traced)
        @test all(is_quarantined, traced)
        # Everyone traced is a household-mate of whoever traced them.
        for t in traced
            src = st.individuals[t.state[:traced_by]]
            @test t.state[:household] == src.state[:household]
        end

        # A trace too late to help must not make things worse (see the network
        # suite for the isolation-pathway interaction this guards).
        late = ContactTracing(probability = 1.0,
            isolation_to_trace_delay = Exponential(500.0))
        @test meansize([iso, late]) <= meansize([iso]) * 1.05
    end

    @testset "conditioned simulation and bare-process household_infections" begin
        m = ModelSpec(HouseholdProcess(fill(4, 300), Exponential(3.0));
            progression = _sir(6.0))
        # `condition` retries until the outbreak size falls in the range
        state = simulate(m; condition = 300:1200, rng = StableRNG(1))
        @test state.cumulative_cases in 300:1200

        # household_infections accepts a bare HouseholdProcess (it delegates
        # through a ModelSpec); the window opens at :infection, matching the SIR
        # simulation above
        bare = HouseholdProcess(fill(4, 50), Exponential(3.0))
        data = household_infections(
            simulate(ModelSpec(bare; progression = _sir(6.0)); rng = StableRNG(2)), bare)
        @test length(data) == 200
    end

    include("test_offspring.jl")
end

include("test_contextual_kernels.jl")
include("test_likelihood_composition.jl")
include("test_initial_cases.jl")
include("test_actions.jl")

include("test_calendar_kernels.jl")

include("test_stateful_kernels.jl")
