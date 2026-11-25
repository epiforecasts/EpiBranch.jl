# Tests for the CensoredDistributions.jl extension: the distribution of the
# serial interval between a linked pair, and its use with
# `double_interval_censored`. Loading CensoredDistributions.jl and
# ConvolvedDistributions.jl together activates `serial_interval`'s only
# method.

@testset "Serial interval (CensoredDistributions.jl extension)" begin
    gt = Gamma(5.95, 1 / 0.77)
    incubation = LogNormal(1.6, 0.5)
    clinical = clinical_presentation(incubation_period = incubation)

    @testset "mean is the generation time's: the incubation periods cancel" begin
        spec = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), gt); attributes = clinical)
        state = simulate(spec; n_initial = 5, rng = StableRNG(1))
        infector, infectee = individuals(state)[1], individuals(state)[2]
        si = serial_interval(spec, infector, infectee)
        @test mean(si) ≈ mean(gt)
        # Variances add: the generation time's, and the two incubation
        # periods' through the convolution and the difference alike.
        @test var(si) ≈ var(gt) + 2 * var(incubation)
    end

    @testset "double_interval_censored gives a finite likelihood" begin
        spec = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), gt); attributes = clinical)
        state = simulate(spec; n_initial = 5, rng = StableRNG(1))
        infector, infectee = individuals(state)[1], individuals(state)[2]
        si = serial_interval(spec, infector, infectee)
        obs = double_interval_censored(si; interval = 1, upper = 21.0)
        @test isfinite(logpdf(obs, 6))
        @test 0 < pdf(obs, 6) < 1
    end

    @testset "a bare BranchingProcess has no attributes to read the incubation period from" begin
        # clinical_presentation attaches through ModelSpec's `attributes`; a
        # bare process alone can never supply the incubation period.
        bare = BranchingProcess(NegBin(2.5, 0.16), gt)
        spec = ModelSpec(bare; attributes = clinical)
        state = simulate(spec; n_initial = 2, rng = StableRNG(1))
        infector, infectee = individuals(state)[1], individuals(state)[2]
        @test_throws "needs the incubation period" serial_interval(bare, infector, infectee)
    end

    @testset "NoGenerationTime has no serial interval" begin
        spec = ModelSpec(BranchingProcess(NegBin(2.5, 0.16)); attributes = clinical)
        state = simulate(spec; n_initial = 2, rng = StableRNG(1))
        @test_throws "needs a fixed generation-time distribution" serial_interval(
            spec, individuals(state)[1], individuals(state)[2]
        )
    end

    @testset "a per-case generation time has no general convolution" begin
        spec = ModelSpec(
            BranchingProcess(NegBin(2.5, 0.16), incubation_linked_generation_time());
            attributes = clinical
        )
        state = simulate(spec; n_initial = 5, rng = StableRNG(1))
        @test_throws "not a per-case function" serial_interval(
            spec, individuals(state)[1], individuals(state)[2]
        )
    end

    @testset "a model with no clinical_presentation has no incubation period" begin
        spec = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), gt))
        state = simulate(spec; n_initial = 2, rng = StableRNG(1))
        @test_throws "needs the incubation period" serial_interval(
            spec, individuals(state)[1], individuals(state)[2]
        )
        err = try
            serial_interval(spec, individuals(state)[1], individuals(state)[2])
        catch e
            e
        end
        @test !occursin("realised_serial_intervals", err.msg)
    end

    @testset "a multi-window model has no single generation time" begin
        windows = (
            Infectiousness(NegBin(1.0, 0.5); kernel = gt),
            Infectiousness(NegBin(1.0, 0.5); kernel = gt, name = :other),
        )
        spec = ModelSpec(BranchingProcess(windows...); attributes = clinical)
        state = simulate(spec; n_initial = 2, rng = StableRNG(1))
        @test_throws ArgumentError serial_interval(
            spec, individuals(state)[1], individuals(state)[2]
        )
    end

    @testset "a shared trajectory behind onset and infectiousness matches the simulated intervals" begin
        function draw_trajectory(rng)
            peak_time = rand(rng, Gamma(6.0, 0.7))
            peak = rand(rng, LogNormal(0.0, 0.6))
            clearance = rand(rng, Gamma(8.0, 0.1))
            load = t -> t < peak_time ? peak * t / peak_time :
                peak * exp(-clearance * (t - peak_time))
            return (; load, peak_time, peak)
        end
        function both_follow(rng, ind)
            trajectory = draw_trajectory(rng)
            ind.state[:trajectory] = trajectory.load
            ind.state[:asymptomatic] = trajectory.peak <= 0.8
            ind.state[:incubation_period] = trajectory.peak_time * 0.8 / trajectory.peak
            return nothing
        end
        shared = Infectiousness(
            Poisson(1.0);
            kernel = ind -> HazardKernel(
                t -> 0.4 * EpiBranch.trajectory_value(ind, t); upper = 60.0
            )
        )
        onset_only = Infectiousness(Poisson(1.0); kernel = Exponential(5.0))
        for window in (shared, onset_only)
            spec = ModelSpec(BranchingProcess(window); attributes = both_follow)
            states = simulate(
                spec, 4000; max_generations = 3, max_cases = 3000, rng = StableRNG(1)
            )
            simulated = realised_serial_intervals(states)
            a = Individual(id = 1)
            si = serial_interval(spec, a, a; n_trajectories = 400, rng = StableRNG(2))
            for x in (-2, 0, 3, 6, 10)
                @test cdf(si, x) ≈ mean(simulated .<= x) atol = 0.025
            end
        end
    end

    @testset "a shared trajectory needs at least two draws and an onset" begin
        spec = ModelSpec(
            BranchingProcess(Poisson(1.0), Exponential(5.0));
            attributes = (rng, ind) -> (ind.state[:asymptomatic] = true)
        )
        a = Individual(id = 1)
        @test_throws "n_trajectories" serial_interval(spec, a, a; n_trajectories = 1)
        @test_throws "no draw of the attributes with an onset" serial_interval(spec, a, a)
    end
end
