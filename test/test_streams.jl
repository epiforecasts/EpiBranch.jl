# A no-op intervention that draws one extra random number straight from
# `state.rng`, for one individual only, the way an intervention with no effect
# on the outbreak still would. It does not route the draw through `stream`, so
# it stands in for any per-person draw a model or intervention has not yet
# migrated to `stream`.
struct _NoiseForOne <: EpiBranch.AbstractIntervention
    id::Int
end
function EpiBranch.initialise_individual!(n::_NoiseForOne, individual, state)
    individual.id == n.id && rand(state.rng)
    return nothing
end

@testset "stream and CommonRandomNumbers" begin
    @testset "stream defaults to the rng itself" begin
        rng = Xoshiro(1)
        @test EpiBranch.stream(rng, 1, :delay) === rng
        @test EpiBranch.stream(rng, (1, 2), :contact) === rng

        srng = StableRNG(1)
        @test EpiBranch.stream(srng, 1, :delay) === srng
    end

    @testset "CommonRandomNumbers keeps one generator per (id, purpose)" begin
        c = CommonRandomNumbers(7)
        s1 = EpiBranch.stream(c, 1, :delay)
        s1_again = EpiBranch.stream(c, 1, :delay)
        @test s1 === s1_again             # same key, same generator object

        s2 = EpiBranch.stream(c, 2, :delay)
        s3 = EpiBranch.stream(c, 1, :acceptance)
        @test s1 !== s2
        @test s1 !== s3

        # Two independently-built `CommonRandomNumbers` with the same seed draw
        # the same value for the same key.
        c2 = CommonRandomNumbers(7)
        @test rand(EpiBranch.stream(c, 1, :delay)) != rand(EpiBranch.stream(c, 1, :delay))
        @test rand(EpiBranch.stream(c2, 5, :contact)) ==
            rand(EpiBranch.stream(CommonRandomNumbers(7), 5, :contact))
    end

    @testset "a draw not routed through stream behaves like an ordinary rng" begin
        c = CommonRandomNumbers(3)
        c2 = CommonRandomNumbers(3)
        @test [rand(c) for _ in 1:5] == [rand(c2) for _ in 1:5]
        @test rand(c, 1:100) isa Int
        @test rand(c, Exponential(2.0)) isa Float64
        @test randexp(c) isa Float64
        @test randperm(c, 10) isa Vector{Int}
    end

    @testset "an intervention's own draw no longer shifts another person's" begin
        prog = [
            Transition(:onset; from = :infection, delay = Exponential(2.0)),
            Transition(:recovered; from = :infection, delay = 6.0, terminal = true),
        ]
        spec(ivs) = ModelSpec(
            BranchingProcess(Poisson(0.0)); progression = prog, interventions = ivs
        )
        run_sim(ivs, rng) = simulate(spec(ivs); n_initial = 5, rng = rng).individuals

        # Under a plain rng, the noise drawn for individual 1 shifts every
        # later individual's onset draw: none of them agree with the baseline.
        baseline = run_sim(AbstractIntervention[], Xoshiro(42))
        noisy = run_sim([_NoiseForOne(1)], Xoshiro(42))
        @test all(onset_time(baseline[i]) != onset_time(noisy[i]) for i in 2:5)

        # Under `CommonRandomNumbers`, every individual's own onset draw comes
        # from its own stream, so the unrelated noise changes nothing.
        baseline_crn = run_sim(AbstractIntervention[], CommonRandomNumbers(42))
        noisy_crn = run_sim([_NoiseForOne(1)], CommonRandomNumbers(42))
        @test all(
            onset_time(baseline_crn[i]) == onset_time(noisy_crn[i]) for i in 1:5
        )
    end

    @testset "population characteristics draw from a per-person stream" begin
        attrs = [
            demographics(age_distribution = Uniform(0, 90)),
            clinical_presentation(
                incubation_period = LogNormal(1.6, 0.5), prob_asymptomatic = 0.3
            ),
            groups(5),
        ]
        spec(ivs) = ModelSpec(
            BranchingProcess(Poisson(0.0), Exponential(5.0));
            attributes = attrs, interventions = ivs
        )
        run_sim(ivs, rng) = simulate(spec(ivs); n_initial = 5, rng = rng).individuals

        # Under `CommonRandomNumbers`, each person's age, symptom status,
        # incubation period and group come from their own stream, so an
        # unrelated draw for another person changes nothing about them.
        baseline_crn = run_sim(AbstractIntervention[], CommonRandomNumbers(42))
        noisy_crn = run_sim([_NoiseForOne(1)], CommonRandomNumbers(42))
        for i in 1:5
            @test baseline_crn[i].state[:age] == noisy_crn[i].state[:age]
            @test baseline_crn[i].state[:sex] == noisy_crn[i].state[:sex]
            @test baseline_crn[i].state[:asymptomatic] == noisy_crn[i].state[:asymptomatic]
            @test isequal(
                baseline_crn[i].state[:incubation_period],
                noisy_crn[i].state[:incubation_period]
            )
            @test baseline_crn[i].state[:group] == noisy_crn[i].state[:group]
        end
    end

    @testset "group_attribute draws from a per-group stream" begin
        attrs = [groups(5), group_attribute(:risk; value = Beta(2, 2))]
        spec(ivs) = ModelSpec(
            BranchingProcess(Poisson(0.0), Exponential(5.0));
            attributes = attrs, interventions = ivs
        )
        run_sim(ivs, rng) = simulate(spec(ivs); n_initial = 5, rng = rng).individuals

        # Under `CommonRandomNumbers`, a group's shared value comes from its
        # own stream. An unrelated draw for another person leaves it
        # unchanged, whichever member happens to populate the cache first.
        baseline_crn = run_sim(AbstractIntervention[], CommonRandomNumbers(42))
        noisy_crn = run_sim([_NoiseForOne(1)], CommonRandomNumbers(42))
        for i in 1:5
            @test baseline_crn[i].state[:risk] == noisy_crn[i].state[:risk]
        end
    end
end
