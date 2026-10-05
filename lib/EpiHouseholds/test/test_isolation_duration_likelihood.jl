# An isolation that lapses leaves the infectious window open and blocks each
# contact over the isolated stretch instead, so the structured likelihood has
# to take that stretch out of each pair's exposure to agree with the simulator.

@testset "Households with a lapsing isolation: simulate to loglikelihood" begin
    clinical = clinical_presentation(incubation_period = Dirac(0.0))
    progression = [
        Transition(:recovered; from = :infection, delay = 30.0, terminal = true),
    ]
    process = HouseholdProcess(fill(5, 120), Exponential(0.4))

    finite = Isolation(
        onset_to_isolation_delay = Dirac(1.0), isolation_duration = Dirac(7.0)
    )
    m = ModelSpec(process; progression, attributes = clinical, interventions = [finite])
    state = simulate(m; rng = StableRNG(17))
    data = household_infections(state, m)

    # The two times the gap is read from are recorded, the model composing an
    # isolation whose duration can lapse.
    @test haskey(data.host_times, :isolation_time)
    @test haskey(data.host_times, :isolation_release_time)
    recorded = coalesce.(data.host_times.isolation_time, Inf)
    @test recorded == EpiBranch.isolation_time.(state.individuals)
    @test any(isfinite, recorded)
    releases = coalesce.(data.host_times.isolation_release_time, Inf)
    @test releases == EpiBranch.isolation_release_time.(state.individuals)
    @test any(isfinite, releases)

    # The window runs to the natural-history close now, so an isolated case is
    # removed at recovery and not at its isolation time.
    isolated = findall(i -> isfinite(EpiBranch.isolation_time(i)), state.individuals)
    @test !isempty(isolated)
    @test all(
        i -> data.removal_time[i] > EpiBranch.isolation_time(state.individuals[i]),
        isolated
    )

    @test isfinite(loglikelihood(data, m))

    # The round trip's job is the plumbing; the arithmetic is pinned exactly
    # on a layer built by hand, where nothing is infected inside a gap, so the
    # whole difference between the two totals is the stretch taken out.
    @test pairwise_surv_loglik(process.kernel, data) > pairwise_surv_loglik(
        process.kernel,
        HouseholdInfections(
            [ind.state[:household]::Int for ind in state.individuals],
            data.infection_time, data.infectious_time, data.removal_time,
            data.is_index; obs_end = data.obs_end
        )
    )
end

@testset "A household layer loses exactly the isolated stretch" begin
    # One household of three. Host 1 is infectious over [0, 30] and isolated
    # over [4, 11]; hosts 2 and 3 are never infected, which makes the total
    # pure escape and the gap's effect exactly the seven days each of them did
    # not spend exposed, at rate `1 / theta`.
    theta = 2.0
    households = [1, 1, 1]
    infection = [0.0, NaN, NaN]
    infectious = [0.0, NaN, NaN]
    removal = [30.0, NaN, NaN]
    index = [true, false, false]

    gapped = HouseholdInfections(
        households, infection, infectious, removal, index; obs_end = Inf,
        host_times = (
            isolation_time = [4.0, Inf, Inf],
            isolation_release_time = [11.0, Inf, Inf],
        )
    )
    plain = HouseholdInfections(
        households, infection, infectious, removal, index; obs_end = Inf
    )

    @test pairwise_surv_loglik(Exponential(theta), plain) ≈ -2 * 30.0 / theta
    @test pairwise_surv_loglik(Exponential(theta), gapped) ≈ -2 * (30.0 - 7.0) / theta
    @test pairwise_surv_loglik(Exponential(theta), gapped) -
        pairwise_surv_loglik(Exponential(theta), plain) ≈ 2 * 7.0 / theta

    # A duration of `Inf` records nothing extra, its window closing at the
    # isolation's own start as before.
    forever = Isolation(onset_to_isolation_delay = Dirac(1.0))
    m_inf = ModelSpec(
        process; progression, attributes = clinical, interventions = [forever]
    )
    inf_state = simulate(m_inf; rng = StableRNG(17))
    @test isempty(household_infections(inf_state, m_inf).host_times)
end

@testset "A wrapped isolation records its stretch like a bare one" begin
    # A wrapper forwards the gate, so a scheduled or capacity-constrained
    # isolation is fitted against the same gapped exposure. Without the
    # forwarding the layer records nothing and the likelihood silently uses the
    # whole exposure.
    clinical = clinical_presentation(incubation_period = Dirac(0.0))
    progression = [
        Transition(:recovered; from = :infection, delay = 30.0, terminal = true),
    ]
    process = HouseholdProcess(fill(5, 60), Exponential(0.4))
    iso = Isolation(
        onset_to_isolation_delay = Dirac(1.0), isolation_duration = Dirac(7.0)
    )

    for wrapped in (
            Scheduled(iso; start_time = 0.0),
            CapacityConstrained(iso; budget_per_period = 1.0e6),
        )
        @test EpiBranch.records_removal_gap(wrapped)
        m = ModelSpec(
            process; progression, attributes = clinical, interventions = [wrapped]
        )
        state = simulate(m; rng = StableRNG(21))
        data = household_infections(state, m)
        @test haskey(data.host_times, :isolation_release_time)
        @test any(isfinite, coalesce.(data.host_times.isolation_release_time, Inf))
        @test isfinite(loglikelihood(data, m))
    end
end

@testset "A quarantine with a duration is fitted on its own" begin
    # `ContactTracing` writes neither key at initialisation, so every host a
    # quarantine never reached reads as `missing` in the recorded column.
    clinical = clinical_presentation(incubation_period = Dirac(0.0))
    progression = [
        Transition(:recovered; from = :infection, delay = 30.0, terminal = true),
    ]
    process = HouseholdProcess(fill(5, 60), Exponential(0.4))
    ct = ContactTracing(
        SymptomaticParent(), 1.0, Exponential(0.5), Quarantine(duration = Dirac(7.0))
    )
    m = ModelSpec(process; progression, attributes = clinical, interventions = [ct])
    state = simulate(m; rng = StableRNG(5))
    data = household_infections(state, m)

    @test EpiBranch.records_removal_gap(ct)
    @test haskey(data.host_times, :isolation_time)
    @test isfinite(loglikelihood(data, m))
end
