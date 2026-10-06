# An isolation that lapses leaves the infectious window open and blocks each
# contact over the isolated stretches instead, so the structured likelihood has
# to take those stretches out of each pair's exposure to agree with the
# simulator.

@testset "Households with a lapsing isolation: simulate to loglikelihood" begin
    clinical = clinical_presentation(incubation_period = Dirac(0.0))
    progression = [
        Transition(:recovered; from = :infection, delay = 30.0, terminal = true),
    ]
    process = HouseholdProcess(fill(5, 120), Exponential(0.4))

    finite = Isolation(
        onset_to_isolation_delay = Dirac(1.0), duration = Dirac(7.0)
    )
    m = ModelSpec(process; progression, attributes = clinical, interventions = [finite])
    state = simulate(m; rng = StableRNG(17))
    data = household_infections(state, m)

    # The stretches the likelihood reads are the ones the simulator recorded.
    @test haskey(data.host_times, :_removal_stretches)
    recorded = data.host_times._removal_stretches
    @test recorded == EpiBranch.removal_stretches.(state.individuals)
    @test any(!isempty, recorded)

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

    # A duration of `Inf` records nothing extra, its window closing at the
    # isolation's own start as before.
    forever = Isolation(onset_to_isolation_delay = Dirac(1.0), duration = Inf)
    m_inf = ModelSpec(
        process; progression, attributes = clinical, interventions = [forever]
    )
    inf_state = simulate(m_inf; rng = StableRNG(17))
    @test isempty(household_infections(inf_state, m_inf).host_times)
end

@testset "A household layer loses exactly the isolated stretches" begin
    # One household of three. Host 1 is infectious over [0, 30]; hosts 2 and 3
    # are never infected, which makes the total pure escape and each stretch's
    # effect exactly the days the two of them did not spend exposed, at rate
    # `1 / theta`.
    theta = 2.0
    households = [1, 1, 1]
    infection = [0.0, NaN, NaN]
    infectious = [0.0, NaN, NaN]
    removal = [30.0, NaN, NaN]
    index = [true, false, false]
    layer(stretches) = HouseholdInfections(
        households, infection, infectious, removal, index; obs_end = Inf,
        host_times = (_removal_stretches = stretches,),
    )
    empty_stretches = Tuple{Float64, Float64}[]

    plain = HouseholdInfections(
        households, infection, infectious, removal, index; obs_end = Inf
    )
    one_gap = layer([[(4.0, 11.0)], empty_stretches, empty_stretches])
    # Quarantined over [4, 11], released, then isolated again over [18, 22]:
    # one pair of times cannot hold both, and both have to come out.
    two_gaps = layer([[(4.0, 11.0), (18.0, 22.0)], empty_stretches, empty_stretches])
    # A stretch running past the end of the window is cut off there.
    overrun = layer([[(25.0, 40.0)], empty_stretches, empty_stretches])

    @test pairwise_surv_loglik(Exponential(theta), plain) ≈ -2 * 30.0 / theta
    @test pairwise_surv_loglik(Exponential(theta), one_gap) ≈ -2 * (30.0 - 7.0) / theta
    @test pairwise_surv_loglik(Exponential(theta), two_gaps) ≈
        -2 * (30.0 - 7.0 - 4.0) / theta
    @test pairwise_surv_loglik(Exponential(theta), overrun) ≈ -2 * 25.0 / theta
    @test pairwise_surv_loglik(Exponential(theta), two_gaps) -
        pairwise_surv_loglik(Exponential(theta), one_gap) ≈ 2 * 4.0 / theta
end

@testset "A wrapper forwards its stretches only when it cannot withdraw them" begin
    iso = Isolation(
        onset_to_isolation_delay = Dirac(1.0), duration = Dirac(7.0)
    )
    key = EpiBranch.REMOVAL_STRETCHES_KEY

    # A schedule with no end, and a capacity budget, both leave the block
    # standing for the stretch they recorded, so both are read back.
    @test EpiBranch.removal_gap_host_times(Scheduled(iso; start_time = 0.0)) == (key,)
    @test EpiBranch.removal_gap_host_times(
        CapacityConstrained(iso; budget_per_period = 1.0e6)
    ) == (key,)

    # A schedule that closes withdraws the block part-way through the stretch,
    # which the record cannot express. It declares none, and narrows the
    # infectious window to the isolation's own start instead, as a duration of
    # `Inf` would.
    lapsing = Scheduled(iso; start_time = 0.0, end_time = 10.0)
    @test EpiBranch.removal_gap_host_times(lapsing) == ()
    case = Individual(id = 1)
    EpiBranch.set_isolated!(case, 8.0; release_time = 15.0)
    @test EpiBranch.infectious_removal_time(lapsing, case) == 8.0
    @test EpiBranch.infectious_removal_time(Scheduled(iso; start_time = 0.0), case) == Inf

    clinical = clinical_presentation(incubation_period = Dirac(0.0))
    progression = [
        Transition(:recovered; from = :infection, delay = 30.0, terminal = true),
    ]
    process = HouseholdProcess(fill(5, 60), Exponential(0.4))
    m = ModelSpec(
        process; progression, attributes = clinical,
        interventions = [Scheduled(iso; start_time = 0.0)]
    )
    state = simulate(m; rng = StableRNG(21))
    data = household_infections(state, m)
    @test data.host_times._removal_stretches ==
        EpiBranch.removal_stretches.(state.individuals)
    @test isfinite(loglikelihood(data, m))

    lapsing_m = ModelSpec(
        process; progression, attributes = clinical, interventions = [lapsing]
    )
    lapsing_data = household_infections(simulate(lapsing_m; rng = StableRNG(21)), lapsing_m)
    @test isempty(lapsing_data.host_times)
end

@testset "A quarantine with a duration is fitted on its own" begin
    # A quarantine keeps its own record, so a leaky isolation composed with
    # tracing does not have the quarantine's perfect block laid over the days
    # it isolated someone itself.
    q = Quarantine(duration = Dirac(7.0))
    contact = Individual(id = 1)
    EpiBranch.set_isolated!(contact, 2.0; release_time = 6.0)
    ct = ContactTracing(OnIsolation(), 1.0, Exponential(0.5), q)
    @test EpiBranch.competing_risk(ct, contact, Individual(id = 2), nothing) === nothing
    @test EpiBranch.removal_gap_host_times(ct) == (EpiBranch.QUARANTINE_STRETCHES_KEY,)

    # Tracing follows a case's own isolation, so one is composed; its duration
    # of `Inf` records no stretch of its own and leaves the quarantine's as the
    # only thing the exposure loses. The household kernel is slow enough that a
    # household is still being exposed when its contacts are quarantined.
    clinical = clinical_presentation(incubation_period = Dirac(0.0))
    progression = [
        Transition(:recovered; from = :infection, delay = 30.0, terminal = true),
    ]
    process = HouseholdProcess(fill(5, 60), Exponential(6.0))
    iso = Isolation(onset_to_isolation_delay = Dirac(1.0), duration = Inf)
    m = ModelSpec(process; progression, attributes = clinical, interventions = [iso, ct])
    state = simulate(m; rng = StableRNG(5))
    data = household_infections(state, m)

    # The column the likelihood reads holds the quarantine's own records. A
    # contact traced twice has its two seven-day quarantines merged, so a
    # stretch runs for seven days or longer and never for less.
    recorded = EpiBranch.removal_stretches.(
        state.individuals, EpiBranch.QUARANTINE_STRETCHES_KEY
    )
    @test data.host_times._removal_stretches == recorded
    quarantined = findall(!isempty, recorded)
    @test !isempty(quarantined)
    lengths = [b - a for i in quarantined for (a, b) in recorded[i]]
    @test all(len -> len > 7.0 || len ≈ 7.0, lengths)
    @test any(len -> len ≈ 7.0, lengths)

    households = [ind.state[:household]::Int for ind in state.individuals]
    plain = HouseholdInfections(
        households, data.infection_time, data.infectious_time, data.removal_time,
        data.is_index; obs_end = data.obs_end
    )
    # The same layer with the stretches put in by hand gives the same total, so
    # the round trip takes out the simulator's own records and nothing else.
    by_hand = HouseholdInfections(
        households, data.infection_time, data.infectious_time, data.removal_time,
        data.is_index; obs_end = data.obs_end,
        host_times = (_removal_stretches = recorded,),
    )
    @test pairwise_surv_loglik(process.kernel, data) ==
        pairwise_surv_loglik(process.kernel, by_hand)
    @test pairwise_surv_loglik(process.kernel, data) >
        pairwise_surv_loglik(process.kernel, plain)
    @test isfinite(loglikelihood(data, m))
end
