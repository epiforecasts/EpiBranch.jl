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

    # Taking the isolated stretch out leaves strictly less exposure than
    # ignoring it, which makes the escape terms larger.
    ignoring = HouseholdInfections(
        [ind.state[:household]::Int for ind in state.individuals],
        data.infection_time, data.infectious_time, data.removal_time, data.is_index;
        obs_end = data.obs_end
    )
    @test pairwise_surv_loglik(process.kernel, data) >
        pairwise_surv_loglik(process.kernel, ignoring)

    # A duration of `Inf` records nothing extra, its window closing at the
    # isolation's own start as before.
    forever = Isolation(onset_to_isolation_delay = Dirac(1.0))
    m_inf = ModelSpec(
        process; progression, attributes = clinical, interventions = [forever]
    )
    inf_state = simulate(m_inf; rng = StableRNG(17))
    @test isempty(household_infections(inf_state, m_inf).host_times)
end
