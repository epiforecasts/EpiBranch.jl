# A component with no recorded stretch of its own, for the unnamed-key tests
# below. It acts only through `removal_gap_host_times`, not any simulation
# hook, so `has_effect` is declared true.
struct _Shield <: EpiBranch.AbstractIntervention end
EpiBranch.has_effect(::_Shield) = true

# As `_Shield`, but withdrawing the scratch key on purpose instead of naming
# it.
struct _WithdrawingShield <: EpiBranch.AbstractIntervention end
EpiBranch.has_effect(::_WithdrawingShield) = true

@testset "Infection likelihood compatibility" begin
    compatible = EpiBranch.infection_likelihood_compatible
    iso = Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf)
    clinical = clinical_presentation(incubation_period = Exponential(1.0))
    @test compatible(NoAttributes())
    @test compatible(clinical)
    @test compatible([clinical])
    @test compatible((clinical, NoAttributes()))
    @test !compatible((rng, ind) -> nothing)
    @test !compatible(transmission_traits(susceptibility = 0.5))
    @test compatible(Transition(:recovered; delay = 1.0))
    @test compatible(Reporting(delay = 1.0))
    @test compatible(Hospitalisation(delay = 1.0))
    @test compatible(Recovery(delay = 1.0))
    @test compatible(Death(delay = 1.0, probability = 0.1))
    @test compatible(iso)
    @test !compatible(
        Isolation(
            onset_to_isolation_delay = Exponential(1.0),
            post_isolation_transmission = 0.5, duration = Inf
        )
    )
    for quarantine in (false, true)
        @test compatible(
            ContactTracing(
                probability = 0.5,
                isolation_to_trace_delay = Exponential(1.0),
                action = quarantine ? Quarantine(duration = Inf) : FlagOnly()
            )
        )
    end
    @test compatible(Scheduled(iso; start_time = 0.0))
    @test compatible(CapacityConstrained(iso; budget_per_period = 2))
    model = ModelSpec(
        BranchingProcess(Poisson(0.0)); attributes = clinical,
        interventions = [iso], progression = [Recovery(delay = 1.0)]
    )
    @test EpiBranch._validate_infection_likelihood(model) === nothing
    bad = ModelSpec(BranchingProcess(Poisson(0.0)); attributes = (rng, ind) -> nothing)
    @test_throws ArgumentError EpiBranch._validate_infection_likelihood(bad)

    @testset "vaccination" begin
        # the susceptibility risk reaches the likelihood through
        # `susceptibility_components`
        @test compatible(MassVaccination(efficacy = 0.6, eligibility_time = 10.0))
        @test compatible(GroupVaccination(efficacy = 0.6))
        @test compatible(RingVaccination(efficacy = 0.6))
        # a labelled dose belongs to a schedule whose doses compose as
        # separate competing risks
        @test !compatible(
            MassVaccination(
                efficacy = 0.6, eligibility_time = 10.0, dose_label = :boost
            )
        )
        @test !compatible(GroupVaccination(efficacy = 0.6, dose_label = :boost))
        # `onward_efficacy`/`post_exposure_efficacy` act on hazards the
        # likelihood's kernel never sees
        @test !compatible(RingVaccination(efficacy = 0.6, onward_efficacy = 0.3))
        @test !compatible(RingVaccination(efficacy = 0.6, post_exposure_efficacy = 0.3))
        @test compatible(
            Scheduled(
                MassVaccination(efficacy = 0.6, eligibility_time = 10.0); start_time = 0.0
            )
        )
    end

    @testset "susceptibility effects of composed components" begin
        mv = MassVaccination(efficacy = 0.7, eligibility_time = 10.0)
        host(τ) = LayerHost(1, NaN, EpiBranch._LayerHostState((; immunity_time = [τ]), 1))
        effect = EpiBranch.susceptibility_components(EpiBranch.vaccine_effect(mv), host(2.0))
        @test EpiBranch.susceptibility_components(iso, host(2.0)) === nothing
        @test EpiBranch.susceptibility_components(mv, host(2.0)) == effect
        @test EpiBranch.susceptibility_components([iso, mv], host(2.0)) == effect
        @test EpiBranch.susceptibility_components(
            [Scheduled(mv; start_time = 0.0)], host(2.0)
        ) == effect
        # a host never vaccinated is left as it is
        @test EpiBranch.susceptibility_components([iso, mv], host(Inf)) === nothing
        # two components modifying one host are not combined silently
        @test_throws ArgumentError EpiBranch.susceptibility_components(
            [mv, GroupVaccination(efficacy = 0.5)], host(2.0)
        )
        # a layer that never recorded the time a vaccination reads says so
        no_times = LayerHost(1, NaN, EpiBranch._LayerHostState((;), 1))
        @test_throws ArgumentError EpiBranch.susceptibility_components(mv, no_times)

        @test EpiBranch.susceptibility_host_times(iso) == ()
        @test EpiBranch.susceptibility_host_times(mv) == (:immunity_time,)
        @test EpiBranch.susceptibility_host_times(
            CapacityConstrained(RingVaccination(efficacy = 0.6); budget_per_period = 2)
        ) == (:immunity_time,)
        labelled = MassVaccination(
            efficacy = 0.6, eligibility_time = 10.0, dose_label = :boost
        )
        @test EpiBranch.susceptibility_host_times(labelled) == (:immunity_time_boost,)
        spec = ModelSpec(
            BranchingProcess(Poisson(0.0)); interventions = [iso, mv, labelled]
        )
        @test EpiBranch._layer_host_time_keys(spec, (:onset_time, :immunity_time)) ==
            [:onset_time, :immunity_time, :immunity_time_boost]
    end

    @testset "A recorded removal must name its key or be withdrawn" begin
        shield_key = :_scratch_shield_stretches
        mini(inds) = SimulationState(
            inds, Int[], 0, StableRNG(1), 0, false, nothing, Inf, nothing,
            AbstractClinicalTransition[]
        )
        bare = ModelSpec(BranchingProcess(Poisson(0.0)))

        # Nothing names the key and nothing withdraws it: the likelihood would
        # otherwise fit on the whole exposure as if the stretch never happened.
        lapsing = [Individual(id = 1)]
        EpiBranch.record_removal!(lapsing[1], 2.0, 9.0; key = shield_key)
        @test_throws ArgumentError EpiBranch._layer_host_times(mini(lapsing), bare, ())

        # Naming the key lets the stretch through, merged into the layer's own
        # removal column.
        EpiBranch.removal_gap_host_times(::_Shield) = (shield_key,)
        named = ModelSpec(
            BranchingProcess(Poisson(0.0)); interventions = [_Shield()]
        )
        result = EpiBranch._layer_host_times(mini(lapsing), named, ())
        @test result[shield_key] == [[(2.0, 9.0)]]
        @test result[EpiBranch.REMOVAL_STRETCHES_KEY] == [[(2.0, 9.0)]]

        # Declaring it withdrawn instead is just as good: another component
        # means to leave it out, so it stays out of `host_times` rather than
        # erroring.
        EpiBranch.removal_gap_host_times_withdrawn(::_WithdrawingShield) = (shield_key,)
        withdrawn = ModelSpec(
            BranchingProcess(Poisson(0.0)); interventions = [_WithdrawingShield()]
        )
        @test EpiBranch._layer_host_times(mini(lapsing), withdrawn, ()) == (;)

        # A stretch that never releases needs no declaration either: the
        # infectious window closes there instead of this column holding it
        # (see `infectious_removal_time`).
        standing = [Individual(id = 1)]
        EpiBranch.record_removal!(standing[1], 2.0, Inf; key = shield_key)
        @test EpiBranch._layer_host_times(mini(standing), bare, ()) == (;)
    end

    @testset "removal_gap_host_times_withdrawn for built-in components" begin
        withdrawn = EpiBranch.removal_gap_host_times_withdrawn
        perfect = Isolation(onset_to_isolation_delay = Exponential(1.0), duration = 5.0)
        leaky = Isolation(
            onset_to_isolation_delay = Exponential(1.0),
            post_isolation_transmission = 0.5, duration = 5.0
        )
        # Perfect isolation names its own key above; leaky isolation never
        # reads the shared history back, so it withdraws it instead.
        @test withdrawn(perfect) == ()
        @test withdrawn(leaky) == (EpiBranch.REMOVAL_STRETCHES_KEY,)

        quarantine = ContactTracing(
            probability = 0.5, isolation_to_trace_delay = Exponential(1.0),
            action = Quarantine(duration = 5.0)
        )
        flagging = ContactTracing(
            probability = 0.5, isolation_to_trace_delay = Exponential(1.0),
            action = FlagOnly()
        )
        # A releasing quarantine's own key already names the days; the shared
        # history that `apply_trace!` also writes is withdrawn rather than
        # read twice.
        @test withdrawn(quarantine) == (EpiBranch.REMOVAL_STRETCHES_KEY,)
        @test withdrawn(flagging) == ()

        source_quarantine = SourceTracing(
            probability = 0.5, isolation_to_trace_delay = Exponential(1.0),
            action = Quarantine(duration = 5.0)
        )
        # `SourceTracing` withdraws the same shared key as `ContactTracing`,
        # for the same reason.
        @test withdrawn(source_quarantine) == (EpiBranch.REMOVAL_STRETCHES_KEY,)

        # A schedule that can lapse withdraws the inner removal's key instead
        # of naming it, since closing the window part-way through a stretch
        # means the record no longer shows when the block ended; one that
        # cannot close forwards the inner key as named, not withdrawn.
        lapsing_schedule = Scheduled(perfect; start_time = 0.0, end_time = 10.0)
        standing_schedule = Scheduled(perfect; start_time = 0.0)
        @test withdrawn(lapsing_schedule) == (EpiBranch.REMOVAL_STRETCHES_KEY,)
        @test withdrawn(standing_schedule) == ()

        # `quarantine` above names its own key through `removal_gap_host_times`
        # and withdraws the shared history at once. A lapsing schedule around
        # it forwards neither, so both must come through as withdrawn here,
        # not just the one `removal_gap_host_times` would have named.
        lapsing_quarantine = Scheduled(quarantine; start_time = 0.0, end_time = 10.0)
        standing_quarantine = Scheduled(quarantine; start_time = 0.0)
        @test Set(withdrawn(lapsing_quarantine)) ==
            Set((EpiBranch.QUARANTINE_STRETCHES_KEY, EpiBranch.REMOVAL_STRETCHES_KEY))
        @test withdrawn(standing_quarantine) == (EpiBranch.REMOVAL_STRETCHES_KEY,)

        capacity_quarantine = CapacityConstrained(quarantine; budget_per_period = 2)
        @test withdrawn(capacity_quarantine) == (EpiBranch.REMOVAL_STRETCHES_KEY,)
    end
end
