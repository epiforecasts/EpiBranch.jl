# A vaccination defined outside the package, taking part in the shared
# vaccination machinery only by holding a `VaccineEffect`.
struct _TestCampaignVaccination{V <: VaccineEffect} <: AbstractVaccination
    effect::V
    campaign_time::Float64
end

EpiBranch.vaccine_effect(v::_TestCampaignVaccination) = v.effect

function EpiBranch.apply_post_transmission!(v::_TestCampaignVaccination, state,
        new_contacts)
    for ind in new_contacts
        EpiBranch._record_vaccination!(v, ind, v.campaign_time, state.rng)
    end
    return nothing
end

@testset "VaccineEffect" begin
    @testset "Waning is shared by built-in and custom vaccinations" begin
        decay = dt -> exp(-dt / 30)
        effect = VaccineEffect(efficacy = 0.8, waning = decay)
        custom = _TestCampaignVaccination(effect, 0.0)
        @test EpiBranch.waning(custom) === decay
        for v in (RingVaccination(efficacy = 0.8, waning = decay),
                GroupVaccination(efficacy = 0.8, waning = decay),
                MassVaccination(efficacy = 0.8, waning = decay, eligibility_time = 0.0))
            @test EpiBranch.vaccine_effect(v).waning === decay
            @test EpiBranch.waning(v) === v.waning === decay
        end
    end

    @testset "Keyword constructors build the shared effect" begin
        rv = RingVaccination(efficacy = 0.7, delay_to_immunity = 14.0,
            severity_efficacy = 0.3, mode = AllOrNothingMode(), dose_label = :prime,
            coverage = 0.9, dose_delay = 2.0)
        effect = EpiBranch.vaccine_effect(rv)
        @test effect isa VaccineEffect
        @test effect.efficacy == 0.7
        @test effect.delay_to_immunity === 14.0
        @test EpiBranch.delay_to_immunity(rv) === 14.0
        @test effect.severity_efficacy == 0.3
        @test effect.mode isa AllOrNothingMode
        @test effect.dose_label === :prime
        @test rv.dose_delay === 2.0

        # A distribution or a function is held as given, for the draw made when
        # the dose is recorded.
        vary = RingVaccination(
            efficacy = Beta(2, 2), delay_to_immunity = Uniform(7.0, 21.0),
            severity_efficacy = (rng, ind) -> 0.4)
        @test vary.efficacy == Beta(2, 2)
        @test EpiBranch.delay_to_immunity(vary) == Uniform(7.0, 21.0)
        @test EpiBranch.severity_efficacy(vary)(nothing, nothing) == 0.4

        mv = MassVaccination(efficacy = Beta(2, 2), eligibility_time = 5.0)
        @test EpiBranch.vaccine_effect(mv) ==
              VaccineEffect(efficacy = Beta(2, 2))
        gv = GroupVaccination(efficacy = 0.6, delay_to_immunity = 3.0)
        @test EpiBranch.vaccine_effect(gv) ==
              VaccineEffect(efficacy = 0.6, delay_to_immunity = 3.0)
    end

    @testset "Effect parameters read as properties of every vaccination" begin
        for v in (RingVaccination(efficacy = 0.8, severity_efficacy = 0.2),
            MassVaccination(efficacy = 0.8, eligibility_time = 1.0,
            severity_efficacy = 0.2),
            GroupVaccination(efficacy = 0.8, severity_efficacy = 0.2))
            @test v.efficacy == EpiBranch.efficacy(v) == 0.8
            @test v.severity_efficacy == EpiBranch.severity_efficacy(v) == 0.2
            @test v.delay_to_immunity == EpiBranch.delay_to_immunity(v) == 0.0
            @test v.mode === EpiBranch.effect_mode(v) === LeakyMode()
            @test v.dose_label === EpiBranch.dose_label(v) === :default
            @test issubset(fieldnames(VaccineEffect), propertynames(v))
            @test (@inferred (x -> x.efficacy)(v)) == 0.8
        end
    end

    @testset "A misspelt keyword names the vaccination and its keywords" begin
        @test_throws UndefKeywordError RingVaccination()
        @test_throws UndefKeywordError MassVaccination(efficacy = 0.5)
        typos = [() -> RingVaccination(efficacy = 0.5, dose_dely = 2.0),
            () -> RingVaccination(efficacy = 0.5, coverge = 0.5),
            () -> RingVaccination(efficacy = 0.5, requires_dse = :prime),
            () -> RingVaccination(efficacy = 0.5, eligibilty_window = 3.0),
            () -> MassVaccination(efficacy = 0.5, eligibility_time = 1.0,
                group_key = :village),
            () -> GroupVaccination(efficacy = 0.5, eligibility_time = 1.0)]
        for make in typos
            err = try
                make()
            catch e
                e
            end
            @test err isa ArgumentError
            # The message names the type the caller wrote, not `VaccineEffect`,
            # and lists the keywords it does take.
            @test occursin("Vaccination has no keyword argument", err.msg)
            @test occursin("`efficacy`", err.msg)
            @test !occursin("VaccineEffect", err.msg)
        end
    end

    @testset "show prints the constructor keywords" begin
        rv = RingVaccination(efficacy = 0.9, dose_label = :boost, requires_dose = :prime)
        @test repr(rv) ==
              "RingVaccination(efficacy = 0.9, severity_efficacy = 0.0, " *
                          "delay_to_immunity = 0.0, waning = nothing, mode = LeakyMode(), dose_label = :boost, " *
              "coverage = 1.0, dose_delay = 0.0, requires_dose = :prime, " *
              "eligibility_window = Inf, post_exposure_efficacy = 0.0, " *
              "onward_efficacy = 0.0)"
        # A distributional parameter prints where a scalar one did.
        @test occursin("delay_to_immunity = Uniform",
            repr(MassVaccination(efficacy = 0.5, eligibility_time = 1.0,
                delay_to_immunity = Uniform(7.0, 21.0))))
        for v in (rv, MassVaccination(efficacy = 0.5, eligibility_time = 3.0),
            GroupVaccination(efficacy = 0.5, group_key = :village))
            @test eval(Meta.parse(repr(v))) == v
        end
    end

    @testset "A user-defined vaccination inherits the shared machinery" begin
        v = _TestCampaignVaccination(
            VaccineEffect(efficacy = 1.0, severity_efficacy = 0.5,
                delay_to_immunity = 2.0, dose_label = :campaign),
            4.0)
        contact = Individual(id = 2, parent_id = 1, infection_time = 10.0)
        EpiBranch.initialise_individual!(v, contact, nothing)
        @test !is_vaccinated(contact; dose_label = :campaign)

        EpiBranch._record_vaccination!(v, contact, 4.0, StableRNG(1))
        @test is_vaccinated(contact; dose_label = :campaign)
        @test immunity_time(contact; dose_label = :campaign) == 6.0
        @test severity_efficacy(contact; dose_label = :campaign) == 0.5

        risk = EpiBranch.competing_risk(v, Individual(id = 1), contact, nothing)
        @test risk.event_time == 6.0
        @test risk.block_probability == 1.0

        # A ring dose can require the campaign dose once it is listed first.
        boost = RingVaccination(efficacy = 0.5, requires_dose = :campaign)
        @test EpiBranch._validate_dose_schedule([v, boost]) === nothing
        @test_throws ArgumentError EpiBranch._validate_dose_schedule([boost, v])

        # Fully effective immunity from day 0 blocks every exposure.
        model = ModelSpec(BranchingProcess(Poisson(3.0), Exponential(5.0));
            interventions = [_TestCampaignVaccination(
                VaccineEffect(efficacy = 1.0), 0.0)])
        results = simulate(model, 20; max_cases = 100, rng = StableRNG(3))
        @test all(s -> s.cumulative_cases == 1, results)
    end
end
