# A vaccination defined outside the package, taking part in the shared
# vaccination machinery only by holding a `VaccineEffect`.
struct _TestCampaignVaccination{V <: VaccineEffect} <: AbstractVaccination
    effect::V
    campaign_time::Float64
end

EpiBranch.vaccine_effect(v::_TestCampaignVaccination) = v.effect

# An `AbstractEffectMode` defined outside the package (#373): deterministic
# instead of either `LeakyMode`'s identity or `AllOrNothingMode`'s Bernoulli
# draw, exercising the two public hooks a new mode needs.
struct _TestThresholdMode <: AbstractEffectMode
    threshold::Float64
end
EpiBranch.realised_efficacy(mode::_TestThresholdMode, eff, rng) =
    eff >= mode.threshold ? 1.0 : 0.0
function EpiBranch.apply_post_transmission!(
        v::_TestCampaignVaccination, state,
        new_contacts
    )
    for ind in new_contacts
        EpiBranch._record_vaccination!(v, ind, v.campaign_time, state.rng)
    end
    return nothing
end

# A third effect mode defined outside the package, as all-or-nothing as
# `AllOrNothingMode` and so unable to combine with `waning`. Used
# below to check that `VaccineEffect` rejects the combination through the
# `supports_waning` trait rather than a hard-coded `AllOrNothingMode` check.
struct _TestBlockedMode <: AbstractEffectMode end
EpiBranch.supports_waning(::_TestBlockedMode) = false
EpiBranch.realised_efficacy(::_TestBlockedMode, eff, rng) =
    float(rand(rng, Bernoulli(eff)))

@testset "VaccineEffect" begin
    @testset "Waning is shared by built-in and custom vaccinations" begin
        decay = dt -> exp(-dt / 30)
        effect = VaccineEffect(efficacy = 0.8, waning = decay)
        custom = _TestCampaignVaccination(effect, 0.0)
        @test EpiBranch.waning(custom) === decay
        for v in (
                RingVaccination(efficacy = 0.8, waning = decay),
                GroupVaccination(efficacy = 0.8, waning = decay),
                MassVaccination(efficacy = 0.8, waning = decay, eligibility_time = 0.0),
            )
            @test EpiBranch.vaccine_effect(v).waning === decay
            @test EpiBranch.waning(v) === v.waning === decay
        end
    end

    @testset "Keyword constructors build the shared effect" begin
        rv = RingVaccination(
            efficacy = 0.7, delay_to_immunity = 14.0,
            severity_efficacy = 0.3, mode = AllOrNothingMode(), dose_label = :prime,
            coverage = 0.9, dose_delay = 2.0
        )
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
            severity_efficacy = (rng, ind) -> 0.4
        )
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
        for v in (
                RingVaccination(efficacy = 0.8, severity_efficacy = 0.2),
                MassVaccination(
                    efficacy = 0.8, eligibility_time = 1.0,
                    severity_efficacy = 0.2
                ),
                GroupVaccination(efficacy = 0.8, severity_efficacy = 0.2),
            )
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
        typos = [
            () -> RingVaccination(efficacy = 0.5, dose_dely = 2.0),
            () -> RingVaccination(efficacy = 0.5, coverge = 0.5),
            () -> RingVaccination(efficacy = 0.5, requires_dse = :prime),
            () -> RingVaccination(efficacy = 0.5, eligibilty_window = 3.0),
            () -> MassVaccination(
                efficacy = 0.5, eligibility_time = 1.0,
                group_key = :community
            ),
            () -> GroupVaccination(efficacy = 0.5, eligibility_time = 1.0),
        ]
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
        @test occursin(
            "delay_to_immunity = Uniform",
            repr(
                MassVaccination(
                    efficacy = 0.5, eligibility_time = 1.0,
                    delay_to_immunity = Uniform(7.0, 21.0)
                )
            )
        )
        for v in (
                rv, MassVaccination(efficacy = 0.5, eligibility_time = 3.0),
                GroupVaccination(efficacy = 0.5, group_key = :community),
            )
            @test eval(Meta.parse(repr(v))) == v
        end
    end

    @testset "A user-defined vaccination inherits the shared machinery" begin
        v = _TestCampaignVaccination(
            VaccineEffect(
                efficacy = 1.0, severity_efficacy = 0.5,
                delay_to_immunity = 2.0, dose_label = :campaign
            ),
            4.0
        )
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
        model = ModelSpec(
            BranchingProcess(Poisson(3.0), Exponential(5.0));
            interventions = [
                _TestCampaignVaccination(
                    VaccineEffect(efficacy = 1.0), 0.0
                ),
            ]
        )
        results = simulate(model, 20; max_cases = 100, rng = StableRNG(3))
        @test all(s -> s.cumulative_cases == 1, results)
    end

    @testset "A user-defined vaccination uses the public action protocol" begin
        v = _TestCampaignVaccination(VaccineEffect(efficacy = 0.7), 5.0)
        function EpiBranch.intervention_actions(v::_TestCampaignVaccination, state, candidates)
            return [EpiBranch.dose_action(v, ind, v.campaign_time) for ind in candidates]
        end
        EpiBranch.capacity_key(v::_TestCampaignVaccination) = EpiBranch._vaccinated_key(:default)
        function EpiBranch.capacity_time_key(v::_TestCampaignVaccination)
            return EpiBranch._vaccination_time_key(:default)
        end

        contact = Individual(id = 2, parent_id = 1, infection_time = 10.0)
        EpiBranch.initialise_individual!(v, contact, nothing)
        @test !is_vaccinated(contact)

        actions = EpiBranch.intervention_actions(v, nothing, [contact])
        @test only(actions) isa EpiBranch.InterventionAction
        state = (; rng = StableRNG(1))
        EpiBranch.apply_actions!(v, state, [contact])
        @test is_vaccinated(contact)
        @test contact.state[:vaccination_time] == 5.0

        # `record_dose!`, unlike `_record_vaccination!`, skips a person already
        # given this dose rather than redrawing their efficacy.
        stored_efficacy = vaccine_efficacy(contact)
        EpiBranch.record_dose!(v, contact, 9.0, StableRNG(2))
        @test vaccine_efficacy(contact) == stored_efficacy
        @test contact.state[:vaccination_time] == 5.0

        # A custom vaccination built on `intervention_actions` is admitted by
        # `CapacityConstrained` exactly as `MassVaccination` is.
        cc = CapacityConstrained(
            _TestCampaignVaccination(VaccineEffect(efficacy = 0.7), 5.0);
            budget_per_period = 1.0
        )
        contacts = [Individual(id = i, parent_id = 1, infection_time = 10.0) for i in 1:3]
        for ind in contacts
            EpiBranch.initialise_individual!(cc.intervention, ind, nothing)
        end
        cc_state = EpiBranch.new_state(
            BranchingProcess(Poisson(0.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), StableRNG(1)
        )
        append!(cc_state.individuals, contacts)
        EpiBranch.apply_actions!(cc, cc_state, contacts)
        @test count(is_vaccinated, contacts) == 1
    end

    @testset "AllOrNothingMode draws a responder once, at dose time" begin
        # At efficacy 1.0 every dose makes a responder: certain block from
        # immunity time on, exactly as LeakyMode would give at that efficacy.
        full = RingVaccination(efficacy = 1.0, mode = AllOrNothingMode())
        responder = Individual(id = 2, parent_id = 1, infection_time = 10.0)
        EpiBranch._record_vaccination!(full, responder, 0.0, StableRNG(1))
        @test vaccine_efficacy(responder) == 1.0
        risk = EpiBranch.competing_risk(full, Individual(id = 1), responder, nothing)
        @test risk.block_probability == 1.0
        # A responder's block never fades (`waning` is disallowed under this
        # mode), so a race can drop a pair it blocks instead of redrawing
        # towards it on an unbounded window.
        @test EpiBranch.standing_block(full)

        # At efficacy 0.0 nobody responds: no risk is built at all, so a
        # non-responder is exposed exactly as an unvaccinated contact.
        none = RingVaccination(efficacy = 0.0, mode = AllOrNothingMode())
        non_responder = Individual(id = 3, parent_id = 1, infection_time = 10.0)
        EpiBranch._record_vaccination!(none, non_responder, 0.0, StableRNG(1))
        @test vaccine_efficacy(non_responder) == 0.0
        @test EpiBranch.competing_risk(none, Individual(id = 1), non_responder, nothing) ===
            nothing

        # At an intermediate efficacy the stored value is always 0 or 1 — a
        # Bernoulli(efficacy) draw — never the raw efficacy LeakyMode would
        # keep, and responders occur with roughly that probability.
        half = RingVaccination(efficacy = 0.5, mode = AllOrNothingMode())
        draws = map(1:1000) do i
            contact = Individual(id = i, parent_id = 0, infection_time = 10.0)
            EpiBranch._record_vaccination!(half, contact, 0.0, StableRNG(i))
            vaccine_efficacy(contact)
        end
        @test all(x -> x == 0.0 || x == 1.0, draws)
        @test 0.4 < count(==(1.0), draws) / length(draws) < 0.6

        # LeakyMode is unaffected: the stored value is the sampled efficacy
        # itself, whatever it is.
        leaky = RingVaccination(efficacy = 0.5, mode = LeakyMode())
        contact = Individual(id = 1, parent_id = 0, infection_time = 10.0)
        EpiBranch._record_vaccination!(leaky, contact, 0.0, StableRNG(1))
        @test vaccine_efficacy(contact) == 0.5
        # Not declared standing even at efficacy 1.0: a waning value could
        # still give a smaller block to a later exposure.
        @test !EpiBranch.standing_block(RingVaccination(efficacy = 1.0, mode = LeakyMode()))
    end

    @testset "AllOrNothingMode draws a responder for a dose recorded beforehand" begin
        # A campaign before the run records its dose through `attributes`.
        # Under AllOrNothingMode its efficacy becomes a responder status once,
        # when the individual is set up; a recorded 0 or 1 is kept as it is.
        prior(eff) = Individual(
            id = 1,
            state = Dict{Symbol, Any}(
                :vaccinated => true, :vaccination_time => -10.0,
                :vaccine_efficacy => eff
            )
        )
        all_or_nothing = RingVaccination(efficacy = 0.5, mode = AllOrNothingMode())
        draws = map(1:1000) do i
            ind = prior(0.5)
            EpiBranch.initialise_individual!(all_or_nothing, ind, (; rng = StableRNG(i)))
            vaccine_efficacy(ind)
        end
        @test all(x -> x == 0.0 || x == 1.0, draws)
        @test 0.4 < count(==(1.0), draws) / length(draws) < 0.6
        for status in (0.0, 1.0)
            ind = prior(status)
            EpiBranch.initialise_individual!(all_or_nothing, ind, (; rng = StableRNG(1)))
            @test vaccine_efficacy(ind) == status
        end
        leaky = RingVaccination(efficacy = 0.5, mode = LeakyMode())
        ind = prior(0.5)
        EpiBranch.initialise_individual!(leaky, ind, (; rng = StableRNG(1)))
        @test vaccine_efficacy(ind) == 0.5
    end

    @testset "Custom AbstractEffectMode defined outside the package" begin
        above = RingVaccination(efficacy = 0.6, mode = _TestThresholdMode(0.5))
        responder = Individual(id = 2, parent_id = 1, infection_time = 10.0)
        EpiBranch._record_vaccination!(above, responder, 0.0, StableRNG(1))
        @test vaccine_efficacy(responder) == 1.0

        below = RingVaccination(efficacy = 0.4, mode = _TestThresholdMode(0.5))
        non_responder = Individual(id = 3, parent_id = 1, infection_time = 10.0)
        EpiBranch._record_vaccination!(below, non_responder, 0.0, StableRNG(1))
        @test vaccine_efficacy(non_responder) == 0.0

        # A dose recorded before the run (via `attributes`) is re-realised too.
        prior = Individual(
            id = 1,
            state = Dict{Symbol, Any}(
                :vaccinated => true, :vaccination_time => -10.0, :vaccine_efficacy => 0.6
            )
        )
        EpiBranch.initialise_individual!(above, prior, (; rng = StableRNG(1)))
        @test vaccine_efficacy(prior) == 1.0
    end

    @testset "waning has no AllOrNothingMode meaning yet" begin
        decay = dt -> exp(-dt / 30)
        @test_throws ArgumentError VaccineEffect(
            efficacy = 0.5, waning = decay, mode = AllOrNothingMode()
        )
        @test_throws ArgumentError RingVaccination(
            efficacy = 0.5, waning = decay, mode = AllOrNothingMode()
        )
        # Either on its own is fine.
        @test VaccineEffect(efficacy = 0.5, waning = decay, mode = LeakyMode()) isa
            VaccineEffect
        @test VaccineEffect(efficacy = 0.5, mode = AllOrNothingMode()) isa VaccineEffect
    end

    @testset "supports_waning is dispatched, not matched against AllOrNothingMode" begin
        @test EpiBranch.supports_waning(LeakyMode())
        @test !EpiBranch.supports_waning(AllOrNothingMode())
        decay = dt -> exp(-dt / 30)
        # A third-party mode rejects `waning` the same way, by declaring
        # itself through the trait rather than requiring a core-level edit.
        @test_throws ArgumentError VaccineEffect(
            efficacy = 0.5, waning = decay, mode = _TestBlockedMode()
        )
        @test VaccineEffect(efficacy = 0.5, mode = _TestBlockedMode()) isa VaccineEffect

        # The trait reaches the race too: a mode that disallows waning composes
        # a block that is certain for good, which is what `standing_block`
        # reports, rather than the race naming `AllOrNothingMode` by type.
        blocked = RingVaccination(efficacy = 0.5, mode = _TestBlockedMode())
        @test EpiBranch.standing_block(blocked)
        @test EpiBranch.standing_block(RingVaccination(efficacy = 1.0, mode = AllOrNothingMode()))
        @test !EpiBranch.standing_block(RingVaccination(efficacy = 1.0))

        # The likelihood needs each mode's own decomposition, so a mode that
        # has not given one says so rather than raising a `MethodError` from
        # inside the mixture.
        @test_throws ArgumentError EpiBranch._dose_components(
            _TestBlockedMode(), 0.5, nothing, 0.0
        )
    end

    @testset "Branching process: the two modes agree in distribution" begin
        # Every contact on a branching process is exposed exactly once, so the
        # marginal probability a vaccinated contact escapes infection is
        # `efficacy` under both modes (see the AbstractVaccination docstring).
        # Vaccinating everyone before the outbreak lets the dose act on every
        # exposure, so both modes must change containment by the same amount.
        function containment(efficacy, mode, seed)
            mv = MassVaccination(efficacy = efficacy, eligibility_time = 0.0, mode = mode)
            results = simulate(
                ModelSpec(
                    BranchingProcess(Poisson(2.0), Exponential(5.0));
                    interventions = [mv]
                ),
                2000; max_cases = 200, rng = StableRNG(seed)
            )
            return containment_probability(results)
        end
        unvaccinated = containment(0.0, LeakyMode(), 101)
        leaky = containment(0.4, LeakyMode(), 102)
        all_or_nothing = containment(0.4, AllOrNothingMode(), 103)
        @test leaky > unvaccinated + 0.3
        @test isapprox(leaky, all_or_nothing; atol = 0.05)
    end
end
