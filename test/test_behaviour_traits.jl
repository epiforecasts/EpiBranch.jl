# Traits the engine asks of an intervention. Each has a fixed default. A
# type with the matching hook declares the trait; building a model throws for
# one that does not.

# Has an `intervention_time` and `reset!` through a forwarding method but
# declares it is not time aware.
struct _ForwardsTime <: EpiBranch.AbstractIntervention
    inner::Isolation
end
EpiBranch.intervention_time(f::_ForwardsTime, ind::Individual) =
    EpiBranch.intervention_time(f.inner, ind)
EpiBranch.time_aware(::_ForwardsTime) = false

struct _OwnsEffect <: EpiBranch.AbstractIntervention end
EpiBranch.capacity_key(::_OwnsEffect) = :_owns_effect
EpiBranch.resolve_individual!(::_OwnsEffect, ind, state) = nothing
EpiBranch.resolves_under_capacity(::_OwnsEffect) = true

# Writes its effect in `resolve_individual!` but declares it is rationed as
# an action.
struct _DeclaredOwnsEffect <: EpiBranch.AbstractIntervention end
EpiBranch.capacity_key(::_DeclaredOwnsEffect) = :_declared_owns_effect
EpiBranch.resolve_individual!(::_DeclaredOwnsEffect, ind, state) = nothing
EpiBranch.resolves_under_capacity(::_DeclaredOwnsEffect) = false

struct _NoRisk <: EpiBranch.AbstractIntervention end
EpiBranch.competing_risk(::_NoRisk, parent, contact, state) = nothing
struct _NoRiskDeclared <: EpiBranch.AbstractIntervention end
EpiBranch.competing_risk(::_NoRiskDeclared, parent, contact, state) = nothing
EpiBranch.has_competing_risk(::_NoRiskDeclared) = false

struct _ProposesNothing <: EpiBranch.AbstractIntervention end
EpiBranch.intervention_actions(::_ProposesNothing, state, candidates) = nothing
EpiBranch.discovers_actions(::_ProposesNothing) = false

# Hooks with no declaration.
struct _UndeclaredTime <: EpiBranch.AbstractIntervention end
EpiBranch.intervention_time(::_UndeclaredTime, ind::Individual) = 1.0
struct _UndeclaredEffect <: EpiBranch.AbstractIntervention end
EpiBranch.capacity_key(::_UndeclaredEffect) = :_undeclared_effect
EpiBranch.resolve_individual!(::_UndeclaredEffect, ind, state) = nothing
struct _UndeclaredActions <: EpiBranch.AbstractIntervention end
EpiBranch.intervention_actions(::_UndeclaredActions, state, candidates) = nothing

# A wrapper with its own `intervention_time`, and an action with its own
# `intervention_actions`, neither declared.
struct _UndeclaredWrapper{I <: EpiBranch.AbstractIntervention} <: EpiBranch.InterventionWrapper
    intervention::I
end
EpiBranch.intervention_time(::_UndeclaredWrapper, ind::Individual) = 1.0
struct _UndeclaredTraceAction <: EpiBranch.TraceAction end
EpiBranch.intervention_actions(::_UndeclaredTraceAction, state, candidates) = nothing

# Declares nothing and defines no hook.
struct _Bare <: EpiBranch.AbstractIntervention end

@testset "Behaviour traits" begin
    iso = Isolation(onset_to_isolation_delay = 1.0, duration = 5.0)

    @testset "time_aware" begin
        @test EpiBranch.time_aware(iso)
        @test !EpiBranch.time_aware(RingVaccination(efficacy = 0.8))
        # Read through a wrapper, in either order.
        @test EpiBranch.time_aware(Scheduled(iso; start_time = 1.0))
        @test !EpiBranch.time_aware(
            Scheduled(RingVaccination(efficacy = 0.8); start_time = 1.0)
        )
        # A declaration overrides the methods, and `Scheduled` follows it.
        fwd = _ForwardsTime(iso)
        @test !EpiBranch.time_aware(fwd)
        @test !Scheduled(fwd; start_time = 1.0).time_aware
        @test Scheduled(iso; start_time = 1.0).time_aware
    end

    @testset "resolves_under_capacity" begin
        @test EpiBranch.resolves_under_capacity(iso)
        @test EpiBranch.resolves_under_capacity(Scheduled(iso; start_time = 1.0))
        @test !EpiBranch.resolves_under_capacity(RingVaccination(efficacy = 0.8))
        @test !EpiBranch.resolves_under_capacity(
            Scheduled(RingVaccination(efficacy = 0.8); start_time = 1.0)
        )
        @test EpiBranch.resolves_under_capacity(_OwnsEffect())
        declared = _DeclaredOwnsEffect()
        @test !EpiBranch.resolves_under_capacity(declared)
        # The wrapper follows the declaration: rationed through the funnel.
        cc = CapacityConstrained(declared; budget_per_period = 1.0)
        @test EpiBranch.proposes_actions(cc)
        @test !EpiBranch.proposes_actions(
            CapacityConstrained(_OwnsEffect(); budget_per_period = 1.0)
        )
        @test !EpiBranch.resolves_under_capacity(Scheduled(declared; start_time = 1.0))
    end

    @testset "discovers_actions and proposes_actions" begin
        @test !EpiBranch.discovers_actions(iso)
        @test !EpiBranch.proposes_actions(iso)
        @test !EpiBranch.discovers_actions(_ProposesNothing())
        @test !EpiBranch.discovers_actions(Scheduled(_ProposesNothing(); start_time = 1.0))
        # Capacity is still offered candidates, so wrapping an intervention
        # with nothing to ration fails when the run starts.
        @test EpiBranch.proposes_actions(
            CapacityConstrained(_ProposesNothing(); budget_per_period = 1.0)
        )
        @test !EpiBranch.discovers_actions(
            CapacityConstrained(_ProposesNothing(); budget_per_period = 1.0)
        )
        # The same holds with the capacity wrapper inside another wrapper.
        cap = CapacityConstrained(_ProposesNothing(); budget_per_period = 1.0)
        @test EpiBranch.proposes_actions(Scheduled(cap; start_time = 1.0))
        @test EpiBranch.proposes_actions(OnRoutes(cap, (:household,)))
        mass = MassVaccination(efficacy = 0.8, coverage = 0.5, eligibility_time = 1.0)
        @test EpiBranch.discovers_actions(mass)
        @test !EpiBranch.proposes_actions(mass)
        ring = RingVaccination(efficacy = 0.8)
        @test EpiBranch.proposes_actions(ring)
        @test EpiBranch.proposes_actions(Scheduled(ring; start_time = 1.0))
        # Isolation is rationed through `resolve_individual!`, not the funnel.
        @test !EpiBranch.proposes_actions(
            CapacityConstrained(iso; budget_per_period = 1.0)
        )
    end

    @testset "has_competing_risk" begin
        @test EpiBranch.has_competing_risk(iso)
        @test EpiBranch.has_competing_risk(_NoRisk())
        @test !EpiBranch.has_competing_risk(_NoRiskDeclared())
        # The default is the conservative answer.
        @test EpiBranch.has_competing_risk(_ProposesNothing())
        # A wrapper forwards the hook, so it answers `true`.
        @test EpiBranch.has_competing_risk(Scheduled(iso; start_time = 1.0))
    end

    @testset "defaults of a type that declares nothing" begin
        bare = _Bare()
        @test !EpiBranch.time_aware(bare)
        @test !EpiBranch.resolves_under_capacity(bare)
        @test !EpiBranch.discovers_actions(bare)
        @test !EpiBranch.proposes_actions(bare)
        @test EpiBranch.has_competing_risk(bare)
        # The same through a wrapper.
        sched = Scheduled(bare; start_time = 1.0)
        @test !EpiBranch.time_aware(sched)
        @test !sched.time_aware
    end

    @testset "a hook without its declaration is an error at build" begin
        bp = BranchingProcess(Poisson(1.5), Exponential(5.0))
        for (iv, trait) in (
                (_UndeclaredTime(), "time_aware"),
                (_UndeclaredEffect(), "resolves_under_capacity"),
                (_UndeclaredActions(), "discovers_actions"),
            )
            err = try
                ModelSpec(bp; interventions = [iv])
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("EpiBranch.$trait(::$(nameof(typeof(iv))))", err.msg)
            # Found under a wrapper too.
            @test_throws ArgumentError ModelSpec(
                bp; interventions = [Scheduled(iv; start_time = 1.0)]
            )
        end
        @test_throws ArgumentError ModelSpec(
            bp; interventions = [_UndeclaredWrapper(_Bare())]
        )
        @test_throws ArgumentError ModelSpec(
            bp; interventions = [
                ContactTracing(
                    probability = 0.5, isolation_to_trace_delay = Exponential(1.0),
                    action = _UndeclaredTraceAction()
                ),
            ]
        )
        # Declared types build.
        @test ModelSpec(
            bp; interventions = [_OwnsEffect(), _ProposesNothing(), _ForwardsTime(Isolation(onset_to_isolation_delay = 1.0, duration = 5.0))]
        ) isa ModelSpec
    end
end
