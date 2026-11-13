# Traits the engine asks of an intervention, whose defaults read the methods a
# type defines and which a type can declare instead.

# Has an `intervention_time` and `reset!` through a forwarding method the
# default does not see as its own.
struct _ForwardsTime <: EpiBranch.AbstractIntervention
    inner::Isolation
end
EpiBranch.intervention_time(f::_ForwardsTime, ind::Individual) =
    EpiBranch.intervention_time(f.inner, ind)
EpiBranch.time_aware(::_ForwardsTime) = false

# Writes its effect in `resolve_individual!` but declares it is rationed as
# an action.
struct _OwnsEffect <: EpiBranch.AbstractIntervention end
EpiBranch.capacity_key(::_OwnsEffect) = :_owns_effect
EpiBranch.resolve_individual!(::_OwnsEffect, ind, state) = nothing

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
        @test !EpiBranch.has_competing_risk(_ProposesNothing())
        # A wrapper forwards the hook, so it answers `true`.
        @test EpiBranch.has_competing_risk(Scheduled(iso; start_time = 1.0))
    end
end
