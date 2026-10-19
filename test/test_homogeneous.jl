using Random

# A leaky vaccine written from outside the package: the whole population is
# dosed at `from_time`, and every exposure from then on is blocked with
# probability `efficacy`. The entire intervention is one competing risk, which
# is the shape #232's seam has to carry.
struct LeakyVaccine <: EpiBranch.AbstractIntervention
    efficacy::Float64
    from_time::Float64
end
function EpiBranch.competing_risk(v::LeakyVaccine, parent, contact, state)
    return Risk(event_time = v.from_time, block_probability = v.efficacy)
end

# Protects every contact exposed from `from_time` on, reading the exposure from
# the contact's `infection_time` as a risk may on the generation engine.
struct ProtectFromExposure <: EpiBranch.AbstractIntervention
    from_time::Float64
end
function EpiBranch.competing_risk(p::ProtectFromExposure, parent, contact, state)
    return Risk(
        block_probability = (rng, parent, contact, state) -> contact.infection_time >=
            p.from_time ? 1.0 : 0.0
    )
end

# Interventions written outside the package that act only through hooks the
# generation engine calls, and one that also traces contacts.
struct DoseNewContacts <: EpiBranch.AbstractIntervention end
EpiBranch.apply_post_transmission!(::DoseNewContacts, state, new_contacts) = nothing
struct KeepFringeActive <: EpiBranch.AbstractIntervention end
EpiBranch.keep_active(::KeepFringeActive, state, targets, is_new) = ()
# The same, with its arguments typed, as the style guide asks for: the check
# that names an unhonoured intervention has to find this method too.
struct DoseNewContactsTyped <: EpiBranch.AbstractIntervention end
function EpiBranch.apply_post_transmission!(
        ::DoseNewContactsTyped,
        state::EpiBranch.SimulationState, new_contacts::Vector{<:Individual}
    )
    return nothing
end
struct DoseAndTrace <: EpiBranch.AbstractIntervention end
EpiBranch.apply_post_transmission!(::DoseAndTrace, state, new_contacts) = nothing
EpiBranch.traces_contacts(::DoseAndTrace) = true
EpiBranch.trace_contacts!(::DoseAndTrace, state, infector, contacts) = nothing
# Written for both engines: `apply_post_transmission!` for the generation
# engine, `on_infection_settled!` for the continuous-time race. Neither traces,
# so this is honoured through the settled hook alone.
struct DoseAndSettle <: EpiBranch.AbstractIntervention end
EpiBranch.apply_post_transmission!(::DoseAndSettle, state, new_contacts) = nothing
EpiBranch.on_infection_settled!(::DoseAndSettle, ind, state, rng) = nothing
# Written for both engines like `DoseAndSettle`, but also grows a ring through
# `keep_active`. That hook's own counterpart is `trace_contacts!`, so the
# settled hook must not paper over a ring the pool cannot grow.
struct DoseSettleAndTrace <: EpiBranch.AbstractIntervention end
EpiBranch.apply_post_transmission!(::DoseSettleAndTrace, state, new_contacts) = nothing
EpiBranch.on_infection_settled!(::DoseSettleAndTrace, ind, state, rng) = nothing
EpiBranch.keep_active(::DoseSettleAndTrace, state, targets, is_new) = ()
EpiBranch.traces_contacts(::DoseSettleAndTrace) = true
EpiBranch.trace_contacts!(::DoseSettleAndTrace, state, infector, contacts) = nothing
# Reaches its targets only through the action-producer path: `continuous_actions`
# opts in, and it defines neither `apply_post_transmission!` nor `keep_active`
# of its own, so `_generation_hooks` has nothing to name.
struct DoseThroughActions <: EpiBranch.AbstractIntervention end
EpiBranch.continuous_actions(::DoseThroughActions) = true
# The same action-producer path as `DoseThroughActions`, but also defines
# `apply_post_transmission!` of its own. `_generation_hooks` has something to
# name here, but naming only that would miss `intervention_actions`, which
# this branch always reaches.
struct DoseThroughActionsAndSettle <: EpiBranch.AbstractIntervention end
EpiBranch.continuous_actions(::DoseThroughActionsAndSettle) = true
EpiBranch.apply_post_transmission!(::DoseThroughActionsAndSettle, state, new_contacts) =
    nothing
# Discovers its own actions, like `MassVaccination`, but without declaring
# `continuous_actions`: the race never delivers what it discovers, and
# `_generation_hooks` has nothing to name either, since it defines neither
# `apply_post_transmission!` nor `keep_active` of its own.
struct DiscoverWithoutContinuous <: EpiBranch.AbstractIntervention end
function EpiBranch.intervention_actions(::DiscoverWithoutContinuous, state, candidates)
    return EpiBranch.InterventionAction[]
end

# The same leaky vaccine with its arguments typed, as the style guide asks for.
struct LeakyVaccineTyped <: EpiBranch.AbstractIntervention
    efficacy::Float64
    from_time::Float64
end
function EpiBranch.competing_risk(
        v::LeakyVaccineTyped, parent::Individual,
        contact::Individual, state::EpiBranch.SimulationState
    )
    return Risk(event_time = v.from_time, block_probability = v.efficacy)
end

struct ResolveInfectiousness <: EpiBranch.AbstractIntervention
    include_seeds::Bool
end
function EpiBranch.resolve_individual!(iv::ResolveInfectiousness, ind, state)
    if iv.include_seeds || ind.parent_id != 0
        ind.infectiousness = isodd(ind.id) ? 0.0 : 1.0
    end
    return nothing
end

struct RouteSelectedBlock <: EpiBranch.AbstractIntervention
    route::Symbol
end
function EpiBranch.risk_applies(iv::RouteSelectedBlock, route)
    return route !== nothing && route.name == iv.route
end
function EpiBranch.competing_risk(::RouteSelectedBlock, parent, contact, state)
    return Risk(block_probability = 1.0)
end
function EpiBranch.resolve_individual!(::RouteSelectedBlock, ind, state)
    return (ind.state[:route_block_resolved] = true)
end

# A terminal clinical transition written outside the package (#373):
# declares `terminal_target` alongside `is_terminal`/`terminal_event`, so
# the `until`-coverage check can see it the same way it sees `Death`/`Recovery`.
struct LostToFollowUp <: EpiBranch.AbstractClinicalTransition end
EpiBranch.is_terminal(::LostToFollowUp) = true
EpiBranch.terminal_target(::LostToFollowUp) = :lost
function EpiBranch.resolve_individual!(::LostToFollowUp, ind, state)
    ind.state[:lost_time] = 3.0
    return nothing
end
function EpiBranch.terminal_event(::LostToFollowUp, individual)
    t = get(individual.state, :lost_time, Inf)
    return isfinite(t) ? (t, :lost) : nothing
end

@testset "HomogeneousProcess (Sellke fixed pool)" begin
    @testset "Pool risk selection preserves other intervention hooks" begin
        process = HomogeneousProcess(; transmission_rate = 20.0, population_size = 20)
        progression = [
            Transition(
                :recovered; from = :infection,
                delay = 10.0, terminal = true
            ),
        ]
        run(ivs) = simulate(
            ModelSpec(process; progression, interventions = ivs);
            rng = StableRNG(1)
        )
        baseline = run(AbstractIntervention[])
        for iv in (
                RouteSelectedBlock(:household),
                Scheduled(
                    Scheduled(
                        RouteSelectedBlock(:household);
                        start_time = 0.0
                    ); start_time = 0.0
                ),
            )
            selected = run([iv])
            @test selected.cumulative_cases == baseline.cumulative_cases == 20
            @test getfield.(selected.individuals, :infection_time) ==
                getfield.(baseline.individuals, :infection_time)
            @test all(
                get(ind.state, :route_block_resolved, false)
                    for ind in selected.individuals
            )
        end
        @test run([RouteSelectedBlock(:transmission)]).cumulative_cases == 1
    end

    @testset "Infector selection uses traits set during resolution" begin
        for include_seeds in (true, false)
            model = ModelSpec(
                HomogeneousProcess(; transmission_rate = 4.0, population_size = 100);
                progression = [
                    Transition(
                        :recovered; from = :infection,
                        delay = 10.0, terminal = true
                    ),
                ],
                interventions = [ResolveInfectiousness(include_seeds)]
            )
            state = simulate(model; n_initial = 20, rng = MersenneTwister(1))
            parents = [ind.parent_id for ind in state.individuals if ind.parent_id != 0]
            @test !isempty(parents)
            @test any(ind.infectiousness == 0 for ind in state.individuals)
            @test all(state.individuals[id].infectiousness > 0 for id in parents)
        end
    end

    @testset "A scheduled block expires after blocked contacts" begin
        model = ModelSpec(
            HomogeneousProcess(; transmission_rate = 20.0, population_size = 20);
            progression = [
                Transition(
                    :recovered; from = :infection,
                    delay = 10.0, terminal = true
                ),
            ],
            interventions = [Scheduled(LeakyVaccine(1.0, 0.0); end_time = 1.0)]
        )
        state = simulate(model; n_initial = 2, rng = StableRNG(1))
        secondary = filter(ind -> ind.parent_id != 0, state.individuals)
        @test !isempty(secondary)
        @test all(ind.infection_time > 1.0 for ind in secondary)
    end

    @testset "deterministic final size (major outbreaks)" begin
        # With β = 2 and mean infectious period 1, R0 = β·E[T] = 2; the
        # deterministic attack rate solves z = 1 - exp(-R0·z), z ≈ 0.7968.
        # Conditioning on major outbreaks, the mean should match.
        N = 3000
        m = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = [
                Transition(
                    :recovered; from = :infection,
                    delay = Exponential(1.0), terminal = true
                ),
            ]
        )
        finals = [
            simulate(m; rng = StableRNG(s), n_initial = 5).cumulative_cases
                for s in 1:40
        ]
        z = 0.7968
        major = filter(x -> x > 0.3 * N, finals)
        @test length(major) > 20                 # most seeds take off at R0 = 2
        @test all(x -> x <= N, finals)            # final size never exceeds the pool
        @test isapprox(mean(major) / N, z; atol = 0.03)
    end

    @testset "sub-critical outbreaks stay small" begin
        # β = 0.5 with mean infectious period 1 gives R0 = 0.5 < 1.
        N = 2000
        m = ModelSpec(
            HomogeneousProcess(; transmission_rate = 0.5, population_size = N);
            progression = [
                Transition(
                    :recovered; from = :infection, delay = 1.0,
                    terminal = true
                ),
            ]
        )
        finals = [
            simulate(m; rng = StableRNG(s), n_initial = 1).cumulative_cases
                for s in 1:100
        ]
        @test mean(finals) < 0.1 * N
    end

    @testset "saturation infects the whole pool" begin
        N = 200
        m = ModelSpec(
            HomogeneousProcess(; transmission_rate = 100.0, population_size = N);
            progression = [
                Transition(
                    :recovered; from = :infection, delay = 1.0,
                    terminal = true
                ),
            ]
        )
        finals = [
            simulate(m; rng = StableRNG(s), n_initial = 1).cumulative_cases
                for s in 1:20
        ]
        @test mean(finals) > 0.98 * N
        @test all(x -> x <= N, finals)
    end

    @testset "depletion is real (final size ≤ N)" begin
        N = 300
        m = ModelSpec(
            HomogeneousProcess(; transmission_rate = 5.0, population_size = N);
            progression = [
                Transition(
                    :recovered; from = :infection, delay = 1.0,
                    terminal = true
                ),
            ]
        )
        for s in 1:20
            @test simulate(m; rng = StableRNG(s), n_initial = 2).cumulative_cases <= N
        end
    end

    @testset "isolation shortens the outbreak" begin
        N = 1000
        base = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = [
                Transition(
                    :recovered; from = :infection,
                    delay = Exponential(1.0), terminal = true
                ),
            ]
        )
        # An early isolation transition closes the infectious window (`:isolated`
        # is in the default `until`), curtailing spread.
        iso = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = [
                Transition(
                    :recovered; from = :infection,
                    delay = Exponential(1.0), terminal = true
                ),
                Transition(:isolated; from = :infection, delay = (rng, ind) -> 0.1),
            ]
        )
        base_mean = mean(
            simulate(base; rng = StableRNG(s), n_initial = 3).cumulative_cases
                for s in 1:30
        )
        iso_mean = mean(
            simulate(iso; rng = StableRNG(s), n_initial = 3).cumulative_cases
                for s in 1:30
        )
        @test iso_mean < base_mean
    end

    @testset "Isolation intervention shortens the outbreak" begin
        # The Isolation *intervention* (not a Transition) must act on the pool:
        # its resolve_individual! runs in the Sellke loop and its isolation time
        # closes the infectious window. Onset comes from a progression transition
        # so it is anchored on the real infection time.
        N = 1000
        prog = [
            Transition(:onset; from = :infection, delay = 0.1),
            Transition(
                :recovered; from = :infection,
                delay = Exponential(1.0), terminal = true
            ),
        ]
        base = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = prog
        )
        iso = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = prog,
            interventions = [Isolation(onset_to_isolation_delay = Exponential(0.1), duration = Inf)]
        )

        base_mean = mean(
            simulate(base; rng = StableRNG(s), n_initial = 3).cumulative_cases
                for s in 1:20
        )
        iso_mean = mean(
            simulate(iso; rng = StableRNG(s), n_initial = 3).cumulative_cases
                for s in 1:20
        )
        # Fast isolation pushes R below 1: the outbreak is curtailed.
        @test iso_mean < 0.1 * base_mean

        # A case's isolation time is actually written (the hook ran).
        state = simulate(iso; rng = StableRNG(1), n_initial = 3)
        @test any(isfinite(EpiBranch.isolation_time(ind)) for ind in state.individuals)
    end

    @testset "Unhonoured intervention warns rather than silently ignoring" begin
        prog = [
            Transition(
                :recovered; from = :infection,
                delay = Exponential(1.0), terminal = true
            ),
        ]
        # A mass-action pool has no pairwise contact structure for tracing to act
        # along, so tracing stays unhonoured there.
        ct = ModelSpec(
            HomogeneousProcess(; transmission_rate = 1.5, population_size = 200);
            progression = prog,
            interventions = [
                ContactTracing(
                    probability = 0.5, isolation_to_trace_delay = Exponential(1.0),
                    action = Quarantine(duration = Inf)
                ),
            ]
        )
        @test_logs (:warn, r"does not honour"i) match_mode = :any simulate(
            ct; rng = StableRNG(1), n_initial = 2
        )

        # `MassVaccination` declares `acts_on_population`, so the pool offers
        # it to every member before the race starts, whether or not it names
        # contacts; it warns about nothing and doses the population.
        mass = ModelSpec(
            HomogeneousProcess(; transmission_rate = 1.5, population_size = 200);
            progression = prog,
            interventions = [MassVaccination(efficacy = 0.8, eligibility_time = 0.0)]
        )
        @test EpiBranch._sellke_honours(mass.process, mass.interventions[1])
        dosed = @test_logs min_level = Base.CoreLogging.Warn simulate(
            mass; rng = StableRNG(1), n_initial = 2
        )
        @test any(is_vaccinated, dosed.individuals)

        # `GroupVaccination` declares `needs_contacts = false`, so it finds a
        # case's group through its `group_key` on the pool, which names no
        # contacts, exactly as it would through a traced contact elsewhere.
        onsets_for_group = clinical_presentation(
            incubation_period = Dirac(0.0), prob_asymptomatic = 0.0
        )
        gv = GroupVaccination(
            efficacy = 0.8, eligibility = OnSymptomOnset(),
            group_key = :group, dose_delay = 0.0
        )
        grouped = ModelSpec(
            HomogeneousProcess(; transmission_rate = 1.5, population_size = 200);
            progression = prog,
            attributes = [onsets_for_group, (rng, ind) -> (ind.state[:group] = 1)],
            interventions = [gv]
        )
        @test EpiBranch._sellke_honours(grouped.process, gv)
        dosed = @test_logs min_level = Base.CoreLogging.Warn simulate(
            grouped; rng = StableRNG(1), n_initial = 2
        )
        @test any(is_vaccinated, dosed.individuals)

        # An intervention from outside the package that acts only through a hook
        # the pool never calls is reported without declaring anything, and so is
        # one that traces, since the pool names no contacts to trace along.
        prog_pool = HomogeneousProcess(; transmission_rate = 1.5, population_size = 200)
        for (iv, name) in (
                (DoseNewContacts(), r"DoseNewContacts"),
                (DoseNewContactsTyped(), r"DoseNewContactsTyped"),
                (KeepFringeActive(), r"KeepFringeActive"), (DoseAndTrace(), r"DoseAndTrace"),
            )
            @test !EpiBranch._sellke_honours(prog_pool, iv)
            @test_logs (:warn, name) match_mode = :any simulate(
                ModelSpec(prog_pool; progression = prog, interventions = [iv]);
                rng = StableRNG(1), n_initial = 2
            )
        end
        # A method of its own for `on_infection_settled!` is `apply_post_transmission!`'s
        # continuous-time counterpart, so this is honoured and warns about nothing,
        # even though the pool still never calls `apply_post_transmission!` itself.
        @test EpiBranch._sellke_honours(prog_pool, DoseAndSettle())
        @test_logs min_level = Base.CoreLogging.Warn simulate(
            ModelSpec(prog_pool; progression = prog, interventions = [DoseAndSettle()]);
            rng = StableRNG(1), n_initial = 2
        )
        # The settled hook covers only `apply_post_transmission!`; `keep_active`
        # still needs its own counterpart, `trace_contacts!`, honoured, and the
        # pool cannot do that, so this stays unhonoured despite the settled hook.
        @test !EpiBranch._sellke_honours(prog_pool, DoseSettleAndTrace())
        @test_logs (:warn, r"DoseSettleAndTrace"i) match_mode = :any simulate(
            ModelSpec(prog_pool; progression = prog, interventions = [DoseSettleAndTrace()]);
            rng = StableRNG(1), n_initial = 2
        )
        # Naming is scoped to the hook the race actually skips: the settled
        # hook still reaches `apply_post_transmission!`, so only `keep_active`
        # is named rather than the whole intervention.
        @test EpiBranch._unhonoured_hooks(prog_pool, DoseSettleAndTrace()) == [:keep_active]
        # Wrapping in `CapacityConstrained` must not lose the settled-hook
        # exemption: the wrapper only overrides `apply_post_transmission!`
        # itself, forwarding `continuous_actions`, `on_infection_settled!` and
        # `keep_active` unchanged, so the wrapped intervention is still
        # honoured through its own settled hook.
        cc = CapacityConstrained(DoseAndSettle(); budget_per_period = 1.0)
        @test EpiBranch._sellke_honours(prog_pool, cc)
        @test EpiBranch._unhonoured_hooks(prog_pool, cc) == Symbol[]
        # Wrapping `DoseSettleAndTrace` keeps it unhonoured for the right
        # reason: its `keep_active`, forwarded unchanged, still needs
        # `trace_contacts!`, which the pool cannot supply.
        cc_trace = CapacityConstrained(DoseSettleAndTrace(); budget_per_period = 1.0)
        @test !EpiBranch._sellke_honours(prog_pool, cc_trace)
        @test EpiBranch._unhonoured_hooks(prog_pool, cc_trace) == [:keep_active]
        # Nesting `Scheduled` inside `CapacityConstrained` must not be mistaken
        # for an action producer: `Scheduled`'s own `intervention_actions`
        # method is only the blanket forwarding every `InterventionWrapper`
        # inherits, so discovery must unwrap past it to `Isolation`, which has
        # no such method at all.
        cc_scheduled = CapacityConstrained(
            Scheduled(
                Isolation(onset_to_isolation_delay = 1.0, duration = 5.0);
                start_time = 1.0
            );
            budget_per_period = 1.0
        )
        @test !EpiBranch._discovers_actions(cc_scheduled)
        # An intervention reached only through the action-producer path is
        # still named, rather than reported with an empty hook list, since the
        # pool cannot name contacts for `intervention_actions` to act on.
        @test !EpiBranch._sellke_honours(prog_pool, DoseThroughActions())
        @test EpiBranch._unhonoured_hooks(prog_pool, DoseThroughActions()) ==
            [:intervention_actions]
        @test_logs (:warn, r"intervention_actions"i) match_mode = :any simulate(
            ModelSpec(prog_pool; progression = prog, interventions = [DoseThroughActions()]);
            rng = StableRNG(1), n_initial = 2
        )
        # Defining `apply_post_transmission!` alongside the action-producer
        # path must not crowd `intervention_actions` out of the naming: the
        # pool still cannot name contacts for it to act on, whatever else
        # `_generation_hooks` finds.
        @test !EpiBranch._sellke_honours(prog_pool, DoseThroughActionsAndSettle())
        @test EpiBranch._unhonoured_hooks(prog_pool, DoseThroughActionsAndSettle()) ==
            [:intervention_actions, :apply_post_transmission!]
        @test_logs (:warn, r"intervention_actions"i) match_mode = :any simulate(
            ModelSpec(
                prog_pool; progression = prog,
                interventions = [DoseThroughActionsAndSettle()]
            );
            rng = StableRNG(1), n_initial = 2
        )
        # An intervention that discovers its own actions without declaring
        # `continuous_actions`, like `MassVaccination`, is unhonoured for the
        # same reason and must be named the same way, not reported with an
        # empty hook list.
        @test !EpiBranch._sellke_honours(prog_pool, DiscoverWithoutContinuous())
        @test EpiBranch._unhonoured_hooks(prog_pool, DiscoverWithoutContinuous()) ==
            [:intervention_actions]
        @test_logs (:warn, r"intervention_actions"i) match_mode = :any simulate(
            ModelSpec(
                prog_pool; progression = prog,
                interventions = [DiscoverWithoutContinuous()]
            );
            rng = StableRNG(1), n_initial = 2
        )
        # The package's own interventions that the pool honours warn about nothing.
        onsets = clinical_presentation(
            incubation_period = LogNormal(-1.0, 0.3),
            prob_asymptomatic = 0.0
        )
        honoured = [
            Isolation(onset_to_isolation_delay = Exponential(0.5), duration = Inf),
            Scheduled(
                Isolation(
                    onset_to_isolation_delay = Exponential(0.5),
                    post_isolation_transmission = 0.5, duration = Inf
                );
                start_time = 1.0
            ),
        ]
        @test all(iv -> EpiBranch._sellke_honours(prog_pool, iv), honoured)
        @test_logs min_level = Base.CoreLogging.Warn simulate(
            ModelSpec(
                prog_pool; progression = prog, interventions = honoured,
                attributes = onsets
            );
            rng = StableRNG(1), n_initial = 2
        )

        # Leaky isolation is honoured: the residual transmission it leaves is a
        # per-contact block, which the pool resolves on each contact it delivers.
        # It warns about nothing, and it bites in proportion to the residual.
        pool = HomogeneousProcess(; transmission_rate = 2.0, population_size = 500)
        onsets = clinical_presentation(
            incubation_period = LogNormal(-1.0, 0.3),
            prob_asymptomatic = 0.0
        )
        leaky(residual) = [
            Isolation(
                onset_to_isolation_delay = Exponential(0.5),
                post_isolation_transmission = residual, duration = Inf
            ),
        ]
        mean_size(ivs) = sum(
            simulate(
                ModelSpec(
                    pool; progression = prog, interventions = ivs,
                    attributes = onsets
                );
                rng = StableRNG(s), n_initial = 5
            ).cumulative_cases
                for s in 1:20
        ) / 20
        base_mean = mean_size(AbstractIntervention[])
        @test EpiBranch._sellke_honours(pool, leaky(0.9)[1])
        # Residual 1.0 is no isolation at all; 0.9 barely reduces transmission;
        # 0.2 cuts most of it. (Isolation draws its own delays either way, so
        # residual 1.0 gives the same process off a different rng stream.)
        @test isapprox(mean_size(leaky(1.0)), base_mean; rtol = 0.1)
        @test 0.8 * base_mean < mean_size(leaky(0.9)) < base_mean
        @test mean_size(leaky(0.2)) < 0.5 * base_mean
    end

    @testset "per-individual susceptibility and infectiousness apply on the pool" begin
        # Both multipliers thin the hazard on the pool: susceptibility scales
        # the pressure a susceptible absorbs and infectiousness the weight an
        # infective adds to the force, so susceptibility m matches beta scaled
        # by m rather than the generation engine's per-contact block.
        N = 500
        prog = [
            Transition(
                :recovered; from = :infection,
                delay = Exponential(1.0), terminal = true
            ),
        ]
        pool = HomogeneousProcess(; transmission_rate = 2.0, population_size = N)
        mean_size(attrs) = sum(
            simulate(
                ModelSpec(pool; progression = prog, attributes = attrs);
                rng = StableRNG(s), n_initial = 3
            ).cumulative_cases
                for s in 1:15
        ) / 15

        full = mean_size(transmission_traits(susceptibility = 1.0))
        @test mean_size(transmission_traits(susceptibility = 0.5)) < full
        @test mean_size(transmission_traits(susceptibility = 0.2)) <
            mean_size(transmission_traits(susceptibility = 0.5))
        # Susceptibility 0 blocks every contact, so only the seeds are infected.
        none = simulate(
            ModelSpec(
                pool; progression = prog,
                attributes = transmission_traits(susceptibility = 0.0)
            );
            rng = StableRNG(1), n_initial = 3
        )
        @test none.cumulative_cases == 3

        # Infectiousness acts on the other side of the same pair.
        @test mean_size(transmission_traits(infectiousness = 0.5)) < full
        silent = simulate(
            ModelSpec(
                pool; progression = prog,
                attributes = transmission_traits(infectiousness = 0.0)
            );
            rng = StableRNG(1), n_initial = 3
        )
        @test silent.cumulative_cases == 3

        # Infectiousness is each infective's weight in the force, so a case
        # contributing none is never named as a parent, and the weighted count
        # returns to exactly zero when the last of them recovers: fractional
        # weights added and subtracted in different orders would otherwise leave
        # a residual force and infect the rest at absurd times.
        half_silent = ModelSpec(
            pool; progression = prog,
            attributes = transmission_traits(
                infectiousness = (rng, ind) -> rand(rng) < 0.5 ? 0.0 : 1.0
            )
        )
        for s in 1:5
            st = simulate(half_silent; rng = StableRNG(s), n_initial = 20)
            parents = [
                ind.parent_id
                    for ind in st.individuals
                    if is_infected(ind) && ind.parent_id > 0
            ]
            @test !isempty(parents)
            @test all(p -> st.individuals[p].infectiousness > 0, parents)
        end
        varied = ModelSpec(
            pool; progression = prog,
            attributes = transmission_traits(infectiousness = Uniform(0, 1))
        )
        @test maximum(
            maximum(
                ind.infection_time
                    for ind in simulate(varied; rng = StableRNG(s), n_initial = 5).individuals
                    if is_infected(ind)
            ) for s in 1:20
        ) < 1.0e3

        # The traits scale every pair's rate, so a susceptibility of s is the
        # same process as a transmission rate scaled by s. Compare attack rates
        # away from R0 = 1, where 40 runs cannot pin the mean down.
        double_beta = HomogeneousProcess(; transmission_rate = 4.0, population_size = N)
        scaled = sum(
            simulate(
                ModelSpec(pool; progression = prog);
                rng = StableRNG(s), n_initial = 3
            ).cumulative_cases
                for s in 1:40
        ) / 40
        blocked = sum(
            simulate(
                ModelSpec(
                    double_beta; progression = prog,
                    attributes = transmission_traits(susceptibility = 0.5)
                );
                rng = StableRNG(s), n_initial = 3
            ).cumulative_cases
                for s in 1:40
        ) / 40
        @test isapprox(blocked, scaled; rtol = 0.15)
    end

    @testset "a user-defined leaky vaccination bites on the pool" begin
        # The seam is open from outside: an intervention returning a time-tagged
        # Risk changes the pool's results without touching the package.
        N = 600
        prog = [
            Transition(
                :recovered; from = :infection,
                delay = Exponential(1.0), terminal = true
            ),
        ]
        pool = HomogeneousProcess(; transmission_rate = 2.0, population_size = N)
        mean_size(ivs) = sum(
            simulate(
                ModelSpec(pool; progression = prog, interventions = ivs);
                rng = StableRNG(s), n_initial = 3
            ).cumulative_cases
                for s in 1:15
        ) / 15

        base = mean_size(AbstractIntervention[])
        # The same risk with typed arguments is found and applied the same way.
        @test mean_size([LeakyVaccineTyped(0.5, 0.0)]) ==
            mean_size([LeakyVaccine(0.5, 0.0)])
        @test mean_size([LeakyVaccine(0.0, 0.0)]) == base
        @test mean_size([LeakyVaccine(0.5, 0.0)]) < 0.9 * base
        @test mean_size([LeakyVaccine(1.0, 0.0)]) == 3         # only the seeds
        # A dose that arrives after the outbreak has burnt out changes nothing.
        @test mean_size([LeakyVaccine(1.0, 1000.0)]) == base
        # A risk that reads the exposure from the contact's `infection_time` sees
        # the time of the contact being resolved, so protection from just after
        # the seeds blocks every contact.
        @test mean_size([ProtectFromExposure(1.0e-9)]) == 3
    end

    @testset "a pool that can never finish is refused, not looped" begin
        # With no removal transition the infectious window never closes, so the
        # force of infection never decays; with every contact certainly blocked
        # by a risk there is no end to reach. The pool says so rather than
        # spinning. (A susceptibility of zero is no such case: it is an infinite
        # threshold, so no contact is ever delivered and the run ends at once.)
        pool40 = HomogeneousProcess(; transmission_rate = 2.0, population_size = 40)
        m = ModelSpec(pool40; interventions = [LeakyVaccine(1.0, 0.0)])
        @test_throws ArgumentError simulate(m; rng = StableRNG(1), n_initial = 2)
        @test simulate(
            ModelSpec(
                pool40;
                attributes = transmission_traits(susceptibility = 0.0)
            );
            rng = StableRNG(1), n_initial = 2
        ).cumulative_cases == 2

        # Even a risk that sometimes permits infection has no finite contact
        # budget here. Use susceptibility for this static proportional effect.
        rare = ModelSpec(pool40; interventions = [LeakyVaccine(1.0 - 1.0e-4, 0.0)])
        @test_throws ArgumentError simulate(rare; rng = StableRNG(1), n_initial = 2)
        scaled = ModelSpec(
            pool40;
            attributes = transmission_traits(susceptibility = 1.0e-4)
        )
        @test simulate(scaled; rng = StableRNG(1), n_initial = 2).cumulative_cases == 40

        # A removal transition is all it takes: the outbreak ends at the seeds.
        with_removal = ModelSpec(
            pool40;
            progression = [
                Transition(
                    :recovered; from = :infection,
                    delay = Exponential(1.0), terminal = true
                ),
            ],
            interventions = [LeakyVaccine(1.0, 0.0)]
        )
        @test simulate(with_removal; rng = StableRNG(1), n_initial = 2).cumulative_cases ==
            2
    end

    @testset "a non-exclusive terminal gate refuses with a pointed hint" begin
        # Two terminal transitions gated independently at p and 1 - p leave
        # about p(1 - p) of cases with neither outcome, so their infectious
        # window never closes, and ten seeds make one such case all but
        # certain; with a blocking risk in play, the rejection sampler cannot
        # find a way out either — and should say why rather than just naming
        # the symptom.
        pool40 = HomogeneousProcess(; transmission_rate = 2.0, population_size = 40)
        non_exclusive = [
            Transition(
                :recovered; from = :infection, delay = Exponential(1.0),
                probability = 0.36, terminal = true
            ),
            Transition(
                :died; from = :infection, delay = Exponential(1.0),
                probability = 0.64, terminal = true
            ),
        ]
        m = @test_logs (:warn, r"gated below.*probability 1") match_mode = :any ModelSpec(
            pool40; progression = non_exclusive, interventions = [LeakyVaccine(1.0, 0.0)]
        )
        err = try
            simulate(m; rng = StableRNG(3), n_initial = 10)
            nothing
        catch caught
            caught
        end
        @test err isa ArgumentError
        @test occursin("infectious window never closes", err.msg)
        @test occursin("exclusive_probabilities", err.msg)
    end

    @testset "Scheduled interventions gate on the isolation's own time on the pool" begin
        # The loop exposes each case's infection time as the running clock,
        # but `Scheduled(Isolation; start_time)` checks the isolation's own
        # time against it, so start_time curtails when the policy takes
        # effect without needing a case's own infection to fall inside it.
        N = 1000
        prog = [
            Transition(:onset; from = :infection, delay = 0.1),
            Transition(
                :recovered; from = :infection,
                delay = Exponential(1.0), terminal = true
            ),
        ]
        mk(start_time) = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = prog,
            interventions = [
                Scheduled(
                    Isolation(onset_to_isolation_delay = Exponential(0.1), duration = Inf); start_time
                ),
            ]
        )
        mean_cc(m) = mean(
            simulate(m; rng = StableRNG(s), n_initial = 3).cumulative_cases
                for s in 1:15
        )

        base = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = prog
        )
        # Active from t = 0: curtails hard. Start far past the outbreak: barely.
        @test mean_cc(mk(0.0)) < 0.1 * mean_cc(base)
        @test mean_cc(mk(1000.0)) > 0.5 * mean_cc(base)

        # Mid start_time gates on the isolation's own time, not the case's
        # infection time: a case infected just before it can still be
        # isolated once its own isolation falls after it.
        st = simulate(mk(3.0); rng = StableRNG(1), n_initial = 10)
        @test all(
            ind -> !is_isolated(ind) || isolation_time(ind) >= 3.0, st.individuals
        )
        @test any(ind -> ind.infection_time < 3.0 && is_isolated(ind), st.individuals)
        @test any(ind -> ind.infection_time >= 3.0 && is_isolated(ind), st.individuals)

        # A Scheduled wrapping an unhonoured intervention still warns.
        sched_ct = ModelSpec(
            HomogeneousProcess(; transmission_rate = 1.5, population_size = 200);
            progression = prog,
            interventions = [
                Scheduled(
                    ContactTracing(
                        probability = 0.5, isolation_to_trace_delay = Exponential(1.0),
                        action = Quarantine(duration = Inf)
                    ); start_time = 5.0
                ),
            ]
        )
        @test_logs (:warn, r"does not honour"i) match_mode = :any simulate(
            sched_ct; rng = StableRNG(1), n_initial = 2
        )
    end

    @testset "removal before infectious onset never infects" begin
        # A latent period opens the window at :infectious, but isolation comes
        # first (close_t <= open_t). Such a case is never infectious: it must be
        # skipped rather than pop a close event against an id never made infectious.
        N = 100
        progression = [
            Transition(:infectious; from = :infection, delay = (rng, ind) -> 5.0),
            Transition(
                :recovered; from = :infectious,
                delay = (rng, ind) -> 1.0, terminal = true
            ),
            Transition(:isolated; from = :infection, delay = (rng, ind) -> 0.1),
        ]
        # The infectious window is derived to open at :infectious (a latent period
        # produces it).
        @test EpiBranch.infectious_from(progression) === :infectious
        m = ModelSpec(
            HomogeneousProcess(; transmission_rate = 5.0, population_size = N);
            progression = progression
        )
        # No index case reaches :infectious, so no secondary transmission occurs
        # and the run completes with only the seeds infected.
        state = simulate(m; rng = StableRNG(1), n_initial = 5)
        @test state.cumulative_cases == 5
    end

    @testset "onset is measured from each case's own infection time" begin
        # Pool members are created, and their incubation periods drawn, before
        # the pool sets their infection times. Isolation depends on onset, so
        # onset must be counted from the time each case was infected.
        clinical = clinical_presentation(incubation_period = LogNormal(1.0, 0.3))
        iso = Isolation(
            onset_to_isolation_delay = Exponential(1.0),
            test_sensitivity = 1.0, duration = Inf
        )
        m = ModelSpec(
            HomogeneousProcess(; transmission_rate = 1.0, population_size = 500);
            progression = [
                Transition(
                    :recovered; from = :infection, delay = 10.0,
                    terminal = true
                ),
            ],
            attributes = clinical, interventions = [iso]
        )
        state = simulate(m; n_initial = 1, rng = StableRNG(3))
        secondary = [
            ind
                for ind in state.individuals
                if is_infected(ind) && ind.parent_id != 0
        ]
        @test !isempty(secondary)
        @test all(onset_time(ind) >= ind.infection_time for ind in secondary)
        @test all(
            onset_time(ind) - ind.infection_time ≈ ind.state[:incubation_period]
                for ind in secondary
        )
        @test all(isolation_time(ind) >= onset_time(ind) for ind in secondary)
    end

    @testset "line list and timing" begin
        progression = [
            Transition(:infectious; from = :infection, delay = Exponential(1.0)),
            Transition(
                :recovered; from = :infectious, delay = Exponential(2.0),
                terminal = true
            ),
        ]
        @test EpiBranch.infectious_from(progression) === :infectious
        m = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 500);
            progression = progression
        )
        state = simulate(m; rng = StableRNG(1), n_initial = 5)
        ll = linelist(state)
        @test size(ll, 1) == state.cumulative_cases
        @test EpiBranch._timetype(state) === Float64
        @test :date_infectious in propertynames(ll)
        @test :date_recovered in propertynames(ll)
    end

    @testset "contacts() lists every infector-infectee pair the race settled" begin
        m = ModelSpec(HomogeneousProcess(; transmission_rate = 2.0, population_size = 200))
        state = simulate(m; rng = StableRNG(1), n_initial = 5)
        df = contacts(state)
        @test size(df, 1) == count(!=(0), linelist(state).parent_id)
        @test all(df.infected)
        expected = Set((ind.parent_id, ind.id) for ind in state.individuals if ind.parent_id != 0)
        @test Set(zip(df.from, df.to)) == expected
    end

    @testset "transmission_rate is the transmission parameter" begin
        # β is the per-infective rate, stored and used directly (no R0 map).
        p = HomogeneousProcess(; transmission_rate = 2.0, population_size = 100)
        @test p.transmission_rate == 2.0
        # `transmission_rate` is required.
        @test_throws UndefKeywordError HomogeneousProcess(; population_size = 10)
        # β must be finite and non-negative; negative, infinite and NaN rates
        # are rejected before they reach the Sellke hazard.
        @test_throws ArgumentError HomogeneousProcess(;
            transmission_rate = -1.0,
            population_size = 10
        )
        @test_throws ArgumentError HomogeneousProcess(;
            transmission_rate = Inf,
            population_size = 10
        )
        @test_throws ArgumentError HomogeneousProcess(;
            transmission_rate = NaN,
            population_size = 10
        )
        # β = 0 (no transmission) is a valid degenerate model.
        @test HomogeneousProcess(;
            transmission_rate = 0.0,
            population_size = 10
        ).transmission_rate == 0.0
        # A non-positive population is rejected at construction.
        @test_throws ArgumentError HomogeneousProcess(;
            transmission_rate = 2.0,
            population_size = 0
        )
    end

    # A small helper that runs structured mass action on the race directly: tag
    # a real attribute (`:age_band`) on a fixed population of size N via
    # `band_of`, name it as the mixing attribute with `mixing_by = (:age_band,)`,
    # supply the pair rate between bands, and return the state.
    function _run_mixing(
            N, band_of, rate; n_initial = 5, rng,
            interventions = AbstractIntervention[], setup! = ind -> nothing
        )
        process = HomogeneousProcess(; transmission_rate = 1.0, population_size = N)
        prog = [
            Transition(:onset; from = :infection, delay = 0.1),
            Transition(
                :recovered; from = :infection, delay = Exponential(1.0),
                terminal = true
            ),
        ]
        state = EpiBranch.new_state(process, prog, NoAttributes(), rng)
        EpiBranch.add_individuals!(
            state, N, interventions;
            setup = (ind, i) -> begin
                ind.state[:age_band] = band_of(ind)
                setup!(ind)
            end
        )
        members = collect(1:N)
        EpiBranch._sellke_race!(
            state, members, rng;
            seed! = (best, _, r) -> (best[randperm(r, N)[1:n_initial]] .= 0),
            targets = EpiBranch._MassAction(state, members; mixing_by = (:age_band,), rate),
            from = :infection, until = (:recovered, EpiBranch.INTERVENTION_REMOVAL),
            interventions
        )
        return state
    end
    band(ind) = ind.state[:age_band]
    infected(state) = filter(is_infected, state.individuals)

    @testset "two-band uniform matrix reduces to one population" begin
        # Two bands with uniform contact behave as a single population of size N:
        # every pair meets at β/N regardless of band, so the major-outbreak
        # attack rate matches the homogeneous law (z ≈ 0.7968 at R0 = 2, since
        # β = 2 and mean infectious period = 1).
        N = 3000
        β = 2.0
        band_of = ind -> (ind.id <= N ÷ 2 ? 1 : 2)
        finals = [
            length(infected(_run_mixing(N, band_of, (_, _) -> β / N; rng = StableRNG(s))))
                for s in 1:40
        ]
        major = filter(x -> x > 0.3 * N, finals)
        @test length(major) > 20
        @test all(x -> x <= N, finals)
        @test isapprox(mean(major) / N, 0.7968; atol = 0.03)
    end

    @testset "asymmetric mixing orders attack rates" begin
        # A 2×2 contact matrix keyed by band, where band 1 mixes far more than
        # band 2: M[b, h] is the rate at which one band-h infective contacts the
        # half of the population in band b, so each pair meets at M[b, h]/half.
        # The high-contact band should suffer a strictly higher attack rate than
        # the low-contact band, over replicates.
        N = 2000
        half = N ÷ 2
        M = [3.0 0.5; 0.5 0.5]
        band_of = ind -> (ind.id <= half ? 1 : 2)
        rate = (from, to) -> M[to[1], from[1]] / half
        ar1 = Float64[]
        ar2 = Float64[]
        for s in 1:40
            bands = band.(infected(_run_mixing(N, band_of, rate; n_initial = 10, rng = StableRNG(s))))
            n1 = count(==(1), bands)
            n2 = count(==(2), bands)
            # Keep major outbreaks only, so the ordering is about who is hit hardest.
            (n1 + n2) > 0.3 * N || continue
            push!(ar1, n1 / half)
            push!(ar2, n2 / half)
        end
        @test length(ar1) > 20
        @test mean(ar1) > mean(ar2)
    end

    @testset "structured mixing attributes each contact to its infector" begin
        # Two bands that never mix. Every contact comes from a member of the
        # infector's own band, so every case's parent shares its band, band 1's
        # attack rate is the SIR final size at R0 = 2 whatever band 2's traits
        # are, and a risk that reads the infector, such as a leaky isolation,
        # acts within each band.
        N = 2000
        half = N ÷ 2
        band_of = ind -> (ind.id <= half ? 1 : 2)
        rate = (from, to) -> from == to ? 2.0 / half : 0.0
        band1_attack(state) = count(ind -> band(ind) == 1, infected(state)) / half
        major(ars) = mean(filter(>(0.2), ars))
        run_bands(s; kwargs...) = _run_mixing(
            N, band_of, rate; n_initial = 20, rng = StableRNG(s), kwargs...
        )

        state = run_bands(1)
        @test all(infected(state)) do ind
            ind.parent_id == 0 || band(state.individuals[ind.parent_id]) == band(ind)
        end

        base = major([band1_attack(run_bands(s)) for s in 1:30])
        immune2 = major(
            [band1_attack(run_bands(s; setup! = ind -> band(ind) == 2 && (ind.susceptibility = 0.0))) for s in 1:30]
        )
        silent2 = major(
            [band1_attack(run_bands(s; setup! = ind -> band(ind) == 2 && (ind.infectiousness = 0.0))) for s in 1:30]
        )
        @test isapprox(base, 0.7968; atol = 0.03)
        @test isapprox(immune2, 0.7968; atol = 0.03)
        @test isapprox(silent2, 0.7968; atol = 0.03)

        leaky = Isolation(
            onset_to_isolation_delay = 0.1, post_isolation_transmission = 0.5,
            duration = Inf
        )
        @test major([band1_attack(run_bands(s; interventions = [leaky])) for s in 1:30]) < base
    end

    @testset "susceptibility thins mass action exactly" begin
        # Proposals are sampled at the largest susceptibility and thinned to each
        # member's own, so half the population at 0.4 must give the same process
        # as two bands whose pair rates are scaled by the contacted band's 0.4.
        N = 400
        β = 2.5
        prog = [
            Transition(
                :recovered; from = :infection, delay = Exponential(1.0),
                terminal = true
            ),
        ]
        lower_half = transmission_traits(
            susceptibility = (rng, ind) -> ind.id <= N ÷ 2 ? 1.0 : 0.4
        )
        thinned = [
            simulate(
                ModelSpec(
                    HomogeneousProcess(; transmission_rate = β, population_size = N);
                    progression = prog, attributes = lower_half
                );
                n_initial = 5, rng = StableRNG(s)
            ).cumulative_cases
                for s in 1:1500
        ]
        band_of = ind -> (ind.id <= N ÷ 2 ? 1 : 2)
        rate = (from, to) -> β / N * (to[1] == 1 ? 1.0 : 0.4)
        banded = [
            length(infected(_run_mixing(N, band_of, rate; rng = StableRNG(10_000 + s))))
                for s in 1:1500
        ]
        se = sqrt(var(thinned) / 1500 + var(banded) / 1500)
        @test abs(mean(thinned) - mean(banded)) < 4 * se
    end

    @testset "conditioned simulation and show" begin
        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        spec = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 500);
            progression = prog
        )
        # `condition` retries until the final size falls in the range
        state = simulate(spec; condition = 100:500, n_initial = 5, rng = StableRNG(1))
        @test state.cumulative_cases in 100:500

        # show renders the β form
        @test occursin(
            "β=",
            repr(HomogeneousProcess(; transmission_rate = 2.0, population_size = 10))
        )
    end

    @testset "max_time ends the pool at that time" begin
        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        spec = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 500);
            progression = prog
        )
        full = simulate(spec; n_initial = 3, rng = StableRNG(4))
        # Honoured, so no warning; the cut run matches the full run up to 2.
        cut = @test_logs simulate(spec; n_initial = 3, max_time = 2.0, rng = StableRNG(4))
        by(state, t) = [is_infected(i) && i.infection_time <= t for i in state.individuals]
        @test is_infected.(cut.individuals) == by(full, 2.0)
        @test count(is_infected, cut.individuals) < count(is_infected, full.individuals)
        # The pool ran to extinction on its own, but the cut run was stopped
        # with candidates still pending, so only the full run is extinct.
        @test is_extinct(full)
        @test !is_extinct(cut)

        # A window that never closes is sampled only up to the time limit, and the
        # contacts it leaves beyond it still make the cut run unfinished. Once
        # everyone is infected nothing is left to reach, and the run is extinct.
        si = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 500);
            progression = Transition[]
        )
        si_cut = simulate(si; n_initial = 3, max_time = 1.0, rng = StableRNG(4))
        @test !is_extinct(si_cut)
        @test all(i -> !is_infected(i) || i.infection_time <= 1.0, si_cut.individuals)
        si_all = simulate(si; n_initial = 3, max_time = 100.0, rng = StableRNG(4))
        @test count(is_infected, si_all.individuals) == 500
        @test is_extinct(si_all)
    end

    @testset "a cut group still reinfectable is not reported extinct" begin
        # A cut group whose sole member has already settled is waiting on
        # their own `susceptible_again_time`, not on susceptibility (which
        # stays > 0 throughout), so it is still reachable once the
        # progression reinfects and their draw gave them a finite wait.
        m = EpiBranch._MassAction(
            [[1]], Dict(1 => 1), reshape([1.0], 1, 1), [1.0], reshape([true], 1, 1)
        )
        waiting = Individual(; id = 1)
        waiting.state[:susceptible_again_time] = 5.0
        race(ind, reinfects) = (;
            processed = [true], pos = Dict(1 => 1),
            state = (; individuals = Dict(1 => ind)), reinfects
        )
        @test EpiBranch._pending_beyond(m, race(waiting, true))
        @test !EpiBranch._pending_beyond(m, race(waiting, false))
        # A settled member whose own draw never let them wane stays immune
        # for good (`susceptible_again_time == Inf`), so they are not
        # pending even though the progression reinfects in general.
        @test !EpiBranch._pending_beyond(m, race(Individual(; id = 1), true))
    end

    @testset "termination controls on the fixed pool" begin
        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        spec = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 200);
            progression = prog
        )
        # An explicit, non-default max_cases now caps the pool, with no
        # warning that it had no effect.
        capped = @test_logs simulate(spec; n_initial = 3, max_cases = 50, rng = StableRNG(1))
        @test capped.cumulative_cases == 50
        @test !is_extinct(capped)
        # No termination keyword set → no warning, and the run still goes to
        # extinction or `max_time` as before.
        @test_logs simulate(spec; n_initial = 3, rng = StableRNG(1))
        # Extinction and MaxTime are both honoured, so neither warns.
        @test_logs simulate(
            spec; n_initial = 3,
            stopping_rules = [Extinction(), MaxTime(2.0)], rng = StableRNG(1)
        )
        # The trait itself: whether the max_cases/max_generations keyword
        # *defaults* apply out of the box. The pool leaves them off (an
        # explicit value still works, as above); the generation engine
        # applies them either way.
        @test !EpiBranch._honours_termination_controls(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 10)
        )
        @test EpiBranch._honours_termination_controls(
            BranchingProcess(Poisson(1.5), Exponential(2.0))
        )
    end

    @testset "uncovered terminal state warns" begin
        process = HomogeneousProcess(; transmission_rate = 2.0, population_size = 200)
        # `:censored` is a terminal transition `until` does not list, so a case
        # reaching it would never have its window closed: composing warns.
        censored = [
            Transition(:infectious; from = :infection, delay = 1.0),
            Transition(:recovered; from = :infectious, delay = 1.0, terminal = true),
            Transition(:censored; from = :infection, delay = 5.0, terminal = true),
        ]
        @test_logs (:warn, r":censored") match_mode = :any ModelSpec(
            process; progression = censored
        )
        # Listing the extra terminal state in `until` silences the warning.
        covered = HomogeneousProcess(;
            transmission_rate = 2.0, population_size = 200,
            until = (:recovered, :died, :isolated, :censored)
        )
        @test_logs ModelSpec(covered; progression = censored)
        # No custom terminal transition beyond the defaults → no warning.
        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        @test_logs ModelSpec(process; progression = prog)
        # A window that only opens once a case reaches a terminal state (e.g.
        # a post-mortem pool with `from = :died`) is not warned about for
        # missing that same state from `until`.
        died = [
            Transition(:recovered; from = :infection, delay = 1.0, terminal = true),
            Transition(:died; from = :infection, delay = 2.0, terminal = true),
        ]
        after_death = HomogeneousProcess(;
            transmission_rate = 2.0,
            population_size = 200, from = :died, until = (:recovered,)
        )
        @test_logs ModelSpec(after_death; progression = died)
        # `Death`/`Recovery` are only covered by the *default* `until`; narrow
        # it and the check must still see them, not exempt them permanently.
        narrowed = HomogeneousProcess(;
            transmission_rate = 2.0,
            population_size = 200, until = (:isolated,)
        )
        @test_logs (:warn, r":recovered") match_mode = :any ModelSpec(
            narrowed; progression = [Recovery(delay = Exponential(5.0))]
        )
        @test_logs (:warn, r":died") match_mode = :any ModelSpec(
            narrowed; progression = [Death(delay = Exponential(5.0), probability = 0.3)]
        )
        # The default `until` covers both without a warning.
        @test_logs ModelSpec(
            process;
            progression = [
                Recovery(delay = Exponential(5.0)),
                Death(delay = Exponential(5.0), probability = 0.3),
            ]
        )
        # A terminal transition defined outside the package (`LostToFollowUp`)
        # is checked the same way once it declares `terminal_target`: missing
        # from `until` warns, and adding it silences the warning.
        lost = [
            Transition(:recovered; from = :infection, delay = 1.0, terminal = true),
            LostToFollowUp(),
        ]
        @test_logs (:warn, r":lost") match_mode = :any ModelSpec(
            process; progression = lost
        )
        covered_lost = HomogeneousProcess(;
            transmission_rate = 2.0, population_size = 200,
            until = (:recovered, :died, :isolated, :lost)
        )
        @test_logs ModelSpec(covered_lost; progression = lost)
    end

    @testset "A settled member is offered again once immunity has waned" begin
        model = ModelSpec(
            HomogeneousProcess(; transmission_rate = 5.0, population_size = 30);
            progression = [
                Transition(:recovered; from = :infection, delay = 1.0),
                Transition(:susceptible_again; from = :recovered, delay = 0.5),
            ]
        )
        state = simulate(model; n_initial = 1, max_time = 50.0, rng = MersenneTwister(1))
        @test any(ind -> !isempty(ind.episodes), state.individuals)
    end
end
