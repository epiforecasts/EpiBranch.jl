# Route windows: the unit that lets one case carry several transmission routes,
# each opening and closing on different states of its natural history.

# Blocks every transmission out of one named infector, whatever the time: the
# smallest per-contact competing risk there is, and one an outside user could
# have written without touching the package.
struct BlockFrom <: EpiBranch.AbstractIntervention
    id::Int
end
function EpiBranch.competing_risk(b::BlockFrom, parent, contact, state)
    parent.id == b.id ? Risk(block_probability = 1.0) : nothing
end

@testset "Route windows" begin
    @testset "construction and show" begin
        w = RouteWindow(:community; from = :infectious, until = (:recovered,),
            kernel = Exponential(3.0))
        @test w.name === :community
        @test w.from === :infectious
        @test w.until == (:recovered,)
        @test w.reach === :community          # defaults to the route name
        @test w.contacts_from === :infection  # standing relationships by default
        @test !occursin("contacts_from", repr(w))
        f = RouteWindow(:funeral; from = :died, kernel = Exponential(1.0),
            contacts_from = :died)
        @test f.contacts_from === :died
        @test occursin("contacts_from=:died", repr(f))
        @test occursin("RouteWindow(:community", repr(w))
        @test occursin("from=:infectious", repr(w))

        # defaults: start derived from the progression, never censored, reach
        # follows the name
        d = RouteWindow(:household; kernel = Exponential(1.0), reach = [[2], [1]])
        @test d.from === nothing
        @test occursin("from=nothing", repr(d))
        @test d.until == ()
        @test d.reach == [[2], [1]]
    end

    @testset "traceability is a validated probability" begin
        # every contact is nameable unless the route says otherwise
        w = RouteWindow(:household; kernel = Exponential(1.0))
        @test w.traceable === 1.0
        @test !occursin("traceable", repr(w))

        c = RouteWindow(:community; kernel = Exponential(1.0), traceable = 0.2)
        @test c.traceable === 0.2
        @test occursin("traceable=0.2", repr(c))

        # a Boolean is the all-or-nothing case, and integers are probabilities too
        @test RouteWindow(:a; kernel = nothing, traceable = false).traceable === 0.0
        @test RouteWindow(:a; kernel = nothing, traceable = true).traceable === 1.0
        @test RouteWindow(:a; kernel = nothing, traceable = 0).traceable === 0.0

        @test_throws ArgumentError RouteWindow(:a; kernel = nothing, traceable = -0.1)
        @test_throws ArgumentError RouteWindow(:a; kernel = nothing, traceable = 1.5)
        @test_throws ArgumentError RouteWindow(:a; kernel = nothing, traceable = NaN)
        # the positional form validates too, so no construction path skips it
        @test_throws ArgumentError RouteWindow(:a, nothing, (), nothing, :a,
            :infection, 2.0)
    end

    @testset "opening and closing read the state-time convention" begin
        ind = Individual(id = 1)
        ind.infection_time = 5.0
        ind.state[:infectious_time] = 7.0
        ind.state[:recovered_time] = 20.0
        ind.state[:died_time] = 14.0

        # from :infection opens at the infection time itself, even when the
        # individual has a later infectious time
        @test window_open(ind, RouteWindow(:c; from = :infection, kernel = nothing)) == 5.0
        # the default derives the start: infectious time when there is one...
        @test window_open(ind, RouteWindow(:c; kernel = nothing)) == 7.0
        # ...and the infection time when there is not
        no_latent = Individual(id = 2)
        no_latent.infection_time = 3.0
        @test window_open(no_latent, RouteWindow(:c; kernel = nothing)) == 3.0
        # any other state opens at that state's `<state>_time`
        @test window_open(ind, RouteWindow(:c; from = :infectious, kernel = nothing)) == 7.0
        # a state never reached never opens the route, so it contributes nothing
        @test window_open(ind, RouteWindow(:f; from = :buried, kernel = nothing)) == Inf

        # closing takes the earliest of the `until` states
        @test window_close(ind, RouteWindow(:c; until = (:recovered,), kernel = nothing)) ==
              20.0
        @test window_close(ind,
            RouteWindow(:c; until = (:recovered, :died), kernel = nothing)) == 14.0
        # no `until` means the route is never censored
        @test window_close(ind, RouteWindow(:c; kernel = nothing)) == Inf
    end

    @testset "isolation reaches a window only through INTERVENTION_REMOVAL" begin
        ind = Individual(id = 1)
        set_isolated!(ind, 4.0)
        @test is_isolated(ind)
        @test isolation_time(ind) == 4.0
        # `:isolated_time` is left to a `Transition(:isolated, …)`, so a window
        # listing `:isolated` is not closed by the intervention, leaky or not
        @test !haskey(ind.state, :isolated_time)
        @test window_close(ind, RouteWindow(:c; until = (:isolated,), kernel = nothing)) ==
              Inf

        clear_isolated!(ind)
        @test !is_isolated(ind)
        @test isolation_time(ind) == Inf
    end

    @testset "INTERVENTION_REMOVAL is opt-in per route" begin
        # Two routes differing only in whether they list the intervention
        # removal. This is the whole mechanism behind a control measure cutting
        # one route and leaving another: the household route survives isolation,
        # the community route does not.
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        ind = Individual(id = 1)
        ind.infection_time = 0.0
        ind.state[:recovered_time] = 10.0
        set_isolated!(ind, 3.0)

        household = RouteWindow(:household; until = (:recovered,), kernel = nothing)
        community = RouteWindow(:community;
            until = (:recovered, EpiBranch.INTERVENTION_REMOVAL), kernel = nothing)

        @test EpiBranch._route_close(ind, household, (iso,)) == 10.0   # not cut
        @test EpiBranch._route_close(ind, community, (iso,)) == 3.0    # cut at isolation

        # Leaky isolation removes no one, so even the opted-in route runs on.
        leaky = Isolation(onset_to_isolation_delay = Exponential(1.0),
            post_isolation_transmission = 0.3)
        @test EpiBranch._route_close(ind, community, (leaky,)) == 10.0

        # With no interventions composed, opting in changes nothing.
        @test EpiBranch._route_close(ind, community, ()) == 10.0

        # The public accessor gives the close the race uses once it is handed
        # the same interventions.
        @test window_close(ind, community, (iso,)) == 3.0
        @test window_close(ind, community) == 10.0
        @test window_close(ind, household, (iso,)) == 10.0
    end

    @testset "the race censors each route on its own window" begin
        # Node 1 is seeded at time 0 and reaches node 2 on a community route and
        # node 3 on a household route, each contact 5 days after infection. It
        # recovers at 10 and, with onset at 1 and immediate isolation, isolates
        # at 1.
        REM = EpiBranch.INTERVENTION_REMOVAL
        prog = [Transition(:recovered; from = :infection, delay = 10.0,
            terminal = true)]
        onsets = clinical_presentation(incubation_period = Dirac(1.0),
            prob_asymptomatic = 0.0)
        function infected_after(routes, interventions)
            rng = StableRNG(1)
            state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(1.0)),
                prog, onsets, rng)
            EpiBranch.add_individuals!(state, 3, interventions)
            EpiBranch._sellke_race!(state, [1, 2, 3], rng; routes = routes,
                interventions = interventions, seed! = (
                    best, members, r) -> (best[1] = 0.0))
            return [get(ind.state, :infected, false) for ind in state.individuals]
        end
        edge(to) = (inf, st) -> inf == 1 &&
                                !get(st.individuals[to].state, :infected, false) ?
                                ((to, Dirac(5.0)),) : ()

        community = RouteWindow(:community; until = (:recovered, REM), kernel = Dirac(5.0))
        household = RouteWindow(:household; until = (:recovered,), kernel = Dirac(5.0))
        routes = ((community, edge(2)), (household, edge(3)))
        isolate = [Isolation(onset_to_isolation_delay = Dirac(0.0))]

        @test infected_after(routes, AbstractIntervention[]) == [true, true, true]
        # isolation at 1 cuts the community contact at 5; the household one runs on
        @test infected_after(routes, isolate) == [true, false, true]

        # a route whose `from` state is never reached contributes no contacts
        funeral = RouteWindow(:funeral; from = :died, until = (:buried,),
            kernel = Dirac(5.0))
        @test infected_after(((funeral, edge(2)), (household, edge(3))),
            AbstractIntervention[]) == [true, false, true]
    end

    @testset "the race traces a settled case's contacts" begin
        # Node 1 infects node 2 on a household route at 5; node 2 infects node 3
        # half a day later on a community route. Every case has onset one day
        # after infection and isolates at onset, so node 2 would isolate at 6,
        # too late to stop the contact at 5.5. Tracing node 1's contacts at its
        # isolation (time 1) quarantines node 2 before it is infected, which
        # closes node 2's community window and spares node 3.
        REM = EpiBranch.INTERVENTION_REMOVAL
        prog = [Transition(:onset; from = :infection, delay = 1.0),
            Transition(:recovered; from = :infection, delay = 20.0, terminal = true)]
        function race(interventions; contacts = nothing)
            rng = StableRNG(1)
            state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(1.0)),
                prog, EpiBranch.NoAttributes(), rng)
            EpiBranch.add_individuals!(state, 3, interventions)
            function edge(from, to, t)
                (inf, st) -> inf == from &&
                             !get(st.individuals[to].state, :infected, false) ?
                             ((to, Dirac(t)),) : ()
            end
            routes = (
                (RouteWindow(:household; until = (:recovered,), kernel = Dirac(5.0)),
                    edge(1, 2, 5.0)),
                (RouteWindow(:community; until = (:recovered, REM), kernel = Dirac(0.5)),
                    edge(2, 3, 0.5)))
            EpiBranch._sellke_race!(state, [1, 2, 3], rng; routes, interventions,
                contacts, seed! = (best, members, r) -> (best[1] = 0.0))
            return state
        end
        infected(state) = [get(ind.state, :infected, false) for ind in state.individuals]
        iso = Isolation(onset_to_isolation_delay = Dirac(0.0))
        ct = ContactTracing(probability = 1.0, isolation_to_trace_delay = Dirac(0.0))
        node1_contacts = (inf, st) -> inf == 1 ? (2,) : ()

        @test infected(race([iso])) == [true, true, true]
        traced = race([iso, ct]; contacts = node1_contacts)
        @test infected(traced) == [true, true, false]
        @test is_quarantined(traced.individuals[2])
        # with no contacts supplied there is nothing to trace along
        @test infected(race([iso, ct])) == [true, true, true]

        # Ring vaccination doses along the trace, so it is honoured exactly where
        # tracing is: on a model that can name a case's contacts.
        nameless = BranchingProcess(Poisson(1.0), Exponential(1.0))
        @test !EpiBranch.supplies_contacts(nameless)
        @test !EpiBranch._sellke_honours(nameless, RingVaccination(efficacy = 0.9))
        # A rollout that doses each newly created contact has nobody to dose
        # when no contacts are created, wherever it runs.
        @test !EpiBranch._sellke_honours(nameless,
            MassVaccination(efficacy = 0.9, eligibility_time = 0.0))
        # A leaky isolation is a per-contact block, and the race now resolves it.
        @test EpiBranch._sellke_honours(nameless,
            Isolation(onset_to_isolation_delay = Dirac(1.0),
                post_isolation_transmission = 0.5))
    end

    @testset "the race resolves competing risks on each proposal" begin
        # Nodes 1 and 2 are both seeded at time 0 and both reach node 3: node 1
        # at time 2, node 2 at time 5. Unblocked, the earlier proposal wins.
        prog = [Transition(:recovered; from = :infection, delay = 20.0, terminal = true)]
        function race(interventions; attributes = EpiBranch.NoAttributes())
            rng = StableRNG(1)
            state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(1.0)),
                prog, attributes, rng)
            EpiBranch.add_individuals!(state, 3, interventions)
            targets = function (inf, st)
                (inf == 3 || get(st.individuals[3].state, :infected, false)) && return ()
                return ((3, Dirac(inf == 1 ? 2.0 : 5.0)),)
            end
            EpiBranch._sellke_race!(state, [1, 2, 3], rng; targets,
                from = :infection, until = (:recovered,), interventions,
                seed! = (best, members, r) -> (best[1] = 0.0; best[2] = 0.0))
            return state
        end

        plain = race(AbstractIntervention[])
        @test plain.individuals[3].parent_id == 1
        @test plain.individuals[3].infection_time == 2.0

        # Blocking node 1's proposal declines that one transmission and nothing
        # else: node 3 stays susceptible and node 2 infects it at 5, later than
        # it would otherwise have been infected.
        blocked = race([BlockFrom(1)])
        @test is_infected(blocked.individuals[3])
        @test blocked.individuals[3].parent_id == 2
        @test blocked.individuals[3].infection_time == 5.0

        # A susceptibility of 0 blocks every proposal, so node 3 is never reached.
        immune = race(AbstractIntervention[];
            attributes = transmission_traits(susceptibility = 0.0))
        @test !is_infected(immune.individuals[3])

        # Traits at their defaults contribute no risk, draw nothing from the rng,
        # and leave the race stream exactly as it was.
        neutral = race(AbstractIntervention[];
            attributes = transmission_traits(susceptibility = 1.0, infectiousness = 1.0))
        @test neutral.individuals[3].parent_id == 1
        @test neutral.individuals[3].infection_time == 2.0
    end

    @testset "an intervention's risks reach only the routes it can cut" begin
        # Node 1 is seeded at 0 and reaches node 2 on a community route and node
        # 3 on a household route, each at 5. An intervention blocking everything
        # node 1 proposes takes out the community contact; the household route
        # never opted into intervention removal, so it runs on.
        REM = EpiBranch.INTERVENTION_REMOVAL
        prog = [Transition(:recovered; from = :infection, delay = 10.0, terminal = true)]
        function infected_after(interventions)
            rng = StableRNG(1)
            state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(1.0)),
                prog, EpiBranch.NoAttributes(), rng)
            EpiBranch.add_individuals!(state, 3, interventions)
            edge(to) = (inf, st) -> inf == 1 &&
                                    !get(st.individuals[to].state, :infected, false) ?
                                    ((to, Dirac(5.0)),) : ()
            routes = (
                (RouteWindow(:community; until = (:recovered, REM), kernel = Dirac(5.0)),
                    edge(2)),
                (RouteWindow(:household; until = (:recovered,), kernel = Dirac(5.0)),
                    edge(3)))
            EpiBranch._sellke_race!(state, [1, 2, 3], rng; routes, interventions,
                seed! = (best, members, r) -> (best[1] = 0.0))
            return [get(ind.state, :infected, false) for ind in state.individuals]
        end
        @test infected_after(AbstractIntervention[]) == [true, true, true]
        @test infected_after([BlockFrom(1)]) == [true, false, true]
    end

    @testset "the race takes routes or the shorthand, not both" begin
        state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(1.0)),
            AbstractClinicalTransition[], nothing, StableRNG(1))
        seed! = (best, members, r) -> nothing
        w = RouteWindow(:c; until = (:recovered,), kernel = Dirac(1.0))
        targets = (inf, st) -> ()
        @test_throws ArgumentError EpiBranch._sellke_race!(state, Int[], StableRNG(1);
            seed!, routes = ((w, targets),), until = (:recovered,))
        @test_throws ArgumentError EpiBranch._sellke_race!(state, Int[], StableRNG(1);
            seed!, routes = ((w, targets),), targets)
        @test_throws ArgumentError EpiBranch._sellke_race!(state, Int[], StableRNG(1);
            seed!)
    end
end
