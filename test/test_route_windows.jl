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
# Its block stands in for taking the infector out of circulation.
EpiBranch.risk_scope(::BlockFrom) = EpiBranch.RemovalRoutes()

# Blocks every transmission into one named contact: a protection that belongs to
# the person, left at the default scope.
struct ProtectTo <: EpiBranch.AbstractIntervention
    id::Int
end
function EpiBranch.competing_risk(p::ProtectTo, parent, contact, state)
    contact.id == p.id ? Risk(block_probability = 1.0) : nothing
end

# Traces every contact a case reaches at a fixed time per tracer, keeping the
# earliest, as `ContactTracing` records `:trace_time`.
struct TraceAtFixedTimes <: EpiBranch.AbstractIntervention
    times::Dict{Int, Float64}
end
EpiBranch.traces_contacts(::TraceAtFixedTimes) = true
function EpiBranch.trace_contacts!(tr::TraceAtFixedTimes, state, infector, contacts)
    for c in contacts
        c.state[:traced] = true
        c.state[:trace_time] = min(get(c.state, :trace_time, Inf), tr.times[infector.id])
    end
end

# A per-contact risk with a constant block probability, of the shape a user
# writes: it thins the pair's contact process rather than scaling its kernel.
struct FlatBlock <: EpiBranch.AbstractIntervention
    p::Float64
end
function EpiBranch.competing_risk(b::FlatBlock, parent, contact, state)
    parent === contact ? nothing : Risk(block_probability = b.p)
end

# The same risk with its arguments typed, as the style guide asks for. The race
# has to find this method as readily as the untyped one above.
struct TypedBlock <: EpiBranch.AbstractIntervention
    p::Float64
end
function EpiBranch.competing_risk(b::TypedBlock, parent::Individual, contact::Individual,
        state::EpiBranch.SimulationState)
    parent === contact ? nothing : Risk(block_probability = b.p)
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
        # Group vaccination doses whole groups as the generation engine creates
        # their members, so it has nobody to dose either.
        @test !EpiBranch._sellke_honours(nameless, GroupVaccination(efficacy = 0.9))
        # A leaky isolation is a per-contact block, and the race now resolves it.
        @test EpiBranch._sellke_honours(nameless,
            Isolation(onset_to_isolation_delay = Dirac(1.0),
                post_isolation_transmission = 0.5))
    end

    @testset "a per-contact risk thins the pair's hazard" begin
        # Two members, one infectious from time 0 for two days, meeting at an
        # Exponential(1) contact interval. A multiplier m on that pair — a
        # susceptibility, an intervention's block of 1 - m, or both — scales the
        # hazard, so the other member is infected with probability 1 - exp(-2m).
        # Halving the transmission probability instead would give 0.43 at
        # m = 0.5, where thinning the hazard gives 0.63.
        function secondary(kernel, sus, ivs, period, seed)
            prog = [Transition(:recovered; from = :infection, delay = period,
                terminal = true)]
            rng = StableRNG(seed)
            state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(1.0)),
                prog, transmission_traits(susceptibility = sus), rng)
            EpiBranch.add_individuals!(state, 2, ivs)
            EpiBranch._sellke_race!(state, [1, 2], rng; from = :infection,
                until = (:recovered,), interventions = ivs,
                targets = (inf, st) -> inf == 1 && !is_infected(st.individuals[2]) ?
                                       ((2, kernel),) : (),
                seed! = (best, members, r) -> (best[1] = 0.0))
            return is_infected(state.individuals[2])
        end
        function share(kernel, sus, ivs = AbstractIntervention[]; period = 2.0)
            count(secondary(kernel, sus, ivs, period, seed) for seed in 1:4000) / 4000
        end

        # The trait folds into the contact-interval draw.
        @test isapprox(share(Exponential(1.0), 1.0), 1 - exp(-2.0); atol = 0.025)
        for m in (0.5, 0.25)
            @test isapprox(share(Exponential(1.0), m), 1 - exp(-2m); atol = 0.025)
        end
        @test share(Exponential(1.0), 0.5) > 0.5 * (1 - exp(-2.0)) + 0.03
        # An intervention's risk is resolved contact by contact instead, and the
        # pair goes on meeting after a blocked one, which thins the same hazard
        # by the same factor. The two compose: half of a half is a quarter.
        @test isapprox(share(Exponential(1.0), 1.0, [FlatBlock(0.5)]), 1 - exp(-1.0);
            atol = 0.025)
        @test isapprox(share(Exponential(1.0), 0.5, [FlatBlock(0.5)]), 1 - exp(-0.5);
            atol = 0.025)
        # A risk written with typed arguments is found and applied the same way.
        @test isapprox(share(Exponential(1.0), 1.0, [TypedBlock(0.5)]), 1 - exp(-1.0);
            atol = 0.025)

        # A small multiplier puts the contact far out in the tail of the kernel's
        # survival, where a linear argument loses it: subtracting from 1 rounds to
        # 1, and `U^(1/m)` underflows to 0 below about 1e-324. Both are taken in
        # logs instead. Ten and a hundred days of an Exponential(0.1) contact
        # interval, at a rate of 10, so transmission is nearly certain, or two in
        # three — never 0.84, 0.51 or 0.31.
        @test isapprox(share(Exponential(0.1), 0.05; period = 10.0), 1 - exp(-5.0);
            atol = 0.02)
        @test isapprox(share(Exponential(0.1), 1.0, [FlatBlock(0.99)]; period = 10.0),
            1 - exp(-1.0); atol = 0.025)
        @test isapprox(share(Exponential(0.1), 0.001; period = 100.0), 1 - exp(-1.0);
            atol = 0.025)
        @test isapprox(share(Exponential(0.1), 1.0, [FlatBlock(0.999)]; period = 100.0),
            1 - exp(-1.0); atol = 0.025)

        # A kernel whose own inverse survival is only defined over part of the
        # unit interval — `Gamma` below shape 1, whose inverse raises a
        # `DomainError` on a small enough probability — is drawn from in logs for
        # the same reason, so a small multiplier runs rather than throwing.
        gamma_law(kernel, m, period) = 1 - ccdf(kernel, period)^m
        for (kernel, m) in ((Gamma(0.3, 2.0), 0.01), (Gamma(0.7, 1.0), 0.005))
            @test isapprox(share(kernel, m; period = 6.0), gamma_law(kernel, m, 6.0);
                atol = 0.025)
        end
        @test isapprox(share(Gamma(0.3, 2.0), 1.0, [FlatBlock(0.99)]; period = 6.0),
            gamma_law(Gamma(0.3, 2.0), 0.01, 6.0); atol = 0.025)

        # A kernel with all its mass inside the window carries an infinite
        # integrated hazard, and no thinning touches that: the pair transmits for
        # certain, whether a multiplier or an intervention's risk is applied to
        # it, because it simply meets again.
        @test share(Uniform(1.5, 1.9), 0.5) == 1.0
        @test share(Uniform(0.1, 0.5), 1.0, [FlatBlock(0.5)]) == 1.0
        # A degenerate contact interval is the exception: it offers one contact
        # and no more, which a multiplier leaves alone and a risk blocks.
        @test share(Dirac(1.0), 0.5) == 1.0
        @test isapprox(share(Dirac(1.0), 1.0, [FlatBlock(0.5)]), 0.5; atol = 0.025)

        # A multiplier of zero never transmits, and draws nothing.
        @test share(Exponential(1.0), 0.0) == 0.0
    end

    @testset "a ring dose follows the earliest trace on the race" begin
        # Nodes 1 and 2 are seeded at 0 and 1, so the race settles node 1 first,
        # and both reach node 3. Node 1 traces it at 4, node 2 sooner, at 2: the
        # dose, given at the trace, must move to the earlier time.
        prog = [Transition(:recovered; from = :infection, delay = 20.0, terminal = true)]
        function dosed(interventions)
            rng = StableRNG(1)
            state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(1.0)),
                prog, EpiBranch.NoAttributes(), rng)
            EpiBranch.add_individuals!(state, 3, interventions)
            EpiBranch._sellke_race!(state, [1, 2, 3], rng; interventions,
                targets = (inf, st) -> (), contacts = (inf, st) -> inf == 3 ? () : (3,),
                from = :infection, until = (:recovered,),
                seed! = (best, members, r) -> (best[1] = 0.0; best[2] = 1.0))
            return state.individuals[3].state
        end
        tracer = TraceAtFixedTimes(Dict(1 => 4.0, 2 => 2.0))

        node3 = dosed([tracer, RingVaccination(efficacy = 0.5, delay_to_immunity = 3.0)])
        @test node3[:trace_time] == 2.0
        @test node3[:vaccination_time] == 2.0
        @test node3[:immunity_time] == 5.0

        # A later dose moves with the trace, and so does a boost timed from it.
        node3 = dosed([tracer,
            RingVaccination(efficacy = 0.5, dose_delay = 1.0, dose_label = :prime),
            RingVaccination(efficacy = 0.5, dose_delay = 7.0, requires_dose = :prime,
                dose_label = :boost)])
        @test node3[:vaccination_time_prime] == 3.0
        @test node3[:vaccination_time_boost] == 9.0

        # A later trace leaves an earlier dose alone.
        node3 = dosed([TraceAtFixedTimes(Dict(1 => 2.0, 2 => 4.0)),
            RingVaccination(efficacy = 0.5)])
        @test node3[:vaccination_time] == 2.0
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

        # Blocking node 1's contact costs node 3 that transmission and nothing
        # else: the contact interval here is degenerate, so the pair has no
        # further contact to offer, and node 2 infects node 3 at 5 instead.
        blocked = race([BlockFrom(1)])
        @test is_infected(blocked.individuals[3])
        @test blocked.individuals[3].parent_id == 2
        @test blocked.individuals[3].infection_time == 5.0

        # A susceptibility of 0 blocks every proposal, so node 3 is never reached.
        immune = race(AbstractIntervention[];
            attributes = transmission_traits(susceptibility = 0.0))
        @test !is_infected(immune.individuals[3])

        # A risk reads the exposure from the contact's `infection_time`
        # (`ProtectFromExposure`, defined with the pool's tests), which
        # holds the proposed time while the risks are resolved. Protection from
        # time 3 lets the proposal at 2 through; protection from time 1 blocks
        # both, and a blocked contact's `infection_time` is left as it was.
        late = race([ProtectFromExposure(3.0)])
        @test late.individuals[3].parent_id == 1
        @test late.individuals[3].infection_time == 2.0
        early = race([ProtectFromExposure(1.0)])
        @test !is_infected(early.individuals[3])
        @test early.individuals[3].infection_time == 0.0

        # Traits at their defaults contribute no risk, draw nothing from the rng,
        # and leave the race stream exactly as it was.
        neutral = race(AbstractIntervention[];
            attributes = transmission_traits(susceptibility = 1.0, infectiousness = 1.0))
        @test neutral.individuals[3].parent_id == 1
        @test neutral.individuals[3].infection_time == 2.0
    end

    @testset "an intervention's risks reach only the routes it can cut" begin
        # Node 1 is seeded at 0 and reaches node 2 on a community route and node
        # 3 on a household route, each at 5. An intervention whose block stands in
        # for removing node 1 takes out the community contact; the household
        # route never opted into intervention removal, so it runs on.
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
                contacts = (inf, st) -> inf == 1 ? (2, 3) : (),
                seed! = (best, members, r) -> (best[1] = 0.0))
            return [get(ind.state, :infected, false) for ind in state.individuals]
        end
        @test infected_after(AbstractIntervention[]) == [true, true, true]
        @test infected_after([BlockFrom(1)]) == [true, false, true]
        @test infected_after([EpiBranch.Scheduled(BlockFrom(1); start_time = 0.0)]) ==
              [true, false, true]
        # A protection scoped to every route applies on the household route too.
        @test EpiBranch.risk_scope(ProtectTo(3)) isa EpiBranch.EveryRoute
        @test infected_after([ProtectTo(3)]) == [true, true, false]
        @test infected_after([ProtectTo(2)]) == [true, false, true]
        # A fully effective ring dose, given when node 1 traces its contacts at
        # time 0, protects on both routes.
        tracer = TraceAtFixedTimes(Dict(1 => 0.0))
        @test infected_after([tracer, RingVaccination(efficacy = 1.0)]) ==
              [true, false, false]
        @test infected_after([tracer, RingVaccination(efficacy = 0.0)]) ==
              [true, true, true]
        # Isolation and quarantine follow the route's removal listing; a vaccine,
        # including its effect on onward transmission, does not.
        @test EpiBranch.risk_scope(Isolation(onset_to_isolation_delay = Dirac(1.0))) isa
              EpiBranch.RemovalRoutes
        @test EpiBranch.risk_scope(ContactTracing(probability = 1.0,
            isolation_to_trace_delay = Dirac(1.0))) isa EpiBranch.RemovalRoutes
        @test EpiBranch.risk_scope(RingVaccination(efficacy = 1.0,
            onward_efficacy = 1.0)) isa EpiBranch.EveryRoute
        @test EpiBranch.risk_scope(MassVaccination(efficacy = 1.0,
            eligibility_time = 0.0)) isa EpiBranch.EveryRoute
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
