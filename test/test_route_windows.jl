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
function EpiBranch.risk_applies(::BlockFrom, route)
    route !== nothing && EpiBranch.INTERVENTION_REMOVAL in route.until
end

# Blocks every transmission into one named contact: a protection that belongs to
# the person, left at the default scope.
struct ProtectTo <: EpiBranch.AbstractIntervention
    id::Int
end
function EpiBranch.competing_risk(p::ProtectTo, parent, contact, state)
    contact.id == p.id ? Risk(block_probability = 1.0) : nothing
end

# A custom route predicate can select a route independently of removal.
struct ProtectOnRoute <: EpiBranch.AbstractIntervention
    name::Symbol
end
EpiBranch.risk_applies(p::ProtectOnRoute, route) = route !== nothing && route.name == p.name
function EpiBranch.competing_risk(::ProtectOnRoute, parent, contact, state)
    Risk(block_probability = 1.0)
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

        # Vaccination delivery still requires generation-based contacts.
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

        # A schedule is evaluated at the proposed contact time, even if the
        # most recent accepted infection is the seed at time zero.
        @test !secondary(Dirac(10.0), 1.0,
            [Scheduled(FlatBlock(1.0); start_time = 5.0)], 20.0, 1)
        @test secondary(Dirac(10.0), 1.0,
            [Scheduled(FlatBlock(1.0); end_time = 5.0)], 20.0, 1)

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
        scaled_law(kernel, m, period) = 1 - ccdf(kernel, period)^m
        for (kernel, m) in ((Gamma(0.3, 2.0), 0.01), (Gamma(0.7, 1.0), 0.005))
            @test isapprox(share(kernel, m; period = 6.0), scaled_law(kernel, m, 6.0);
                atol = 0.025)
        end
        @test isapprox(share(Gamma(0.3, 2.0), 1.0, [FlatBlock(0.99)]; period = 6.0),
            scaled_law(Gamma(0.3, 2.0), 0.01, 6.0); atol = 0.025)

        # A kernel with no inverse survival of its own falls back on one that
        # rebuilds the argument in linear space and hands back the top of the
        # support, losing contacts that are still inside the window: the same law
        # would then give different answers written as `Rayleigh(1)` or as
        # `Weibull(2, sqrt(2))`. Inverting the kernel's own log-survival instead
        # keeps them together, and covers a mixture, a truncated and a shifted
        # distribution too.
        @test isapprox(share(Rayleigh(1.0), 0.005; period = 20.0),
            share(Weibull(2.0, sqrt(2)), 0.005; period = 20.0); atol = 0.025)
        fallbacks = ((Rayleigh(1.0), 20.0, 0.005), (Rayleigh(1.0), 20.0, 0.05),
            (1.0 + Exponential(1.0), 100.0, 0.01),
            (MixtureModel([Exponential(1.0), Exponential(5.0)]), 100.0, 0.01),
            (truncated(Exponential(1.0), 0.0, 5.0), 20.0, 0.05))
        for (kernel, period, m) in fallbacks
            @test isapprox(share(kernel, m; period), scaled_law(kernel, m, period);
                atol = 0.025)
        end
        # and through the blocked-contact continuation, which inverts the same
        # survival from the time of the contact it blocked
        for (kernel, period, m) in fallbacks[1:(end - 1)]
            @test isapprox(share(kernel, 1.0, [FlatBlock(1 - m)]; period),
                scaled_law(kernel, m, period); atol = 0.03)
        end
        # Rejection sampling cannot enumerate infinitely many contacts before
        # a bounded continuous kernel reaches the end of its support.
        @test_throws ArgumentError share(truncated(Exponential(1.0), 0.0, 5.0),
            1.0, [FlatBlock(0.95)]; period = 20.0)
        @test share(Uniform(1.5, 1.9), 0.5) == 1.0
        @test_throws ArgumentError share(Uniform(0.1, 0.5), 1.0, [FlatBlock(1.0)])
        @test_throws ArgumentError share(Exponential(1.0), 1.0, [FlatBlock(1.0)];
            period = Inf)
        # A degenerate contact interval is the exception: it offers one contact
        # and no more, which a multiplier leaves alone and a risk blocks.
        @test share(Dirac(1.0), 0.5) == 1.0
        @test isapprox(share(Dirac(1.0), 1.0, [FlatBlock(0.5)]), 0.5; atol = 0.025)

        # A multiplier of zero never transmits, and draws nothing.
        @test share(Exponential(1.0), 0.0) == 0.0
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
        @test isnan(early.individuals[3].infection_time)

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
        @test EpiBranch.risk_applies(ProtectTo(3), nothing)
        @test infected_after([ProtectTo(3)]) == [true, true, false]
        @test infected_after([ProtectTo(2)]) == [true, false, true]
        @test infected_after([ProtectOnRoute(:household)]) == [true, true, false]
        @test infected_after([Scheduled(
            Scheduled(ProtectOnRoute(:community);
                start_time = 0.0); start_time = 0.0)]) == [true, false, true]
        @test !EpiBranch.risk_applies(ProtectOnRoute(:household), nothing)
        # Isolation and quarantine follow the route's removal listing; a vaccine,
        # including its effect on onward transmission, does not.
        for route in (nothing, RouteWindow(:household; until = (:recovered,), kernel = Dirac(1.0)),
            RouteWindow(:community; until = (REM,), kernel = Dirac(1.0)))
            removes = route !== nothing && REM in route.until
            @test EpiBranch.risk_applies(Isolation(onset_to_isolation_delay = Dirac(1.0)), route) ==
                  removes
            @test EpiBranch.risk_applies(
                ContactTracing(probability = 1.0,
                    isolation_to_trace_delay = Dirac(1.0)),
                route) == removes
            @test EpiBranch.risk_applies(
                RingVaccination(efficacy = 1.0,
                    onward_efficacy = 1.0), route)
            @test EpiBranch.risk_applies(
                MassVaccination(efficacy = 1.0,
                    eligibility_time = 0.0), route)
        end
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

@testset "Community introductions respect susceptibility" begin
    for source in (0.5, Gamma(2.0, 3.0))
        @test EpiBranch._ext_draw(StableRNG(7), source, 1.0) ==
              EpiBranch._ext_draw(StableRNG(7), source)
        rng = StableRNG(7)
        @test EpiBranch._ext_draw(rng, source, 0.0) == Inf
        @test rand(rng) == rand(StableRNG(7))
    end
    # Scaling a constant hazard by susceptibility scales its mean waiting time.
    draws = [EpiBranch._ext_draw(StableRNG(seed), 0.5, 0.25) for seed in 1:4000]
    @test isapprox(sum(draws) / length(draws), 8.0; rtol = 0.04)
end

@testset "A withdrawn edge cannot offer another contact after a block" begin
    rng = StableRNG(11)
    prog = [Transition(:recovered; from = :infection, delay = 100.0, terminal = true)]
    state = EpiBranch.new_state(BranchingProcess(Poisson(1.0), Exponential(1.0)),
        prog, NoAttributes(), rng)
    interventions = [FlatBlock(1.0)]
    EpiBranch.add_individuals!(state, 2, interventions)
    enquiries = Ref(0)
    targets = function (inf, st)
        enquiries[] += 1
        # The edge is available for the original proposal, then withdrawn.
        return enquiries[] == 1 ? ((2, Exponential(1.0)),) : ()
    end
    EpiBranch._sellke_race!(state, [1, 2], rng; targets,
        from = :infection, until = (:recovered,), interventions,
        seed! = (best, members, r) -> (best[1] = 0.0))
    @test !is_infected(state.individuals[2])
    @test enquiries[] == 2
end

@testset "Unbounded blocked introductions fail promptly" begin
    rng = StableRNG(1)
    state = EpiBranch.new_state(BranchingProcess(Poisson(0.0), Exponential(1.0)),
        AbstractClinicalTransition[], NoAttributes(), rng)
    interventions = [ProtectTo(1)]
    EpiBranch.add_individuals!(state, 1, interventions)
    @test_throws ArgumentError EpiBranch._sellke_race!(state, [1], rng;
        from = :infection, until = (), interventions,
        introduction = (Exponential(1.0), Inf), targets = (i, st) -> (),
        seed! = (best, members, r) -> (best[1] = 1.0))
    @test !is_infected(only(state.individuals))
    @test state.max_infection_time == 0.0
end
