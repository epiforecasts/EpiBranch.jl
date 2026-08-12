# Route windows: the unit that lets one case carry several transmission routes,
# each opening and closing on different states of its natural history.

@testset "Route windows" begin
    @testset "construction and show" begin
        w = RouteWindow(:community; from = :infectious, until = (:recovered,),
            kernel = Exponential(3.0))
        @test w.name === :community
        @test w.from === :infectious
        @test w.until == (:recovered,)
        @test w.reach === :community          # defaults to the route name
        @test occursin("RouteWindow(:community", repr(w))
        @test occursin("from=:infectious", repr(w))

        # defaults: opens at infection, never censored, reach follows the name
        d = RouteWindow(:household; kernel = Exponential(1.0), reach = [[2], [1]])
        @test d.from === :infection
        @test d.until == ()
        @test d.reach == [[2], [1]]
    end

    @testset "opening and closing read the state-time convention" begin
        ind = Individual(id = 1)
        ind.infection_time = 5.0
        ind.state[:infectious_time] = 7.0
        ind.state[:recovered_time] = 20.0
        ind.state[:died_time] = 14.0

        # from :infection opens at the infection time itself
        @test window_open(ind, RouteWindow(:c; kernel = nothing)) == 5.0
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

    @testset "isolation is a removal state readable by a window" begin
        ind = Individual(id = 1)
        set_isolated!(ind, 4.0)
        @test is_isolated(ind)
        # both keys carry the time: the intervention layer's name and the
        # `<state>_time` convention a window's `until` reads
        @test isolation_time(ind) == 4.0
        @test ind.state[:isolated_time] == 4.0
        @test window_close(ind, RouteWindow(:c; until = (:isolated,), kernel = nothing)) ==
              4.0

        # clearing must clear both, or a stale removal time keeps censoring
        clear_isolated!(ind)
        @test !is_isolated(ind)
        @test isolation_time(ind) == Inf
        @test window_close(ind, RouteWindow(:c; until = (:isolated,), kernel = nothing)) ==
              Inf
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
    end
end
