# `docs/src/tutorials/networks.md` prints numbers that its prose then reads:
# the traceability comparison asserts that the more community contacts a case
# can name, the more tracing prevents. The docs build renders those numbers
# without checking them, so an edit that flattens or reverses the comparison
# ships a page contradicting itself. The claim is pinned here rather than the
# figures, which drift with any change to the engine's draws.

function households_and_community(n_households, household_size, rng)
    n = n_households * household_size
    hh = [Int[] for _ in 1:n]
    for h in 0:(n_households - 1)
        members = (h * household_size + 1):(h * household_size + household_size)
        for i in members, j in members
            i != j && push!(hh[i], j)
        end
    end
    comm = [Int[] for _ in 1:n]
    for _ in 1:(3 * n)
        a, b = rand(rng, 1:n), rand(rng, 1:n)
        if a != b && !(b in comm[a])
            push!(comm[a], b)
            push!(comm[b], a)
        end
    end
    return hh, comm
end

@testset "Tracing prevents more where more contacts can be named" begin
    hh_adj, comm_adj = households_and_community(150, 4, StableRNG(99))
    clinical = clinical_presentation(
        incubation_period = LogNormal(0.5, 0.3), prob_asymptomatic = 0.0
    )
    removal = EpiBranch.INTERVENTION_REMOVAL
    iso = Isolation(
        onset_to_isolation_delay = Exponential(4.0), test_sensitivity = 1.0,
        duration = 7.0
    )
    # The quarantine is the one the tutorial uses, and it has to be the one that
    # never releases: a quarantine that hands the contact back leaves the
    # infectious window open, and against a community kernel this slow it
    # prevents almost nothing, which flattens the comparison the prose reads.
    ct = ContactTracing(
        probability = 0.9, isolation_to_trace_delay = Exponential(0.5),
        action = Quarantine(duration = Inf)
    )
    traced_routes(traceable) = [
        RouteWindow(
            :household; until = (:recovered, removal),
            kernel = Weibull(1.5, 4.0), reach = hh_adj
        ),
        RouteWindow(
            :community; until = (:recovered, removal),
            kernel = Exponential(15.0), reach = comm_adj, traceable = traceable
        ),
    ]
    function mean_size(windows, interventions)
        m = ModelSpec(
            RoutedNetwork(windows);
            progression = [
                Transition(
                    :recovered; from = :infection, delay = 10.0, terminal = true
                ),
            ],
            interventions = interventions, attributes = clinical
        )
        return sum(
            simulate(m; n_initial = 3, rng = StableRNG(s)).cumulative_cases
                for s in 1:30
        ) / 30
    end

    sizes = [mean_size(traced_routes(p), [iso, ct]) for p in (0.0, 0.5, 1.0)]
    @test issorted(sizes; rev = true)
    @test last(sizes) < first(sizes) / 2
end
