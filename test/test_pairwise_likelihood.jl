using ForwardDiff
using DifferentiationInterface: DifferentiationInterface
import Mooncake
using ADTypes: AutoMooncake

# A minimal infection layer over an explicit contact structure, as a companion
# package would define one: the fields the likelihood reads plus the structure.
# A layer that also records the stretch a lapsing removal took each host out
# for, which `household_infections`/`network_infections` put in `host_times`.
struct _GappedInfections{S} <: InfectionLayer
    structure::S
    infection_time::Vector{Float64}
    infectious_time::Vector{Float64}
    removal_time::Vector{Float64}
    is_index::Vector{Bool}
    obs_end::Float64
    followup_end::Float64
    host_times::NamedTuple
end
EpiBranch.contact_structure(d::_GappedInfections) = d.structure

struct _TestInfections{S} <: InfectionLayer
    structure::S
    infection_time::Vector{Float64}
    infectious_time::Vector{Float64}
    removal_time::Vector{Float64}
    is_index::Vector{Bool}
    obs_end::Float64
    followup_end::Float64
end
function _TestInfections(
        structure, inf, infectious, removal, index; obs_end = Inf,
        followup_end = Inf
    )
    return _TestInfections(
        structure, Float64.(inf), Float64.(infectious),
        Float64.(removal), Vector{Bool}(index), Float64(obs_end),
        Float64(followup_end)
    )
end
EpiBranch.contact_structure(d::_TestInfections) = d.structure

# The same layer without a follow-up time, as a subtype written before it existed.
struct _NoFollowup{S} <: InfectionLayer
    structure::S
    infection_time::Vector{Float64}
    infectious_time::Vector{Float64}
    removal_time::Vector{Float64}
    is_index::Vector{Bool}
    obs_end::Float64
end
EpiBranch.contact_structure(d::_NoFollowup) = d.structure

# `d` as observed up to `tf`: later infections unseen, windows and community
# introductions cut there.
function _truncate(d::_TestInfections, tf)
    late = .!isnan.(d.infection_time) .& (d.infection_time .> tf)
    inf = _nan_where(d.infection_time, late)
    infectious = _nan_where(d.infectious_time, late)
    removal = _nan_where(min.(d.removal_time, tf), late)
    return _TestInfections(
        d.structure, inf, infectious, removal, d.is_index;
        obs_end = min(d.obs_end, tf)
    )
end
_nan_where(x, mask) = [m ? NaN : v for (v, m) in zip(x, mask)]

# A layer that forgets to name its structure.
struct _NoStructure <: InfectionLayer end

# A layer holding its per-host times under a name of its own, as a subtype
# written before `host_times` existed might.
struct _NamedTimes{S} <: InfectionLayer
    structure::S
    times::NamedTuple
end
EpiBranch.contact_structure(d::_NamedTimes) = d.structure

# A grouping the package does not define, written the same way a household or
# network grouping would be: every host's rows add into one of two groups by
# parity of its id, with no change to pairwise_survival.jl.
struct _Parity <: EpiBranch.PairwiseReduction end
EpiBranch.ngroups(::_Parity) = 2
EpiBranch.group(::_Parity, host) = host % 2 == 0 ? 2 : 1

# Contiguous cliques of the given sizes, as a membership vector and as the
# equivalent adjacency list (each member lists its household-mates in order).
function _cliques(sizes)
    membership = reduce(vcat, [fill(h, s) for (h, s) in enumerate(sizes)])
    adjacency = [Int[] for _ in membership]
    for i in eachindex(membership), j in eachindex(membership)

        i != j && membership[i] == membership[j] && push!(adjacency[i], j)
    end
    return membership, adjacency
end

@testset "Pairwise survival likelihood" begin
    @testset "counting-process rows" begin
        rows = PairwiseSurvivalData(
            [1, 1, 2, 2], [0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 1.5, 5.0], [true, false, true, false]
        )
        k = Exponential(3.0)
        # one event row per susceptible, so each adds its log-hazard; every row
        # subtracts its cumulative hazard (t/3 for this kernel)
        @test pairwise_surv_loglik(k, rows) ≈ 2 * log(1 / 3) - (2 + 4 + 1.5 + 5) / 3
        @test pairwise_surv_loglik(r -> k, rows) ≈ pairwise_surv_loglik(k, rows)
        @test_throws ArgumentError PairwiseSurvivalData([1], [2.0], [1.0], [true])
        # differentiable in a log-scale parameter
        f(θ) = pairwise_surv_loglik(Exponential(exp(θ)), rows)
        fd = (f(log(3.0) + 1.0e-6) - f(log(3.0) - 1.0e-6)) / 2.0e-6
        @test ForwardDiff.derivative(f, log(3.0)) ≈ fd rtol = 1.0e-4
    end

    @testset "a zero hazard adds no NaN to a gradient" begin
        # A pair before a shifted kernel's support has log-hazard -Inf, whose
        # partials can be NaN; the reduction must skip it.
        D = ForwardDiff.Dual{Nothing, Float64, 1}
        acc = EpiBranch._LogSumExpAcc{D}()
        EpiBranch._push!(acc, D(-Inf, ForwardDiff.Partials((NaN,))))
        EpiBranch._push!(acc, D(0.5, ForwardDiff.Partials((1.0,))))
        @test ForwardDiff.value(EpiBranch._value(acc)) == 0.5
        @test ForwardDiff.partials(EpiBranch._value(acc))[1] == 1.0
    end

    @testset "infection-layer columns read out of a simulation" begin
        # the homogeneous pool runs the same one-window race as a household or a
        # network, so its state reads back the same way: the window opens at
        # :infectious and closes at recovery or isolation, whichever is first
        progression = [
            Transition(:onset; from = :infection, delay = 0.1),
            Transition(:infectious; from = :infection, delay = 0.5),
            Transition(
                :recovered; from = :infectious, delay = Exponential(1.0),
                terminal = true
            ),
        ]
        m = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 300);
            progression,
            interventions = [Isolation(onset_to_isolation_delay = Exponential(1.0), isolation_duration = Inf)]
        )
        state = simulate(m; rng = StableRNG(1), n_initial = 3)
        columns = EpiBranch._infection_layer_columns(state, m)

        infected = findall(is_infected, state.individuals)
        @test findall(!isnan, columns.infection_time) == infected
        @test count(columns.is_index) == 3
        inds = state.individuals[infected]
        infectious = [ind.state[:infectious_time] for ind in inds]
        recovered = [ind.state[:recovered_time] for ind in inds]
        @test columns.infectious_time[infected] == infectious
        @test columns.removal_time[infected] == min.(recovered, isolation_time.(inds))
        @test any(columns.removal_time[infected] .< recovered)
        uninfected = setdiff(eachindex(state.individuals), infected)
        @test all(isinf, columns.removal_time[uninfected])
    end

    @testset "infection-layer fields share one number type" begin
        fields = EpiBranch._infection_layer_fields(
            2, [0, 1], [0.0f0, 1.0f0],
            [2.0, Inf], [1, 0]; obs_end = 3, followup_end = Inf
        )
        @test fields == ([0.0, 1.0], [0.0, 1.0], [2.0, Inf], [true, false], 3.0, Inf, (;))
        onsets = EpiBranch._infection_layer_fields(
            2, [0.0, 1.0], [0.0, 1.0],
            [2.0, Inf], [true, false]; obs_end = Inf, followup_end = Inf,
            host_times = (onset_time = [1, NaN], trace_time = [missing, 2])
        )[end]
        @test onsets.onset_time isa Vector{Float64}
        @test isequal(onsets.onset_time, [1.0, NaN])
        @test onsets.trace_time isa Vector{Union{Missing, Float64}}
        @test isequal(onsets.trace_time, [missing, 2.0])
        @test_throws ArgumentError EpiBranch._infection_layer_fields(
            2, [0.0, 1.0],
            [0.0, 1.0], [2.0, Inf], [true, false]; obs_end = Inf, followup_end = Inf,
            host_times = (onset_time = [1.0],)
        )
        @test all(v -> eltype(v) == Float64, fields[1:3])
        @test fields[4] isa Vector{Bool}
        dual = ForwardDiff.Dual(1.0, 1.0)
        @test eltype(
            first(
                EpiBranch._infection_layer_fields(
                    1, [dual], [0.0], [2.0],
                    [true]; obs_end = Inf, followup_end = Inf
                )
            )
        ) == typeof(dual)
        @test_throws ArgumentError EpiBranch._infection_layer_fields(
            2, [0.0], [0.0],
            [1.0], [true]; obs_end = Inf, followup_end = Inf
        )
    end

    @testset "community hazard helpers" begin
        @test EpiBranch._valid_external(0)
        @test EpiBranch._valid_external(Gamma(2.0, 3.0))
        @test !EpiBranch._valid_external(-0.1)
        @test !EpiBranch._valid_external(Normal())
        @test !EpiBranch._valid_external("0.1")
        @test EpiBranch._normalise_external(1) === 1.0
        @test EpiBranch._normalise_external(Gamma(2.0, 3.0)) == Gamma(2.0, 3.0)
        @test !EpiBranch._ext_active(0.0)
        @test EpiBranch._ext_active(Gamma(2.0, 3.0))
        # a constant rate draws its exponential waiting time
        @test EpiBranch._ext_draw(StableRNG(1), 0.5) ==
            rand(StableRNG(1), Exponential(2.0))
        @test EpiBranch._ext_draw(StableRNG(1), Gamma(2.0, 3.0)) ==
            rand(StableRNG(1), Gamma(2.0, 3.0))
    end

    @testset "layout on a household partition" begin
        # households {1,2,3} and {4,5}; 1 and 4 are indexes, 2 is infected too
        membership = [1, 1, 1, 2, 2]
        is_index = [true, false, false, true, false]
        infected = [true, true, false, true, false]

        L = compile_contact_pairs(membership, is_index, infected)
        @test L isa ContactPairsLayout
        @test !L.external
        @test L.nhosts == 5
        # 2←1, 3←1, 3←2, 5←4: only infected hosts infect, indexes are conditioned on
        @test L.sus == [2, 3, 3, 5]
        @test L.infector == [1, 1, 2, 4]
        @test L.contact_index == [0, 0, 0, 0]
        @test !any(L.is_ext)
        @test L.sus_unique == [2, 3, 5]
        @test L.sus_row_ranges == [1:1, 2:3, 4:4]
        @test isempty(L.no_rows)

        # with a community hazard every host is explained and gets an external row
        Le = compile_contact_pairs(membership, is_index, infected; external = true)
        @test Le.sus == [1, 1, 2, 2, 3, 3, 3, 4, 5, 5]
        @test Le.infector == [0, 2, 0, 1, 0, 1, 2, 0, 0, 4]
        @test Le.is_ext == (Le.infector .== 0)

        @test_throws ArgumentError compile_contact_pairs([1, 1], [true], [true, false])
        @test length(compile_contact_pairs(Int[], Bool[], Bool[])) == 0

        # each household is its own connected component, numbered in label order
        @test L.component == [1, 1, 1, 2, 2]
        @test L.ncomponents == 2
    end

    @testset "layout on a directed graph" begin
        # 1→2, 1→3, 2→3, 3→1; host 4 has no edges. Infected: 1 (index) and 2.
        contacts = [[2, 3], [3], [1], Int[]]
        is_index = [true, false, false, false]
        infected = [true, true, false, false]

        L = compile_contact_pairs(contacts, is_index, infected)
        # possible infectors are in-neighbours: 2←1, 3←{1, 2}; 3→1 is not a row
        # because 3 is uninfected, and 1 is conditioned on
        @test L.sus == [2, 3, 3]
        @test L.infector == [1, 1, 2]
        # the edge each row travels: 2 is 1's first contact, 3 its second, and
        # 3 is 2's first
        @test L.contact_index == [1, 2, 1]
        @test L.sus_unique == [2, 3]
        # 4 is not conditioned on but no infected host lists it as a contact
        @test L.no_rows == [4]

        Le = compile_contact_pairs(contacts, is_index, infected; external = true)
        @test Le.sus == [1, 2, 2, 3, 3, 3, 4]
        @test Le.infector == [0, 0, 1, 0, 1, 2, 0]
        @test Le.contact_index == [0, 0, 1, 0, 2, 1, 0]
        @test isempty(Le.no_rows)

        # a self-loop is never a row; an out-of-range contact is rejected
        @test length(
            compile_contact_pairs(
                [[1, 2], Int[]], [true, false],
                [true, false]
            )
        ) == 1
        @test_throws ArgumentError compile_contact_pairs(
            [[3], Int[]], [true, false],
            [true, false]
        )

        # host 4 has no edges at all, so it is its own component
        @test L.component == [1, 1, 1, 2]
        @test L.ncomponents == 2
    end

    @testset "evaluation matches hand-built counting-process rows" begin
        # the directed graph above with times: 1 infected at 0, infectious [0, 5];
        # 2 infected at 2, infectious [2, 6]; 3 never infected
        contacts = [[2, 3], [3], [1], Int[]]
        data = _TestInfections(
            contacts, [0.0, 2.0, NaN, NaN], [0.0, 2.0, NaN, NaN],
            [5.0, 6.0, Inf, Inf], [true, false, false, false]
        )
        k = Weibull(1.5, 3.0)
        # 2←1 at risk for 2 with an event; 3←1 at risk for 5, 3←2 for 4, no event
        rows = PairwiseSurvivalData(
            [2, 3, 3], zeros(3), [2.0, 5.0, 4.0],
            [true, false, false]
        )
        L = compile_contact_pairs(data)
        @test pairwise_surv_loglik(k, data, L) ≈ pairwise_surv_loglik(k, rows)
        @test pairwise_surv_loglik(k, data) == pairwise_surv_loglik(k, data, L)

        # per-edge kernels index the edge a row travels; callables see the pair
        per_edge = [
            [Weibull(1.5, 3.0), Weibull(1.5, 7.0)], [Weibull(1.5, 11.0)],
            [Weibull(1.5, 1.0)], Weibull{Float64}[],
        ]
        rowk = r -> (Weibull(1.5, 3.0), Weibull(1.5, 7.0), Weibull(1.5, 11.0))[r]
        @test pairwise_surv_loglik(per_edge, data, L) ≈
            pairwise_surv_loglik(rowk, rows)
        pairk = (i, j) -> per_edge[i][findfirst(==(j), contacts[i])]
        @test pairwise_surv_loglik(pairk, data, L) ≈
            pairwise_surv_loglik(per_edge, data, L)

        # a partition layout has no edge list for a per-edge kernel to index
        hh = _TestInfections(
            [1, 1], [0.0, 1.0], [0.0, 1.0], [3.0, 4.0],
            [true, false]
        )
        @test_throws ArgumentError pairwise_surv_loglik([[k], [k]], hh)

        # the external mode and the layout must agree
        @test_throws ArgumentError pairwise_surv_loglik(
            k, data, L;
            external_hazard = 0.1
        )
        short = _TestInfections(
            contacts[1:2], [0.0, 2.0], [0.0, 2.0], [5.0, 6.0],
            [true, false]
        )
        @test_throws DimensionMismatch pairwise_surv_loglik(k, short, L)
    end

    @testset "an infection layer must name its contact structure" begin
        @test_throws ArgumentError EpiBranch.contact_structure(_NoStructure())
    end

    @testset "a partition and its clique graph give the same likelihood" begin
        membership, adjacency = _cliques([3, 4, 2, 4])
        n = length(membership)
        inf = [0.0, 1.2, NaN, 0.0, 2.1, 3.5, NaN, 0.0, NaN, 0.0, 0.7, NaN, 4.2]
        infectious = inf .+ 0.5
        removal = infectious .+ 4.0
        is_index = [
            true, false, false, true, false, false, false, true, false,
            true, false, false, false,
        ]
        @test length(inf) == n
        hh = _TestInfections(
            membership, inf, infectious, removal, is_index;
            obs_end = 10.0
        )
        net = _TestInfections(
            adjacency, inf, infectious, removal, is_index;
            obs_end = 10.0
        )

        for external in (false, true)
            Lh = compile_contact_pairs(hh; external)
            Ln = compile_contact_pairs(net; external)
            @test Lh.sus == Ln.sus
            @test Lh.infector == Ln.infector
            @test Lh.is_ext == Ln.is_ext
            @test Lh.sus_unique == Ln.sus_unique
            α = external ? 0.05 : 0.0
            for k in (Exponential(2.5), Weibull(1.5, 3.0))
                @test pairwise_surv_loglik(k, hh, Lh; external_hazard = α) ==
                    pairwise_surv_loglik(k, net, Ln; external_hazard = α)
            end
            cov = (i, j) -> Exponential(2.0 + 0.1 * i)
            @test pairwise_surv_loglik(cov, hh, Lh; external_hazard = α) ==
                pairwise_surv_loglik(cov, net, Ln; external_hazard = α)
        end
    end

    @testset "a community hazard introduces cases only up to obs_end" begin
        # 1 is a community case at 1; 2 is infected by 1 at 4, after obs_end = 3;
        # 3 is never infected and can be reached by 1 and 2; 4 has no contacts
        contacts = [[2, 3], [3], [1], Int[]]
        data = _TestInfections(
            contacts, [1.0, 4.0, NaN, NaN], [1.0, 4.0, NaN, NaN],
            [6.0, 8.0, Inf, Inf], [true, false, false, false]; obs_end = 3.0
        )
        L = compile_contact_pairs(data; external = true)
        k = Weibull(1.5, 3.0)
        # community rows stop at the earlier of infection and obs_end, and 2 has no
        # community event; 3 is exposed to 1 over [1, 6] and to 2 over [4, 8]
        rows = PairwiseSurvivalData(
            [1, 2, 2, 3, 3, 3, 4],
            zeros(7), [1.0, 3.0, 3.0, 3.0, 5.0, 4.0, 3.0],
            [true, false, true, false, false, false, false]
        )
        is_ext = [true, true, false, true, false, false, true]
        for ext in (0.05, Gamma(2.0, 5.0))
            extdist = ext isa Real ? Exponential(1 / ext) : ext
            rowk = r -> is_ext[r] ? extdist : k
            @test pairwise_surv_loglik(k, data, L; external_hazard = ext) ≈
                pairwise_surv_loglik(rowk, rows)
        end
    end

    @testset "a community case at time 0 is counted" begin
        # households {1, 2} and {3}: 1 and 3 are community cases at 0, 1 is
        # infectious over [0, 3] and 2 is never infected
        data = _TestInfections(
            [1, 1, 2], [0.0, NaN, 0.0], [0.0, NaN, 0.0],
            [3.0, Inf, 3.0], [true, false, true]; obs_end = 5.0
        )
        L = compile_contact_pairs(data; external = true)
        k = Exponential(2.0)
        α = 0.1
        # each case at 0 adds log α; 2 escapes α·5 from the community and 3/2 from 1
        @test pairwise_surv_loglik(k, data, L; external_hazard = α) ≈
            2 * log(α) - 5α - 3 / 2
        # a community hazard that is zero at 0 cannot have introduced them
        @test pairwise_surv_loglik(k, data, L; external_hazard = Gamma(2.0, 5.0)) ==
            -Inf
    end

    @testset "an infection where every hazard is zero has zero density" begin
        # 1 and 2 are indexes at 0 and 3 is infected at 1, but the kernel has no
        # hazard before 2, so neither of its possible infectors can explain it
        data = _TestInfections(
            [1, 1, 1], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            [5.0, 5.0, 6.0], [true, true, false]
        )
        @test pairwise_surv_loglik(Uniform(2.0, 10.0), data) == -Inf
    end

    @testset "an infection no source can explain has zero density" begin
        k = Exponential(2.0)
        # without a community hazard: index 1 is infectious over [0, 3] and 2 is
        # infected at t, which 1 can explain only up to 3
        for (t, expected) in ((2.9, log(1 / 2) - 2.9 / 2), (3.5, -Inf), (5.0, -Inf))
            data = _TestInfections(
                [1, 1], [0.0, t], [0.0, t], [3.0, t + 3.0],
                [true, false]
            )
            @test pairwise_surv_loglik(k, data) ≈ expected
        end

        # with a community hazard 0.1 up to obs_end = 4: 1 is a community case at
        # 0.5, infectious over [0.5, 3], and after 4 only 1 could infect 2
        for (t, expected) in (
                (3.5, 2 * log(0.1) - 0.05 - 0.35 - 2.5 / 2),
                (4.5, -Inf), (6.0, -Inf),
            )
            data = _TestInfections(
                [1, 1], [0.5, t], [0.5, t], [3.0, t + 3.0],
                [false, false]; obs_end = 4.0
            )
            @test pairwise_surv_loglik(k, data; external_hazard = 0.1) ≈ expected
        end

        # a host with no possible infector at all: 1 infects 2 and nobody lists 3
        contacts = [[2], Int[], Int[]]
        infected_3 = _TestInfections(
            contacts, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
            [4.0, 5.0, 6.0], [true, false, false]
        )
        @test pairwise_surv_loglik(k, infected_3) == -Inf
        escaped_3 = _TestInfections(
            contacts, [0.0, 1.0, NaN], [0.0, 1.0, NaN],
            [4.0, 5.0, Inf], [true, false, false]
        )
        @test pairwise_surv_loglik(k, escaped_3) ≈ log(1 / 2) - 1 / 2
        # an index case is conditioned on, so it needs no possible infector
        index_3 = _TestInfections(
            contacts, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
            [4.0, 5.0, 6.0], [true, false, true]
        )
        @test isempty(compile_contact_pairs(index_3).no_rows)
        @test pairwise_surv_loglik(k, index_3) ≈ log(1 / 2) - 1 / 2
    end

    @testset "nothing after the end of follow-up contributes" begin
        # path 1-2-3 followed up to 5: index 1 infectious over [0, 2] infects 2 at
        # 1, which is still infectious at 5, and 3 has escaped until then
        contacts = [[2], [1, 3], [2]]
        k = Exponential(2.0)
        ongoing = _TestInfections(
            contacts, [0.0, 1.0, NaN], [0.0, 1.0, NaN],
            [2.0, Inf, NaN], [true, false, false]; obs_end = 5.0, followup_end = 5.0
        )
        @test pairwise_surv_loglik(k, ongoing) ≈ log(1 / 2) - 1 / 2 - 4 / 2
        # 3 is also exposed to the community until 5
        @test pairwise_surv_loglik(k, ongoing; external_hazard = 0.1) ≈
            log(0.1) + log(1 / 2 + 0.1) - 0.1 - 1 / 2 - 4 / 2 - 0.5
        # without a follow-up time 3 is exposed to 2 for ever
        unbounded = _TestInfections(
            contacts, [0.0, 1.0, NaN], [0.0, 1.0, NaN],
            [2.0, Inf, NaN], [true, false, false]; obs_end = 5.0
        )
        @test pairwise_surv_loglik(k, unbounded) == -Inf

        # Cliques and a graph with latent periods, ongoing windows, infections
        # after follow-up (including one nobody could explain then, and an index
        # case) and a window opening after it: evaluating up to 6 matches
        # evaluating the data truncated at 6.
        membership, adjacency = _cliques([3, 4, 2, 4])
        inf = [0.0, 1.2, NaN, 0.0, 2.1, 5.5, 8.0, 0.0, 7.0, 0.0, 0.7, NaN, 9.0]
        infectious = inf .+ [0.5, 0.3, 0, 0.4, 0.2, 0.1, 0.3, 0.6, 0.2, 0.1, 0.2, 0, 0.3]
        removal = [3.0, Inf, NaN, 4.0, Inf, Inf, Inf, 2.0, Inf, 5.0, Inf, NaN, Inf]
        index = [
            true, false, false, true, false, false, false, true, true,
            true, false, false, false,
        ]
        wk = Weibull(1.5, 3.0)
        for structure in (membership, adjacency), obs_end in (3.0, 10.0)

            full = _TestInfections(
                structure, inf, infectious, removal, index;
                obs_end, followup_end = 6.0
            )
            cut = _truncate(full, 6.0)
            @test isfinite(pairwise_surv_loglik(wk, full))
            @test pairwise_surv_loglik(wk, full) ≈ pairwise_surv_loglik(wk, cut)
            for α in (0.05, Weibull(1.0, 20.0))
                v = pairwise_surv_loglik(wk, full; external_hazard = α)
                @test isfinite(v)
                @test v ≈ pairwise_surv_loglik(wk, cut; external_hazard = α)
                L = compile_contact_pairs(full; external = true)
                @test pairwise_surv_loglik(wk, full, L; external_hazard = α) ≈ v
            end
        end

        # the gradient in the kernel and community hazard matches finite differences
        full = _TestInfections(
            adjacency, inf, infectious, removal, index;
            obs_end = 10.0, followup_end = 6.0
        )
        L = compile_contact_pairs(full; external = true)
        f(θ) = pairwise_surv_loglik(
            Weibull(exp(θ[1]), exp(θ[2])), full, L;
            external_hazard = exp(θ[3])
        )
        θ = [log(1.5), log(3.0), log(0.05)]
        g = ForwardDiff.gradient(f, θ)
        h = 1.0e-6
        fd = [
            (f(θ .+ h .* e) - f(θ .- h .* e)) / 2h
                for e in ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0])
        ]
        @test all(isfinite, g)
        @test g ≈ fd rtol = 1.0e-5
        @test (@inferred pairwise_surv_loglik(wk, full, L; external_hazard = 0.05)) isa
            Float64

        # a layer without the field is followed up for ever, as `Inf` is
        @test EpiBranch.followup_end(
            _NoFollowup(
                adjacency, inf, infectious, removal,
                index, 10.0
            )
        ) == Inf
        closed = min.(removal, 12.0)
        for α in (0.0, 0.05)
            @test pairwise_surv_loglik(
                wk,
                _NoFollowup(adjacency, inf, infectious, closed, index, 10.0);
                external_hazard = α
            ) ==
                pairwise_surv_loglik(
                wk,
                _TestInfections(
                    adjacency, inf, infectious, closed, index;
                    obs_end = 10.0
                ); external_hazard = α
            )
        end

        bad = _TestInfections(
            contacts, [0.0, 1.0, NaN], [0.0, 1.0, NaN],
            [2.0, Inf, NaN], [true, false, false]; followup_end = NaN
        )
        @test_throws ArgumentError pairwise_surv_loglik(k, bad)
    end

    @testset "host_times reads a fixed field name, overridable" begin
        # a layer without the field holds no extra host times
        plain = _TestInfections(
            contacts, [0.0, 1.0, NaN], [0.0, 1.0, NaN],
            [2.0, Inf, NaN], [true, false, false]; obs_end = 5.0
        )
        @test EpiBranch.host_times(plain) == (;)

        # a layer holding per-host times under a name of its own is invisible
        # until it defines its own method, the shape `followup_end` sets
        layer = _NamedTimes([[2], [1]], (onset_time = [1.0, 2.0],))
        @test EpiBranch.host_times(layer) == (;)
        EpiBranch.host_times(d::_NamedTimes) = d.times
        @test EpiBranch.host_times(layer) == (onset_time = [1.0, 2.0],)

        if VERSION >= v"1.11"
            @test Base.ispublic(EpiBranch, :host_times)
        else
            @info "Skipping Base.ispublic check on Julia $VERSION"
            @test_skip true
        end
    end

    @testset "an impossible configuration has a zero gradient" begin
        # Two components. In {1, 2} host 1 is a community case at 0 and infects 2
        # at 1.0, which the parameters do move. In {3, 4} host 4 is infected at
        # 8.0, after obs_end and before its only possible infector is infectious,
        # so nothing can explain it. Whether a configuration is possible at all
        # depends on the times alone, which makes the density -Inf over a whole
        # neighbourhood of the parameters. The derivative must then be exactly
        # zero, and must not pick up the other component's finite terms.
        inf = [0.0, 1.0, 10.0, 8.0]
        removal = [5.0, 6.0, 12.0, 13.0]
        index = [true, false, true, false]
        membership = [1, 1, 2, 2]
        adjacency = [[2], [1], [4], [3]]
        for structure in (membership, adjacency)
            data = _TestInfections(structure, inf, inf, removal, index; obs_end = 5.0)
            L = compile_contact_pairs(data; external = true)
            f(θ) = pairwise_surv_loglik(
                Exponential(exp(θ[1])), data;
                external_hazard = exp(θ[2])
            )
            g(θ) = pairwise_surv_loglik(
                Exponential(exp(θ[1])), data, L;
                external_hazard = exp(θ[2])
            )
            θ = [log(3.0), log(0.1)]
            @test f(θ) == -Inf
            @test g(θ) == -Inf
            @test ForwardDiff.gradient(f, θ) == [0.0, 0.0]
            @test ForwardDiff.gradient(g, θ) == [0.0, 0.0]
            @test DifferentiationInterface.gradient(g, AutoMooncake(), θ) == [0.0, 0.0]
            @test (
                @inferred pairwise_surv_loglik(
                    Exponential(3.0), data, L; external_hazard = 0.1
                )
            ) == -Inf

            # the possible component alone is finite and does move with both
            # parameters, so the zero gradient above is the fix and not an
            # artefact of a flat likelihood
            ok = _TestInfections(
                structure, [0.0, 1.0, NaN, NaN], [0.0, 1.0, NaN, NaN],
                [5.0, 6.0, Inf, Inf], index; obs_end = 5.0
            )
            okL = compile_contact_pairs(ok; external = true)
            h(θ) = pairwise_surv_loglik(
                Exponential(exp(θ[1])), ok, okL;
                external_hazard = exp(θ[2])
            )
            @test isfinite(h(θ))
            @test all(!iszero, ForwardDiff.gradient(h, θ))
        end
    end

    @testset "differentiable in the kernel parameters" begin
        _, adjacency = _cliques([3, 4, 2, 4])
        inf = [0.0, 1.2, NaN, 0.0, 2.1, 3.5, NaN, 0.0, NaN, 0.0, 0.7, NaN, 4.2]
        data = _TestInfections(
            adjacency, inf, inf, inf .+ 4.0,
            .!isnan.(inf) .& (inf .== 0.0); obs_end = 10.0
        )
        L = compile_contact_pairs(data)
        f(θ) = pairwise_surv_loglik(Weibull(exp(θ[1]), exp(θ[2])), data, L)
        θ = [log(1.5), log(3.0)]
        g = ForwardDiff.gradient(f, θ)
        h = 1.0e-6
        fd = [(f(θ .+ h .* e) - f(θ .- h .* e)) / 2h for e in ([1.0, 0.0], [0.0, 1.0])]
        @test all(isfinite, g)
        @test g ≈ fd rtol = 1.0e-5

        # the dual passes through the edge lookup of a per-edge kernel
        pe(s) = [[Exponential(s) for _ in nbrs] for nbrs in adjacency]
        fe(s) = pairwise_surv_loglik(pe(s), data, L)
        fs(s) = pairwise_surv_loglik(Exponential(s), data, L)
        @test ForwardDiff.derivative(fe, 2.5) ≈ ForwardDiff.derivative(fs, 2.5)

        # a kernel whose first internal pair holds no fitted parameter
        first_inf = L.infector[findfirst(!, L.is_ext)]
        fc(s) = pairwise_surv_loglik(
            (i, j) -> i == first_inf ? Exponential(3.0) : Exponential(s), data, L
        )
        dc = ForwardDiff.derivative(fc, 2.5)
        @test dc ≈ (fc(2.5 + 1.0e-6) - fc(2.5 - 1.0e-6)) / 2.0e-6 rtol = 1.0e-5
        pm(s) = [
            Distribution[
                i == first_inf ? Exponential(3.0) : Exponential(s)
                    for _ in nbrs
            ] for (i, nbrs) in enumerate(adjacency)
        ]
        @test ForwardDiff.derivative(s -> pairwise_surv_loglik(pm(s), data, L), 2.5) ≈ dc
    end

    @testset "per-component contributions" begin
        # a sampler updating one household at a time needs that household's
        # share of the log-likelihood, not the total
        membership, adjacency = _cliques([3, 4, 2, 4])
        inf = [0.0, 1.2, NaN, 0.0, 2.1, 3.5, NaN, 0.0, NaN, 0.0, 0.7, NaN, 4.2]
        infectious = inf .+ 0.5
        removal = infectious .+ 4.0
        is_index = [
            true, false, false, true, false, false, false, true, false,
            true, false, false, false,
        ]
        hh = _TestInfections(
            membership, inf, infectious, removal, is_index;
            obs_end = 10.0
        )
        net = _TestInfections(
            adjacency, inf, infectious, removal, is_index;
            obs_end = 10.0
        )
        for external in (false, true)
            Lh = compile_contact_pairs(hh; external)
            Ln = compile_contact_pairs(net; external)
            @test Lh.component == membership
            @test Lh.component == Ln.component
            @test Lh.ncomponents == Ln.ncomponents == 4
            α = external ? 0.05 : 0.0
            for k in (Exponential(2.5), Weibull(1.5, 3.0))
                total = pairwise_surv_loglik(k, hh, Lh; external_hazard = α)
                by_component = pairwise_surv_loglik_by_component(k, hh, Lh; external_hazard = α)
                @test length(by_component) == 4
                @test sum(by_component) ≈ total
                # a household on a network layout gives the same breakdown
                @test by_component ≈
                    pairwise_surv_loglik_by_component(k, net, Ln; external_hazard = α)
                # the two-argument form compiles its own layout
                @test pairwise_surv_loglik_by_component(k, hh; external_hazard = α) ≈
                    by_component
            end
        end

        # an infection no possible infector can explain only zeroes out its own
        # household's density; the other household stays finite
        structure = [1, 1, 2, 2]
        inf2 = [0.0, 1.0, 10.0, 8.0]
        removal2 = [5.0, 6.0, 12.0, 13.0]
        index2 = [true, false, true, false]
        data = _TestInfections(structure, inf2, inf2, removal2, index2; obs_end = 5.0)
        L = compile_contact_pairs(data; external = true)
        k = Exponential(3.0)
        α = 0.1
        by_component = pairwise_surv_loglik_by_component(k, data, L; external_hazard = α)
        @test pairwise_surv_loglik(k, data, L; external_hazard = α) == -Inf
        broken_household = L.component[3]
        ok_household = L.component[1]
        @test by_component[broken_household] == -Inf
        @test isfinite(by_component[ok_household])

        # the finite household's contribution matches evaluating it on its own
        solo = _TestInfections(
            [1, 1], [0.0, 1.0], [0.0, 1.0], [5.0, 6.0], [true, false];
            obs_end = 5.0
        )
        @test by_component[ok_household] ≈
            pairwise_surv_loglik(k, solo; external_hazard = α)
    end

    @testset "per-component contributions differentiable in the kernel parameters" begin
        # as for the total's "a kernel whose first internal pair holds no
        # fitted parameter", but summed by component rather than into one
        # scalar: a row the `T` probe missed still makes `_add!`, now shared
        # machinery for both entry points, widen the running totals
        _, adjacency = _cliques([3, 4, 2, 4])
        inf = [0.0, 1.2, NaN, 0.0, 2.1, 3.5, NaN, 0.0, NaN, 0.0, 0.7, NaN, 4.2]
        data = _TestInfections(
            adjacency, inf, inf, inf .+ 4.0,
            .!isnan.(inf) .& (inf .== 0.0); obs_end = 10.0
        )
        L = compile_contact_pairs(data)
        first_inf = L.infector[findfirst(!, L.is_ext)]
        fc(s) = pairwise_surv_loglik_by_component(
            (i, j) -> i == first_inf ? Exponential(3.0) : Exponential(s), data, L
        )
        dc = ForwardDiff.derivative(fc, 2.5)
        fd = (fc(2.5 + 1.0e-6) .- fc(2.5 - 1.0e-6)) ./ 2.0e-6
        @test dc ≈ fd rtol = 1.0e-5
        @test sum(dc) ≈ ForwardDiff.derivative(
            s -> pairwise_surv_loglik(
                (i, j) -> i == first_inf ? Exponential(3.0) : Exponential(s), data, L
            ), 2.5
        )
    end

    @testset "a new grouping needs no change to pairwise_survival.jl" begin
        # pairwise_surv_loglik and pairwise_surv_loglik_by_component are two
        # groupings of the same two passes; a third (here, by parity) is
        # written from outside with the same two tiny methods and reuses them
        # unchanged.
        membership, _ = _cliques([3, 4, 2, 4])
        inf = [0.0, 1.2, NaN, 0.0, 2.1, 3.5, NaN, 0.0, NaN, 0.0, 0.7, NaN, 4.2]
        infectious = inf .+ 0.5
        removal = infectious .+ 4.0
        is_index = [
            true, false, false, true, false, false, false, true, false,
            true, false, false, false,
        ]
        hh = _TestInfections(membership, inf, infectious, removal, is_index; obs_end = 10.0)
        L = compile_contact_pairs(hh; external = true)
        k = Exponential(2.5)

        total = pairwise_surv_loglik(k, hh, L; external_hazard = 0.05)
        by_parity = EpiBranch.pairwise_reduce(_Parity(), k, hh, L; external_hazard = 0.05)
        @test length(by_parity) == 2
        @test sum(by_parity) ≈ total
    end
end

@testset "Removals that lapse leave their stretches out of the exposure" begin
    # Two hosts in one household: host 1 infectious over [0, 30], host 2 never
    # infected, so the pair is pure escape. An exponential kernel integrates to
    # `exposed / theta`, which makes the subtraction exact to read.
    adj = [1, 1]
    theta = 4.0
    k = Exponential(theta)
    none = Tuple{Float64, Float64}[]
    layer(host_times) = _GappedInfections(
        adj, [0.0, NaN], [0.0, NaN], [30.0, NaN], [true, false], Inf, Inf, host_times
    )
    gapped(stretches) = layer((_removal_stretches = [stretches, none],))

    @test pairwise_surv_loglik(k, layer((;))) ≈ -30.0 / theta

    # Isolated over [4, 11]: seven days out of the thirty.
    @test pairwise_surv_loglik(k, gapped([(4.0, 11.0)])) ≈ -(30.0 - 7.0) / theta

    # Quarantined over [4, 11], released, then isolated again over [18, 22]:
    # both stretches come out, and one pair of times could not have held them.
    @test pairwise_surv_loglik(k, gapped([(4.0, 11.0), (18.0, 22.0)])) ≈
        -(30.0 - 7.0 - 4.0) / theta

    # A removal with no release ends the exposure where it starts. The built-in
    # removals close the infectious window there through
    # `infectious_removal_time`, so the two agree and nothing is taken out
    # twice; a layer whose removal time runs past such a stretch, which a
    # removal written outside the package can leave, loses the days after it
    # rather than being fitted on days the simulation blocked.
    @test pairwise_surv_loglik(k, gapped([(4.0, Inf)])) ≈ -4.0 / theta
    closed = _GappedInfections(
        adj, [0.0, NaN], [0.0, NaN], [4.0, NaN], [true, false], Inf, Inf,
        (_removal_stretches = [[(4.0, Inf)], none],)
    )
    @test pairwise_surv_loglik(k, closed) ≈ -4.0 / theta
    # A stretch that lapses and a later one that never does: the first comes
    # out, and the exposure ends at the second.
    @test pairwise_surv_loglik(k, gapped([(4.0, 11.0), (18.0, Inf)])) ≈
        -(18.0 - 7.0) / theta

    # A stretch reaching past the end of the exposure is clamped to it, and one
    # starting after the end takes nothing out.
    @test pairwise_surv_loglik(k, gapped([(20.0, 40.0)])) ≈ -20.0 / theta
    @test pairwise_surv_loglik(k, gapped([(4.0, 11.0), (40.0, 50.0)])) ≈
        -(30.0 - 7.0) / theta
end

@testset "An infector removed at the infection cannot be the one" begin
    adj = [1, 1]
    k = Exponential(4.0)
    none = Tuple{Float64, Float64}[]
    gap = (_removal_stretches = [[(4.0, 11.0), (18.0, 22.0)], none],)

    # Host 1 is host 2's only possible infector and was removed at day 7, so
    # the configuration has density zero. The same holds inside the second
    # stretch, which one pair of times would have lost.
    for t in (7.0, 20.0)
        inside = _GappedInfections(
            adj, [0.0, t], [0.0, t], [30.0, 30.0 + t], [true, false], Inf, Inf, gap
        )
        @test pairwise_surv_loglik(k, inside) == -Inf
    end

    # Between the two stretches, and after the last, it is an ordinary event.
    for t in (15.0, 25.0)
        between = _GappedInfections(
            adj, [0.0, t], [0.0, t], [30.0, 30.0 + t], [true, false], Inf, Inf, gap
        )
        @test isfinite(pairwise_surv_loglik(k, between))
    end
end

@testset "A gap past a bounded kernel's support keeps the finite head" begin
    # A kernel of bounded support has an infinite cumulative hazard past it,
    # and taking a gap out as the whole exposure less the gap would subtract
    # an infinity from itself. Once survival reaches zero no mass is left, so
    # the stretch after the release contributes nothing and the head alone is
    # the answer: escape past the support is possible precisely because the
    # host was removed over the tail.
    adj = [1, 1]
    none = Tuple{Float64, Float64}[]
    layer(stretches) = _GappedInfections(
        adj, [0.0, NaN], [0.0, NaN], [30.0, NaN], [true, false], Inf, Inf,
        (_removal_stretches = [stretches, none],)
    )
    head = log(1 - 4 / 10)

    # Each of these has a gap end at or past the support while the exposure
    # runs on to day 30, so each is a place the two infinities would meet.
    for (a, b) in ((4.0, 10.0), (4.0, 20.0), (4.0, 40.0))
        v = pairwise_surv_loglik(Uniform(0, 10), layer([(a, b)]))
        @test !isnan(v)
        @test v ≈ head
    end

    # Two stretches, the second reaching past the support. What survives is
    # [0, 4] and [6, 8], both inside it, so the answer stays finite: the
    # cumulative hazards add to `-log(0.6) - log(0.5)`.
    v = pairwise_surv_loglik(Uniform(0, 10), layer([(4.0, 6.0), (8.0, 40.0)]))
    @test !isnan(v)
    @test v ≈ log(0.3)

    # A gap wholly past the support leaves the whole of the kernel's mass in
    # the exposure, so the susceptible cannot escape.
    @test pairwise_surv_loglik(Uniform(0, 10), layer([(15.0, 20.0)])) == -Inf

    # A gap strictly inside the support leaves the tail uncovered, which is
    # impossible to escape for the same reason.
    @test pairwise_surv_loglik(Uniform(0, 10), layer([(2.0, 4.0)])) == -Inf

    # An unbounded kernel takes the ordinary path.
    @test pairwise_surv_loglik(Exponential(4.0), layer([(4.0, 11.0)])) ≈
        -(30.0 - 7.0) / 4.0
end

@testset "A host no removal reached contributes no gap" begin
    # `_host_time_columns` writes `missing` for a host that never had the key,
    # which a quarantine-only model leaves for everyone it did not reach.
    adj = [1, 1]
    theta = 4.0
    none = Tuple{Float64, Float64}[]
    column(values) = Union{Missing, Vector{Tuple{Float64, Float64}}}[values...]
    layer(values) = _GappedInfections(
        adj, [0.0, NaN], [0.0, NaN], [30.0, NaN], [true, false], Inf, Inf,
        (_removal_stretches = column(values),)
    )

    @test pairwise_surv_loglik(Exponential(theta), layer((missing, missing))) ≈
        -30.0 / theta

    # Mixed: host 1 was quarantined, host 2 never reached.
    @test pairwise_surv_loglik(
        Exponential(theta), layer(([(4.0, 11.0)], missing))
    ) ≈ -(30.0 - 7.0) / theta

    # An empty list is the same as a missing column.
    @test pairwise_surv_loglik(Exponential(theta), layer((none, missing))) ≈
        -30.0 / theta
end

struct _IdentitySusceptibility <: EpiBranch.AbstractIntervention end
EpiBranch.susceptibility_components(::_IdentitySusceptibility, host) =
    (1.0 => EpiBranch.HazardScaling(0.0, 1.0),)
EpiBranch.infection_likelihood_compatible(::_IdentitySusceptibility) = true

@testset "The mixture path takes out the same stretch" begin
    # `_component_loglik` keeps its own exposure loop for a susceptible whose
    # susceptibility an effect modifies, so it has to take the infector's
    # isolated stretch out as well. Under a modifier that changes nothing the
    # two paths must agree exactly.
    adj = [1, 1]
    theta = 4.0
    layer(host_times) = _GappedInfections(
        adj, [0.0, NaN], [0.0, NaN], [30.0, NaN], [true, false], Inf, Inf, host_times
    )
    none = Tuple{Float64, Float64}[]
    gap = (_removal_stretches = [[(4.0, 11.0)], none],)
    two = (_removal_stretches = [[(4.0, 11.0), (18.0, 22.0)], none],)

    flat = pairwise_surv_loglik(Exponential(theta), layer(gap))
    mixed = pairwise_surv_loglik(
        Exponential(theta), layer(gap); susceptibility = _IdentitySusceptibility()
    )
    @test flat ≈ -(30.0 - 7.0) / theta
    @test mixed ≈ flat

    # Two stretches on one host, so the path is not reading only the first.
    @test pairwise_surv_loglik(
        Exponential(theta), layer(two); susceptibility = _IdentitySusceptibility()
    ) ≈ -(30.0 - 7.0 - 4.0) / theta

    # Without the gap both paths give the whole exposure, so the mixture path
    # is not simply ignoring the host times.
    @test pairwise_surv_loglik(
        Exponential(theta), layer((;)); susceptibility = _IdentitySusceptibility()
    ) ≈ -30.0 / theta

    # An infector removed at the moment of infection is dropped from the
    # susceptible's log-sum-exp on this path too, so a susceptible whose only
    # possible infector was isolated then has density zero.
    inside = _GappedInfections(
        adj, [0.0, 7.0], [0.0, 7.0], [30.0, 37.0], [true, false], Inf, Inf, gap
    )
    @test pairwise_surv_loglik(
        Exponential(theta), inside; susceptibility = _IdentitySusceptibility()
    ) == -Inf

    # A bounded kernel reaches the same infinity guard on this path.
    bounded = pairwise_surv_loglik(
        Uniform(0, 10), layer((_removal_stretches = [[(4.0, 20.0)], none],));
        susceptibility = _IdentitySusceptibility()
    )
    @test !isnan(bounded)
    @test bounded ≈ log(1 - 4 / 10)
end
