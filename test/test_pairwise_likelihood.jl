using ForwardDiff

# A minimal infection layer over an explicit contact structure, as a companion
# package would define one: the fields the likelihood reads plus the structure.
struct _TestInfections{S} <: InfectionLayer
    structure::S
    infection_time::Vector{Float64}
    infectious_time::Vector{Float64}
    removal_time::Vector{Float64}
    is_index::Vector{Bool}
    obs_end::Float64
end
function _TestInfections(structure, inf, infectious, removal, index; obs_end = Inf)
    _TestInfections(structure, Float64.(inf), Float64.(infectious),
        Float64.(removal), Vector{Bool}(index), Float64(obs_end))
end
EpiBranch.contact_structure(d::_TestInfections) = d.structure

# A layer that forgets to name its structure.
struct _NoStructure <: InfectionLayer end

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
        rows = PairwiseSurvivalData([1, 1, 2, 2], [0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 1.5, 5.0], [true, false, true, false])
        k = Exponential(3.0)
        # one event row per susceptible, so each adds its log-hazard; every row
        # subtracts its cumulative hazard (t/3 for this kernel)
        @test pairwise_surv_loglik(k, rows) ≈ 2 * log(1 / 3) - (2 + 4 + 1.5 + 5) / 3
        @test pairwise_surv_loglik(r -> k, rows) ≈ pairwise_surv_loglik(k, rows)
        @test_throws ArgumentError PairwiseSurvivalData([1], [2.0], [1.0], [true])
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
        @test length(compile_contact_pairs([[1, 2], Int[]], [true, false],
            [true, false])) == 1
        @test_throws ArgumentError compile_contact_pairs([[3], Int[]], [true, false],
            [true, false])
    end

    @testset "evaluation matches hand-built counting-process rows" begin
        # the directed graph above with times: 1 infected at 0, infectious [0, 5];
        # 2 infected at 2, infectious [2, 6]; 3 never infected
        contacts = [[2, 3], [3], [1], Int[]]
        data = _TestInfections(contacts, [0.0, 2.0, NaN, NaN], [0.0, 2.0, NaN, NaN],
            [5.0, 6.0, Inf, Inf], [true, false, false, false])
        k = Weibull(1.5, 3.0)
        # 2←1 at risk for 2 with an event; 3←1 at risk for 5, 3←2 for 4, no event
        rows = PairwiseSurvivalData([2, 3, 3], zeros(3), [2.0, 5.0, 4.0],
            [true, false, false])
        L = compile_contact_pairs(data)
        @test pairwise_surv_loglik(k, data, L) ≈ pairwise_surv_loglik(k, rows)
        @test pairwise_surv_loglik(k, data) == pairwise_surv_loglik(k, data, L)

        # per-edge kernels index the edge a row travels; callables see the pair
        per_edge = [[Weibull(1.5, 3.0), Weibull(1.5, 7.0)], [Weibull(1.5, 11.0)],
            [Weibull(1.5, 1.0)], Weibull{Float64}[]]
        rowk = r -> (Weibull(1.5, 3.0), Weibull(1.5, 7.0), Weibull(1.5, 11.0))[r]
        @test pairwise_surv_loglik(per_edge, data, L) ≈
              pairwise_surv_loglik(rowk, rows)
        pairk = (i, j) -> per_edge[i][findfirst(==(j), contacts[i])]
        @test pairwise_surv_loglik(pairk, data, L) ≈
              pairwise_surv_loglik(per_edge, data, L)

        # a partition layout has no edge list for a per-edge kernel to index
        hh = _TestInfections([1, 1], [0.0, 1.0], [0.0, 1.0], [3.0, 4.0],
            [true, false])
        @test_throws ArgumentError pairwise_surv_loglik([[k], [k]], hh)

        # the external mode and the layout must agree
        @test_throws ArgumentError pairwise_surv_loglik(k, data, L;
            external_hazard = 0.1)
        short = _TestInfections(contacts[1:2], [0.0, 2.0], [0.0, 2.0], [5.0, 6.0],
            [true, false])
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
        is_index = [true, false, false, true, false, false, false, true, false,
            true, false, false, false]
        @test length(inf) == n
        hh = _TestInfections(membership, inf, infectious, removal, is_index;
            obs_end = 10.0)
        net = _TestInfections(adjacency, inf, infectious, removal, is_index;
            obs_end = 10.0)

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
        data = _TestInfections(contacts, [1.0, 4.0, NaN, NaN], [1.0, 4.0, NaN, NaN],
            [6.0, 8.0, Inf, Inf], [true, false, false, false]; obs_end = 3.0)
        L = compile_contact_pairs(data; external = true)
        k = Weibull(1.5, 3.0)
        # community rows stop at the earlier of infection and obs_end, and 2 has no
        # community event; 3 is exposed to 1 over [1, 6] and to 2 over [4, 8]
        rows = PairwiseSurvivalData([1, 2, 2, 3, 3, 3, 4],
            zeros(7), [1.0, 3.0, 3.0, 3.0, 5.0, 4.0, 3.0],
            [true, false, true, false, false, false, false])
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
        data = _TestInfections([1, 1, 2], [0.0, NaN, 0.0], [0.0, NaN, 0.0],
            [3.0, Inf, 3.0], [true, false, true]; obs_end = 5.0)
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
        data = _TestInfections([1, 1, 1], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            [5.0, 5.0, 6.0], [true, true, false])
        @test pairwise_surv_loglik(Uniform(2.0, 10.0), data) == -Inf
    end

    @testset "an infection no source can explain has zero density" begin
        k = Exponential(2.0)
        # without a community hazard: index 1 is infectious over [0, 3] and 2 is
        # infected at t, which 1 can explain only up to 3
        for (t, expected) in ((2.9, log(1 / 2) - 2.9 / 2), (3.5, -Inf), (5.0, -Inf))
            data = _TestInfections([1, 1], [0.0, t], [0.0, t], [3.0, t + 3.0],
                [true, false])
            @test pairwise_surv_loglik(k, data) ≈ expected
        end

        # with a community hazard 0.1 up to obs_end = 4: 1 is a community case at
        # 0.5, infectious over [0.5, 3], and after 4 only 1 could infect 2
        for (t, expected) in ((3.5, 2 * log(0.1) - 0.05 - 0.35 - 2.5 / 2),
            (4.5, -Inf), (6.0, -Inf))
            data = _TestInfections([1, 1], [0.5, t], [0.5, t], [3.0, t + 3.0],
                [false, false]; obs_end = 4.0)
            @test pairwise_surv_loglik(k, data; external_hazard = 0.1) ≈ expected
        end

        # a host with no possible infector at all: 1 infects 2 and nobody lists 3
        contacts = [[2], Int[], Int[]]
        infected_3 = _TestInfections(contacts, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
            [4.0, 5.0, 6.0], [true, false, false])
        @test pairwise_surv_loglik(k, infected_3) == -Inf
        escaped_3 = _TestInfections(contacts, [0.0, 1.0, NaN], [0.0, 1.0, NaN],
            [4.0, 5.0, Inf], [true, false, false])
        @test pairwise_surv_loglik(k, escaped_3) ≈ log(1 / 2) - 1 / 2
        # an index case is conditioned on, so it needs no possible infector
        index_3 = _TestInfections(contacts, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
            [4.0, 5.0, 6.0], [true, false, true])
        @test isempty(compile_contact_pairs(index_3).no_rows)
        @test pairwise_surv_loglik(k, index_3) ≈ log(1 / 2) - 1 / 2
    end

    @testset "a case still infectious at the end of follow-up" begin
        # path 1-2-3 followed up to 5: index 1 infectious over [0, 2] infects 2 at
        # 1, which is still infectious at 5, and 3 has escaped until then
        contacts = [[2], [1, 3], [2]]
        k = Exponential(2.0)
        censored = _TestInfections(contacts, [0.0, 1.0, NaN], [0.0, 1.0, NaN],
            [2.0, 5.0, NaN], [true, false, false]; obs_end = 5.0)
        @test pairwise_surv_loglik(k, censored) ≈ log(1 / 2) - 1 / 2 - 4 / 2
        # 3 is also exposed to the community until 5
        @test pairwise_surv_loglik(k, censored; external_hazard = 0.1) ≈
              log(0.1) + log(1 / 2 + 0.1) - 0.1 - 1 / 2 - 4 / 2 - 0.5
        # without the follow-up time 3 is exposed to 2 for ever
        unbounded = _TestInfections(contacts, [0.0, 1.0, NaN], [0.0, 1.0, NaN],
            [2.0, Inf, NaN], [true, false, false]; obs_end = 5.0)
        @test pairwise_surv_loglik(k, unbounded) == -Inf
    end

    @testset "an impossible configuration has a zero gradient" begin
        # Two components. In {1, 2} host 1 is a community case at 0 and infects 2
        # at 1.0, which the parameters do move. In {3, 4} host 4 is infected at
        # 8.0, after obs_end and before its only possible infector is infectious,
        # so nothing can explain it. The density is -Inf over a whole
        # neighbourhood of the parameters — possibility is fixed by the times —
        # so the derivative must be exactly zero, and must not pick up the other
        # component's finite terms.
        inf = [0.0, 1.0, 10.0, 8.0]
        removal = [5.0, 6.0, 12.0, 13.0]
        index = [true, false, true, false]
        membership = [1, 1, 2, 2]
        adjacency = [[2], [1], [4], [3]]
        for structure in (membership, adjacency)
            data = _TestInfections(structure, inf, inf, removal, index; obs_end = 5.0)
            L = compile_contact_pairs(data; external = true)
            f(θ) = pairwise_surv_loglik(Exponential(exp(θ[1])), data;
                external_hazard = exp(θ[2]))
            g(θ) = pairwise_surv_loglik(Exponential(exp(θ[1])), data, L;
                external_hazard = exp(θ[2]))
            θ = [log(3.0), log(0.1)]
            @test f(θ) == -Inf
            @test g(θ) == -Inf
            @test ForwardDiff.gradient(f, θ) == [0.0, 0.0]
            @test ForwardDiff.gradient(g, θ) == [0.0, 0.0]
            @test (@inferred pairwise_surv_loglik(
                Exponential(3.0), data, L; external_hazard = 0.1)) == -Inf

            # the possible component alone is finite and does move with both
            # parameters, so the zero gradient above is the fix and not an
            # artefact of a flat likelihood
            ok = _TestInfections(structure, [0.0, 1.0, NaN, NaN], [0.0, 1.0, NaN, NaN],
                [5.0, 6.0, Inf, Inf], index; obs_end = 5.0)
            okL = compile_contact_pairs(ok; external = true)
            h(θ) = pairwise_surv_loglik(Exponential(exp(θ[1])), ok, okL;
                external_hazard = exp(θ[2]))
            @test isfinite(h(θ))
            @test all(!iszero, ForwardDiff.gradient(h, θ))
        end
    end

    @testset "differentiable in the kernel parameters" begin
        _, adjacency = _cliques([3, 4, 2, 4])
        inf = [0.0, 1.2, NaN, 0.0, 2.1, 3.5, NaN, 0.0, NaN, 0.0, 0.7, NaN, 4.2]
        data = _TestInfections(adjacency, inf, inf, inf .+ 4.0,
            .!isnan.(inf) .& (inf .== 0.0); obs_end = 10.0)
        L = compile_contact_pairs(data)
        f(θ) = pairwise_surv_loglik(Weibull(exp(θ[1]), exp(θ[2])), data, L)
        θ = [log(1.5), log(3.0)]
        g = ForwardDiff.gradient(f, θ)
        h = 1e-6
        fd = [(f(θ .+ h .* e) - f(θ .- h .* e)) / 2h for e in ([1.0, 0.0], [0.0, 1.0])]
        @test all(isfinite, g)
        @test g ≈ fd rtol = 1e-5

        # the dual passes through the edge lookup of a per-edge kernel
        pe(s) = [[Exponential(s) for _ in nbrs] for nbrs in adjacency]
        fe(s) = pairwise_surv_loglik(pe(s), data, L)
        fs(s) = pairwise_surv_loglik(Exponential(s), data, L)
        @test ForwardDiff.derivative(fe, 2.5) ≈ ForwardDiff.derivative(fs, 2.5)

        # a kernel whose first internal pair carries no fitted parameter
        first_inf = L.infector[findfirst(!, L.is_ext)]
        fc(s) = pairwise_surv_loglik(
            (i, j) -> i == first_inf ? Exponential(3.0) : Exponential(s), data, L)
        dc = ForwardDiff.derivative(fc, 2.5)
        @test dc ≈ (fc(2.5 + 1e-6) - fc(2.5 - 1e-6)) / 2e-6 rtol = 1e-5
        pm(s) = [Distribution[i == first_inf ? Exponential(3.0) : Exponential(s)
                              for _ in nbrs] for (i, nbrs) in enumerate(adjacency)]
        @test ForwardDiff.derivative(s -> pairwise_surv_loglik(pm(s), data, L), 2.5) ≈ dc
    end
end
