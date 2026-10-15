# `SourceTracing` traces a case back to its own infector, the reverse of
# `ContactTracing`'s forward reach, through the same eligibility/rate/delay/
# action protocol and the same `trace_contacts!` verb, which both engines call
# a second time per case for an intervention declaring `traces_source`.

@testset "SourceTracing trait seams" begin
    @testset "Constructors lower kwargs to trait form" begin
        st = SourceTracing(
            probability = 0.5, isolation_to_trace_delay = Exponential(1.0),
            action = Quarantine(duration = Inf)
        )
        @test st.eligibility isa SymptomaticParent
        @test st.trace_rate isa ConstantRate
        @test st.trace_rate.p == 0.5
        @test st.isolation_to_trace_delay isa ConstantDelay
        @test st.action isa Quarantine

        st2 = SourceTracing(OnSymptomOnset(), 0.7, Exponential(2.0), FlagOnly())
        @test st2.eligibility isa OnSymptomOnset
        @test st2.trace_rate isa ConstantRate
        @test st2.trace_rate.p == 0.7
        @test st2.action isa FlagOnly
    end

    @testset "action is required" begin
        @test_throws UndefKeywordError SourceTracing(
            probability = 1.0, isolation_to_trace_delay = Exponential(1.0)
        )
    end

    @testset "SourceTracing traces backward only; ContactTracing forward only" begin
        st = SourceTracing(
            probability = 1.0, isolation_to_trace_delay = Exponential(1.0),
            action = Quarantine(duration = Inf)
        )
        @test !EpiBranch.traces_contacts(st)
        @test EpiBranch.traces_source(st)

        ct = ContactTracing(
            probability = 1.0, isolation_to_trace_delay = Exponential(1.0),
            action = Quarantine(duration = Inf)
        )
        @test EpiBranch.traces_contacts(ct)
        @test !EpiBranch.traces_source(ct)
    end

    @testset "required_fields mirrors the eligibility, as for ContactTracing" begin
        st = SourceTracing(
            OnLabConfirmation(), 1.0, Exponential(1.0), Quarantine(duration = Inf)
        )
        @test EpiBranch.required_fields(st) == EpiBranch.required_fields(OnLabConfirmation())
    end

    @testset "trace_contacts! quarantines an eligible case's own infector" begin
        st = SourceTracing(TraceEveryone(), 1.0, Dirac(0.0), Quarantine(duration = Inf))
        source = Individual(id = 1)
        case = Individual(id = 2, parent_id = 1)
        # `TraceEveryone`'s default `trigger_time` is the case's own isolation.
        set_isolated!(case, 0.0; release_time = Inf)
        state = SimulationState(
            [source, case], Int[], 1, StableRNG(1), 0, false, nothing, Inf,
            nothing, AbstractClinicalTransition[]
        )
        EpiBranch.trace_contacts!(st, state, case, (source,))
        @test is_traced(source)
        @test is_quarantined(source)
        @test source.state[:source_traced_by] == case.id
        @test source.state[:trace_time] == 0.0
        # The reverse key, `:traced_by`, is `ContactTracing`'s own and is left
        # untouched so `compute_trace_level!`'s invariant (it always names an
        # earlier-infected node) survives combining the two.
        @test !haskey(source.state, :traced_by)
    end

    @testset "An ineligible case never reaches its infector" begin
        # The default eligibility, SymptomaticParent, needs the case isolated;
        # this one never is.
        st = SourceTracing(
            probability = 1.0, isolation_to_trace_delay = Dirac(0.0),
            action = Quarantine(duration = Inf)
        )
        source = Individual(id = 1)
        case = Individual(id = 2, parent_id = 1)
        case.state[:asymptomatic] = false
        state = SimulationState(
            [source, case], Int[], 1, StableRNG(1), 0, false, nothing, Inf,
            nothing, AbstractClinicalTransition[]
        )
        EpiBranch.trace_contacts!(st, state, case, (source,))
        @test !is_traced(source)
    end

    @testset "reset! clears the backward tracing keys" begin
        st = SourceTracing(
            probability = 1.0, isolation_to_trace_delay = Dirac(0.0),
            action = Quarantine(duration = Inf)
        )
        ind = Individual(id = 1)
        ind.state[:traced] = true
        ind.state[:source_traced_by] = 7
        set_isolated!(ind, 6.0; release_time = Inf)
        EpiBranch.reset!(st, ind)
        @test ind.state[:traced] == false
        @test !haskey(ind.state, :source_traced_by)
        @test !EpiBranch._isolation_in_force(ind)
    end
end

# Records, on `state.scratch`, every case whose own infector was offered to
# it, so the engine wiring can be checked without depending on which contact
# the infector happens to be stored under. Both engines also call this
# unconditionally in the ordinary forward role, with `case`'s real contacts
# (see `trace_contacts!`'s docstring), which the id check below ignores.
struct _RecordSource <: AbstractIntervention end
EpiBranch.traces_source(::_RecordSource) = true
function EpiBranch.trace_contacts!(::_RecordSource, state, case, contacts)
    length(contacts) == 1 && only(contacts).id == case.parent_id || return nothing
    push!(get!(Vector{Int}, state.scratch, :_contacts_sizes), length(contacts))
    push!(get!(Set{Int}, state.scratch, :_sources_found), case.id)
    return nothing
end

@testset "Both engines offer a settled case's own infector" begin
    @testset "Generation engine, including a leaf case with no offspring" begin
        state = simulate(
            ModelSpec(
                BranchingProcess(Poisson(0.8), Exponential(3.0));
                interventions = [_RecordSource()]
            );
            max_cases = 300, max_generations = 10, n_initial = 1, rng = StableRNG(21)
        )
        found = get(state.scratch, :_sources_found, Set{Int}())
        @test all(==(1), state.scratch[:_contacts_sizes])

        children = filter(ind -> ind.parent_id != 0, state.individuals)
        @test !isempty(children)
        for child in children
            @test child.id in found
        end

        # A leaf (no offspring of its own) still triggers the backward call:
        # unlike the forward hooks, it does not depend on the case having made
        # a contact this generation.
        leaves = filter(c -> isempty(c.secondary_case_ids), children)
        @test !isempty(leaves)
        for leaf in leaves
            @test leaf.id in found
        end

        # The index case has no infector, so it never triggers the hook.
        index_case = only(filter(ind -> ind.parent_id == 0, state.individuals))
        @test index_case.id ∉ found
    end

    @testset "The continuous-time race, with no model contact structure at all" begin
        interventions = AbstractIntervention[_RecordSource()]
        rng = StableRNG(5)
        state = EpiBranch.new_state(
            BranchingProcess(Poisson(1.0), Exponential(1.0)),
            EpiBranch.AbstractClinicalTransition[], NoAttributes(), rng
        )
        EpiBranch.add_individuals!(state, 3, interventions)
        neighbours = Dict(1 => [2], 2 => [1, 3], 3 => [2])
        targets(inf, st) = (
            (to, Dirac(1.0)) for to in neighbours[inf]
                if !get(st.individuals[to].state, :infected, false)
        )
        # No `contacts` function is passed: backward tracing needs none, unlike
        # the forward ring walk, since the infector comes straight off the
        # transmission tree rather than the model's own contact structure.
        EpiBranch._sellke_race!(
            state, [1, 2, 3], rng;
            routes = ((RouteWindow(:one; kernel = Dirac(1.0)), targets),),
            interventions, seed! = (best, members, r) -> (best[1] = 0.0)
        )
        found = get(state.scratch, :_sources_found, Set{Int}())
        # Node 1 is the seed (no infector); 2's infector is 1, 3's is 2.
        @test found == Set([2, 3])
    end
end

@testset "SourceTracing blocks a settled infector's own future transmission" begin
    # A mass-action pool has no pairwise structure for `ContactTracing` to act
    # along (see the "Unhonoured intervention" test in test_homogeneous.jl);
    # `SourceTracing` needs none, since the infector is read off the
    # transmission tree, so this is where its marginal value over `Isolation`
    # alone shows up most clearly. `test_sensitivity` leaves most infectors
    # themselves undetected; their own detected secondary cases find and
    # quarantine them instead.
    N = 500
    prog = [
        Transition(:onset; from = :infection, delay = 0.1),
        Transition(:recovered; from = :infection, delay = Exponential(5.0), terminal = true),
    ]
    isolation = Isolation(
        onset_to_isolation_delay = Exponential(0.5), duration = Inf,
        test_sensitivity = 0.3
    )
    pool = HomogeneousProcess(; transmission_rate = 1.5, population_size = N)
    base = ModelSpec(pool; progression = prog, interventions = [isolation])
    with_source = ModelSpec(
        pool; progression = prog,
        interventions = [
            isolation,
            SourceTracing(
                probability = 1.0, isolation_to_trace_delay = Dirac(0.0),
                action = Quarantine(duration = Inf)
            ),
        ]
    )

    base_mean = mean(
        simulate(base; rng = StableRNG(s), n_initial = 3).cumulative_cases
            for s in 1:30
    )
    source_mean = mean(
        simulate(with_source; rng = StableRNG(s), n_initial = 3).cumulative_cases
            for s in 1:30
    )
    @test source_mean < base_mean

    st = SourceTracing(
        probability = 1.0, isolation_to_trace_delay = Dirac(0.0),
        action = Quarantine(duration = Inf)
    )
    @test EpiBranch._sellke_honours(pool, st)
    @test_logs min_level = Base.CoreLogging.Warn simulate(
        with_source; rng = StableRNG(1), n_initial = 3
    )
end

@testset "On the generation engine, SourceTracing only records the link" begin
    # A case's secondary cases are drawn as one batch, before the case itself
    # is confirmed and can be traced back, so there is nothing left for
    # `SourceTracing` to shorten there (see its docstring): the record is its
    # whole contribution on this engine.
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.0
    )
    iso = Isolation(onset_to_isolation_delay = Exponential(0.1), duration = Inf)
    st = SourceTracing(
        probability = 1.0, isolation_to_trace_delay = Dirac(0.0),
        action = Quarantine(duration = Inf)
    )
    state = simulate(
        ModelSpec(
            BranchingProcess(Poisson(2.0), Exponential(5.0));
            interventions = [iso, st], attributes = clinical
        );
        max_cases = 200, rng = StableRNG(31)
    )
    traced_sources = filter(
        ind -> haskey(ind.state, :source_traced_by), state.individuals
    )
    @test !isempty(traced_sources)
    for source in traced_sources
        case = state.individuals[source.state[:source_traced_by]]
        @test case.parent_id == source.id
    end
end

@testset "ContactTracing and SourceTracing coexist without crossing wires" begin
    clinical = clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5), prob_asymptomatic = 0.0
    )
    iso = Isolation(onset_to_isolation_delay = Exponential(0.5), duration = Inf)
    ct = ContactTracing(
        probability = 1.0, isolation_to_trace_delay = Exponential(0.5),
        action = Quarantine(duration = Inf)
    )
    st = SourceTracing(
        probability = 1.0, isolation_to_trace_delay = Dirac(0.0),
        action = Quarantine(duration = Inf)
    )
    state = simulate(
        ModelSpec(
            BranchingProcess(Poisson(2.0), Exponential(5.0));
            interventions = [iso, ct, st], attributes = clinical
        );
        max_cases = 200, rng = StableRNG(41)
    )

    # `ContactTracing`'s `:traced_by` always names an earlier-infected node,
    # the invariant `compute_trace_level!` relies on; `SourceTracing` never
    # writes it.
    for ind in state.individuals
        haskey(ind.state, :traced_by) || continue
        tracer = state.individuals[ind.state[:traced_by]]
        @test tracer.infection_time <= ind.infection_time
    end
    @test compute_trace_level!(state) === state  # must not cycle or error

    # `:source_traced_by` always names a later-infected node: the reverse.
    for ind in state.individuals
        haskey(ind.state, :source_traced_by) || continue
        tracer = state.individuals[ind.state[:source_traced_by]]
        @test tracer.infection_time >= ind.infection_time
    end
end
