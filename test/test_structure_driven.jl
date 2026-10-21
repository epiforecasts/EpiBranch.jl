using StableRNGs
using Random

# A minimal structure-driven model — a ring of nodes — exercising the race's
# `contacts_of` route from within core: a node exposed by both neighbours
# (decided once, by the earliest exposure that transmits), a model-contributed
# competing risk, and the "exposed but not infected" branch. The network and household
# subpackages cover these too, but only under their own coverage flags; a tiny
# model here keeps the extension contract tested in core.

struct RingModel <: EpiBranch.TransmissionModel
    n::Int
    p::Float64          # per-edge transmission probability
end

# Depletion comes from the ring structure (a node is infected at most once), not
# from a population-level susceptible pool — so, like a network, no population.
EpiBranch.population_size(::RingModel) = EpiBranch.NoPopulation()

function EpiBranch.initialise_state(
        m::RingModel, sim_opts::EpiBranch.SimOpts,
        interventions, transitions, attributes, rng::AbstractRNG
    )
    state = EpiBranch.new_state(m, transitions, attributes, rng)
    EpiBranch.add_individuals!(state, m.n, interventions)
    EpiBranch.seed!(state, 1:(sim_opts.n_initial), interventions, transitions)
    return state
end

function EpiBranch.contacts_of(m::RingModel, parent, state::EpiBranch.SimulationState)
    i = parent.id
    neighbours = (i == 1 ? m.n : i - 1, i == m.n ? 1 : i + 1)
    result = Tuple{eltype(state.individuals), Float64}[]
    for nb in neighbours
        target = state.individuals[nb]
        EpiBranch.is_infected(target) && continue
        push!(result, (target, parent.infection_time + rand(state.rng)))
    end
    return result
end

# The per-edge probability enters as a model competing risk (the `model_risks`
# the race resolves on each exposure); p < 1 lets some exposed nodes escape.
struct RingRisk
    p::Float64
end
EpiBranch.transmission_risks(m::RingModel) = (RingRisk(m.p),)
function EpiBranch.competing_risk(r::RingRisk, parent, contact, state)
    return r.p < 1.0 ? EpiBranch.Risk(block_probability = 1.0 - r.p) : nothing
end

@testset "Structure-driven model (core extension path)" begin
    # p < 1 so some exposures fail — exercises the exposed-but-not-infected
    # branch; the ring guarantees a node is reached from both sides.
    state = simulate(
        RingModel(50, 0.5); n_initial = 1, rng = StableRNG(11),
        stopping_rules = [Extinction(), MaxGenerations(50)]
    )
    infected = count(ind -> get(ind.state, :infected, false), state.individuals)
    @test length(state.individuals) == 50            # fixed, pre-instantiated pool
    @test 1 <= infected <= 50
    @test EpiBranch._timetype(state) === Float64     # generic (non-BranchingProcess)

    # a fully-transmitting ring (p = 1) infects the whole ring
    full = simulate(
        RingModel(20, 1.0); n_initial = 1, rng = StableRNG(3),
        stopping_rules = [Extinction(), MaxGenerations(50)]
    )
    @test count(ind -> get(ind.state, :infected, false), full.individuals) == 20
end

# A structure-driven model whose `contacts_of` *mints* fresh contacts with
# `make_contact!`, so the race grows its members as it goes. It uses the default
# `initialise_state` (offspring-style seeds).
struct MintModel <: EpiBranch.TransmissionModel
    k::Int
end
EpiBranch.population_size(::MintModel) = EpiBranch.NoPopulation()
function EpiBranch.contacts_of(m::MintModel, parent, state::EpiBranch.SimulationState)
    result = Tuple{eltype(state.individuals), Float64}[]
    for _ in 1:(m.k)
        t = parent.infection_time + rand(state.rng)
        push!(result, (EpiBranch.make_contact!(state, parent, t), t))
    end
    return result
end

@testset "Structure-driven model minting fresh contacts" begin
    state = simulate(
        MintModel(2); n_initial = 1, rng = StableRNG(7),
        stopping_rules = [Extinction(), MaxGenerations(4)]
    )
    @test length(state.individuals) > 1            # fresh contacts were minted
    @test EpiBranch._timetype(state) === Float64
end

struct MintRiskModel <: EpiBranch.TransmissionModel
    k::Int
end
EpiBranch.population_size(::MintRiskModel) = EpiBranch.NoPopulation()
EpiBranch.contacts_of(m::MintRiskModel, parent, state::EpiBranch.SimulationState) =
    EpiBranch.contacts_of(MintModel(m.k), parent, state)
EpiBranch.transmission_risks(::MintRiskModel) = (RingRisk(0.5),)

@testset "A blocked contact a model creates is listed once" begin
    spec = ModelSpec(MintRiskModel(2); recorder = ExposedContactRecorder())
    state = simulate(
        spec; n_initial = 1, rng = StableRNG(1),
        stopping_rules = [Extinction(), MaxGenerations(4)]
    )
    pairs = contacts(state)
    @test any(!, pairs.infected)
    @test allunique(zip(pairs.from, pairs.to))
end

# A pool of pre-created nodes, each parent reaching random members, so a node
# can be exposed, escape, and be exposed again by a later case. That is
# the path on which a post-exposure abort drawn for one exposure could outlive
# it.
struct AbortPoolModel <: EpiBranch.TransmissionModel
    n::Int
    k::Int
    p::Float64
end
EpiBranch.population_size(::AbortPoolModel) = EpiBranch.NoPopulation()
function EpiBranch.initialise_state(
        m::AbortPoolModel, sim_opts::EpiBranch.SimOpts,
        interventions, transitions, attributes, rng::AbstractRNG
    )
    state = EpiBranch.new_state(m, transitions, attributes, rng)
    EpiBranch.add_individuals!(state, m.n, interventions)
    EpiBranch.seed!(state, 1:(sim_opts.n_initial), interventions, transitions)
    return state
end
function EpiBranch.contacts_of(m::AbortPoolModel, parent, state::EpiBranch.SimulationState)
    result = Tuple{eltype(state.individuals), Float64}[]
    for _ in 1:(m.k)
        target = state.individuals[rand(state.rng, 1:(m.n))]
        (EpiBranch.is_infected(target) || target.id == parent.id) && continue
        push!(result, (target, parent.infection_time + rand(state.rng, Exponential(4.0))))
    end
    return result
end
EpiBranch.transmission_risks(m::AbortPoolModel) = (RingRisk(m.p),)

@testset "A post-exposure abort belongs to the exposure it was drawn for" begin
    spec = ModelSpec(
        AbortPoolModel(300, 4, 0.4);
        interventions = [
            Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf),
            ContactTracing(
                probability = 1.0,
                isolation_to_trace_delay = Exponential(0.5),
                action = FlagOnly()
            ),
            RingVaccination(efficacy = 0.0, post_exposure_efficacy = 0.5),
        ],
        attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
    )
    aborted = 0
    escaped = 0
    escaped_with_abort = 0
    escaped_with_onset = 0
    abort_before_infection = 0
    for seed in 1:100
        state = simulate(
            spec; n_initial = 3, rng = StableRNG(seed),
            stopping_rules = [Extinction(), MaxGenerations(30)]
        )
        for ind in state.individuals
            t = get(ind.state, :infection_aborted_time, nothing)
            if !EpiBranch.is_infected(ind)
                escaped += 1
                t === nothing || (escaped_with_abort += 1)
                # A never-infected individual has no onset time, because
                # onset is derived from the infection time.
                isnan(onset_time(ind)) || (escaped_with_onset += 1)
            elseif t !== nothing
                aborted += 1
                # An abort at or before the infection was drawn for an earlier
                # exposure: the dose already acted on this one as a
                # contact-side block, and would act a second time.
                t > ind.infection_time || (abort_before_infection += 1)
            end
        end
    end
    @test aborted > 0
    @test escaped > 0
    @test escaped_with_abort == 0
    @test escaped_with_onset == 0
    @test abort_before_infection == 0
end

@testset "FlagOnly tracing isolates pool members through the traced pathway" begin
    # Pool members are traced before their exposure is resolved, when their
    # onset is not yet known. With no test-positive cases only the traced
    # pathway can isolate, and it must still do so once the onset is known,
    # while a member whose infection was aborted before onset never isolates.
    spec = ModelSpec(
        AbortPoolModel(300, 4, 0.4);
        interventions = [
            Isolation(onset_to_isolation_delay = Exponential(1.0), test_sensitivity = 0.0, duration = Inf),
            ContactTracing(OnSymptomOnset(), 1.0, Exponential(0.5), FlagOnly()),
            RingVaccination(efficacy = 0.0, post_exposure_efficacy = 0.5),
        ],
        attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
    )
    isolated = 0
    isolated_before_onset = 0
    aborted = 0
    aborted_isolated = 0
    for seed in 1:50
        state = simulate(
            spec; n_initial = 3, rng = StableRNG(seed),
            stopping_rules = [Extinction(), MaxGenerations(30)]
        )
        for ind in state.individuals
            EpiBranch.is_infected(ind) || continue
            if haskey(ind.state, :infection_aborted_time)
                aborted += 1
                isfinite(isolation_time(ind)) && (aborted_isolated += 1)
            elseif isfinite(isolation_time(ind))
                isolated += 1
                isolation_time(ind) >= onset_time(ind) || (isolated_before_onset += 1)
            end
        end
    end
    @test isolated > 0
    @test isolated_before_onset == 0
    @test aborted > 0
    @test aborted_isolated == 0
end

# Records the generation of each node's latest exposure and of the first one in
# which it was found vaccinated. Listed after the ring vaccination, and recorded
# as an action, so a node dosed at an exposure records that exposure's
# generation.
struct ExposureGenerations <: AbstractIntervention end
function _record_exposure_generation!(t, at, state)
    t.state[:exposure_generation] = state.current_generation
    get(t.state, :vaccinated, false) || return nothing
    get!(t.state, :vaccination_generation, state.current_generation)
    return nothing
end
function EpiBranch.intervention_actions(::ExposureGenerations, state, targets)
    return [
        EpiBranch.InterventionAction(t, state.max_infection_time, _record_exposure_generation!)
            for t in targets
    ]
end

@testset "A dose given at an earlier exposure can abort the infecting one" begin
    # A node dosed at an exposure it escapes, then infected in a later
    # generation before its immunity arrives, with immunity arriving before its
    # onset, is aborted with the post-exposure efficacy like any other.
    spec = ModelSpec(
        AbortPoolModel(300, 4, 0.4);
        interventions = [
            Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf),
            ContactTracing(
                probability = 1.0,
                isolation_to_trace_delay = Exponential(0.5),
                action = FlagOnly()
            ),
            RingVaccination(efficacy = 0.0, post_exposure_efficacy = 0.5),
            ExposureGenerations(),
        ],
        attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
    )
    aborted = 0
    not_aborted = 0
    for seed in 1:300
        state = simulate(
            spec; n_initial = 3, rng = StableRNG(seed),
            stopping_rules = [Extinction(), MaxGenerations(30)]
        )
        for ind in state.individuals
            EpiBranch.is_infected(ind) || continue
            dosed = get(ind.state, :vaccination_generation, nothing)
            dosed !== nothing && dosed < ind.state[:exposure_generation] || continue
            # Immunity is immediate (`delay_to_immunity = 0`).
            immunity = ind.state[:vaccination_time]
            onset = ind.infection_time + ind.state[:incubation_period]
            ind.infection_time < immunity < onset || continue
            if haskey(ind.state, :infection_aborted_time)
                aborted += 1
            else
                not_aborted += 1
            end
        end
    end
    @test aborted + not_aborted >= 50
    @test 0.35 < aborted / (aborted + not_aborted) < 0.65
end

# Two cases expose the same person, who is half as susceptible as usual.
struct TwoToOneModel <: EpiBranch.TransmissionModel end
EpiBranch.population_size(::TwoToOneModel) = EpiBranch.NoPopulation()
function EpiBranch.initialise_state(
        m::TwoToOneModel, sim_opts::EpiBranch.SimOpts,
        interventions, transitions, attributes, rng::AbstractRNG
    )
    state = EpiBranch.new_state(m, transitions, attributes, rng)
    EpiBranch.add_individuals!(state, 3, interventions)
    EpiBranch.seed!(state, 1:2, interventions, transitions)
    state.individuals[3].susceptibility = 0.5
    return state
end
function EpiBranch.contacts_of(::TwoToOneModel, parent, state::EpiBranch.SimulationState)
    result = Tuple{eltype(state.individuals), Float64}[]
    parent.id == 3 && return result
    push!(result, (state.individuals[3], parent.infection_time + parent.id))
    return result
end

@testset "A person whose earliest exposure is blocked can be infected by a later one" begin
    n = 4000
    infected = count(1:n) do seed
        s = simulate(
            TwoToOneModel(); n_initial = 2, rng = StableRNG(seed),
            stopping_rules = [Extinction(), MaxGenerations(5)]
        )
        is_infected(s.individuals[3])
    end
    # Each exposure gets through with probability 0.5, so at least one does
    # with probability 0.75.
    @test abs(infected / n - 0.75) < 4 * sqrt(0.75 * 0.25 / n)
end

@testset "A contact a model creates lists no exposure after the run ends" begin
    late = 0
    for seed in 1:10
        s = simulate(
            MintModel(2); n_initial = 1, rng = StableRNG(seed), max_time = 2.0,
            max_cases = 10^6, max_generations = 100
        )
        late += count(i -> i.infection_time > 2.0, s.individuals)
    end
    @test late == 0
end

# A line of four people, the first an index case and the second never
# infected. `LineEarly` says its `contacts_of` reads nothing settling writes.
abstract type AbstractLine <: EpiBranch.TransmissionModel end
struct LineLate <: AbstractLine end
struct LineEarly <: AbstractLine end
EpiBranch.reads_settled_state(::LineEarly) = false
EpiBranch.population_size(::AbstractLine) = EpiBranch.NoPopulation()
function EpiBranch.initialise_state(
        m::AbstractLine, sim_opts::EpiBranch.SimOpts,
        interventions, transitions, attributes, rng::AbstractRNG
    )
    state = EpiBranch.new_state(m, transitions, attributes, rng)
    EpiBranch.add_individuals!(state, 4, interventions)
    state.individuals[2].susceptibility = 0.0
    EpiBranch.seed!(state, [1], interventions, transitions)
    return state
end
function EpiBranch.contacts_of(::AbstractLine, parent, state::EpiBranch.SimulationState)
    neighbours = filter(in(1:4), [parent.id - 1, parent.id + 1])
    return [(state.individuals[j], parent.infection_time + 1.0) for j in neighbours]
end

@testset "A ring grows past an uninfected person on a model that says it can" begin
    ct = ContactTracing(OnIsolation(), 1.0, Dirac(0.1), Quarantine(duration = Inf); depth = 2)
    iso = Isolation(onset_to_isolation_delay = Dirac(0.1), duration = Inf)
    attributes = clinical_presentation(incubation_period = Dirac(0.2))
    for (model, reached) in ((LineEarly(), true), (LineLate(), false))
        s = simulate(
            ModelSpec(model; interventions = [iso, ct], attributes);
            n_initial = 1, rng = StableRNG(1), max_generations = 10
        )
        @test is_traced(s.individuals[2]) && !is_infected(s.individuals[2])
        @test is_traced(s.individuals[3]) == reached
    end
end

# A model that says how many contacts each case makes, with and without saying
# the count reads nothing settling writes.
struct CountLate <: EpiBranch.TransmissionModel end
struct CountEarly <: EpiBranch.TransmissionModel end
const _CountModel = Union{CountLate, CountEarly}
EpiBranch.reads_settled_state(::CountEarly) = false
EpiBranch.population_size(::_CountModel) = EpiBranch.NoPopulation()
EpiBranch.model_generation_time(::_CountModel) = Gamma(2.0, 2.0)
EpiBranch.generate_offspring(::_CountModel, parent, state) = rand(state.rng, Poisson(2.0))

@testset "A model that counts its contacts can let a ring grow early" begin
    ct = ContactTracing(OnIsolation(), 0.8, Exponential(0.5), Quarantine(duration = Inf); depth = 2)
    iso = Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf)
    attributes = clinical_presentation(incubation_period = Gamma(3.0, 1.0))
    run(model, seed) = simulate(
        ModelSpec(model; interventions = [iso, ct], attributes);
        max_cases = 200, rng = StableRNG(seed)
    )
    sizes(model) = [length(run(model, seed).individuals) for seed in 1:50]
    bp = sizes(BranchingProcess(Poisson(2.0), Gamma(2.0, 2.0)))
    @test sizes(CountEarly()) == bp
    @test sizes(CountLate()) != bp
end

# `MintModel` that says its `contacts_of` reads nothing settling writes.
struct MintEarly <: EpiBranch.TransmissionModel end
EpiBranch.reads_settled_state(::MintEarly) = false
EpiBranch.population_size(::MintEarly) = EpiBranch.NoPopulation()
EpiBranch.contacts_of(::MintEarly, parent, state::EpiBranch.SimulationState) =
    EpiBranch.contacts_of(MintModel(2), parent, state)

@testset "A ring that asks early names the contacts a case goes on to expose" begin
    ct = ContactTracing(OnIsolation(), 1.0, Dirac(0.1), Quarantine(duration = Inf); depth = 2)
    iso = Isolation(onset_to_isolation_delay = Dirac(0.1), duration = Inf)
    attributes = clinical_presentation(incubation_period = Dirac(0.2))
    early = 0
    for seed in 1:10
        s = simulate(
            ModelSpec(MintEarly(); interventions = [iso, ct], attributes);
            n_initial = 1, rng = StableRNG(seed), max_generations = 6
        )
        for ind in s.individuals
            # Each person the model is asked about has two contacts, however
            # often it is asked.
            @test length(ind.secondary_case_ids) in (0, 2)
            is_infected(ind) || isempty(ind.secondary_case_ids) || (early += 1)
            # A contact's exposure follows its case's infection.
            for cid in ind.secondary_case_ids
                c = s.individuals[cid]
                is_infected(c) && @test c.infection_time > ind.infection_time
            end
        end
    end
    # Some uninfected ring members had their contacts named early.
    @test early > 0
    # A case at the generation bound has no contacts, as without the opt-in.
    for seed in 1:20
        s = simulate(
            ModelSpec(MintEarly(); interventions = [iso, ct], attributes);
            n_initial = 1, rng = StableRNG(seed), max_generations = 3
        )
        for ind in s.individuals
            ind.generation >= 3 && @test isempty(ind.secondary_case_ids)
        end
    end
end
