using StableRNGs
using Random

# A minimal structure-driven model — a ring of nodes — exercising the shared
# engine's structure-driven path from within core: `gather_by_target` (shared,
# deduplicated exposures), a model-contributed competing risk, and the
# "exposed but not infected" resolve branch. The network and household
# subpackages cover these too, but only under their own coverage flags; a tiny
# model here keeps the extension contract tested in core.

struct RingModel <: EpiBranch.TransmissionModel
    n::Int
    p::Float64          # per-edge transmission probability
end

# Depletion comes from the ring structure (a node is infected at most once), not
# from a population-level susceptible pool — so, like a network, no population.
EpiBranch.population_size(::RingModel) = EpiBranch.NoPopulation()

function EpiBranch.initialise_state(m::RingModel, sim_opts::EpiBranch.SimOpts,
        interventions, transitions, attributes, rng::AbstractRNG)
    state = EpiBranch.new_state(m, transitions, attributes, rng)
    EpiBranch.add_individuals!(state, m.n, interventions)
    EpiBranch.seed!(state, 1:(sim_opts.n_initial), interventions, transitions)
    return state
end

# Shared contacts across parents in a generation ⇒ gather by target.
function EpiBranch.collect_exposures(m::RingModel, state::EpiBranch.SimulationState)
    return EpiBranch.gather_by_target(m, state)
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
# loop in the engine's resolution); p < 1 lets some exposed nodes escape.
struct RingRisk
    p::Float64
end
EpiBranch.transmission_risks(m::RingModel) = (RingRisk(m.p),)
function EpiBranch.competing_risk(r::RingRisk, parent, contact, state)
    r.p < 1.0 ? EpiBranch.Risk(block_probability = 1.0 - r.p) : nothing
end

@testset "Structure-driven model (core extension path)" begin
    # p < 1 so some exposures fail — exercises the exposed-but-not-infected
    # resolve branch; the ring guarantees a node is reached from both sides.
    state = simulate(RingModel(50, 0.5); n_initial = 1, rng = StableRNG(11),
        stopping_rules = [Extinction(), MaxGenerations(50)])
    infected = count(ind -> get(ind.state, :infected, false), state.individuals)
    @test length(state.individuals) == 50            # fixed, pre-instantiated pool
    @test 1 <= infected <= 50
    @test EpiBranch._timetype(state) === Float64     # generic (non-BranchingProcess)

    # a fully-transmitting ring (p = 1) infects the whole ring
    full = simulate(RingModel(20, 1.0); n_initial = 1, rng = StableRNG(3),
        stopping_rules = [Extinction(), MaxGenerations(50)])
    @test count(ind -> get(ind.state, :infected, false), full.individuals) == 20
end

# A structure-driven model whose `contacts_of` *mints* fresh contacts (the other
# `gather_by_target` branch, where a contact's id is past the generation's
# starting count). It uses the default `initialise_state` (offspring-style seeds).
struct MintModel <: EpiBranch.TransmissionModel
    k::Int
end
EpiBranch.population_size(::MintModel) = EpiBranch.NoPopulation()
function EpiBranch.collect_exposures(m::MintModel, state::EpiBranch.SimulationState)
    return EpiBranch.gather_by_target(m, state)
end
function EpiBranch.contacts_of(m::MintModel, parent, state::EpiBranch.SimulationState)
    result = Tuple{eltype(state.individuals), Float64}[]
    for _ in 1:(m.k)
        t = parent.infection_time + rand(state.rng)
        push!(result, (EpiBranch.make_contact!(state, parent, t), t))
    end
    return result
end

@testset "Structure-driven model minting fresh contacts" begin
    state = simulate(MintModel(2); n_initial = 1, rng = StableRNG(7),
        stopping_rules = [Extinction(), MaxGenerations(4)])
    @test length(state.individuals) > 1            # fresh contacts were minted
    @test EpiBranch._timetype(state) === Float64
end

# A pool of pre-created nodes, each parent reaching random members, so a node
# can be exposed, escape, and be exposed again in a later generation. That is
# the path on which a post-exposure abort drawn for one exposure could outlive
# it.
struct AbortPoolModel <: EpiBranch.TransmissionModel
    n::Int
    k::Int
    p::Float64
end
EpiBranch.population_size(::AbortPoolModel) = EpiBranch.NoPopulation()
function EpiBranch.initialise_state(m::AbortPoolModel, sim_opts::EpiBranch.SimOpts,
        interventions, transitions, attributes, rng::AbstractRNG)
    state = EpiBranch.new_state(m, transitions, attributes, rng)
    EpiBranch.add_individuals!(state, m.n, interventions)
    EpiBranch.seed!(state, 1:(sim_opts.n_initial), interventions, transitions)
    return state
end
function EpiBranch.collect_exposures(m::AbortPoolModel, state::EpiBranch.SimulationState)
    return EpiBranch.gather_by_target(m, state)
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
    spec = ModelSpec(AbortPoolModel(300, 4, 0.4);
        interventions = [
            Isolation(onset_to_isolation_delay = Exponential(1.0)),
            ContactTracing(probability = 1.0,
                isolation_to_trace_delay = Exponential(0.5),
                quarantine_on_trace = false),
            RingVaccination(efficacy = 0.0, post_exposure_efficacy = 0.5)],
        attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)))
    aborted = 0
    escaped = 0
    escaped_with_abort = 0
    escaped_with_onset = 0
    abort_before_infection = 0
    for seed in 1:100
        state = simulate(spec; n_initial = 3, rng = StableRNG(seed),
            stopping_rules = [Extinction(), MaxGenerations(30)])
        for ind in state.individuals
            t = get(ind.state, :infection_aborted_time, nothing)
            if !EpiBranch.is_infected(ind)
                escaped += 1
                t === nothing || (escaped_with_abort += 1)
                # A never-infected individual has no onset either: onset
                # follows the infection time, and an infection that never
                # happened carries no time of its own to derive one from.
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

# Records the generation of each node's latest exposure and of the first one in
# which it was found vaccinated. Listed after the ring vaccination, so a node
# dosed at an exposure records that exposure's generation.
struct ExposureGenerations <: AbstractIntervention end
function EpiBranch.apply_post_transmission!(::ExposureGenerations, state, targets)
    for t in targets
        t.state[:exposure_generation] = state.current_generation
        get(t.state, :vaccinated, false) || continue
        get!(t.state, :vaccination_generation, state.current_generation)
    end
    return nothing
end

@testset "A dose given at an earlier exposure can abort the infecting one" begin
    # A node dosed at an exposure it escapes, then infected in a later
    # generation before its immunity arrives, with immunity arriving before its
    # onset, is aborted with the post-exposure efficacy like any other.
    spec = ModelSpec(AbortPoolModel(300, 4, 0.4);
        interventions = [
            Isolation(onset_to_isolation_delay = Exponential(1.0)),
            ContactTracing(probability = 1.0,
                isolation_to_trace_delay = Exponential(0.5),
                quarantine_on_trace = false),
            RingVaccination(efficacy = 0.0, post_exposure_efficacy = 0.5),
            ExposureGenerations()],
        attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)))
    aborted = 0
    not_aborted = 0
    for seed in 1:300
        state = simulate(spec; n_initial = 3, rng = StableRNG(seed),
            stopping_rules = [Extinction(), MaxGenerations(30)])
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

@testset "An abort is dropped when resolution does not bear its exposure out" begin
    function exposed(infected, infection_time)
        ind = Individual(id = 1, infection_time = infection_time)
        ind.state[:infected] = infected
        ind.state[:incubation_period] = 6.0
        ind.state[:infection_aborted_time] = 4.0
        ind.state[:onset_time] = NaN
        return ind
    end
    # Infected before the abort: the abort stands.
    ind = exposed(true, 1.0)
    EpiBranch._drop_stale_abort!(ind)
    @test ind.state[:infection_aborted_time] == 4.0
    @test isnan(onset_time(ind))
    # Infected through a later exposure at or after the abort time.
    ind = exposed(true, 4.0)
    EpiBranch._drop_stale_abort!(ind)
    @test !haskey(ind.state, :infection_aborted_time)
    @test onset_time(ind) == 10.0
    # Not infected at all.
    ind = exposed(false, 0.0)
    EpiBranch._drop_stale_abort!(ind)
    @test !haskey(ind.state, :infection_aborted_time)
    @test onset_time(ind) == 6.0
end
