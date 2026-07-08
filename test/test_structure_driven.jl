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
