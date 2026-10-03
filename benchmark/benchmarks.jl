# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Benchmark suite definition. Build a BenchmarkTools `BenchmarkGroup` named
# `SUITE`; the managed `run.jl` / `compare.jl` consume it. Edit freely.
#
# One flat group, with the area in each name. `compare_comment` files every
# benchmark that is not an AD-gradient one under a single heading of its own
# choosing, so a group named here would not reach the comment and a reader
# would see a simulation regression under a heading about evaluation.

using BenchmarkTools
using Distributions
using StableRNGs
using EpiBranch
using EpiHouseholds
using EpiNetwork

const SUITE = BenchmarkGroup()

# A fixed graph, so a timing shift means the engine changed rather than the
# population did.
function fixed_graph(n, degree; seed = 42)
    rng = StableRNG(seed)
    adjacency = [Int[] for _ in 1:n]
    seen = Set{Tuple{Int, Int}}()
    while length(seen) < (n * degree) ÷ 2
        i, j = rand(rng, 1:n), rand(rng, 1:n)
        i == j && continue
        a, b = minmax(i, j)
        (a, b) in seen && continue
        push!(seen, (a, b))
        push!(adjacency[a], b)
        push!(adjacency[b], a)
    end
    return adjacency
end

const GRAPH = fixed_graph(200, 6)
const PROGRESSION = [Transition(:recovered; delay = 3.0, terminal = true)]

# The Sellke race over a network, which is the engine's hot path.
network_model(kernel) = ModelSpec(NetworkProcess(GRAPH, kernel); progression = PROGRESSION)

SUITE["simulation: network, shared kernel"] = @benchmarkable simulate(
    $(network_model(Exponential(1.2))); initial_cases = [1], rng = StableRNG(1)
)

SUITE["simulation: households"] = @benchmarkable simulate(
    $(
        ModelSpec(
            HouseholdProcess(fill(4, 300), Weibull(1.5, 3.0));
            progression = PROGRESSION
        )
    ); rng = StableRNG(1)
)

# A `PairKernel` with a projection re-reads host records as the outbreak runs.
# Its cost depends on how often a record moves, so it is benchmarked with and
# without an intervention that moves one.
const LIVE_KERNEL = PairKernel(
    (c, a, b) -> Exponential(1.2);
    state = ind -> (tag = get(ind.state, :tag, 0.0)::Float64,), watches = (:tag,)
)

SUITE["simulation: network, live kernel"] = @benchmarkable simulate(
    $(network_model(LIVE_KERNEL)); initial_cases = [1], rng = StableRNG(1)
)

struct BenchmarkPolicy <: AbstractIntervention end
function EpiBranch.resolve_individual!(::BenchmarkPolicy, ind, state)
    state.cumulative_cases == 5 || return nothing
    for person in state.individuals
        person.state[:tag] = ind.infection_time
    end
    return nothing
end

SUITE["simulation: network, live kernel with policy"] = @benchmarkable simulate(
    $(
        ModelSpec(
            NetworkProcess(GRAPH, LIVE_KERNEL); progression = PROGRESSION,
            interventions = [BenchmarkPolicy()]
        )
    ); initial_cases = [1], rng = StableRNG(1)
)

# The pairwise survival likelihood, which inference evaluates repeatedly. The
# compiled layout is reused across evaluations, so both are timed.
const LIK_MODEL = network_model(Exponential(1.2))
const LIK_DATA = network_infections(
    simulate(LIK_MODEL; initial_cases = [1], rng = StableRNG(1)), LIK_MODEL
)
const LIK_LAYOUT = compile_contact_pairs(LIK_DATA)

SUITE["evaluation: compile_contact_pairs"] = @benchmarkable compile_contact_pairs($LIK_DATA)
SUITE["evaluation: pairwise_surv_loglik"] = @benchmarkable pairwise_surv_loglik(
    $(Exponential(1.2)), $LIK_DATA, $LIK_LAYOUT
)

# Closed-form analytics. `chain_size_distribution` returns a law rather than
# computing one, so the size law is timed by evaluating it.
SUITE["evaluation: extinction_probability"] = @benchmarkable extinction_probability(
    $(BranchingProcess(Poisson(1.5)))
)
const CHAIN_LAW = chain_size_distribution(BranchingProcess(Poisson(0.8)))
SUITE["evaluation: chain size loglikelihood"] = @benchmarkable loglikelihood(
    $CHAIN_LAW, $(collect(1:50))
)
