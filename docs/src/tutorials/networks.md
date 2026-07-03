# Network models

`NetworkProcess` spreads infection over a fixed contact network. Each
node is a person. An infectious node infects its neighbours, and each
node can be infected once. The graph does the job the offspring
distribution does in [`BranchingProcess`](@ref): it sets who can infect
whom. Interventions, node attributes and the clinical timeline all
behave exactly as they do for a branching process.

## Defining a network

Pass an adjacency list, where `adjacency[i]` holds the nodes connected
to node `i`, plus a per-edge transmission probability and a generation
time.

```@example networks
using EpiBranch
using EpiNetwork
using Distributions
using StableRNGs

# Five households of four, wired into a ring of households by one
# bridge edge between consecutive households.
function household_ring(n_households, household_size)
    n = n_households * household_size
    adj = [Int[] for _ in 1:n]
    for h in 0:(n_households - 1)
        members = (h * household_size + 1):(h * household_size + household_size)
        for i in members, j in members
            i != j && push!(adj[i], j)
        end
        # bridge to the next household
        a = h * household_size + 1
        b = (mod(h + 1, n_households)) * household_size + 1
        push!(adj[a], b)
        push!(adj[b], a)
    end
    return [sort(unique(a)) for a in adj]
end

adjacency = household_ring(20, 4)
clinical = clinical_presentation(incubation_period = LogNormal(1.6, 0.5))
model = NetworkProcess(adjacency, 0.4, LogNormal(1.6, 0.5); attributes = clinical)
```

A weighted adjacency matrix works too. `NetworkProcess(A, gt)` reads any
nonzero `A[i, j]` as an undirected edge whose value is the per-edge
transmission probability.

## Simulating

```@example networks
rng = StableRNG(42)
state = simulate(model;
    n_initial = 1, stopping_rules = [Extinction(), MaxGenerations(50)],
    rng = rng)

println("Final outbreak size: ", state.cumulative_cases, " of ", length(adjacency))
```

The population is the graph, so an outbreak saturates at the number of
nodes instead of growing without bound. For a batch, use
`simulate(model, n)`:

```@example networks
results = simulate(model, 200;
    n_initial = 1, stopping_rules = [Extinction(), MaxGenerations(50)],
    rng = StableRNG(1))

sizes = [s.cumulative_cases for s in results]
println("Mean size: ", round(sum(sizes) / length(sizes), digits = 1))
```

## Attributes belong to the node

Each node is built once, so its attributes are drawn once and stay
fixed for the run. Node properties like age, risk group, or
susceptibility are part of the network. Below, susceptibility rises
with age, and the older, more susceptible nodes end up over-represented
among cases:

```@example networks
attrs = [
    demographics(age_distribution = Uniform(0, 80)),
    clinical_presentation(incubation_period = LogNormal(1.6, 0.5)),
    transmission_traits(
        susceptibility = (rng, ind) -> ind.state[:age] >= 60 ? 0.9 : 0.3),
]

model_attrs = NetworkProcess(adjacency, 0.4, LogNormal(1.6, 0.5); attributes = attrs)
state = simulate(model_attrs;
    n_initial = 1, stopping_rules = [Extinction(), MaxGenerations(50)],
    rng = StableRNG(7))

infected = filter(is_infected, state.individuals)
frac_old = count(ind -> ind.state[:age] >= 60, infected) / max(length(infected), 1)
println("Fraction of cases aged ≥60: ", round(frac_old, digits = 2))
```

## Interventions

Isolation, contact tracing, and ring vaccination work without changes.
A node traced on the network keeps its identity, so its intervention
state carries across generations.

```@example networks
iso = Isolation(onset_to_isolation_delay = Exponential(2.0))
ct = ContactTracing(probability = 0.6, isolation_to_trace_delay = Exponential(1.5))

model_ct = NetworkProcess(adjacency, 0.4, LogNormal(1.6, 0.5);
    interventions = [iso, ct], attributes = clinical)
results = simulate(model_ct, 200;
    n_initial = 1, stopping_rules = [Extinction(), MaxGenerations(50)],
    rng = StableRNG(2))

sizes = [s.cumulative_cases for s in results]
println("Mean size with isolation + tracing: ", round(sum(sizes) / length(sizes), digits = 1))
```
