# Network models

`NetworkProcess` spreads infection over a fixed contact network. Each
node is a person, and each graph edge is a potential route of
transmission. Transmission is driven by a contact **rate** rather than a
single coin flip per edge: along each edge the time from an infector
becoming infectious to its next infectious contact with a neighbour is
drawn from a contact-interval distribution (the *kernel*), and
transmission happens only while the infector is still inside its
infectious window. Because that hazard races the infector's recovery or
isolation, shortening the infectious window genuinely curtails onward
spread — something the earlier coin-flip-per-edge model could not
express.

Each node can be infected once, and the graph does the job the offspring
distribution does in [`BranchingProcess`](@ref): it sets who can infect
whom. Node attributes and the clinical timeline behave as they do
elsewhere.

## Defining a network

The process is a pure transmission kernel: pass an adjacency list, where
`adjacency[i]` holds the nodes connected to node `i`, and the
contact-interval kernel — a `Distributions.jl` distribution shared by
every edge. The disease natural history is a `progression` of
`Transition`s attached with a [`ModelSpec`](@ref): a terminal removal
transition sets how long a node stays infectious once infected, and an
optional latent-period transition delays the start of that window. The
kernel times contacts from the window's start state, derived from the
progression (`:infectious` when a latent period produces it, otherwise
`:infection`).

```@example networks
using EpiBranch
using EpiNetwork
using Distributions
using StableRNGs

# Twenty households of four, wired into a ring of households by one
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
model = ModelSpec(NetworkProcess(adjacency, Exponential(3.0));
    progression = [
        Transition(:infectious; from = :infection, delay = LogNormal(1.6, 0.5)),
        Transition(:recovered; from = :infectious, delay = 7.0, terminal = true),
    ])
```

The kernel here has a mean contact interval of three days, and each node
stays infectious for seven days, so most edges have time to transmit but
not all do.

A weighted adjacency matrix works too. `NetworkProcess(A, kernel)` reads
any nonzero `A[i, j]` as an undirected edge; the matrix marks the graph
structure only, and every edge shares the kernel.

## Generating a network with Graphs.jl

Building an adjacency list by hand suits small or bespoke structures, but
for realistic contact networks it is easier to use the generators in
[Graphs.jl](https://juliagraphs.org/Graphs.jl/), the standard Julia graph
library. Load Graphs.jl and pass a graph straight to `NetworkProcess`: each
vertex becomes a node and each vertex's neighbours become its contacts.

```@example networks
using Graphs

# A small-world network: mostly local contacts (high clustering) with a
# few long-range links, from the Watts–Strogatz model.
g = watts_strogatz(400, 6, 0.1)

model_ws = ModelSpec(NetworkProcess(g, Exponential(3.0));
    progression = [Transition(:recovered; from = :infection, delay = 7.0, terminal = true)])
state = simulate(model_ws; n_initial = 1, rng = StableRNG(3))
println("Final size: ", state.cumulative_cases, " of ", nv(g))
```

Any generator that returns a graph works, so the structure the outbreak
spreads on is a modelling choice. A few that map onto common assumptions:

- `watts_strogatz(n, k, β)` — small-world: local clustering with a few
  long-range links.
- `barabasi_albert(n, k)` — scale-free: a heavy-tailed degree distribution,
  so a minority of highly-connected nodes drive spread.
- `stochastic_block_model(...)` — block structure: dense within blocks and
  sparse between them, a natural fit for households or communities.
- `euclidean_graph(n, d; cutoff)` — a random geometric graph where nodes
  close in space are linked, so clustering emerges from proximity (the
  mechanism spatial outbreak-network models use); it returns the graph and
  the distances, so take the first element.

The kernel, progression and attributes attach exactly as before — only the
source of the graph changes. Interventions attach through the infectious
window: one that removes a case from transmission — `Isolation` — shortens
that window and curtails spread (see below), while an intervention whose
effect is a per-contact competing risk (contact tracing, leaky vaccination)
has no representation on the continuous-time network path and is reported
with a warning rather than applied. Graphs.jl is an optional
dependency: this constructor becomes available once you load Graphs.jl,
and the adjacency-list and matrix constructors need nothing extra. For a
directed graph, a node's out-neighbours are the contacts it can infect.

## Simulating

`simulate` returns a `SimulationState`, and [`linelist`](@ref) renders it
as a one-row-per-case DataFrame.

```@example networks
rng = StableRNG(42)
state = simulate(model; n_initial = 1, rng = rng)

println("Final outbreak size: ", state.cumulative_cases, " of ", length(adjacency))
```

The population is the graph, so an outbreak saturates at the number of
nodes instead of growing without bound. For a batch of independent runs,
simulate over a sequence of seeds:

```@example networks
sizes = [simulate(model; n_initial = 1, rng = StableRNG(i)).cumulative_cases
         for i in 1:200]
println("Mean size: ", round(sum(sizes) / length(sizes), digits = 1))
```

## Attributes belong to the node

Each node is built once, so its attributes are drawn once and stay
fixed for the run. Node properties like age or risk group are part of
the network and are carried into the line list.

```@example networks
attrs = [
    demographics(age_distribution = Uniform(0, 80)),
    clinical_presentation(incubation_period = LogNormal(1.6, 0.5)),
]

model_attrs = ModelSpec(NetworkProcess(adjacency, Exponential(3.0));
    progression = [Transition(:recovered; from = :infection, delay = 7.0, terminal = true)],
    attributes = attrs)
state = simulate(model_attrs; n_initial = 1, rng = StableRNG(7))

infected = filter(is_infected, state.individuals)
ages = [ind.state[:age] for ind in infected]
println("Cases: ", length(infected),
    "; mean age: ", round(sum(ages) / length(ages), digits = 1))
```

## Isolation curtails onward spread

Because transmission is a hazard racing removal, shortening a node's
infectious window cuts onward transmission — the feature a
coin-flip-per-edge model cannot express. Isolation is part of the
clinical timeline: adding an `:isolated` transition closes the
infectious window early, so a node that isolates soon after becoming
infectious contacts fewer neighbours before it stops transmitting.

Here the same fast kernel spreads through the whole ring when nothing
stops it, but isolating each case a couple of days after infection holds
the outbreak back.

```@example networks
kernel = Exponential(1.5)

baseline = ModelSpec(NetworkProcess(adjacency, kernel);
    progression = [
        Transition(:recovered; from = :infection,
            delay = (rng, ind) -> 10.0, terminal = true)])

isolating = ModelSpec(NetworkProcess(adjacency, kernel);
    progression = [
        Transition(:recovered; from = :infection,
            delay = (rng, ind) -> 10.0, terminal = true),
        Transition(:isolated; from = :infection, delay = Exponential(2.0))])

base_sizes = [simulate(baseline; n_initial = 1, rng = StableRNG(i)).cumulative_cases
              for i in 1:200]
iso_sizes = [simulate(isolating; n_initial = 1, rng = StableRNG(i)).cumulative_cases
             for i in 1:200]

println("Mean size, no isolation:   ",
    round(sum(base_sizes) / length(base_sizes), digits = 1))
println("Mean size, with isolation: ",
    round(sum(iso_sizes) / length(iso_sizes), digits = 1))
```

## Community introductions

Without an external hazard the outbreak starts from the seeded index
nodes and spreads only along the edges. An `external_hazard` adds a
community force of infection, so fresh introductions appear over time; a
finite `obs_end` on the process bounds the window over which they can
arrive.

```@example networks
model_ext = ModelSpec(NetworkProcess(adjacency, Exponential(3.0);
        external_hazard = 0.02, obs_end = 60.0);
    progression = [Transition(:recovered; from = :infection, delay = 7.0, terminal = true)])
state = simulate(model_ext; rng = StableRNG(11))
df = linelist(state)

println("Cases: ", size(df, 1),
    "; community introductions: ", count(df.index))
```
