# EpiNetwork.jl

Transmission over a fixed contact network, built on
[EpiBranch.jl](https://github.com/epiforecasts/EpiBranch.jl).

`NetworkProcess` transmits along the edges of a fixed graph: each node is a
person, an infectious node infects its neighbours with a per-edge probability,
and each node can be infected at most once. Timing, interventions, attributes
and competing-risks resolution route through the EpiBranch engine, so
`Isolation`, `ContactTracing`, `RingVaccination`, `clinical_presentation` and
the rest apply directly.

```julia
using EpiBranch, EpiNetwork, Distributions

# Ring of 5 nodes, uniform per-edge probability
adjacency = [[2, 5], [1, 3], [2, 4], [3, 5], [4, 1]]
model = NetworkProcess(adjacency, 0.5, LogNormal(1.6, 0.5))

state = simulate(model; n_initial = 1,
    stopping_rules = [Extinction(), MaxGenerations(100)])
```

`NetworkProcess` was originally part of EpiBranch.jl and was split into this
subpackage so the core package stays focused on the branching-process models;
the structured network model lives here alongside future network and household
extensions.

## Status

EpiBranch.jl is not yet registered, so this subpackage depends on it by path
(`[sources]` in `Project.toml`, pointing at the repository root). Once EpiBranch
is registered the path source can be dropped.
