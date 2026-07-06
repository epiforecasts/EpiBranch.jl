# EpiNetwork.jl

Transmission over a fixed contact network, built on
[EpiBranch.jl](https://github.com/epiforecasts/EpiBranch.jl).

`NetworkProcess` transmits along the edges of a fixed graph: each node is a
person, and along each edge the time from an infector becoming infectious to
its next infectious contact with a neighbour is drawn from a contact-interval
kernel (Kenah 2011). Transmission happens only while the infector is still
infectious, so shortening that window — through recovery or isolation —
curtails onward spread; each node can be infected at most once. It is a
structure-driven model simulated by the **Sellke construction** in continuous
time, like `HouseholdProcess` in EpiHouseholds. The latent period, infectious
period, onset and testing are flexible `Transition`s on the shared
natural-history timeline, so a network line list is an EpiBranch line list.

```julia
using EpiBranch, EpiNetwork, Distributions

# Ring of 5 nodes; every edge shares a contact-interval kernel
adjacency = [[2, 5], [1, 3], [2, 4], [3, 5], [4, 1]]
model = NetworkProcess(adjacency, Exponential(2.0); infectious_period = 6.0)

state = simulate(model; n_initial = 1)
```

`NetworkProcess` was originally part of EpiBranch.jl and was split into this
subpackage so the core package stays focused on the branching-process models;
the structured network model lives here alongside future network and household
extensions.

## Status

EpiBranch.jl is not yet registered, so this subpackage depends on it by path
(`[sources]` in `Project.toml`, pointing at the repository root). Once EpiBranch
is registered the path source can be dropped.
